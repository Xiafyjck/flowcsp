"""
DiffCSP网络实现 - 基于图神经网络的晶体结构生成模型
实现CSPNet架构，联合建模晶格参数和原子坐标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter_mean, scatter_sum
from typing import Dict, Optional, Tuple
import math

from . import BaseCSPNetwork, register_network


class SinusoidalTimeEmbedding(nn.Module):
    """时间步的正弦编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch_size, 1] 时间步
        Returns:
            [batch_size, dim] 时间编码
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DistanceEmbedding(nn.Module):
    """距离的正弦余弦编码"""
    
    def __init__(self, n_frequencies: int = 10):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.frequencies = torch.arange(1, n_frequencies + 1) * math.pi
        
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [num_edges, 1] 边的距离
        Returns:
            [num_edges, 2*n_frequencies] 距离编码
        """
        # 扩展频率到设备
        freqs = self.frequencies.to(distances.device)
        
        # 计算正弦余弦编码
        scaled = distances * freqs  # [num_edges, n_frequencies]
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class CSPMessagePassing(MessagePassing):
    """
    CSP消息传递层
    处理原子间的相互作用，考虑周期性边界条件
    """
    
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 20,
        act_fn: str = 'silu'
    ):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        
        # 边特征编码
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU() if act_fn == 'silu' else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 消息网络
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.SiLU() if act_fn == 'silu' else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 更新网络
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU() if act_fn == 'silu' else nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征（距离编码）[num_edges, edge_dim]
        Returns:
            更新后的节点特征 [num_nodes, hidden_dim]
        """
        # 编码边特征
        edge_features = self.edge_encoder(edge_attr)
        
        # 消息传递
        out = self.propagate(edge_index, x=x, edge_features=edge_features)
        
        # 残差连接
        return x + out
        
    def message(self, x_i, x_j, edge_features):
        """构建消息"""
        msg_input = torch.cat([x_i, x_j, edge_features], dim=-1)
        return self.message_net(msg_input)
    
    def update(self, aggr_out, x):
        """更新节点"""
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(update_input)


class CSPLayer(nn.Module):
    """
    CSP层 - 联合处理晶格和原子
    """
    
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 20,
        act_fn: str = 'silu',
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        
        # 消息传递
        self.message_passing = CSPMessagePassing(hidden_dim, edge_dim, act_fn)
        
        # 晶格-原子交互
        self.lattice_to_atom = nn.Linear(9, hidden_dim)
        self.atom_to_lattice = nn.Linear(hidden_dim, 9)
        
        # 层归一化
        if use_layer_norm:
            self.node_norm = nn.LayerNorm(hidden_dim)
            self.lattice_norm = nn.LayerNorm(9)
        
        # 激活函数
        self.act = nn.SiLU() if act_fn == 'silu' else nn.ReLU()
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: 节点特征 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
            lattice: 晶格参数 [batch_size, 3, 3]
            batch: 节点批次索引 [num_nodes]
        Returns:
            更新的节点特征和晶格参数
        """
        batch_size = lattice.shape[0]
        
        # 1. 原子间消息传递
        node_features = self.message_passing(node_features, edge_index, edge_attr)
        
        # 2. 晶格到原子的信息传递
        lattice_flat = lattice.view(batch_size, 9)  # [batch_size, 9]
        lattice_info = self.lattice_to_atom(lattice_flat)  # [batch_size, hidden_dim]
        
        # 广播晶格信息到对应批次的原子
        lattice_broadcast = lattice_info[batch]  # [num_nodes, hidden_dim]
        node_features = node_features + lattice_broadcast
        
        # 3. 原子到晶格的信息聚合
        atom_info = scatter_mean(node_features, batch, dim=0, dim_size=batch_size)  # [batch_size, hidden_dim]
        lattice_update = self.atom_to_lattice(atom_info)  # [batch_size, 9]
        lattice_flat = lattice_flat + lattice_update
        
        # 4. 层归一化
        if self.use_layer_norm:
            node_features = self.node_norm(node_features)
            lattice_flat = self.lattice_norm(lattice_flat)
        
        # 重塑晶格
        lattice = lattice_flat.view(batch_size, 3, 3)
        
        return node_features, lattice


@register_network('diffcsp')
class DiffCSPNetwork(BaseCSPNetwork):
    """
    DiffCSP网络 - 基于图神经网络的晶体结构生成
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # 网络参数
        self.hidden_dim = config.get('hidden_dim', 128)
        self.latent_dim = config.get('latent_dim', 256)
        self.num_layers = config.get('num_layers', 4)
        self.max_atoms = config.get('max_atoms', 60)
        self.pxrd_dim = config.get('pxrd_dim', 11501)
        
        # 图构建参数
        # 全连接图不需要半径和最大邻居数参数
        self.edge_dim = config.get('edge_dim', 20)
        
        # 原子类型嵌入
        self.atom_embedding = nn.Embedding(100, self.hidden_dim)
        
        # PXRD编码器
        self.pxrd_encoder = nn.Sequential(
            nn.Linear(self.pxrd_dim, 512),
            nn.SiLU(),
            nn.Linear(512, self.latent_dim)
        )
        
        # 时间嵌入
        self.time_embedding = SinusoidalTimeEmbedding(self.latent_dim)
        
        # 距离嵌入
        self.distance_embedding = DistanceEmbedding(n_frequencies=self.edge_dim // 2)
        
        # 原子特征融合
        self.atom_encoder = nn.Linear(
            self.hidden_dim + self.latent_dim * 3,  # atom_emb + pxrd + t + r
            self.hidden_dim
        )
        
        # CSP层
        self.csp_layers = nn.ModuleList([
            CSPLayer(
                self.hidden_dim,
                self.edge_dim,
                act_fn='silu',
                use_layer_norm=True
            ) for _ in range(self.num_layers)
        ])
        
        # 输出头
        self.coord_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3)
        )
        
        self.lattice_output = nn.Sequential(
            nn.Linear(9, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 9)
        )
        
        # 注册缓冲区（用于DDP）
        self.register_buffer('_dummy', torch.zeros(1))
        
    def prepare_batch(self, batch: dict) -> dict:
        """
        准备批次数据 - 将统一格式转换为图格式
        """
        # 检查是否已经包含PyG批次
        if 'pyg_batch' in batch:
            return batch
        
        # 否则需要动态构建全连接图（降级方案）
        from ..gnn_data import fully_connected_graph_pbc
        
        batch_size = batch['z'].shape[0]
        lattice = batch['z'][:, :3, :]  # [B, 3, 3]
        frac_coords = batch['z'][:, 3:, :]  # [B, 60, 3]
        num_atoms = batch['num_atoms']
        
        # 展平分数坐标
        valid_frac = []
        for b in range(batch_size):
            n = num_atoms[b].item()
            valid_frac.append(frac_coords[b, :n, :])
        
        if len(valid_frac) > 0:
            flat_frac = torch.cat(valid_frac, dim=0)
            
            # 构建全连接图
            edge_index, edge_attr, cell_offsets = fully_connected_graph_pbc(
                flat_frac,
                lattice,
                num_atoms
            )
            
            # 创建批次索引
            batch_idx = torch.repeat_interleave(
                torch.arange(batch_size, device=batch['z'].device),
                num_atoms
            )
            
            # 添加到批次
            batch['edge_index'] = edge_index
            batch['edge_attr'] = edge_attr
            batch['cell_offsets'] = cell_offsets
            batch['batch_idx'] = batch_idx
            batch['flat_frac_coords'] = flat_frac
        
        return batch
    
    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        conditions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 当前状态 [batch_size, 63, 3]
            t: 时间步 [batch_size, 1]
            r: 另一个时间步 [batch_size, 1]
            conditions: 条件字典，包含comp, pxrd, num_atoms等
            
        Returns:
            预测的速度场 [batch_size, 63, 3]
        """
        batch_size = z.shape[0]
        device = z.device
        
        # 解析输入
        lattice = z[:, :3, :]  # [B, 3, 3]
        frac_coords_full = z[:, 3:, :]  # [B, 60, 3]
        
        comp = conditions['comp']  # [B, 60]
        pxrd = conditions['pxrd']  # [B, 11501]
        num_atoms = conditions['num_atoms']  # [B]
        
        # 准备图数据（检查是否有预构建的图）
        if 'pyg_batch' in conditions:
            # 使用预构建的图
            pyg_batch = conditions['pyg_batch']
            edge_index = pyg_batch.edge_index
            edge_attr = pyg_batch.edge_attr
            batch_idx = pyg_batch.batch
            
            # 展平的有效原子
            flat_atom_types = pyg_batch.x
            flat_frac_coords = pyg_batch.pos
        else:
            # 动态构建全连接图
            from ..gnn_data import fully_connected_graph_pbc
            
            # 展平有效原子
            flat_atom_types = []
            flat_frac_coords = []
            for b in range(batch_size):
                n = num_atoms[b].item()
                flat_atom_types.append(comp[b, :n])
                flat_frac_coords.append(frac_coords_full[b, :n, :])
            
            flat_atom_types = torch.cat(flat_atom_types, dim=0)
            flat_frac_coords = torch.cat(flat_frac_coords, dim=0)
            
            # 构建全连接图
            edge_index, edge_attr, _ = fully_connected_graph_pbc(
                flat_frac_coords,
                lattice,
                num_atoms
            )
            
            # 批次索引
            batch_idx = torch.repeat_interleave(
                torch.arange(batch_size, device=device),
                num_atoms
            )
        
        # 时间编码
        t_emb = self.time_embedding(t)  # [B, latent_dim]
        r_emb = self.time_embedding(r)  # [B, latent_dim]
        
        # PXRD编码
        pxrd_emb = self.pxrd_encoder(pxrd)  # [B, latent_dim]
        
        # 原子嵌入
        node_features = self.atom_embedding(flat_atom_types.long())  # [num_nodes, hidden_dim]
        
        # 融合条件信息
        # 将批次级别的特征广播到节点
        t_broadcast = t_emb[batch_idx]  # [num_nodes, latent_dim]
        r_broadcast = r_emb[batch_idx]  # [num_nodes, latent_dim]
        pxrd_broadcast = pxrd_emb[batch_idx]  # [num_nodes, latent_dim]
        
        # 组合特征
        combined_features = torch.cat([
            node_features,
            t_broadcast,
            r_broadcast,
            pxrd_broadcast
        ], dim=-1)
        
        # 编码原子特征
        node_features = self.atom_encoder(combined_features)  # [num_nodes, hidden_dim]
        
        # 距离编码
        if edge_attr.shape[1] == 1:
            # 如果只有距离，进行编码
            edge_features = self.distance_embedding(edge_attr)  # [num_edges, edge_dim]
        else:
            edge_features = edge_attr
        
        # 通过CSP层
        for layer in self.csp_layers:
            node_features, lattice = layer(
                node_features,
                edge_index,
                edge_features,
                lattice,
                batch_idx
            )
        
        # 输出预测
        # 1. 预测分数坐标更新
        coord_update = self.coord_output(node_features)  # [num_nodes, 3]
        
        # 2. 预测晶格更新
        lattice_flat = lattice.view(batch_size, 9)
        lattice_update = self.lattice_output(lattice_flat)  # [B, 9]
        lattice_update = lattice_update.view(batch_size, 3, 3)
        
        # 3. 组装输出
        output = torch.zeros_like(z)  # [B, 63, 3]
        output[:, :3, :] = lattice_update  # 晶格更新
        
        # 将节点更新填充回完整的60个原子位置
        node_ptr = torch.cat([
            torch.tensor([0], device=device),
            torch.cumsum(num_atoms, dim=0)
        ])
        
        for b in range(batch_size):
            start = node_ptr[b].item()
            end = node_ptr[b + 1].item()
            n_atoms = end - start
            
            if n_atoms > 0:
                output[b, 3:3+n_atoms, :] = coord_update[start:end, :]
        
        return output
    
    def prepare_batch(self, batch: dict) -> dict:
        """
        为GNN准备批次数据
        如果使用GNN dataloader，批次中应该已经包含pyg_batch
        """
        return batch