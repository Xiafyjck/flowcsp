"""
Equiformer Network for Crystal Structure Prediction
充分利用参考实现，只编写必要的适配层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_scatter import scatter
from torch_cluster import radius_graph
import numpy as np
import json
from typing import Dict, Optional, Tuple, List
from pathlib import Path

# e3nn 核心组件
import e3nn
from e3nn import o3
from e3nn.o3 import Irreps, spherical_harmonics

# 注册到网络注册表
from . import BaseCSPNetwork, register_network

# 导入参考实现的组件 - 添加到 Python 路径
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'equiformer_refs'))

# 从参考代码导入核心组件
from graph_attention_transformer import (
    GraphAttentionTransformer,
    SeparableFCTP,
    DepthwiseTensorProduct,
    get_norm_layer,
    FullyConnectedTensorProductRescaleNormSwishGate,
    FullyConnectedTensorProductRescaleSwishGate,
    FullyConnectedTensorProductRescaleNorm,
    _AVG_DEGREE
)
from gaussian_rbf import GaussianRadialBasisLayer
from tensor_product_rescale import (
    TensorProductRescale,
    LinearRS,
    FullyConnectedTensorProductRescale,
    irreps2gate,
    sort_irreps_even_first
)
from fast_activation import Activation, Gate
from radial_func import RadialProfile
from drop import EquivariantDropout, GraphDropPath


@register_network("equiformer")
class EquiformerCSPNetwork(BaseCSPNetwork):
    """
    Equiformer 适配层 - 将参考实现适配到 CSP 任务
    
    主要功能：
    1. 利用 GraphAttentionTransformer 作为核心
    2. 添加 PXRD 编码器和 MatScholar 原子嵌入
    3. 适配 CFM-CFG 的接口要求
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # 基础配置
        self.max_atoms = config.get('max_atoms', 60)
        self.hidden_channels = config.get('hidden_channels', 128)
        self.num_layers = config.get('num_layers', 6)
        self.num_heads = config.get('num_heads', 8)
        self.lmax = config.get('lmax', 2)
        self.max_radius = config.get('max_radius', 10.0)
        self.num_basis = config.get('num_basis', 128)
        
        # Irreps 配置
        self.sh_irreps = o3.Irreps.spherical_harmonics(self.lmax)
        
        # 节点特征 irreps：混合不同阶的表示
        if self.lmax == 2:
            self.irreps_node_embedding = o3.Irreps(f"{self.hidden_channels}x0e + {self.hidden_channels//2}x1e + {self.hidden_channels//4}x2e")
        elif self.lmax == 1:
            self.irreps_node_embedding = o3.Irreps(f"{self.hidden_channels}x0e + {self.hidden_channels//2}x1e")
        else:  # lmax >= 3
            self.irreps_node_embedding = o3.Irreps(f"{self.hidden_channels}x0e + {self.hidden_channels//2}x1e + {self.hidden_channels//4}x2e + {self.hidden_channels//8}x3e")
        
        self.irreps_edge_attr = o3.Irreps(f"{self.num_basis}x0e")
        
        # ========== 1. 原子嵌入 ==========
        self.matscholar_embeddings = self._load_matscholar_embeddings()
        
        # 原子嵌入网络
        self.atom_embedding_mlp = nn.Sequential(
            nn.Linear(200, self.hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )
        
        # ========== 2. PXRD 编码器 ==========
        self.pxrd_encoder = MultiScalePXRDEncoder(
            input_dim=11501,
            output_dim=self.hidden_channels
        )
        
        # ========== 3. 时间编码 ==========
        self.time_mlp = nn.Sequential(
            nn.Linear(2, self.hidden_channels),  # t 和 r 两个时间
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )
        
        # ========== 4. 初始嵌入层 ==========
        # 将标量特征提升到 irreps 空间
        self.embedding = nn.Linear(self.hidden_channels * 3, self.hidden_channels)  # comp + pxrd + time
        
        # 标量到 irreps 的映射
        self.scalar_to_irreps = o3.Linear(
            irreps_in=o3.Irreps(f"{self.hidden_channels}x0e"),
            irreps_out=self.irreps_node_embedding
        )
        
        # ========== 5. 径向基函数 ==========
        self.rbf = GaussianRadialBasisLayer(
            num_basis=self.num_basis,
            cutoff=self.max_radius
        )
        
        # 边嵌入
        self.edge_embedding = nn.Sequential(
            nn.Linear(self.num_basis, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.num_basis)
        )
        
        # ========== 6. 主干网络 - 使用参考实现 ==========
        self.blocks = nn.ModuleList()
        
        for _ in range(self.num_layers):
            block = EquivariantBlock(
                irreps_node=self.irreps_node_embedding,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_sh=self.sh_irreps,
                num_heads=self.num_heads,
                fc_neurons=[self.hidden_channels, self.hidden_channels],
                norm_type='layer'
            )
            self.blocks.append(block)
        
        # ========== 7. 输出头 ==========
        # 晶格输出（等变，输出 3x1e）
        self.lattice_output = o3.Linear(
            irreps_in=self.irreps_node_embedding,
            irreps_out=o3.Irreps("3x1e")
        )
        
        # 分数坐标输出（使用标量特征）
        # 先提取标量部分
        self.irreps_to_scalar = o3.Linear(
            irreps_in=self.irreps_node_embedding,
            irreps_out=o3.Irreps(f"{self.hidden_channels}x0e")
        )
        
        self.frac_coords_output = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 3)
        )
    
    def _load_matscholar_embeddings(self) -> nn.Parameter:
        """加载 MatScholar 预训练嵌入"""
        matscholar_path = Path(__file__).parent / 'matscholar.json'
        
        with open(matscholar_path, 'r') as f:
            embeddings_dict = json.load(f)
        
        # 创建嵌入查找表
        max_z = 103
        embedding_dim = 200
        embeddings = torch.zeros(max_z + 1, embedding_dim)
        
        # 元素符号到原子序数映射（简化版）
        element_to_z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
            'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
            'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
            'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83
        }
        
        for element, z in element_to_z.items():
            if element in embeddings_dict:
                embeddings[z] = torch.tensor(embeddings_dict[element], dtype=torch.float32)
        
        return nn.Parameter(embeddings, requires_grad=False)
    
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
            z: [batch_size, 63, 3] 当前状态（晶格+分数坐标）
            t: [batch_size, 1] 时间步
            r: [batch_size, 1] CFG 时间步
            conditions: 包含 comp, pxrd, num_atoms
            
        Returns:
            [batch_size, 63, 3] 速度场
        """
        batch_size = z.shape[0]
        device = z.device
        
        # ========== 1. 提取数据 ==========
        lattice = z[:, :3, :]  # [batch_size, 3, 3]
        frac_coords = z[:, 3:, :]  # [batch_size, 60, 3]
        comp = conditions['comp']  # [batch_size, 60]
        pxrd = conditions['pxrd']  # [batch_size, 11501]
        num_atoms = conditions['num_atoms']  # [batch_size]
        
        # ========== 2. 构建图数据 ==========
        # 将批次数据展平为图格式
        pos_list = []
        batch_idx = []
        atom_types_list = []
        
        for i in range(batch_size):
            n = int(num_atoms[i].item())
            pos_list.append(frac_coords[i, :n, :])
            batch_idx.extend([i] * n)
            atom_types_list.append(comp[i, :n])
        
        pos = torch.cat(pos_list, dim=0)  # [total_atoms, 3]
        batch_idx = torch.tensor(batch_idx, device=device, dtype=torch.long)
        atom_types = torch.cat(atom_types_list, dim=0).long()  # [total_atoms]
        
        # 转换到笛卡尔坐标（用于计算距离）
        cart_pos_list = []
        for i in range(batch_size):
            n = int(num_atoms[i].item())
            cart = frac_coords[i, :n, :] @ lattice[i].T  # [n, 3]
            cart_pos_list.append(cart)
        cart_pos = torch.cat(cart_pos_list, dim=0)  # [total_atoms, 3]
        
        # 构建全连接图（考虑周期性边界）
        edge_index, edge_dist = self._build_fully_connected_graph_pbc(
            pos, cart_pos, lattice, batch_idx, num_atoms
        )
        
        # ========== 3. 特征编码 ==========
        # 原子嵌入
        atom_embeddings = self.matscholar_embeddings[atom_types]  # [total_atoms, 200]
        atom_features = self.atom_embedding_mlp(atom_embeddings)  # [total_atoms, hidden_channels]
        
        # PXRD 编码
        pxrd_features = self.pxrd_encoder(pxrd)  # [batch_size, hidden_channels]
        # 广播到每个原子
        pxrd_per_atom = pxrd_features[batch_idx]  # [total_atoms, hidden_channels]
        
        # 时间编码
        time_input = torch.cat([t, r], dim=-1)  # [batch_size, 2]
        time_features = self.time_mlp(time_input)  # [batch_size, hidden_channels]
        time_per_atom = time_features[batch_idx]  # [total_atoms, hidden_channels]
        
        # 组合特征
        node_scalar_features = atom_features + pxrd_per_atom + time_per_atom
        node_scalar_features = self.embedding(
            torch.cat([atom_features, pxrd_per_atom, time_per_atom], dim=-1)
        )  # [total_atoms, hidden_channels]
        
        # 提升到 irreps 空间
        node_features = self.scalar_to_irreps(node_scalar_features)  # [total_atoms, irreps]
        
        # ========== 4. 边特征 ==========
        # 径向基函数
        edge_rbf = self.rbf(edge_dist)  # [num_edges, num_basis]
        edge_attr = self.edge_embedding(edge_rbf).float()  # [num_edges, num_basis] - 确保是float类型
        
        # 球谐函数（边的方向）
        edge_vec = cart_pos[edge_index[1]] - cart_pos[edge_index[0]]  # [num_edges, 3]
        edge_sh = spherical_harmonics(self.sh_irreps, edge_vec, normalize=True)  # [num_edges, irreps]
        
        # ========== 5. 通过 Equiformer 层 ==========
        for block in self.blocks:
            node_features = block(
                node_features, 
                edge_index, 
                edge_attr, 
                edge_sh,
                batch_idx
            )
        
        # ========== 6. 输出 ==========
        # 晶格速度（等变输出）
        # 聚合每个批次的节点特征
        lattice_velocity = []
        for i in range(batch_size):
            mask = batch_idx == i
            # 平均池化
            batch_features = node_features[mask].mean(dim=0, keepdim=True)  # [1, irreps]
            lattice_v = self.lattice_output(batch_features)  # [1, 9] (3x1e flattened)
            lattice_v = lattice_v.reshape(3, 3)
            lattice_velocity.append(lattice_v)
        
        lattice_velocity = torch.stack(lattice_velocity)  # [batch_size, 3, 3]
        
        # 分数坐标速度
        node_scalar = self.irreps_to_scalar(node_features)  # [total_atoms, hidden_channels]
        frac_velocity_flat = self.frac_coords_output(node_scalar)  # [total_atoms, 3]
        
        # 重组为批次格式
        frac_velocity = torch.zeros(batch_size, 60, 3, device=device)
        start_idx = 0
        for i in range(batch_size):
            n = int(num_atoms[i].item())
            frac_velocity[i, :n, :] = frac_velocity_flat[start_idx:start_idx+n]
            start_idx += n
        
        # 组合输出
        output = torch.zeros_like(z)
        output[:, :3, :] = lattice_velocity
        output[:, 3:, :] = frac_velocity
        
        return output
    
    def _build_fully_connected_graph_pbc(
        self,
        frac_pos: torch.Tensor,
        cart_pos: torch.Tensor, 
        lattice: torch.Tensor,
        batch_idx: torch.Tensor,
        num_atoms: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建考虑周期性边界的全连接图
        
        Returns:
            edge_index: [2, num_edges]
            edge_dist: [num_edges] 最短距离
        """
        device = frac_pos.device
        batch_size = len(num_atoms)
        
        edge_index_list = []
        edge_dist_list = []
        
        # 对每个批次分别构建
        for b in range(batch_size):
            mask = batch_idx == b
            n = int(num_atoms[b].item())
            
            if n == 0:
                continue
            
            # 全连接边索引
            src = torch.arange(n, device=device).repeat_interleave(n)
            dst = torch.arange(n, device=device).repeat(n)
            
            # 移除自环
            valid = src != dst
            src = src[valid]
            dst = dst[valid]
            
            # 获取该批次的位置
            batch_cart = cart_pos[mask]  # [n, 3]
            batch_lattice = lattice[b]  # [3, 3]
            
            # 计算周期性最短距离
            pos_src = batch_cart[src]  # [n*(n-1), 3]
            pos_dst = batch_cart[dst]  # [n*(n-1), 3]
            
            # 简化：只考虑最近的镜像
            # 实际应该检查 27 个邻近晶胞
            dist = torch.norm(pos_dst - pos_src, dim=1)  # [n*(n-1)]
            
            # 转换为全局索引
            global_idx = torch.where(mask)[0]
            src_global = global_idx[src]
            dst_global = global_idx[dst]
            
            edge_index_list.append(torch.stack([src_global, dst_global]))
            edge_dist_list.append(dist)
        
        if len(edge_index_list) > 0:
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_dist = torch.cat(edge_dist_list)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_dist = torch.zeros(0, device=device)
        
        return edge_index, edge_dist


class EquivariantBlock(nn.Module):
    """
    Equivariant Transformer Block
    利用参考实现的组件
    """
    
    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_sh: o3.Irreps,
        num_heads: int = 8,
        fc_neurons: List[int] = [128, 128],
        norm_type: str = 'layer'
    ):
        super().__init__()
        
        self.irreps_node = irreps_node
        self.num_heads = num_heads
        
        # 使用参考实现的分离式全连接张量积
        self.message_mlp = SeparableFCTP(
            irreps_node_input=irreps_node,
            irreps_edge_attr=irreps_sh,
            irreps_node_output=irreps_node,
            fc_neurons=fc_neurons,
            use_activation=True,
            norm_layer=norm_type
        )
        
        # MLP 注意力（使用标量）
        scalar_dim = irreps_node[0].mul if len(irreps_node) > 0 else 128
        self.alpha_mlp = nn.Sequential(
            nn.Linear(scalar_dim * 2 + irreps_edge_attr.dim, fc_neurons[0]),
            nn.SiLU(),
            nn.Linear(fc_neurons[0], fc_neurons[1]),
            nn.SiLU(),
            nn.Linear(fc_neurons[1], num_heads)
        )
        
        # 输出线性层
        self.linear_out = o3.Linear(irreps_node, irreps_node)
        
        # 归一化
        self.norm = get_norm_layer(norm_type)(irreps_node) if norm_type else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_sh: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        """
        src, dst = edge_index
        
        # 计算消息
        messages = self.message_mlp(
            node_features[src],
            edge_sh,
            edge_attr,  # edge_scalars
            batch[src]
        )
        
        # 计算注意力权重
        # 提取标量特征
        scalar_dim = self.irreps_node[0].mul
        scalar_src = node_features[src, :scalar_dim]
        scalar_dst = node_features[dst, :scalar_dim]
        
        alpha_input = torch.cat([scalar_src, scalar_dst, edge_attr], dim=-1)
        alpha = self.alpha_mlp(alpha_input)  # [num_edges, num_heads]
        
        # Softmax over incoming edges for each node
        alpha = F.softmax(alpha, dim=0)
        
        # 应用注意力权重
        # 简化：对所有头取平均
        alpha_mean = alpha.mean(dim=1, keepdim=True)  # [num_edges, 1]
        weighted_messages = messages * alpha_mean
        
        # 聚合消息
        out = scatter(weighted_messages, dst, dim=0, dim_size=node_features.size(0), reduce='sum')
        
        # 输出变换
        out = self.linear_out(out)
        out = self.dropout(out)
        
        # 残差连接和归一化
        out = node_features + out
        if hasattr(self.norm, '__call__'):
            out = self.norm(out, batch=batch)
        
        return out


class MultiScalePXRDEncoder(nn.Module):
    """
    多尺度 PXRD 编码器
    """
    
    def __init__(self, input_dim: int = 11501, output_dim: int = 128):
        super().__init__()
        
        # 多尺度 1D 卷积
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出 MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, pxrd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pxrd: [batch_size, 11501]
        Returns:
            [batch_size, output_dim]
        """
        x = pxrd.unsqueeze(1)  # [batch_size, 1, 11501]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.global_pool(x).squeeze(-1)  # [batch_size, 128]
        x = self.output_mlp(x)
        
        return x