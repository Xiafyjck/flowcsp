"""
Equiformer Network for Crystal Structure Prediction
优化版本：参数量100M+，强化PXRD编码，集成MatScholar嵌入
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
import math

# e3nn 核心组件
import e3nn
from e3nn import o3
from e3nn.o3 import Irreps, spherical_harmonics

# 注册到网络注册表
from . import BaseCSPNetwork, register_network

# 导入参考实现的组件
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'equiformer_refs'))

# 从参考代码导入核心组件
from graph_attention_transformer import (
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
    增强版 Equiformer 网络 - 参数量100M+
    
    主要特性：
    1. 强化的PXRD编码器（CNN+Transformer）
    2. 更深的网络架构（8层）
    3. 更宽的隐藏层（256维）
    4. MatScholar原子嵌入
    5. 多头注意力机制（16头）
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # 扩展的网络配置
        self.max_atoms = config.get('max_atoms', 60)
        self.hidden_channels = config.get('hidden_channels', 256)  # 增加到256
        self.num_layers = config.get('num_layers', 8)  # 增加到8层
        self.num_heads = config.get('num_heads', 16)  # 增加到16头
        self.lmax = config.get('lmax', 3)  # 增加到3阶球谐
        self.max_radius = config.get('max_radius', 12.0)  # 增加感受野
        self.num_basis = config.get('num_basis', 256)  # 增加基函数数量
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Irreps 配置 - 更丰富的表示
        self.sh_irreps = o3.Irreps.spherical_harmonics(self.lmax)
        
        # 构建更丰富的节点特征 irreps
        irreps_list = [f"{self.hidden_channels}x0e"]  # 标量
        if self.lmax >= 1:
            irreps_list.append(f"{self.hidden_channels//2}x1e")  # 向量
        if self.lmax >= 2:
            irreps_list.append(f"{self.hidden_channels//4}x2e")  # 二阶张量
        if self.lmax >= 3:
            irreps_list.append(f"{self.hidden_channels//8}x3e")  # 三阶张量
        
        self.irreps_node_embedding = o3.Irreps(" + ".join(irreps_list))
        self.irreps_edge_attr = o3.Irreps(f"{self.num_basis}x0e")
        
        # ========== 1. MatScholar 原子嵌入 ==========
        self._init_matscholar_embeddings(config.get('matscholar_path', 'src/networks/matscholar.json'))
        
        # 增强的原子嵌入网络
        self.atom_embedding_net = nn.Sequential(
            nn.Linear(200, self.hidden_channels * 2),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_channels * 2, self.hidden_channels * 2),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_channels * 2),
            nn.Linear(self.hidden_channels * 2, self.hidden_channels)
        )
        
        # ========== 2. 增强的 PXRD 编码器 ==========
        self.pxrd_encoder = AdvancedPXRDEncoder(
            input_dim=11501,
            output_dim=self.hidden_channels,
            hidden_dim=self.hidden_channels * 2
        )
        
        # ========== 3. 时间编码（使用正弦位置编码） ==========
        self.time_encoder = TimeEncoder(
            hidden_dim=self.hidden_channels,
            max_period=1000
        )
        
        # ========== 4. 初始特征融合 ==========
        # 三种特征的融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 2),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels)
        )
        
        # 标量到 irreps 的映射
        self.scalar_to_irreps = o3.Linear(
            irreps_in=o3.Irreps(f"{self.hidden_channels}x0e"),
            irreps_out=self.irreps_node_embedding,
            internal_weights=True,
            shared_weights=True
        )
        
        # ========== 5. 径向基函数和边嵌入 ==========
        self.rbf = GaussianRadialBasisLayer(
            num_basis=self.num_basis,
            cutoff=self.max_radius
        )
        
        # 增强的边嵌入网络
        self.edge_embedding = nn.Sequential(
            nn.Linear(self.num_basis, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.num_basis)
        )
        
        # ========== 6. 主干网络 - 深层 Equivariant Blocks ==========
        self.blocks = nn.ModuleList()
        
        for layer_idx in range(self.num_layers):
            block = EnhancedEquivariantBlock(
                irreps_node=self.irreps_node_embedding,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_sh=self.sh_irreps,
                num_heads=self.num_heads,
                fc_neurons=[self.hidden_channels * 2, self.hidden_channels * 2],
                norm_type='layer',
                dropout=self.dropout_rate,
                use_gate=True  # 使用门控机制
            )
            self.blocks.append(block)
        
        # ========== 7. 输出头 ==========
        # 晶格输出（等变）
        self.lattice_output_net = nn.ModuleList([
            o3.Linear(
                self.irreps_node_embedding,
                o3.Irreps(f"{self.hidden_channels//2}x0e + {self.hidden_channels//4}x1e"),
                internal_weights=True,
                shared_weights=True
            ),
            o3.Linear(
                o3.Irreps(f"{self.hidden_channels//2}x0e + {self.hidden_channels//4}x1e"),
                o3.Irreps("3x1e"),
                internal_weights=True,
                shared_weights=True
            )
        ])
        
        # 分数坐标输出
        self.irreps_to_scalar = o3.Linear(
            irreps_in=self.irreps_node_embedding,
            irreps_out=o3.Irreps(f"{self.hidden_channels}x0e"),
            internal_weights=True,
            shared_weights=True
        )
        
        self.frac_coords_output = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels * 2),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_channels),
            nn.Linear(self.hidden_channels, 3)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_matscholar_embeddings(self, matscholar_path: str):
        """加载并初始化 MatScholar 嵌入"""
        matscholar_path = Path(matscholar_path)
        
        with open(matscholar_path, 'r') as f:
            embeddings_dict = json.load(f)
        
        # 创建完整的元素符号到原子序数映射
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
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
            'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
            'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103
        }
        
        # 创建嵌入矩阵
        max_z = 103
        embedding_dim = 200
        embeddings = torch.zeros(max_z + 1, embedding_dim)
        
        # 填充已知元素的嵌入
        for element, z in element_to_z.items():
            if element in embeddings_dict:
                embeddings[z] = torch.tensor(embeddings_dict[element], dtype=torch.float32)
            else:
                # 对未知元素使用随机初始化
                embeddings[z] = torch.randn(embedding_dim) * 0.02
        
        # 注册为buffer而不是parameter，因为不需要梯度
        self.register_buffer('matscholar_embeddings', embeddings)
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
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
        
        # ========== 1. 数据提取 ==========
        lattice = z[:, :3, :]  # [batch_size, 3, 3]
        frac_coords = z[:, 3:, :]  # [batch_size, 60, 3]
        comp = conditions['comp']  # [batch_size, 60]
        pxrd = conditions['pxrd']  # [batch_size, 11501]
        num_atoms = conditions['num_atoms']  # [batch_size]
        
        # ========== 2. 构建图数据 ==========
        pos_list = []
        batch_idx = []
        atom_types_list = []
        
        for i in range(batch_size):
            n = int(num_atoms[i].item())
            if n > 0:
                pos_list.append(frac_coords[i, :n, :])
                batch_idx.extend([i] * n)
                atom_types_list.append(comp[i, :n])
        
        if len(pos_list) == 0:
            # 处理空批次
            return torch.zeros_like(z)
        
        pos = torch.cat(pos_list, dim=0)  # [total_atoms, 3]
        batch_idx = torch.tensor(batch_idx, device=device, dtype=torch.long)
        atom_types = torch.cat(atom_types_list, dim=0).long()  # [total_atoms]
        
        # 转换到笛卡尔坐标
        cart_pos_list = []
        for i in range(batch_size):
            n = int(num_atoms[i].item())
            if n > 0:
                cart = frac_coords[i, :n, :] @ lattice[i].T
                cart_pos_list.append(cart)
        cart_pos = torch.cat(cart_pos_list, dim=0) if cart_pos_list else torch.zeros(0, 3, device=device)
        
        # 构建图（考虑周期性）
        edge_index, edge_dist, edge_vec = self._build_graph_with_pbc(
            pos, cart_pos, lattice, batch_idx, num_atoms
        )
        
        # ========== 3. 特征编码 ==========
        # 原子嵌入
        atom_embeddings = self.matscholar_embeddings[atom_types]  # [total_atoms, 200]
        atom_features = self.atom_embedding_net(atom_embeddings)  # [total_atoms, hidden_channels]
        
        # PXRD 编码
        pxrd_features = self.pxrd_encoder(pxrd)  # [batch_size, hidden_channels]
        pxrd_per_atom = pxrd_features[batch_idx]  # [total_atoms, hidden_channels]
        
        # 时间编码
        time_features = self.time_encoder(t, r)  # [batch_size, hidden_channels]
        time_per_atom = time_features[batch_idx]  # [total_atoms, hidden_channels]
        
        # 特征融合
        combined_features = torch.cat([atom_features, pxrd_per_atom, time_per_atom], dim=-1)
        node_scalar_features = self.feature_fusion(combined_features)  # [total_atoms, hidden_channels]
        
        # 转换到 irreps 空间
        node_features = self.scalar_to_irreps(node_scalar_features)  # [total_atoms, irreps]
        
        # ========== 4. 边特征 ==========
        if edge_index.shape[1] > 0:
            # 径向基函数
            edge_rbf = self.rbf(edge_dist)  # [num_edges, num_basis]
            edge_attr = self.edge_embedding(edge_rbf)  # [num_edges, num_basis]
            
            # 球谐函数
            edge_sh = spherical_harmonics(self.sh_irreps, edge_vec, normalize=True)
        else:
            edge_attr = torch.zeros(0, self.num_basis, device=device)
            edge_sh = torch.zeros(0, self.sh_irreps.dim, device=device)
        
        # ========== 5. 通过 Equivariant 层 ==========
        for block in self.blocks:
            node_features = block(
                node_features,
                edge_index,
                edge_attr,
                edge_sh,
                batch_idx
            )
        
        # ========== 6. 输出 ==========
        # 晶格速度
        lattice_velocity = []
        for i in range(batch_size):
            mask = batch_idx == i
            if mask.any():
                # 平均池化
                batch_features = node_features[mask].mean(dim=0, keepdim=True)
                # 两层输出网络
                for layer in self.lattice_output_net:
                    batch_features = layer(batch_features)
                lattice_v = batch_features.reshape(3, 3)
            else:
                lattice_v = torch.zeros(3, 3, device=device)
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
            if n > 0:
                frac_velocity[i, :n, :] = frac_velocity_flat[start_idx:start_idx+n]
                start_idx += n
        
        # 组合输出
        output = torch.zeros_like(z)
        output[:, :3, :] = lattice_velocity
        output[:, 3:, :] = frac_velocity
        
        return output
    
    def _build_graph_with_pbc(
        self,
        frac_pos: torch.Tensor,
        cart_pos: torch.Tensor,
        lattice: torch.Tensor,
        batch_idx: torch.Tensor,
        num_atoms: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建全连接图，考虑周期性边界条件计算最短距离
        使用27个邻近晶胞来找到真正的最短周期性距离
        
        Returns:
            edge_index: [2, num_edges] 全连接图的边索引
            edge_dist: [num_edges] 最短周期性距离
            edge_vec: [num_edges, 3] 最短距离对应的边向量
        """
        device = frac_pos.device
        batch_size = len(num_atoms)
        
        if len(cart_pos) == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, device=device),
                torch.zeros((0, 3), device=device)
            )
        
        # 生成27个邻近单元的偏移（3x3x3），用于计算最短周期性距离
        offsets = torch.tensor(
            [[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]],
            device=device, dtype=torch.float32
        )
        
        edge_index_list = []
        edge_dist_list = []
        edge_vec_list = []
        
        for b in range(batch_size):
            mask = batch_idx == b
            n = int(num_atoms[b].item())
            
            if n <= 1:
                continue
            
            batch_cart = cart_pos[mask]  # [n, 3]
            batch_frac = frac_pos[mask]  # [n, 3]
            batch_lattice = lattice[b]  # [3, 3]
            
            # 创建全连接图的边索引（除了自环）
            # 生成所有可能的原子对
            src = torch.arange(n, device=device).repeat_interleave(n)
            dst = torch.arange(n, device=device).repeat(n)
            
            # 移除自环
            mask_no_self = src != dst
            src = src[mask_no_self]
            dst = dst[mask_no_self]
            
            # 向量化计算每条边的最短周期性距离
            num_edges = src.shape[0]
            
            # 获取所有边的起点和终点坐标
            pos_src = batch_cart[src]  # [num_edges, 3]
            pos_dst = batch_cart[dst]  # [num_edges, 3]
            
            # 扩展维度以计算所有可能的周期性距离
            pos_src_expanded = pos_src.unsqueeze(1)  # [num_edges, 1, 3]
            pos_dst_expanded = pos_dst.unsqueeze(1)  # [num_edges, 1, 3]
            
            # 计算所有偏移向量
            offset_vecs = offsets @ batch_lattice.T  # [27, 3] 转换到笛卡尔坐标
            offset_vecs_expanded = offset_vecs.unsqueeze(0)  # [1, 27, 3]
            
            # 计算所有可能的周期性位置
            shifted_pos_dst = pos_dst_expanded + offset_vecs_expanded  # [num_edges, 27, 3]
            
            # 计算所有距离
            diffs = pos_src_expanded - shifted_pos_dst  # [num_edges, 27, 3]
            distances = torch.norm(diffs, dim=2)  # [num_edges, 27]
            
            # 找到最短距离及其对应的偏移和向量
            min_distances, min_indices = torch.min(distances, dim=1)  # [num_edges]
            
            # 获取最短距离对应的边向量
            # 使用gather来选择正确的向量
            min_indices_expanded = min_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3)  # [num_edges, 1, 3]
            edge_vec = torch.gather(diffs, 1, min_indices_expanded).squeeze(1)  # [num_edges, 3]
            edge_vec = -edge_vec  # 从src指向dst
            
            # 可选：根据最大半径过滤边（全连接图可能不需要）
            if self.max_radius < float('inf'):
                valid_edges = min_distances < self.max_radius
                src = src[valid_edges]
                dst = dst[valid_edges]
                min_distances = min_distances[valid_edges]
                edge_vec = edge_vec[valid_edges]
            
            # 转换为全局索引
            global_idx = torch.where(mask)[0]
            src_global = global_idx[src]
            dst_global = global_idx[dst]
            
            edge_index_list.append(torch.stack([src_global, dst_global]))
            edge_dist_list.append(min_distances)
            edge_vec_list.append(edge_vec)
        
        if len(edge_index_list) > 0:
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_dist = torch.cat(edge_dist_list)
            edge_vec = torch.cat(edge_vec_list, dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_dist = torch.zeros(0, device=device)
            edge_vec = torch.zeros((0, 3), device=device)
        
        return edge_index, edge_dist, edge_vec


class EnhancedEquivariantBlock(nn.Module):
    """
    增强的 Equivariant Transformer Block
    使用门控机制和改进的注意力
    """
    
    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_sh: o3.Irreps,
        num_heads: int = 16,
        fc_neurons: List[int] = [256, 256],
        norm_type: str = 'layer',
        dropout: float = 0.1,
        use_gate: bool = True
    ):
        super().__init__()
        
        self.irreps_node = irreps_node
        self.num_heads = num_heads
        self.use_gate = use_gate
        
        # 消息传递网络（使用参考实现的组件）
        if use_gate:
            self.message_mlp = FullyConnectedTensorProductRescaleNormSwishGate(
                irreps_in1=irreps_node,
                irreps_in2=irreps_sh,
                irreps_out=irreps_node,
                bias=True,
                rescale=True,
                norm_layer=norm_type
            )
        else:
            self.message_mlp = FullyConnectedTensorProductRescaleNorm(
                irreps_in1=irreps_node,
                irreps_in2=irreps_sh,
                irreps_out=irreps_node,
                bias=True,
                rescale=True,
                norm_layer=norm_type
            )
        
        # 多头注意力
        scalar_dim = irreps_node[0].mul if len(irreps_node) > 0 else fc_neurons[0]
        
        # Query, Key, Value 投影
        self.q_proj = nn.Linear(scalar_dim, num_heads * fc_neurons[0] // num_heads)
        self.k_proj = nn.Linear(scalar_dim, num_heads * fc_neurons[0] // num_heads)
        self.v_proj = o3.Linear(irreps_node, irreps_node)
        
        # 边特征投影
        self.edge_proj = nn.Linear(irreps_edge_attr.dim, fc_neurons[0])
        
        # 输出投影
        self.out_proj = o3.Linear(irreps_node, irreps_node)
        
        # 归一化和 dropout
        self.norm1 = get_norm_layer(norm_type)(irreps_node) if norm_type else nn.Identity()
        self.norm2 = get_norm_layer(norm_type)(irreps_node) if norm_type else nn.Identity()
        self.dropout = EquivariantDropout(irreps_node, drop_prob=dropout)
        
        # FFN with proper Gate
        # 使用 irreps2gate 正确分解 irreps
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_node)
        
        # 创建Gate的输入irreps: scalars + gates + gated
        gate_irreps_in = irreps_scalars + irreps_gates + irreps_gated
        
        # 创建FFN层
        self.ffn_linear1 = o3.Linear(irreps_node, gate_irreps_in)
        
        # 创建Gate - 为每个部分提供激活函数
        act_scalars = [nn.SiLU()] * len(irreps_scalars) if len(irreps_scalars) > 0 else []
        act_gates = [nn.Sigmoid()] * len(irreps_gates) if len(irreps_gates) > 0 else []
        
        self.ffn_gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=act_scalars,
            irreps_gates=irreps_gates, 
            act_gates=act_gates,
            irreps_gated=irreps_gated
        )
        self.ffn_linear2 = o3.Linear(self.ffn_gate._irreps_out, irreps_node)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_sh: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        
        if edge_index.shape[1] == 0:
            # 没有边的情况，只应用 FFN
            residual = node_features
            out = self.norm1(node_features, batch=batch)
            out = self.ffn_linear1(out)
            out = self.ffn_gate(out)
            out = self.ffn_linear2(out)
            out = self.dropout(out)
            out = residual + out
            out = self.norm2(out, batch=batch)
            return out
        
        src, dst = edge_index
        
        # 提取标量特征用于注意力
        scalar_dim = self.irreps_node[0].mul
        node_scalar = node_features[:, :scalar_dim]
        
        # 计算 Q, K, V
        q = self.q_proj(node_scalar[dst])  # [num_edges, hidden]
        k = self.k_proj(node_scalar[src])  # [num_edges, hidden]
        v = self.v_proj(node_features[src])  # [num_edges, irreps]
        
        # 计算注意力分数
        edge_features = self.edge_proj(edge_attr)  # [num_edges, hidden]
        
        # 多头注意力
        head_dim = q.shape[1] // self.num_heads
        q = q.view(-1, self.num_heads, head_dim)
        k = k.view(-1, self.num_heads, head_dim)
        edge_features = edge_features.view(-1, self.num_heads, head_dim)
        
        # 计算注意力权重
        attn_scores = (q * k).sum(dim=-1) / math.sqrt(head_dim)  # [num_edges, num_heads]
        attn_scores = attn_scores + (q * edge_features).sum(dim=-1) / math.sqrt(head_dim)
        
        # 对每个节点的入边进行 softmax
        attn_weights = torch.zeros_like(attn_scores)
        for node_idx in torch.unique(dst):
            mask = dst == node_idx
            if mask.any():
                attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)
        
        # 应用注意力权重
        attn_weights = attn_weights.mean(dim=1, keepdim=True)  # [num_edges, 1]
        
        # 计算消息
        messages = self.message_mlp(v, edge_sh, batch[src])
        weighted_messages = messages * attn_weights
        
        # 聚合消息
        out = scatter(weighted_messages, dst, dim=0, dim_size=node_features.size(0), reduce='sum')
        
        # 输出投影
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # 残差连接
        out = node_features + out
        out = self.norm1(out, batch=batch)
        
        # FFN
        residual = out
        ffn_out = self.ffn_linear1(out)
        ffn_out = self.ffn_gate(ffn_out)
        ffn_out = self.ffn_linear2(ffn_out)
        ffn_out = self.dropout(ffn_out)
        out = residual + ffn_out
        out = self.norm2(out, batch=batch)
        
        return out


class AdvancedPXRDEncoder(nn.Module):
    """
    高级 PXRD 编码器
    结合 CNN、Transformer 和 LSTM 进行多尺度特征提取
    """
    
    def __init__(self, input_dim: int = 11501, output_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        
        # ========== 1. 多尺度 CNN 分支 ==========
        # 不同尺度的卷积核
        self.conv_branch1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv_branch2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv_branch3 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # ========== 2. Transformer 编码器 ==========
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model=384, max_len=3000)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # ========== 3. LSTM 分支（捕获序列模式） ==========
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # ========== 4. 特征融合 ==========
        # CNN 特征降维
        self.cnn_pool = nn.AdaptiveAvgPool1d(64)
        
        # 最终融合层
        # CNN: 128*3*64 = 24576 -> 需要池化
        # Transformer: 384
        # LSTM: 256
        self.fusion = nn.Sequential(
            nn.Linear(384 + 384 + 256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 注释掉未使用的stats_mlp
        # self.stats_mlp = nn.Sequential(
        #     nn.Linear(5, 64),  # mean, std, max, min, peak_count
        #     nn.ReLU(),
        #     nn.Linear(64, 64)
        # )
    
    def forward(self, pxrd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pxrd: [batch_size, 11501]
        Returns:
            [batch_size, output_dim]
        """
        batch_size = pxrd.shape[0]
        
        # 扩展维度用于卷积
        x_conv = pxrd.unsqueeze(1)  # [batch_size, 1, 11501]
        
        # ========== CNN 分支 ==========
        conv1 = self.conv_branch1(x_conv)  # [batch_size, 128, ~2875]
        conv2 = self.conv_branch2(x_conv)  # [batch_size, 128, ~2875]
        conv3 = self.conv_branch3(x_conv)  # [batch_size, 128, ~2875]
        
        # 池化到固定大小
        conv1 = self.cnn_pool(conv1)  # [batch_size, 128, 64]
        conv2 = self.cnn_pool(conv2)  # [batch_size, 128, 64]
        conv3 = self.cnn_pool(conv3)  # [batch_size, 128, 64]
        
        # 拼接并展平
        cnn_features = torch.cat([conv1, conv2, conv3], dim=1)  # [batch_size, 384, 64]
        cnn_flat = cnn_features.mean(dim=2)  # [batch_size, 384]
        
        # ========== Transformer 分支 ==========
        # 下采样 PXRD 用于 Transformer
        x_trans = F.avg_pool1d(x_conv, kernel_size=4, stride=4)  # [batch_size, 1, ~2875]
        x_trans = x_trans.transpose(1, 2)  # [batch_size, ~2875, 1]
        x_trans = x_trans.expand(-1, -1, 384)  # [batch_size, ~2875, 384]
        
        # 添加位置编码
        x_trans = self.pos_encoding(x_trans)
        
        # Transformer 编码
        trans_out = self.transformer(x_trans)  # [batch_size, ~2875, 384]
        trans_features = trans_out.mean(dim=1)  # [batch_size, 384]
        
        # ========== LSTM 分支 ==========
        # 下采样用于 LSTM
        x_lstm = F.avg_pool1d(x_conv, kernel_size=8, stride=8)  # [batch_size, 1, ~1437]
        x_lstm = x_lstm.transpose(1, 2)  # [batch_size, ~1437, 1]
        
        lstm_out, _ = self.lstm(x_lstm)  # [batch_size, ~1437, 256]
        lstm_features = lstm_out.mean(dim=1)  # [batch_size, 256]
        
        # ========== 全局统计特征 ==========
        stats = torch.stack([
            pxrd.mean(dim=1),
            pxrd.std(dim=1),
            pxrd.max(dim=1)[0],
            pxrd.min(dim=1)[0],
            (pxrd > pxrd.mean(dim=1, keepdim=True) + 2*pxrd.std(dim=1, keepdim=True)).float().sum(dim=1)
        ], dim=1)  # [batch_size, 5]
        
        # ========== 特征融合 ==========
        combined = torch.cat([cnn_flat, trans_features, lstm_features], dim=1)
        output = self.fusion(combined)
        
        return output


class TimeEncoder(nn.Module):
    """
    时间编码器
    使用正弦位置编码 + MLP
    """
    
    def __init__(self, hidden_dim: int = 256, max_period: int = 1000):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_period = max_period
        
        # 正弦编码的频率
        half_dim = hidden_dim // 4
        self.frequencies = torch.exp(
            torch.arange(half_dim) * -(math.log(max_period) / half_dim)
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 修正：输入是hidden_dim，不是hidden_dim//2
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch_size, 1] 时间步
            r: [batch_size, 1] CFG 时间步
        Returns:
            [batch_size, hidden_dim]
        """
        device = t.device
        self.frequencies = self.frequencies.to(device)
        
        # 正弦编码
        t_emb = self._sinusoidal_embedding(t)
        r_emb = self._sinusoidal_embedding(r)
        
        # 拼接
        time_emb = torch.cat([t_emb, r_emb], dim=-1)  # [batch_size, hidden_dim]
        
        # MLP
        output = self.mlp(time_emb)
        
        return output
    
    def _sinusoidal_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """计算正弦位置编码"""
        x = x.squeeze(-1) if x.dim() > 1 else x
        args = x.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class PositionalEncoding(nn.Module):
    """Transformer 的位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]