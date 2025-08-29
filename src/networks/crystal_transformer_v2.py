"""
Crystal Transformer V2 - 增强版晶体结构生成网络
集成MatScholar预训练元素嵌入，增强晶体结构表达能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import math
import json
import os
from . import BaseCSPNetwork, register_network


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码，用于时间步编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size] or [batch_size, 1] 时间步
        Returns:
            [batch_size, dim] 位置编码
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        if x.dim() == 1:
            x = x.unsqueeze(1)
        emb = x[:, :, None] * emb[None, None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.squeeze(1)


class ElementEmbedding(nn.Module):
    """
    基于MatScholar预训练的元素嵌入层
    将原子序数映射到预训练的元素表征
    """
    def __init__(self, embedding_path: str, output_dim: int):
        super().__init__()
        
        # 加载MatScholar元素嵌入
        with open(embedding_path, 'r') as f:
            matscholar_data = json.load(f)
        
        # 元素符号到原子序数的映射
        element_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                          'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                          'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                          'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                          'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                          'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                          'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                          'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                          'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                          'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
                          'Md', 'No', 'Lr']
        
        # 构建嵌入矩阵 [num_elements + 1, 200]，第0个位置用于padding
        embedding_dim = 200
        num_elements = 104  # 103个元素 + 1个padding位置
        embeddings = torch.zeros(num_elements, embedding_dim)
        
        # 填充元素嵌入
        for i, symbol in enumerate(element_symbols):
            if symbol in matscholar_data:
                embeddings[i + 1] = torch.tensor(matscholar_data[symbol])
        
        # 冻结预训练嵌入
        self.register_buffer('embeddings', embeddings)
        
        # 投影层：将200维嵌入投影到所需维度
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_types: [batch_size, num_atoms] 原子序数 (0表示padding)
        Returns:
            [batch_size, num_atoms, output_dim] 元素嵌入
        """
        # 将原子序数作为索引获取嵌入
        atom_types = atom_types.long()
        elem_embeddings = self.embeddings[atom_types]  # [batch_size, num_atoms, 200]
        
        # 投影到目标维度
        return self.projection(elem_embeddings)


class LatticeEncoder(nn.Module):
    """
    专门的晶格参数编码器
    考虑晶格的几何性质和对称性
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # 晶格向量编码
        self.vector_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 几何特征提取
        self.geometry_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim // 2),  # 输入：3个长度+3个角度+6个点积
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, lattice: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lattice: [batch_size, 3, 3] 晶格矩阵
        Returns:
            [batch_size, 3, hidden_dim] 晶格编码
        """
        batch_size = lattice.shape[0]
        
        # 1. 向量编码
        vector_features = self.vector_encoder(lattice)  # [batch_size, 3, hidden_dim]
        
        # 2. 提取几何特征
        # 计算晶格向量长度
        lengths = torch.norm(lattice, dim=-1)  # [batch_size, 3]
        
        # 计算晶格向量间的角度（余弦值）
        a, b, c = lattice[:, 0], lattice[:, 1], lattice[:, 2]
        cos_alpha = F.cosine_similarity(b, c, dim=-1, eps=1e-8).unsqueeze(-1)
        cos_beta = F.cosine_similarity(a, c, dim=-1, eps=1e-8).unsqueeze(-1)
        cos_gamma = F.cosine_similarity(a, b, dim=-1, eps=1e-8).unsqueeze(-1)
        angles = torch.cat([cos_alpha, cos_beta, cos_gamma], dim=-1)  # [batch_size, 3]
        
        # 计算点积矩阵（晶格向量间的相互作用）
        gram_matrix = torch.bmm(lattice, lattice.transpose(-2, -1))  # [batch_size, 3, 3]
        # 提取上三角部分（6个独立值）
        triu_indices = torch.triu_indices(3, 3, offset=0)
        dot_products = gram_matrix[:, triu_indices[0], triu_indices[1]]  # [batch_size, 6]
        
        # 组合几何特征
        geometry_features = torch.cat([lengths, angles, dot_products], dim=-1)  # [batch_size, 12]
        geometry_encoded = self.geometry_encoder(geometry_features)  # [batch_size, hidden_dim]
        
        # 扩展几何特征到3个晶格向量
        geometry_encoded = geometry_encoded.unsqueeze(1).expand(-1, 3, -1)  # [batch_size, 3, hidden_dim]
        
        # 3. 融合向量和几何特征
        combined = torch.cat([vector_features, geometry_encoded], dim=-1)  # [batch_size, 3, hidden_dim*2]
        lattice_features = self.fusion(combined)  # [batch_size, 3, hidden_dim]
        
        return lattice_features


class MultiScalePXRDEncoder(nn.Module):
    """
    多尺度PXRD编码器
    分层提取不同尺度的衍射峰特征
    """
    def __init__(self, pxrd_dim: int, hidden_dim: int):
        super().__init__()
        
        # 多尺度CNN特征提取
        self.conv_blocks = nn.ModuleList()
        self.scale_dims = []
        
        # Scale 1: 细粒度峰（局部特征）
        self.conv_blocks.append(nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(128)
        ))
        self.scale_dims.append(128 * 128)
        
        # Scale 2: 中等尺度峰（区域特征）
        self.conv_blocks.append(nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, stride=8, padding=15),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(64)
        ))
        self.scale_dims.append(64 * 64)
        
        # Scale 3: 粗粒度峰（全局特征）
        self.conv_blocks.append(nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=101, stride=32, padding=50),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32)
        ))
        self.scale_dims.append(16 * 32)
        
        # 特征融合
        total_dim = sum(self.scale_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, pxrd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pxrd: [batch_size, 11501] PXRD谱
        Returns:
            [batch_size, hidden_dim] PXRD特征
        """
        x = pxrd.unsqueeze(1)  # [batch_size, 1, 11501]
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv_block in self.conv_blocks:
            features = conv_block(x)  # [batch_size, C, L]
            features = features.flatten(1)  # [batch_size, C*L]
            multi_scale_features.append(features)
        
        # 拼接并融合
        combined = torch.cat(multi_scale_features, dim=-1)
        return self.fusion(combined)


class CrystalTransformerLayer(nn.Module):
    """
    晶体Transformer层，包含自注意力和晶格-原子交互
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 晶格-原子交互注意力
        self.lattice_atom_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.atom_lattice_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # 交互门控机制
        self.interaction_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len]
        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        # 自注意力
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out
        
        # 晶格-原子交互
        x_norm = self.norm2(x)
        lattice = x_norm[:, :3]  # [batch_size, 3, hidden_dim]
        atoms = x_norm[:, 3:]    # [batch_size, 60, hidden_dim]
        
        # 原子关注晶格
        atoms_mask = mask[:, 3:] if mask is not None else None
        atoms_to_lattice, _ = self.atom_lattice_attn(
            atoms, lattice, lattice, key_padding_mask=None
        )
        
        # 晶格关注原子
        lattice_to_atoms, _ = self.lattice_atom_attn(
            lattice, atoms, atoms, key_padding_mask=atoms_mask
        )
        
        # 门控融合交互结果
        interaction = torch.cat([lattice_to_atoms, atoms_to_lattice], dim=1)
        gate = self.interaction_gate(torch.cat([x, interaction], dim=-1))
        x = x + gate * interaction
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        
        return x


class ConditionalModulation(nn.Module):
    """
    条件调制模块，使用FiLM (Feature-wise Linear Modulation)
    """
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        
        self.scale_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.shift_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            condition: [batch_size, condition_dim]
        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        scale = self.scale_proj(condition).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        shift = self.shift_proj(condition).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        return x * (1 + scale) + shift


@register_network("crystal_transformer_v2")
class CrystalTransformerV2(BaseCSPNetwork):
    """
    增强版晶体结构生成Transformer
    - 集成MatScholar预训练元素嵌入
    - 专门的晶格编码器
    - 多尺度PXRD编码器
    - 增强的条件融合机制
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # 模型配置
        self.hidden_dim = config.get('hidden_dim', 768)
        self.num_layers = config.get('num_layers', 12)
        self.num_heads = config.get('num_heads', 12)
        self.dropout = config.get('dropout', 0.1)
        self.max_atoms = config.get('max_atoms', 60)
        self.pxrd_dim = config.get('pxrd_dim', 11501)
        
        # 元素嵌入路径
        matscholar_path = config.get('matscholar_path', 'src/networks/matscholar.json')
        
        # 1. 晶格编码器
        self.lattice_encoder = LatticeEncoder(self.hidden_dim)
        
        # 2. 元素嵌入层
        self.element_embedding = ElementEmbedding(matscholar_path, self.hidden_dim)
        
        # 3. 分数坐标编码器
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
        # 4. PXRD编码器
        self.pxrd_encoder = MultiScalePXRDEncoder(self.pxrd_dim, self.hidden_dim)
        
        # 5. 时间编码
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 6. 位置编码
        self.lattice_pos_emb = nn.Parameter(torch.randn(1, 3, self.hidden_dim) * 0.02)
        self.atom_pos_emb = nn.Parameter(torch.randn(1, self.max_atoms, self.hidden_dim) * 0.02)
        
        # 7. 输入融合
        self.input_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # 8. Transformer层
        self.transformer_layers = nn.ModuleList([
            CrystalTransformerLayer(self.hidden_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # 9. 条件调制（每3层一次）
        self.condition_modulation = nn.ModuleList([
            ConditionalModulation(self.hidden_dim, self.hidden_dim * 2)
            if i % 3 == 2 else None
            for i in range(self.num_layers)
        ])
        
        # 10. 输出投影
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 3)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)
                
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
            z: [batch_size, 63, 3] 晶格(3) + 分数坐标(60)
            t: [batch_size, 1] 时间步
            r: [batch_size, 1] 另一个时间步
            conditions: 包含 'comp', 'pxrd', 'num_atoms'
        
        Returns:
            [batch_size, 63, 3] 预测的速度场
        """
        batch_size = z.shape[0]
        device = z.device
        
        # 提取条件
        comp = conditions['comp']  # [batch_size, 60]
        pxrd = conditions['pxrd']  # [batch_size, 11501]
        num_atoms = conditions.get('num_atoms', torch.full((batch_size,), self.max_atoms, device=device))
        
        # 1. 编码晶格参数
        lattice = z[:, :3]  # [batch_size, 3, 3]
        lattice_features = self.lattice_encoder(lattice)  # [batch_size, 3, hidden_dim]
        
        # 2. 编码原子信息
        # 从comp提取原子类型（comp的每个值是原子序数）
        atom_types = comp.long()  # [batch_size, 60]
        element_features = self.element_embedding(atom_types)  # [batch_size, 60, hidden_dim]
        
        # 编码分数坐标
        frac_coords = z[:, 3:]  # [batch_size, 60, 3]
        coord_features = self.coord_encoder(frac_coords)  # [batch_size, 60, hidden_dim]
        
        # 融合元素和坐标特征
        atom_features = self.input_fusion(
            torch.cat([element_features, coord_features], dim=-1)
        )  # [batch_size, 60, hidden_dim]
        
        # 3. 组合晶格和原子特征
        x = torch.cat([lattice_features, atom_features], dim=1)  # [batch_size, 63, hidden_dim]
        
        # 4. 添加位置编码
        x[:, :3] += self.lattice_pos_emb
        x[:, 3:] += self.atom_pos_emb
        
        # 5. 编码条件信息
        pxrd_features = self.pxrd_encoder(pxrd)  # [batch_size, hidden_dim]
        t_emb = self.time_emb(t.squeeze(-1))  # [batch_size, hidden_dim]
        r_emb = self.time_emb(r.squeeze(-1))  # [batch_size, hidden_dim]
        
        # 组合全局条件
        global_condition = torch.cat([pxrd_features, t_emb + r_emb], dim=-1)  # [batch_size, hidden_dim*2]
        
        # 6. 创建padding mask
        mask = torch.zeros(batch_size, 63, dtype=torch.bool, device=device)
        for i, n in enumerate(num_atoms):
            if n < self.max_atoms:
                mask[i, 3 + n:] = True
        
        # 7. 通过Transformer层
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, mask)
            
            # 条件调制（每3层一次）
            if self.condition_modulation[i] is not None:
                x = self.condition_modulation[i](x, global_condition)
        
        # 8. 输出投影
        x = self.output_norm(x)
        output = self.output_proj(x)  # [batch_size, 63, 3]
        
        # 9. Mask padding位置
        for i, n in enumerate(num_atoms):
            if n < self.max_atoms:
                output[i, 3 + n:] = 0
        
        return output
    
    def prepare_batch(self, batch: dict) -> dict:
        """准备批次数据"""
        return batch