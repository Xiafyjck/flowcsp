"""
Transformer-based CSP Network
~150M parameters for initial testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import math
from . import BaseCSPNetwork, register_network


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class TimeEmbedding(nn.Module):
    """Time embedding similar to diffusion models"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim // 4, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        
    def forward(self, t):
        # t: [batch_size] or [batch_size, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)  # 确保是 [batch_size, 1]
        half_dim = self.dim // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, :, None] * emb[None, None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb.squeeze(1)  # [batch_size, dim]


@register_network("transformer")
class TransformerCSPNetwork(BaseCSPNetwork):
    """
    Transformer network for CSP generation
    Processes lattice and coordinates together with attention mechanism
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Default configuration
        hidden_dim = config.get('hidden_dim', 512)
        num_layers = config.get('num_layers', 8)
        num_heads = config.get('num_heads', 8)
        dropout = config.get('dropout', 0.1)
        max_atoms = config.get('max_atoms', 60)
        pxrd_dim = config.get('pxrd_dim', 11501)
        
        # Calculate total sequence length: 3 lattice vectors + 60 atom positions
        self.seq_len = 3 + max_atoms  # 63
        self.max_atoms = max_atoms
        
        # Input projections
        self.coord_proj = nn.Linear(3, hidden_dim)  # Each vector/position is 3D
        self.comp_proj = nn.Linear(max_atoms, hidden_dim)  # Composition embedding
        
        # 目标PXRD投影（要生成的结构对应的PXRD）
        # 路径1: MLP分支 - 全局特征提取
        self.pxrd_target_proj = nn.Sequential(
            nn.Linear(pxrd_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        
        # 路径2: CNN分支 - 局部模式提取
        # 1D CNN for PXRD spectrum processing
        self.pxrd_cnn = nn.Sequential(
            # 第一层卷积：捕捉细粒度的峰形特征
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 降采样到 ~5750
            
            # 第二层卷积：捕捉中等尺度特征
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 降采样到 ~2875
            
            # 第三层卷积：捕捉更大尺度的模式
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 降采样到 ~1437
            
            # 第四层卷积：进一步抽象
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 降采样到 ~718
            
            # 第五层卷积：高级特征提取
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),  # 固定输出长度为64
        )
        
        # CNN特征投影到hidden_dim
        self.pxrd_cnn_proj = nn.Sequential(
            nn.Linear(1024 * 64, 2048),  # 将flattened CNN特征映射
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, hidden_dim)
        )
        
        # 特征融合层：融合MLP和CNN两路特征
        self.pxrd_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 门控机制：自适应调节两路特征的权重
        self.pxrd_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 输出2个权重
            nn.Softmax(dim=-1)
        )
        
        # 移除了实时PXRD encoder、门控机制和PXRD质量评估器，不再支持实时PXRD
        
        # Time embeddings for t and r
        self.time_emb_t = TimeEmbedding(hidden_dim)
        self.time_emb_r = TimeEmbedding(hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Type embeddings to distinguish lattice vs atoms
        self.type_embedding = nn.Embedding(2, hidden_dim)  # 0: lattice, 1: atom
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)  # Output 3D coordinates
        )
        
        # ResNet-style MLP for condition fusion
        # 输入: 4个条件（comp_emb, pxrd_features, t_emb, r_emb），每个都是hidden_dim维
        self.cond_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # ResNet连接：将融合后的特征与原始comp_emb相加，保留重要信息
        self.cond_residual = nn.Linear(hidden_dim, hidden_dim)
        
        # Condition projection for adaptive normalization
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim * 2)  # For scale and shift
        
    def forward(
        self, 
        z: torch.Tensor,           
        t: torch.Tensor,           
        r: torch.Tensor,           
        conditions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            z: [batch_size, 63, 3] - lattice (3) + frac_coords (60)
            t: [batch_size, 1] - time step
            r: [batch_size, 1] - another time step for flow
            conditions: dict with 'comp', 'pxrd', 'num_atoms'
        
        Returns:
            [batch_size, 63, 3] - predicted update/velocity
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Extract conditions
        comp = conditions['comp']  # [batch_size, 60] - 原子组成
        pxrd = conditions['pxrd']  # [batch_size, 11501] - 目标PXRD谱
        num_atoms = conditions.get('num_atoms', torch.full((batch_size,), self.max_atoms, device=device))
        
        # Project input coordinates [batch_size, 63, 3] -> [batch_size, 63, hidden_dim]
        x = self.coord_proj(z)
        
        # 调试：检查x的形状
        if x.shape[1] != self.seq_len:
            print(f"WARNING: x shape after proj: {x.shape}, expected seq_len: {self.seq_len}")
            print(f"z shape: {z.shape}")
        
        # Add type embeddings (0 for lattice, 1 for atoms)
        type_ids = torch.cat([
            torch.zeros(batch_size, 3, dtype=torch.long, device=device),
            torch.ones(batch_size, self.max_atoms, dtype=torch.long, device=device)
        ], dim=1)
        x = x + self.type_embedding(type_ids)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process conditions
        comp_emb = self.comp_proj(comp)  # [batch_size, hidden_dim]
        
        # Time embeddings - 先计算，因为门控机制需要t_emb
        t_emb = self.time_emb_t(t)  # [batch_size, hidden_dim]
        r_emb = self.time_emb_r(r)  # [batch_size, hidden_dim]
        
        # PXRD embeddings - 双路径处理
        # 路径1: MLP处理全局特征
        pxrd_mlp_features = self.pxrd_target_proj(pxrd)  # [batch_size, hidden_dim]
        
        # 路径2: CNN处理局部模式
        # 将PXRD谱重塑为CNN输入格式 [batch_size, 1, 11501]
        pxrd_cnn_input = pxrd.unsqueeze(1)  # 添加channel维度
        pxrd_cnn_out = self.pxrd_cnn(pxrd_cnn_input)  # [batch_size, 1024, 64]
        
        # Flatten CNN输出并投影
        pxrd_cnn_flat = pxrd_cnn_out.view(batch_size, -1)  # [batch_size, 1024*64]
        pxrd_cnn_features = self.pxrd_cnn_proj(pxrd_cnn_flat)  # [batch_size, hidden_dim]
        
        # 融合两路特征
        # 方法1: 门控融合 - 自适应权重
        pxrd_concat = torch.cat([pxrd_mlp_features, pxrd_cnn_features], dim=-1)  # [batch_size, hidden_dim*2]
        gate_weights = self.pxrd_gate(pxrd_concat)  # [batch_size, 2]
        
        # 应用门控权重
        pxrd_gated = (pxrd_mlp_features * gate_weights[:, 0:1] + 
                     pxrd_cnn_features * gate_weights[:, 1:2])  # [batch_size, hidden_dim]
        
        # 方法2: 特征融合层 - 深度融合
        pxrd_fused = self.pxrd_fusion(pxrd_concat)  # [batch_size, hidden_dim]
        
        # 最终PXRD特征：门控特征 + 深度融合特征
        pxrd_features = pxrd_gated + pxrd_fused  # [batch_size, hidden_dim]
        
        # Combine all conditions
        global_cond = comp_emb + pxrd_features + t_emb + r_emb  # [batch_size, hidden_dim]
        
        # 调试：检查维度
        if x.dim() != 3 or global_cond.dim() != 2:
            print(f"Shape mismatch: x={x.shape}, global_cond={global_cond.shape}")
        
        # Add global conditioning to each position
        x = x + global_cond.unsqueeze(1)
        
        # Create attention mask for padding (atoms beyond num_atoms should be masked)
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=device)
        for i, n in enumerate(num_atoms):
            if n < self.max_atoms:
                mask[i, 3 + n:] = True  # Mask positions after actual atoms
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Adaptive normalization using conditions with ResNet-style fusion
        # 步骤1: 融合所有4个条件
        cond_combined = torch.cat([comp_emb, pxrd_features, t_emb, r_emb], dim=-1)  # [batch_size, hidden_dim * 4]
        cond_fused = self.cond_fusion(cond_combined)  # [batch_size, hidden_dim]
        
        # 步骤2: ResNet残差连接 - 保留原始comp_emb信息
        cond_final = cond_fused + self.cond_residual(comp_emb)  # [batch_size, hidden_dim]
        
        # 步骤3: 生成scale和shift用于adaptive normalization
        scale_shift = self.cond_proj(cond_final)  # [batch_size, hidden_dim * 2]
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        shift = shift.unsqueeze(1)
        
        # 步骤4: 应用adaptive normalization
        x = x * (1 + scale) + shift
        
        # Project to output space [batch_size, 63, 3]
        output = self.output_proj(x)
        
        # Zero out padded positions
        for i, n in enumerate(num_atoms):
            if n < self.max_atoms:
                output[i, 3 + n:] = 0
        
        return output
    
    def prepare_batch(self, batch: dict) -> dict:
        """Prepare batch - transformer doesn't need special format"""
        return batch