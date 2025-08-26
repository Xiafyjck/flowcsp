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
        # t: [batch_size, 1]
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
        max_atoms = config.get('max_atoms', 52)
        pxrd_dim = config.get('pxrd_dim', 11501)
        
        # Calculate total sequence length: 3 lattice vectors + 52 atom positions
        self.seq_len = 3 + max_atoms  # 55
        self.max_atoms = max_atoms
        
        # Input projections
        self.coord_proj = nn.Linear(3, hidden_dim)  # Each vector/position is 3D
        self.comp_proj = nn.Linear(max_atoms, hidden_dim)  # Composition embedding
        
        # 目标PXRD投影（要生成的结构对应的PXRD）
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
        
        # 实时PXRD encoder（用于真实计算的PXRD，如果有）
        self.pxrd_real_encoder = nn.Sequential(
            nn.Linear(pxrd_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        
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
        
        # Condition projection for adaptive normalization
        self.cond_proj = nn.Linear(hidden_dim * 3, hidden_dim * 2)  # For scale and shift
        
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
            z: [batch_size, 55, 3] - lattice (3) + frac_coords (52)
            t: [batch_size, 1] - time step
            r: [batch_size, 1] - another time step for flow
            conditions: dict with 'comp', 'pxrd', 'pxrd_realtime', 'num_atoms'
        
        Returns:
            [batch_size, 55, 3] - predicted update/velocity
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Extract conditions
        comp = conditions['comp']  # [batch_size, 52] - 原子组成
        pxrd_target = conditions['pxrd']  # [batch_size, 11501] - 目标PXRD谱
        pxrd_realtime = conditions.get('pxrd_realtime', None)  # [batch_size, 11501] - 实时PXRD谱（可选）
        num_atoms = conditions.get('num_atoms', torch.full((batch_size,), self.max_atoms, device=device))
        
        # Project input coordinates [batch_size, 55, 3] -> [batch_size, 55, hidden_dim]
        x = self.coord_proj(z)
        
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
        pxrd_target_emb = self.pxrd_target_proj(pxrd_target)  # [batch_size, hidden_dim]
        
        # 处理PXRD特征融合
        # 基础特征：总是使用目标PXRD
        pxrd_features = pxrd_target_emb  # [batch_size, hidden_dim]
        
        # 如果有真实计算的PXRD，添加其特征
        if pxrd_realtime is not None:
            pxrd_sim_emb = self.pxrd_real_encoder(pxrd_realtime)  # [batch_size, hidden_dim]
            pxrd_features = pxrd_features + pxrd_sim_emb  # 特征融合
        
        # Time embeddings
        t_emb = self.time_emb_t(t)  # [batch_size, hidden_dim]
        r_emb = self.time_emb_r(r)  # [batch_size, hidden_dim]
        
        # Combine all conditions
        global_cond = comp_emb + pxrd_features + t_emb + r_emb  # [batch_size, hidden_dim]
        
        # Add global conditioning to each position
        x = x + global_cond.unsqueeze(1)
        
        # Create attention mask for padding (atoms beyond num_atoms should be masked)
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=device)
        for i, n in enumerate(num_atoms):
            if n < self.max_atoms:
                mask[i, 3 + n:] = True  # Mask positions after actual atoms
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Adaptive normalization using conditions - 现在包含两个PXRD
        cond_combined = torch.cat([comp_emb, pxrd_target_emb + pxrd_realtime_emb, t_emb + r_emb], dim=-1)
        scale_shift = self.cond_proj(cond_combined)  # [batch_size, hidden_dim * 2]
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        shift = shift.unsqueeze(1)
        
        x = x * (1 + scale) + shift
        
        # Project to output space [batch_size, 55, 3]
        output = self.output_proj(x)
        
        # Zero out padded positions
        for i, n in enumerate(num_atoms):
            if n < self.max_atoms:
                output[i, 3 + n:] = 0
        
        return output
    
    def prepare_batch(self, batch: dict) -> dict:
        """Prepare batch - transformer doesn't need special format"""
        return batch