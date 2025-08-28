"""
Crystal-Specific Transformer Network
~100M parameters optimized for crystal structure generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math
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


class CrossAttentionBlock(nn.Module):
    """交叉注意力块，用于条件融合"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim] 查询序列
            context: [batch_size, context_len, dim] 键值序列
            mask: [batch_size, seq_len] padding mask
        """
        # 交叉注意力
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, context, context, key_padding_mask=mask)
        x = x + attn_out
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class EfficientPXRDEncoder(nn.Module):
    """高效的PXRD编码器，使用分层CNN + 注意力池化"""
    def __init__(self, pxrd_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 分层CNN特征提取器
        # 使用更少的通道和更激进的下采样
        self.conv_layers = nn.ModuleList([
            # Stage 1: 细粒度特征 (11501 -> 2875)
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.GELU(),
            ),
            # Stage 2: 中等尺度特征 (2875 -> 359)
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=9, stride=8, padding=4),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.GELU(),
            ),
            # Stage 3: 粗粒度特征 (359 -> 44)
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=9, stride=8, padding=4),
                nn.BatchNorm1d(128),
                nn.GELU(),
            )
        ])
        
        # 多尺度特征融合
        # 将不同尺度的特征投影到相同维度并融合
        self.scale_projections = nn.ModuleList([
            nn.Conv1d(32, hidden_dim // 4, kernel_size=1),  # 细粒度
            nn.Conv1d(64, hidden_dim // 4, kernel_size=1),  # 中等
            nn.Conv1d(128, hidden_dim // 2, kernel_size=1), # 粗粒度
        ])
        
        # 移除未使用的attention_pool层
        # 使用全局平均池化替代
        
        # 最终投影层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, pxrd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pxrd: [batch_size, 11501] PXRD谱
        Returns:
            [batch_size, hidden_dim] PXRD特征
        """
        batch_size = pxrd.shape[0]
        x = pxrd.unsqueeze(1)  # [batch_size, 1, 11501]
        
        # 多尺度特征提取
        features = []
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            # 投影并池化每个尺度的特征
            feat = self.scale_projections[i](x)  # [batch_size, hidden_dim/n, seq_len]
            # 全局平均池化
            feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # [batch_size, hidden_dim/n]
            features.append(feat)
        
        # 拼接多尺度特征
        multi_scale_feat = torch.cat(features, dim=-1)  # [batch_size, hidden_dim]
        
        # 最终投影
        output = self.output_proj(multi_scale_feat)
        
        return output


class CrystalTransformerBlock(nn.Module):
    """晶体结构专用Transformer块，增强对称性感知"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # 标准自注意力
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # 晶格-原子交互注意力
        # 晶格参数和原子位置之间的特殊交互
        self.norm2 = nn.LayerNorm(dim)
        self.lattice_atom_attn = nn.MultiheadAttention(dim // 2, num_heads // 2, dropout=dropout, batch_first=True)
        self.lattice_proj = nn.Linear(dim, dim // 2)
        self.atom_proj = nn.Linear(dim, dim // 2)
        self.interaction_proj = nn.Linear(dim // 2, dim)
        
        # FFN
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim] 输入序列（晶格+原子）
            mask: [batch_size, seq_len] padding mask
        Returns:
            [batch_size, seq_len, dim] 输出序列
        """
        batch_size, seq_len, dim = x.shape
        
        # 自注意力
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out
        
        # 晶格-原子交互（仅前3个是晶格向量）
        x_norm = self.norm2(x)
        lattice = self.lattice_proj(x_norm[:, :3])  # [batch_size, 3, dim/2]
        atoms = self.atom_proj(x_norm[:, 3:])  # [batch_size, 60, dim/2]
        
        # 原子关注晶格信息
        atoms_mask = mask[:, 3:] if mask is not None else None
        atom_lattice_attn, _ = self.lattice_atom_attn(atoms, lattice, lattice, key_padding_mask=None)
        
        # 晶格关注原子信息
        lattice_atom_attn, _ = self.lattice_atom_attn(lattice, atoms, atoms, key_padding_mask=atoms_mask)
        
        # 组合交互结果
        interaction = torch.cat([lattice_atom_attn, atom_lattice_attn], dim=1)  # [batch_size, 63, dim/2]
        interaction = self.interaction_proj(interaction)  # [batch_size, 63, dim]
        x = x + interaction
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        
        return x


@register_network("crystal_transformer")
class CrystalTransformer(BaseCSPNetwork):
    """
    优化的晶体结构生成Transformer网络
    ~100M参数，专门针对晶体结构生成任务设计
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # 配置参数 - 调整为约100M参数量
        self.hidden_dim = config.get('hidden_dim', 640)  # 增大隐藏维度以达到100M参数
        self.num_layers = config.get('num_layers', 16)  # 更深的网络以增强表达能力
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        self.max_atoms = config.get('max_atoms', 60)
        self.pxrd_dim = config.get('pxrd_dim', 11501)
        
        self.seq_len = 3 + self.max_atoms  # 63
        
        # 输入投影
        # 使用更高效的投影方式
        self.coord_proj = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        # 组分编码器：comp是60维向量，每维表示原子序数
        # 使用线性投影处理comp向量
        self.comp_proj = nn.Sequential(
            nn.Linear(self.max_atoms, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 高效PXRD编码器
        self.pxrd_encoder = EfficientPXRDEncoder(self.pxrd_dim, self.hidden_dim, self.dropout)
        
        # 时间编码
        self.time_emb_dim = self.hidden_dim
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(self.time_emb_dim // 2),
            nn.Linear(self.time_emb_dim // 2, self.time_emb_dim),
            nn.GELU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )
        
        # 位置和类型编码
        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, self.hidden_dim) * 0.02)
        self.type_emb = nn.Embedding(2, self.hidden_dim)  # 0: lattice, 1: atom
        
        # 条件融合层
        # 使用交叉注意力机制进行条件融合
        self.cond_cross_attn = CrossAttentionBlock(self.hidden_dim, self.num_heads, self.dropout)
        
        # 主Transformer层
        self.transformer_blocks = nn.ModuleList([
            CrystalTransformerBlock(self.hidden_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # 输出层
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 3)  # 输出3D坐标
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
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
        
        # 1. 输入编码
        x = self.coord_proj(z)  # [batch_size, 63, hidden_dim]
        
        # 2. 添加位置编码
        x = x + self.pos_emb
        
        # 3. 添加类型编码
        type_ids = torch.cat([
            torch.zeros(batch_size, 3, dtype=torch.long, device=device),
            torch.ones(batch_size, self.max_atoms, dtype=torch.long, device=device)
        ], dim=1)
        x = x + self.type_emb(type_ids)
        
        # 4. 处理组分信息
        # comp是[batch_size, 60]的向量，每维表示原子序数(atom.Z)，0表示padding
        comp_feat = self.comp_proj(comp)  # [batch_size, hidden_dim]
        
        # 5. PXRD编码
        pxrd_feat = self.pxrd_encoder(pxrd)  # [batch_size, hidden_dim]
        
        # 6. 时间编码
        t_emb = self.time_emb(t.squeeze(-1))  # [batch_size, hidden_dim]
        r_emb = self.time_emb(r.squeeze(-1))  # [batch_size, hidden_dim]
        
        # 7. 组合所有条件
        # 使用加法组合，更简单高效
        global_cond = comp_feat + pxrd_feat + t_emb + r_emb  # [batch_size, hidden_dim]
        
        # 8. 条件融合（使用交叉注意力）
        # 将全局条件作为context
        global_cond_seq = global_cond.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        x = self.cond_cross_attn(x, global_cond_seq, mask=None)
        
        # 9. 创建padding mask
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=device)
        for i, n in enumerate(num_atoms):
            if n < self.max_atoms:
                mask[i, 3 + n:] = True
        
        # 10. 应用Transformer块
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # 11. 最终投影
        x = self.final_norm(x)
        output = self.output_proj(x)  # [batch_size, 63, 3]
        
        # 12. Mask掉padding位置
        for i, n in enumerate(num_atoms):
            if n < self.max_atoms:
                output[i, 3 + n:] = 0
        
        return output
    
    def prepare_batch(self, batch: dict) -> dict:
        """准备批次数据"""
        return batch