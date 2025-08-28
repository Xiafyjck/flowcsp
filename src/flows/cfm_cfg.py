"""
Classifier-Free Guidance 版本的 Conditional Flow Matching
支持训练时随机dropout条件，推理时通过条件和无条件预测的加权组合增强生成质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from . import BaseFlow, register_flow


@register_flow("cfm_cfg")
class CFMFlowCFG(BaseFlow):
    """
    支持Classifier-Free Guidance的CFM流模型
    
    核心思想：
    - 训练时以一定概率将条件置零，让模型同时学习条件和无条件分布
    - 推理时通过加权组合增强条件控制：v = v_uncond + w*(v_cond - v_uncond)
    """
    
    def __init__(self, network: nn.Module, config: dict):
        """
        初始化CFG版本的CFM流模型
        
        Args:
            network: 神经网络模型
            config: 配置字典，新增参数：
                - cfg_prob: 训练时条件dropout概率（默认0.1）
                - cfg_scale: 推理时的默认引导强度（默认1.5）
                其他参数与标准CFM相同
        """
        super().__init__(network, config)
        
        # CFM 超参数
        self.sigma_min = config.get('sigma_min', 1e-4)
        self.sigma_max = config.get('sigma_max', 1.0)
        
        # 损失权重
        self.loss_weight_lattice = config.get('loss_weight_lattice', 2.0)
        self.loss_weight_coords = config.get('loss_weight_coords', 1.0)
        
        # CFG 特定参数
        self.cfg_prob = config.get('cfg_prob', 0.1)  # 训练时条件dropout概率
        self.cfg_scale = config.get('cfg_scale', 1.5)  # 默认引导强度
        
        # 采样配置
        self.default_num_steps = config.get('default_num_steps', 50)
        
        # 归一化器
        from src.normalizer import DataNormalizer
        self.normalizer = DataNormalizer(
            normalize_lattice=config.get('normalize_lattice', True),
            normalize_frac_coords=config.get('normalize_frac_coords', False),
            use_global_stats=config.get('use_global_stats', True)
        )
    
    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        计算CFM训练损失（遵循BaseFlow接口）
        
        与标准CFM的区别：
        - 训练时以cfg_prob概率将条件（PXRD和comp）置零
        - 让模型同时学习有条件和无条件的速度场
        
        Args:
            batch: 包含训练数据的字典
            
        Returns:
            loss: 标量损失张量
            metrics: 用于日志记录的指标字典
        """
        # 准备数据
        batch = self.network.prepare_batch(batch)
        
        # 归一化数据
        batch = self.normalizer.normalize(batch, inplace=False)
        
        # 提取归一化后的数据
        x1 = batch['z']  # [batch_size, 63, 3]
        batch_size = x1.shape[0]
        device = x1.device
        
        # 准备条件
        conditions = {
            'comp': batch['comp'].clone(),  # [batch_size, 60]
            'pxrd': batch['pxrd'].clone(),  # [batch_size, 11501]
            'num_atoms': batch['num_atoms']  # [batch_size]
        }
        
        # ========== CFG核心：随机dropout条件 ==========
        if self.training and self.cfg_prob > 0:
            # 为每个样本独立决定是否dropout
            dropout_mask = torch.rand(batch_size, device=device) < self.cfg_prob
            
            # 对需要dropout的样本，将条件置零
            for i in range(batch_size):
                if dropout_mask[i]:
                    conditions['comp'][i] = torch.zeros_like(conditions['comp'][i])
                    conditions['pxrd'][i] = torch.zeros_like(conditions['pxrd'][i])
        
        # ========== 标准CFM训练流程 ==========
        # 采样时间步
        t = torch.rand(batch_size, 1, device=device)
        r = torch.rand(batch_size, 1, device=device)
        
        # 采样噪声（cosine schedule）
        alpha = torch.cos(t * torch.pi / 2)
        sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * alpha
        
        if sigma_t.dim() == 1:
            sigma_t = sigma_t.unsqueeze(-1)
        
        # 生成噪声
        x0 = torch.randn_like(x1) * sigma_t.unsqueeze(-1)
        x0 = torch.nan_to_num(x0, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 计算插值
        t_expanded = t.unsqueeze(-1)  # [batch_size, 1, 1]
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        # 计算目标速度场
        v_target = x1 - x0
        
        # 通过网络预测速度场
        v_pred = self.network(xt, t, r, conditions)
        
        # 创建mask处理padding
        mask = torch.zeros_like(x1[..., 0])  # [batch_size, 63]
        for i, n in enumerate(batch['num_atoms']):
            mask[i, :3] = 1.0  # 晶格参数
            mask[i, 3:3+n] = 1.0  # 有效原子
        mask = mask.unsqueeze(-1)  # [batch_size, 63, 1]
        
        # 计算晶格损失
        lattice_pred = v_pred[:, :3]
        lattice_target = v_target[:, :3]
        
        if torch.isnan(lattice_pred).any() or torch.isinf(lattice_pred).any():
            lattice_loss = torch.tensor(1.0, device=device, requires_grad=True)
        else:
            lattice_loss = F.mse_loss(lattice_pred, lattice_target)
            lattice_loss = torch.clamp(lattice_loss, max=10.0)
        
        # 计算坐标损失
        coords_diff = (v_pred[:, 3:] - v_target[:, 3:]) * mask[:, 3:]
        coords_sq_error = (coords_diff ** 2).sum(dim=-1)
        valid_atoms = mask[:, 3:, 0].sum(dim=1)
        coords_loss = (coords_sq_error.sum(dim=1) / torch.clamp(valid_atoms * 3, min=1.0)).mean()
        
        # 总损失
        loss = self.loss_weight_lattice * lattice_loss + self.loss_weight_coords * coords_loss
        loss = torch.clamp(loss, max=20.0)
        
        if torch.isnan(loss):
            loss = torch.tensor(5.0, device=device, requires_grad=True)
        
        # 计算指标
        with torch.no_grad():
            v_pred_norm = torch.norm(v_pred * mask, dim=-1).mean()
            v_target_norm = torch.norm(v_target * mask, dim=-1).mean()
            
            v_target_norm_masked = torch.norm(v_target * mask, dim=-1)
            valid_mask = v_target_norm_masked > 1e-4
            if valid_mask.any():
                relative_error = torch.norm((v_pred - v_target) * mask, dim=-1)[valid_mask] / (
                    v_target_norm_masked[valid_mask] + 1e-4
                )
                relative_error = relative_error.mean()
            else:
                relative_error = torch.tensor(0.0, device=device)
        
        metrics = {
            'loss': loss.item(),
            'lattice_loss': lattice_loss.item(),
            'coords_loss': coords_loss.item(),
            'v_pred_norm': v_pred_norm.item(),
            'v_target_norm': v_target_norm.item(),
            'relative_error': relative_error.item(),
            't_mean': t.mean().item(),
        }
        
        return loss, metrics
    
    def sample(
        self, 
        conditions: Dict[str, torch.Tensor],
        num_steps: int = None,
        temperature: float = 1.0,
        guidance_scale: float = None,  # CFG特有参数
        return_trajectory: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        使用Classifier-Free Guidance生成样本（遵循BaseFlow接口）
        
        CFG采样公式：v = v_uncond + w*(v_cond - v_uncond)
        - w=1: 标准条件生成
        - w>1: 增强条件控制（更精确但可能过拟合）
        - w<1: 增加多样性（更随机但可能偏离条件）
        
        Args:
            conditions: 条件信息字典
            num_steps: 采样步数（默认50）
            temperature: 温度参数
            guidance_scale: CFG引导强度（默认使用配置值）
            return_trajectory: 是否返回整个轨迹
            
        Returns:
            生成的晶体结构 [batch_size, 63, 3]（物理空间）
        """
        if num_steps is None:
            num_steps = self.default_num_steps
        
        if guidance_scale is None:
            guidance_scale = self.cfg_scale
            
        device = conditions['pxrd'].device
        batch_size = conditions['pxrd'].shape[0]
        
        # 初始化噪声（归一化空间）
        x = torch.randn(batch_size, 63, 3, device=device) * temperature * self.sigma_max
        
        # 时间步
        dt = 1.0 / num_steps
        
        # 保存轨迹
        trajectory = [x.clone()] if return_trajectory else None
        
        # 准备无条件的条件（用于CFG）
        if guidance_scale != 1.0:
            conditions_uncond = {
                'comp': torch.zeros_like(conditions['comp']),
                'pxrd': torch.zeros_like(conditions['pxrd']),
                'num_atoms': conditions['num_atoms']  # 原子数量保持不变
            }
        
        # ODE积分
        for step in range(num_steps):
            t = torch.full((batch_size, 1), step * dt, device=device)
            r = t.clone()
            
            with torch.no_grad():
                if guidance_scale != 1.0:
                    # 计算条件速度场
                    v_cond = self.network(x, t, r, conditions)
                    
                    # 计算无条件速度场
                    v_uncond = self.network(x, t, r, conditions_uncond)
                    
                    # Classifier-Free Guidance组合
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    # 标准条件生成
                    v = self.network(x, t, r, conditions)
            
            # Euler步进
            x = x + v * dt
            
            # 保存轨迹
            if return_trajectory:
                trajectory.append(x.clone())
        
        # 反归一化到物理空间
        x = self.normalizer.denormalize_z(x)
        
        # 后处理：确保分数坐标在[0, 1]范围
        x[:, 3:] = x[:, 3:] % 1.0
        
        if return_trajectory:
            trajectory = [self.normalizer.denormalize_z(z) for z in trajectory]
            return torch.stack(trajectory, dim=0)
        else:
            return x