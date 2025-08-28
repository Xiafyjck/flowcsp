"""
标准CFM (Conditional Flow Matching) 流模型实现
使用pytorch-cfm库，实现晶体结构生成的条件流匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from . import BaseFlow, register_flow


@register_flow("cfm")
class CFMFlow(BaseFlow):
    """
    标准CFM流模型
    用于晶体结构生成的条件流匹配
    
    原理：
    - CFM通过学习从噪声分布到数据分布的向量场来生成样本
    - 相比扩散模型，CFM使用更直接的传输路径，训练更稳定
    - 支持条件生成，可以根据PXRD谱生成对应的晶体结构
    """
    
    def __init__(self, network: nn.Module, config: dict):
        """
        初始化CFM流模型
        
        Args:
            network: 神经网络模型（如Transformer或Equiformer）
            config: 配置字典，包含：
                - sigma_min: 最小噪声水平
                - sigma_max: 最大噪声水平
                - loss_weight_lattice: 晶格损失权重
                - loss_weight_coords: 坐标损失权重
                - normalize_lattice: 是否归一化晶格参数
                - lattice_scale: 晶格归一化尺度
        """
        super().__init__(network, config)
        
        # CFM 超参数
        self.sigma_min = config.get('sigma_min', 1e-4)
        self.sigma_max = config.get('sigma_max', 1.0)
        
        # 损失权重（晶格参数通常更重要，给予更高权重）
        self.loss_weight_lattice = config.get('loss_weight_lattice', 2.0)
        self.loss_weight_coords = config.get('loss_weight_coords', 1.0)
        
        # 采样配置
        self.default_num_steps = config.get('default_num_steps', 50)
        
        # 归一化器（用于推理时的反归一化）
        # 注意：训练时数据已在DataLoader中归一化，这里只用于采样时的反归一化
        from src.normalizer import DataNormalizer
        self.normalizer = DataNormalizer(
            normalize_lattice=config.get('normalize_lattice', True),
            normalize_frac_coords=config.get('normalize_frac_coords', False),
            use_global_stats=config.get('use_global_stats', True)
        )
        
    def interpolate(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        线性插值函数，用于构建从x0到x1的传输路径
        
        Args:
            x0: 起始点（噪声）[batch_size, 63, 3]
            x1: 目标点（数据）[batch_size, 63, 3]
            t: 时间步 [batch_size, 1, 1]
            
        Returns:
            插值结果 [batch_size, 63, 3]
        """
        # t需要广播到正确的形状
        if t.dim() == 2:
            t = t.unsqueeze(-1)  # [batch_size, 1, 1]
        return (1 - t) * x0 + t * x1
    
    def _sample_cfm_training_data(self, x1: torch.Tensor, conditions: Dict) -> Dict:
        """
        CFM训练数据采样（独立函数，便于理解和复用）
        
        采样过程：
        1. 采样时间步 t ~ U(0, 1)
        2. 生成噪声 x0 ~ N(0, sigma*I)
        3. 计算插值 xt = (1-t)*x0 + t*x1
        4. 计算目标速度场 v_target = x1 - x0
        5. （可选）生成实时PXRD
        
        Args:
            x1: 目标数据（归一化空间）[batch_size, 63, 3]
            conditions: 条件字典
            
        Returns:
            包含采样数据的字典：
                - t: 时间步 [batch_size, 1]
                - r: 第二个时间步 [batch_size, 1]
                - x0: 噪声 [batch_size, 63, 3]
                - xt: 插值 [batch_size, 63, 3]
                - v_target: 目标速度场 [batch_size, 63, 3]
                - pxrd_realtime: 实时PXRD（如果有）
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # 采样时间步 t ~ U(0, 1)
        t = torch.rand(batch_size, 1, device=device)
        
        # 采样另一个时间步 r（用于某些高级CFM变体）
        r = torch.rand(batch_size, 1, device=device)
        
        # 采样噪声 x0 ~ N(0, sigma*I)
        # 使用cosine schedule，更平滑的噪声调度
        alpha = torch.cos(t * torch.pi / 2)  # cos schedule: 1->0 as t: 0->1
        sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * alpha
        
        # 确保sigma_t形状正确 [batch_size, 1]
        if sigma_t.dim() == 1:
            sigma_t = sigma_t.unsqueeze(-1)
        
        # 添加噪声
        x0 = torch.randn_like(x1) * sigma_t.unsqueeze(-1)
        
        # 额外的数值稳定性检查
        x0 = torch.nan_to_num(x0, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 计算插值 xt
        xt = self.interpolate(x0, x1, t.unsqueeze(-1))
        
        # 计算目标速度场
        v_target = x1 - x0
        
        # 不再计算realtime PXRD
        pxrd_realtime = None
        
        return {
            't': t,
            'r': r,
            'x0': x0,
            'xt': xt,
            'v_target': v_target,
            'pxrd_realtime': pxrd_realtime
        }
    
    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        计算CFM训练损失
        
        CFM损失计算步骤：
        1. 采样时间步t ~ U(0, 1)
        2. 采样噪声x0 ~ N(0, I)
        3. 计算插值xt = (1-t)*x0 + t*x1
        4. 预测速度场v = network(xt, t, conditions)
        5. 计算目标速度v_target = x1 - x0
        6. 损失 = MSE(v, v_target)
        
        Args:
            batch: 包含训练数据的字典
            
        Returns:
            loss: 标量损失张量
            metrics: 用于日志记录的指标字典
        """
        # 准备数据
        batch = self.network.prepare_batch(batch)
        
        # 归一化数据（重要：在噪声采样前归一化）
        batch = self.normalizer.normalize(batch, inplace=False)
        
        # 提取归一化后的数据
        x1 = batch['z']  # 归一化的目标晶体结构 [batch_size, 63, 3]
        batch_size = x1.shape[0]
        device = x1.device
        
        # 准备条件
        conditions = {
            'comp': batch['comp'],  # [batch_size, 60]
            'pxrd': batch['pxrd'],  # [batch_size, 11501]
            'num_atoms': batch['num_atoms']  # [batch_size]
        }
        
        # ========== CFM训练数据采样 ==========
        # 原始采样代码已移至 _sample_cfm_training_data 函数
        # 这里可以选择：
        # 1. 使用原始的在线采样（默认）
        # 2. 使用预计算的采样数据（如果batch中包含）
        
        # 检查是否有预计算数据
        if 't' in batch and 'x0' in batch and 'xt' in batch:
            # 使用预计算数据（来自DataLoader）
            t = batch['t']  # [batch_size] or [batch_size, 1]
            r = batch['r']
            x0 = batch['x0']  # 已经在归一化空间
            xt = batch['xt']  # 已经在归一化空间
            v_target = x1 - x0
            
            # 确保t和r的形状正确
            if t.dim() == 1:
                t = t.unsqueeze(-1)  # [batch_size, 1]
            if r.dim() == 1:
                r = r.unsqueeze(-1)
            
            # 不再处理realtime PXRD
            conditions['pxrd_realtime'] = None
            
            # 记录使用了预计算
            self.using_precomputed = True
            
        else:
            # 使用原始的在线采样（向后兼容）
            sampled_data = self._sample_cfm_training_data(x1, conditions)
            t = sampled_data['t']
            r = sampled_data['r']
            x0 = sampled_data['x0']
            xt = sampled_data['xt']
            v_target = sampled_data['v_target']
            # 不再使用realtime PXRD
            
            # 记录使用了在线采样
            self.using_precomputed = False
        
        # 通过网络预测速度场
        v_pred = self.network(xt, t, r, conditions)
        
        # v_target已经在采样阶段计算（无论是预计算还是在线）
        
        # 对padding位置进行mask
        mask = torch.zeros_like(x1[..., 0])  # [batch_size, 63]
        for i, n in enumerate(batch['num_atoms']):
            mask[i, :3] = 1.0  # 晶格参数始终有效
            mask[i, 3:3+n] = 1.0  # 有效原子位置
        mask = mask.unsqueeze(-1)  # [batch_size, 63, 1]
        
        # 计算加权MSE损失 - 只对有效位置计算，避免padding影响
        # 分别计算晶格和坐标的损失
        
        # 晶格损失 - 晶格始终有效，直接计算MSE
        # 添加数值稳定性检查，防止NaN
        lattice_pred = v_pred[:, :3]
        lattice_target = v_target[:, :3]
        
        # 检查是否有NaN或inf
        if torch.isnan(lattice_pred).any() or torch.isinf(lattice_pred).any():
            # 如果有NaN，返回一个稳定的损失值避免训练崩溃
            lattice_loss = torch.tensor(1.0, device=lattice_pred.device, requires_grad=True)
        elif torch.isnan(lattice_target).any() or torch.isinf(lattice_target).any():
            # 目标也可能有问题
            lattice_loss = torch.tensor(1.0, device=lattice_pred.device, requires_grad=True)
        else:
            # 使用更稳健的损失计算
            lattice_loss = F.mse_loss(lattice_pred, lattice_target)
            # 裁剪极端值
            lattice_loss = torch.clamp(lattice_loss, max=10.0)
        
        # 坐标损失 - 需要处理不同原子数的情况
        # 方法：计算每个有效位置的平方误差，然后求平均
        coords_diff = (v_pred[:, 3:] - v_target[:, 3:]) * mask[:, 3:]  # 只保留有效位置的差值
        coords_sq_error = (coords_diff ** 2).sum(dim=-1)  # [batch_size, 60] 每个位置的平方误差
        
        # 计算每个样本的有效原子数，用于正确的平均
        valid_atoms = mask[:, 3:, 0].sum(dim=1)  # [batch_size] 每个样本的有效原子数
        # 使用更稳健的平均方式
        coords_loss_per_sample = coords_sq_error.sum(dim=1) / torch.clamp(valid_atoms * 3, min=1.0)
        coords_loss = coords_loss_per_sample.mean()  # batch平均
        
        # 总损失
        loss = (self.loss_weight_lattice * lattice_loss + 
                self.loss_weight_coords * coords_loss)
        
        # 最终的损失裁剪，防止梯度爆炸
        loss = torch.clamp(loss, max=20.0)
        
        # 如果损失是NaN，返回一个固定值
        if torch.isnan(loss):
            loss = torch.tensor(5.0, device=loss.device, requires_grad=True)
        
        # 计算额外的指标用于监控
        with torch.no_grad():
            # 速度场的平均幅度
            v_pred_norm = torch.norm(v_pred * mask, dim=-1).mean()
            v_target_norm = torch.norm(v_target * mask, dim=-1).mean()
            
            # 预测误差（相对误差）
            # 使用更大的epsilon避免除零
            v_target_norm_masked = torch.norm(v_target * mask, dim=-1)
            # 只对非零目标计算相对误差
            valid_mask = v_target_norm_masked > 1e-4
            if valid_mask.any():
                relative_error = torch.norm((v_pred - v_target) * mask, dim=-1)[valid_mask] / (
                    v_target_norm_masked[valid_mask] + 1e-4
                )
                relative_error = relative_error.mean()
            else:
                relative_error = torch.tensor(0.0, device=x1.device)
            
            # 获取门控权重（如果网络支持）
            gate_weight = getattr(self.network, 'last_gate_weight', 0.0)
            quality_score = getattr(self.network, 'last_quality_score', 0.0)
        
        metrics = {
            'loss': loss.item(),
            'lattice_loss': lattice_loss.item(),
            'coords_loss': coords_loss.item(),
            'v_pred_norm': v_pred_norm.item(),
            'v_target_norm': v_target_norm.item(),
            'relative_error': relative_error.item(),
            't_mean': t.mean().item(),  # 监控时间步分布
            'pxrd_gate_weight': gate_weight,  # 监控门控权重
            'pxrd_quality_score': quality_score,  # 监控质量分数
            'using_precomputed': int(getattr(self, 'using_precomputed', False)),  # 是否使用预计算
        }
        
        return loss, metrics
    
    def sample(
        self, 
        conditions: Dict[str, torch.Tensor],
        num_steps: int = None,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        return_trajectory: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        从CFM生成样本（对用户透明的接口）
        
        采样过程：
        1. 从噪声分布x0 ~ N(0, I)开始
        2. 通过ODE积分：dx/dt = v(x, t)
        3. 使用Euler方法或更高阶的积分器
        
        Args:
            conditions: 条件信息字典，包含：
                - comp: 原子组成 [batch_size, 60]，原子序数
                - pxrd: PXRD谱 [batch_size, 11501]，已归一化的强度
                - num_atoms: 原子数量 [batch_size]
            num_steps: 采样步数（默认50）
            temperature: 温度参数，控制生成多样性
            guidance_scale: 引导强度（用于加强条件控制）
            return_trajectory: 是否返回整个采样轨迹
            
        Returns:
            生成的晶体结构 [batch_size, 63, 3]（物理空间）
            - 前3行：晶格参数矩阵（Angstrom单位）
            - 后60行：分数坐标 [0, 1]
            如果return_trajectory=True，返回 [num_steps+1, batch_size, 63, 3]
        
        注意：
            输入和输出都在物理空间，归一化在内部自动处理
        """
        if num_steps is None:
            num_steps = self.default_num_steps
            
        device = conditions['pxrd'].device
        batch_size = conditions['pxrd'].shape[0]
        
        # 条件不需要归一化，因为comp和pxrd本身就是标准化的特征
        # comp: 原子序数，范围[0, 118]
        # pxrd: 谱强度，已经是归一化的
        # num_atoms: 整数，不需要归一化
        
        # 初始化噪声（在归一化空间）
        x = torch.randn(batch_size, 63, 3, device=device) * temperature * self.sigma_max
        
        # 时间步
        dt = 1.0 / num_steps
        
        # 保存轨迹（如果需要）
        trajectory = [x.clone()] if return_trajectory else None
        
        # ODE积分（从t=0到t=1）
        for step in range(num_steps):
            t = torch.full((batch_size, 1), step * dt, device=device)
            r = t.clone()  # 对于标准CFM，r可以等于t
            
            # 不再计算realtime PXRD
            
            # 预测速度场
            with torch.no_grad():
                v = self.network(x, t, r, conditions)
                
                # 应用引导（可选）
                if guidance_scale != 1.0:
                    # 计算无条件速度（使用零PXRD）
                    conditions_uncond = conditions.copy()
                    conditions_uncond['pxrd'] = torch.zeros_like(conditions['pxrd'])
                    # 不再处理realtime PXRD
                    v_uncond = self.network(x, t, r, conditions_uncond)
                    
                    # 应用classifier-free guidance
                    v = v_uncond + guidance_scale * (v - v_uncond)
            
            # Euler步进
            x = x + v * dt
            
            # 保存轨迹
            if return_trajectory:
                trajectory.append(x.clone())
        
        # 反归一化生成的结构（重要：将归一化空间的结果转换回物理空间）
        x = self.normalizer.denormalize_z(x)
        
        # 后处理：确保分数坐标在[0, 1]范围内
        x[:, 3:] = x[:, 3:] % 1.0
        
        if return_trajectory:
            # 如果返回轨迹，也需要反归一化整个轨迹
            trajectory = [self.normalizer.denormalize_z(z) for z in trajectory]
            return torch.stack(trajectory, dim=0)
        else:
            return x
    
    # 移除了sample_with_pxrd_feedback方法，不再支持实时PXRD反馈
    # 如需此功能，请在采样后进行后处理优化