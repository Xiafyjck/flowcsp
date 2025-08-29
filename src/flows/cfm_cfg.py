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
    - 分数坐标损失计算考虑周期性边界条件
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
        
        # 损失权重（调整为更平衡的默认值）
        self.loss_weight_lattice = config.get('loss_weight_lattice', 1.0)  # 降低晶格权重
        self.loss_weight_coords = config.get('loss_weight_coords', 1.0)
        
        # CFG 特定参数
        self.cfg_prob = config.get('cfg_prob', 0.1)  # 训练时条件dropout概率
        self.cfg_scale = config.get('cfg_scale', 1.5)  # 默认引导强度
        
        # 不变量损失权重
        self.invariant_loss_weight = config.get('invariant_loss_weight', 0.01)
        
        # 采样配置
        self.default_num_steps = config.get('default_num_steps', 50)
        
        # 归一化器
        from src.normalizer import DataNormalizer
        stats_file = config.get('stats_file', None)
        if stats_file is None:
            raise ValueError("必须提供stats_file参数用于归一化")
        
        self.normalizer = DataNormalizer(
            stats_file=stats_file,
            normalize_lattice=config.get('normalize_lattice', True),
            normalize_frac_coords=config.get('normalize_frac_coords', True),  # 现在默认归一化分数坐标
            use_global_stats=config.get('use_global_stats', True)
        )
    
    # CDVAE 风格的晶格镜像偏移列表（3×3×3 = 27个镜像）
    OFFSET_LIST = torch.tensor([
        [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
        [-1, 0, -1],  [-1, 0, 0],  [-1, 0, 1],
        [-1, 1, -1],  [-1, 1, 0],  [-1, 1, 1],
        [0, -1, -1],  [0, -1, 0],  [0, -1, 1],
        [0, 0, -1],   [0, 0, 0],   [0, 0, 1],
        [0, 1, -1],   [0, 1, 0],   [0, 1, 1],
        [1, -1, -1],  [1, -1, 0],  [1, -1, 1],
        [1, 0, -1],   [1, 0, 0],   [1, 0, 1],
        [1, 1, -1],   [1, 1, 0],   [1, 1, 1],
    ], dtype=torch.float32)
    
    def sample_wrapped_normal(self, mean: torch.Tensor, std: float, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        从Wrapped Normal分布采样（适用于周期性边界条件）
        
        Wrapped Normal是将正态分布"包裹"到[0,1]区间的分布，
        通过将正态分布的值模1来实现周期性。
        
        Args:
            mean: 均值张量（在[0,1]范围内）
            std: 标准差（控制分布的集中程度）
            shape: 输出形状
            device: 设备
            
        Returns:
            采样结果（在[0,1]范围内）
        """
        # 从正态分布采样
        samples = torch.randn(shape, device=device) * std + mean
        
        # 包裹到[0,1]范围（应用周期性边界）
        wrapped_samples = samples - torch.floor(samples)
        
        return wrapped_samples
    
    def sample_wrapped_normal_normalized(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        在归一化空间中从Wrapped Normal分布采样
        
        考虑归一化变换：[0,1] → [(mean-std/2), (mean+std/2)]
        需要相应地调整Wrapped Normal的参数
        
        Args:
            shape: 输出形状
            device: 设备
            
        Returns:
            归一化空间中的采样结果
        """
        if self.normalizer.normalize_frac_coords:
            # 获取归一化参数
            frac_mean = self.normalizer.frac_mean.to(device)  # [3]
            frac_std = self.normalizer.frac_std.to(device)    # [3]
            
            # 在原始[0,1]空间采样Wrapped Normal
            # 使用较小的std以保持在单个周期内
            wrapped_std = 0.15  # 经验值，可调整
            original_samples = self.sample_wrapped_normal(
                mean=torch.tensor(0.5, device=device),  # 中心在0.5
                std=wrapped_std,
                shape=shape,
                device=device
            )
            
            # 变换到归一化空间：(x - mean) / std
            # 其中mean和std是从数据统计得出的
            normalized_samples = (original_samples - frac_mean) / (frac_std + 1e-6)
            
            return normalized_samples
        else:
            # 如果不归一化，直接在[0,1]空间采样
            return self.sample_wrapped_normal(
                mean=torch.tensor(0.5, device=device),
                std=0.15,
                shape=shape,
                device=device
            )
    
    @staticmethod
    def periodic_distance_cdvae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        使用CDVAE风格的晶格镜像方法计算周期性距离
        考虑27个可能的镜像位置，找到最短距离
        
        Args:
            x, y: 分数坐标张量 [batch_size, num_atoms, 3]
            
        Returns:
            最短周期性距离的平方 [batch_size, num_atoms, 3]
        """
        device = x.device
        batch_size, num_atoms, _ = x.shape
        
        # 将偏移列表移到正确的设备
        offsets = CFMFlowCFG.OFFSET_LIST.to(device)  # [27, 3]
        num_images = offsets.shape[0]
        
        # 扩展维度以便批处理
        # x: [batch_size, num_atoms, 1, 3]
        # y: [batch_size, num_atoms, 1, 3]
        # offsets: [1, 1, 27, 3]
        x_expanded = x.unsqueeze(2)  # [batch_size, num_atoms, 1, 3]
        y_expanded = y.unsqueeze(2)  # [batch_size, num_atoms, 1, 3]
        offsets_expanded = offsets.view(1, 1, num_images, 3)
        
        # 计算所有镜像位置
        y_images = y_expanded + offsets_expanded  # [batch_size, num_atoms, 27, 3]
        
        # 计算到所有镜像的距离
        diffs = x_expanded - y_images  # [batch_size, num_atoms, 27, 3]
        distances_sq = diffs ** 2  # [batch_size, num_atoms, 27, 3]
        
        # 对每个坐标维度，找到最小距离
        min_distances_sq, _ = distances_sq.min(dim=2)  # [batch_size, num_atoms, 3]
        
        return min_distances_sq
    
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
        
        # 采样噪声（改进的线性schedule，避免初期噪声过大）
        # 使用线性插值代替cosine schedule，更稳定
        sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)  # [batch_size, 1]
        
        # 生成噪声：晶格参数使用正态分布，分数坐标使用Wrapped Normal
        x0 = torch.zeros_like(x1)  # [batch_size, 63, 3]
        
        # 晶格参数：使用标准正态分布
        x0[:, :3, :] = torch.randn(batch_size, 3, 3, device=device) * sigma_t.unsqueeze(-1)
        
        # 分数坐标：使用Wrapped Normal先验分布（归一化空间）
        # 这确保了噪声也遵循周期性边界条件
        frac_coords_noise = self.sample_wrapped_normal_normalized(
            shape=(batch_size, 60, 3),
            device=device
        )
        
        # 应用噪声强度调制
        x0[:, 3:, :] = frac_coords_noise * sigma_t.unsqueeze(-1)
        
        # 数值稳定性处理
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
        
        # =============== 等变性感知的晶格损失 ===============
        lattice_v_pred = v_pred[:, :3]  # [batch_size, 3, 3]
        lattice_v_target = v_target[:, :3]  # [batch_size, 3, 3]
        
        # 晶格损失：直接计算MSE，因为晶格参数不需要周期性边界条件
        lattice_velocity_loss = F.smooth_l1_loss(lattice_v_pred, lattice_v_target, beta=2.5)
        
        # =============== 坐标损失（考虑周期性边界条件）===============
        # 分离出坐标部分的速度场
        coords_v_pred = v_pred[:, 3:]  # [batch_size, 60, 3]
        coords_v_target = v_target[:, 3:]  # [batch_size, 60, 3]
        
        # 周期性边界处理：
        # 分数坐标归一化后，[0,1] → [mean-std/2, mean+std/2]
        # 但周期性的本质不变，只是尺度改变了
        # 速度场 v = x1 - x0，考虑最短路径
        
        # 计算速度场的差异
        coords_v_diff = coords_v_pred - coords_v_target
        
        # 获取归一化后的周期（对应原始空间的1.0）
        # 如果std是标准差，那么归一化后的周期长度就是 1.0/std
        device = coords_v_diff.device
        if self.normalizer.normalize_frac_coords:
            # 归一化后的周期长度 = 原始周期(1.0) / std
            frac_std = self.normalizer.frac_std.to(device).view(1, 1, 3)  # [1, 1, 3]
            period = 1.0 / (frac_std + 1e-6)  # 归一化空间中的周期
            
            # 将差值映射到最短路径（[-period/2, period/2]）
            coords_v_diff = coords_v_diff - torch.round(coords_v_diff / period) * period
        else:
            # 如果没有归一化，使用原始的周期性处理
            print("Warning: 没有归一化，使用原始的周期性处理")
            coords_v_diff = coords_v_diff - torch.round(coords_v_diff)
        
        # 使用修正后的目标计算损失
        coords_v_target_corrected = coords_v_pred - coords_v_diff
        
        # 使用reduction='none'以便手动应用mask
        coords_velocity_loss_unreduced = F.smooth_l1_loss(
            coords_v_pred, 
            coords_v_target_corrected,
            beta=2.5,
            reduction='none'
        )  # [batch_size, 60, 3]
        
        # 应用mask排除padding
        coords_mask = mask[:, 3:]  # [batch_size, 60, 1]
        masked_coords_loss = coords_velocity_loss_unreduced * coords_mask  # [batch_size, 60, 3]
        
        # 计算每个样本的平均损失
        valid_atoms = coords_mask.sum(dim=[1, 2])  # [batch_size] - 总的有效维度数
        coords_velocity_loss = masked_coords_loss.sum(dim=[1, 2]) / (valid_atoms + 1e-6)  # [batch_size]
        coords_velocity_loss = coords_velocity_loss.mean()  # 标量
        
        # 总损失
        v_loss = self.loss_weight_lattice * lattice_velocity_loss + self.loss_weight_coords * coords_velocity_loss
        
        # 记录原始损失值用于调试
        original_loss = v_loss.item() if not torch.isnan(v_loss) else float('inf')
        
        # 只在极端情况下进行截断，避免梯度爆炸
        if torch.isnan(v_loss) or torch.isinf(v_loss):
            print(f"Warning: NaN/Inf loss detected! lattice_loss={lattice_velocity_loss.item():.4f}, coords_loss={coords_velocity_loss.item():.4f}")
            v_loss = torch.tensor(10.0, device=device, requires_grad=True)
        elif v_loss > 100.0:  # 提高阈值，让我们看到真实的损失
            print(f"Warning: Very high loss={original_loss:.4f}")
            v_loss = torch.clamp(v_loss, max=100.0)
        
        # 计算指标
        with torch.no_grad():
            # 计算相对误差
            v_error = v_pred - v_target
            
            # 对坐标部分应用周期性修正（与损失计算保持一致）
            if self.normalizer.normalize_frac_coords:
                frac_std = self.normalizer.frac_std.to(device).view(1, 1, 3)
                period = 1.0 / (frac_std + 1e-6)
                v_error[:, 3:] = v_error[:, 3:] - torch.round(v_error[:, 3:] / period) * period
            else:
                v_error[:, 3:] = v_error[:, 3:] - torch.round(v_error[:, 3:])
            
            # 计算误差范数
            v_error_norm = torch.norm(v_error * mask, dim=-1)
            v_target_norm_masked = torch.norm(v_target * mask, dim=-1)
            
            valid_mask = v_target_norm_masked > 1e-4
            
            if valid_mask.any():
                relative_error = (v_error_norm[valid_mask] / (v_target_norm_masked[valid_mask] + 1e-4)).mean()
            else:
                relative_error = torch.tensor(0.0, device=device)
        
        metrics = {
            'loss': v_loss.item(),
            'lattice_velocity_loss': lattice_velocity_loss.item(),
            'coords_velocity_loss': coords_velocity_loss.item(),
            't_mean': t.mean().item(),
            'sigma_mean': sigma_t.mean().item() if sigma_t.numel() > 0 else 0.0,  # 监控噪声水平
            'original_loss': original_loss,  # 记录未截断的损失
            'relative_error': relative_error.item() if 'relative_error' in locals() else 0.0,
        }
        
        return v_loss, metrics
    
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
        # 分别处理晶格参数和分数坐标
        x = torch.zeros(batch_size, 63, 3, device=device)
        
        # 晶格参数：使用标准正态分布
        x[:, :3, :] = torch.randn(batch_size, 3, 3, device=device) * temperature * self.sigma_max
        
        # 分数坐标：使用Wrapped Normal先验分布
        # 这确保了初始采样就遵循周期性边界条件
        frac_coords_init = self.sample_wrapped_normal_normalized(
            shape=(batch_size, 60, 3),
            device=device
        )
        x[:, 3:, :] = frac_coords_init * temperature * self.sigma_max
        
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
        # 将x包装成字典格式以使用denormalize方法
        batch = {'z': x}
        batch = self.normalizer.denormalize(batch, inplace=False)
        x = batch['z']
        
        # 后处理：确保分数坐标在[0, 1]范围（应用周期性边界条件）
        # 使用模运算将任何超出[0,1]的坐标映射回正确范围
        x[:, 3:] = x[:, 3:] % 1.0
        
        if return_trajectory:
            trajectory = [self.normalizer.denormalize_z(z) for z in trajectory]
            return torch.stack(trajectory, dim=0)
        else:
            return x