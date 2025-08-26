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
            x0: 起始点（噪声）[batch_size, 55, 3]
            x1: 目标点（数据）[batch_size, 55, 3]
            t: 时间步 [batch_size, 1, 1]
            
        Returns:
            插值结果 [batch_size, 55, 3]
        """
        # t需要广播到正确的形状
        if t.dim() == 2:
            t = t.unsqueeze(-1)  # [batch_size, 1, 1]
        return (1 - t) * x0 + t * x1
    
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
        x1 = batch['z']  # 归一化的目标晶体结构 [batch_size, 55, 3]
        batch_size = x1.shape[0]
        device = x1.device
        
        # 准备条件
        conditions = {
            'comp': batch['comp'],  # [batch_size, 52]
            'pxrd': batch['pxrd'],  # [batch_size, 11501]
            'num_atoms': batch['num_atoms']  # [batch_size]
        }
        
        # 采样时间步 t ~ U(0, 1)
        t = torch.rand(batch_size, 1, device=device)
        
        # 采样另一个时间步 r（用于某些高级CFM变体）
        r = torch.rand(batch_size, 1, device=device)
        
        # 采样噪声 x0 ~ N(0, sigma*I)
        # 使用adaptive sigma基于时间步
        sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)
        x0 = torch.randn_like(x1) * sigma_t.unsqueeze(-1)
        
        # 计算插值 xt
        xt = self.interpolate(x0, x1, t.unsqueeze(-1))
        
        # 训练时模拟realtime PXRD的可用性
        # 早期时间步（t < 0.25）不提供realtime PXRD（结构太嘈杂）
        # 后期时间步有一定概率提供（模拟实际计算可能失败的情况）
        provide_realtime = (t > 0.25).float() * (torch.rand_like(t) > 0.3)  # 70%概率在后期提供
        
        if provide_realtime.any():
            # 为部分样本提供模拟的realtime PXRD
            # 实际应用中应该计算xt的真实PXRD
            simulated_pxrd = conditions['pxrd'] + torch.randn_like(conditions['pxrd']) * 0.1 * (1 - t.squeeze(-1).unsqueeze(-1))
            # 只为provide_realtime=1的样本设置
            conditions['pxrd_realtime'] = simulated_pxrd * provide_realtime.unsqueeze(-1)
        else:
            conditions['pxrd_realtime'] = None
        
        # 通过网络预测速度场
        v_pred = self.network(xt, t, r, conditions)
        
        # 计算目标速度场（从x0指向x1的速度）
        v_target = x1 - x0
        
        # 对padding位置进行mask
        mask = torch.zeros_like(x1[..., 0])  # [batch_size, 55]
        for i, n in enumerate(batch['num_atoms']):
            mask[i, :3] = 1.0  # 晶格参数始终有效
            mask[i, 3:3+n] = 1.0  # 有效原子位置
        mask = mask.unsqueeze(-1)  # [batch_size, 55, 1]
        
        # 计算加权MSE损失
        # 分别计算晶格和坐标的损失
        lattice_loss = F.mse_loss(
            v_pred[:, :3] * mask[:, :3],
            v_target[:, :3] * mask[:, :3]
        )
        
        coords_loss = F.mse_loss(
            v_pred[:, 3:] * mask[:, 3:],
            v_target[:, 3:] * mask[:, 3:]
        )
        
        # 总损失
        loss = (self.loss_weight_lattice * lattice_loss + 
                self.loss_weight_coords * coords_loss)
        
        # 计算额外的指标用于监控
        with torch.no_grad():
            # 速度场的平均幅度
            v_pred_norm = torch.norm(v_pred * mask, dim=-1).mean()
            v_target_norm = torch.norm(v_target * mask, dim=-1).mean()
            
            # 预测误差（相对误差）
            relative_error = torch.norm((v_pred - v_target) * mask, dim=-1) / (
                torch.norm(v_target * mask, dim=-1) + 1e-8
            )
            relative_error = relative_error.mean()
        
        metrics = {
            'loss': loss.item(),
            'lattice_loss': lattice_loss.item(),
            'coords_loss': coords_loss.item(),
            'v_pred_norm': v_pred_norm.item(),
            'v_target_norm': v_target_norm.item(),
            'relative_error': relative_error.item(),
            't_mean': t.mean().item(),  # 监控时间步分布
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
                - comp: 原子组成 [batch_size, 52]，原子序数
                - pxrd: PXRD谱 [batch_size, 11501]，已归一化的强度
                - num_atoms: 原子数量 [batch_size]
            num_steps: 采样步数（默认50）
            temperature: 温度参数，控制生成多样性
            guidance_scale: 引导强度（用于加强条件控制）
            return_trajectory: 是否返回整个采样轨迹
            
        Returns:
            生成的晶体结构 [batch_size, 55, 3]（物理空间）
            - 前3行：晶格参数矩阵（Angstrom单位）
            - 后52行：分数坐标 [0, 1]
            如果return_trajectory=True，返回 [num_steps+1, batch_size, 55, 3]
        
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
        x = torch.randn(batch_size, 55, 3, device=device) * temperature * self.sigma_max
        
        # 时间步
        dt = 1.0 / num_steps
        
        # 保存轨迹（如果需要）
        trajectory = [x.clone()] if return_trajectory else None
        
        # ODE积分（从t=0到t=1）
        for step in range(num_steps):
            t = torch.full((batch_size, 1), step * dt, device=device)
            r = t.clone()  # 对于标准CFM，r可以等于t
            
            # 计算当前位置的实时PXRD（简化版本）
            # 实际应用中应该调用PXRDSimulator
            conditions['pxrd_realtime'] = conditions['pxrd'] * t.squeeze(-1).unsqueeze(-1)
            
            # 预测速度场
            with torch.no_grad():
                v = self.network(x, t, r, conditions)
                
                # 应用引导（可选）
                if guidance_scale != 1.0:
                    # 计算无条件速度（使用零PXRD）
                    conditions_uncond = conditions.copy()
                    conditions_uncond['pxrd'] = torch.zeros_like(conditions['pxrd'])
                    conditions_uncond['pxrd_realtime'] = torch.zeros_like(conditions['pxrd'])
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
    
    def sample_with_pxrd_feedback(
        self, 
        conditions: Dict[str, torch.Tensor],
        pxrd_simulator,
        num_steps: int = None,
        temperature: float = 1.0,
        pxrd_weight: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        """
        带有实时PXRD反馈的采样（对用户透明的接口）
        
        在采样过程中实时计算PXRD并用于引导生成
        
        Args:
            conditions: 条件信息字典，包含：
                - comp: 原子组成 [batch_size, 52]，原子序数
                - pxrd: 目标PXRD谱 [batch_size, 11501]，已归一化的强度
                - num_atoms: 原子数量 [batch_size]
            pxrd_simulator: PXRD计算器实例
            num_steps: 采样步数
            temperature: 温度参数
            pxrd_weight: PXRD反馈权重
            
        Returns:
            生成的晶体结构 [batch_size, 55, 3]（物理空间）
            - 前3行：晶格参数矩阵（Angstrom单位）
            - 后52行：分数坐标 [0, 1]
        
        注意：
            输入和输出都在物理空间，归一化在内部自动处理
        """
        if num_steps is None:
            num_steps = self.default_num_steps
            
        device = conditions['pxrd'].device
        batch_size = conditions['pxrd'].shape[0]
        
        # 初始化噪声
        x = torch.randn(batch_size, 55, 3, device=device) * temperature * self.sigma_max
        
        # 目标PXRD
        target_pxrd = conditions['pxrd']
        
        # 时间步
        dt = 1.0 / num_steps
        
        # ODE积分
        for step in range(num_steps):
            t = torch.full((batch_size, 1), step * dt, device=device)
            r = t.clone()
            
            # 尝试计算当前结构的真实PXRD
            # 早期阶段（前25%）不尝试计算，因为结构太嘈杂
            if step > num_steps // 4:
                try:
                    # 将当前状态转换为晶体结构（需要先反归一化到物理空间）
                    x_physical = self.normalizer.denormalize_z(x.detach())
                    lattice = x_physical[:, :3].cpu().numpy()
                    frac_coords = x_physical[:, 3:].cpu().numpy()
                    comp = conditions['comp'].cpu().numpy()
                    num_atoms = conditions['num_atoms'].cpu().numpy()
                    
                    # 计算实时PXRD（批量计算）
                    realtime_pxrd = []
                    success_mask = []
                    
                    for i in range(batch_size):
                        try:
                            # TODO: 真正调用pxrd_simulator
                            # structure = create_structure(lattice[i], frac_coords[i][:num_atoms[i]], comp[i][:num_atoms[i]])
                            # _, pxrd_i = pxrd_simulator.simulate(structure)
                            # 暂时模拟：后期阶段假设能计算
                            if step > num_steps // 2:
                                # 模拟成功计算（实际应该调用simulator）
                                pxrd_i = target_pxrd[i] * 0.8 + torch.randn_like(target_pxrd[i]) * 0.2
                                realtime_pxrd.append(pxrd_i)
                                success_mask.append(True)
                            else:
                                realtime_pxrd.append(None)
                                success_mask.append(False)
                        except:
                            realtime_pxrd.append(None)
                            success_mask.append(False)
                    
                    # 只有当有成功计算的PXRD时才设置
                    if any(success_mask):
                        # 对于失败的样本，不提供realtime PXRD
                        processed_pxrd = torch.zeros(batch_size, target_pxrd.shape[1], device=device)
                        for i, (success, pxrd) in enumerate(zip(success_mask, realtime_pxrd)):
                            if success and pxrd is not None:
                                processed_pxrd[i] = pxrd
                        
                        # 只为成功的样本提供realtime PXRD
                        conditions['pxrd_realtime'] = processed_pxrd if any(success_mask) else None
                    else:
                        conditions['pxrd_realtime'] = None
                except:
                    # 计算失败，不提供realtime PXRD
                    conditions['pxrd_realtime'] = None
            else:
                # 早期阶段不提供realtime PXRD
                conditions['pxrd_realtime'] = None
            
            # 预测速度场
            with torch.no_grad():
                v = self.network(x, t, r, conditions)
                # 不再需要显式的PXRD引导项，因为网络通过encoder融合自动学习引导
            
            # Euler步进
            x = x + v * dt
        
        # 反归一化生成的结构（重要：将归一化空间的结果转换回物理空间）
        x = self.normalizer.denormalize_z(x)
        
        # 后处理：确保分数坐标在[0, 1]范围内
        x[:, 3:] = x[:, 3:] % 1.0
        
        return x