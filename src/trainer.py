"""
PyTorch Lightning训练模块
管理训练、验证和测试流程，网络无关、flow无关
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Optional
import numpy as np

# 导入网络和流的构建函数
from src.networks import build_network
from src.flows import build_flow
from src.data import get_dataloader  


class CrystalGenerationModule(pl.LightningModule):
    """
    晶体生成的Lightning模块
    
    功能：
    - 管理网络和流模型的训练
    - 处理优化器和学习率调度
    - 记录训练指标
    - 支持DDP双卡训练
    """
    
    def __init__(
        self,
        network_name: str,
        flow_name: str,
        network_config: dict,
        flow_config: dict,
        optimizer_config: dict,
        scheduler_config: Optional[dict] = None,
    ):
        """
        初始化Lightning模块
        
        Args:
            network_name: 网络名称（'transformer' 或 'equiformer'）
            flow_name: 流模型名称（'cfm' 或 'meanflow'）
            network_config: 网络配置
            flow_config: 流模型配置
            optimizer_config: 优化器配置
            scheduler_config: 学习率调度器配置（可选）
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 构建网络和流模型
        self.network = build_network(network_name, network_config)
        self.flow = build_flow(flow_name, self.network, flow_config)
        
        # 配置
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        # 训练状态追踪
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
        
    def forward(self, batch: Dict) -> torch.Tensor:
        """
        前向传播（用于推理）
        
        Args:
            batch: 输入批次
            
        Returns:
            生成的晶体结构
        """
        # 准备条件
        conditions = {
            'comp': batch['comp'],
            'pxrd': batch['pxrd'],
            'num_atoms': batch['num_atoms']
        }
        
        # 使用流模型采样
        return self.flow.sample(conditions)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        训练步骤
        
        Args:
            batch: 训练批次
            batch_idx: 批次索引
            
        Returns:
            损失值
        """
        # 计算损失和指标
        loss, metrics = self.flow.compute_loss(batch)
        
        # 记录指标 - 使用sync_dist=True确保DDP兼容
        # 注意：on_step=True会显示当前步的损失，on_epoch=True会自动计算epoch平均
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch['z'].shape[0])
        
        # 记录其他指标
        for key, value in metrics.items():
            if key != 'loss':  # 避免重复记录loss
                self.log(f'train/{key}', value, on_step=False, on_epoch=True, sync_dist=True)
        
        # 每100步记录学习率
        if batch_idx % 100 == 0:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('train/lr', lr, on_step=True, on_epoch=False, sync_dist=True)
        
        # 保存输出用于epoch结束时的汇总
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'batch_size': batch['z'].shape[0]
        })
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """
        验证步骤
        
        Args:
            batch: 验证批次
            batch_idx: 批次索引
        """
        # 计算损失和指标
        loss, metrics = self.flow.compute_loss(batch)
        
        # 记录指标 - 添加batch_size以确保正确的加权平均
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch['z'].shape[0])
        
        for key, value in metrics.items():
            if key != 'loss':
                self.log(f'val/{key}', value, on_step=False, on_epoch=True, sync_dist=True)
        
        # 保存输出
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'batch_size': batch['z'].shape[0]
        })
        
        # 每个epoch第一个批次进行生成测试
        if batch_idx == 0:
            self._sample_and_evaluate(batch)
    
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """
        测试步骤
        
        Args:
            batch: 测试批次
            batch_idx: 批次索引
        """
        # 计算损失和指标
        loss, metrics = self.flow.compute_loss(batch)
        
        # 记录指标
        self.log('test/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        
        for key, value in metrics.items():
            if key != 'loss':
                self.log(f'test/{key}', value, on_step=False, on_epoch=True, sync_dist=True)
        
        # 生成样本并评估
        if batch_idx < 3:  # 只对前几个批次生成样本
            self._sample_and_evaluate(batch, prefix='test')
    
    def on_train_epoch_end(self) -> None:
        """训练epoch结束时的处理"""
        if self.training_step_outputs:
            # 计算加权平均损失
            total_loss = 0
            total_samples = 0
            
            for output in self.training_step_outputs:
                total_loss += output['loss'] * output['batch_size']
                total_samples += output['batch_size']
            
            if total_samples > 0:
                avg_loss = total_loss / total_samples
                # 打印epoch训练损失
                print(f"\n[Epoch {self.current_epoch}] Training Loss: {avg_loss:.6f}")
            
            # 清空输出列表
            self.training_step_outputs = []
    
    def on_validation_epoch_end(self) -> None:
        """验证epoch结束时的处理"""
        if self.validation_step_outputs:
            # 计算加权平均损失
            total_loss = 0
            total_samples = 0
            
            for output in self.validation_step_outputs:
                total_loss += output['loss'] * output['batch_size']
                total_samples += output['batch_size']
            
            if total_samples > 0:
                avg_loss = total_loss / total_samples
                
                # 检查是否是最佳模型
                if avg_loss < self.best_val_loss:
                    self.best_val_loss = avg_loss
                    print(f"[Epoch {self.current_epoch}] New best validation loss: {avg_loss:.6f} ✨")
                else:
                    print(f"[Epoch {self.current_epoch}] Validation Loss: {avg_loss:.6f}")
            
            # 清空输出列表
            self.validation_step_outputs = []
    
    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        
        Returns:
            优化器和调度器配置
        """
        # 创建优化器
        optimizer_type = self.optimizer_config.get('type', 'AdamW')
        lr = self.optimizer_config.get('lr', 1e-4)
        weight_decay = self.optimizer_config.get('weight_decay', 0.01)
        betas = self.optimizer_config.get('betas', (0.9, 0.999))
        
        if optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.flow.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas
            )
        elif optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(
                self.flow.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # 如果没有调度器配置，只返回优化器
        if self.scheduler_config is None:
            return optimizer
        
        # 创建学习率调度器
        scheduler_type = self.scheduler_config.get('type', 'CosineAnnealingLR')
        
        if scheduler_type == 'CosineAnnealingLR':
            # Cosine annealing with warm restarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 100),
                eta_min=self.scheduler_config.get('eta_min', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
            
        elif scheduler_type == 'CosineAnnealingWarmup':
            # 带warmup的cosine调度器
            warmup_steps = self.scheduler_config.get('warmup_steps', 1000)
            max_steps = self.scheduler_config.get('max_steps', 100000)
            
            def lr_lambda(step):
                # 添加除零保护
                if warmup_steps > 0 and step < warmup_steps:
                    # Linear warmup
                    return max(0.01, step / warmup_steps)  # 最低1%学习率
                else:
                    # Cosine annealing
                    if max_steps <= warmup_steps:
                        return 1.0  # 没有退火阶段
                    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                    return max(0.01, 0.5 * (1 + np.cos(np.pi * min(progress, 1.0))))  # 最低1%
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
            
        elif scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get('step_size', 30),
                gamma=self.scheduler_config.get('gamma', 0.1)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
            
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 10),
                min_lr=self.scheduler_config.get('min_lr', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _sample_and_evaluate(self, batch: Dict, prefix: str = 'val') -> None:
        """
        生成样本并评估质量
        
        Args:
            batch: 输入批次
            prefix: 日志前缀
        """
        with torch.no_grad():
            # 只使用前4个样本进行评估
            batch_size = min(4, batch['z'].shape[0])
            
            # 准备条件
            conditions = {
                'comp': batch['comp'][:batch_size],
                'pxrd': batch['pxrd'][:batch_size],
                'num_atoms': batch['num_atoms'][:batch_size]
            }
            
            # 生成样本（使用较少步数加快速度）
            samples = self.flow.sample(conditions, num_steps=30)
            
            # 确保samples在正确的设备上
            device = samples.device
            
            # 计算生成质量指标
            # 1. 晶格体积（在CPU上计算，然后转回GPU）
            lattice_vols = []
            for i in range(samples.shape[0]):
                lattice = samples[i, :3].cpu().numpy()
                vol = np.abs(np.linalg.det(lattice))
                lattice_vols.append(vol)
            
            avg_vol = torch.tensor(np.mean(lattice_vols), device=device)
            self.log(f'{prefix}/generated_volume', avg_vol, sync_dist=True)
            
            # 2. 坐标范围检查（保持在GPU上计算）
            coords = samples[:, 3:]  # 不要转到CPU
            coords_in_range = ((coords >= 0) & (coords <= 1)).float().mean()
            self.log(f'{prefix}/coords_in_range', coords_in_range, sync_dist=True)
            
            # 3. 计算与目标的MSE（确保在同一设备上）
            target = batch['z'][:batch_size].to(device)
            mse = torch.mean((samples - target) ** 2)
            self.log(f'{prefix}/sample_mse', mse, sync_dist=True)


class CrystalGenerationDataModule(pl.LightningDataModule):
    """
    数据模块，管理数据加载和处理
    """
    
    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        augment_permutation: bool = False,
        augment_so3: bool = False,
        augment_prob: float = 0.5,
        so3_augment_prob: float = 0.5,
        **dataset_kwargs
    ):
        """
        初始化数据模块
        
        Args:
            train_path: 训练数据路径
            val_path: 验证数据路径（可选，如果不提供则从训练集分割）
            test_path: 测试数据路径（可选）
            batch_size: 批量大小
            num_workers: 数据加载线程数
            augment_permutation: 是否使用置换数据增强
            augment_so3: 是否使用SO3数据增强
            augment_prob: 置换数据增强概率
            so3_augment_prob: SO3数据增强概率
            compute_realtime_pxrd: 是否计算实时PXRD
            realtime_pxrd_prob: 实时PXRD计算概率
            realtime_pxrd_min_t: 最小时间步
            cache_pxrd: 是否缓存PXRD
            **dataset_kwargs: 数据集的其他参数
        """
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_permutation = augment_permutation
        self.augment_so3 = augment_so3
        self.augment_prob = augment_prob
        self.so3_augment_prob = so3_augment_prob
        self.dataset_kwargs = dataset_kwargs
    
    def setup(self, stage: Optional[str] = None):
        """
        设置数据集
        
        Args:
            stage: 当前阶段（'fit', 'validate', 'test', 'predict'）
        """
        # 数据集会在各自的dataloader方法中创建
        pass
    
    def train_dataloader(self):
        """创建训练数据加载器"""
        return get_dataloader(
            self.train_path,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            augment_permutation=self.augment_permutation,
            augment_so3=self.augment_so3,
            augment_prob=self.augment_prob,
            so3_augment_prob=self.so3_augment_prob,
            **self.dataset_kwargs
        )
    
    def val_dataloader(self):
        """创建验证数据加载器"""
        if self.val_path is None:
            # 如果没有单独的验证集，使用训练集但不打乱
            return get_dataloader(
                self.train_path,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                augment_permutation=False,  # 验证时不使用数据增强
                augment_so3=False,
                **self.dataset_kwargs
            )
        
        return get_dataloader(
            self.val_path,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            augment_permutation=False,
            augment_so3=False,
            **self.dataset_kwargs
        )
    
    def test_dataloader(self):
        """创建测试数据加载器"""
        if self.test_path is None:
            return None
        
        return get_dataloader(
            self.test_path,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            augment_permutation=False,  # 测试时不使用数据增强
            augment_so3=False,
            **self.dataset_kwargs
        )