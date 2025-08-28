"""
EMA (Exponential Moving Average) 回调
用于生成模型的标准技巧，显著提升生成质量
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
from typing import Optional


class EMACallback(pl.Callback):
    """
    指数移动平均回调
    维护模型参数的移动平均版本，用于验证和推理
    
    原理：
    - 训练时更新EMA参数：ema_param = decay * ema_param + (1-decay) * param  
    - 验证时使用EMA模型替换原模型
    - 验证后恢复原模型继续训练
    
    优势：
    - 平滑参数波动，提升生成质量
    - 特别适合生成模型（扩散模型、Flow模型等）
    - 几乎无额外计算成本
    """
    
    def __init__(self, decay: float = 0.9999, update_every: int = 1):
        """
        Args:
            decay: EMA衰减率，越接近1越平滑（通常0.999-0.9999）
            update_every: 每N步更新一次EMA
        """
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.step = 0
        self.ema_model = None
        self.original_state = None
        
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """训练开始时创建EMA模型"""
        # 深拷贝网络参数作为EMA模型
        self.ema_model = deepcopy(pl_module.network.state_dict())
        print(f"✨ EMA initialized with decay={self.decay}")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """每个训练批次结束后更新EMA参数"""
        self.step += 1
        
        # 按照设定频率更新
        if self.step % self.update_every != 0:
            return
            
        # 更新EMA参数
        with torch.no_grad():
            for name, param in pl_module.network.named_parameters():
                if name in self.ema_model:
                    # EMA更新公式
                    self.ema_model[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )
                    
    def on_validation_epoch_start(self, trainer, pl_module):
        """验证开始时切换到EMA模型"""
        # 保存原始模型参数
        self.original_state = deepcopy(pl_module.network.state_dict())
        
        # 加载EMA参数
        if self.ema_model is not None:
            pl_module.network.load_state_dict(self.ema_model)
            
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证结束后恢复原始模型"""
        # 恢复原始参数继续训练
        if self.original_state is not None:
            pl_module.network.load_state_dict(self.original_state)
            self.original_state = None
            
    def on_test_epoch_start(self, trainer, pl_module):
        """测试时使用EMA模型"""
        # 测试时也使用EMA参数
        if self.ema_model is not None:
            self.original_state = deepcopy(pl_module.network.state_dict())
            pl_module.network.load_state_dict(self.ema_model)
            
    def on_test_epoch_end(self, trainer, pl_module):
        """测试结束后恢复"""
        if self.original_state is not None:
            pl_module.network.load_state_dict(self.original_state)
            self.original_state = None
            
    def state_dict(self):
        """返回EMA状态用于保存"""
        return {
            'ema_model': self.ema_model,
            'step': self.step,
            'decay': self.decay
        }
        
    def load_state_dict(self, state_dict):
        """加载EMA状态"""
        self.ema_model = state_dict.get('ema_model')
        self.step = state_dict.get('step', 0)
        self.decay = state_dict.get('decay', self.decay)