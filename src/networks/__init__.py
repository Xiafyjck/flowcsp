"""
统一网络接口和注册机制
"""

import torch.nn as nn
from typing import Dict, Optional
import torch


class BaseCSPNetwork(nn.Module):
    """所有CSP网络的基类 - 统一接口"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        comp: torch.Tensor,
        pxrd: torch.Tensor,
        noise_level: torch.Tensor,
        lattice: Optional[torch.Tensor] = None,
        frac_coords: Optional[torch.Tensor] = None,
        real_time_pxrd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        统一的前向传播接口 - 输出合并矩阵
        
        Args:
            comp: 晶胞原子数量 [B, num_elements]
            pxrd: 目标PXRD谱 [B, 11051]
            noise_level: 噪声水平 [B, 1]
            lattice: 晶格参数 [B, 3, 3]（可选，训练时提供）
            frac_coords: 分数坐标 [B, max_atoms, 3]（可选，训练时提供）
            real_time_pxrd: 实时计算的PXRD [B, 11051]（可选，采样时提供）
            
        Returns:
            output: 合并的输出矩阵 [B, 55, 3] - 前3行是晶格，后52行是分数坐标
        """
        raise NotImplementedError
        
    def prepare_batch(self, batch: dict) -> dict:
        """
        准备批次数据 - 子类可覆盖以进行特殊格式转换
        
        Args:
            batch: 原始数据字典
            
        Returns:
            处理后的数据字典
        """
        return batch


# 网络注册表
NETWORK_REGISTRY = {}


def register_network(name: str):
    """网络注册装饰器"""
    def decorator(cls):
        NETWORK_REGISTRY[name] = cls
        return cls
    return decorator


def build_network(name: str, config: dict) -> BaseCSPNetwork:
    """构建网络实例"""
    if name not in NETWORK_REGISTRY:
        raise ValueError(f"Unknown network: {name}. Available: {list(NETWORK_REGISTRY.keys())}")
    return NETWORK_REGISTRY[name](config)


# 导入具体网络实现（延迟导入避免循环依赖）
def _register_networks():
    try:
        from .transformer import TransformerCSPNetwork
    except ImportError:
        pass
        
    try:
        from .equiformer import EquiformerCSPNetwork
    except ImportError:
        pass


_register_networks()