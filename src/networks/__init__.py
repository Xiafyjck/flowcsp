"""
统一网络接口和注册机制
"""

import torch.nn as nn
from typing import Dict, Optional
import torch
from abc import ABC, abstractmethod


class BaseCSPNetwork(nn.Module, ABC):
    """所有CSP网络的基类 - 统一接口"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(
        self, 
        z: torch.Tensor,           # [batch_size, 3+60, 3] (lattice + frac_coords)
        t: torch.Tensor,           # [batch_size, 1] time step
        r: torch.Tensor,           # [batch_size, 1] another time step  
        conditions: Dict[str, torch.Tensor] # conditions including comp, pxrd, etc
    ) -> torch.Tensor:
        """
        Forward pass of the network - IMPORTANT: Do not modify this signature
        
        Args:
            z: Current state [batch_size, 3+60, 3]
               - z[:, :3, :] are lattice vectors
               - z[:, 3:, :] are fractional coordinates
            t: Time step [batch_size, 1]
            r: Another time step for flow [batch_size, 1]
            conditions: Dictionary containing:
                - 'comp': [batch_size, 60] atomic composition 
                - 'pxrd': [batch_size, 11501] PXRD pattern
                - 'pxrd_realtime': [batch_size, 11501] real-time PXRD (optional)
                - 'num_atoms': [batch_size] number of atoms per sample
                
        Returns:
            Output in same shape as z: [batch_size, 3+60, 3]
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
    
    try:
        from .crystal_transformer import CrystalTransformer
    except ImportError:
        pass


_register_networks()