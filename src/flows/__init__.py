"""
Flow registration and interface  
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

# Registry for different flow models
FLOW_REGISTRY = {}

def register_flow(name):
    """Decorator to register flow classes"""
    def decorator(cls):
        FLOW_REGISTRY[name] = cls
        return cls
    return decorator

def get_flow(name: str, **kwargs):
    """Get flow by name"""
    if name not in FLOW_REGISTRY:
        raise ValueError(f"Unknown flow: {name}. Available: {list(FLOW_REGISTRY.keys())}")
    return FLOW_REGISTRY[name](**kwargs)

def build_flow(name: str, network: nn.Module, config: dict):
    """Build flow instance with network"""
    if name not in FLOW_REGISTRY:
        raise ValueError(f"Unknown flow: {name}. Available: {list(FLOW_REGISTRY.keys())}")
    return FLOW_REGISTRY[name](network, config)


class BaseFlow(nn.Module, ABC):
    """Base class for all flow models"""
    
    def __init__(self, network: nn.Module, config: dict):
        super().__init__()
        self.network = network
        self.config = config
        
    @abstractmethod
    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute training loss
        
        Args:
            batch: Dictionary containing training data
            
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of additional metrics for logging
        """
        pass
    
    @abstractmethod  
    def sample(
        self, 
        conditions: Dict[str, torch.Tensor],
        num_steps: int = 50,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples from the flow
        
        Args:
            conditions: Dictionary of conditioning information
            num_steps: Number of sampling steps (default: 50)
            temperature: Sampling temperature
            
        Returns:
            samples: Generated samples [batch_size, 63, 3]
        """
        pass


# Import concrete implementations
def _register_flows():
    try:
        from .cfm import CFMFlow
    except ImportError:
        pass
    
    try:
        from .cfm_cfg import CFMFlowCFG
    except ImportError:
        pass
        
    try:
        from .meanflow import MeanFlow
    except ImportError:
        pass


_register_flows()