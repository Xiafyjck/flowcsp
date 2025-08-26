"""
Flow Matching 框架集合
"""

from .cfm_flow import CrystalCFM

__all__ = [
    'CrystalCFM',
]

# 工厂函数
def create_flow(flow_type: str = 'cfm', **kwargs):
    """
    创建指定类型的 Flow Matching 实例
    
    Args:
        flow_type: 'cfm' (标准 Conditional Flow Matching)
        **kwargs: 传递给构造函数的参数
    """
    flows = {
        'cfm': CrystalCFM,
    }
    
    if flow_type not in flows:
        raise ValueError(f"Unknown flow type: {flow_type}. Choose from {list(flows.keys())}")
    
    return flows[flow_type](**kwargs)