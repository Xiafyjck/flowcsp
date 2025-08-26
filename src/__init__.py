"""
CSP (Crystal Structure Prediction) Package
"""

# Networks
from .networks import BaseCSPNetwork, build_network, NETWORK_REGISTRY

# Core modules
from .trainer import CSPLightningModule
from .meanflow import CrystalMeanFlow
from .data import CSPDataset, CSPDataModule, collate_batch
from .metrics import *
from .pxrd_simulator import *

__version__ = "3.0.0"  # 新架构版本

__all__ = [
    # Networks
    "BaseCSPNetwork",
    "build_network",
    "NETWORK_REGISTRY",
    # Trainer
    "CSPLightningModule",
    # MeanFlow
    "CrystalMeanFlow",
    # Data
    "CSPDataset",
    "CSPDataModule",
    "collate_batch",
]