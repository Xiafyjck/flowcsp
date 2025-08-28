"""
晶体结构数据归一化模块
负责晶格参数和分数坐标的归一化和反归一化
"""

import torch
import json
from pathlib import Path
from typing import Dict, Optional


class DataNormalizer:
    """
    数据归一化和反归一化类
    主要用于晶格参数的标准化，提升训练稳定性
    
    归一化策略：
    - 晶格参数：使用全局或分维度的mean/std进行标准化
    - 分数坐标：通常不需要归一化（已在[0,1]范围）
    """
    
    def __init__(
        self,
        stats_file: Optional[str] = None,
        normalize_lattice: bool = True,
        normalize_frac_coords: bool = False,
        use_global_stats: bool = True
    ):
        """
        初始化归一化器
        
        Args:
            stats_file: 统计信息文件路径（JSON格式）
            normalize_lattice: 是否归一化晶格参数
            normalize_frac_coords: 是否归一化分数坐标（通常不需要，已在[0,1]）
            use_global_stats: 使用全局统计还是分维度统计
        """
        self.normalize_lattice = normalize_lattice
        self.normalize_frac_coords = normalize_frac_coords
        self.use_global_stats = use_global_stats
        
        # 默认统计值（基于 merged_cdvae_total.pkl 计算，全部74310样本）
        self.default_stats = {
            'lattice_global_mean': 0.5582,
            'lattice_global_std': 3.3490,
            'lattice_mean': [1.8232, 0.3099, 0.0494, -0.1238, 2.0166, -0.3753, -0.4834, 0.8804, 0.9270],
            'lattice_std': [2.7895, 2.6121, 1.8521, 2.3274, 3.7704, 1.9673, 2.4782, 2.7006, 6.2662],
            'frac_coords_mean': [0.4454, 0.4369, 0.4450],
            'frac_coords_std': [0.2938, 0.2935, 0.2940]
        }
        
        # 加载统计信息
        if stats_file and Path(stats_file).exists():
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
        else:
            self.stats = self.default_stats
        
        # 转换为张量以便GPU计算
        self._prepare_tensors()
    
    def _prepare_tensors(self):
        """预处理统计张量，便于GPU加速计算"""
        if self.normalize_lattice:
            if self.use_global_stats:
                # 使用全局mean/std（推荐）
                self.lattice_mean = torch.tensor(self.stats['lattice_global_mean'], dtype=torch.float32)
                self.lattice_std = torch.tensor(self.stats['lattice_global_std'], dtype=torch.float32)
            else:
                # 使用分维度mean/std
                self.lattice_mean = torch.tensor(self.stats['lattice_mean'], dtype=torch.float32).reshape(3, 3)
                self.lattice_std = torch.tensor(self.stats['lattice_std'], dtype=torch.float32).reshape(3, 3)
        
        if self.normalize_frac_coords:
            self.frac_mean = torch.tensor(self.stats['frac_coords_mean'], dtype=torch.float32)
            self.frac_std = torch.tensor(self.stats['frac_coords_std'], dtype=torch.float32)
    
    def normalize(self, batch: Dict[str, torch.Tensor], inplace: bool = True) -> Dict[str, torch.Tensor]:
        """
        归一化batch数据
        
        数据流：原始物理空间 -> 归一化空间
        
        Args:
            batch: 包含z, comp等的batch字典
            inplace: 是否原地修改（节省内存）
            
        Returns:
            归一化后的batch
        """
        if not inplace:
            batch = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        if self.normalize_lattice and 'z' in batch:
            device = batch['z'].device
            
            # 归一化晶格参数（z的前3行）
            if self.use_global_stats:
                # 全局归一化：所有晶格参数使用相同的mean/std
                # 添加epsilon保护防止除零（即使在FP32下也需要）
                safe_std = torch.clamp(self.lattice_std.to(device), min=1e-6)
                batch['z'][:, :3, :] = (batch['z'][:, :3, :] - self.lattice_mean.to(device)) / safe_std
            else:
                # 分维度归一化：每个晶格参数使用独立的mean/std
                safe_std = torch.clamp(self.lattice_std.to(device), min=1e-6)
                batch['z'][:, :3, :] = (batch['z'][:, :3, :] - self.lattice_mean.to(device)) / safe_std
            
            # 分数坐标归一化（如果需要）
            if self.normalize_frac_coords:
                # z的第3行之后是分数坐标
                safe_frac_std = torch.clamp(self.frac_std.to(device), min=1e-6)
                batch['z'][:, 3:, :] = (batch['z'][:, 3:, :] - self.frac_mean.to(device)) / safe_frac_std
        
        return batch
    
    def denormalize(self, batch: Dict[str, torch.Tensor], inplace: bool = True) -> Dict[str, torch.Tensor]:
        """
        反归一化batch数据
        
        数据流：归一化空间 -> 原始物理空间
        
        Args:
            batch: 归一化的batch
            inplace: 是否原地修改
            
        Returns:
            反归一化后的batch（物理空间）
        """
        if not inplace:
            batch = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        if self.normalize_lattice and 'z' in batch:
            device = batch['z'].device
            
            # 反归一化晶格参数
            if self.use_global_stats:
                safe_std = torch.clamp(self.lattice_std.to(device), min=1e-6)
                batch['z'][:, :3, :] = batch['z'][:, :3, :] * safe_std + self.lattice_mean.to(device)
            else:
                safe_std = torch.clamp(self.lattice_std.to(device), min=1e-6)
                batch['z'][:, :3, :] = batch['z'][:, :3, :] * safe_std + self.lattice_mean.to(device)
            
            # 反归一化分数坐标
            if self.normalize_frac_coords:
                safe_frac_std = torch.clamp(self.frac_std.to(device), min=1e-6)
                batch['z'][:, 3:, :] = batch['z'][:, 3:, :] * safe_frac_std + self.frac_mean.to(device)
        
        return batch
    
    def normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        仅归一化z张量（便捷方法）
        
        Args:
            z: 原始z张量 [batch, 63, 3]
            
        Returns:
            归一化的z张量
        """
        z_norm = z.clone()
        device = z.device
        
        if self.normalize_lattice:
            if self.use_global_stats:
                safe_std = torch.clamp(self.lattice_std.to(device), min=1e-6)
                z_norm[:, :3, :] = (z[:, :3, :] - self.lattice_mean.to(device)) / safe_std
            else:
                safe_std = torch.clamp(self.lattice_std.to(device), min=1e-6)
                z_norm[:, :3, :] = (z[:, :3, :] - self.lattice_mean.to(device)) / safe_std
            
            if self.normalize_frac_coords:
                safe_frac_std = torch.clamp(self.frac_std.to(device), min=1e-6)
                z_norm[:, 3:, :] = (z[:, 3:, :] - self.frac_mean.to(device)) / safe_frac_std
        
        return z_norm
    
    def denormalize_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        仅反归一化z张量（便捷方法）
        
        用于采样后的后处理，将归一化空间的结果转换回物理空间
        
        Args:
            z: 归一化的z张量 [batch, 63, 3]
            
        Returns:
            反归一化的z张量（物理空间）
        """
        z_denorm = z.clone()
        device = z.device
        
        if self.normalize_lattice:
            if self.use_global_stats:
                safe_std = torch.clamp(self.lattice_std.to(device), min=1e-6)
                z_denorm[:, :3, :] = z[:, :3, :] * safe_std + self.lattice_mean.to(device)
            else:
                safe_std = torch.clamp(self.lattice_std.to(device), min=1e-6)
                z_denorm[:, :3, :] = z[:, :3, :] * safe_std + self.lattice_mean.to(device)
            
            if self.normalize_frac_coords:
                safe_frac_std = torch.clamp(self.frac_std.to(device), min=1e-6)
                z_denorm[:, 3:, :] = z[:, 3:, :] * safe_frac_std + self.frac_mean.to(device)
        
        return z_denorm
    
    def compute_stats_from_dataset(self, dataset, num_samples: Optional[int] = None) -> Dict:
        """
        从数据集计算统计信息
        
        Args:
            dataset: CrystalDataset实例
            num_samples: 用于计算的样本数（None表示使用全部）
            
        Returns:
            统计信息字典
        """
        # 采样数据
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))
        
        # 收集晶格参数
        lattices = []
        frac_coords_list = []
        
        for i in range(num_samples):
            data = dataset[i]
            z = data['z']  # [63, 3]
            
            # 晶格参数（前3行）
            lattice = z[:3].flatten()  # [9]
            lattices.append(lattice)
            
            # 分数坐标（第3行之后）
            frac_coords = z[3:]  # [60, 3]
            frac_coords_list.append(frac_coords.flatten())
        
        # 转换为张量
        lattices = torch.stack(lattices)  # [num_samples, 9]
        frac_coords_tensor = torch.stack(frac_coords_list)  # [num_samples, 60*3]
        
        # 计算统计
        stats = {
            # 全局晶格统计
            'lattice_global_mean': float(lattices.mean()),
            'lattice_global_std': float(lattices.std()),
            
            # 分维度晶格统计
            'lattice_mean': lattices.mean(dim=0).tolist(),
            'lattice_std': lattices.std(dim=0).tolist(),
            
            # 分数坐标统计（按xyz分组）
            'frac_coords_mean': [
                float(frac_coords_tensor[:, 0::3].mean()),  # x坐标
                float(frac_coords_tensor[:, 1::3].mean()),  # y坐标
                float(frac_coords_tensor[:, 2::3].mean()),  # z坐标
            ],
            'frac_coords_std': [
                float(frac_coords_tensor[:, 0::3].std()),
                float(frac_coords_tensor[:, 1::3].std()),
                float(frac_coords_tensor[:, 2::3].std()),
            ]
        }
        
        return stats
    
    def save_stats(self, stats: Dict, file_path: str):
        """
        保存统计信息到JSON文件
        
        Args:
            stats: 统计信息字典
            file_path: 保存路径
        """
        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"统计信息已保存到: {file_path}")