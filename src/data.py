"""
Data loading module for crystal structure generation
Handles pickle format from preprocessing
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Optional, Tuple, List
from pathlib import Path

class SO3Augmentation:
    """
    SO3等变性数据增强模块
    通过对晶格参数应用随机旋转变换来增强数据
    
    原理：
    - 晶格向量起点固定在原点，只能旋转不能平移
    - 只对初始数据的晶格参数施加SO3旋转变换
    - Flow Matching的线性插值会自动保持等变性
    - 三种等变性通过同一种方式实现：
      1. 晶格参数SO3变换对分数坐标的不变性：model_frac(SO3(lattice), ...) = model_frac(lattice, ...)
      2. 晶格参数SO3变换对条件(comp,pxrd)的不变性
      3. 晶格参数去噪网络的SO3等变性：model_lattice(SO3(lattice), ...) = SO3(model_lattice(lattice, ...))
    """
    
    def __init__(self, augment_prob: float = 0.5):
        """
        Args:
            augment_prob: 应用数据增强的概率
        """
        self.augment_prob = augment_prob
    
    def _random_rotation_matrix(self, device: torch.device) -> torch.Tensor:
        """
        生成随机SO(3)旋转矩阵
        使用Rodrigues旋转公式生成均匀分布的旋转
        
        Returns:
            旋转矩阵 [3, 3]
        """
        # 生成随机旋转轴（单位向量）
        axis = torch.randn(3, device=device)
        axis = axis / torch.norm(axis)
        
        # 生成随机旋转角度 [0, 2π]
        angle = torch.rand(1, device=device) * 2 * torch.pi
        
        # 使用Rodrigues公式构建旋转矩阵
        # R = I + sin(θ)K + (1-cos(θ))K²
        # 其中K是轴向量的叉乘矩阵
        K = torch.zeros(3, 3, device=device)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]
        
        I = torch.eye(3, device=device)
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
        
        return R
    
    
    def apply_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        对batch应用SO3增强
        
        重要：这个增强应用于初始数据（t=0时刻），用于训练Flow Matching模型
        Flow Matching的线性插值会自动保持等变性
        
        Args:
            batch: 包含z, comp, pxrd等的batch数据（初始数据）
            
        Returns:
            增强后的batch
        """
        batch_size = batch['z'].shape[0]
        device = batch['z'].device
        
        # 对每个样本独立应用增强
        for i in range(batch_size):
            if torch.rand(1).item() > self.augment_prob:
                continue  # 跳过不需要增强的样本
            
            # 生成随机SO(3)旋转矩阵
            rotation = self._random_rotation_matrix(device)
            
            # 提取晶格参数（前3行）
            lattice = batch['z'][i, :3, :]  # [3, 3]
            
            # 应用SO(3)旋转到晶格参数
            # lattice' = R @ lattice
            # 注意：不能平移，晶格向量起点必须在原点
            transformed_lattice = rotation @ lattice
            
            # 更新晶格参数
            batch['z'][i, :3, :] = transformed_lattice
            
            # 注意：分数坐标不需要变换（它们相对于晶格保持不变）
            # 这是因为分数坐标是相对于晶格基向量定义的
            # 当晶格旋转时，分数坐标自动"跟随"旋转
        
        return batch
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """使SO3Augmentation可以作为函数调用"""
        return self.apply_batch(batch)


class PermutationAugmentation:
    """
    置换等变性数据增强模块
    通过随机置换原子顺序来增强数据，保持晶体结构的物理等价性
    
    原理：
    - 对于相同元素的原子，交换它们的位置不改变晶体的物理性质
    - 使用置换矩阵P对comp和frac_coords进行相同的变换
    - 保证模型满足：model(P(comp), ...) = P(model(comp, ...))
    """
    
    def __init__(self, augment_prob: float = 0.5, max_atoms: int = 60):
        """
        Args:
            augment_prob: 应用数据增强的概率
            max_atoms: 最大原子数量
        """
        self.augment_prob = augment_prob
        self.max_atoms = max_atoms
    
    def _group_atoms_by_type(self, comp: torch.Tensor, num_atoms: int) -> Dict[int, List[int]]:
        """
        按原子类型分组原子索引
        
        Args:
            comp: 原子组成向量 [max_atoms]
            num_atoms: 实际原子数量
            
        Returns:
            字典，键为原子类型(Z)，值为该类型原子的索引列表
        """
        groups = {}
        for i in range(num_atoms):
            atom_type = int(comp[i].item())
            if atom_type > 0:  # 忽略padding的0值
                if atom_type not in groups:
                    groups[atom_type] = []
                groups[atom_type].append(i)
        return groups
    
    def _create_permutation_matrix(self, n: int, perm: torch.Tensor) -> torch.Tensor:
        """
        创建置换矩阵
        使用F.one_hot实现，更简洁高效
        
        Args:
            n: 矩阵维度
            perm: 置换索引向量
            
        Returns:
            置换矩阵 [n, n]
        """
        # 使用one_hot创建置换矩阵，符合CLAUDE.md的建议
        import torch.nn.functional as F
        perm_matrix = F.one_hot(perm, num_classes=n).float()
        return perm_matrix
    
    def apply_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        对batch应用置换增强
        
        Args:
            batch: 包含z, comp, pxrd等的batch数据
            
        Returns:
            增强后的batch
        """
        batch_size = batch['z'].shape[0]
        device = batch['z'].device
        
        # 对每个样本独立应用增强
        for i in range(batch_size):
            if torch.rand(1).item() > self.augment_prob:
                continue  # 跳过不需要增强的样本
            
            num_atoms = int(batch['num_atoms'][i].item())
            if num_atoms <= 1:
                continue  # 单原子或无原子不需要置换
            
            # 按原子类型分组
            atom_groups = self._group_atoms_by_type(batch['comp'][i], num_atoms)
            
            # 为每组相同类型的原子创建随机置换
            perm = torch.arange(self.max_atoms, device=device)
            
            for atom_type, indices in atom_groups.items():
                if len(indices) <= 1:
                    continue  # 单个原子不需要置换
                
                # 对该组原子进行随机置换
                group_indices = torch.tensor(indices, device=device)
                shuffled = group_indices[torch.randperm(len(indices), device=device)]
                perm[group_indices] = shuffled
            
            # 应用置换到comp和frac_coords
            # 注意：z的前3行是晶格参数，不参与置换
            batch['comp'][i] = batch['comp'][i][perm]
            batch['atom_types'][i] = batch['atom_types'][i][perm]
            
            # 对frac_coords进行置换（z的第3行之后）
            frac_coords = batch['z'][i, 3:, :]  # [60, 3]
            batch['z'][i, 3:, :] = frac_coords[perm[:60], :]
        
        return batch
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """使PermutationAugmentation可以作为函数调用"""
        return self.apply_batch(batch)


class CrystalDataset(Dataset):
    """
    Dataset for crystal structures with PXRD patterns
    Loads preprocessed pickle files
    """
    
    def __init__(
        self,
        data_path: str,
        max_atoms: int = 60,
        pxrd_dim: int = 11501,
        transform = None
    ):
        """
        Args:
            data_path: Path to pickle file
            max_atoms: Maximum number of atoms (for padding)
            pxrd_dim: Dimension of PXRD pattern
            use_niggli: Use Niggli reduced cell (True) or primitive cell (False)
            transform: Optional data transformation
        """
        self.max_atoms = max_atoms
        self.pxrd_dim = pxrd_dim
        self.transform = transform
        
        # Load data
        print(f"Loading data from {data_path}...")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Convert to list if DataFrame
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.to_dict('records')
        
        original_count = len(self.data)
        
        # 过滤掉原子数超过max_atoms的样品
        filtered_data = []
        for sample in self.data:
            # 获取结构来检查原子数
            structure = sample.get('niggli_structure')
            if structure is not None:
                num_atoms = len(structure)
                if num_atoms <= max_atoms:
                    filtered_data.append(sample)
        
        self.data = filtered_data
        filtered_count = len(self.data)
        
        if filtered_count < original_count:
            print(f"⚠️ 过滤了 {original_count - filtered_count} 个原子数超过 {max_atoms} 的样品")
        
        print(f"Loaded {filtered_count} samples (original: {original_count})")
        
        # Verify first sample
        if len(self.data) > 0:
            sample = self.data[0]
            print(f"Sample keys: {list(sample.keys())}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - z: Combined lattice and fractional coordinates [63, 3]
                - comp: Atomic composition [60]
                - pxrd: PXRD pattern [11501]
                - num_atoms: Number of actual atoms (scalar)
                - atom_types: Atomic numbers [60]
        """
        sample = self.data[idx]
        
        # Choose structure type
        structure = sample['niggli_structure']
        
        # Extract data
        lattice_matrix = np.array(structure.lattice.matrix, dtype=np.float32)  # [3, 3]
        frac_coords = np.array(structure.frac_coords, dtype=np.float32)  # [num_atoms, 3]
        atom_types = np.array([site.specie.Z for site in structure], dtype=np.int32)  # [num_atoms]
        num_atoms = len(atom_types)
        pxrd = sample['pxrd'].astype(np.float32)  # [11501]
        
        # Pad fractional coordinates and atom types to max_atoms
        padded_frac_coords = np.zeros((self.max_atoms, 3), dtype=np.float32)
        padded_atom_types = np.zeros(self.max_atoms, dtype=np.int32)
        
        # 由于已经在加载时过滤，这里num_atoms一定 <= max_atoms
        if num_atoms > 0:
            padded_frac_coords[:num_atoms] = frac_coords
            padded_atom_types[:num_atoms] = atom_types
        
        # Create composition vector (atomic numbers at each position)
        comp = padded_atom_types.astype(np.float32)
        
        # Combine lattice and coordinates into single matrix z
        z = np.concatenate([lattice_matrix, padded_frac_coords], axis=0)  # [63, 3]
        
        # Convert to tensors
        output = {
            'z': torch.from_numpy(z),
            'comp': torch.from_numpy(comp),
            'pxrd': torch.from_numpy(pxrd),
            'num_atoms': torch.tensor(num_atoms, dtype=torch.long),  # 现在可以安全使用原始值
            'atom_types': torch.from_numpy(padded_atom_types),
            'id': str(sample.get('id', idx))
        }
        
        # Apply transform if provided
        if self.transform:
            output = self.transform(output)
        
        return output


def collate_crystal_batch(
    batch, 
    permutation_aug: Optional[PermutationAugmentation] = None,
    so3_aug: Optional[SO3Augmentation] = None
):
    """
    Custom collate function with optional augmentations
    
    Args:
        batch: List of dictionaries from dataset
        permutation_aug: Optional PermutationAugmentation for atom permutation
        so3_aug: Optional SO3Augmentation for lattice rotation
    
    Returns:
        Collated and optionally augmented batch dictionary
    """
    # Stack all tensors
    z = torch.stack([item['z'] for item in batch])
    comp = torch.stack([item['comp'] for item in batch])
    pxrd = torch.stack([item['pxrd'] for item in batch])
    num_atoms = torch.stack([item['num_atoms'] for item in batch])
    atom_types = torch.stack([item['atom_types'] for item in batch])
    ids = [item['id'] for item in batch]
    
    collated_batch = {
        'z': z,
        'comp': comp,
        'pxrd': pxrd,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'ids': ids
    }
    
    # 先应用SO3增强（晶格旋转）
    if so3_aug is not None:
        collated_batch = so3_aug(collated_batch)
    
    # 再应用置换增强（原子顺序）
    if permutation_aug is not None:
        collated_batch = permutation_aug(collated_batch)
    
    return collated_batch


def get_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    augment_permutation: bool = False,
    augment_so3: bool = False,
    augment_prob: float = 0.5,
    so3_augment_prob: float = 0.5,
    **dataset_kwargs
) -> DataLoader:
    """
    Create dataloader with optional augmentations
    
    Args:
        data_path: Path to pickle file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        augment_permutation: Whether to apply permutation augmentation
        augment_so3: Whether to apply SO3 augmentation
        augment_prob: Probability of applying permutation augmentation
        so3_augment_prob: Probability of applying SO3 augmentation
        **dataset_kwargs: Additional arguments for CrystalDataset
    
    Returns:
        DataLoader instance with optional augmentations
    """
    dataset = CrystalDataset(data_path, **dataset_kwargs)
    
    # 创建数据增强实例（如果需要）
    permutation_aug = PermutationAugmentation(augment_prob=augment_prob) if augment_permutation else None
    so3_aug = SO3Augmentation(augment_prob=so3_augment_prob) if augment_so3 else None
    
    # 创建带有增强的collate函数
    def collate_fn(batch):
        return collate_crystal_batch(
            batch, 
            permutation_aug=permutation_aug,
            so3_aug=so3_aug
        )
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return dataloader


def split_dataset(
    dataset: CrystalDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test
    
    Args:
        dataset: CrystalDataset instance
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed for splitting
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    return train_dataset, val_dataset, test_dataset