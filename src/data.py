"""
数据加载模块 - 晶体结构生成
仅支持npz格式，使用fp16存储PXRD以节省内存
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time


class SO3Augmentation:
    """
    SO3等变性数据增强模块
    通过对晶格参数应用随机旋转变换来增强数据
    """
    
    def __init__(self, augment_prob: float = 0.5):
        self.augment_prob = augment_prob
    
    def _random_rotation_matrix(self, device: torch.device) -> torch.Tensor:
        """生成随机SO(3)旋转矩阵"""
        # 生成随机旋转轴（单位向量）
        axis = torch.randn(3, device=device)
        axis = axis / torch.norm(axis)
        
        # 生成随机旋转角度 [0, 2π]
        angle = torch.rand(1, device=device) * 2 * torch.pi
        
        # 使用Rodrigues公式构建旋转矩阵
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
        """对batch应用SO3增强"""
        batch_size = batch['z'].shape[0]
        device = batch['z'].device
        
        for i in range(batch_size):
            if torch.rand(1).item() > self.augment_prob:
                continue
            
            # 生成随机SO(3)旋转矩阵
            rotation = self._random_rotation_matrix(device)
            
            # 提取并旋转晶格参数（前3行）
            lattice = batch['z'][i, :3, :]  # [3, 3]
            batch['z'][i, :3, :] = rotation @ lattice
        
        return batch
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.apply_batch(batch)


class PermutationAugmentation:
    """
    置换等变性数据增强模块
    通过随机置换原子顺序来增强数据
    """
    
    def __init__(self, augment_prob: float = 0.5, max_atoms: int = 60):
        self.augment_prob = augment_prob
        self.max_atoms = max_atoms
    
    def _group_atoms_by_type(self, comp: torch.Tensor, num_atoms: int) -> Dict[int, List[int]]:
        """按原子类型分组原子索引"""
        groups = {}
        for i in range(num_atoms):
            atom_type = int(comp[i].item())
            if atom_type > 0:  # 忽略padding的0值
                if atom_type not in groups:
                    groups[atom_type] = []
                groups[atom_type].append(i)
        return groups
    
    def apply_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """对batch应用置换增强"""
        batch_size = batch['z'].shape[0]
        device = batch['z'].device
        
        for i in range(batch_size):
            if torch.rand(1).item() > self.augment_prob:
                continue
            
            num_atoms = int(batch['num_atoms'][i].item())
            if num_atoms <= 1:
                continue
            
            # 按原子类型分组
            atom_groups = self._group_atoms_by_type(batch['comp'][i], num_atoms)
            
            # 为每组相同类型的原子创建随机置换
            perm = torch.arange(self.max_atoms, device=device)
            
            for atom_type, indices in atom_groups.items():
                if len(indices) <= 1:
                    continue
                
                # 对该组原子进行随机置换
                group_indices = torch.tensor(indices, device=device)
                shuffled = group_indices[torch.randperm(len(indices), device=device)]
                perm[group_indices] = shuffled
            
            # 应用置换到comp和atom_types
            batch['comp'][i] = batch['comp'][i][perm]
            batch['atom_types'][i] = batch['atom_types'][i][perm]
            
            # 对frac_coords进行置换（z的第3行之后）
            frac_coords = batch['z'][i, 3:, :]  # [60, 3]
            batch['z'][i, 3:, :] = frac_coords[perm[:60], :]
        
        return batch
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.apply_batch(batch)


class CrystalDataset(Dataset):
    """
    晶体结构数据集 - 仅支持npz格式
    使用fp16存储PXRD以节省内存，初始化时一次性加载所有数据到内存
    """
    
    def __init__(
        self,
        data_path: str,
        max_atoms: int = 60,
        pxrd_dim: int = 11501,
        transform = None,
        cache_in_memory: bool = True  # 默认开启内存缓存
    ):
        """
        Args:
            data_path: npz缓存目录路径
            max_atoms: 最大原子数量
            pxrd_dim: PXRD维度
            transform: 可选的数据变换
            cache_in_memory: 是否将所有数据缓存到内存（默认True）
        """
        self.max_atoms = max_atoms
        self.pxrd_dim = pxrd_dim
        self.transform = transform
        self.data_path = Path(data_path)
        self.cache_in_memory = True or cache_in_memory
        
        # 验证路径
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
        
        if not self.data_path.is_dir():
            raise ValueError(f"数据路径必须是目录: {data_path}")
        
        # 加载元数据
        metadata_path = self.data_path / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"找不到metadata.json文件。\n"
                f"请使用 scripts/warmup_cache.py 生成数据集缓存。"
            )
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # 验证格式
        if self.metadata.get('format') != 'npz_batched':
            raise ValueError(
                f"数据格式不正确，期望'npz_batched'，得到'{self.metadata.get('format')}'"
            )
        
        self.n_samples = self.metadata['n_samples']
        self.n_batches = self.metadata['n_batches']
        self.batch_size = self.metadata['batch_size']
        self.ids = self.metadata.get('ids', [str(i) for i in range(self.n_samples)])
        
        # 获取所有批次文件
        self.batch_files = sorted(self.data_path.glob('batch_*.npz'))
        if len(self.batch_files) == 0:
            raise FileNotFoundError(f"找不到批次文件 (batch_*.npz) in {self.data_path}")
        
        print(f"Loading npz dataset from {self.data_path}...")
        print(f"  Samples: {self.n_samples}")
        print(f"  Batches: {len(self.batch_files)}")
        
        # 一次性加载所有数据到内存
        if self.cache_in_memory:
            print(f"  Loading all data into memory...")
            start_time = time.time()
            
            # 初始化存储所有数据的列表
            all_z = []
            all_comp = []
            all_atom_types = []
            all_pxrd = []
            all_num_atoms = []
            all_ids = []
            
            # 加载所有批次文件
            for i, batch_file in enumerate(self.batch_files):
                with np.load(batch_file) as data:
                    all_z.append(data['z'])
                    all_comp.append(data['comp'])
                    all_atom_types.append(data['atom_types'])
                    all_pxrd.append(data['pxrd'])  # 保持fp16格式
                    all_num_atoms.append(data['num_atoms'])
                    if 'ids' in data:
                        all_ids.extend(data['ids'])
                
                # 打印进度
                if (i + 1) % 10 == 0 or (i + 1) == len(self.batch_files):
                    print(f"    Loaded {i+1}/{len(self.batch_files)} batches...")
            
            # 合并所有批次为单个numpy数组
            self.all_data = {
                'z': np.concatenate(all_z, axis=0),
                'comp': np.concatenate(all_comp, axis=0),
                'atom_types': np.concatenate(all_atom_types, axis=0),
                'pxrd': np.concatenate(all_pxrd, axis=0),  # 保持fp16格式
                'num_atoms': np.concatenate(all_num_atoms, axis=0)
            }
            
            # 处理IDs
            if all_ids:
                self.all_data['ids'] = np.array(all_ids)
            else:
                self.all_data['ids'] = np.array([str(i) for i in range(self.n_samples)])
            
            # 计算加载时间
            load_time = time.time() - start_time
            
            # 显示内存使用信息
            total_memory = 0
            for key, arr in self.all_data.items():
                if key != 'ids':
                    memory_gb = arr.nbytes / (1024**3)
                    total_memory += memory_gb
                    print(f"    {key}: {arr.shape} {arr.dtype} ({memory_gb:.3f} GB)")
            
            print(f"  Total memory usage: {total_memory:.3f} GB")
            print(f"  Loading time: {load_time:.2f} seconds")
        else:
            # 不使用内存缓存时，保持原有的延迟加载逻辑
            self.all_data = None
            print(f"  Using lazy loading mode (cache_in_memory=False)")
            
            # 创建索引映射（样本索引 -> (批次文件索引, 批次内索引)）
            self.index_map = []
            for batch_idx, batch_file in enumerate(self.batch_files):
                # 获取该批次的样本数
                with np.load(batch_file) as data:
                    batch_samples = len(data['z'])
                for sample_idx in range(batch_samples):
                    self.index_map.append((batch_idx, sample_idx))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - z: 晶格和分数坐标 [63, 3]
                - comp: 原子组成 [60]
                - pxrd: PXRD谱图 [11501] (从fp16转换为fp32)
                - num_atoms: 实际原子数量
                - atom_types: 原子类型 [60]
                - id: 样本ID
        """
        if self.cache_in_memory:
            # 从内存中直接读取数据
            z = self.all_data['z'][idx]
            comp = self.all_data['comp'][idx]
            atom_types = self.all_data['atom_types'][idx]
            pxrd = self.all_data['pxrd'][idx]  # fp16格式
            num_atoms = int(self.all_data['num_atoms'][idx])
            sample_id = str(self.all_data['ids'][idx])
            
            # 转换为tensor
            output = {
                'z': torch.from_numpy(z.copy()),
                'comp': torch.from_numpy(comp.copy()),
                'pxrd': torch.from_numpy(pxrd.astype(np.float32)),  # fp16 -> fp32
                'num_atoms': torch.tensor(num_atoms, dtype=torch.long),
                'atom_types': torch.from_numpy(atom_types.copy()),
                'id': sample_id
            }
        else:
            # 延迟加载模式（从文件读取）
            batch_idx, sample_idx = self.index_map[idx]
            batch_file = self.batch_files[batch_idx]
            
            with np.load(batch_file) as data:
                z = data['z'][sample_idx]
                comp = data['comp'][sample_idx]
                atom_types = data['atom_types'][sample_idx]
                pxrd = data['pxrd'][sample_idx]  # fp16格式
                num_atoms = int(data['num_atoms'][sample_idx])
                
                # 获取ID
                if 'ids' in data:
                    sample_id = str(data['ids'][sample_idx])
                else:
                    sample_id = self.ids[idx] if idx < len(self.ids) else str(idx)
            
            # 转换为tensor
            output = {
                'z': torch.from_numpy(z.copy()),
                'comp': torch.from_numpy(comp.copy()),
                'pxrd': torch.from_numpy(pxrd.astype(np.float32)),  # fp16 -> fp32
                'num_atoms': torch.tensor(num_atoms, dtype=torch.long),
                'atom_types': torch.from_numpy(atom_types.copy()),
                'id': sample_id
            }
        
        # 应用可选的变换
        if self.transform:
            output = self.transform(output)
        
        return output


def collate_crystal_batch(
    batch, 
    permutation_aug: Optional[PermutationAugmentation] = None,
    so3_aug: Optional[SO3Augmentation] = None
):
    """
    自定义批处理函数，包含数据增强
    
    Args:
        batch: 数据集返回的字典列表
        permutation_aug: 可选的置换增强
        so3_aug: 可选的SO3旋转增强
    
    Returns:
        批处理后的字典
    """
    # Stack所有tensor
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
    cache_in_memory: bool = False,
    drop_last: bool = False,
    **dataset_kwargs
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_path: npz数据集路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载进程数
        pin_memory: 是否固定内存（GPU训练时使用）
        augment_permutation: 是否应用置换增强
        augment_so3: 是否应用SO3旋转增强
        augment_prob: 置换增强概率
        so3_augment_prob: SO3增强概率
        cache_in_memory: 是否将数据缓存到内存
        drop_last: 是否丢弃最后一个不完整的批次
        **dataset_kwargs: 传递给CrystalDataset的额外参数
    
    Returns:
        DataLoader实例
    """
    dataset = CrystalDataset(
        data_path, 
        cache_in_memory=cache_in_memory,
        **dataset_kwargs
    )
    
    # 创建数据增强实例
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
        collate_fn=collate_fn,
        drop_last=drop_last  # 丢弃最后一个不完整的批次（避免批次大小不一致）
    )
    
    return dataloader


def split_dataset(
    dataset: CrystalDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    将数据集分割为训练/验证/测试集
    
    Args:
        dataset: CrystalDataset实例
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    
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