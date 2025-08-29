"""
GNN数据处理模块 - 为图神经网络提供数据加载和图构建功能
支持全连接图构建，考虑周期性边界条件
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from typing import Dict, Optional, List, Tuple
import time
from pathlib import Path

from .data import CrystalDataset, SO3Augmentation, PermutationAugmentation


def fully_connected_graph_pbc(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    num_atoms: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构建全连接图，考虑周期性边界条件计算最短距离
    
    全连接图意味着每个原子都与所有其他原子连接，
    让网络自己学习哪些连接是重要的。
    
    Args:
        frac_coords: 分数坐标 [total_atoms, 3]
        lattice: 晶格参数 [batch_size, 3, 3]
        num_atoms: 每个样本的原子数 [batch_size]
        
    Returns:
        edge_index: 边索引 [2, num_edges] - 全连接
        edge_attr: 边属性（最短周期性距离）[num_edges, 1]
        cell_offsets: 最短距离对应的周期性偏移 [num_edges, 3]
    """
    batch_size = len(num_atoms)
    device = frac_coords.device
    
    # 构建批次索引
    batch_idx = torch.repeat_interleave(
        torch.arange(batch_size, device=device), num_atoms
    )
    
    # 生成27个邻近单元的偏移（3x3x3），用于计算最短周期性距离
    offsets = torch.tensor(
        [[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]],
        device=device, dtype=torch.float32
    )
    
    edge_index_list = []
    edge_attr_list = []
    cell_offset_list = []
    
    # 对每个批次分别处理
    for b in range(batch_size):
        # 获取当前批次的原子
        mask = batch_idx == b
        batch_frac = frac_coords[mask]  # [n_atoms, 3]
        n_atoms = batch_frac.shape[0]
        
        if n_atoms == 0:
            continue
            
        # 转换到笛卡尔坐标
        batch_lattice = lattice[b]  # [3, 3]
        batch_cart = batch_frac @ batch_lattice.T  # [n_atoms, 3]
        
        # 创建全连接图的边索引（除了自环）
        # 生成所有可能的原子对
        src = torch.arange(n_atoms, device=device).repeat_interleave(n_atoms)
        dst = torch.arange(n_atoms, device=device).repeat(n_atoms)
        
        # 移除自环
        mask_no_self = src != dst
        src = src[mask_no_self]
        dst = dst[mask_no_self]
        
        # 向量化计算每条边的最短周期性距离
        num_edges = src.shape[0]
        
        # 获取所有边的起点和终点坐标
        pos_src = batch_cart[src]  # [num_edges, 3]
        pos_dst = batch_cart[dst]  # [num_edges, 3]
        
        # 扩展维度以计算所有可能的周期性距离
        pos_src_expanded = pos_src.unsqueeze(1)  # [num_edges, 1, 3]
        pos_dst_expanded = pos_dst.unsqueeze(1)  # [num_edges, 1, 3]
        
        # 计算所有偏移向量
        offset_vecs = offsets @ batch_lattice.T  # [27, 3]
        offset_vecs_expanded = offset_vecs.unsqueeze(0)  # [1, 27, 3]
        
        # 计算所有可能的周期性位置
        shifted_pos_dst = pos_dst_expanded + offset_vecs_expanded  # [num_edges, 27, 3]
        
        # 计算所有距离
        diffs = pos_src_expanded - shifted_pos_dst  # [num_edges, 27, 3]
        distances = torch.norm(diffs, dim=2)  # [num_edges, 27]
        
        # 找到最短距离及其对应的偏移
        min_distances, min_indices = torch.min(distances, dim=1)  # [num_edges]
        min_offsets = offsets[min_indices]  # [num_edges, 3]
        
        # 调整索引到全局
        global_offset = mask.nonzero(as_tuple=False)[0, 0].item()
        edge_index = torch.stack([src + global_offset, dst + global_offset])
        
        edge_index_list.append(edge_index)
        edge_attr_list.append(min_distances.unsqueeze(1))  # [num_edges, 1]
        cell_offset_list.append(min_offsets)  # [num_edges, 3]
    
    # 合并所有批次的边
    if len(edge_index_list) > 0:
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
        cell_offsets = torch.cat(cell_offset_list, dim=0)
    else:
        # 如果没有边，返回空tensor
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32, device=device)
        cell_offsets = torch.zeros((0, 3), dtype=torch.float32, device=device)
    
    return edge_index, edge_attr, cell_offsets


class GNNCrystalDataset(CrystalDataset):
    """
    图神经网络晶体数据集
    继承自CrystalDataset，添加全连接图构建功能
    """
    
    def __init__(
        self,
        data_path: str,
        max_atoms: int = 60,
        pxrd_dim: int = 11501,
        build_graph_cache: bool = True,
        transform=None,
        cache_in_memory: bool = True
    ):
        """
        Args:
            data_path: npz缓存目录路径
            max_atoms: 最大原子数量
            pxrd_dim: PXRD维度
            build_graph_cache: 是否预构建并缓存图结构
            transform: 可选的数据变换
            cache_in_memory: 是否将所有数据缓存到内存
        """
        super().__init__(
            data_path=data_path,
            max_atoms=max_atoms,
            pxrd_dim=pxrd_dim,
            transform=transform,
            cache_in_memory=cache_in_memory
        )
        
        self.build_graph_cache = build_graph_cache
        
        # 预构建全连接图结构并缓存
        if self.build_graph_cache and self.cache_in_memory:
            print(f"Building fully connected graph cache...")
            start_time = time.time()
            self._build_graph_cache()
            print(f"Graph cache built in {time.time() - start_time:.2f} seconds")
    
    def _build_graph_cache(self):
        """
        预构建所有样本的全连接图结构并缓存
        """
        self.cached_graphs = []
        
        # 批量处理以提高效率
        batch_size = 100
        for i in range(0, self.n_samples, batch_size):
            batch_indices = range(i, min(i + batch_size, self.n_samples))
            
            # 收集批次数据
            batch_z = []
            batch_num_atoms = []
            
            for idx in batch_indices:
                z = self.all_data['z'][idx]
                num_atoms = int(self.all_data['num_atoms'][idx])
                batch_z.append(torch.from_numpy(z.copy()))
                batch_num_atoms.append(num_atoms)
            
            batch_z = torch.stack(batch_z)
            batch_num_atoms = torch.tensor(batch_num_atoms, dtype=torch.long)
            
            # 提取晶格和分数坐标
            lattice = batch_z[:, :3, :]  # [batch, 3, 3]
            frac_coords = batch_z[:, 3:, :]  # [batch, 60, 3]
            
            # 处理变长序列 - 展平并移除padding
            valid_frac_coords = []
            for b, n in enumerate(batch_num_atoms):
                valid_frac_coords.append(frac_coords[b, :n, :])
            
            if len(valid_frac_coords) > 0:
                flat_frac_coords = torch.cat(valid_frac_coords, dim=0)
                
                # 构建全连接图
                edge_index, edge_attr, cell_offsets = fully_connected_graph_pbc(
                    flat_frac_coords,
                    lattice,
                    batch_num_atoms
                )
                
                # 分割并存储每个样本的图
                node_ptr = torch.cat([
                    torch.tensor([0]), torch.cumsum(batch_num_atoms, dim=0)
                ])
                
                for j, idx in enumerate(batch_indices):
                    # 找到属于当前样本的边
                    start_node = node_ptr[j].item()
                    end_node = node_ptr[j + 1].item()
                    
                    # 筛选边
                    mask = (edge_index[0] >= start_node) & (edge_index[0] < end_node)
                    sample_edges = edge_index[:, mask] - start_node
                    sample_attr = edge_attr[mask]
                    sample_cells = cell_offsets[mask]
                    
                    self.cached_graphs.append({
                        'edge_index': sample_edges.numpy(),
                        'edge_attr': sample_attr.numpy(),
                        'cell_offsets': sample_cells.numpy()
                    })
            
            # 打印进度
            if (i + batch_size) % 1000 == 0:
                print(f"  Processed {min(i + batch_size, self.n_samples)}/{self.n_samples} samples...")
    
    def __getitem__(self, idx) -> Data:
        """
        返回PyTorch Geometric的Data对象
        
        Returns:
            Data对象，包含：
                - x: 节点特征（原子类型）[num_atoms]
                - pos: 分数坐标 [num_atoms, 3]
                - lattice: 晶格参数 [3, 3]
                - edge_index: 边索引 [2, num_edges]
                - edge_attr: 边属性 [num_edges, 1]
                - cell_offsets: 周期性偏移 [num_edges, 3]
                - comp: 完整原子组成 [60]
                - pxrd: PXRD谱图 [11501]
                - num_atoms: 实际原子数量
                - z: 原始z数据 [63, 3]（用于兼容性）
        """
        # 获取基础数据
        base_data = super().__getitem__(idx)
        
        # 提取数据
        z = base_data['z']
        comp = base_data['comp']
        pxrd = base_data['pxrd']
        num_atoms = base_data['num_atoms'].item()
        atom_types = base_data['atom_types']
        
        # 分离晶格和分数坐标
        lattice = z[:3, :]  # [3, 3]
        frac_coords = z[3:3+num_atoms, :]  # [num_atoms, 3]
        
        # 获取或构建图
        if self.build_graph_cache and hasattr(self, 'cached_graphs'):
            # 使用缓存的图
            graph = self.cached_graphs[idx]
            edge_index = torch.from_numpy(graph['edge_index'])
            edge_attr = torch.from_numpy(graph['edge_attr'])
            cell_offsets = torch.from_numpy(graph['cell_offsets'])
        else:
            # 动态构建全连接图
            edge_index, edge_attr, cell_offsets = fully_connected_graph_pbc(
                frac_coords,  # 不需要添加batch维度，已经是[num_atoms, 3]
                lattice.unsqueeze(0),  # 添加batch维度 [1, 3, 3]
                torch.tensor([num_atoms])
            )
        
        # 创建PyG Data对象
        data = Data(
            x=atom_types[:num_atoms],  # 节点特征（原子类型）
            pos=frac_coords,  # 分数坐标
            lattice=lattice,  # 晶格参数
            edge_index=edge_index.long(),  # 边索引
            edge_attr=edge_attr,  # 边属性（距离）
            cell_offsets=cell_offsets,  # 周期性偏移
            comp=comp,  # 完整原子组成（包含padding）
            pxrd=pxrd,  # PXRD谱图
            num_atoms=num_atoms,  # 实际原子数
            num_nodes=num_atoms,  # PyG批处理需要
            z=z,  # 保留原始z用于兼容性
            id=base_data['id']  # 样本ID
        )
        
        return data


def collate_gnn_batch(
    batch: List[Data],
    permutation_aug: Optional[PermutationAugmentation] = None,
    so3_aug: Optional[SO3Augmentation] = None
) -> Dict[str, torch.Tensor]:
    """
    自定义批处理函数，将PyG Data对象转换为统一格式
    
    Args:
        batch: PyG Data对象列表
        permutation_aug: 可选的置换增强
        so3_aug: 可选的SO3增强
        
    Returns:
        批处理后的字典，包含：
            - z: [batch_size, 63, 3]
            - comp: [batch_size, 60]
            - pxrd: [batch_size, 11501]
            - num_atoms: [batch_size]
            - pyg_batch: PyG Batch对象（用于图神经网络）
    """
    # 使用PyG的批处理
    pyg_batch = Batch.from_data_list(batch)
    
    # 提取统一格式的数据
    batch_size = len(batch)
    z_list = [data.z for data in batch]
    comp_list = [data.comp for data in batch]
    pxrd_list = [data.pxrd for data in batch]
    num_atoms_list = [torch.tensor(data.num_atoms) for data in batch]
    atom_types_list = [data.x for data in batch]
    ids = [data.id for data in batch]
    
    # Stack成批次
    z = torch.stack(z_list)
    comp = torch.stack(comp_list)
    pxrd = torch.stack(pxrd_list)
    num_atoms = torch.stack(num_atoms_list)
    
    # 处理原子类型（变长，需要padding）
    atom_types = torch.zeros(batch_size, 60, dtype=torch.long)
    for i, types in enumerate(atom_types_list):
        atom_types[i, :len(types)] = types
    
    # 构建输出字典
    collated_batch = {
        'z': z,
        'comp': comp,
        'pxrd': pxrd,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'ids': ids,
        'pyg_batch': pyg_batch  # 保留PyG批次对象供GNN使用
    }
    
    # 应用数据增强
    if so3_aug is not None:
        collated_batch = so3_aug(collated_batch)
        # 同步更新pyg_batch中的晶格
        pyg_batch.lattice = collated_batch['z'][:, :3, :].reshape(-1, 3, 3)
    
    if permutation_aug is not None:
        collated_batch = permutation_aug(collated_batch)
        # 注意：置换增强可能需要更新图结构
    
    return collated_batch


def get_gnn_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    augment_permutation: bool = False,
    augment_so3: bool = False,
    augment_prob: float = 0.5,
    so3_augment_prob: float = 0.5,
    cache_in_memory: bool = True,
    drop_last: bool = False,
    build_graph_cache: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    创建GNN数据加载器（全连接图）
    
    Args:
        data_path: npz数据集路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载进程数
        pin_memory: 是否固定内存
        augment_permutation: 是否应用置换增强
        augment_so3: 是否应用SO3增强
        augment_prob: 置换增强概率
        so3_augment_prob: SO3增强概率
        cache_in_memory: 是否缓存数据到内存
        drop_last: 是否丢弃最后批次
        build_graph_cache: 是否预构建图缓存
        **dataset_kwargs: 其他数据集参数
        
    Returns:
        DataLoader实例
    """
    dataset = GNNCrystalDataset(
        data_path,
        cache_in_memory=cache_in_memory,
        build_graph_cache=build_graph_cache,
        **dataset_kwargs
    )
    
    # 创建数据增强实例
    permutation_aug = PermutationAugmentation(augment_prob=augment_prob) if augment_permutation else None
    so3_aug = SO3Augmentation(augment_prob=so3_augment_prob) if augment_so3 else None
    
    # 创建collate函数
    def collate_fn(batch):
        return collate_gnn_batch(batch, permutation_aug, so3_aug)
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn,
        drop_last=drop_last
    )
    
    return dataloader