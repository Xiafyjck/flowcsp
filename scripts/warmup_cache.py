#!/usr/bin/env python
"""
Warmup缓存生成脚本
将CDVAE的CSV格式转换为内存映射的NumPy数组，支持大规模数据集和多进程访问
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from tqdm import tqdm
import argparse
from src.pxrd_simulator import PXRDSimulator
from multiprocessing import Pool, cpu_count
import warnings
import json
from pathlib import Path


def process_single_sample(row_data):
    """处理单个样本的函数（用于并行处理）"""
    idx, material_id, cif = row_data
    
    # 忽略pymatgen的CIF解析警告
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF")
        warnings.filterwarnings("ignore", category=UserWarning)
        
        try:
            # 解析CIF
            structure = Structure.from_str(cif, fmt="cif")
            
            # 获取Niggli结构
            niggli_structure = structure.get_reduced_structure()
            
            # 获取原子信息
            atom_types = np.array([site.specie.Z for site in niggli_structure], dtype=np.int32)
            num_atoms = len(atom_types)
            
            # 超过60个原子的样本直接跳过
            if num_atoms > 60:
                return None
            
            # 模拟PXRD
            pxrd_simulator = PXRDSimulator()
            _, pxrd = pxrd_simulator.simulate(structure)
            
            # 获取晶格和坐标
            lattice_matrix = niggli_structure.lattice.matrix.astype(np.float32)  # [3, 3]
            frac_coords = niggli_structure.frac_coords.astype(np.float32)  # [num_atoms, 3]
            
            # 准备输出数据，所有数据都padding到固定尺寸
            # z: [63, 3] - 前3行是晶格，后60行是原子坐标
            z = np.zeros((63, 3), dtype=np.float32)
            z[:3, :] = lattice_matrix
            if num_atoms > 0:
                z[3:3+num_atoms, :] = frac_coords
            
            # comp: [60] - 原子组成（原子序数）
            comp = np.zeros(60, dtype=np.float32)
            if num_atoms > 0:
                comp[:num_atoms] = atom_types.astype(np.float32)
            
            # atom_types: [60] - 原子类型（整数）
            atom_types_padded = np.zeros(60, dtype=np.int32)
            if num_atoms > 0:
                atom_types_padded[:num_atoms] = atom_types
            
            return {
                'idx': idx,
                'id': material_id,
                'z': z,
                'comp': comp,
                'atom_types': atom_types_padded,
                'pxrd': pxrd.astype(np.float32),
                'num_atoms': num_atoms
            }
            
        except Exception as e:
            print(f"\n处理 {material_id} 时出错: {e}")
            return None


def create_memmap_arrays(output_dir, n_samples):
    """创建内存映射数组文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建各个数组的内存映射文件
    arrays = {
        'z': np.memmap(output_dir / 'z.npy', dtype='float32', mode='w+', shape=(n_samples, 63, 3)),
        'comp': np.memmap(output_dir / 'comp.npy', dtype='float32', mode='w+', shape=(n_samples, 60)),
        'atom_types': np.memmap(output_dir / 'atom_types.npy', dtype='int32', mode='w+', shape=(n_samples, 60)),
        'pxrd': np.memmap(output_dir / 'pxrd.npy', dtype='float32', mode='w+', shape=(n_samples, 11501)),
        'num_atoms': np.memmap(output_dir / 'num_atoms.npy', dtype='int32', mode='w+', shape=(n_samples,))
    }
    
    return arrays


def main():
    parser = argparse.ArgumentParser(description='生成warmup内存映射缓存')
    parser.add_argument('--csv', type=str, 
                       default='/home/ma-user/work/mincycle4csp/data/full_mp/full_mp.csv',
                       help='输入CSV文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='/home/ma-user/work/mincycle4csp/data/full_mp_cache',
                       help='输出目录路径')
    parser.add_argument('--id_col', type=str, default='id',
                       help='ID列名')
    parser.add_argument('--cif_col', type=str, default='cif',
                       help='CIF列名')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理样本数（调试用）')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行工作进程数（默认为CPU核心数）')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='每批处理的样本数')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # 读取数据
    print(f"读取数据: {args.csv}")
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
        print(f"限制处理 {args.limit} 个样本")
    
    total_samples = len(df)
    print(f"总样本数: {total_samples}")
    
    # 准备并行处理的数据（添加索引）
    row_data = [(idx, row[args.id_col], row[args.cif_col]) for idx, (_, row) in enumerate(df.iterrows())]
    
    # 设置工作进程数
    n_workers = args.workers or cpu_count()
    print(f"使用 {n_workers} 个进程并行处理")
    
    # 第一遍：并行处理所有样本，收集有效样本
    print("\n第一步：处理样本并收集有效数据...")
    valid_samples = []
    ids_list = []
    
    batch_size = args.batch_size
    total_batches = (len(row_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(row_data))
        batch_data = row_data[start_idx:end_idx]
        
        print(f"\n处理批次 {batch_idx + 1}/{total_batches} (样本 {start_idx + 1}-{end_idx})")
        
        # 并行处理当前批次
        with Pool(n_workers) as pool:
            processed_batch = list(tqdm(
                pool.imap(process_single_sample, batch_data),
                total=len(batch_data),
                desc=f"批次 {batch_idx + 1}"
            ))
        
        # 收集有效样本
        for item in processed_batch:
            if item is not None:
                valid_samples.append(item)
                ids_list.append(item['id'])
    
    n_valid = len(valid_samples)
    n_filtered = total_samples - n_valid
    
    print(f"\n有效样本数: {n_valid}")
    print(f"过滤样本数（处理失败或原子数>60）: {n_filtered}")
    
    if n_valid == 0:
        print("没有有效样本，退出")
        return
    
    # 第二步：创建内存映射数组并写入数据
    print(f"\n第二步：创建内存映射数组...")
    arrays = create_memmap_arrays(output_dir, n_valid)
    
    print("写入数据到内存映射数组...")
    for i, sample in enumerate(tqdm(valid_samples, desc="写入数据")):
        arrays['z'][i] = sample['z']
        arrays['comp'][i] = sample['comp']
        arrays['atom_types'][i] = sample['atom_types']
        arrays['pxrd'][i] = sample['pxrd']
        arrays['num_atoms'][i] = sample['num_atoms']
    
    # 刷新到磁盘
    print("刷新数据到磁盘...")
    for array in arrays.values():
        array.flush()
    
    # 保存元数据
    metadata = {
        'n_samples': n_valid,
        'n_filtered': n_filtered,
        'max_atoms': 60,
        'pxrd_dim': 11501,
        'source_csv': args.csv,
        'ids': ids_list,
        'shapes': {
            'z': [n_valid, 63, 3],
            'comp': [n_valid, 60],
            'atom_types': [n_valid, 60],
            'pxrd': [n_valid, 11501],
            'num_atoms': [n_valid]
        },
        'dtypes': {
            'z': 'float32',
            'comp': 'float32',
            'atom_types': 'int32',
            'pxrd': 'float32',
            'num_atoms': 'int32'
        }
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 输出统计信息
    print(f"\n处理完成:")
    print(f"  有效样本数: {n_valid}")
    print(f"  过滤样本数: {n_filtered}")
    print(f"  数据目录: {output_dir}")
    print(f"  元数据文件: {metadata_path}")
    
    # 验证数据
    print(f"\n验证数据...")
    z_test = np.memmap(output_dir / 'z.npy', dtype='float32', mode='r', shape=(n_valid, 63, 3))
    print(f"  z数组形状: {z_test.shape}")
    print(f"  z数组前3行（晶格）非零元素数: {np.count_nonzero(z_test[0, :3, :])}")
    
    pxrd_test = np.memmap(output_dir / 'pxrd.npy', dtype='float32', mode='r', shape=(n_valid, 11501))
    print(f"  pxrd数组形状: {pxrd_test.shape}")
    print(f"  pxrd数组第一个样本非零元素数: {np.count_nonzero(pxrd_test[0])}")
    
    print("\n✅ 内存映射缓存创建成功！")
    print("\n使用方法：")
    print(f"  from src.data import CrystalDataset")
    print(f"  dataset = CrystalDataset('{output_dir}')")


if __name__ == "__main__":
    main()