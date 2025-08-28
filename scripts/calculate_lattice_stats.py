#!/usr/bin/env python
"""
计算晶格参数统计信息脚本
从多个cache目录读取内存映射数据，计算晶格参数的均值和标准差
用于生成归一化所需的统计信息
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple


def load_cache_metadata(cache_dir: Path) -> Dict:
    """
    加载cache目录的元数据
    
    Args:
        cache_dir: cache目录路径
        
    Returns:
        元数据字典
    """
    metadata_path = cache_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def compute_lattice_stats_incremental(cache_dirs: List[Path], 
                                     batch_size: int = 1000) -> Dict:
    """
    增量计算晶格参数的统计信息
    使用Welford算法在线计算均值和方差，避免内存溢出
    
    Args:
        cache_dirs: cache目录列表
        batch_size: 每批处理的样本数
        
    Returns:
        包含统计信息的字典
    """
    # 初始化统计变量
    n_total = 0  # 总样本数
    
    # 晶格参数是3x3矩阵，展平成9个元素
    lattice_sum = np.zeros(9, dtype=np.float64)  # 使用float64避免精度损失
    lattice_sum_sq = np.zeros(9, dtype=np.float64)
    
    # 分数坐标统计（3个维度）
    frac_sum = np.zeros(3, dtype=np.float64)
    frac_sum_sq = np.zeros(3, dtype=np.float64)
    frac_count = 0  # 有效原子总数
    
    # 遍历所有cache目录
    for cache_dir in cache_dirs:
        print(f"\n处理cache目录: {cache_dir}")
        
        # 加载元数据
        metadata = load_cache_metadata(cache_dir)
        n_samples = metadata['n_samples']
        
        print(f"  样本数: {n_samples}")
        
        # 打开内存映射数组（只读模式）
        z_shape = tuple(metadata['shapes']['z'])
        z_array = np.memmap(
            cache_dir / 'z.npy',
            dtype='float32',
            mode='r',
            shape=z_shape
        )
        
        num_atoms_shape = tuple(metadata['shapes']['num_atoms'])
        num_atoms_array = np.memmap(
            cache_dir / 'num_atoms.npy',
            dtype='int32',
            mode='r',
            shape=num_atoms_shape
        )
        
        # 分批处理数据
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(n_batches), desc="  计算统计"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            # 读取当前批次的数据
            z_batch = z_array[start_idx:end_idx]  # [batch_size, 63, 3]
            num_atoms_batch = num_atoms_array[start_idx:end_idx]  # [batch_size]
            
            # 提取晶格参数（前3行）
            lattice_batch = z_batch[:, :3, :]  # [batch_size, 3, 3]
            lattice_flat = lattice_batch.reshape(-1, 9)  # [batch_size, 9]
            
            # 更新晶格参数统计
            batch_count = len(lattice_flat)
            lattice_sum += np.sum(lattice_flat, axis=0)
            lattice_sum_sq += np.sum(lattice_flat ** 2, axis=0)
            n_total += batch_count
            
            # 提取分数坐标并计算统计（只统计有效原子）
            for i in range(len(z_batch)):
                n_atoms = num_atoms_batch[i]
                if n_atoms > 0:
                    # 获取该样本的有效分数坐标
                    frac_coords = z_batch[i, 3:3+n_atoms, :]  # [n_atoms, 3]
                    
                    # 更新分数坐标统计
                    frac_sum += np.sum(frac_coords, axis=0)
                    frac_sum_sq += np.sum(frac_coords ** 2, axis=0)
                    frac_count += n_atoms
    
    print(f"\n计算最终统计信息...")
    
    # 计算晶格参数的均值和标准差
    lattice_mean = lattice_sum / n_total  # [9]
    lattice_var = (lattice_sum_sq / n_total) - (lattice_mean ** 2)
    lattice_std = np.sqrt(np.maximum(lattice_var, 0))  # 避免负方差
    
    # 计算全局均值和标准差（用于整体归一化）
    lattice_global_mean = np.mean(lattice_mean)
    lattice_global_std = np.mean(lattice_std)
    
    # 计算分数坐标的均值和标准差
    if frac_count > 0:
        frac_mean = frac_sum / frac_count
        frac_var = (frac_sum_sq / frac_count) - (frac_mean ** 2)
        frac_std = np.sqrt(np.maximum(frac_var, 0))
    else:
        frac_mean = np.zeros(3)
        frac_std = np.ones(3)
    
    # 构建输出字典（与模板格式一致）
    stats = {
        "lattice_global_mean": float(lattice_global_mean),
        "lattice_global_std": float(lattice_global_std),
        "lattice_mean": lattice_mean.tolist(),
        "lattice_std": lattice_std.tolist(),
        "frac_coords_mean": frac_mean.tolist(),
        "frac_coords_std": frac_std.tolist(),
        "num_samples_used": int(n_total),
        "total_samples": int(n_total),
        "cache_dirs": [str(d) for d in cache_dirs]
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='计算晶格参数统计信息')
    parser.add_argument('--cache_dirs', type=str, nargs='+', 
                       required=True,
                       help='cache目录路径列表')
    parser.add_argument('--output', type=str,
                       default='dataset_lattice_stats.json',
                       help='输出JSON文件路径')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='批处理大小（用于控制内存使用）')
    args = parser.parse_args()
    
    # 转换为Path对象并验证目录存在
    cache_dirs = []
    for dir_str in args.cache_dirs:
        cache_dir = Path(dir_str)
        if not cache_dir.exists():
            print(f"警告: 目录不存在 {cache_dir}")
            continue
        if not (cache_dir / 'metadata.json').exists():
            print(f"警告: 目录缺少metadata.json {cache_dir}")
            continue
        cache_dirs.append(cache_dir)
    
    if not cache_dirs:
        print("错误: 没有有效的cache目录")
        return
    
    print(f"将处理 {len(cache_dirs)} 个cache目录:")
    for cache_dir in cache_dirs:
        print(f"  - {cache_dir}")
    
    # 计算统计信息
    stats = compute_lattice_stats_incremental(cache_dirs, batch_size=args.batch_size)
    
    # 保存到JSON文件
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 打印统计摘要
    print(f"\n统计信息已保存到: {output_path}")
    print(f"\n统计摘要:")
    print(f"  处理样本总数: {stats['num_samples_used']}")
    print(f"  晶格参数全局均值: {stats['lattice_global_mean']:.4f}")
    print(f"  晶格参数全局标准差: {stats['lattice_global_std']:.4f}")
    print(f"  晶格参数均值范围: [{min(stats['lattice_mean']):.4f}, {max(stats['lattice_mean']):.4f}]")
    print(f"  晶格参数标准差范围: [{min(stats['lattice_std']):.4f}, {max(stats['lattice_std']):.4f}]")
    print(f"  分数坐标均值: {[f'{v:.4f}' for v in stats['frac_coords_mean']]}")
    print(f"  分数坐标标准差: {[f'{v:.4f}' for v in stats['frac_coords_std']]}")


if __name__ == "__main__":
    main()