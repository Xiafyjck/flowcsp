#!/usr/bin/env python
"""
Warmup缓存生成脚本
将CSV格式转换为npz格式，使用fp16存储PXRD以节省内存
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
from pathlib import Path


def process_single_sample(args):
    """处理单个样本的函数（用于并行处理）"""
    idx, material_id, cif = args
    
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
            
            # 模拟PXRD - 使用fp16存储以节省内存
            pxrd_simulator = PXRDSimulator()
            _, pxrd = pxrd_simulator.simulate(structure)
            pxrd = pxrd.astype(np.float16)  # 转换为fp16
            
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
                'pxrd': pxrd,  # 已经是fp16
                'num_atoms': num_atoms
            }
            
        except Exception as e:
            print(f"\n处理 {material_id} 时出错: {e}")
            return None


def save_batch_to_npz(batch_data, output_dir, batch_start_idx):
    """将一批数据保存为npz文件"""
    output_dir = Path(output_dir)
    
    # 收集有效样本
    valid_samples = [s for s in batch_data if s is not None]
    if not valid_samples:
        return 0
    
    # 分别收集每个数组
    z_list = [s['z'] for s in valid_samples]
    comp_list = [s['comp'] for s in valid_samples]
    atom_types_list = [s['atom_types'] for s in valid_samples]
    pxrd_list = [s['pxrd'] for s in valid_samples]
    num_atoms_list = [s['num_atoms'] for s in valid_samples]
    ids_list = [s['id'] for s in valid_samples]
    
    # 堆叠成批次数组
    z_batch = np.stack(z_list)
    comp_batch = np.stack(comp_list)
    atom_types_batch = np.stack(atom_types_list)
    pxrd_batch = np.stack(pxrd_list)  # fp16格式
    num_atoms_batch = np.array(num_atoms_list, dtype=np.int32)
    
    # 保存为单个npz文件（每批一个文件）
    batch_file = output_dir / f'batch_{batch_start_idx:06d}.npz'
    np.savez_compressed(
        batch_file,
        z=z_batch,
        comp=comp_batch,
        atom_types=atom_types_batch,
        pxrd=pxrd_batch,  # fp16格式，节省50%空间
        num_atoms=num_atoms_batch,
        ids=ids_list
    )
    
    return len(valid_samples)


def main():
    parser = argparse.ArgumentParser(description='生成warmup npz缓存')
    parser.add_argument('--csv', type=str, 
                       default='/home/ma-user/work/mincycle4csp/data/merged_cdvae/test.csv',
                       help='输入CSV文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='/home/ma-user/work/mincycle4csp/data/test_cache',
                       help='输出目录路径')
    parser.add_argument('--id_col', type=str, default='material_id',
                       help='ID列名')
    parser.add_argument('--cif_col', type=str, default='cif',
                       help='CIF列名')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理样本数（调试用）')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行工作进程数（默认为CPU核心数）')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='每个npz文件包含的样本数')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取数据
    print(f"读取数据: {args.csv}")
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
        print(f"限制处理 {args.limit} 个样本")
    
    total_samples = len(df)
    print(f"总样本数: {total_samples}")
    
    # 准备并行处理的数据（添加索引）
    row_data = [(idx, row[args.id_col], row[args.cif_col]) 
                for idx, (_, row) in enumerate(df.iterrows())]
    
    # 设置工作进程数
    n_workers = args.workers or cpu_count()
    print(f"使用 {n_workers} 个进程并行处理")
    
    # 分批处理
    batch_size = args.batch_size
    total_batches = (len(row_data) + batch_size - 1) // batch_size
    
    success_count = 0
    fail_count = 0
    all_ids = []
    
    print(f"\n开始处理（每批 {batch_size} 个样本，共 {total_batches} 批）...")
    
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
        
        # 保存批次数据
        n_valid = save_batch_to_npz(processed_batch, output_dir, batch_idx)
        
        # 统计
        success_count += n_valid
        fail_count += len(batch_data) - n_valid
        
        # 收集有效ID
        valid_samples = [s for s in processed_batch if s is not None]
        all_ids.extend([s['id'] for s in valid_samples])
        
        print(f"  批次完成：成功 {n_valid}，失败 {len(batch_data) - n_valid}")
    
    # 保存元数据
    metadata = {
        'format': 'npz_batched',
        'n_samples': success_count,
        'n_batches': total_batches,
        'batch_size': batch_size,
        'n_filtered': fail_count,
        'max_atoms': 60,
        'pxrd_dim': 11501,
        'pxrd_dtype': 'float16',  # 标记使用fp16
        'source_csv': args.csv,
        'ids': all_ids,
        'data_info': {
            'z_shape': [63, 3],
            'comp_shape': [60],
            'atom_types_shape': [60],
            'pxrd_shape': [11501],
            'z_dtype': 'float32',
            'comp_dtype': 'float32',
            'atom_types_dtype': 'int32',
            'pxrd_dtype': 'float16',  # fp16节省内存
            'num_atoms_dtype': 'int32'
        }
    }
    
    import json
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 计算内存占用
    fp32_size = success_count * 11501 * 4 / (1024**3)  # 如果用fp32
    fp16_size = success_count * 11501 * 2 / (1024**3)  # 实际用fp16
    saved_gb = fp32_size - fp16_size
    saved_percent = (saved_gb / fp32_size) * 100
    
    # 输出统计信息
    print(f"\n处理完成:")
    print(f"  成功样本数: {success_count}")
    print(f"  失败样本数: {fail_count}")
    print(f"  生成批次文件: {total_batches} 个")
    print(f"  数据目录: {output_dir}")
    print(f"  元数据文件: {metadata_path}")
    print(f"\n内存优化:")
    print(f"  PXRD使用fp16格式")
    print(f"  理论内存占用: {fp32_size:.2f} GB (fp32)")
    print(f"  实际内存占用: {fp16_size:.2f} GB (fp16)")
    print(f"  节省内存: {saved_gb:.2f} GB ({saved_percent:.1f}%)")
    
    print("\n✅ npz缓存创建成功！")
    print("\n使用方法：")
    print(f"  from src.data import CrystalDataset")
    print(f"  dataset = CrystalDataset('{output_dir}')")


if __name__ == "__main__":
    main()