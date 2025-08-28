#!/usr/bin/env python
"""
MP20数据集预处理脚本
将CDVAE的CSV格式转换为训练所需的格式
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm
import pickle
import argparse
from src.pxrd_simulator import PXRDSimulator
from multiprocessing import Pool, cpu_count
import warnings


def process_single_sample(row_data):
    """处理单个样本的函数（用于并行处理）"""
    material_id, cif = row_data
    
    # 忽略pymatgen的CIF解析警告
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF")
        warnings.filterwarnings("ignore", category=UserWarning)
        
        try:
            # 解析CIF
            structure = Structure.from_str(cif, fmt="cif")
            
            # 获取Niggli和primitive结构
            # primitive_structure = structure.get_primitive_structure()
            niggli_structure = structure.get_reduced_structure()
            
            # 模拟PXRD
            pxrd_simulator = PXRDSimulator()
            _, pxrd = pxrd_simulator.simulate(structure)
            
            # 获取原子信息
            atom_types = np.array([site.specie.Z for site in structure], dtype=np.int32)
            num_atoms = len(atom_types)
            
            # 获取晶格和坐标
            lattice_matrix = structure.lattice.matrix  # [3, 3]
            frac_coords = structure.frac_coords  # [num_atoms, 3]
            
            # 获取成分字符串
            def get_comp_str(struct):
                comp = struct.composition.get_el_amt_dict()
                return " ".join([f"{el}{int(amt) if amt == int(amt) else amt}" 
                               for el, amt in sorted(comp.items())])
            
            return {
                'id': material_id,
                'structure': structure,
                'niggli_structure': niggli_structure,
                # 'primitive_structure': primitive_structure,
                'pxrd': pxrd.astype(np.float32),
                'niggli_comp': get_comp_str(niggli_structure),
                # 'primitive_comp': get_comp_str(primitive_structure),
                'atom_types': atom_types,
                'num_atoms': num_atoms,
                'lattice_matrix': lattice_matrix.astype(np.float32),
                'frac_coords': frac_coords.astype(np.float32)
            }
        except Exception as e:
            print(f"\n处理 {material_id} 时出错: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='预处理数据集')
    parser.add_argument('--csv', type=str, 
                       default='/home/ma-user/work/mincycle4csp/data/full_mp/full_mp.csv',
                       help='CSV文件路径')
    parser.add_argument('--output', type=str,
                       default='/home/ma-user/work/mincycle4csp/full_mp.pkl',
                       help='输出pickle文件路径')
    parser.add_argument('--id_col', type=str, default='extra_id',
                       help='ID列名')
    parser.add_argument('--cif_col', type=str, default='cif',
                       help='CIF列名')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理样本数（调试用）')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行工作进程数（默认为CPU核心数）')
    args = parser.parse_args()
    
    # 读取数据
    print(f"读取数据: {args.csv}")
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
        print(f"限制处理 {args.limit} 个样本")
    
    print(f"总样本数: {len(df)}")
    
    # 准备并行处理的数据
    row_data = [(row[args.id_col], row[args.cif_col]) for _, row in df.iterrows()]
    
    # 设置工作进程数
    n_workers = args.workers or cpu_count()
    print(f"使用 {n_workers} 个进程并行处理")
    
    # 并行处理
    with Pool(n_workers) as pool:
        # 使用tqdm显示进度
        processed_data = list(tqdm(
            pool.imap(process_single_sample, row_data),
            total=len(row_data),
            desc="处理样本"
        ))
    
    # 过滤掉处理失败的样本
    processed_data = [item for item in processed_data if item is not None]
    
    # 创建DataFrame
    result_df = pd.DataFrame(processed_data)
    
    # 简单统计
    max_atoms = result_df['num_atoms'].max() if len(result_df) > 0 else 0
    print(f"\n处理完成:")
    print(f"  成功样本数: {len(result_df)}")
    print(f"  失败样本数: {len(df) - len(result_df)}")
    print(f"  最大原子数: {max_atoms}")
    
    # 保存
    with open(args.output, 'wb') as f:
        pickle.dump(result_df, f)
    print(f"\n数据已保存到: {args.output}")


if __name__ == "__main__":
    main()