#!/usr/bin/env python
"""
合并CDVAE的所有数据集并重新分割
将mp_20, carbon_24, perov_5的所有数据合并，按8:1:1比例重新分割
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

def load_dataset(dataset_path):
    """加载一个数据集的所有CSV文件"""
    train_csv = dataset_path / 'train.csv'
    val_csv = dataset_path / 'val.csv'
    test_csv = dataset_path / 'test.csv'
    
    dfs = []
    for csv_file in [train_csv, val_csv, test_csv]:
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            # 添加数据集来源标记
            df['source_dataset'] = dataset_path.name
            df['source_split'] = csv_file.stem  # train/val/test
            dfs.append(df)
            print(f"  加载 {csv_file.name}: {len(df)} 样本")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def main():
    parser = argparse.ArgumentParser(description='合并CDVAE数据集')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/ma-user/work/mincycle4csp/docs/repo/cdvae/data',
                       help='CDVAE数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/ma-user/work/mincycle4csp/data',
                       help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--sample_limit', type=int, default=None,
                       help='限制总样本数（用于测试）')
    parser.add_argument('--datasets', nargs='+', 
                       default=['mp_20', 'carbon_24', 'perov_5'],
                       help='要合并的数据集')
    args = parser.parse_args()
    
    # 验证比例
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "训练、验证、测试比例之和必须为1"
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载所有数据集
    print("="*60)
    print("加载数据集...")
    print("="*60)
    
    all_dfs = []
    dataset_stats = {}
    
    for dataset_name in args.datasets:
        dataset_path = Path(args.data_dir) / dataset_name
        if not dataset_path.exists():
            print(f"警告: 数据集 {dataset_name} 不存在，跳过")
            continue
        
        print(f"\n加载 {dataset_name}:")
        df = load_dataset(dataset_path)
        if df is not None:
            all_dfs.append(df)
            dataset_stats[dataset_name] = len(df)
            
            # 打印列信息
            print(f"  总计: {len(df)} 样本")
            print(f"  列: {df.columns.tolist()}")
            
            # 检查是否有CIF数据
            if 'cif' in df.columns:
                # 检查CIF非空
                non_empty_cif = df['cif'].notna().sum()
                print(f"  有效CIF: {non_empty_cif}/{len(df)}")
    
    if not all_dfs:
        print("错误: 没有加载到任何数据")
        return
    
    # 合并所有数据
    print("\n" + "="*60)
    print("合并数据...")
    print("="*60)
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # 打乱数据
    merged_df = merged_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    # 如果有样本限制
    if args.sample_limit:
        merged_df = merged_df.head(args.sample_limit)
        print(f"限制样本数到: {args.sample_limit}")
    
    total_samples = len(merged_df)
    print(f"合并后总样本数: {total_samples}")
    
    # 统计信息
    print("\n数据来源分布:")
    for dataset_name in dataset_stats:
        count = (merged_df['source_dataset'] == dataset_name).sum()
        print(f"  {dataset_name}: {count} ({count/total_samples*100:.1f}%)")
    
    # 分割数据
    print("\n" + "="*60)
    print(f"分割数据 (比例 {args.train_ratio}:{args.val_ratio}:{args.test_ratio})...")
    print("="*60)
    
    # 首先分出训练集
    train_size = int(total_samples * args.train_ratio)
    val_size = int(total_samples * args.val_ratio)
    test_size = total_samples - train_size - val_size
    
    # 使用索引进行分割
    indices = np.arange(total_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    train_df = merged_df.iloc[train_idx].reset_index(drop=True)
    val_df = merged_df.iloc[val_idx].reset_index(drop=True)
    test_df = merged_df.iloc[test_idx].reset_index(drop=True)
    
    print(f"训练集: {len(train_df)} 样本")
    print(f"验证集: {len(val_df)} 样本")
    print(f"测试集: {len(test_df)} 样本")
    
    # 保存分割后的数据
    print("\n" + "="*60)
    print("保存数据...")
    print("="*60)
    
    # 创建merged子目录
    merged_dir = output_dir / 'merged_cdvae'
    merged_dir.mkdir(exist_ok=True)
    
    # 保存CSV文件
    train_path = merged_dir / 'train.csv'
    val_path = merged_dir / 'val.csv'
    test_path = merged_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"训练集保存到: {train_path}")
    print(f"验证集保存到: {val_path}")
    print(f"测试集保存到: {test_path}")
    
    # 保存统计信息
    stats = {
        'total_samples': total_samples,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'datasets': dataset_stats,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'seed': args.seed
    }
    
    stats_path = merged_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n统计信息保存到: {stats_path}")
    
    # 打印数据集摘要
    print("\n" + "="*60)
    print("数据集摘要")
    print("="*60)
    
    # 检查材料ID分布
    if 'material_id' in merged_df.columns:
        unique_materials = merged_df['material_id'].nunique()
        print(f"独特材料数: {unique_materials}")
        
        # 检查是否有重复
        duplicates = merged_df['material_id'].duplicated().sum()
        if duplicates > 0:
            print(f"警告: 发现 {duplicates} 个重复的material_id")
    
    # 检查各数据集在训练/验证/测试中的分布
    print("\n各数据集在分割中的分布:")
    for split_name, split_df in [('训练', train_df), ('验证', val_df), ('测试', test_df)]:
        print(f"\n{split_name}集:")
        for dataset_name in dataset_stats:
            count = (split_df['source_dataset'] == dataset_name).sum()
            if len(split_df) > 0:
                print(f"  {dataset_name}: {count} ({count/len(split_df)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✓ 数据合并完成！")
    print("="*60)
    
    # 生成预处理命令
    print("\n下一步：运行预处理脚本")
    print("建议命令：")
    print(f"python scripts/preprocess_mp20.py --csv {train_path} --output data/merged_cdvae_train.pkl")
    print(f"python scripts/preprocess_mp20.py --csv {val_path} --output data/merged_cdvae_val.pkl")
    print(f"python scripts/preprocess_mp20.py --csv {test_path} --output data/merged_cdvae_test.pkl")

if __name__ == "__main__":
    main()