#!/usr/bin/env python
"""
CSV文件合并去重脚本
合并指定文件夹中的所有CSV文件，根据ID列去重，只保留ID和CIF两列
"""

import argparse
import pandas as pd
from pathlib import Path


def merge_and_deduplicate(
    folder_path: str,
    input_id_col: str,
    input_cif_col: str,
    output_id_col: str,
    output_cif_col: str,
    output_path: str,
    split_ratio: str = "9:1:0"
):
    """
    合并CSV文件并去重，支持train/val/test分割
    
    Args:
        folder_path: 包含CSV文件的文件夹路径
        input_id_col: 输入CSV中的ID列名
        input_cif_col: 输入CSV中的CIF列名
        output_id_col: 输出CSV中的ID列名
        output_cif_col: 输出CSV中的CIF列名
        output_path: 输出文件路径
        split_ratio: train:val:test的比例，默认9:1:0
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    # 找到所有CSV文件
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"文件夹中没有CSV文件: {folder_path}")
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 读取并合并所有CSV
    all_data = []
    for csv_file in csv_files:
        try:
            # 只读取需要的两列
            df = pd.read_csv(csv_file, usecols=[input_id_col, input_cif_col])
            all_data.append(df)
            print(f"  读取 {csv_file.name}: {len(df)} 行")
        except Exception as e:
            print(f"  跳过 {csv_file.name}: {e}")
    
    if not all_data:
        raise ValueError("没有成功读取任何CSV文件")
    
    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"\n合并后总行数: {len(merged_df)}")
    
    # 根据ID列去重，保留第一次出现的记录
    merged_df = merged_df.drop_duplicates(subset=[input_id_col], keep='first')
    print(f"去重后总行数: {len(merged_df)}")
    
    # 重命名列
    merged_df = merged_df.rename(columns={
        input_id_col: output_id_col,
        input_cif_col: output_cif_col
    })
    
    # 只保留需要的两列
    merged_df = merged_df[[output_id_col, output_cif_col]]
    
    # 解析分割比例
    ratios = [float(x) for x in split_ratio.split(':')]
    if len(ratios) != 3:
        raise ValueError(f"分割比例必须是3个数字，格式为train:val:test，当前为: {split_ratio}")
    
    total_ratio = sum(ratios)
    if total_ratio == 0:
        raise ValueError("分割比例总和不能为0")
    
    # 归一化比例
    train_ratio = ratios[0] / total_ratio
    val_ratio = ratios[1] / total_ratio
    test_ratio = ratios[2] / total_ratio
    
    # 打乱数据以确保随机性
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_total = len(merged_df)
    
    # 计算分割点
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 分割数据
    train_df = merged_df.iloc[:n_train]
    val_df = merged_df.iloc[n_train:n_train + n_val]
    test_df = merged_df.iloc[n_train + n_val:]
    
    # 准备输出路径
    output = Path(output_path)
    output_dir = output.parent
    output_stem = output.stem
    output_suffix = output.suffix
    
    # 保存结果
    if len(train_df) > 0:
        train_path = output_dir / f"{output_stem}_train{output_suffix}"
        train_df.to_csv(train_path, index=False)
        print(f"\nTrain集已保存至: {train_path}")
        print(f"  Train集行数: {len(train_df)} ({len(train_df)/n_total*100:.1f}%)")
    
    if len(val_df) > 0:
        val_path = output_dir / f"{output_stem}_val{output_suffix}"
        val_df.to_csv(val_path, index=False)
        print(f"Val集已保存至: {val_path}")
        print(f"  Val集行数: {len(val_df)} ({len(val_df)/n_total*100:.1f}%)")
    
    if len(test_df) > 0:
        test_path = output_dir / f"{output_stem}_test{output_suffix}"
        test_df.to_csv(test_path, index=False)
        print(f"Test集已保存至: {test_path}")
        print(f"  Test集行数: {len(test_df)} ({len(test_df)/n_total*100:.1f}%)")
    
    print(f"\n最终总行数: {len(merged_df)}")


def main():
    parser = argparse.ArgumentParser(description="合并CSV文件并去重，支持train/val/test分割")
    parser.add_argument("folder", help="包含CSV文件的文件夹路径")
    parser.add_argument("--input-id", required=True, help="输入CSV的ID列名")
    parser.add_argument("--input-cif", required=True, help="输入CSV的CIF列名")
    parser.add_argument("--output-id", required=True, help="输出CSV的ID列名")
    parser.add_argument("--output-cif", required=True, help="输出CSV的CIF列名")
    parser.add_argument("--output", required=True, help="输出文件路径（将自动添加_train/_val/_test后缀）")
    parser.add_argument("--split", default="9:1:0", help="train:val:test分割比例，默认9:1:0（不分割）")
    
    args = parser.parse_args()
    
    merge_and_deduplicate(
        folder_path=args.folder,
        input_id_col=args.input_id,
        input_cif_col=args.input_cif,
        output_id_col=args.output_id,
        output_cif_col=args.output_cif,
        output_path=args.output,
        split_ratio=args.split
    )


if __name__ == "__main__":
    main()