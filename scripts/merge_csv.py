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
    output_path: str
):
    """
    合并CSV文件并去重
    
    Args:
        folder_path: 包含CSV文件的文件夹路径
        input_id_col: 输入CSV中的ID列名
        input_cif_col: 输入CSV中的CIF列名
        output_id_col: 输出CSV中的ID列名
        output_cif_col: 输出CSV中的CIF列名
        output_path: 输出文件路径
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
    
    # 保存结果
    merged_df.to_csv(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")
    print(f"最终行数: {len(merged_df)}")


def main():
    parser = argparse.ArgumentParser(description="合并CSV文件并去重")
    parser.add_argument("folder", help="包含CSV文件的文件夹路径")
    parser.add_argument("--input-id", required=True, help="输入CSV的ID列名")
    parser.add_argument("--input-cif", required=True, help="输入CSV的CIF列名")
    parser.add_argument("--output-id", required=True, help="输出CSV的ID列名")
    parser.add_argument("--output-cif", required=True, help="输出CSV的CIF列名")
    parser.add_argument("--output", required=True, help="输出文件路径")
    
    args = parser.parse_args()
    
    merge_and_deduplicate(
        folder_path=args.folder,
        input_id_col=args.input_id,
        input_cif_col=args.input_cif,
        output_id_col=args.output_id,
        output_cif_col=args.output_cif,
        output_path=args.output
    )


if __name__ == "__main__":
    main()