#!/usr/bin/env python
"""
CSV文件随机抽样合并脚本
从指定文件夹中的多个CSV文件中随机抽样，合并成一个文件
只处理id和cif两列，支持按总数抽样
"""

import argparse
import pandas as pd
from pathlib import Path
import random
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()


def sample_merge_csvs(
    folder_path: str,
    input_id_col: str,
    input_cif_col: str,
    output_id_col: str,
    output_cif_col: str,
    output_path: str,
    total_samples: int,
    seed: int = 42,
    shuffle: bool = True
):
    """
    从多个CSV文件中随机抽样并合并
    
    Args:
        folder_path: 包含CSV文件的文件夹路径
        input_id_col: 输入CSV中的ID列名
        input_cif_col: 输入CSV中的CIF列名  
        output_id_col: 输出CSV中的ID列名
        output_cif_col: 输出CSV中的CIF列名
        output_path: 输出文件路径
        total_samples: 总共抽样的数量（跨所有文件）
        seed: 随机种子
        shuffle: 是否在最后打乱合并的数据
    """
    # 设置随机种子
    random.seed(seed)
    
    folder = Path(folder_path)
    if not folder.exists():
        console.print(f"[red]错误: 文件夹不存在: {folder_path}[/red]")
        return
    
    # 找到所有CSV文件
    csv_files = list(folder.glob("*.csv"))
    # 排除已经抽样的文件
    csv_files = [f for f in csv_files if not f.stem.endswith('_sampled')]
    
    if not csv_files:
        console.print(f"[red]错误: 文件夹中没有CSV文件: {folder_path}[/red]")
        return
    
    console.print(f"\n[cyan]找到 {len(csv_files)} 个CSV文件[/cyan]")
    
    # 创建列名映射
    column_mapping = {input_id_col: output_id_col, input_cif_col: output_cif_col}
    
    # 收集所有数据
    all_samples = []
    file_stats = []
    input_cols = [input_id_col, input_cif_col]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        # 统计所有文件的实际记录数（使用pandas解析）
        task = progress.add_task("[cyan]统计文件记录数...", total=len(csv_files))
        total_rows = 0
        file_rows = {}
        
        for csv_file in csv_files:
            try:
                # 使用pandas统计实际记录数（处理包含换行符的字段）
                df_temp = pd.read_csv(csv_file, usecols=input_cols, nrows=None)
                rows = len(df_temp)
                file_rows[csv_file] = rows
                total_rows += rows
                progress.update(task, advance=1)
                del df_temp  # 释放内存
            except Exception as e:
                console.print(f"[yellow]警告: 无法读取 {csv_file.name}: {e}[/yellow]")
                file_rows[csv_file] = 0
                progress.update(task, advance=1)
        
        if total_rows == 0:
            console.print("[red]错误: 所有文件都是空的[/red]")
            return
        
        # 基于随机生成的分配算法（避免除法计算）
        actual_total_samples = min(total_samples, total_rows)
        console.print(f"[yellow]总行数: {total_rows}, 目标抽样数: {actual_total_samples}[/yellow]")
        
        # 构建全局行索引映射：每一行对应哪个文件
        non_empty_files = [f for f in csv_files if file_rows.get(f, 0) > 0]
        
        if len(non_empty_files) == 0:
            console.print("[red]错误: 没有非空文件[/red]")
            return
        
        console.print(f"[cyan]使用随机抽样分配模式（避免除法计算）[/cyan]")
        
        # 构建全局行索引到文件的映射表
        global_row_to_file = []  # 每个元素是(file, local_row_index)
        
        for csv_file in non_empty_files:
            file_row_count = file_rows[csv_file]
            for local_row in range(file_row_count):
                global_row_to_file.append((csv_file, local_row))
        
        # 验证映射表大小
        assert len(global_row_to_file) == total_rows, f"映射表大小不匹配: {len(global_row_to_file)} != {total_rows}"
        
        # 随机选择全局行索引
        selected_global_indices = random.sample(range(total_rows), actual_total_samples)
        
        # 统计每个文件被选中的次数
        file_allocations = {f: 0 for f in csv_files}  # 初始化所有文件为0
        selected_rows_by_file = {f: [] for f in non_empty_files}  # 记录每个文件被选中的具体行号
        
        for global_idx in selected_global_indices:
            selected_file, local_row = global_row_to_file[global_idx]
            file_allocations[selected_file] += 1
            selected_rows_by_file[selected_file].append(local_row)
        
        # 验证总数
        final_total = sum(file_allocations.values())
        console.print(f"[cyan]分配验证: 期望={actual_total_samples}, 实际分配={final_total}[/cyan]")
        
        # 读取并抽样数据（使用预先计算的行号）
        task = progress.add_task("[cyan]读取并抽样文件...", total=len(csv_files))
        
        for csv_file in csv_files:
            try:
                progress.update(task, description=f"[cyan]处理: {csv_file.name}")
                
                # 获取该文件被选中的行号
                file_sample_size = file_allocations.get(csv_file, 0)
                
                # 读取CSV文件
                if file_sample_size > 0 and csv_file in selected_rows_by_file:
                    selected_local_rows = selected_rows_by_file[csv_file]
                    total_file_rows = file_rows[csv_file]
                    
                    if len(selected_local_rows) > 0:
                        # 读取整个文件，然后根据选中的行号进行筛选
                        # 这种方法对于包含换行符的CIF数据更可靠
                        df = pd.read_csv(csv_file, usecols=input_cols)
                        
                        # 根据选中的行索引筛选数据
                        selected_local_rows = [idx for idx in selected_local_rows if idx < len(df)]
                        if selected_local_rows:
                            df = df.iloc[selected_local_rows].copy()
                        else:
                            df = df.iloc[:0].copy()  # 空DataFrame
                        
                        # 重命名列
                        df = df.rename(columns=column_mapping)
                        all_samples.append(df)
                        
                        file_stats.append({
                            'name': csv_file.name,
                            'total_rows': total_file_rows,
                            'sampled_rows': len(df),
                            'status': 'success'
                        })
                    else:
                        file_stats.append({
                            'name': csv_file.name,
                            'total_rows': total_file_rows,
                            'sampled_rows': 0,
                            'status': 'no_selection'
                        })
                else:
                    # 文件未被选中或为空文件
                    file_stats.append({
                        'name': csv_file.name,
                        'total_rows': file_rows.get(csv_file, 0),
                        'sampled_rows': 0,
                        'status': 'skipped' if file_sample_size == 0 else 'empty'
                    })
                
                progress.update(task, advance=1)
                
            except Exception as e:
                console.print(f"[red]错误: 处理 {csv_file.name} 时出错: {e}[/red]")
                file_stats.append({
                    'name': csv_file.name,
                    'total_rows': file_rows.get(csv_file, 0),
                    'sampled_rows': 0,
                    'status': f'error: {str(e)}'
                })
                progress.update(task, advance=1)
    
    # 合并所有样本
    if not all_samples:
        console.print("[red]错误: 没有成功读取任何数据[/red]")
        return
    
    merged_df = pd.concat(all_samples, ignore_index=True)
    
    # 如果需要打乱数据
    if shuffle:
        merged_df = merged_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 保存结果
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output, index=False)
    
    # 显示统计结果
    console.print("\n[bold cyan]抽样统计:[/bold cyan]")
    
    # 创建统计表格
    stats_table = Table(title="文件抽样详情")
    stats_table.add_column("文件名", style="cyan", no_wrap=True)
    stats_table.add_column("原始行数", style="yellow", justify="right")
    stats_table.add_column("抽样行数", style="green", justify="right")
    stats_table.add_column("抽样比例", style="magenta", justify="right")
    stats_table.add_column("状态", style="bold")
    
    total_original = 0
    total_sampled = 0
    
    for stat in file_stats:
        if stat['status'] == 'success':
            ratio = f"{stat['sampled_rows']/stat['total_rows']*100:.1f}%" if stat['total_rows'] > 0 else "-"
            status_display = "[green]✓[/green]"
            total_original += stat['total_rows']
            total_sampled += stat['sampled_rows']
        elif stat['status'] == 'empty':
            ratio = "-"
            status_display = "[yellow]空文件[/yellow]"
        elif stat['status'] == 'skipped':
            ratio = "-"
            status_display = "[yellow]跳过[/yellow]"
        elif stat['status'] == 'no_selection':
            ratio = "-"
            status_display = "[yellow]未选中[/yellow]"
            total_original += stat['total_rows']
        else:
            ratio = "-"
            status_display = f"[red]✗[/red]"
        
        stats_table.add_row(
            stat['name'],
            str(stat['total_rows']),
            str(stat['sampled_rows']),
            ratio,
            status_display
        )
    
    # 添加总计行
    stats_table.add_row(
        "[bold]总计[/bold]",
        f"[bold]{total_original}[/bold]",
        f"[bold]{total_sampled}[/bold]",
        f"[bold]{total_sampled/total_original*100:.1f}%[/bold]" if total_original > 0 else "-",
        ""
    )
    
    console.print(stats_table)
    
    # 输出最终信息
    console.print(f"\n[bold green]✓ 合并完成![/bold green]")
    console.print(f"  输出文件: {output}")
    console.print(f"  最终行数: {len(merged_df)}")
    console.print(f"  数据列: {output_id_col}, {output_cif_col}")


def main():
    parser = argparse.ArgumentParser(
        description="从多个CSV文件中随机抽样并合并成一个文件（只处理id和cif两列）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 总共抽样5000行（按文件大小比例分配）
  %(prog)s /data/csvs --samples 5000 --id-col id --cif-col cif --output-id-col ID --output-cif-col CIF -o merged.csv
  
  # 抽样10000行
  %(prog)s /data/csvs --samples 10000 --id-col material_id --cif-col structure --output-id-col id --output-cif-col cif -o output.csv
        """
    )
    
    parser.add_argument('folder', help='包含CSV文件的文件夹路径')
    parser.add_argument('--samples', type=int, required=True,
                       help='总共抽样的行数（按文件大小比例分配）')
    parser.add_argument('--id-col', required=True, 
                       help='输入CSV的ID列名')
    parser.add_argument('--cif-col', required=True,
                       help='输入CSV的CIF列名')
    parser.add_argument('--output-id-col', required=True,
                       help='输出CSV的ID列名')
    parser.add_argument('--output-cif-col', required=True,
                       help='输出CSV的CIF列名')
    parser.add_argument('-o', '--output', required=True,
                       help='输出文件路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='不打乱最终的合并数据')
    
    args = parser.parse_args()
    
    # 显示配置
    console.print("\n[bold cyan]随机抽样合并配置:[/bold cyan]")
    console.print(f"  输入文件夹: {args.folder}")
    console.print(f"  输入列: {args.id_col}, {args.cif_col}")
    console.print(f"  输出列: {args.output_id_col}, {args.output_cif_col}")
    console.print(f"  输出文件: {args.output}")
    console.print(f"  抽样数量: {args.samples} 行")
    console.print(f"  随机种子: {args.seed}")
    console.print(f"  最终打乱: {'是' if not args.no_shuffle else '否'}")
    
    # 执行抽样合并
    sample_merge_csvs(
        folder_path=args.folder,
        input_id_col=args.id_col,
        input_cif_col=args.cif_col,
        output_id_col=args.output_id_col,
        output_cif_col=args.output_cif_col,
        output_path=args.output,
        total_samples=args.samples,
        seed=args.seed,
        shuffle=not args.no_shuffle
    )


if __name__ == '__main__':
    main()