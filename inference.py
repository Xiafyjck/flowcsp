"""
晶体结构推理程序
使用训练好的模型进行推理，严格复用现有标准接口
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import asyncio
import threading
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from pymatgen.core import Structure, Lattice, Composition, Element
from pymatgen.io.cif import CifWriter
import warnings
warnings.filterwarnings('ignore')

# 导入标准接口
from src.networks import build_network
from src.flows import build_flow
from src.pxrd_simulator import PXRDSimulator
from postprocess import quick_optimize

console = Console()


class InferenceEngine:
    """推理引擎 - 严格使用标准接口"""
    
    def __init__(
        self, 
        checkpoint_path: str,
        data_path: str,
        output_path: str = "submission.csv",
        device: str = "cuda",
        batch_size: int = 8,
        num_workers: int = 4
    ):
        """
        初始化推理引擎
        
        Args:
            checkpoint_path: 模型检查点路径
            data_path: 输入数据路径
            output_path: 输出CSV路径
            device: 设备
            batch_size: 批量大小
            num_workers: 并行工作进程数
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_path = Path(output_path)
        
        # 加载数据
        self.data_path = Path(data_path)
        self.load_data()
        
        # 硬编码配置（与训练保持一致）
        self.network_config = {
            'max_atoms': 60,
            'hidden_channels': 256,
            'num_layers': 8,
            'num_heads': 16,
            'lmax': 3,
            'max_radius': 12.0,
            'num_basis': 256,
            'dropout_rate': 0.0,  # 推理时不用dropout
            'matscholar_path': 'src/networks/matscholar.json'
        }
        
        self.flow_config = {
            'sigma_min': 1e-4,
            'sigma_max': 1.0,
            'loss_weight_lattice': 1.0,
            'loss_weight_coords': 1.0,
            'cfg_prob': 0.0,  # 推理时不用dropout
            'cfg_scale': 1.5,  # CFG引导强度
            'invariant_loss_weight': 0.01,
            'default_num_steps': 100,
            'stats_file': 'data/stats.npz',  # 归一化统计文件
            'normalize_lattice': True,
            'normalize_frac_coords': True,
            'use_global_stats': True
        }
        
        # 构建网络和流模型（使用标准接口）
        console.print("[cyan]构建模型...[/cyan]")
        self.network = build_network("equiformer", self.network_config)
        self.flow = build_flow("cfm_cfg", self.network, self.flow_config)
        
        # 加载检查点
        if checkpoint_path and Path(checkpoint_path).exists():
            console.print(f"[cyan]加载检查点: {checkpoint_path}[/cyan]")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 处理state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 移除'flow.'前缀（如果有）
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('flow.'):
                    new_state_dict[k[5:]] = v
                else:
                    new_state_dict[k] = v
            
            self.flow.load_state_dict(new_state_dict, strict=False)
            console.print("[green]模型加载成功[/green]")
        else:
            console.print("[yellow]警告: 未找到检查点，使用随机初始化的模型[/yellow]")
        
        self.flow.to(self.device)
        self.flow.eval()
        
        # PXRD模拟器
        self.pxrd_simulator = PXRDSimulator()
        
        # 结果存储
        self.results = {}  # {sample_id: (structure, rwp)}
        self.results_lock = threading.Lock()
        
        # GPU资源锁（生成和后处理互斥）
        self.gpu_lock = threading.Lock()
        
    def load_data(self):
        """加载比赛数据"""
        console.print("[cyan]加载数据...[/cyan]")
        
        # 加载组成数据
        comp_file = self.data_path / "composition.json"
        with open(comp_file, 'r') as f:
            comp_data = json.load(f)
        
        # 加载PXRD数据
        self.samples = []
        pattern_dir = self.data_path / "pattern"
        
        for sample_id, comp_info in comp_data.items():
            # 处理组成数据格式
            if isinstance(comp_info, dict) and 'composition' in comp_info:
                comp_str = comp_info['composition'][0]  # 取第一个组成
            else:
                comp_str = comp_info
            
            # 解析组成
            comp = self.parse_composition(comp_str)
            
            # 加载PXRD (.xy格式)
            pxrd_file = pattern_dir / f"{sample_id}.xy"
            if pxrd_file.exists():
                # 读取.xy文件
                pxrd_data = []
                with open(pxrd_file, 'r') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                # 只读取强度值（第二列）
                                intensity = float(parts[1])
                                pxrd_data.append(intensity)
                            except ValueError:
                                continue
                
                pxrd = np.array(pxrd_data, dtype=np.float32)
                
                # 确保是11501个点
                if len(pxrd) != 11501:
                    console.print(f"[yellow]警告: {sample_id} 有 {len(pxrd)} 个点[/yellow]")
                    if len(pxrd) > 11501:
                        pxrd = pxrd[:11501]
                    else:
                        pxrd = np.pad(pxrd, (0, 11501 - len(pxrd)), 'constant')
                
                self.samples.append({
                    'id': sample_id,
                    'comp': comp,
                    'pxrd': pxrd,
                    'num_atoms': np.sum(comp > 0)
                })
        
        console.print(f"[green]加载了 {len(self.samples)} 个样本[/green]")
    
    def parse_composition(self, comp_str: str) -> np.ndarray:
        """解析组成字符串为原子数组（使用pymatgen）"""
        # 使用pymatgen解析组成
        composition = Composition(comp_str)
        
        # 创建原子数组
        atom_list = []
        for element, count in composition.items():
            # 获取元素的原子序数
            atomic_num = Element(element).Z
            # 根据数量重复原子序数
            atom_list.extend([atomic_num] * int(count))
        
        # 填充到60维数组
        comp = np.zeros(60, dtype=np.float32)
        for i, atomic_num in enumerate(atom_list[:60]):
            comp[i] = atomic_num
        
        return comp
    
    def prepare_batch(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """准备批次数据"""
        batch_size = len(samples)
        
        # 准备条件
        conditions = {
            'comp': torch.zeros(batch_size, 60, dtype=torch.float32),
            'pxrd': torch.zeros(batch_size, 11501, dtype=torch.float32),
            'num_atoms': torch.zeros(batch_size, dtype=torch.long)
        }
        
        for i, sample in enumerate(samples):
            conditions['comp'][i] = torch.from_numpy(sample['comp'])
            conditions['pxrd'][i] = torch.from_numpy(sample['pxrd'])
            conditions['num_atoms'][i] = sample['num_atoms']
        
        # 移到设备
        for k, v in conditions.items():
            conditions[k] = v.to(self.device)
        
        return conditions
    
    def generate_structures(self, samples: List[Dict], num_candidates: int = 5) -> List[Tuple[str, Structure, float]]:
        """
        生成晶体结构（使用标准flow.sample接口）
        
        Returns:
            [(sample_id, structure, rwp), ...]
        """
        results = []
        
        with self.gpu_lock:
            # 准备批次
            conditions = self.prepare_batch(samples)
            batch_size = len(samples)
            
            # 生成多个候选
            all_structures = []
            
            with torch.no_grad():
                for _ in range(num_candidates):
                    # 使用标准sample接口
                    z = self.flow.sample(
                        conditions=conditions,
                        num_steps=100,
                        temperature=1.0,
                        guidance_scale=1.5  # CFG引导强度
                    )
                    
                    # 转换为Structure对象
                    for i in range(batch_size):
                        sample_id = samples[i]['id']
                        structure = self.z_to_structure(
                            z[i].cpu().numpy(),
                            samples[i]['comp'],
                            samples[i]['num_atoms']
                        )
                        all_structures.append((sample_id, structure))
            
            # 评估RWP（CPU密集）
            for sample_id, structure in all_structures:
                try:
                    rwp = self.calculate_rwp(
                        structure,
                        next(s for s in samples if s['id'] == sample_id)
                    )
                    results.append((sample_id, structure, rwp))
                except Exception as e:
                    console.print(f"[yellow]RWP计算失败 {sample_id}: {e}[/yellow]")
        
        return results
    
    def z_to_structure(self, z: np.ndarray, comp: np.ndarray, num_atoms: int) -> Structure:
        """将模型输出转换为pymatgen Structure"""
        # 提取晶格参数和分数坐标
        lattice_matrix = z[:3, :]  # [3, 3]
        frac_coords = z[3:3+num_atoms, :]  # [num_atoms, 3]
        
        # 创建Lattice对象
        lattice = Lattice(lattice_matrix)
        
        # 原子类型 - 使用pymatgen的Element类
        species = []
        for i in range(num_atoms):
            atomic_num = int(comp[i])
            if atomic_num > 0:
                # 使用原子序数创建Element对象
                element = Element.from_Z(atomic_num)
                species.append(element.symbol)
            else:
                species.append('H')  # 默认元素（不应该发生）
        
        # 创建Structure
        structure = Structure(lattice, species, frac_coords)
        
        return structure
    
    def calculate_rwp(self, structure: Structure, sample: Dict) -> float:
        """计算RWP指标"""
        try:
            # 模拟PXRD
            _, y_pred = self.pxrd_simulator.simulate(structure)
            
            # 获取目标PXRD
            y_true = sample['pxrd']
            
            # 计算RWP
            numerator = np.sum((y_true - y_pred) ** 2)
            denominator = np.sum(y_true ** 2)
            rwp = np.sqrt(numerator / (denominator + 1e-8))
            
            return rwp
        except Exception:
            return float('inf')
    
    def postprocess_structure(self, structure: Structure) -> Structure:
        """后处理优化结构"""
        with self.gpu_lock:
            try:
                # 使用quick_optimize进行快速优化
                optimized = quick_optimize(structure, mode='fast')
                return optimized
            except Exception as e:
                console.print(f"[yellow]后处理失败: {e}[/yellow]")
                return structure
    
    def update_submission(self, sample_id: str, structure: Structure, rwp: float):
        """更新submission.csv"""
        with self.results_lock:
            # 检查是否有更好的结果
            if sample_id not in self.results or self.results[sample_id][1] > rwp:
                self.results[sample_id] = (structure, rwp)
                
                # 写入CSV
                self.write_submission_csv()
    
    def write_submission_csv(self):
        """写入submission.csv"""
        rows = []
        
        for sample_id, (structure, rwp) in self.results.items():
            # 转换为CIF字符串
            cif = CifWriter(structure)
            cif_str = str(cif)
            
            # 移除换行符，用分号分隔
            cif_str = cif_str.replace('\n', ';')
            
            rows.append({
                'id': sample_id,
                'cif': cif_str,
                'rwp': rwp
            })
        
        # 保存为CSV
        df = pd.DataFrame(rows)
        df.to_csv(self.output_path, index=False)
        console.print(f"[green]更新 {self.output_path} ({len(rows)} 个样本)[/green]")
    
    async def process_sample_async(self, sample: Dict, max_attempts: int = 10):
        """异步处理单个样本"""
        sample_id = sample['id']
        best_rwp = float('inf')
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            # 生成候选结构
            results = self.generate_structures([sample], num_candidates=3)
            
            for sid, structure, rwp in results:
                if rwp < best_rwp:
                    best_rwp = rwp
                    
                    # 如果RWP足够好，进行后处理
                    if rwp < 3.0:
                        # 后处理优化
                        optimized = self.postprocess_structure(structure)
                        
                        # 重新计算RWP
                        rwp_opt = self.calculate_rwp(optimized, sample)
                        
                        if rwp_opt < rwp:
                            structure = optimized
                            rwp = rwp_opt
                    
                    # 更新结果
                    self.update_submission(sid, structure, rwp)
                    
                    # 如果足够好就停止
                    if rwp < 0.3:
                        console.print(f"[green]✓ {sample_id}: RWP={rwp:.4f} (尝试 {attempts} 次)[/green]")
                        return
        
        console.print(f"[yellow]! {sample_id}: 最佳RWP={best_rwp:.4f} (达到最大尝试次数)[/yellow]")
    
    async def run_async(self):
        """异步运行推理"""
        console.print("[bold cyan]开始推理...[/bold cyan]")
        
        # 创建任务
        tasks = []
        for sample in self.samples:
            task = asyncio.create_task(self.process_sample_async(sample))
            tasks.append(task)
        
        # 等待所有任务完成
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task_id = progress.add_task(
                f"处理 {len(self.samples)} 个样本...",
                total=len(self.samples)
            )
            
            for task in asyncio.as_completed(tasks):
                await task
                progress.update(task_id, advance=1)
        
        # 最终统计
        self.print_statistics()
    
    def run(self):
        """运行推理（同步接口）"""
        # 使用批处理方式，更高效
        console.print("[bold cyan]开始批量推理...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            # 分批处理
            num_batches = (len(self.samples) + self.batch_size - 1) // self.batch_size
            task_id = progress.add_task(
                f"处理 {len(self.samples)} 个样本...",
                total=len(self.samples)
            )
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.samples))
                batch_samples = self.samples[start_idx:end_idx]
                
                # 生成并评估
                max_attempts = 5
                for attempt in range(max_attempts):
                    results = self.generate_structures(batch_samples, num_candidates=3)
                    
                    # 处理每个结果
                    for sample_id, structure, rwp in results:
                        # 如果需要后处理
                        if 0.3 < rwp < 3.0:
                            optimized = self.postprocess_structure(structure)
                            sample = next(s for s in batch_samples if s['id'] == sample_id)
                            rwp_opt = self.calculate_rwp(optimized, sample)
                            if rwp_opt < rwp:
                                structure = optimized
                                rwp = rwp_opt
                        
                        # 更新结果
                        self.update_submission(sample_id, structure, rwp)
                    
                    # 检查是否所有样本都足够好
                    all_good = True
                    for sample in batch_samples:
                        if sample['id'] in self.results:
                            if self.results[sample['id']][1] > 0.3:
                                all_good = False
                        else:
                            all_good = False
                    
                    if all_good:
                        break
                
                progress.update(task_id, advance=len(batch_samples))
        
        self.print_statistics()
    
    def print_statistics(self):
        """打印统计信息"""
        table = Table(title="推理结果统计")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        
        rwps = [rwp for _, rwp in self.results.values()]
        
        table.add_row("总样本数", str(len(self.samples)))
        table.add_row("完成样本数", str(len(self.results)))
        
        if rwps:
            table.add_row("平均RWP", f"{np.mean(rwps):.4f}")
            table.add_row("最小RWP", f"{np.min(rwps):.4f}")
            table.add_row("最大RWP", f"{np.max(rwps):.4f}")
            table.add_row("RWP<0.3", str(sum(1 for r in rwps if r < 0.3)))
            table.add_row("RWP<1.0", str(sum(1 for r in rwps if r < 1.0)))
            table.add_row("RWP<3.0", str(sum(1 for r in rwps if r < 3.0)))
        
        console.print(table)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="晶体结构推理")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='模型检查点路径')
    parser.add_argument('--data', type=str, default='/home/ma-user/work/mincycle4csp/raw_data/A_sample',
                        help='输入数据路径')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='输出CSV路径')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批量大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--async-mode', action='store_true',
                        help='使用异步模式')
    
    args = parser.parse_args()
    
    # 创建推理引擎
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # 运行推理
    if args.async_mode:
        asyncio.run(engine.run_async())
    else:
        engine.run()
    
    console.print(f"[bold green]推理完成！结果保存到 {args.output}[/bold green]")


if __name__ == "__main__":
    main()