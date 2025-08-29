"""
晶体结构后处理模块
使用 ASE (Atomic Simulation Environment) 对 pymatgen.Structure 进行能量优化
"""

import warnings
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ASE 相关导入
from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS
from ase.filters import ExpCellFilter
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

# pymatgen 相关导入
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor

# 用于进度显示
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console

console = Console()


def structure_to_atoms(structure: Structure) -> Atoms:
    """
    将 pymatgen.Structure 转换为 ASE.Atoms
    
    参数:
        structure: pymatgen.Structure 对象
        
    返回:
        ASE.Atoms 对象
    """
    # 使用 pymatgen 内置的适配器进行转换
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    
    # 确保周期性边界条件正确设置
    atoms.pbc = True
    
    return atoms


def atoms_to_structure(atoms: Atoms) -> Structure:
    """
    将 ASE.Atoms 转换为 pymatgen.Structure
    
    参数:
        atoms: ASE.Atoms 对象
        
    返回:
        pymatgen.Structure 对象
    """
    # 使用 pymatgen 内置的适配器进行转换
    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(atoms)
    
    return structure


def get_calculator(calculator_name: str = "chgnet", **kwargs) -> Any:
    """
    获取 ASE 计算器（势函数）
    
    参数:
        calculator_name: 计算器名称 ('emt', 'lj', 'mace', 'chgnet')
        **kwargs: 传递给计算器的额外参数
        
    返回:
        ASE 计算器对象
    """
    calculator_name = calculator_name.lower()
    
    if calculator_name == "emt":
        # EMT (Effective Medium Theory) - 适用于金属
        return EMT()
    
    elif calculator_name == "lj":
        # Lennard-Jones 势 - 简单通用势
        return LennardJones(**kwargs)
    
    elif calculator_name == "mace":
        # MACE - 机器学习势（需要额外安装 mace-torch）
        try:
            from mace.calculators import MACECalculator  # type: ignore
            model_path = kwargs.get('model_path', 'small')  # 使用预训练模型
            device = kwargs.get('device', 'cpu')
            return MACECalculator(model_path=model_path, device=device)
        except ImportError:
            warnings.warn("MACE 未安装，回退到 LJ 计算器")
            return LennardJones(**kwargs)
    
    elif calculator_name == "chgnet":
        # CHGNet - 通用机器学习势（需要额外安装 chgnet）
        try:
            from chgnet.model import CHGNetCalculator  # type: ignore
            model = kwargs.get('model', None)  # 使用默认预训练模型
            use_device = kwargs.get('use_device', 'cpu')
            return CHGNetCalculator(model=model, use_device=use_device)
        except ImportError:
            warnings.warn("CHGNet 未安装，回退到 LJ 计算器")
            return LennardJones(**kwargs)
    
    else:
        warnings.warn(f"未知的计算器 {calculator_name}，使用默认 LJ")
        return LennardJones(**kwargs)


def optimize_structure(
    structure: Structure,
    calculator: str = "chgnet",
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    steps: int = 500,
    fix_lattice: bool = False,
    fix_atoms_indices: Optional[List[int]] = None,
    calculator_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Tuple[Structure, Dict[str, Any]]:
    """
    对单个晶体结构进行能量优化
    
    参数:
        structure: pymatgen.Structure 对象
        calculator: 计算器名称 ('emt', 'lj', 'mace', 'chgnet')
        optimizer: 优化器名称 ('bfgs', 'fire', 'lbfgs')
        fmax: 力的收敛标准 (eV/Å)
        steps: 最大优化步数
        fix_lattice: 是否固定晶格参数（仅优化原子位置）
        fix_atoms_indices: 需要固定的原子索引列表
        calculator_kwargs: 传递给计算器的额外参数
        verbose: 是否显示优化过程
        
    返回:
        (优化后的结构, 优化信息字典)
    """
    # 转换为 ASE Atoms
    atoms = structure_to_atoms(structure)
    
    # 设置计算器
    calc_kwargs = calculator_kwargs or {}
    calc = get_calculator(calculator, **calc_kwargs)
    atoms.calc = calc
    
    # 设置约束
    constraints = []
    if fix_atoms_indices:
        constraints.append(FixAtoms(indices=fix_atoms_indices))
    
    if constraints:
        atoms.set_constraint(constraints)
    
    # 如果不固定晶格，使用 ExpCellFilter 允许晶格优化
    if not fix_lattice:
        atoms = ExpCellFilter(atoms)
    
    # 选择优化器
    optimizer_name = optimizer.lower()
    if optimizer_name == "bfgs":
        opt = BFGS(atoms, logfile=None if not verbose else '-')
    elif optimizer_name == "fire":
        opt = FIRE(atoms, logfile=None if not verbose else '-')
    elif optimizer_name == "lbfgs":
        opt = LBFGS(atoms, logfile=None if not verbose else '-')
    else:
        warnings.warn(f"未知的优化器 {optimizer}，使用默认 BFGS")
        opt = BFGS(atoms, logfile=None if not verbose else '-')
    
    # 记录初始能量
    try:
        initial_energy = atoms.get_potential_energy()
    except:
        initial_energy = None
    
    # 执行优化
    converged = False
    n_steps = 0
    try:
        converged = opt.run(fmax=fmax, steps=steps)
        n_steps = opt.nsteps
    except Exception as e:
        warnings.warn(f"优化过程出错: {e}")
    
    # 获取优化后的结构
    if not fix_lattice:
        atoms = atoms.atoms  # 从 ExpCellFilter 中提取原始 atoms
    
    optimized_structure = atoms_to_structure(atoms)
    
    # 记录最终能量
    try:
        final_energy = atoms.get_potential_energy()
    except:
        final_energy = None
    
    # 收集优化信息
    info = {
        'converged': converged,
        'n_steps': n_steps,
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'energy_change': final_energy - initial_energy if (initial_energy and final_energy) else None,
        'max_force': opt.fmax if hasattr(opt, 'fmax') else None
    }
    
    return optimized_structure, info


def _optimize_single_structure(args):
    """
    优化单个结构的辅助函数（用于并行处理）
    """
    idx, struct, calculator, optimizer, fmax, steps, fix_lattice, calculator_kwargs = args
    try:
        opt_struct, info = optimize_structure(
            struct,
            calculator=calculator,
            optimizer=optimizer,
            fmax=fmax,
            steps=steps,
            fix_lattice=fix_lattice,
            calculator_kwargs=calculator_kwargs,
            verbose=False
        )
        return idx, (opt_struct, info)
    except Exception as e:
        return idx, (struct, {'error': str(e), 'converged': False})


def optimize_structures_batch(
    structures: List[Structure],
    calculator: str = "chgnet",
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    steps: int = 500,
    fix_lattice: bool = False,
    n_workers: int = 4,
    calculator_kwargs: Optional[Dict[str, Any]] = None,
    show_progress: bool = True
) -> List[Tuple[Structure, Dict[str, Any]]]:
    """
    批量优化多个晶体结构（并行处理）
    
    参数:
        structures: pymatgen.Structure 对象列表
        calculator: 计算器名称
        optimizer: 优化器名称
        fmax: 力的收敛标准
        steps: 最大优化步数
        fix_lattice: 是否固定晶格参数
        n_workers: 并行工作进程数
        calculator_kwargs: 传递给计算器的额外参数
        show_progress: 是否显示进度条
        
    返回:
        [(优化后的结构, 优化信息)] 列表
    """
    results = []
    
    # 准备任务参数
    task_args = [
        (idx, struct, calculator, optimizer, fmax, steps, fix_lattice, calculator_kwargs)
        for idx, struct in enumerate(structures)
    ]
    
    if show_progress:
        # 使用进度条
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"优化 {len(structures)} 个结构...", 
                total=len(structures)
            )
            
            # 使用线程池处理（避免EMT序列化问题）
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_optimize_single_structure, args): args[0] 
                          for args in task_args}
                
                # 收集结果
                temp_results = {}
                for future in as_completed(futures):
                    idx, result = future.result()
                    temp_results[idx] = result
                    progress.update(task, advance=1)
                
                # 按原始顺序排列结果
                results = [temp_results[i] for i in range(len(structures))]
    else:
        # 不显示进度条的并行处理
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_optimize_single_structure, args): args[0] 
                      for args in task_args}
            
            temp_results = {}
            for future in as_completed(futures):
                idx, result = future.result()
                temp_results[idx] = result
            
            results = [temp_results[i] for i in range(len(structures))]
    
    # 统计优化结果
    n_converged = sum(1 for _, info in results if info.get('converged', False))
    console.print(f"\n[green]优化完成: {n_converged}/{len(structures)} 个结构收敛[/green]")
    
    return results


def optimize_with_multiple_calculators(
    structure: Structure,
    calculators: List[str] = ["chgnet", "lj"],
    select_best: str = "energy",
    **optimize_kwargs
) -> Tuple[Structure, Dict[str, Any]]:
    """
    使用多个计算器优化结构，选择最佳结果
    
    参数:
        structure: 输入结构
        calculators: 计算器列表
        select_best: 选择标准 ('energy' - 最低能量, 'converged' - 优先选择收敛的)
        **optimize_kwargs: 传递给 optimize_structure 的参数
        
    返回:
        最佳的 (优化后结构, 优化信息)
    """
    results = []
    
    for calc_name in calculators:
        try:
            opt_struct, info = optimize_structure(
                structure,
                calculator=calc_name,
                **optimize_kwargs
            )
            info['calculator'] = calc_name
            results.append((opt_struct, info))
        except Exception as e:
            console.print(f"[yellow]计算器 {calc_name} 优化失败: {e}[/yellow]")
    
    if not results:
        raise ValueError("所有计算器都失败了")
    
    # 选择最佳结果
    if select_best == "energy":
        # 选择能量最低的
        best_result = min(
            results, 
            key=lambda x: x[1].get('final_energy') if x[1].get('final_energy') is not None else float('inf')
        )
    elif select_best == "converged":
        # 优先选择收敛的，然后选择能量最低的
        converged_results = [r for r in results if r[1].get('converged', False)]
        if converged_results:
            best_result = min(
                converged_results,
                key=lambda x: x[1].get('final_energy') if x[1].get('final_energy') is not None else float('inf')
            )
        else:
            best_result = results[0]  # 如果都没收敛，返回第一个
    else:
        best_result = results[0]
    
    console.print(f"[cyan]选择了计算器: {best_result[1]['calculator']}[/cyan]")
    return best_result


# 便捷函数：针对竞赛需求的快速优化
def quick_optimize(
    structure: Structure,
    mode: str = "fast"
) -> Structure:
    """
    快速优化函数，预设参数
    
    参数:
        structure: 输入结构
        mode: 优化模式
            - 'fast': 快速优化，使用 EMT，较松的收敛标准
            - 'accurate': 精确优化，使用机器学习势，严格收敛标准
            - 'lattice_only': 仅优化晶格
            - 'atoms_only': 仅优化原子位置
            
    返回:
        优化后的结构
    """
    mode_configs = {
        'fast': {
            'calculator': 'chgnet',  # 使用CHGNet机器学习势
            'optimizer': 'fire',
            'fmax': 0.1,
            'steps': 200,
            'fix_lattice': False
        },
        'accurate': {
            'calculator': 'chgnet',  # 使用CHGNet进行精确优化
            'optimizer': 'bfgs',
            'fmax': 0.01,
            'steps': 1000,
            'fix_lattice': False
        },
        'lattice_only': {
            'calculator': 'chgnet',  # 使用CHGNet
            'optimizer': 'bfgs',
            'fmax': 0.05,
            'steps': 300,
            'fix_lattice': False,
            'fix_atoms_indices': list(range(len(structure)))  # 固定所有原子
        },
        'atoms_only': {
            'calculator': 'chgnet',  # 使用CHGNet
            'optimizer': 'bfgs',
            'fmax': 0.05,
            'steps': 300,
            'fix_lattice': True
        }
    }
    
    if mode not in mode_configs:
        raise ValueError(f"未知的优化模式: {mode}")
    
    config = mode_configs[mode]
    opt_struct, info = optimize_structure(structure, **config)
    
    if not info.get('converged', False):
        console.print(f"[yellow]警告: 优化未完全收敛 (模式: {mode})[/yellow]")
    
    return opt_struct


if __name__ == "__main__":
    # 测试代码
    console.print("[bold cyan]后处理模块测试[/bold cyan]\n")
    
    # 创建一个简单的测试结构（NaCl）
    lattice = Lattice.cubic(4.0)
    species = ["Na", "Cl"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    test_structure = Structure(lattice, species, coords)
    
    console.print("原始结构:")
    console.print(f"  晶格常数: {test_structure.lattice.abc}")
    console.print(f"  体积: {test_structure.volume:.2f} Å³\n")
    
    # 测试基本优化
    console.print("[green]测试基本优化...[/green]")
    opt_structure, info = optimize_structure(
        test_structure,
        calculator="emt",
        verbose=True
    )
    
    console.print(f"\n优化后结构:")
    console.print(f"  晶格常数: {opt_structure.lattice.abc}")
    console.print(f"  体积: {opt_structure.volume:.2f} Å³")
    console.print(f"  优化信息: {info}\n")
    
    # 测试快速优化模式
    console.print("[green]测试快速优化模式...[/green]")
    quick_opt = quick_optimize(test_structure, mode='fast')
    console.print(f"  快速优化后体积: {quick_opt.volume:.2f} Å³\n")
    
    console.print("[bold green]测试完成！[/bold green]")