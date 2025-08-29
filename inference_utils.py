"""
推理工具函数模块

包含晶体结构生成推理所需的所有工具函数：
- 数据加载和处理
- 模型加载和推理（支持CFG）
- PXRD计算和质量评估
- 后处理和优化
- 文件I/O操作
"""
import time

import json
import os
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from pymatgen.core import Structure, Lattice, Composition, Element
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')


# ============== 数据加载函数 ==============

def read_xy_file(file_path):
    """
    读取.xy格式的PXRD数据
    
    Args:
        file_path: .xy文件路径
        
    Returns:
        np.array: PXRD强度数组
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                intensity = float(parts[1])
                data.append(intensity)
    return np.array(data, dtype=np.float32)


def parse_composition(comp_str):
    """
    解析组成字符串为原子类型和数量
    
    Args:
        comp_str: 组成字符串，如"Li2 Mn1 O4"
        
    Returns:
        tuple: (原子数量, 60维原子类型数组)
    """
    comp = Composition(comp_str)
    atom_list = []
    
    for element, count in comp.items():
        atomic_num = Element(element).Z
        atom_list.extend([atomic_num] * int(count))
    
    # 填充到60维
    atom_types = np.zeros(60, dtype=np.int32)
    atom_types[:len(atom_list)] = atom_list[:60]
    
    return len(atom_list), atom_types


def load_competition_data(data_dir):
    """
    加载比赛格式数据
    
    Args:
        data_dir: 数据目录路径，包含composition.json和pattern/目录
        
    Returns:
        pd.DataFrame: 包含所有样本信息的数据框
    """
    data_dir = Path(data_dir)
    
    # 读取composition
    with open(data_dir / "composition.json", 'r') as f:
        compositions = json.load(f)
    
    # 准备数据列表
    data_list = []
    
    for sample_id, comp_info in tqdm(compositions.items(), desc="加载数据"):
        # 获取组成信息
        comp_list = comp_info["composition"]
        niggli_comp = comp_list[0]
        primitive_comp = comp_list[1] if len(comp_list) > 1 else comp_list[0]
        
        # 解析原子信息
        num_atoms, atom_types = parse_composition(niggli_comp)
        
        # 读取PXRD数据
        pattern_file = data_dir / "pattern" / f"{sample_id}.xy"
        if pattern_file.exists():
            pxrd = read_xy_file(pattern_file)
            # 确保长度为11501
            if len(pxrd) < 11501:
                pxrd_full = np.zeros(11501, dtype=np.float32)
                pxrd_full[:len(pxrd)] = pxrd
                pxrd = pxrd_full
            elif len(pxrd) > 11501:
                pxrd = pxrd[:11501]
        else:
            pxrd = np.zeros(11501, dtype=np.float32)
        
        data_list.append({
            'id': sample_id,
            'niggli_comp': niggli_comp,
            'primitive_comp': primitive_comp,
            'atom_types': atom_types,
            'num_atoms': num_atoms,
            'pxrd': pxrd  # 观测的PXRD谱
        })
    
    return pd.DataFrame(data_list)


def load_lattice_stats(stats_path="data/lattice_stats.json"):
    """
    加载晶格参数归一化统计信息
    
    Args:
        stats_path: 统计信息JSON文件路径，默认为data/lattice_stats.json
        
    Returns:
        dict: 包含mean和std的统计信息字典
    """
    stats_path = Path(stats_path)
    
    if not stats_path.exists():
        print(f"⚠️ 归一化统计文件不存在: {stats_path}")
        print("  使用默认统计值（可能影响模型性能）")
        # 返回默认统计值
        return {
            'lattice_mean': np.array([[5.0, 0.0, 0.0],
                                     [0.0, 5.0, 0.0],
                                     [0.0, 0.0, 5.0]], dtype=np.float32),
            'lattice_std': np.array([[2.0, 2.0, 2.0],
                                    [2.0, 2.0, 2.0],
                                    [2.0, 2.0, 2.0]], dtype=np.float32)
        }
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # 转换为numpy数组
    stats_dict = {
        'lattice_mean': np.array(stats.get('lattice_mean', stats.get('mean', [[5,0,0],[0,5,0],[0,0,5]])), dtype=np.float32),
        'lattice_std': np.array(stats.get('lattice_std', stats.get('std', [[2,2,2],[2,2,2],[2,2,2]])), dtype=np.float32)
    }
    
    print(f"✅ 加载归一化统计信息: {stats_path}")
    print(f"   晶格均值范围: [{stats_dict['lattice_mean'].min():.2f}, {stats_dict['lattice_mean'].max():.2f}]")
    print(f"   晶格标准差范围: [{stats_dict['lattice_std'].min():.2f}, {stats_dict['lattice_std'].max():.2f}]")
    
    return stats_dict


# ============== 模型加载和推理函数 ==============

def load_model(model_path, device='auto'):
    """
    加载训练好的模型（支持cfm_cfg流）
    
    Args:
        model_path: checkpoint文件路径
        device: 设备选择 - 'auto'(自动选择), 'cuda', 'cpu', 或torch.device对象
        
    Returns:
        加载好的Lightning模块
    """
    from src.trainer import CrystalGenerationModule
    
    print(f"正在加载模型: {model_path}")
    
    # 处理设备选择
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    # 如果已经是torch.device对象，直接使用
    
    # 从checkpoint加载模型
    model = CrystalGenerationModule.load_from_checkpoint(
        model_path,
        map_location=device
    )
    
    # 验证流模型类型
    flow_name = model.hparams.get('flow_name', 'cfm')
    print(f"  检测到流模型: {flow_name}")
    
    # 设置为评估模式
    model.eval()
    model = model.to(device)
    
    print(f"✅ 模型加载成功，设备: {device}")
    return model


def initialize_normalizer(stats_path="data/lattice_stats.json", lattice_stats=None):
    """
    初始化数据归一化器
    
    Args:
        stats_path: 统计信息文件路径
        lattice_stats: 已加载的统计信息字典（可选）
        
    Returns:
        DataNormalizer实例
    """
    from src.normalizer import DataNormalizer
    
    try:
        # 如果提供了stats_path，使用它初始化
        if stats_path and Path(stats_path).exists():
            data_normalizer = DataNormalizer(stats_file=stats_path)
            print(f"✅ 使用文件初始化归一化器: {stats_path}")
        else:
            # 否则创建临时文件并初始化
            import tempfile
            import json
            
            # 使用提供的统计信息或默认值
            if lattice_stats is None:
                lattice_stats = load_lattice_stats(stats_path)
            
            # 准备符合DataNormalizer期望格式的统计信息
            normalizer_stats = {
                'lattice_global_mean': float(lattice_stats['lattice_mean'].mean()),
                'lattice_global_std': float(lattice_stats['lattice_std'].mean()),
                'lattice_mean': lattice_stats['lattice_mean'].flatten().tolist(),
                'lattice_std': lattice_stats['lattice_std'].flatten().tolist(),
                'frac_coords_mean': [0.5, 0.5, 0.5],  # 分数坐标默认值
                'frac_coords_std': [0.3, 0.3, 0.3]
            }
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(normalizer_stats, f)
                temp_stats_file = f.name
            
            # 使用临时文件初始化
            data_normalizer = DataNormalizer(stats_file=temp_stats_file)
            
            # 删除临时文件
            Path(temp_stats_file).unlink()
            print("✅ 使用统计信息初始化归一化器")
            
        return data_normalizer
        
    except Exception as e:
        print(f"⚠️ 归一化器初始化失败: {e}")
        print("  将使用备用方案（可能影响模型性能）")
        
        # 创建一个简单的归一化器作为备用
        class SimpleNormalizer:
            def __init__(self):
                self.lattice_mean = torch.zeros(3, 3)
                self.lattice_std = torch.ones(3, 3)
            
            def denormalize_z(self, z):
                # 简单的反归一化：直接返回
                return z * 5.0  # 假设典型晶格参数在5埃左右
        
        return SimpleNormalizer()


def generate_crystal_structures_batch_cfg(samples_df, model, data_normalizer, 
                                         batch_size=32, guidance_scale=1.5,
                                         adaptive_mode=False, 
                                         min_scale=0.8, max_scale=2.5):
    """
    批量生成晶体结构（使用CFG引导）
    
    Args:
        samples_df: 包含多个样本的DataFrame
        model: 训练好的模型
        data_normalizer: 数据归一化器
        batch_size: 批处理大小
        guidance_scale: CFG引导强度（None使用自适应）
        adaptive_mode: 是否使用自适应引导强度
        min_scale: 自适应模式下的最小引导强度
        max_scale: 自适应模式下的最大引导强度
    
    Returns:
        tuple: (Structure对象列表, 使用的guidance_scale列表)
    """
    device = next(model.parameters()).device
    structures = []
    scales_used = []
    
    # 按批次处理
    num_samples = len(samples_df)
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_df = samples_df.iloc[batch_start:batch_end]
        
        # 准备批次数据
        batch = {
            'comp': torch.tensor(
                np.stack(batch_df['atom_types'].values), 
                dtype=torch.float32
            ).to(device),
            'pxrd': torch.tensor(
                np.stack(batch_df['pxrd'].values), 
                dtype=torch.float32
            ).to(device),
            'num_atoms': torch.tensor(
                batch_df['num_atoms'].values, 
                dtype=torch.long
            ).to(device),
        }
        
        # 自适应选择引导强度
        if adaptive_mode:
            # 根据样本复杂度（原子数量）动态调整引导强度
            complexities = batch_df['num_atoms'].values / 60.0  # 归一化到[0,1]
            batch_scales = min_scale + (max_scale - min_scale) * complexities
        else:
            batch_scales = [guidance_scale] * len(batch_df)
        
        # 对每个不同的scale值分组处理
        unique_scales = np.unique(batch_scales)
        
        for scale in unique_scales:
            scale_mask = (batch_scales == scale)
            scale_indices = np.where(scale_mask)[0]
            
            if len(scale_indices) == 0:
                continue
            
            # 准备子批次
            sub_batch = {
                'comp': batch['comp'][scale_indices],
                'pxrd': batch['pxrd'][scale_indices],
                'num_atoms': batch['num_atoms'][scale_indices],
            }
            
            # 使用CFG采样
            print(f"使用CFG采样: {scale}")
            with torch.no_grad():
                generated = model.flow.sample(
                    sub_batch, 
                    guidance_scale=float(scale),
                    temperature=1.0,
                    num_steps=50
                )  # [sub_batch_size, 63, 3]
            
            # 反归一化
            generated_denorm = data_normalizer.denormalize_z(generated)
            generated_denorm = generated_denorm.cpu().numpy()
            
            # 处理每个样本
            for i, local_idx in enumerate(scale_indices):
                row = batch_df.iloc[local_idx]
                num_atoms = row.num_atoms
                
                # 提取晶格和分数坐标
                single_output = generated_denorm[i]  # [63, 3]
                lattice_matrix = single_output[:3, :]  # [3, 3]
                frac_coords = single_output[3:3+num_atoms, :]  # [num_atoms, 3]
                frac_coords = np.mod(frac_coords, 1.0)
                
                # 获取元素列表
                species = []
                for j in range(num_atoms):
                    atomic_num = int(row.atom_types[j])
                    if atomic_num > 0:
                        species.append(Element.from_Z(atomic_num))
                
                # 创建Structure对象
                try:
                    lattice = Lattice(lattice_matrix)
                    structure = Structure(
                        lattice=lattice,
                        species=species,
                        coords=frac_coords,
                        coords_are_cartesian=False
                    )
                    structures.append(structure)
                    scales_used.append(scale)
                except Exception as e:
                    # 如果创建失败，使用随机结构
                    structures.append(generate_random_structure(row.to_dict()))
                    scales_used.append(scale)
    
    return structures, scales_used


def generate_crystal_structures_batch(samples_df, model, data_normalizer, batch_size=32):
    """
    批量生成晶体结构（兼容接口，使用默认CFG设置）
    
    Args:
        samples_df: 包含多个样本的DataFrame
        model: 训练好的模型
        data_normalizer: 数据归一化器
        batch_size: 批处理大小
        
    Returns:
        list: Structure对象列表
    """
    structures, _ = generate_crystal_structures_batch_cfg(
        samples_df, model, data_normalizer, 
        batch_size=batch_size,
        guidance_scale=1.5,
        adaptive_mode=False
    )
    return structures


def generate_random_structure(sample):
    """
    生成随机晶体结构（备用方案）
    
    Args:
        sample: 样本数据字典，包含num_atoms和atom_types
        
    Returns:
        Structure对象
    """
    num_atoms = sample['num_atoms']
    
    # 随机晶格参数
    a = np.random.uniform(3, 10)
    b = np.random.uniform(3, 10)
    c = np.random.uniform(3, 10)
    alpha = np.random.uniform(60, 120)
    beta = np.random.uniform(60, 120)
    gamma = np.random.uniform(60, 120)
    
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    
    # 获取实际的原子（排除padding的0）
    atom_types = sample['atom_types']
    species = []
    actual_atom_count = 0
    for i in range(num_atoms):
        atomic_num = int(atom_types[i])
        if atomic_num > 0:
            species.append(Element.from_Z(atomic_num))
            actual_atom_count += 1
    
    # 生成与实际原子数量匹配的分数坐标
    frac_coords = np.random.rand(actual_atom_count, 3)
    
    return Structure(
        lattice=lattice,
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False
    )


# ============== PXRD计算和质量评估 ==============

def calculate_pxrd_worker(structure):
    """
    用于多进程的PXRD计算worker函数（带超时控制）
    
    Args:
        structure: pymatgen Structure对象
        
    Returns:
        np.array: 11501维PXRD强度数组
    """
    import signal
    from contextlib import contextmanager
    
    @contextmanager
    def timeout(seconds):
        """超时上下文管理器"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"PXRD计算超时 ({seconds}秒)")
        
        # 设置超时信号
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    from src.pxrd_simulator import PXRDSimulator
    simulator = PXRDSimulator()
    start_time = time.time()
    
    try:
        # 使用2分钟超时
        with timeout(120):
            x_angles, pxrd_intensities = simulator.simulate(structure)
            elapsed = time.time() - start_time
            if elapsed > 10:  # 只有超过10秒才打印警告
                print(f"⚠️ PXRD计算耗时较长: {elapsed:.2f}秒")
            return pxrd_intensities
    except TimeoutError:
        print(f"❌ PXRD计算超时（>120秒），使用随机值")
        # 返回随机PXRD作为备用
        pxrd_calc = np.random.rand(11501) * 100
        pxrd_calc[pxrd_calc < 10] = 0
        return pxrd_calc
    except Exception as e:
        print(f"⚠️ PXRD计算失败: {str(e)[:100]}")
        # 返回随机PXRD作为备用
        pxrd_calc = np.random.rand(11501) * 100
        pxrd_calc[pxrd_calc < 10] = 0
        return pxrd_calc


def calculate_pxrd_batch(structures, n_workers=4, timeout_seconds=120):
    """
    批量计算PXRD谱（使用多进程并行，带超时控制）
    
    Args:
        structures: Structure对象列表
        n_workers: 并行工作进程数
        timeout_seconds: 每个PXRD计算的超时时间（秒）
    
    Returns:
        list: PXRD数组列表
    """
    from concurrent.futures import as_completed, TimeoutError as FutureTimeoutError
    
    pxrd_results = [None] * len(structures)
    
    print(f"批量计算 {len(structures)} 个PXRD谱 (并行数={n_workers}, 超时={timeout_seconds}秒)")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务
        future_to_index = {}
        for i, structure in enumerate(structures):
            future = executor.submit(calculate_pxrd_worker, structure)
            future_to_index[future] = i
        
        # 收集结果（带进度显示）
        completed = 0
        failed = 0
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                # 等待单个任务完成，设置超时
                result = future.result(timeout=timeout_seconds)
                pxrd_results[index] = result
                completed += 1
            except FutureTimeoutError:
                print(f"  ❌ 样本 {index} PXRD计算超时")
                # 使用随机PXRD作为备用
                pxrd_calc = np.random.rand(11501) * 100
                pxrd_calc[pxrd_calc < 10] = 0
                pxrd_results[index] = pxrd_calc
                failed += 1
            except Exception as e:
                print(f"  ⚠️ 样本 {index} PXRD计算失败: {str(e)[:50]}")
                # 使用随机PXRD作为备用
                pxrd_calc = np.random.rand(11501) * 100
                pxrd_calc[pxrd_calc < 10] = 0
                pxrd_results[index] = pxrd_calc
                failed += 1
            
            # 每10个样本显示一次进度
            if (completed + failed) % 10 == 0:
                print(f"  进度: {completed + failed}/{len(structures)} (成功={completed}, 失败={failed})")
    
    if failed > 0:
        print(f"⚠️ PXRD计算完成: {completed}成功, {failed}失败（使用随机值替代）")
    else:
        print(f"✅ PXRD计算完成: 全部{completed}个成功")
    
    return pxrd_results


def evaluate_structure_quality(structure, observed_pxrd, pxrd_simulator=None):
    """
    评估生成结构的质量
    
    Args:
        structure: 生成的Structure对象
        observed_pxrd: 观测的PXRD谱
        pxrd_simulator: PXRD仿真器实例（可选）
    
    Returns:
        float: RWP值（越小越好）
    """
    if pxrd_simulator is None:
        from src.pxrd_simulator import PXRDSimulator
        pxrd_simulator = PXRDSimulator()
    
    # 计算生成结构的PXRD
    try:
        x_angles, calculated_pxrd = pxrd_simulator.simulate(structure)
    except:
        # 备用：随机PXRD
        calculated_pxrd = np.random.rand(11501) * 100
    
    # 计算RWP
    try:
        from src.metrics import rwp
        rwp_value = rwp(calculated_pxrd, observed_pxrd)
    except ImportError:
        # 备用RWP计算
        diff = calculated_pxrd - observed_pxrd
        weighted_diff = diff * np.sqrt(np.maximum(observed_pxrd, 1e-10))
        rwp_value = np.sqrt(np.sum(weighted_diff**2) / np.sum(observed_pxrd**2 + 1e-10))
    
    return rwp_value


def evaluate_structures_batch(structures, observed_pxrds, n_workers=4, timeout_seconds=120):
    """
    批量评估结构质量（带超时控制）
    
    Args:
        structures: Structure对象列表
        observed_pxrds: 观测PXRD列表
        n_workers: 并行工作进程数
        timeout_seconds: 每个PXRD计算的超时时间（秒）
    
    Returns:
        list: RWP值列表
    """
    print(f"开始批量评估 {len(structures)} 个结构...")
    start_time = time.time()
    
    # 批量计算PXRD（带超时控制）
    calculated_pxrds = calculate_pxrd_batch(
        structures, 
        n_workers=n_workers,
        timeout_seconds=timeout_seconds
    )
    
    # 计算RWP值
    rwp_values = []
    failed_count = 0
    
    for i, (calc_pxrd, obs_pxrd) in enumerate(zip(calculated_pxrds, observed_pxrds)):
        try:
            # 尝试导入metrics模块的rwp函数
            try:
                from src.metrics import rwp
                rwp_value = rwp(calc_pxrd, obs_pxrd)
            except ImportError:
                # 备用RWP计算
                diff = calc_pxrd - obs_pxrd
                weighted_diff = diff * np.sqrt(np.maximum(obs_pxrd, 1e-10))
                rwp_value = np.sqrt(np.sum(weighted_diff**2) / np.sum(obs_pxrd**2 + 1e-10))
            
            rwp_values.append(rwp_value)
            
        except Exception as e:
            # 如果RWP计算失败，使用一个大值表示质量差
            print(f"  ⚠️ 样本 {i} RWP计算失败: {str(e)[:50]}")
            rwp_values.append(999.0)  # 使用大值表示失败
            failed_count += 1
    
    elapsed = time.time() - start_time
    print(f"评估完成: 耗时 {elapsed:.2f}秒")
    if failed_count > 0:
        print(f"  ⚠️ {failed_count} 个样本评估失败（RWP=999）")
    
    return rwp_values


# ============== 流水线处理函数 ==============

def generate_and_evaluate_pipeline(samples_df, model, data_normalizer, observed_pxrds_dict,
                                  batch_size=32, n_workers=4, timeout_seconds=120,
                                  guidance_scale=1.5, adaptive_mode=False):
    """
    流水线处理：生成一批立即评测，避免等待所有批次完成
    
    Args:
        samples_df: 包含样本信息的DataFrame
        model: 训练好的模型
        data_normalizer: 数据归一化器
        observed_pxrds_dict: 样本ID到观测PXRD的字典
        batch_size: 每批生成的大小
        n_workers: PXRD计算并行数
        timeout_seconds: PXRD计算超时时间
        guidance_scale: CFG引导强度
        adaptive_mode: 是否使用自适应引导
        
    Returns:
        dict: {sample_id: (structure, rwp_value)}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import queue
    import threading
    
    results = {}
    num_samples = len(samples_df)
    
    print(f"流水线处理 {num_samples} 个样本")
    print(f"  批大小: {batch_size}")
    print(f"  PXRD并行: {n_workers} 进程")
    print(f"  超时设置: {timeout_seconds}秒")
    
    # 创建队列用于批次间通信
    generation_queue = queue.Queue(maxsize=2)  # 最多缓存2批
    
    def generation_worker():
        """生成线程：负责批量生成结构"""
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_df = samples_df.iloc[batch_start:batch_end]
            
            print(f"\n生成批次 [{batch_start}:{batch_end}]...")
            
            # 生成结构
            if adaptive_mode:
                structures, scales = generate_crystal_structures_batch_cfg(
                    batch_df, model, data_normalizer,
                    batch_size=batch_size,
                    guidance_scale=guidance_scale,
                    adaptive_mode=True
                )
            else:
                structures = generate_crystal_structures_batch(
                    batch_df, model, data_normalizer,
                    batch_size=batch_size
                )
                scales = [guidance_scale] * len(structures)
            
            # 将生成的结构放入队列
            batch_data = {
                'df': batch_df,
                'structures': structures,
                'scales': scales if adaptive_mode else None,
                'batch_idx': (batch_start, batch_end)
            }
            generation_queue.put(batch_data)
            print(f"  ✓ 批次 [{batch_start}:{batch_end}] 生成完成，开始评测...")
        
        # 标记生成完成
        generation_queue.put(None)
    
    def evaluation_worker():
        """评测线程：从队列获取结构并评测"""
        batch_count = 0
        total_evaluated = 0
        
        while True:
            # 从队列获取批次数据
            batch_data = generation_queue.get()
            if batch_data is None:  # 生成完成
                break
            
            batch_df = batch_data['df']
            structures = batch_data['structures']
            batch_start, batch_end = batch_data['batch_idx']
            batch_count += 1
            
            print(f"\n评测批次 {batch_count} [{batch_start}:{batch_end}]...")
            
            # 获取该批次的观测PXRD
            observed_pxrds = []
            for sample_id in batch_df['id']:
                if sample_id in observed_pxrds_dict:
                    observed_pxrds.append(observed_pxrds_dict[sample_id])
                else:
                    # 如果没有观测数据，使用DataFrame中的
                    idx = batch_df[batch_df['id'] == sample_id].index[0]
                    observed_pxrds.append(batch_df.loc[idx, 'pxrd'])
            
            # 批量评测（带超时控制）
            rwp_values = evaluate_structures_batch(
                structures, observed_pxrds,
                n_workers=n_workers,
                timeout_seconds=timeout_seconds
            )
            
            # 保存结果
            for i, (sample_id, structure, rwp) in enumerate(
                zip(batch_df['id'], structures, rwp_values)
            ):
                results[sample_id] = {
                    'structure': structure,
                    'rwp': rwp,
                    'scale': batch_data['scales'][i] if batch_data['scales'] else guidance_scale
                }
                total_evaluated += 1
            
            # 显示进度
            print(f"  ✓ 批次 {batch_count} 评测完成")
            print(f"  总进度: {total_evaluated}/{num_samples} ({total_evaluated/num_samples*100:.1f}%)")
            
            # 显示质量统计
            batch_rwps = [r for r in rwp_values if r < 999]
            if batch_rwps:
                print(f"  批次RWP: 最小={min(batch_rwps):.4f}, 平均={np.mean(batch_rwps):.4f}, 最大={max(batch_rwps):.4f}")
    
    # 启动两个线程
    print("\n启动流水线...")
    generation_thread = threading.Thread(target=generation_worker, name="Generation")
    evaluation_thread = threading.Thread(target=evaluation_worker, name="Evaluation")
    
    generation_thread.start()
    evaluation_thread.start()
    
    # 等待完成
    generation_thread.join()
    evaluation_thread.join()
    
    print(f"\n✅ 流水线完成！共处理 {len(results)} 个样本")
    
    # 统计结果
    all_rwps = [r['rwp'] for r in results.values() if r['rwp'] < 999]
    if all_rwps:
        print(f"整体RWP统计:")
        print(f"  最小: {min(all_rwps):.4f}")
        print(f"  平均: {np.mean(all_rwps):.4f}")
        print(f"  中位数: {np.median(all_rwps):.4f}")
        print(f"  最大: {max(all_rwps):.4f}")
    
    return results


def generate_and_evaluate_batch_simple(samples_df, model, data_normalizer,
                                      batch_size=32, n_workers=4, timeout_seconds=120):
    """
    简化版批量生成和评测（顺序处理每个批次）
    
    Args:
        samples_df: 包含样本信息的DataFrame（必须包含'pxrd'列）
        model: 训练好的模型
        data_normalizer: 数据归一化器
        batch_size: 批处理大小
        n_workers: PXRD计算并行数
        timeout_seconds: PXRD计算超时时间
        
    Returns:
        tuple: (structures列表, rwp_values列表)
    """
    all_structures = []
    all_rwp_values = []
    num_samples = len(samples_df)
    
    print(f"批量处理 {num_samples} 个样本（批大小={batch_size}）")
    
    # 按批次处理
    for batch_idx, batch_start in enumerate(range(0, num_samples, batch_size), 1):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_df = samples_df.iloc[batch_start:batch_end]
        
        print(f"\n批次 {batch_idx}: [{batch_start}:{batch_end}]")
        
        # 1. 生成结构
        print(f"  生成结构...")
        batch_structures = generate_crystal_structures_batch(
            batch_df, model, data_normalizer, batch_size=len(batch_df)
        )
        
        # 2. 立即评测
        print(f"  评测质量...")
        batch_pxrds = batch_df['pxrd'].tolist()
        batch_rwps = evaluate_structures_batch(
            batch_structures, batch_pxrds,
            n_workers=n_workers,
            timeout_seconds=timeout_seconds
        )
        
        # 3. 保存结果
        all_structures.extend(batch_structures)
        all_rwp_values.extend(batch_rwps)
        
        # 4. 显示批次统计
        valid_rwps = [r for r in batch_rwps if r < 999]
        if valid_rwps:
            print(f"  批次统计: RWP均值={np.mean(valid_rwps):.4f}, 最小={min(valid_rwps):.4f}")
        
        print(f"  累计进度: {len(all_structures)}/{num_samples} ({len(all_structures)/num_samples*100:.1f}%)")
    
    print(f"\n✅ 完成！共生成并评测 {len(all_structures)} 个结构")
    
    return all_structures, all_rwp_values


# ============== 后处理函数 ==============

def energy_optimization(structure):
    """
    能量优化（占位实现）
    
    Args:
        structure: 待优化的Structure对象
    
    Returns:
        Structure: 优化后的结构
    """
    # TODO: 实现真实的能量优化（GULP、VASP等）
    
    # 占位：稍微调整晶格参数模拟优化
    new_lattice = structure.lattice.matrix * np.random.uniform(0.98, 1.02)
    optimized = Structure(
        lattice=Lattice(new_lattice),
        species=structure.species,
        coords=structure.frac_coords,
        coords_are_cartesian=False
    )
    
    return optimized


def rietveld_refinement(structure, observed_pxrd, rwp_threshold=0.15):
    """
    Rietveld精修（占位实现）
    
    Args:
        structure: 待精修的Structure对象
        observed_pxrd: 观测的PXRD谱
        rwp_threshold: RWP阈值
    
    Returns:
        tuple: (精修后的Structure, 是否进行了精修)
    """
    # 判断是否需要精修
    current_rwp = evaluate_structure_quality(structure, observed_pxrd)
    needs_refinement = current_rwp > rwp_threshold * 1.5
    
    if not needs_refinement:
        return structure, False
    
    # TODO: 实现真实的Rietveld精修（GSAS-II、TOPAS等）
    
    # 占位：稍微调整原子位置模拟精修
    new_coords = structure.frac_coords + np.random.randn(*structure.frac_coords.shape) * 0.01
    new_coords = np.clip(new_coords, 0, 1)
    
    refined = Structure(
        lattice=structure.lattice,
        species=structure.species,
        coords=new_coords,
        coords_are_cartesian=False
    )
    
    return refined, True


def post_process_structure(structure, observed_pxrd, rwp_threshold=0.15):
    """
    完整的后处理流程
    
    Args:
        structure: 待处理的Structure对象
        observed_pxrd: 观测的PXRD谱
        rwp_threshold: RWP阈值
    
    Returns:
        tuple: (处理后的Structure, 最终RWP值)
    """
    # 1. 能量优化
    optimized = energy_optimization(structure)
    rwp_after_opt = evaluate_structure_quality(optimized, observed_pxrd)
    
    # 2. Rietveld精修（如果需要）
    refined, was_refined = rietveld_refinement(optimized, observed_pxrd, rwp_threshold)
    
    if was_refined:
        rwp_after_refine = evaluate_structure_quality(refined, observed_pxrd)
        return refined, rwp_after_refine
    else:
        return optimized, rwp_after_opt


# ============== 迭代优化控制函数 ==============

def check_termination_conditions(sample_status, start_time, max_runtime, max_attempts):
    """
    检查是否满足终止条件
    
    Args:
        sample_status: 样本状态字典
        start_time: 开始时间戳
        max_runtime: 最大运行时间（秒）
        max_attempts: 单样本最大尝试次数
    
    Returns:
        tuple: (是否终止, 终止原因)
    """
    # 检查运行时间
    elapsed_time = time.time() - start_time
    if elapsed_time > max_runtime:
        return True, f"达到最大运行时间 {max_runtime/3600:.1f} 小时"
    
    # 检查所有样本状态
    all_done = all(
        status['satisfied'] or status['attempts'] >= max_attempts
        for status in sample_status.values()
    )
    
    if all_done:
        satisfied_count = sum(1 for s in sample_status.values() if s['satisfied'])
        return True, f"所有样本处理完成（{satisfied_count}/{len(sample_status)}满足要求）"
    
    return False, None


def get_samples_to_regenerate(sample_status, batch_size=32, max_attempts=10):
    """
    获取需要重新生成的样本
    
    Args:
        sample_status: 样本状态字典
        batch_size: 批次大小
        max_attempts: 单样本最大尝试次数
    
    Returns:
        list: 需要重新生成的样本ID列表
    """
    # 找出未满足要求且未超过尝试次数的样本
    candidates = [
        sample_id for sample_id, status in sample_status.items()
        if not status['satisfied'] and status['attempts'] < max_attempts
    ]
    
    # 按RWP值排序，优先处理质量最差的
    candidates.sort(key=lambda x: sample_status[x]['best_rwp'], reverse=True)
    
    return candidates[:batch_size]


def get_adaptive_cfg_scale(iteration, base_scale=1.5, min_scale=0.8, max_scale=2.5):
    """
    根据迭代次数获取自适应的CFG引导强度
    
    Args:
        iteration: 当前迭代轮次
        base_scale: 基础引导强度
        min_scale: 最小引导强度
        max_scale: 最大引导强度
    
    Returns:
        float: 推荐的CFG引导强度
    """
    if iteration <= 2:
        # 早期：标准引导
        return base_scale
    elif iteration <= 5:
        # 中期：增强引导
        return min(base_scale * 1.5, max_scale)
    else:
        # 后期：降低引导增加多样性
        return max(base_scale * 0.8, min_scale)


# ============== 文件I/O函数 ==============

def update_submission_incrementally(sample_status, data_dir, output_file="submission.csv"):
    """
    增量更新submission.csv文件
    
    Args:
        sample_status: 样本状态字典，包含每个样本的最佳结构
        data_dir: 数据目录路径
        output_file: 输出文件名
    
    Returns:
        pd.DataFrame: submission数据框
    """
    # 准备submission数据
    rows = []
    
    for sample_id, status in sample_status.items():
        try:
            structure = status['best_structure']
            
            if structure is not None:
                # 转换为CIF格式
                cif_str = structure.to(fmt="cif")
            else:
                # 如果还没有结构，创建占位CIF
                cif_str = f"data_{sample_id}\n_cell_length_a 5.0\n_cell_length_b 5.0\n_cell_length_c 5.0\n_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            
            rows.append({
                'ID': sample_id,
                'cif': cif_str
            })
        except Exception as e:
            # 出错时创建占位CIF
            min_cif = f"data_{sample_id}\n_cell_length_a 5.0\n_cell_length_b 5.0\n_cell_length_c 5.0\n_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            rows.append({
                'ID': sample_id,
                'cif': min_cif
            })
    
    # 创建DataFrame
    submission_df = pd.DataFrame(rows)
    
    # 保存为CSV（覆盖原文件）
    submission_df.to_csv(output_file, index=False)
    
    return submission_df


def log_submission_update(iteration, sample_status, submission_file="submission.csv"):
    """
    记录submission更新信息
    
    Args:
        iteration: 当前迭代轮次（0表示初始推理）
        sample_status: 样本状态字典
        submission_file: submission文件路径
    """
    satisfied_count = sum(1 for s in sample_status.values() if s['satisfied'])
    total_count = len(sample_status)
    
    if iteration == 0:
        print(f"\n📝 初始submission.csv已生成")
    else:
        print(f"\n📝 submission.csv已更新 (迭代{iteration})")
    
    print(f"   满足要求: {satisfied_count}/{total_count} ({satisfied_count/total_count*100:.1f}%)")
    
    if os.path.exists(submission_file):
        file_size = os.path.getsize(submission_file) / 1024
        print(f"   文件大小: {file_size:.2f} KB")


def print_final_statistics(sample_status, start_time, rwp_threshold=0.15, max_attempts=10):
    """
    打印最终统计信息
    
    Args:
        sample_status: 样本状态字典
        start_time: 开始时间戳
        rwp_threshold: RWP阈值
        max_attempts: 单样本最大尝试次数
    """
    print("\n" + "="*60)
    print("最终统计")
    print("="*60)
    
    # 计算各项统计指标
    satisfied_samples = [s for s in sample_status.values() if s['satisfied']]
    unsatisfied_samples = [s for s in sample_status.values() if not s['satisfied']]
    
    print(f"\n质量统计:")
    print(f"  满足RWP<{rwp_threshold}: {len(satisfied_samples)}/{len(sample_status)} ({len(satisfied_samples)/len(sample_status)*100:.1f}%)")
    print(f"  未满足要求: {len(unsatisfied_samples)}")
    
    if satisfied_samples:
        satisfied_rwps = [s['best_rwp'] for s in satisfied_samples]
        print(f"\n满足要求样本的RWP:")
        print(f"  最小: {np.min(satisfied_rwps):.4f}")
        print(f"  最大: {np.max(satisfied_rwps):.4f}")
        print(f"  平均: {np.mean(satisfied_rwps):.4f}")
    
    if unsatisfied_samples:
        unsatisfied_rwps = [s['best_rwp'] for s in unsatisfied_samples]
        print(f"\n未满足要求样本的RWP:")
        print(f"  最小: {np.min(unsatisfied_rwps):.4f}")
        print(f"  最大: {np.max(unsatisfied_rwps):.4f}")
        print(f"  平均: {np.mean(unsatisfied_rwps):.4f}")
    
    # 尝试次数统计
    attempts_list = [s['attempts'] for s in sample_status.values()]
    print(f"\n尝试次数统计:")
    print(f"  最少: {np.min(attempts_list)}")
    print(f"  最多: {np.max(attempts_list)}")
    print(f"  平均: {np.mean(attempts_list):.1f}")
    print(f"  达到上限({max_attempts}次): {sum(1 for a in attempts_list if a >= max_attempts)}")
    
    # 运行时间
    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time/3600:.2f}小时")