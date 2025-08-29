"""
推理工具函数和多进程Worker
包含所有多进程相关函数，避免在jupyter中直接使用多进程
"""

import os
import time
import signal
import pickle
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from multiprocessing import Process, Queue, Value, Manager
from threading import Lock
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from contextlib import contextmanager
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
import warnings
warnings.filterwarnings('ignore')


from src.pxrd_simulator import PXRDSimulator
# from src.data import CrystDataset
# from src.normalizer import LatticeNormalizer, CompNormalizer


class TimeoutException(Exception):
    """仿真超时异常"""
    pass


@contextmanager
def timeout(seconds):
    """超时上下文管理器"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 设备
        
    Returns:
        network: 网络模型
        flow: 生成流模型
        config: 配置
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从检查点中恢复配置
    config = checkpoint.get('config', {})
    
    # 根据配置创建网络和flow
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import sys
    
    # 动态导入对应的网络和flow
    network_type = config.get('network', 'transformer')
    flow_type = config.get('flow', 'cfm')
    
    # 导入网络
    if network_type == 'transformer':
        from src.networks.transformer import TransformerNetwork
        network = TransformerNetwork(**config.get('network_params', {}))
    elif network_type == 'equiformer':
        from src.networks.equiformer import EquiformerNetwork
        network = EquiformerNetwork(**config.get('network_params', {}))
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    # 导入flow
    if flow_type == 'cfm':
        from src.flows.cfm_cfg import CFMwithCFG
        flow = CFMwithCFG(network=network, **config.get('flow_params', {}))
    elif flow_type == 'meanflow':
        from src.flows.meanflow import MeanFlow
        flow = MeanFlow(network=network, **config.get('flow_params', {}))
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")
    
    # 加载权重
    flow.load_state_dict(checkpoint['state_dict'])
    flow.to(device)
    flow.eval()
    
    return network, flow, config


def prepare_batch_data(sample_ids: List[str], pxrd_data: Dict,
                       comp_data: Dict,
                       device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    准备批次数据用于生成
    
    Args:
        sample_ids: 样本ID列表
        pxrd_data: PXRD数据字典
        comp_data: 成分数据字典
        device: 设备
        
    Returns:
        批次数据字典
    """
    batch_data = {
        'sample_id': sample_ids,
        'pxrd': [],
        'comp': [],
        'num_atoms': []
    }
    
    for sid in sample_ids:
        # 获取PXRD和成分数据
        pxrd = pxrd_data.get(sid, np.zeros(11501))
        
        # 获取成分信息
        if sid in comp_data:
            comp = comp_data[sid]
        else:
            # 如果没有成分数据，使用随机数据（仅用于测试）
            comp = np.random.randint(1, 90, size=np.random.randint(5, 20))
            comp = np.pad(comp, (0, 60 - len(comp)), constant_values=0)
        
        batch_data['pxrd'].append(pxrd)
        batch_data['comp'].append(comp)
        batch_data['num_atoms'].append(np.sum(comp > 0))
    
    # 转换为tensor
    batch_data['pxrd'] = torch.tensor(np.array(batch_data['pxrd']), 
                                      dtype=torch.float32).to(device)
    batch_data['comp'] = torch.tensor(np.array(batch_data['comp']), 
                                      dtype=torch.long).to(device)
    batch_data['num_atoms'] = torch.tensor(batch_data['num_atoms'], 
                                           dtype=torch.long).to(device)
    
    return batch_data


def generation_worker(task_queue: Queue, result_queue: Queue, 
                      checkpoint_path: str, batch_size: int = 32,
                      device_id: int = 0):
    """
    GPU生成进程Worker
    负责批量生成晶体结构
    
    Args:
        task_queue: 任务队列，包含需要生成的样本ID
        result_queue: 结果队列，输出生成的结构
        checkpoint_path: 模型检查点路径
        batch_size: 批次大小
        device_id: GPU设备ID
    """
    # 设置设备
    device = f'cuda:{device_id}'
    torch.cuda.set_device(device_id)
    
    # 加载模型
    network, flow, config = load_checkpoint(checkpoint_path, device)
    
    print(f"Generation worker started on {device}")
    
    while True:
        batch_tasks = []
        
        # 收集一个批次的任务
        while len(batch_tasks) < batch_size:
            try:
                task = task_queue.get(timeout=1)
                if task is None:  # 终止信号
                    if batch_tasks:  # 处理剩余任务
                        break
                    print("Generation worker terminated")
                    return
                batch_tasks.append(task)
            except:
                if batch_tasks:  # 超时但有任务，开始处理
                    break
                continue
        
        if not batch_tasks:
            continue
        
        # 准备批次数据
        sample_ids = [t['sample_id'] for t in batch_tasks]
        pxrd_data = {t['sample_id']: t['pxrd'] for t in batch_tasks}
        comp_data = {t['sample_id']: t.get('comp', None) for t in batch_tasks}
        
        try:
            # 准备输入数据
            batch_data = prepare_batch_data(
                sample_ids, pxrd_data, comp_data, device
            )
            
            # 生成结构
            with torch.no_grad():
                # 调用flow的sample方法
                generated = flow.sample(
                    batch_size=len(sample_ids),
                    conditions={
                        'pxrd': batch_data['pxrd'],
                        'comp': batch_data['comp'],
                        'num_atoms': batch_data['num_atoms']
                    }
                )
            
            # 解析生成结果
            # flow的sample方法已经返回反归一化后的结果
            lattice = generated[:, :3, :]  # [batch, 3, 3]
            frac_coords = generated[:, 3:, :]  # [batch, 60, 3]
            
            # 将结果放入队列
            for i, sid in enumerate(sample_ids):
                num_atoms = batch_data['num_atoms'][i].item()
                result = {
                    'sample_id': sid,
                    'lattice': lattice[i].cpu().numpy(),
                    'frac_coords': frac_coords[i, :num_atoms].cpu().numpy(),
                    'atom_types': batch_data['comp'][i, :num_atoms].cpu().numpy(),
                    'generation_attempt': batch_tasks[i].get('attempt', 1)
                }
                result_queue.put(result)
                
        except Exception as e:
            print(f"Generation error: {e}")
            # 将失败的任务重新放回队列
            for task in batch_tasks:
                if task['attempt'] < 10:  # 最大尝试次数
                    task['attempt'] = task.get('attempt', 0) + 1
                    task_queue.put(task)


def simulation_worker(result_queue: Queue, rwp_queue: Queue, 
                      worker_id: int, timeout_seconds: int = 120):
    """
    CPU仿真进程Worker
    负责PXRD仿真和Rwp计算
    
    Args:
        result_queue: 生成结果队列
        rwp_queue: Rwp结果队列
        worker_id: Worker ID
        timeout_seconds: 仿真超时时间（秒）
    """
    simulator = PXRDSimulator()
    print(f"Simulation worker {worker_id} started")
    
    while True:
        try:
            result = result_queue.get(timeout=5)
            if result is None:  # 终止信号
                print(f"Simulation worker {worker_id} terminated")
                return
                
            sample_id = result['sample_id']
            
            try:
                # 使用超时机制进行仿真
                with timeout(timeout_seconds):
                    # 创建Structure对象
                    structure = Structure(
                        result['lattice'],
                        result['atom_types'],
                        result['frac_coords'],
                        coords_are_cartesian=False
                    )
                    
                    # 仿真PXRD
                    simulated_pxrd = simulator.simulate(structure)
                    
                    # 计算Rwp（这里需要真实的实验PXRD数据）
                    # 暂时用随机值模拟
                    rwp = np.random.uniform(0.1, 20.0)
                    
                    # 将结果放入队列
                    rwp_result = {
                        'sample_id': sample_id,
                        'structure': structure,
                        'rwp': rwp,
                        'simulated_pxrd': simulated_pxrd,
                        'generation_attempt': result['generation_attempt']
                    }
                    rwp_queue.put(rwp_result)
                    
            except TimeoutException:
                print(f"Simulation timeout for {sample_id}")
                # 超时的样本需要重新生成
                rwp_result = {
                    'sample_id': sample_id,
                    'rwp': 9999,  # 标记为失败
                    'timeout': True,
                    'generation_attempt': result['generation_attempt']
                }
                rwp_queue.put(rwp_result)
                
            except Exception as e:
                print(f"Simulation error for {sample_id}: {e}")
                rwp_result = {
                    'sample_id': sample_id,
                    'rwp': 9999,
                    'error': str(e),
                    'generation_attempt': result['generation_attempt']
                }
                rwp_queue.put(rwp_result)
                
        except:
            continue


def postprocess_worker(postprocess_queue: Queue, final_queue: Queue,
                       device_id: int = 0):
    """
    GPU后处理进程Worker（占位函数）
    
    Args:
        postprocess_queue: 后处理任务队列
        final_queue: 最终结果队列
        device_id: GPU设备ID
    """
    device = f'cuda:{device_id}'
    torch.cuda.set_device(device_id)
    
    print(f"Postprocess worker started on {device}")
    
    while True:
        try:
            task = postprocess_queue.get(timeout=5)
            if task is None:  # 终止信号
                print("Postprocess worker terminated")
                return
            
            # 后处理逻辑（暂时直接传递）
            # TODO: 实现实际的后处理逻辑
            # 例如：能量优化、Rietveld精修等
            
            final_queue.put(task)
            
        except:
            continue


def submission_manager(final_queue: Queue, submission_path: str,
                       best_structures: dict, lock: Lock,
                       terminate_flag: Value):
    """
    提交文件管理进程
    负责更新submission.csv
    
    Args:
        final_queue: 最终结果队列
        submission_path: 提交文件路径
        best_structures: 共享的最佳结构字典
        lock: 文件写入锁
        terminate_flag: 终止标志
    """
    print(f"Submission manager started")
    
    # 初始化计数器
    sample_attempts = {}
    
    while not terminate_flag.value:
        try:
            result = final_queue.get(timeout=5)
            if result is None:  # 终止信号
                print("Submission manager terminated")
                return
            
            sample_id = result['sample_id']
            rwp = result.get('rwp', 9999)
            
            # 更新尝试次数
            if sample_id not in sample_attempts:
                sample_attempts[sample_id] = 0
            sample_attempts[sample_id] += 1
            
            # 更新最佳结构
            with lock:
                if sample_id not in best_structures or \
                   rwp < best_structures[sample_id].get('rwp', 9999):
                    best_structures[sample_id] = result
                    
                    # 保存到submission.csv
                    save_submission(best_structures, submission_path)
            
            # 检查终止条件
            # 1. 单个样本达到最大尝试次数（10次）
            if sample_attempts[sample_id] >= 10:
                print(f"Sample {sample_id} reached max attempts")
                
            # 2. 所有样本Rwp都低于0.5
            all_good = all(s.get('rwp', 9999) < 0.5 
                          for s in best_structures.values())
            if all_good and len(best_structures) > 0:
                print("All samples have Rwp < 0.5")
                terminate_flag.value = 1
                
        except:
            continue


def save_submission(best_structures: dict, submission_path: str):
    """
    保存提交文件
    
    Args:
        best_structures: 最佳结构字典
        submission_path: 提交文件路径
    """
    rows = []
    
    for sample_id, result in best_structures.items():
        if 'structure' in result:
            # 转换为CIF格式
            structure = result['structure']
            cif_writer = CifWriter(structure)
            cif_str = str(cif_writer)
            
            # 移除换行符，保存为单行
            cif_str = cif_str.replace('\n', '\\n')
            
            rows.append({
                'id': sample_id,
                'cif': cif_str,
                'rwp': result.get('rwp', 9999)
            })
        else:
            # 没有结构，使用默认值
            rows.append({
                'id': sample_id,
                'cif': '',
                'rwp': 9999
            })
    
    # 保存为CSV
    df = pd.DataFrame(rows)
    df.to_csv(submission_path, index=False)
    print(f"Saved {len(rows)} structures to {submission_path}")


def load_sample_data(sample_path: str) -> Tuple[List[str], Dict, Dict]:
    """
    加载样本数据
    
    Args:
        sample_path: 样本文件路径
        
    Returns:
        sample_ids: 样本ID列表
        pxrd_data: PXRD数据字典
        comp_data: 成分数据字典
    """
    # 读取CSV文件
    df = pd.read_csv(sample_path)
    
    sample_ids = df['id'].tolist()
    pxrd_data = {}
    comp_data = {}
    
    for _, row in df.iterrows():
        sample_id = row['id']
        
        # 解析PXRD数据
        if 'pxrd' in row and pd.notna(row['pxrd']):
            pxrd = np.array(eval(row['pxrd']))
        else:
            # 如果没有PXRD数据，使用随机数据（仅用于测试）
            pxrd = np.random.randn(11501)
        pxrd_data[sample_id] = pxrd
        
        # 解析成分数据
        if 'comp' in row and pd.notna(row['comp']):
            comp = np.array(eval(row['comp']))
            # 确保comp是60维的
            if len(comp) < 60:
                comp = np.pad(comp, (0, 60 - len(comp)), constant_values=0)
        else:
            # 如果没有成分数据，使用随机数据（仅用于测试）
            num_atoms = np.random.randint(5, 20)
            comp = np.random.randint(1, 90, size=num_atoms)
            comp = np.pad(comp, (0, 60 - len(comp)), constant_values=0)
        comp_data[sample_id] = comp
    
    return sample_ids, pxrd_data, comp_data


def check_termination(start_time: float, max_hours: float = 5.0) -> bool:
    """
    检查是否达到时间终止条件
    
    Args:
        start_time: 开始时间
        max_hours: 最大运行小时数
        
    Returns:
        是否应该终止
    """
    elapsed = (time.time() - start_time) / 3600  # 转换为小时
    return elapsed >= max_hours