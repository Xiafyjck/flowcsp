#!/usr/bin/env python
"""
基于Hydra配置管理的晶体结构生成模型训练脚本
支持配置组合、参数覆盖和实验管理
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import warnings

# 导入自定义模块
from src.trainer import CrystalGenerationModule, CrystalGenerationDataModule
from src.ema_callback import EMACallback

# 忽略一些无害的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.parallel')

console = Console()


def print_config(cfg: DictConfig) -> None:
    """美观地打印配置信息"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]                    训练配置                                [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    # 网络配置
    network_table = Table(title="🧠 网络配置", show_header=True, header_style="bold magenta")
    network_table.add_column("参数", style="cyan", width=20)
    network_table.add_column("值", style="green")
    
    network_table.add_row("网络类型", cfg.networks.name)
    for key, value in cfg.networks.config.items():
        network_table.add_row(f"  {key}", str(value))
    
    console.print(network_table)
    console.print()
    
    # 流模型配置
    flow_table = Table(title="🌊 流模型配置", show_header=True, header_style="bold magenta")
    flow_table.add_column("参数", style="cyan", width=20)
    flow_table.add_column("值", style="green")
    
    flow_table.add_row("流模型类型", cfg.flows.name)
    for key, value in cfg.flows.config.items():
        if isinstance(value, float):
            flow_table.add_row(f"  {key}", f"{value:.6f}")
        else:
            flow_table.add_row(f"  {key}", str(value))
    
    console.print(flow_table)
    console.print()
    
    # 数据配置
    data_table = Table(title="📊 数据配置", show_header=True, header_style="bold magenta")
    data_table.add_column("参数", style="cyan", width=20)
    data_table.add_column("值", style="green")
    
    data_table.add_row("训练路径", cfg.data.train_path)
    data_table.add_row("验证路径", cfg.data.val_path)
    data_table.add_row("测试路径", str(cfg.data.test_path))
    data_table.add_row("批次大小", str(cfg.data.batch_size))
    data_table.add_row("工作线程", str(cfg.data.num_workers))
    data_table.add_row("置换增强", f"{cfg.data.augmentation.permutation.enabled} (p={cfg.data.augmentation.permutation.prob})")
    data_table.add_row("SO3增强", f"{cfg.data.augmentation.so3.enabled} (p={cfg.data.augmentation.so3.prob})")
    
    console.print(data_table)
    console.print()
    
    # 训练配置
    trainer_table = Table(title="🎯 训练配置", show_header=True, header_style="bold magenta")
    trainer_table.add_column("参数", style="cyan", width=20)
    trainer_table.add_column("值", style="green")
    
    trainer_table.add_row("最大轮数", str(cfg.trainer.max_epochs))
    trainer_table.add_row("学习率", f"{cfg.optimizer.lr:.2e}")
    trainer_table.add_row("优化器", cfg.optimizer.type)
    trainer_table.add_row("调度器", cfg.scheduler.type)
    trainer_table.add_row("梯度裁剪", str(cfg.trainer.gradient_clip_val))
    trainer_table.add_row("梯度累积", str(cfg.trainer.accumulate_grad_batches))
    trainer_table.add_row("精度", cfg.trainer.precision)
    trainer_table.add_row("设备", f"{cfg.trainer.accelerator} ({cfg.trainer.devices})")
    
    console.print(trainer_table)
    console.print()


def setup_callbacks(cfg: DictConfig, checkpoint_dir: Path, use_wandb: bool = False) -> list:
    """设置训练回调"""
    callbacks = []
    
    # Rich进度条
    callbacks.append(RichProgressBar(refresh_rate=10))
    
    # EMA回调
    if cfg.ema.use_ema:
        ema_callback = EMACallback(decay=cfg.ema.decay, update_every=cfg.ema.update_every)
        callbacks.append(ema_callback)
        console.print(f"✨ EMA已启用 (decay={cfg.ema.decay})")
    
    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='epoch={epoch:03d}-val_loss={val/loss:.4f}',
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # 早停回调
    if cfg.early_stopping.patience > 0:
        early_stopping = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            mode=cfg.early_stopping.mode,
            verbose=True,
        )
        callbacks.append(early_stopping)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_loggers(cfg: DictConfig, logs_dir: Path) -> list:
    """设置日志记录器"""
    loggers = []
    
    # CSV日志记录器
    csv_logger = CSVLogger(
        save_dir=logs_dir,
        name="metrics",
        version=""
    )
    loggers.append(csv_logger)
    
    # WandB日志记录器（如果配置了）
    if cfg.get('wandb', {}).get('enabled', False):
        wandb_logger = WandbLogger(
            project=cfg.wandb.get('project', 'crystal-generation'),
            name=cfg.experiment.name,
            save_dir=logs_dir,
            log_model=cfg.wandb.get('log_model', False),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)
        console.print("📊 WandB日志记录已启用")
    
    return loggers


def validate_paths(cfg: DictConfig) -> bool:
    """验证必要的路径是否存在"""
    errors = []
    
    # 获取原始工作目录（Hydra改变目录前的路径）
    from hydra.core.hydra_config import HydraConfig
    orig_cwd = HydraConfig.get().runtime.cwd
    
    # 检查训练数据（使用原始工作目录的相对路径）
    train_path = Path(orig_cwd) / cfg.data.train_path
    if not train_path.exists():
        errors.append(f"训练缓存未找到: {train_path}")
        errors.append(f"  请运行: python scripts/warmup_cache.py --csv your_train.csv --output_dir {cfg.data.train_path}")
    
    # 检查验证数据
    val_path = Path(orig_cwd) / cfg.data.val_path
    if not val_path.exists():
        errors.append(f"验证缓存未找到: {val_path}")
        errors.append(f"  请运行: python scripts/warmup_cache.py --csv your_val.csv --output_dir {cfg.data.val_path}")
    
    # 检查统计文件
    stats_file = Path(orig_cwd) / cfg.flows.config.stats_file
    if not stats_file.exists():
        errors.append(f"统计文件未找到: {stats_file}")
        errors.append(f"  请运行: python scripts/calculate_lattice_stats.py --cache_dirs {cfg.data.train_path} {cfg.data.val_path} --output {cfg.flows.config.stats_file}")
    
    # 检查MatScholar嵌入文件（如果使用V2网络）
    if cfg.networks.name == "crystal_transformer_v2":
        matscholar_path = cfg.networks.config.get('matscholar_path', '')
        if matscholar_path:
            matscholar_full_path = Path(orig_cwd) / matscholar_path
            if not matscholar_full_path.exists():
                errors.append(f"MatScholar嵌入文件未找到: {matscholar_full_path}")
    
    if errors:
        console.print("\n[bold red]❌ 路径验证失败：[/bold red]")
        for error in errors:
            console.print(f"  {error}")
        return False
    
    return True


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """主训练函数"""
    
    # 设置随机种子
    pl.seed_everything(cfg.experiment.seed, workers=True)
    
    # 生成实验名称
    if cfg.experiment.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.experiment.name = f"{cfg.networks.name}_{cfg.flows.name}_{timestamp}"
    
    # 创建输出目录
    exp_dir = Path.cwd()  # Hydra已经创建了工作目录
    checkpoint_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    config_dir = exp_dir / "config"
    
    # 打印Hydra生成的输出目录
    console.print(f"\n[bold green]实验目录: {exp_dir}[/bold green]")
    
    checkpoint_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)
    
    # 保存完整配置
    config_path = config_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    
    # 打印配置
    print_config(cfg)
    
    # 验证路径
    if not validate_paths(cfg):
        sys.exit(1)
    
    console.print(Panel.fit(
        f"[bold cyan]实验: {cfg.experiment.name}[/bold cyan]\n"
        f"📁 检查点: {checkpoint_dir}\n"
        f"📊 日志: {logs_dir}\n"
        f"⚙️  配置: {config_dir}",
        title="🚀 开始训练",
        border_style="cyan"
    ))
    
    # 获取原始工作目录
    from hydra.core.hydra_config import HydraConfig
    orig_cwd = HydraConfig.get().runtime.cwd
    
    # 准备配置字典，修正路径
    network_config = OmegaConf.to_container(cfg.networks.config, resolve=True)
    flow_config = OmegaConf.to_container(cfg.flows.config, resolve=True)
    
    # 修正文件路径为绝对路径
    if 'stats_file' in flow_config:
        flow_config['stats_file'] = str(Path(orig_cwd) / flow_config['stats_file'])
    if 'matscholar_path' in network_config:
        network_config['matscholar_path'] = str(Path(orig_cwd) / network_config['matscholar_path'])
    
    optimizer_config = {
        'type': cfg.optimizer.type,
        'lr': cfg.optimizer.lr,
        'weight_decay': cfg.optimizer.weight_decay,
        'betas': cfg.optimizer.betas,
    }
    
    scheduler_config = None
    if cfg.scheduler.type != 'none':
        scheduler_config = OmegaConf.to_container(cfg.scheduler, resolve=True)
    
    # 创建模型
    console.print(f"\n📊 创建模型: {cfg.networks.name} + {cfg.flows.name}")
    model = CrystalGenerationModule(
        network_name=cfg.networks.name,
        flow_name=cfg.flows.name,
        network_config=network_config,
        flow_config=flow_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"📈 模型参数: {total_params:,} (可训练: {trainable_params:,})")
    console.print(f"   约 {total_params/1e6:.1f}M 参数")
    
    # 创建数据模块（使用绝对路径）
    # 如果使用diffcsp网络，自动启用GNN数据加载器
    use_gnn_dataloader = (cfg.networks.name == 'diffcsp')
    
    data_module = CrystalGenerationDataModule(
        train_path=str(Path(orig_cwd) / cfg.data.train_path),
        val_path=str(Path(orig_cwd) / cfg.data.val_path),
        test_path=str(Path(orig_cwd) / cfg.data.test_path) if cfg.data.test_path else None,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        augment_permutation=cfg.data.augmentation.permutation.enabled,
        augment_so3=cfg.data.augmentation.so3.enabled,
        augment_prob=cfg.data.augmentation.permutation.prob,
        so3_augment_prob=cfg.data.augmentation.so3.prob,
        pin_memory=cfg.data.get('pin_memory', True),
        persistent_workers=cfg.data.get('persistent_workers', True),
        use_gnn_dataloader=use_gnn_dataloader,
        gnn_radius=cfg.networks.config.get('cutoff', 6.0) if use_gnn_dataloader else 6.0,
        gnn_max_neighbors=cfg.networks.config.get('max_neighbors', 50) if use_gnn_dataloader else 50,
    )
    
    # 设置回调和日志
    use_wandb = cfg.get('wandb', {}).get('enabled', False)
    callbacks = setup_callbacks(cfg, checkpoint_dir, use_wandb)
    loggers = setup_loggers(cfg, logs_dir)
    
    # 准备Trainer参数
    trainer_kwargs = {
        'max_epochs': cfg.trainer.max_epochs,
        'gradient_clip_val': cfg.trainer.gradient_clip_val,
        'accumulate_grad_batches': cfg.trainer.accumulate_grad_batches,
        'val_check_interval': cfg.trainer.val_check_interval,
        'log_every_n_steps': cfg.trainer.log_every_n_steps,
        'callbacks': callbacks,
        'logger': loggers,
        'precision': cfg.trainer.precision,
        'deterministic': cfg.trainer.deterministic,
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
    }
    
    # 设置加速器和设备
    if cfg.trainer.accelerator == 'gpu':
        if cfg.trainer.devices == -1:
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = 'auto'
        else:
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = cfg.trainer.devices
    else:
        trainer_kwargs['accelerator'] = 'cpu'
        trainer_kwargs['devices'] = 1
    
    # 设置分布式策略
    if cfg.trainer.accelerator == 'gpu' and (cfg.trainer.devices == -1 or cfg.trainer.devices > 1):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                console.print(f"🚀 使用DDP策略，{num_gpus}个GPU")
                trainer_kwargs['strategy'] = DDPStrategy(
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                )
            else:
                trainer_kwargs['strategy'] = 'auto'
    
    # 调试选项
    if cfg.trainer.fast_dev_run:
        trainer_kwargs['fast_dev_run'] = True
        console.print("⚡ 快速开发运行模式 (1 batch)")
    
    if cfg.trainer.overfit_batches > 0:
        trainer_kwargs['overfit_batches'] = cfg.trainer.overfit_batches
        console.print(f"🔄 过拟合测试: {cfg.trainer.overfit_batches} batches")
    
    # 性能分析
    if cfg.trainer.profile:
        from pytorch_lightning.profilers import PyTorchProfiler
        trainer_kwargs['profiler'] = PyTorchProfiler(
            dirpath=logs_dir,
            filename="profiler_report",
        )
        console.print("📊 启用PyTorch性能分析")
    
    # 创建Trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # 开始训练
    console.print("\n[bold green]🚀 开始训练...[/bold green]")
    
    # 检查是否有恢复点
    resume_path = cfg.get('resume_from', None)
    if resume_path and Path(resume_path).exists():
        console.print(f"📂 从检查点恢复: {resume_path}")
        trainer.fit(model, data_module, ckpt_path=resume_path)
    else:
        trainer.fit(model, data_module)
    
    # 测试（如果有测试集）
    if cfg.data.test_path:
        console.print("\n🧪 运行测试...")
        trainer.test(model, data_module)
    
    # 打印最终结果
    console.print("\n" + "="*60)
    
    # 查找ModelCheckpoint回调以获取最佳模型路径
    best_model_path = "N/A"
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_model_path = getattr(callback, 'best_model_path', 'N/A')
            break
    
    console.print(Panel.fit(
        f"[bold green]✅ 训练完成！[/bold green]\n\n"
        f"📁 实验输出: {exp_dir}\n"
        f"📊 指标CSV: {logs_dir}/metrics/metrics.csv\n"
        f"🏆 最佳模型: {best_model_path}",
        title="训练总结",
        border_style="green"
    ))


if __name__ == '__main__':
    main()