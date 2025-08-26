#!/usr/bin/env python
"""
晶体结构生成模型训练脚本
支持命令行参数和DDP双卡训练
"""

import argparse
import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import warnings

# 导入自定义模块
from src.trainer import CrystalGenerationModule, CrystalGenerationDataModule

# 忽略一些无害的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.parallel')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Crystal Structure Generation Model')
    
    # 基本参数
    parser.add_argument('--network', type=str, default='transformer', 
                        choices=['transformer', 'equiformer'],
                        help='Network architecture to use')
    parser.add_argument('--flow', type=str, default='cfm',
                        choices=['cfm', 'meanflow'],
                        help='Flow model to use')
    
    # 数据参数
    parser.add_argument('--train_path', type=str, 
                        default='data/merged_cdvae_train.pkl',
                        help='Path to training data')
    parser.add_argument('--val_path', type=str,
                        default='data/merged_cdvae_val.pkl',
                        help='Path to validation data')
    parser.add_argument('--test_path', type=str,
                        default='data/merged_cdvae_test.pkl',
                        help='Path to test data (optional)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation (permutation)')
    parser.add_argument('--augment_prob', type=float, default=0.5,
                        help='Probability of applying augmentation')
    
    # 网络参数
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for transformer')
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # 流模型参数
    parser.add_argument('--sigma_min', type=float, default=1e-4,
                        help='Minimum noise level for CFM')
    parser.add_argument('--sigma_max', type=float, default=1.0,
                        help='Maximum noise level for CFM')
    parser.add_argument('--loss_weight_lattice', type=float, default=2.0,
                        help='Loss weight for lattice parameters')
    parser.add_argument('--loss_weight_coords', type=float, default=1.0,
                        help='Loss weight for fractional coordinates')
    parser.add_argument('--default_num_steps', type=int, default=50,
                        help='Default number of sampling steps')
    
    # 优化器参数
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['AdamW', 'Adam'],
                        help='Optimizer type')
    
    # 调度器参数
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingWarmup',
                        choices=['none', 'CosineAnnealingLR', 'CosineAnnealingWarmup', 
                                 'StepLR', 'ReduceLROnPlateau'],
                        help='Learning rate scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup steps for CosineAnnealingWarmup')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Max steps for CosineAnnealingWarmup')
    parser.add_argument('--T_max', type=int, default=100,
                        help='T_max for CosineAnnealingLR')
    
    # 训练参数
    parser.add_argument('--max_epochs', type=int, default=350,
                        help='Maximum number of epochs')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Accumulate gradients over k batches')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='How often to check validation set')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                        help='Log metrics every n steps')
    
    # Checkpoint参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_top_k', type=int, default=3,
                        help='Save top k checkpoints')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')
    
    # 硬件参数
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs to use (-1 for all available)')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['32', '16-mixed', 'bf16-mixed'],
                        help='Training precision')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # 调试参数
    parser.add_argument('--fast_dev_run', action='store_true',
                        help='Fast development run (1 batch)')
    parser.add_argument('--overfit_batches', type=float, default=0,
                        help='Overfit on a fraction of batches')
    parser.add_argument('--profile', action='store_true',
                        help='Profile training performance')
    
    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed, workers=True)
    
    # 创建checkpoint目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备网络配置
    network_config = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'max_atoms': 52,
        'pxrd_dim': 11501,
    }
    
    # 准备流模型配置
    flow_config = {
        'sigma_min': args.sigma_min,
        'sigma_max': args.sigma_max,
        'loss_weight_lattice': args.loss_weight_lattice,
        'loss_weight_coords': args.loss_weight_coords,
        'default_num_steps': args.default_num_steps,
        # TODO(human): 添加归一化相关配置
    }
    
    # 准备优化器配置
    optimizer_config = {
        'type': args.optimizer,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'betas': (0.9, 0.999),
    }
    
    # 准备调度器配置
    scheduler_config = None
    if args.scheduler != 'none':
        scheduler_config = {
            'type': args.scheduler,
            'warmup_steps': args.warmup_steps,
            'max_steps': args.max_steps,
            'T_max': args.T_max,
            'eta_min': 1e-6,
            'step_size': 30,
            'gamma': 0.1,
            'factor': 0.5,
            'patience': 10,
            'min_lr': 1e-6,
        }
    
    # 创建模型
    print(f"\n📊 Creating model with {args.network} network and {args.flow} flow...")
    model = CrystalGenerationModule(
        network_name=args.network,
        flow_name=args.flow,
        network_config=network_config,
        flow_config=flow_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )
    
    # 创建数据模块
    print(f"📁 Loading data from {args.train_path}...")
    data_module = CrystalGenerationDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        augment_prob=args.augment_prob,
    )
    
    # 设置回调
    callbacks = []
    
    # Checkpoint回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{args.network}-{args.flow}-{{epoch:03d}}-{{val_loss:.4f}}',
        monitor='val/loss',
        mode='min',
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # 早停回调
    if args.patience > 0:
        early_stopping = EarlyStopping(
            monitor='val/loss',
            patience=args.patience,
            mode='min',
            verbose=True,
        )
        callbacks.append(early_stopping)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # 准备Trainer参数
    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'gradient_clip_val': args.gradient_clip_val,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'val_check_interval': args.val_check_interval,
        'log_every_n_steps': args.log_every_n_steps,
        'callbacks': callbacks,
        'precision': args.precision,
        'deterministic': args.deterministic,
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
    }
    
    # 设置GPU/设备
    if args.gpus == -1:
        # 使用所有可用GPU
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = 'auto'
    elif args.gpus > 0:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = args.gpus
    else:
        trainer_kwargs['accelerator'] = 'cpu'
        trainer_kwargs['devices'] = 1
    
    # 设置分布式策略（双卡DDP）
    if args.gpus > 1 or args.gpus == -1:
        # 检查可用GPU数量
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"🚀 Using DDP strategy with {num_gpus} GPUs")
                trainer_kwargs['strategy'] = DDPStrategy(
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,  # 优化内存使用
                )
            else:
                print(f"💻 Using single GPU")
                trainer_kwargs['strategy'] = 'auto'
        else:
            print("⚠️ No GPU available, using CPU")
            trainer_kwargs['accelerator'] = 'cpu'
            trainer_kwargs['devices'] = 1
    
    # 调试选项
    if args.fast_dev_run:
        trainer_kwargs['fast_dev_run'] = True
        print("⚡ Running fast dev run (1 batch)")
    
    if args.overfit_batches > 0:
        trainer_kwargs['overfit_batches'] = args.overfit_batches
        print(f"🔄 Overfitting on {args.overfit_batches} batches")
    
    if args.profile:
        trainer_kwargs['profiler'] = 'simple'
    
    # 创建Trainer
    print("\n🏋️ Creating trainer...")
    trainer = pl.Trainer(**trainer_kwargs)
    
    # 打印模型信息
    print(f"\n📊 Model Information:")
    print(f"  Network: {args.network}")
    print(f"  Flow: {args.flow}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Scheduler: {args.scheduler}")
    
    # 开始训练
    print("\n🚀 Starting training...")
    if args.resume_from:
        print(f"📂 Resuming from checkpoint: {args.resume_from}")
        trainer.fit(model, data_module, ckpt_path=args.resume_from)
    else:
        trainer.fit(model, data_module)
    
    # 测试（如果有测试集）
    if args.test_path:
        print("\n🧪 Running test...")
        trainer.test(model, data_module)
    
    print("\n✅ Training completed!")
    print(f"📁 Checkpoints saved to: {checkpoint_dir}")
    print(f"🏆 Best model: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()