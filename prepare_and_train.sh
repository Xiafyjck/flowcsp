#!/bin/bash
# 准备数据并训练的完整流程示例

echo "=== 晶体结构生成模型训练流程 ==="
echo ""

# Step 1: 生成训练集缓存
echo "Step 1: 生成训练集内存映射缓存..."
python scripts/warmup_cache.py \
    --csv data/merged_cdvae/train.csv \
    --output_dir data/train_cache \
    --id_col material_id \
    --cif_col cif \
    --workers 32

# Step 2: 生成验证集缓存  
echo ""
echo "Step 2: 生成验证集内存映射缓存..."
python scripts/warmup_cache.py \
    --csv data/merged_cdvae/val.csv \
    --output_dir data/val_cache \
    --id_col material_id \
    --cif_col cif \
    --workers 32

# Step 3: 生成测试集缓存（可选）
echo ""
echo "Step 3: 生成测试集内存映射缓存..."
python scripts/warmup_cache.py \
    --csv data/merged_cdvae/test.csv \
    --output_dir data/test_cache \
    --id_col material_id \
    --cif_col cif \
    --workers 32

# Step 4: 开始训练
echo ""
echo "Step 4: 开始训练..."
python train.py \
    --train_path data/train_cache \
    --val_path data/val_cache \
    --test_path data/test_cache \
    --network transformer \
    --flow cfm \
    --batch_size 32 \
    --num_workers 16 \
    --max_epochs 500 \
    --gpus -1

echo ""
echo "=== 完成 ===="