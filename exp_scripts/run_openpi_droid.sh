#!/bin/bash
###############################################################################
# OpenPI π₀.₅ DROID（RLDS）微调训练脚本（宿主机直接运行）
#
# 用法: cd /data/Embobrain && bash run_openpi_droid.sh
#
# 说明:
#   - 使用 openpi 内置配置: pi05_full_droid_finetune
#   - 数据为 RLDS 格式目录（父目录下含 droid/<version>/）
#   - 训练步数按 EPOCHS 自动换算
###############################################################################

set -euo pipefail

# ======================= 配置区 =======================
OPENPI_DIR="/data/Embobrain/openpi"

# RLDS 数据根目录（父目录；其下应有 droid/0.0.1/）
RLDS_DATA_ROOT="/data/Embobrain/dataset/droid_rlds_split_padded"
DROID_VERSION="0.0.1"

# 训练配置名（openpi/src/openpi/training/config.py）
CONFIG_NAME="pi05_full_droid_finetune"

# 实验名称（checkpoint 保存子目录）
EXP_NAME="droid_pi05_full_finetune_100k"

# 训练参数（采用官方默认配置）
TRAIN_STEPS=100000
BATCH_SIZE=256
FSDP_DEVICES=8
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

cd "$OPENPI_DIR"

echo "============================================"
echo "  OpenPI π₀.₅ DROID RLDS 微调训练"
echo "============================================"
echo "OpenPI 目录:      $OPENPI_DIR"
echo "训练配置:         $CONFIG_NAME"
echo "实验名称:         $EXP_NAME"
echo "RLDS 根目录:      $RLDS_DATA_ROOT"
echo "DROID 版本目录:   $DROID_VERSION"
echo "训练步数:         $TRAIN_STEPS (官方默认)"
echo "Batch Size:       $BATCH_SIZE"
echo "FSDP 设备数:      $FSDP_DEVICES"
echo "GPU 内存占比:     $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo ""

echo "🔍 验证环境..."
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装"; exit 1
fi
if [ ! -d "$OPENPI_DIR/.venv" ]; then
    echo "❌ openpi .venv 不存在，请先运行 'cd $OPENPI_DIR && uv sync'"; exit 1
fi

DATASET_INFO="$RLDS_DATA_ROOT/droid/$DROID_VERSION/dataset_info.json"
if [ ! -f "$DATASET_INFO" ]; then
    echo "❌ 数据集元信息未找到: $DATASET_INFO"; exit 1
fi
echo "✅ 数据集:        $DATASET_INFO"

LOCAL_BASE_CKPT="/data/Embobrain/modelsrepo/pi05_base/params"
if [ -d "$LOCAL_BASE_CKPT" ]; then
    echo "✅ 本地基础权重:  $LOCAL_BASE_CKPT"
else
    echo "⚠️  本地基础权重不存在，训练将回退到配置中的远端 checkpoint"
fi

echo "🔍 验证 JAX GPU..."
GPU_COUNT=$(uv run --group rlds python -c "import jax; print(len(jax.devices()))" 2>/dev/null)
echo "✅ JAX 可见 GPU 数: $GPU_COUNT"
echo ""

###############################################################################
# 阶段 1: 计算归一化统计量
###############################################################################
echo "========== [1/2] 计算归一化统计量 =========="
echo "配置: $CONFIG_NAME"
echo ""

# 检查是否已经计算过
NORM_STATS_DIR="$OPENPI_DIR/assets/$CONFIG_NAME/droid"
if [ -f "$NORM_STATS_DIR/norm_stats.json" ]; then
    echo "⏭️  归一化统计量已存在，跳过计算"
    echo "   路径: $NORM_STATS_DIR/norm_stats.json"
else
    echo "📊 开始计算归一化统计量（RLDS 数据集，可能需要较长时间）..."
    uv run --group rlds scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
    echo "✅ 归一化统计量计算完成"
fi
echo ""

###############################################################################
# 阶段 2: 微调训练
###############################################################################
echo "========== [2/2] DROID 微调训练 =========="
echo ""

uv run --group rlds scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --overwrite \
    --num-train-steps=$TRAIN_STEPS \
    --batch-size=$BATCH_SIZE \
    --fsdp-devices=$FSDP_DEVICES

echo ""
echo "============================================"
echo "  ✅ DROID 训练流程完成！"
echo "============================================"
