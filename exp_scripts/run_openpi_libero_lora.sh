#!/bin/bash
###############################################################################
# OpenPI π₀.₅ LIBERO 微调训练 + 推理评测 一键脚本（宿主机直接运行）
#
# 用法: cd /home/dataset-local/zxlei/code && bash run_openpi_libero.sh
#
# 支持两种微调模式（在下方 CONFIG_NAME 中切换）:
#   pi05_libero_lora  = LoRA 微调（<40GB/GPU, 可与其他任务共享 GPU）
#   pi05_libero       = 全量微调（需要 >70GB/GPU 或 FSDP 分片）
#
# 前置条件:
#   1. openpi 仓库已克隆且子模块已初始化, uv 已安装
#   2. 模型权重在 /home/dataset-local/zxlei/modelsrepo/pi05_base
#   3. 数据集在 HF_HOME/lerobot/modified_libero_lerobot_split_padded/libero_10_no_noops/
#
# 环境: Kubernetes Pod, 8x A100-SXM4-80GB, CUDA 12.1, JAX 0.5.3
# 注意: 此环境不支持 Docker (缺少内核能力)，故直接使用 uv run
###############################################################################

set -euo pipefail

# ======================= 配置区 =======================
OPENPI_DIR="/home/dataset-local/zxlei/code/openpi"

# HuggingFace 缓存路径（数据集在 $HF_HOME/lerobot/<REPO_ID> 下）
export HF_HOME="/home/dataset-local/zxlei/.cache/huggingface"

# 训练配置名（对应 config.py 中注册的名称）
# pi05_libero       = 全量微调（需要 >70GB/GPU）
# pi05_libero_lora  = LoRA 微调（<40GB/GPU，适合与其他任务共享 GPU）
CONFIG_NAME="pi05_libero_lora"

# 实验名称（checkpoint 保存子目录）
EXP_NAME="libero_lora_finetune_256"

# 训练步数 & batch size
# LoRA 微调使用较小 batch size 以适应有限显存
TRAIN_STEPS=30000
BATCH_SIZE=256

# GPU 内存占比 (0.0-1.0)
# 当前 GPU 已被其他训练占用约 40GB，仅使用剩余部分
# 0.45 × 80GB ≈ 36GB/GPU，确保不超过 40GB 可用空间
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# FSDP 设置
# LoRA 微调显存需求低，fsdp_devices=1 即可（数据并行）
# 如果仍 OOM 请把 FSDP_DEVICES 改为 8
FSDP_DEVICES=8

# 评测参数
EVAL_TASK_SUITE="libero_10"
# =====================================================

cd "$OPENPI_DIR"

echo "============================================"
echo "  OpenPI π₀.₅ LIBERO 微调 & 评测"
echo "============================================"
echo ""
echo "OpenPI 目录:   $OPENPI_DIR"
echo "HF_HOME:       $HF_HOME"
echo "训练配置:      $CONFIG_NAME"
echo "实验名称:      $EXP_NAME"
echo "GPU 内存占比:  $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "FSDP 设备数:   $FSDP_DEVICES"
echo "训练步数:      $TRAIN_STEPS"
echo "Batch Size:    $BATCH_SIZE"
echo ""

# 验证环境
echo "🔍 验证环境..."
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装"; exit 1
fi
if [ ! -d "$OPENPI_DIR/.venv" ]; then
    echo "❌ openpi .venv 不存在，请先运行 'cd $OPENPI_DIR && uv sync'"; exit 1
fi

# 验证数据集路径
DATASET_PATH="$HF_HOME/lerobot/modified_libero_lerobot_split_padded/libero_10_no_noops"
if [ ! -f "$DATASET_PATH/meta/info.json" ]; then
    echo "❌ 数据集未找到: $DATASET_PATH/meta/info.json"; exit 1
fi
echo "✅ 数据集:    $DATASET_PATH"

# 验证模型权重
CKPT_PATH="/home/dataset-local/zxlei/modelsrepo/pi05_base/params"
if [ ! -d "$CKPT_PATH" ]; then
    echo "❌ 模型权重未找到: $CKPT_PATH"; exit 1
fi
echo "✅ 模型权重:  $CKPT_PATH"

# 验证 JAX GPU
echo "🔍 验证 JAX GPU..."
GPU_COUNT=$(uv run python -c "import jax; print(len(jax.devices()))" 2>/dev/null)
echo "✅ JAX 可见 GPU 数: $GPU_COUNT"
echo ""

###############################################################################
# 阶段 1: 计算归一化统计量
###############################################################################
echo "========== [1/3] 计算归一化统计量 =========="
echo "配置: $CONFIG_NAME"
echo "输出目录: $OPENPI_DIR/assets/$CONFIG_NAME/"
echo ""

# 检查是否已经计算过
NORM_STATS_DIR="$OPENPI_DIR/assets/$CONFIG_NAME/modified_libero_lerobot_split_padded/libero_10_no_noops"
ORIG_STATS_DIR="$OPENPI_DIR/assets/pi05_libero/modified_libero_lerobot_split_padded/libero_10_no_noops"
if [ -f "$NORM_STATS_DIR/norm_stats.json" ]; then
    echo "⏭️  归一化统计量已存在，跳过计算"
    echo "   路径: $NORM_STATS_DIR/norm_stats.json"
elif [ -f "$ORIG_STATS_DIR/norm_stats.json" ] && [ "$CONFIG_NAME" != "pi05_libero" ]; then
    # LoRA 和全量微调使用同一数据集，norm stats 可复用
    echo "📋 从 pi05_libero 复用归一化统计量..."
    mkdir -p "$NORM_STATS_DIR"
    cp "$ORIG_STATS_DIR/norm_stats.json" "$NORM_STATS_DIR/norm_stats.json"
    echo "✅ 已复制: $NORM_STATS_DIR/norm_stats.json"
else
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
    echo "✅ 归一化统计量计算完成"
fi
echo ""

###############################################################################
# 阶段 2: 微调训练
###############################################################################
echo "========== [2/3] 微调训练 =========="
echo "实验: $EXP_NAME"
echo "步数: $TRAIN_STEPS, Batch: $BATCH_SIZE"
echo ""

uv run scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --overwrite \
    --num-train-steps=$TRAIN_STEPS \
    --batch-size=$BATCH_SIZE \
    --fsdp-devices=$FSDP_DEVICES

echo "✅ 微调训练完成"
echo ""
