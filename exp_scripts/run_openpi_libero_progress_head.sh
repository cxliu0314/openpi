#!/bin/bash
###############################################################################
# OpenPI pi0.5 LIBERO progress-head-only finetune script
#
# Usage:
#   cd /data/Embobrain && bash run_openpi_libero_progress_head.sh
#
# Fixed configuration script (no external env overrides).
###############################################################################

set -euo pipefail

OPENPI_DIR="/data/Embobrain/openpi"
CONFIG_NAME="pi05_libero_lora_progress_head"
EXP_NAME="pi05_libero_lora_progress_head_low_noise_action"

# Base checkpoint (without progress head), step directory -> params
INIT_CKPT_BASE="/data/Embobrain/openpi/checkpoints/pi05_libero_lora/libero_lora_finetune_256/29999"
INIT_CKPT="$INIT_CKPT_BASE/params"

TRAIN_STEPS="30000"
BATCH_SIZE="64"
FSDP_DEVICES="8"
NUM_WORKERS="4"

LOG_INTERVAL="100"
SAVE_INTERVAL="1000"
VAL_INTERVAL="1000"
VAL_NUM_BATCHES="20"
VAL_SPLIT_RATIO="0.1"
PROGRESS_LOSS_WEIGHT="1"
PROGRESS_READOUT_MODE="low_noise_action"

WANDB_ENABLED="1"
PROGRESS_INPUT="first_action_token_at_low_noise_t"
PROGRESS_TARGET_RULE="frame_index / max(episode_len - 1, 1)"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.90"
export PYTHONPATH="$OPENPI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/data/HF_Cache_dataevo"
export HF_LEROBOT_HOME="/data/HF_Cache_dataevo/lerobot"

# Route JAX/XLA caches to /data to avoid filling root filesystem.
JAX_CACHE_ROOT="$OPENPI_DIR/.cache/jax"
AUTOTUNE_CACHE_DIR="$JAX_CACHE_ROOT/xla_gpu_per_fusion_autotune_cache_dir"
COMPILE_CACHE_DIR="$JAX_CACHE_ROOT/compilation_cache"

mkdir -p "$AUTOTUNE_CACHE_DIR" "$COMPILE_CACHE_DIR"
export XDG_CACHE_HOME="$OPENPI_DIR/.cache"
export JAX_COMPILATION_CACHE_DIR="$COMPILE_CACHE_DIR"
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_per_fusion_autotune_cache_dir="* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$AUTOTUNE_CACHE_DIR"
fi

cd "$OPENPI_DIR"

echo "============================================"
echo "  OpenPI pi0.5 LIBERO progress-head-only train"
echo "============================================"
echo "OpenPI dir:            $OPENPI_DIR"
echo "Config:                $CONFIG_NAME"
echo "Experiment:            $EXP_NAME"
echo "Init checkpoint:       $INIT_CKPT"
echo "Train steps:           $TRAIN_STEPS"
echo "Batch size:            $BATCH_SIZE"
echo "FSDP devices:          $FSDP_DEVICES"
echo "Num workers:           $NUM_WORKERS"
echo "Progress mode:         $PROGRESS_READOUT_MODE"
echo "Progress input:        $PROGRESS_INPUT"
echo "Progress target rule:  $PROGRESS_TARGET_RULE"
echo "Progress loss weight:  $PROGRESS_LOSS_WEIGHT"
echo "HF home:               $HF_HOME"
echo "HF lerobot home:       $HF_LEROBOT_HOME"
echo "XDG cache home:        $XDG_CACHE_HOME"
echo "Autotune cache dir:    $AUTOTUNE_CACHE_DIR"
echo ""

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed"
  exit 1
fi

if [ ! -d "$OPENPI_DIR/.venv" ]; then
  echo "ERROR: $OPENPI_DIR/.venv not found. Run: cd $OPENPI_DIR && uv sync"
  exit 1
fi

if [ ! -d "$INIT_CKPT" ]; then
  echo "ERROR: init checkpoint not found: $INIT_CKPT"
  exit 1
fi

if [ "$WANDB_ENABLED" = "1" ]; then
  WANDB_FLAG="--wandb-enabled"
else
  WANDB_FLAG=""
fi

uv run python scripts/train_progress_head_only.py \
  --config-name "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --init-ckpt "$INIT_CKPT" \
  --num-train-steps "$TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --log-interval "$LOG_INTERVAL" \
  --save-interval "$SAVE_INTERVAL" \
  --val-interval "$VAL_INTERVAL" \
  --val-num-batches "$VAL_NUM_BATCHES" \
  --val-split-ratio "$VAL_SPLIT_RATIO" \
  --progress-loss-weight "$PROGRESS_LOSS_WEIGHT" \
  --progress-readout-mode "$PROGRESS_READOUT_MODE" \
  --overwrite \
  $WANDB_FLAG

echo ""
echo "============================================"
echo "  Progress head only training finished"
echo "============================================"
