#!/bin/bash
set -euo pipefail

OPENPI_DIR="/data/Embobrain/openpi"
UV_BIN="$OPENPI_DIR/.venv/bin/uv"

CONFIG_NAME="pi05_libero"
EXP_NAME="${EXP_NAME:-pi05_libero_full_chunk_progress_30k}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
VAL_SPLIT_RATIO="${VAL_SPLIT_RATIO:-0.1}"

PROGRESS_TARGET_MODE="chunk"
PROGRESS_READOUT_MODE="chunk_prefix"
PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.1}"

BASE_INIT_CKPT="/data/Embobrain/modelsrepo/pi05_base/params"
LIBERO_LEROBOT_REPO_ID="modified_libero_lerobot_split_padded/libero_10_no_noops"
LIBERO_LEROBOT_LOCAL_DIR="/data/HF_Cache_dataevo/lerobot/${LIBERO_LEROBOT_REPO_ID}"
LIBERO_NORM_STATS_DIR="$OPENPI_DIR/assets/$CONFIG_NAME/${LIBERO_LEROBOT_REPO_ID}"

# Set these to 1 only when you intentionally want to rebuild cached artifacts.
RECOMPUTE_NORM_STATS="${RECOMPUTE_NORM_STATS:-0}"
OVERWRITE="${OVERWRITE:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export PYTHONPATH="$OPENPI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/data/HF_Cache_dataevo"
export HF_LEROBOT_HOME="/data/HF_Cache_dataevo/lerobot"
export XDG_CACHE_HOME="$OPENPI_DIR/.cache"

JAX_CACHE_ROOT="$OPENPI_DIR/.cache/jax"
AUTOTUNE_CACHE_DIR="$JAX_CACHE_ROOT/xla_gpu_per_fusion_autotune_cache_dir"
COMPILE_CACHE_DIR="$JAX_CACHE_ROOT/compilation_cache"
mkdir -p "$AUTOTUNE_CACHE_DIR" "$COMPILE_CACHE_DIR"
export JAX_COMPILATION_CACHE_DIR="$COMPILE_CACHE_DIR"
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_per_fusion_autotune_cache_dir="* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$AUTOTUNE_CACHE_DIR"
fi

cd "$OPENPI_DIR"

echo "============================================"
echo "  OpenPI pi0.5 LIBERO full finetune + chunk progress"
echo "============================================"
echo "OpenPI dir:              $OPENPI_DIR"
echo "Config:                  $CONFIG_NAME"
echo "Experiment:              $EXP_NAME"
echo "Base checkpoint:         $BASE_INIT_CKPT"
echo "Train steps:             $TRAIN_STEPS"
echo "Batch size:              $BATCH_SIZE"
echo "FSDP devices:            $FSDP_DEVICES"
echo "Num workers:             $NUM_WORKERS"
echo "LeRobot local dir:       $LIBERO_LEROBOT_LOCAL_DIR"
echo "Norm stats dir:          $LIBERO_NORM_STATS_DIR"
echo "Progress target mode:    $PROGRESS_TARGET_MODE"
echo "Progress readout mode:   $PROGRESS_READOUT_MODE"
echo "Progress loss weight:    $PROGRESS_LOSS_WEIGHT"
echo "Validation split:        $VAL_SPLIT_RATIO"
echo "Validation interval:     $VAL_INTERVAL"
echo "Validation batches:      $VAL_NUM_BATCHES"
echo "Log interval:            $LOG_INTERVAL"
echo "Save interval:           $SAVE_INTERVAL"
echo "Keep period:             $KEEP_PERIOD"
echo "Overwrite checkpoint:    $OVERWRITE"
echo "WandB enabled:           $WANDB_ENABLED"
echo "HF home:                 $HF_HOME"
echo "HF lerobot home:         $HF_LEROBOT_HOME"
echo "XLA mem fraction:        $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "CUDA visible devices:    $CUDA_VISIBLE_DEVICES"
echo "XDG cache home:          $XDG_CACHE_HOME"
echo "Compile cache dir:       $COMPILE_CACHE_DIR"
echo "Autotune cache dir:      $AUTOTUNE_CACHE_DIR"
echo ""

if [ ! -x "$UV_BIN" ]; then
  echo "ERROR: uv not found: $UV_BIN"
  exit 1
fi

if [ ! -d "$OPENPI_DIR/.venv" ]; then
  echo "ERROR: $OPENPI_DIR/.venv not found. Run: cd $OPENPI_DIR && uv sync"
  exit 1
fi

if [ ! -d "$BASE_INIT_CKPT" ]; then
  echo "ERROR: base checkpoint not found: $BASE_INIT_CKPT"
  exit 1
fi

if [ ! -f "$LIBERO_LEROBOT_LOCAL_DIR/meta/info.json" ]; then
  echo "ERROR: LeRobot dataset not found: $LIBERO_LEROBOT_LOCAL_DIR/meta/info.json"
  exit 1
fi

echo "[Check] JAX devices..."
GPU_COUNT=$("$UV_BIN" run python -c "import jax; print(len(jax.devices()))" 2>/dev/null)
echo "JAX visible GPU:         $GPU_COUNT"
if [ "$GPU_COUNT" -lt "$FSDP_DEVICES" ]; then
  echo "ERROR: JAX sees $GPU_COUNT devices, but FSDP_DEVICES=$FSDP_DEVICES"
  exit 1
fi
echo ""

if [ "$RECOMPUTE_NORM_STATS" = "1" ]; then
  echo "========== Recompute LIBERO norm stats =========="
  "$UV_BIN" run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
  echo ""
fi

if [ ! -f "$LIBERO_NORM_STATS_DIR/norm_stats.json" ]; then
  echo "ERROR: norm stats missing: $LIBERO_NORM_STATS_DIR/norm_stats.json"
  echo "Set RECOMPUTE_NORM_STATS=1 to recompute it."
  exit 1
fi

TRAIN_FLAGS=()
if [ "$OVERWRITE" = "1" ]; then
  TRAIN_FLAGS+=(--overwrite)
else
  TRAIN_FLAGS+=(--resume)
fi

if [ "$WANDB_ENABLED" = "1" ]; then
  TRAIN_FLAGS+=(--wandb-enabled)
else
  TRAIN_FLAGS+=(--no-wandb-enabled)
fi

echo "========== Train LIBERO full finetune + chunk-prefix progress =========="
"$UV_BIN" run scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  "${TRAIN_FLAGS[@]}" \
  --num-train-steps="$TRAIN_STEPS" \
  --batch-size="$BATCH_SIZE" \
  --fsdp-devices="$FSDP_DEVICES" \
  --num-workers="$NUM_WORKERS" \
  --log-interval="$LOG_INTERVAL" \
  --save-interval="$SAVE_INTERVAL" \
  --keep-period="$KEEP_PERIOD" \
  --val-interval="$VAL_INTERVAL" \
  --val-num-batches="$VAL_NUM_BATCHES" \
  --val-split-ratio="$VAL_SPLIT_RATIO" \
  --progress-target-mode="$PROGRESS_TARGET_MODE" \
  --progress-readout-mode="$PROGRESS_READOUT_MODE" \
  --progress-loss-weight="$PROGRESS_LOSS_WEIGHT"

echo ""
echo "============================================"
echo "  pi0.5 full finetune + chunk progress done"
echo "============================================"
