#!/bin/bash
set -euo pipefail

OPENPI_DIR="/data/Embobrain/openpi"
UV_BIN="$OPENPI_DIR/.venv/bin/uv"
LIBERO_REVISED_RLDS_DIR="${LIBERO_REVISED_RLDS_DIR:-/data/Embobrain/dataset_revised/libero/libero_10_split_padded}"
LIBERO_RLDS_DATASET_NAME="libero_10_no_noops"

# We reuse the same local LeRobot repo id/config family as pi05_libero_lora so train.py
# can consume the rebuilt dataset and matching norm stats without extra code changes.
LIBERO_LEROBOT_REPO_ID="modified_libero_lerobot_split_padded/libero_10_no_noops"
LIBERO_LEROBOT_LOCAL_DIR="/data/HF_Cache_dataevo/lerobot/${LIBERO_LEROBOT_REPO_ID}"
LIBERO_NORM_STATS_DIR="/data/Embobrain/openpi/assets/pi05_libero_lora/${LIBERO_LEROBOT_REPO_ID}"
BASE_INIT_CKPT="/data/Embobrain/modelsrepo/pi05_base/params"

CONFIG_NAME="pi05_libero_lora"
EXP_NAME="${EXP_NAME:-pi05_libero_lora_chunk_progress_prefix_from_base_revised_rlds}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"

LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
VAL_SPLIT_RATIO="${VAL_SPLIT_RATIO:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

PROGRESS_TARGET_MODE="chunk"
PROGRESS_READOUT_MODE="chunk_prefix"
PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.1}"

SKIP_REBUILD="${SKIP_REBUILD:-0}"
SKIP_NORM_STATS="${SKIP_NORM_STATS:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export PYTHONPATH="$OPENPI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/data/HF_Cache_dataevo"
export HF_LEROBOT_HOME="/data/HF_Cache_dataevo/lerobot"

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
echo "  OpenPI pi0.5 LIBERO chunk-progress from pi05_base"
echo "============================================"
echo "OpenPI dir:              $OPENPI_DIR"
echo "Config:                  $CONFIG_NAME"
echo "Experiment:              $EXP_NAME"
echo "Base checkpoint:         $BASE_INIT_CKPT"
echo "Train steps:             $TRAIN_STEPS"
echo "Batch size:              $BATCH_SIZE"
echo "FSDP devices:            $FSDP_DEVICES"
echo "Num workers:             $NUM_WORKERS"
echo "Revised RLDS dir:        $LIBERO_REVISED_RLDS_DIR"
echo "RLDS dataset name:       $LIBERO_RLDS_DATASET_NAME"
echo "LeRobot repo id:         $LIBERO_LEROBOT_REPO_ID"
echo "LeRobot local dir:       $LIBERO_LEROBOT_LOCAL_DIR"
echo "Norm stats dir:          $LIBERO_NORM_STATS_DIR"
echo "Progress target mode:    $PROGRESS_TARGET_MODE"
echo "Progress readout mode:   $PROGRESS_READOUT_MODE"
echo "Progress loss weight:    $PROGRESS_LOSS_WEIGHT"
echo "Skip rebuild:            $SKIP_REBUILD"
echo "Skip norm stats:         $SKIP_NORM_STATS"
echo "HF home:                 $HF_HOME"
echo "HF lerobot home:         $HF_LEROBOT_HOME"
echo "XDG cache home:          $XDG_CACHE_HOME"
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

if [ ! -d "$LIBERO_REVISED_RLDS_DIR/$LIBERO_RLDS_DATASET_NAME" ]; then
  echo "ERROR: revised LIBERO RLDS dataset not found: $LIBERO_REVISED_RLDS_DIR/$LIBERO_RLDS_DATASET_NAME"
  exit 1
fi

echo "[Check] JAX devices..."
GPU_COUNT=$("$UV_BIN" run python -c "import jax; print(len(jax.devices()))" 2>/dev/null)
echo "JAX visible GPU:         $GPU_COUNT"
echo ""

###############################################################################
# Stage 1/3: rebuild local LeRobot dataset from revised RLDS
###############################################################################
echo "========== [1/3] Rebuild LIBERO LeRobot dataset =========="
if [ "$SKIP_REBUILD" = "1" ]; then
  echo "Skipping LeRobot rebuild because SKIP_REBUILD=1"
else
  if [ -d "$LIBERO_LEROBOT_LOCAL_DIR" ]; then
    echo "Removing existing LeRobot dataset: $LIBERO_LEROBOT_LOCAL_DIR"
    rm -rf "$LIBERO_LEROBOT_LOCAL_DIR"
  fi

  "$UV_BIN" run --group rlds python convert_libero10_to_lerobot.py \
    --data-dir "$LIBERO_REVISED_RLDS_DIR" \
    --output-dir "$LIBERO_LEROBOT_REPO_ID"
fi

if [ ! -d "$LIBERO_LEROBOT_LOCAL_DIR/meta" ]; then
  echo "ERROR: LeRobot dataset missing after stage 1: $LIBERO_LEROBOT_LOCAL_DIR"
  exit 1
fi
echo "Done: LeRobot dataset ready"
echo ""

###############################################################################
# Stage 2/3: recompute norm stats for the rebuilt LeRobot dataset
###############################################################################
echo "========== [2/3] Recompute LIBERO norm stats =========="
if [ "$SKIP_NORM_STATS" = "1" ]; then
  echo "Skipping norm stats recompute because SKIP_NORM_STATS=1"
else
  if [ -d "$LIBERO_NORM_STATS_DIR" ]; then
    echo "Removing existing norm stats dir: $LIBERO_NORM_STATS_DIR"
    rm -rf "$LIBERO_NORM_STATS_DIR"
  fi

  "$UV_BIN" run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
fi

if [ ! -f "$LIBERO_NORM_STATS_DIR/norm_stats.json" ]; then
  echo "ERROR: norm stats missing after stage 2: $LIBERO_NORM_STATS_DIR/norm_stats.json"
  exit 1
fi
echo "Done: norm stats ready"
echo ""

###############################################################################
# Stage 3/3: train LoRA trunk + progress head jointly from pi05_base
###############################################################################
echo "========== [3/3] Train LIBERO LoRA + Chunk Prefix Progress from pi05_base =========="
if [ "$WANDB_ENABLED" = "1" ]; then
  WANDB_FLAG="--wandb-enabled"
else
  WANDB_FLAG=""
fi

"$UV_BIN" run scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  --overwrite \
  --num-train-steps="$TRAIN_STEPS" \
  --batch-size="$BATCH_SIZE" \
  --fsdp-devices="$FSDP_DEVICES" \
  --num-workers="$NUM_WORKERS" \
  --log-interval="$LOG_INTERVAL" \
  --save-interval="$SAVE_INTERVAL" \
  --val-interval="$VAL_INTERVAL" \
  --val-num-batches="$VAL_NUM_BATCHES" \
  --val-split-ratio="$VAL_SPLIT_RATIO" \
  --progress-target-mode="$PROGRESS_TARGET_MODE" \
  --progress-readout-mode="$PROGRESS_READOUT_MODE" \
  --progress-loss-weight="$PROGRESS_LOSS_WEIGHT" \
  $WANDB_FLAG

echo ""
echo "============================================"
echo "  LIBERO LoRA + chunk-prefix progress done"
echo "============================================"
