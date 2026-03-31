#!/usr/bin/env bash
set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-/data/Embobrain/openpi}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_lora_progress_head}"
OUTPUT_CONFIG_NAME="${OUTPUT_CONFIG_NAME:-pi05_libero_lora_progress_head_chunk}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/data/Embobrain/openpi/checkpoints}"
INIT_CKPT="${INIT_CKPT:-/data/Embobrain/openpi/checkpoints/pi05_libero_lora/libero_lora_finetune/29999/params}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
VAL_SPLIT_RATIO="${VAL_SPLIT_RATIO:-0.1}"

PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.1}"
PROGRESS_RELATIVE_LOSS_WEIGHT="${PROGRESS_RELATIVE_LOSS_WEIGHT:-0.25}"

PEAK_LR="${PEAK_LR:-2e-4}"
DECAY_LR="${DECAY_LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
DECAY_STEPS="${DECAY_STEPS:-30000}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export PYTHONPATH="$OPENPI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/data/HF_Cache_dataevo}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/HF_Cache_dataevo/lerobot}"

JAX_CACHE_ROOT="$OPENPI_DIR/.cache/jax"
AUTOTUNE_CACHE_DIR="$JAX_CACHE_ROOT/xla_gpu_per_fusion_autotune_cache_dir"
COMPILE_CACHE_DIR="$JAX_CACHE_ROOT/compilation_cache"

mkdir -p "$AUTOTUNE_CACHE_DIR" "$COMPILE_CACHE_DIR"
export XDG_CACHE_HOME="$OPENPI_DIR/.cache"
export JAX_COMPILATION_CACHE_DIR="$COMPILE_CACHE_DIR"
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_per_fusion_autotune_cache_dir="* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$AUTOTUNE_CACHE_DIR"
fi

usage() {
  cat <<'EOF'
Usage:
  bash /data/Embobrain/exp_scripts/train_openpi_libero_chunk_progress_suite.sh <exp_key>

Supported exp_key values:
  prefix_base
  prefix_large
  action_self
  hybrid_base
  hybrid_large
  multilayer_hybrid
  all
EOF
}

resolve_exp() {
  local key="$1"
  case "$key" in
    prefix_base)
      MODE="chunk_prefix"
      EXP_NAME="pi05_libero_lora_chunk_progress_prefix_base"
      PROGRESS_INPUT_DESC="pooled final prefix_out + chunk step embedding"
      ;;
    prefix_large)
      MODE="chunk_prefix_large"
      EXP_NAME="pi05_libero_lora_chunk_progress_prefix_large"
      PROGRESS_INPUT_DESC="pooled final prefix_out + chunk step embedding with larger MLP head"
      ;;
    action_self)
      MODE="chunk_self_action"
      EXP_NAME="pi05_libero_lora_chunk_progress_action_self"
      PROGRESS_INPUT_DESC="sampler-consistent self-sampled suffix_out from the final inference step"
      ;;
    hybrid_base)
      MODE="chunk_hybrid_self_action"
      EXP_NAME="pi05_libero_lora_chunk_progress_prefix_action_self_base"
      PROGRESS_INPUT_DESC="pooled final prefix_out + sampler-consistent self-sampled suffix_out"
      ;;
    hybrid_large)
      MODE="chunk_hybrid_self_action_large"
      EXP_NAME="pi05_libero_lora_chunk_progress_prefix_action_self_large"
      PROGRESS_INPUT_DESC="pooled final prefix_out + sampler-consistent self-sampled suffix_out with larger MLP head"
      ;;
    multilayer_hybrid)
      MODE="chunk_multilayer_self_action"
      EXP_NAME="pi05_libero_lora_chunk_progress_multilayer_prefix_action_self"
      PROGRESS_INPUT_DESC="multilayer prefix captures (5/11/17) + sampler-consistent self-sampled suffix_out"
      ;;
    *)
      echo "ERROR: unknown exp_key '$key'"
      usage
      exit 1
      ;;
  esac
}

run_one() {
  local key="$1"
  resolve_exp "$key"

  if [ ! -d "$OPENPI_DIR/.venv" ]; then
    echo "ERROR: $OPENPI_DIR/.venv not found. Run: cd $OPENPI_DIR && uv sync"
    exit 1
  fi

  if [ ! -d "$INIT_CKPT" ]; then
    echo "ERROR: init checkpoint not found: $INIT_CKPT"
    exit 1
  fi

  if [ ! -x "$OPENPI_DIR/.venv/bin/python" ]; then
    echo "ERROR: missing existing project python: $OPENPI_DIR/.venv/bin/python"
    exit 1
  fi

  local wandb_args=()
  if [ "$WANDB_ENABLED" = "1" ]; then
    wandb_args+=(--wandb-enabled)
  fi

  cd "$OPENPI_DIR"

  echo "============================================"
  echo "  OpenPI pi0.5 LIBERO chunk-progress train"
  echo "============================================"
  echo "Experiment key:               $key"
  echo "Experiment:                   $EXP_NAME"
  echo "Progress readout mode:        $MODE"
  echo "Progress input:               $PROGRESS_INPUT_DESC"
  echo "OpenPI dir:                   $OPENPI_DIR"
  echo "Config:                       $CONFIG_NAME"
  echo "Output config:                $OUTPUT_CONFIG_NAME"
  echo "Checkpoint base dir:          $CHECKPOINT_BASE_DIR"
  echo "Init checkpoint:              $INIT_CKPT"
  echo "Runner:                       $OPENPI_DIR/.venv/bin/python"
  echo "Train steps:                  $NUM_TRAIN_STEPS"
  echo "Batch size:                   $BATCH_SIZE"
  echo "FSDP devices:                 $FSDP_DEVICES"
  echo "Num workers:                  $NUM_WORKERS"
  echo "Progress loss weight:         $PROGRESS_LOSS_WEIGHT"
  echo "Progress relative weight:     $PROGRESS_RELATIVE_LOSS_WEIGHT"
  echo "Peak lr:                      $PEAK_LR"
  echo "Decay lr:                     $DECAY_LR"
  echo "Warmup steps:                 $WARMUP_STEPS"
  echo "Decay steps:                  $DECAY_STEPS"
  echo ""

  "$OPENPI_DIR/.venv/bin/python" scripts/train_chunk_progress_head_only.py \
    --config-name "$CONFIG_NAME" \
    --output-config-name "$OUTPUT_CONFIG_NAME" \
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
    --exp-name "$EXP_NAME" \
    --init-ckpt "$INIT_CKPT" \
    --num-train-steps "$NUM_TRAIN_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --fsdp-devices "$FSDP_DEVICES" \
    --num-workers "$NUM_WORKERS" \
    --log-interval "$LOG_INTERVAL" \
    --save-interval "$SAVE_INTERVAL" \
    --val-interval "$VAL_INTERVAL" \
    --val-num-batches "$VAL_NUM_BATCHES" \
    --val-split-ratio "$VAL_SPLIT_RATIO" \
    --progress-loss-weight "$PROGRESS_LOSS_WEIGHT" \
    --progress-relative-loss-weight "$PROGRESS_RELATIVE_LOSS_WEIGHT" \
    --progress-readout-mode "$MODE" \
    --peak-lr "$PEAK_LR" \
    --decay-lr "$DECAY_LR" \
    --warmup-steps "$WARMUP_STEPS" \
    --decay-steps "$DECAY_STEPS" \
    --overwrite \
    "${wandb_args[@]}"
}

main() {
  local key="${1:-}"
  if [ -z "$key" ]; then
    usage
    exit 1
  fi

  if [ "$key" = "all" ]; then
    run_one prefix_base
    run_one prefix_large
    run_one action_self
    run_one hybrid_base
    run_one hybrid_large
    run_one multilayer_hybrid
    return
  fi

  run_one "$key"
}

main "$@"
