#!/usr/bin/env bash
set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-/data/Embobrain/openpi}"
OUTPUT_DIR="${OUTPUT_DIR:-$OPENPI_DIR/eval_results/progress_inference/latest}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
LOADER_BATCH_SIZE="${LOADER_BATCH_SIZE:-256}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VAL_SPLIT_RATIO="${VAL_SPLIT_RATIO:-0.1}"
GROUP="${GROUP:-all}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_lora_progress_head}"
EXP_NAMES="${EXP_NAMES:-}"
SAMPLE_CACHE="${SAMPLE_CACHE:-}"
SKIP_SAMPLE_PREP="${SKIP_SAMPLE_PREP:-0}"
PREPARE_SAMPLES_ONLY="${PREPARE_SAMPLES_ONLY:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}"
export PYTHONPATH="$OPENPI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/data/HF_Cache_dataevo}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/HF_Cache_dataevo/lerobot}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

JAX_CACHE_ROOT="$OPENPI_DIR/.cache/jax"
AUTOTUNE_CACHE_DIR="$JAX_CACHE_ROOT/xla_gpu_per_fusion_autotune_cache_dir"
COMPILE_CACHE_DIR="$JAX_CACHE_ROOT/compilation_cache"
mkdir -p "$AUTOTUNE_CACHE_DIR" "$COMPILE_CACHE_DIR" "$OUTPUT_DIR"
export XDG_CACHE_HOME="$OPENPI_DIR/.cache"
export JAX_COMPILATION_CACHE_DIR="$COMPILE_CACHE_DIR"
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_per_fusion_autotune_cache_dir="* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$AUTOTUNE_CACHE_DIR"
fi

cd "$OPENPI_DIR"

echo "============================================"
echo "  OpenPI progress inference evaluation"
echo "============================================"
echo "OpenPI dir:            $OPENPI_DIR"
echo "Output dir:            $OUTPUT_DIR"
echo "Config name:           $CONFIG_NAME"
echo "Group:                 $GROUP"
echo "Exp names:             ${EXP_NAMES:-<all>}"
echo "Num samples:           $NUM_SAMPLES"
echo "Num seeds:             $NUM_SEEDS"
echo "Seed:                  $SEED"
echo "Num steps:             $NUM_STEPS"
echo "Replan steps:          $REPLAN_STEPS"
echo "Loader batch size:     $LOADER_BATCH_SIZE"
echo "Infer batch size:      $INFER_BATCH_SIZE"
echo "Num workers:           $NUM_WORKERS"
echo "Val split ratio:       $VAL_SPLIT_RATIO"
echo "Sample cache:          ${SAMPLE_CACHE:-<default>}"
echo "Skip sample prep:      $SKIP_SAMPLE_PREP"
echo "Prepare only:          $PREPARE_SAMPLES_ONLY"
echo "Runner:                $OPENPI_DIR/.venv/bin/python -u"
echo ""

ARGS=(
  --config-name "$CONFIG_NAME"
  --num-samples "$NUM_SAMPLES"
  --num-seeds "$NUM_SEEDS"
  --seed "$SEED"
  --num-steps "$NUM_STEPS"
  --replan-steps "$REPLAN_STEPS"
  --loader-batch-size "$LOADER_BATCH_SIZE"
  --infer-batch-size "$INFER_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --val-split-ratio "$VAL_SPLIT_RATIO"
  --group "$GROUP"
  --output-dir "$OUTPUT_DIR"
  --overwrite
)

if [[ -n "$EXP_NAMES" ]]; then
  ARGS+=(--exp-names "$EXP_NAMES")
fi
if [[ -n "$SAMPLE_CACHE" ]]; then
  ARGS+=(--sample-cache "$SAMPLE_CACHE")
fi
if [[ "$SKIP_SAMPLE_PREP" == "1" ]]; then
  ARGS+=(--skip-sample-prep)
fi
if [[ "$PREPARE_SAMPLES_ONLY" == "1" ]]; then
  ARGS+=(--prepare-samples-only)
fi

"$OPENPI_DIR/.venv/bin/python" -u scripts/eval_progress_inference.py "${ARGS[@]}"
