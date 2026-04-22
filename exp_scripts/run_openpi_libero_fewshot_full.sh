#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash exp_scripts/run_openpi_libero_fewshot_full.sh <original|split_padded> <10|20|30|50|100>"
    exit 2
fi

DATA_KIND="$1"
SHOT="$2"

case "$DATA_KIND" in
    original|split_padded) ;;
    *) echo "Unknown DATA_KIND: $DATA_KIND"; exit 2 ;;
esac

case "$SHOT" in
    10|20|30|50|100) ;;
    *) echo "Unknown SHOT: $SHOT"; exit 2 ;;
esac

OPENPI_DIR="${OPENPI_DIR:-/data/Embobrain/openpi}"
export HF_HOME="${HF_HOME:-/data/HF_Cache_dataevo}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
UV_BIN="${UV_BIN:-${OPENPI_DIR}/.venv/bin/uv}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
FEWSHOT_LEROBOT_PREFIX="${FEWSHOT_LEROBOT_PREFIX:-libero_fewshot_lerobot_trainval_2x}"
DATASET_NAME="${DATASET_NAME:-libero_10_no_noops}"
TRAIN_REPO_ID="${TRAIN_REPO_ID:-${FEWSHOT_LEROBOT_PREFIX}/${DATA_KIND}_n${SHOT}_train/${DATASET_NAME}}"
VAL_REPO_ID="${VAL_REPO_ID:-${FEWSHOT_LEROBOT_PREFIX}/${DATA_KIND}_n${SHOT}_val/${DATASET_NAME}}"
EXP_SUFFIX="${EXP_SUFFIX:-}"
EXP_NAME="${EXP_NAME:-pi05-full-${DATA_KIND}-n${SHOT}-fewshot${EXP_SUFFIX}}"
PROJECT_NAME="${PROJECT_NAME:-fewshot_full_exp}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OPENPI_DIR}/checkpoints}"

TRAIN_STEPS="${TRAIN_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_INTERVAL="${VAL_INTERVAL:-100}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
KEEP_PERIOD="${KEEP_PERIOD:-2000}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
RECOMPUTE_NORM_STATS="${RECOMPUTE_NORM_STATS:-0}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-1}"

WANDB_ARGS=()
if [ "$DISABLE_WANDB" = "1" ]; then
    WANDB_ARGS=(--no-wandb-enabled)
    export WANDB_MODE=disabled
else
    WANDB_ARGS=(--wandb-enabled)
fi

TRAIN_FLAGS=()
if [ "$OVERWRITE" = "1" ]; then
    TRAIN_FLAGS=(--overwrite)
elif [ "$RESUME" = "1" ]; then
    TRAIN_FLAGS=(--resume)
fi

if [ "$DATA_KIND" = "split_padded" ]; then
    PROGRESS_ARGS=(
        --model.enable-progress-head True
        --enable-progress-loss
        --progress-target-mode chunk
        --progress-readout-mode chunk_prefix
    )
else
    PROGRESS_ARGS=(
        --model.enable-progress-head False
        --no-enable-progress-loss
    )
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OPENPI_DIR}/.cache}"
export PYTHONPATH="${OPENPI_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

JAX_CACHE_ROOT="${OPENPI_DIR}/.cache/jax"
AUTOTUNE_CACHE_DIR="${JAX_CACHE_ROOT}/xla_gpu_per_fusion_autotune_cache_dir"
COMPILE_CACHE_DIR="${JAX_CACHE_ROOT}/compilation_cache"
mkdir -p "$AUTOTUNE_CACHE_DIR" "$COMPILE_CACHE_DIR"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$COMPILE_CACHE_DIR}"
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_enable_command_buffer"* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
fi
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_graph_level="* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_graph_level=0"
fi
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_per_fusion_autotune_cache_dir="* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$AUTOTUNE_CACHE_DIR"
fi

cd "$OPENPI_DIR"

if [ ! -x "$UV_BIN" ]; then
    echo "uv is not available: $UV_BIN"
    exit 1
fi

for repo_id in "$TRAIN_REPO_ID" "$VAL_REPO_ID"; do
    if [ ! -f "$HF_HOME/lerobot/$repo_id/meta/info.json" ]; then
        echo "LeRobot dataset is not ready: $HF_HOME/lerobot/$repo_id/meta/info.json"
        echo "Convert it first, for example:"
        echo "  cd $OPENPI_DIR && $UV_BIN run scripts/convert_fewshot_rlds_to_lerobot.py --data-kind $DATA_KIND --shot $SHOT"
        exit 1
    fi
done

NORM_STATS_DIR="$OPENPI_DIR/assets/$CONFIG_NAME/$TRAIN_REPO_ID"
if [ "$RECOMPUTE_NORM_STATS" = "1" ] || [ ! -f "$NORM_STATS_DIR/norm_stats.json" ]; then
    "$UV_BIN" run scripts/compute_fewshot_norm_stats.py \
        --config-name "$CONFIG_NAME" \
        --data-repo-id "$TRAIN_REPO_ID"
fi

"$UV_BIN" run scripts/train_fewshot.py "$CONFIG_NAME" \
    --exp-name "$EXP_NAME" \
    --project-name "$PROJECT_NAME" \
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
    "${WANDB_ARGS[@]}" \
    "${TRAIN_FLAGS[@]}" \
    --data.repo-id "$TRAIN_REPO_ID" \
    --val-repo-id "$VAL_REPO_ID" \
    --num-train-steps "$TRAIN_STEPS" \
    --log-interval "$LOG_INTERVAL" \
    --batch-size "$BATCH_SIZE" \
    --fsdp-devices "$FSDP_DEVICES" \
    --num-workers "$NUM_WORKERS" \
    --lr-schedule.warmup-steps 0 \
    --lr-schedule.peak-lr 5e-5 \
    --lr-schedule.decay-steps 1000000000 \
    --lr-schedule.decay-lr 5e-5 \
    "${PROGRESS_ARGS[@]}" \
    --val-interval "$VAL_INTERVAL" \
    --val-num-batches "$VAL_NUM_BATCHES" \
    --save-interval "$SAVE_INTERVAL" \
    --keep-period "$KEEP_PERIOD"
