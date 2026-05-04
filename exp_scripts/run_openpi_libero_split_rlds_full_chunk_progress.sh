#!/usr/bin/env bash
set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-/data/Embobrain/openpi}"
UV_BIN="${UV_BIN:-${OPENPI_DIR}/.venv/bin/uv}"

CONFIG_NAME="${CONFIG_NAME:-pi05_rlds_libero_uni}"
EXP_NAME="${EXP_NAME:-pi05-rlds-libero-split-full-$(date -u +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-openpi}"

FSDP_DEVICES="${FSDP_DEVICES:-8}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-1}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
PROGRESS_LOSS_WEIGHT="${PROGRESS_LOSS_WEIGHT:-0.1}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OPENPI_DIR}/.cache}"

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
    echo "uv not found: $UV_BIN"
    exit 1
fi
if [ ! -d "$OPENPI_DIR/.venv" ]; then
    echo "openpi .venv not found: $OPENPI_DIR/.venv"
    exit 1
fi

readarray -t CONFIG_INFO < <(CONFIG_NAME="$CONFIG_NAME" "$UV_BIN" run --group rlds python - <<'PY'
import os

from openpi.training import config as _config

cfg = _config.get_config(os.environ["CONFIG_NAME"])
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
print(cfg.batch_size)
print(cfg.num_train_steps)
print(cfg.save_interval)
print(data_cfg.rlds_data_dir)
print(data_cfg.datasets[0].name)
print(str(cfg.assets_dirs / data_cfg.repo_id / "norm_stats.json"))
print(cfg.weight_loader.params_path)
print(int(bool(getattr(cfg.model, "enable_progress_head", False))))
print(int(cfg.enable_progress_loss))
print(int(cfg.use_val_set))
PY
)

BATCH_SIZE="${CONFIG_INFO[0]}"
TRAIN_STEPS="${CONFIG_INFO[1]}"
SAVE_INTERVAL="${CONFIG_INFO[2]}"
DATA_ROOT="${CONFIG_INFO[3]}"
DATASET_NAME="${CONFIG_INFO[4]}"
NORM_STATS_PATH="${CONFIG_INFO[5]}"
BASE_CKPT="${CONFIG_INFO[6]}"
ENABLE_PROGRESS_HEAD="${CONFIG_INFO[7]}"
ENABLE_PROGRESS_LOSS="${CONFIG_INFO[8]}"
USE_VAL_SET="${CONFIG_INFO[9]}"

WANDB_ARGS=()
if [ "$DISABLE_WANDB" = "1" ]; then
    WANDB_ARGS=(--no-wandb-enabled)
    export WANDB_MODE=disabled
else
    WANDB_ARGS=(--wandb-enabled)
fi

TRAIN_FLAGS=()
if [ "$OVERWRITE" = "1" ]; then
    TRAIN_FLAGS+=(--overwrite)
elif [ "$RESUME" = "1" ]; then
    TRAIN_FLAGS+=(--resume)
fi

echo "============================================"
echo "  OpenPI pi0.5 full finetune: LIBERO split RLDS + chunk progress"
echo "============================================"
echo "OpenPI dir:           $OPENPI_DIR"
echo "Config:               $CONFIG_NAME"
echo "Experiment:           $EXP_NAME"
echo "Project:              $PROJECT_NAME"
echo "Base checkpoint:      $BASE_CKPT"
echo "Data root:            $DATA_ROOT"
echo "Dataset:              $DATASET_NAME"
echo "Batch size:           $BATCH_SIZE"
echo "Train steps:          $TRAIN_STEPS"
echo "Save interval:        $SAVE_INTERVAL"
echo "FSDP devices:         $FSDP_DEVICES"
echo "Norm stats:           $NORM_STATS_PATH"
echo "Progress head:        $ENABLE_PROGRESS_HEAD"
echo "Progress loss:        $ENABLE_PROGRESS_LOSS"
echo "Progress loss weight: $PROGRESS_LOSS_WEIGHT"
echo "Use val set:          $USE_VAL_SET (script overrides to false)"
echo "WandB disabled:       $DISABLE_WANDB"
echo "Dry run:              $DRY_RUN"
echo ""

GPU_COUNT=$("$UV_BIN" run python -c "import jax; print(len(jax.devices()))" 2>/dev/null)
echo "JAX visible GPU:      $GPU_COUNT"
echo ""

if [ ! -d "$BASE_CKPT" ]; then
    echo "Missing base checkpoint: $BASE_CKPT"
    exit 1
fi
if [ ! -d "$DATA_ROOT/$DATASET_NAME" ]; then
    echo "Missing dataset: $DATA_ROOT/$DATASET_NAME"
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1, checked config and paths; training not started."
    exit 0
fi

if [ "$COMPUTE_NORM_STATS" = "1" ] && [ ! -f "$NORM_STATS_PATH" ]; then
    echo "========== Compute norm stats =========="
    "$UV_BIN" run --group rlds scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
    echo ""
else
    echo "Skip norm stats."
fi
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "Missing norm stats: $NORM_STATS_PATH"
    exit 1
fi

echo "========== Train pi0.5 full finetune with chunk progress =========="
"$UV_BIN" run --group rlds scripts/train.py "$CONFIG_NAME" \
    --exp-name "$EXP_NAME" \
    --project-name "$PROJECT_NAME" \
    --fsdp-devices "$FSDP_DEVICES" \
    --no-use-val-set \
    --model.enable-progress-head True \
    --enable-progress-loss \
    --progress-target-mode chunk \
    --progress-readout-mode chunk_prefix \
    --progress-loss-weight "$PROGRESS_LOSS_WEIGHT" \
    "${WANDB_ARGS[@]}" \
    "${TRAIN_FLAGS[@]}"

echo ""
echo "Checkpoint dir: ${OPENPI_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
