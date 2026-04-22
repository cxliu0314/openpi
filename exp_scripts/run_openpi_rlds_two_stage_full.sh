#!/usr/bin/env bash
set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-/data/Embobrain/openpi}"
UV_BIN="${UV_BIN:-${OPENPI_DIR}/.venv/bin/uv}"

DROID_CONFIG="${DROID_CONFIG:-pi05_rlds_droid_full_uni}"
LIBERO_CONFIG="${LIBERO_CONFIG:-pi05_rlds_libero_uni}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
DROID_EXP_NAME="${DROID_EXP_NAME:-pi05-rlds-droid-full-1ep-${RUN_TAG}}"
LIBERO_EXP_NAME="${LIBERO_EXP_NAME:-pi05-rlds-libero-full-from-droid-${RUN_TAG}}"

FSDP_DEVICES="${FSDP_DEVICES:-8}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
COMPUTE_DROID_NORM_STATS="${COMPUTE_DROID_NORM_STATS:-1}"
COMPUTE_LIBERO_NORM_STATS="${COMPUTE_LIBERO_NORM_STATS:-1}"
OVERWRITE_STAGE1="${OVERWRITE_STAGE1:-0}"
OVERWRITE_STAGE2="${OVERWRITE_STAGE2:-0}"
RESUME_STAGE1="${RESUME_STAGE1:-0}"
RESUME_STAGE2="${RESUME_STAGE2:-0}"
DRY_RUN="${DRY_RUN:-0}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OPENPI_DIR}/.cache}"
export HF_HOME="${HF_HOME:-/data/HF_Cache_dataevo}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"

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
    echo "uv 不可用: $UV_BIN"
    exit 1
fi
if [ ! -d "$OPENPI_DIR/.venv" ]; then
    echo "openpi .venv 不存在: $OPENPI_DIR/.venv"
    exit 1
fi

readarray -t CONFIG_INFO < <(DROID_CONFIG="$DROID_CONFIG" LIBERO_CONFIG="$LIBERO_CONFIG" "$UV_BIN" run --group rlds python - <<'PY'
import json
import math
import os
from pathlib import Path

from openpi.training import config as _config


def load_config_info(config_name: str):
    cfg = _config.get_config(config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    candidates = [
        Path(data_cfg.rlds_data_dir) / "rlds_dataset_stats.json",
        Path(data_cfg.rlds_data_dir) / data_cfg.datasets[0].name / "rlds_dataset_stats.json",
    ]
    total_steps = None
    for candidate in candidates:
        if candidate.exists():
            total_steps = json.loads(candidate.read_text())["total_steps"]
            break
    print(cfg.batch_size)
    print(cfg.num_train_steps)
    print(cfg.save_interval)
    print(data_cfg.rlds_data_dir)
    print(str(cfg.assets_dirs / data_cfg.repo_id / "norm_stats.json"))
    print(0 if total_steps is None else math.ceil(total_steps / cfg.batch_size))

load_config_info(os.environ["DROID_CONFIG"])
load_config_info(os.environ["LIBERO_CONFIG"])
PY
)

DROID_BATCH_SIZE="${CONFIG_INFO[0]}"
DROID_DEFAULT_TRAIN_STEPS="${CONFIG_INFO[1]}"
DROID_SAVE_INTERVAL="${CONFIG_INFO[2]}"
DROID_DATA_ROOT="${CONFIG_INFO[3]}"
DROID_NORM_STATS_PATH="${CONFIG_INFO[4]}"
DROID_EPOCH_STEPS="${CONFIG_INFO[5]}"

LIBERO_BATCH_SIZE="${CONFIG_INFO[6]}"
LIBERO_DEFAULT_TRAIN_STEPS="${CONFIG_INFO[7]}"
LIBERO_SAVE_INTERVAL="${CONFIG_INFO[8]}"
LIBERO_DATA_ROOT="${CONFIG_INFO[9]}"
LIBERO_NORM_STATS_PATH="${CONFIG_INFO[10]}"
LIBERO_EPOCH_STEPS_IGNORED="${CONFIG_INFO[11]}"
unset LIBERO_EPOCH_STEPS_IGNORED

DROID_LAST_STEP=$((DROID_EPOCH_STEPS - 1))
DROID_CKPT_PATH="${OPENPI_DIR}/checkpoints/${DROID_CONFIG}/${DROID_EXP_NAME}/${DROID_LAST_STEP}/params"

WANDB_ARGS=()
if [ "$DISABLE_WANDB" = "1" ]; then
    WANDB_ARGS=(--no-wandb-enabled)
    export WANDB_MODE=disabled
else
    WANDB_ARGS=(--wandb-enabled)
fi

STAGE1_FLAGS=()
if [ "$OVERWRITE_STAGE1" = "1" ]; then
    STAGE1_FLAGS+=(--overwrite)
elif [ "$RESUME_STAGE1" = "1" ]; then
    STAGE1_FLAGS+=(--resume)
fi

STAGE2_FLAGS=()
if [ "$OVERWRITE_STAGE2" = "1" ]; then
    STAGE2_FLAGS+=(--overwrite)
elif [ "$RESUME_STAGE2" = "1" ]; then
    STAGE2_FLAGS+=(--resume)
fi

echo "============================================"
echo "  OpenPI RLDS 两阶段全参训练"
echo "============================================"
echo "OpenPI 目录:             $OPENPI_DIR"
echo "DROID config:            $DROID_CONFIG"
echo "DROID 数据根目录:        $DROID_DATA_ROOT"
echo "DROID 默认 batch:        $DROID_BATCH_SIZE"
echo "DROID 默认 train steps:  $DROID_DEFAULT_TRAIN_STEPS"
echo "DROID 一 epoch steps:    $DROID_EPOCH_STEPS"
echo "DROID exp:               $DROID_EXP_NAME"
echo "DROID 最终 ckpt:         $DROID_CKPT_PATH"
echo "LIBERO config:           $LIBERO_CONFIG"
echo "LIBERO 数据根目录:       $LIBERO_DATA_ROOT"
echo "LIBERO 默认 batch:       $LIBERO_BATCH_SIZE"
echo "LIBERO 默认 train steps: $LIBERO_DEFAULT_TRAIN_STEPS"
echo "LIBERO exp:              $LIBERO_EXP_NAME"
echo "FSDP devices:            $FSDP_DEVICES"
echo "DROID norm stats:        $DROID_NORM_STATS_PATH"
echo "LIBERO norm stats:       $LIBERO_NORM_STATS_PATH"
echo "WandB disabled:          $DISABLE_WANDB"
echo "Dry run:                 $DRY_RUN"
echo ""

GPU_COUNT=$("$UV_BIN" run python -c "import jax; print(len(jax.devices()))" 2>/dev/null)
echo "JAX 可见 GPU 数:         $GPU_COUNT"
echo ""

if [ ! -f "$DROID_DATA_ROOT/rlds_dataset_stats.json" ]; then
    echo "缺少 DROID 统计文件: $DROID_DATA_ROOT/rlds_dataset_stats.json"
    exit 1
fi
if [ ! -d "$LIBERO_DATA_ROOT/libero_10_no_noops" ]; then
    echo "缺少 LIBERO 数据目录: $LIBERO_DATA_ROOT/libero_10_no_noops"
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1，完成配置检查后退出，不启动训练。"
    exit 0
fi

echo "========== [1/4] 准备 DROID norm stats =========="
if [ "$COMPUTE_DROID_NORM_STATS" = "1" ] && [ ! -f "$DROID_NORM_STATS_PATH" ]; then
    "$UV_BIN" run --group rlds scripts/compute_norm_stats.py --config-name "$DROID_CONFIG"
else
    echo "跳过 DROID norm stats"
fi
if [ ! -f "$DROID_NORM_STATS_PATH" ]; then
    echo "缺少 DROID norm stats: $DROID_NORM_STATS_PATH"
    exit 1
fi
echo ""

echo "========== [2/4] DROID 一阶段全参训练（1 epoch） =========="
"$UV_BIN" run --group rlds scripts/train.py "$DROID_CONFIG" \
    --exp-name "$DROID_EXP_NAME" \
    --num-train-steps "$DROID_EPOCH_STEPS" \
    --fsdp-devices "$FSDP_DEVICES" \
    "${WANDB_ARGS[@]}" \
    "${STAGE1_FLAGS[@]}"
echo ""

if [ ! -d "$DROID_CKPT_PATH" ]; then
    echo "未找到 DROID 最终 checkpoint: $DROID_CKPT_PATH"
    exit 1
fi

echo "========== [3/4] 准备 LIBERO norm stats =========="
if [ "$COMPUTE_LIBERO_NORM_STATS" = "1" ] && [ ! -f "$LIBERO_NORM_STATS_PATH" ]; then
    "$UV_BIN" run --group rlds scripts/compute_norm_stats.py --config-name "$LIBERO_CONFIG"
else
    echo "跳过 LIBERO norm stats"
fi
if [ ! -f "$LIBERO_NORM_STATS_PATH" ]; then
    echo "缺少 LIBERO norm stats: $LIBERO_NORM_STATS_PATH"
    exit 1
fi
echo ""

echo "========== [4/4] 基于 DROID checkpoint 的 LIBERO 全参训练 =========="
"$UV_BIN" run --group rlds scripts/train.py "$LIBERO_CONFIG" \
    --exp-name "$LIBERO_EXP_NAME" \
    --weight-loader.params-path "$DROID_CKPT_PATH" \
    --fsdp-devices "$FSDP_DEVICES" \
    "${WANDB_ARGS[@]}" \
    "${STAGE2_FLAGS[@]}"

echo ""
echo "============================================"
echo "  两阶段全参训练已启动完成"
echo "============================================"
echo "DROID checkpoint: $DROID_CKPT_PATH"
echo "LIBERO checkpoint dir: ${OPENPI_DIR}/checkpoints/${LIBERO_CONFIG}/${LIBERO_EXP_NAME}"
