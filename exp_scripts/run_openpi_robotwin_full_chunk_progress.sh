#!/bin/bash
set -euo pipefail

OPENPI_DIR="/data/Embobrain/openpi"
UV_BIN="$OPENPI_DIR/.venv/bin/uv"
FFMPEG7_PREFIX="${FFMPEG7_PREFIX:-$OPENPI_DIR/.ffmpeg7}"

CONFIG_NAME="${CONFIG_NAME:-pi05_franka_full_base}"
EXP_NAME="${EXP_NAME:-pi05_robotwin_full_chunk_progress_30k_0422}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

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
ROBOTWIN_LEROBOT_REPO_ID="${ROBOTWIN_LEROBOT_REPO_ID:-robotwin_9tasks_0331_split}"
ROBOTWIN_LEROBOT_LOCAL_DIR="${ROBOTWIN_LEROBOT_LOCAL_DIR:-/data/Embobrain/RoboTwin/.cache/lerobot/robotwin_9tasks_0331_split}"
ROBOTWIN_NORM_STATS_DIR="$OPENPI_DIR/assets/$CONFIG_NAME/${ROBOTWIN_LEROBOT_REPO_ID}"
ROBOTWIN_LEROBOT_ROOT="$(dirname "$ROBOTWIN_LEROBOT_LOCAL_DIR")"

# Set these to 1 only when you intentionally want to rebuild cached artifacts.
RECOMPUTE_NORM_STATS="${RECOMPUTE_NORM_STATS:-0}"
OVERWRITE="${OVERWRITE:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export PYTHONPATH="$OPENPI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/data/HF_Cache_dataevo"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$ROBOTWIN_LEROBOT_ROOT}"
export XDG_CACHE_HOME="$OPENPI_DIR/.cache"
export OPENPI_LEROBOT_VIDEO_BACKEND="${OPENPI_LEROBOT_VIDEO_BACKEND:-pyav}"
if [ -d "$FFMPEG7_PREFIX/lib" ]; then
  export LD_LIBRARY_PATH="$FFMPEG7_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

JAX_CACHE_ROOT="$OPENPI_DIR/.cache/jax"
AUTOTUNE_CACHE_DIR="$JAX_CACHE_ROOT/xla_gpu_per_fusion_autotune_cache_dir"
COMPILE_CACHE_DIR="$JAX_CACHE_ROOT/compilation_cache"
mkdir -p "$AUTOTUNE_CACHE_DIR" "$COMPILE_CACHE_DIR"
export JAX_COMPILATION_CACHE_DIR="$COMPILE_CACHE_DIR"
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_per_fusion_autotune_cache_dir="* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_per_fusion_autotune_cache_dir=$AUTOTUNE_CACHE_DIR"
fi

cd "$OPENPI_DIR"
# Avoid uv warning when caller has another active virtualenv.
unset VIRTUAL_ENV || true

echo "============================================"
echo "  OpenPI pi0.5 RobotWin full finetune + chunk progress"
echo "============================================"
echo "OpenPI dir:              $OPENPI_DIR"
echo "Config:                  $CONFIG_NAME"
echo "Experiment:              $EXP_NAME"
echo "Base checkpoint:         $BASE_INIT_CKPT"
echo "Train steps:             $TRAIN_STEPS"
echo "Batch size:              $BATCH_SIZE"
echo "FSDP devices:            $FSDP_DEVICES"
echo "Num workers:             $NUM_WORKERS"
echo "LeRobot repo id:         $ROBOTWIN_LEROBOT_REPO_ID"
echo "LeRobot local dir:       $ROBOTWIN_LEROBOT_LOCAL_DIR"
echo "Norm stats dir:          $ROBOTWIN_NORM_STATS_DIR"
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
echo "LeRobot video backend:   $OPENPI_LEROBOT_VIDEO_BACKEND"
echo "FFmpeg prefix:           $FFMPEG7_PREFIX"
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

if [ ! -f "$ROBOTWIN_LEROBOT_LOCAL_DIR/meta/info.json" ]; then
  echo "ERROR: LeRobot dataset not found: $ROBOTWIN_LEROBOT_LOCAL_DIR/meta/info.json"
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

if [ "$RECOMPUTE_NORM_STATS" = "1" ] || [ ! -f "$ROBOTWIN_NORM_STATS_DIR/norm_stats.json" ]; then
  echo "========== Recompute RobotWin norm stats =========="
  ROBOTWIN_CFG_NAME="$CONFIG_NAME" \
  ROBOTWIN_REPO_ID="$ROBOTWIN_LEROBOT_REPO_ID" \
  "$UV_BIN" run python - <<'PY'
import dataclasses
import os

import numpy as np
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class RepackStateActionsOnly(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        state = x.get("observation.state", x.get("state"))
        actions = x.get("action", x.get("actions"))
        if state is None or actions is None:
            raise KeyError(f"Cannot locate state/actions keys for norm stats. Available keys: {sorted(x.keys())}")
        return {"state": state, "actions": actions}


class EnsureStateActionsNumpy(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        x = dict(x)
        x["state"] = np.asarray(x["state"])
        x["actions"] = np.asarray(x["actions"])
        return x


config_name = os.environ["ROBOTWIN_CFG_NAME"]
repo_id = os.environ["ROBOTWIN_REPO_ID"]

config = _config.get_config(config_name)
config = dataclasses.replace(
    config,
    data=dataclasses.replace(config.data, repo_id=repo_id),
)
data_config = config.data.create(config.assets_dirs, config.model)
dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)

# Norm stats only needs numeric state/action tensors. Some environments do not
# have a torchcodec/ffmpeg stack compatible with LeRobot video decoding.
# Strip video features from metadata to skip frame decoding in __getitem__.
raw_dataset = dataset
while hasattr(raw_dataset, "_dataset"):
    raw_dataset = raw_dataset._dataset
if hasattr(raw_dataset, "meta") and hasattr(raw_dataset.meta, "info"):
    features = dict(raw_dataset.meta.info.get("features", {}))
    raw_dataset.meta.info["features"] = {
        k: v for k, v in features.items() if v.get("dtype") != "video"
    }

dataset = _data_loader.TransformedDataset(
    dataset,
    [
        # Norm stats only needs state/actions. Avoid image-dependent transforms.
        RepackStateActionsOnly(),
        EnsureStateActionsNumpy(),
        *[
            t
            for t in data_config.data_transforms.inputs
            if t.__class__.__name__ not in ("AlohaInputs", "LiberoInputs")
        ],
        RemoveStrings(),
    ],
)
num_batches = len(dataset) // config.batch_size
data_loader = _data_loader.TorchDataLoader(
    dataset,
    local_batch_size=config.batch_size,
    # Inline Python runs from <stdin>; multiprocessing spawn cannot re-import it.
    # Force single-process dataloading for norm-stats computation.
    num_workers=0,
    shuffle=False,
    num_batches=num_batches,
)

stats = {key: normalize.RunningStats() for key in ("state", "actions")}
for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
    for key in ("state", "actions"):
        stats[key].update(np.asarray(batch[key]))

norm_stats = {key: stats[key].get_statistics() for key in ("state", "actions")}
output_path = config.assets_dirs / repo_id
print(f"Writing stats to: {output_path}")
normalize.save(output_path, norm_stats)
PY
  echo ""
fi

if [ ! -f "$ROBOTWIN_NORM_STATS_DIR/norm_stats.json" ]; then
  echo "ERROR: norm stats missing: $ROBOTWIN_NORM_STATS_DIR/norm_stats.json"
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

echo "========== Train RobotWin full finetune + chunk-prefix progress =========="
"$UV_BIN" run scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  "${TRAIN_FLAGS[@]}" \
  --data.repo-id "$ROBOTWIN_LEROBOT_REPO_ID" \
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
