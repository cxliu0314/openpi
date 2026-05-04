#!/bin/bash
set -euo pipefail

OPENPI_DIR="/data/Embobrain/openpi"
UV_BIN="$OPENPI_DIR/.venv/bin/uv"
FFMPEG7_PREFIX="${FFMPEG7_PREFIX:-$OPENPI_DIR/.ffmpeg7}"

if [ "$#" -eq 3 ]; then
  CONFIG_NAME="$1"
  EXP_NAME="$2"
  export CUDA_VISIBLE_DEVICES="$3"
elif [ "$#" -eq 0 ]; then
  CONFIG_NAME="${CONFIG_NAME:-pi05_robotwin_endpose_full_base}"
  EXP_NAME="${EXP_NAME:-pi05_robotwin_baseline_0427_endpose}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
else
  echo "usage: bash $0 [<train_config_name> <model_name> <gpu_use>]"
  echo "example: bash $0 pi05_robotwin_endpose_full_base demo_clean_dual_baseline 0,1,2,3,4,5,6,7"
  exit 1
fi

ROBOTWIN_LEROBOT_REPO_ID="${ROBOTWIN_LEROBOT_REPO_ID:-robotwin_9tasks_0331_dual_endpose}"
ROBOTWIN_LEROBOT_LOCAL_DIR="${ROBOTWIN_LEROBOT_LOCAL_DIR:-/data/Embobrain/RoboTwin/.cache/lerobot/robotwin_9tasks_0331_dual_endpose}"
ROBOTWIN_NORM_STATS_DIR="$OPENPI_DIR/assets/$CONFIG_NAME/${ROBOTWIN_LEROBOT_REPO_ID}"
ROBOTWIN_LEROBOT_ROOT="$(dirname "$ROBOTWIN_LEROBOT_LOCAL_DIR")"
ROBOTWIN_ENDPOSE_QUAT_ORDER="wxyz"
ROBOTWIN_NORM_STATS_QUAT_ORDER_FILE="$ROBOTWIN_NORM_STATS_DIR/robotwin_endpose_quat_order.txt"

# Set these to 1 only when you intentionally want to rebuild cached artifacts.
RECOMPUTE_NORM_STATS="${RECOMPUTE_NORM_STATS:-0}"
OVERWRITE="${OVERWRITE:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
# Local-only W&B; sync later with: wandb sync <run_dir> (or set WANDB_MODE=online to override)
export WANDB_MODE="${WANDB_MODE:-offline}"

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

if [ ! -x "$UV_BIN" ]; then
  echo "ERROR: uv not found: $UV_BIN"
  exit 1
fi

if [ ! -d "$OPENPI_DIR/.venv" ]; then
  echo "ERROR: $OPENPI_DIR/.venv not found. Run: cd $OPENPI_DIR && uv sync"
  exit 1
fi

readarray -t CONFIG_INFO < <(ROBOTWIN_CFG_NAME="$CONFIG_NAME" "$UV_BIN" run python - <<'PY'
import os

from openpi.training import config as _config

cfg = _config.get_config(os.environ["ROBOTWIN_CFG_NAME"])
print(cfg.weight_loader.params_path)
print(cfg.num_train_steps)
print(cfg.batch_size)
print(cfg.fsdp_devices)
print(cfg.num_workers)
print(cfg.log_interval)
print(cfg.save_interval)
print(cfg.keep_period)
print(int(cfg.use_val_set))
print(cfg.val_interval)
print(cfg.val_num_batches)
print(cfg.val_split_ratio)
PY
)

BASE_INIT_CKPT="${CONFIG_INFO[0]}"
TRAIN_STEPS="${CONFIG_INFO[1]}"
BATCH_SIZE="${CONFIG_INFO[2]}"
FSDP_DEVICES="${CONFIG_INFO[3]}"
NUM_WORKERS="${CONFIG_INFO[4]}"
LOG_INTERVAL="${CONFIG_INFO[5]}"
SAVE_INTERVAL="${CONFIG_INFO[6]}"
KEEP_PERIOD="${CONFIG_INFO[7]}"
USE_VAL_SET="${CONFIG_INFO[8]}"
VAL_INTERVAL="${CONFIG_INFO[9]}"
VAL_NUM_BATCHES="${CONFIG_INFO[10]}"
VAL_SPLIT_RATIO="${CONFIG_INFO[11]}"

echo "============================================"
echo "  OpenPI pi0.5 RobotWin baseline full finetune"
echo "============================================"
echo "OpenPI dir:              $OPENPI_DIR"
echo "Config:                  $CONFIG_NAME"
echo "Experiment:              $EXP_NAME"
echo "Base checkpoint:         $BASE_INIT_CKPT"
echo "Train steps(config):     $TRAIN_STEPS"
echo "Batch size(config):      $BATCH_SIZE"
echo "FSDP devices(config):    $FSDP_DEVICES"
echo "Num workers(config):     $NUM_WORKERS"
echo "LeRobot repo id:         $ROBOTWIN_LEROBOT_REPO_ID"
echo "LeRobot local dir:       $ROBOTWIN_LEROBOT_LOCAL_DIR"
echo "Norm stats dir:          $ROBOTWIN_NORM_STATS_DIR"
echo "Endpose quat order:      $ROBOTWIN_ENDPOSE_QUAT_ORDER"
echo "Use val set(config):     $USE_VAL_SET"
echo "Validation split(config): $VAL_SPLIT_RATIO"
echo "Validation interval(config): $VAL_INTERVAL"
echo "Validation batches(config): $VAL_NUM_BATCHES"
echo "Log interval(config):    $LOG_INTERVAL"
echo "Save interval(config):   $SAVE_INTERVAL"
echo "Keep period(config):     $KEEP_PERIOD"
echo "Overwrite checkpoint:    $OVERWRITE"
echo "WandB enabled:           $WANDB_ENABLED"
echo "WandB mode:              $WANDB_MODE"
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

if [ ! -d "$BASE_INIT_CKPT" ]; then
  echo "ERROR: base checkpoint not found: $BASE_INIT_CKPT"
  exit 1
fi

if [ ! -f "$ROBOTWIN_LEROBOT_LOCAL_DIR/meta/info.json" ]; then
  echo "ERROR: LeRobot dataset not found: $ROBOTWIN_LEROBOT_LOCAL_DIR/meta/info.json"
  exit 1
fi

if [ -f "$ROBOTWIN_NORM_STATS_DIR/norm_stats.json" ]; then
  readarray -t NORM_STATS_DIMS < <(
    ROBOTWIN_INFO_PATH="$ROBOTWIN_LEROBOT_LOCAL_DIR/meta/info.json" \
    ROBOTWIN_NORM_STATS_PATH="$ROBOTWIN_NORM_STATS_DIR/norm_stats.json" \
    ROBOTWIN_CFG_NAME="$CONFIG_NAME" \
    ROBOTWIN_REPO_ID="$ROBOTWIN_LEROBOT_REPO_ID" \
    "$UV_BIN" run python - <<'PY'
import dataclasses
import json
import os

import numpy as np

from openpi.training import config as _config

info_path = os.environ["ROBOTWIN_INFO_PATH"]
norm_stats_path = os.environ["ROBOTWIN_NORM_STATS_PATH"]
config_name = os.environ["ROBOTWIN_CFG_NAME"]
repo_id = os.environ["ROBOTWIN_REPO_ID"]

with open(info_path) as f:
    info = json.load(f)
with open(norm_stats_path) as f:
    stats = json.load(f)["norm_stats"]

raw_state_dim = int(info["features"]["observation.state"]["shape"][0])
raw_action_dim = int(info["features"]["action"]["shape"][0])

config = _config.get_config(config_name)
config = dataclasses.replace(
    config,
    data=dataclasses.replace(config.data, repo_id=repo_id),
)
data_config = config.data.create(config.assets_dirs, config.model)
sample = {
    "state": np.zeros((raw_state_dim,), dtype=np.float32),
    "actions": np.zeros((1, raw_action_dim), dtype=np.float32),
}
if raw_state_dim == 16:
    sample["state"][3] = 1.0
    sample["state"][11] = 1.0
if raw_action_dim == 16:
    sample["actions"][..., 3] = 1.0
    sample["actions"][..., 11] = 1.0
for transform in data_config.data_transforms.inputs:
    if transform.__class__.__name__ in ("AlohaInputs", "LiberoInputs"):
        continue
    sample = transform(sample)

dataset_state_dim = int(np.asarray(sample["state"]).shape[-1])
dataset_action_dim = int(np.asarray(sample["actions"]).shape[-1])
stats_state_dim = len(stats["state"]["mean"])
stats_action_dim = len(stats["actions"]["mean"])

print(raw_state_dim)
print(raw_action_dim)
print(dataset_state_dim)
print(dataset_action_dim)
print(stats_state_dim)
print(stats_action_dim)
PY
  )
  RAW_STATE_DIM="${NORM_STATS_DIMS[0]}"
  RAW_ACTION_DIM="${NORM_STATS_DIMS[1]}"
  DATASET_STATE_DIM="${NORM_STATS_DIMS[2]}"
  DATASET_ACTION_DIM="${NORM_STATS_DIMS[3]}"
  STATS_STATE_DIM="${NORM_STATS_DIMS[4]}"
  STATS_ACTION_DIM="${NORM_STATS_DIMS[5]}"
  echo "Raw dataset dims:        state=$RAW_STATE_DIM action=$RAW_ACTION_DIM"
  echo "Transformed dims:        state=$DATASET_STATE_DIM action=$DATASET_ACTION_DIM"
  echo "Norm stats dims:         state=$STATS_STATE_DIM action=$STATS_ACTION_DIM"
  if [ "$DATASET_STATE_DIM" != "$STATS_STATE_DIM" ] || [ "$DATASET_ACTION_DIM" != "$STATS_ACTION_DIM" ]; then
    echo "Norm stats dimension mismatch; will recompute norm stats."
    RECOMPUTE_NORM_STATS=1
  fi
  if [ ! -f "$ROBOTWIN_NORM_STATS_QUAT_ORDER_FILE" ] || \
    [ "$(tr -d '[:space:]' < "$ROBOTWIN_NORM_STATS_QUAT_ORDER_FILE")" != "$ROBOTWIN_ENDPOSE_QUAT_ORDER" ]; then
    echo "Norm stats quaternion order marker missing or stale; will recompute norm stats."
    RECOMPUTE_NORM_STATS=1
  fi
fi

echo "[Check] JAX devices..."
GPU_COUNT=$("$UV_BIN" run python -c "import jax; print(len(jax.devices()))" 2>/dev/null)
echo "JAX visible GPU:         $GPU_COUNT"
if [ "$GPU_COUNT" -lt "$FSDP_DEVICES" ]; then
  echo "ERROR: JAX sees $GPU_COUNT devices, but config fsdp_devices=$FSDP_DEVICES"
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
  printf "%s\n" "$ROBOTWIN_ENDPOSE_QUAT_ORDER" > "$ROBOTWIN_NORM_STATS_QUAT_ORDER_FILE"
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

echo "========== Train RobotWin baseline full finetune =========="
"$UV_BIN" run scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  "${TRAIN_FLAGS[@]}" \
  --data.repo-id="$ROBOTWIN_LEROBOT_REPO_ID"

echo ""
echo "============================================"
echo "  pi0.5 baseline full finetune done"
echo "============================================"
