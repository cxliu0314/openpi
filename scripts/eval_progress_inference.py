#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Any

import jax
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
DEFAULT_HF_HOME = pathlib.Path("/data/HF_Cache_dataevo")
DEFAULT_HF_LEROBOT_HOME = DEFAULT_HF_HOME / "lerobot"

if "HF_HOME" not in os.environ and DEFAULT_HF_HOME.exists():
    os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
if "HF_LEROBOT_HOME" not in os.environ and DEFAULT_HF_LEROBOT_HOME.exists():
    os.environ["HF_LEROBOT_HOME"] = str(DEFAULT_HF_LEROBOT_HOME)

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as train_config
from openpi.training import data_loader as data_loader

logger = logging.getLogger("openpi.eval_progress_inference")

VAL_BUCKET = 1000


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    group: str
    exp_name: str
    checkpoint_root: str
    notes: str = ""


CANONICAL_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        group="chunk_prefix",
        exp_name="pi05_libero_lora_chunk_progress_prefix",
        checkpoint_root="/data/Embobrain/openpi/checkpoints/pi05_libero_lora_progress_head",
        notes="old chunk_prefix baseline",
    ),
    ModelSpec(
        group="chunk_prefix",
        exp_name="pi05_libero_lora_chunk_progress_prefix_base",
        checkpoint_root="/data/Embobrain/openpi/checkpoints/pi05_libero_lora_progress_head_chunk",
        notes="new prefix base",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference-mode evaluation for OpenPI progress-head checkpoints.")
    parser.add_argument("--config-name", default="pi05_libero_lora_progress_head")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--loader-batch-size", type=int, default=256)
    parser.add_argument("--infer-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-split-ratio", type=float, default=0.1)
    parser.add_argument("--group", choices=("all", "chunk_prefix"), default="all")
    parser.add_argument("--exp-names", default=None, help="Optional comma-separated exp names to evaluate.")
    parser.add_argument("--sample-cache", default=None, help="Path to raw selected validation samples (.npz).")
    parser.add_argument("--skip-sample-prep", action="store_true")
    parser.add_argument("--prepare-samples-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to /data/Embobrain/openpi/eval_results/progress_inference/<timestamp>.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_float(value: float | np.floating[Any]) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _latest_checkpoint_step(exp_dir: pathlib.Path) -> tuple[int, pathlib.Path] | None:
    steps: list[int] = []
    for child in exp_dir.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / "params").exists():
            steps.append(int(child.name))
    if not steps:
        return None
    step = max(steps)
    return step, exp_dir / str(step)


def _filter_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = list(CANONICAL_MODEL_SPECS)
    if args.group != "all":
        specs = [spec for spec in specs if spec.group == args.group]
    if args.exp_names:
        names = {name.strip() for name in args.exp_names.split(",") if name.strip()}
        specs = [spec for spec in specs if spec.exp_name in names]
    return specs


def _make_output_dir(args: argparse.Namespace) -> pathlib.Path:
    if args.output_dir is not None:
        output_dir = pathlib.Path(args.output_dir).resolve()
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output_dir = (PROJECT_ROOT / "eval_results" / "progress_inference" / timestamp).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir already exists: {output_dir}. Pass --overwrite to reuse it.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_model").mkdir(parents=True, exist_ok=True)
    return output_dir


def _val_mask(episode_hash: np.ndarray, val_split_ratio: float) -> np.ndarray:
    threshold = int(val_split_ratio * VAL_BUCKET)
    hashes = np.asarray(episode_hash, dtype=np.int64).reshape(-1)
    return np.mod(np.abs(hashes.astype(np.int32)), VAL_BUCKET) < threshold


def _sample_cache_path(args: argparse.Namespace, output_dir: pathlib.Path) -> pathlib.Path:
    if args.sample_cache is not None:
        return pathlib.Path(args.sample_cache).resolve()
    return output_dir / "selected_samples_raw.npz"


def _batched_value(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.asarray(value)
    return arr[None, ...] if arr.ndim > 0 else np.asarray([arr.item() if hasattr(arr, "item") else arr])


def _decode_prompt(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = arr.item()
        return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)
    return str(arr)


def _collect_validation_samples_raw(
    cfg: train_config.TrainConfig,
    *,
    num_samples: int,
    val_split_ratio: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)

    image_samples: list[np.ndarray] = []
    wrist_image_samples: list[np.ndarray] = []
    state_samples: list[np.ndarray] = []
    prompt_samples: list[str] = []
    action_samples: list[np.ndarray] = []
    progress_samples: list[np.ndarray] = []
    hash_samples: list[np.ndarray] = []

    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        sample_batch = {key: _batched_value(value) for key, value in sample.items()}
        progress_np, hash_np = data_loader._compute_progress_and_episode_hash(
            sample_batch,
            progress_target_mode="chunk",
        )
        if not _val_mask(hash_np, val_split_ratio)[0]:
            if sample_index < 20 or sample_index % 200 == 0:
                logger.info(
                    "Validation sample scan: sample=%d collected=%d/%d",
                    sample_index,
                    len(action_samples),
                    num_samples,
                )
            continue

        image_samples.append(np.asarray(sample["image"]))
        wrist_image_samples.append(np.asarray(sample["wrist_image"]))
        state_samples.append(np.asarray(sample["state"], dtype=np.float32))
        prompt_samples.append(_decode_prompt(sample.get("prompt", "")))
        action_arr = np.asarray(sample["actions"], dtype=np.float32)
        if action_arr.ndim == 2 and action_arr.shape[-1] > 7:
            action_arr = action_arr[:, :7]
        action_samples.append(action_arr)
        progress_samples.append(np.asarray(progress_np, dtype=np.float32))
        hash_samples.append(np.asarray(hash_np, dtype=np.int64))

        if len(action_samples) >= num_samples:
            logger.info(
                "Validation sample scan: sample=%d collected=%d/%d",
                sample_index,
                len(action_samples),
                num_samples,
            )
            obs_batch = {
                "observation/image": np.stack(image_samples, axis=0),
                "observation/wrist_image": np.stack(wrist_image_samples, axis=0),
                "observation/state": np.stack(state_samples, axis=0),
                "prompt": np.asarray(prompt_samples),
            }
            action_batch = np.stack(action_samples, axis=0)
            progress_batch = np.concatenate(progress_samples, axis=0)
            hash_batch = np.concatenate(hash_samples, axis=0)
            return obs_batch, action_batch, progress_batch, hash_batch

        if sample_index < 20 or sample_index % 200 == 0:
            logger.info(
                "Validation sample scan: sample=%d collected=%d/%d",
                sample_index,
                len(action_samples),
                num_samples,
            )

    raise RuntimeError(f"Could not collect {num_samples} validation samples.")


def _load_or_prepare_samples(
    cfg: train_config.TrainConfig,
    args: argparse.Namespace,
    output_dir: pathlib.Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, pathlib.Path]:
    cache_path = _sample_cache_path(args, output_dir)
    if args.skip_sample_prep:
        if not cache_path.exists():
            raise FileNotFoundError(f"Sample cache not found: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            obs_batch_np = {
                "observation/image": data["observation_image"],
                "observation/wrist_image": data["observation_wrist_image"],
                "observation/state": data["observation_state"],
                "prompt": data["prompt"],
            }
            return (
                obs_batch_np,
                data["gt_actions"],
                data["gt_progress_chunk"],
                data["episode_hash"],
                cache_path,
            )

    obs_batch_np, gt_actions_np, gt_progress_chunk_np, episode_hash_np = _collect_validation_samples_raw(
        cfg,
        num_samples=args.num_samples,
        val_split_ratio=args.val_split_ratio,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        observation_image=obs_batch_np["observation/image"],
        observation_wrist_image=obs_batch_np["observation/wrist_image"],
        observation_state=obs_batch_np["observation/state"],
        prompt=obs_batch_np["prompt"],
        gt_actions=gt_actions_np,
        gt_progress_chunk=gt_progress_chunk_np,
        episode_hash=episode_hash_np,
    )
    return obs_batch_np, gt_actions_np, gt_progress_chunk_np, episode_hash_np, cache_path


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return _safe_float(np.corrcoef(x, y)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(sorted_x):
        end = start + 1
        while end < len(sorted_x) and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size == 0 or y.size == 0:
        return float("nan")
    return _safe_pearson(_rankdata(x), _rankdata(y))


def _mean_pairwise_mae(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    maes: list[float] = []
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            maes.append(float(np.mean(np.abs(values[i] - values[j]))))
    return float(np.mean(maes)) if maes else 0.0


def _compute_metrics(
    *,
    pred_actions: np.ndarray,
    pred_progress: np.ndarray | None,
    gt_actions: np.ndarray,
    gt_progress_chunk: np.ndarray,
    replan_steps: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    action_err = pred_actions - gt_actions[None, ...]
    replan_steps = max(1, min(replan_steps, gt_actions.shape[1]))
    metrics["action_mse"] = float(np.mean(np.square(action_err)))
    metrics["action_rmse"] = float(np.sqrt(metrics["action_mse"]))
    metrics["action_mae"] = float(np.mean(np.abs(action_err)))
    metrics["action_step0_mae"] = float(np.mean(np.abs(action_err[:, :, 0, :])))
    metrics["action_laststep_mae"] = float(np.mean(np.abs(action_err[:, :, -1, :])))
    metrics["action_replan_mae"] = float(np.mean(np.abs(action_err[:, :, :replan_steps, :])))
    metrics["action_seed_std_mean"] = float(np.mean(np.std(pred_actions, axis=0)))
    metrics["action_pairwise_mae"] = _mean_pairwise_mae(pred_actions)

    current_target = gt_progress_chunk[:, 0]
    last_target = gt_progress_chunk[:, -1]
    replan_target = gt_progress_chunk[:, :replan_steps]

    if pred_progress is None:
        return metrics

    chunk_pred = np.asarray(pred_progress, dtype=np.float32)
    chunk_err = chunk_pred - gt_progress_chunk[None, ...]
    current_pred = chunk_pred[:, :, 0]
    last_pred = chunk_pred[:, :, -1]
    current_err = current_pred - current_target[None, :]
    last_err = last_pred - last_target[None, :]
    mean_chunk_pred = chunk_pred.mean(axis=0)
    diffs = np.diff(chunk_pred, axis=-1)
    replan_pred = chunk_pred[:, :, :replan_steps]
    replan_err = replan_pred - replan_target[None, ...]
    replan_diffs = np.diff(replan_pred, axis=-1)

    metrics["chunk_progress_mse"] = float(np.mean(np.square(chunk_err)))
    metrics["chunk_progress_rmse"] = float(np.sqrt(metrics["chunk_progress_mse"]))
    metrics["chunk_progress_mae"] = float(np.mean(np.abs(chunk_err)))
    metrics["chunk_progress_replan_mae"] = float(np.mean(np.abs(replan_err)))
    metrics["current_progress_mse"] = float(np.mean(np.square(current_err)))
    metrics["current_progress_rmse"] = float(np.sqrt(metrics["current_progress_mse"]))
    metrics["current_progress_mae"] = float(np.mean(np.abs(current_err)))
    metrics["current_progress_bias"] = float(np.mean(current_err))
    metrics["last_progress_mse"] = float(np.mean(np.square(last_err)))
    metrics["last_progress_rmse"] = float(np.sqrt(metrics["last_progress_mse"]))
    metrics["last_progress_mae"] = float(np.mean(np.abs(last_err)))
    metrics["last_progress_bias"] = float(np.mean(last_err))
    metrics["current_progress_within_0.05"] = float(np.mean(np.abs(current_err) <= 0.05))
    metrics["current_progress_within_0.10"] = float(np.mean(np.abs(current_err) <= 0.10))
    metrics["last_progress_within_0.05"] = float(np.mean(np.abs(last_err) <= 0.05))
    metrics["last_progress_within_0.10"] = float(np.mean(np.abs(last_err) <= 0.10))
    metrics["current_progress_pearson"] = _safe_pearson(mean_chunk_pred[:, 0], current_target)
    metrics["current_progress_spearman"] = _safe_spearman(mean_chunk_pred[:, 0], current_target)
    metrics["last_progress_pearson"] = _safe_pearson(mean_chunk_pred[:, -1], last_target)
    metrics["last_progress_spearman"] = _safe_spearman(mean_chunk_pred[:, -1], last_target)
    metrics["chunk_progress_seed_std_mean"] = float(np.mean(np.std(chunk_pred, axis=0)))
    metrics["current_progress_seed_std_mean"] = float(np.mean(np.std(current_pred, axis=0)))
    metrics["last_progress_seed_std_mean"] = float(np.mean(np.std(last_pred, axis=0)))
    metrics["chunk_progress_pairwise_mae"] = _mean_pairwise_mae(chunk_pred)
    metrics["current_progress_pairwise_mae"] = _mean_pairwise_mae(current_pred)
    metrics["last_progress_pairwise_mae"] = _mean_pairwise_mae(last_pred)
    metrics["progress_monotonic_violation_rate"] = float(np.mean(diffs < -1e-6))
    metrics["progress_max_drop"] = float(np.min(diffs)) if diffs.size else 0.0
    metrics["progress_replan_monotonic_violation_rate"] = float(np.mean(replan_diffs < -1e-6)) if replan_diffs.size else 0.0
    return metrics


def _save_json(path: pathlib.Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _evaluate_model(
    spec: ModelSpec,
    step_dir: pathlib.Path,
    base_cfg: train_config.TrainConfig,
    obs_batch_np: dict[str, np.ndarray],
    gt_actions_np: np.ndarray,
    gt_progress_chunk_np: np.ndarray,
    episode_hash_np: np.ndarray,
    args: argparse.Namespace,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    logger.info("Loading %s from %s", spec.exp_name, step_dir)
    num_samples = gt_actions_np.shape[0]
    pred_actions_seeds: list[np.ndarray] = []
    pred_progress_seeds: list[np.ndarray] = []
    policy_cfg = dataclasses.replace(
        base_cfg,
        model=dataclasses.replace(base_cfg.model, enable_progress_head=True),
    )

    for seed_offset in range(args.num_seeds):
        seed = args.seed + seed_offset
        logger.info("Evaluating %s seed=%d/%d", spec.exp_name, seed_offset + 1, args.num_seeds)
        policy = _policy_config.create_trained_policy(
            policy_cfg,
            step_dir,
            sample_kwargs={"num_steps": args.num_steps},
            rng=jax.random.key(seed),
            allow_partial_params=True,
        )
        seed_actions: list[np.ndarray] = []
        seed_progress: list[np.ndarray] = []

        for sample_index in range(num_samples):
            obs = {
                "observation/image": obs_batch_np["observation/image"][sample_index],
                "observation/wrist_image": obs_batch_np["observation/wrist_image"][sample_index],
                "observation/state": obs_batch_np["observation/state"][sample_index],
                "prompt": str(obs_batch_np["prompt"][sample_index]),
            }
            outputs = policy.infer_with_progress(
                obs,
            )
            seed_actions.append(np.asarray(outputs["actions"], dtype=np.float32))
            if outputs.get("progress") is not None:
                seed_progress.append(np.asarray(outputs["progress"], dtype=np.float32))
            if sample_index < 3 or (sample_index + 1) % 20 == 0 or sample_index + 1 == num_samples:
                logger.info(
                    "Evaluating %s seed=%d/%d sample=%d/%d",
                    spec.exp_name,
                    seed_offset + 1,
                    args.num_seeds,
                    sample_index + 1,
                    num_samples,
                )

        pred_actions_seeds.append(np.stack(seed_actions, axis=0))
        if seed_progress:
            pred_progress_seeds.append(np.stack(seed_progress, axis=0))
        logger.info("Finished %s seed=%d/%d", spec.exp_name, seed_offset + 1, args.num_seeds)

    pred_actions_np = np.stack(pred_actions_seeds, axis=0)
    pred_progress_np = np.stack(pred_progress_seeds, axis=0) if pred_progress_seeds else None
    metrics = _compute_metrics(
        pred_actions=pred_actions_np,
        pred_progress=pred_progress_np,
        gt_actions=gt_actions_np,
        gt_progress_chunk=gt_progress_chunk_np,
        replan_steps=args.replan_steps,
    )

    result = {
        "group": spec.group,
        "exp_name": spec.exp_name,
        "checkpoint_root": spec.checkpoint_root,
        "checkpoint_step": int(step_dir.name),
        "checkpoint_dir": str(step_dir),
        "readout_mode": "chunk_prefix",
        "progress_target_mode": "chunk",
        "notes": spec.notes,
        "num_samples": int(num_samples),
        "num_seeds": int(args.num_seeds),
        "num_steps": int(args.num_steps),
        "metrics": metrics,
    }

    per_model_dir = output_dir / "per_model" / spec.exp_name
    per_model_dir.mkdir(parents=True, exist_ok=True)
    _save_json(per_model_dir / "metrics.json", result)
    np.savez_compressed(
        per_model_dir / "predictions.npz",
        episode_hash=episode_hash_np,
        gt_actions=gt_actions_np,
        gt_progress_chunk=gt_progress_chunk_np,
        pred_actions=pred_actions_np,
        pred_progress=pred_progress_np if pred_progress_np is not None else np.asarray([], dtype=np.float32),
    )
    return result


def _write_summary(output_dir: pathlib.Path, results: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    summary_rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "group": result["group"],
            "exp_name": result["exp_name"],
            "checkpoint_step": result["checkpoint_step"],
            "readout_mode": result["readout_mode"],
            "progress_target_mode": result["progress_target_mode"],
            **result["metrics"],
        }
        summary_rows.append(row)

    if summary_rows:
        fieldnames: list[str] = []
        for row in summary_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    _save_json(output_dir / "summary.json", {"results": results, "skipped": skipped})

    lines = [
        "# Progress Inference Evaluation",
        "",
        "## Evaluated Models",
    ]
    for result in sorted(results, key=lambda item: item["metrics"].get("current_progress_mae", math.inf)):
        metrics = result["metrics"]
        line = (
            f"- `{result['exp_name']}` [{result['readout_mode']}] "
            f"step={result['checkpoint_step']} "
            f"action_mae={metrics.get('action_mae', float('nan')):.4f} "
            f"current_progress_mae={metrics.get('current_progress_mae', float('nan')):.4f}"
        )
        if "chunk_progress_mae" in metrics:
            line += f" chunk_progress_mae={metrics['chunk_progress_mae']:.4f}"
        if "progress_monotonic_violation_rate" in metrics:
            line += f" monotonic_violation_rate={metrics['progress_monotonic_violation_rate']:.4f}"
        lines.append(line)
    lines.extend(["", "## Skipped Models"])
    if skipped:
        for item in skipped:
            lines.append(f"- `{item['exp_name']}`: {item['reason']}")
    else:
        lines.append("- None")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    specs = _filter_specs(args)
    output_dir = _make_output_dir(args)
    logger.info("Output dir: %s", output_dir)

    base_cfg = train_config.get_config(args.config_name)
    eval_cfg = dataclasses.replace(
        base_cfg,
        batch_size=args.loader_batch_size,
        num_workers=args.num_workers,
        enable_progress_loss=True,
        progress_target_mode="chunk",
        use_val_set=True,
        val_split_ratio=args.val_split_ratio,
    )

    obs_batch_np, gt_actions_np, gt_progress_chunk_np, episode_hash_np, sample_cache_path = _load_or_prepare_samples(
        eval_cfg,
        args,
        output_dir,
    )
    logger.info("Collected %d validation samples", gt_actions_np.shape[0])
    logger.info("Sample cache: %s", sample_cache_path)

    if args.prepare_samples_only:
        logger.info("Prepared samples only. Exiting without model evaluation.")
        return

    np.savez_compressed(
        output_dir / "selected_samples.npz",
        observation_image=obs_batch_np["observation/image"],
        observation_wrist_image=obs_batch_np["observation/wrist_image"],
        observation_state=obs_batch_np["observation/state"],
        prompt=obs_batch_np["prompt"],
        episode_hash=episode_hash_np,
        gt_actions=gt_actions_np,
        gt_progress_chunk=gt_progress_chunk_np,
    )
    _save_json(
        output_dir / "eval_config.json",
        {
            "args": vars(args),
            "config_name": args.config_name,
            "loader_batch_size": args.loader_batch_size,
            "infer_batch_size": args.infer_batch_size,
            "replan_steps": args.replan_steps,
            "sample_cache": str(sample_cache_path),
            "num_devices": len(jax.devices()),
            "devices": [str(device) for device in jax.devices()],
        },
    )

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for spec in specs:
        exp_dir = pathlib.Path(spec.checkpoint_root) / spec.exp_name
        if not exp_dir.exists():
            skipped.append({"exp_name": spec.exp_name, "reason": f"missing experiment dir: {exp_dir}"})
            continue
        latest = _latest_checkpoint_step(exp_dir)
        if latest is None:
            skipped.append({"exp_name": spec.exp_name, "reason": f"no numeric checkpoint step under: {exp_dir}"})
            continue
        checkpoint_step, step_dir = latest
        logger.info("Evaluating %s at step %d", spec.exp_name, checkpoint_step)
        results.append(
            _evaluate_model(
                spec,
                step_dir,
                base_cfg,
                obs_batch_np,
                gt_actions_np,
                gt_progress_chunk_np,
                episode_hash_np,
                args,
                output_dir,
            )
        )

    _write_summary(output_dir, results, skipped)
    logger.info("Finished. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
