#!/usr/bin/env python3
"""Discover RLDS datasets under a root directory and export sampled videos into each dataset's vis/ folder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
import tensorflow_datasets as tfds

try:
    from .export_droid_rlds_videos import (
        build_decoders,
        build_output_path,
        prepare_episode_video_data,
        resolve_builder,
        write_episode_video,
    )
except ImportError:
    from export_droid_rlds_videos import (
        build_decoders,
        build_output_path,
        prepare_episode_video_data,
        resolve_builder,
        write_episode_video,
    )


DEFAULT_ROOT = "/data/Embobrain/dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export first N subtitle videos for every RLDS dataset under a root directory.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help="Root directory containing RLDS datasets.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=20,
        help="Maximum number of sampled episodes to export per dataset.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=100,
        help="Sample one episode every N episodes.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Video FPS.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="TFDS split to export from.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default=None,
        help="Preferred observation image key.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing videos.",
    )
    return parser.parse_args()


def count_dataset_infos(path: Path) -> int:
    return sum(1 for _ in path.rglob("dataset_info.json"))


def highest_unique_dataset_root(top_level_root: Path, dataset_info_path: Path) -> Path:
    ancestors: list[Path] = []
    current = dataset_info_path.parent.parent
    while True:
        ancestors.append(current)
        if current == top_level_root:
            break
        current = current.parent
        if top_level_root not in current.parents and current != top_level_root:
            break

    chosen = ancestors[0]
    for ancestor in reversed(ancestors):
        if count_dataset_infos(ancestor) == 1:
            chosen = ancestor
            break
    return chosen


def discover_dataset_roots(root: Path) -> list[Path]:
    dataset_roots: list[Path] = []
    seen: set[Path] = set()
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        dataset_infos = sorted(child.rglob("dataset_info.json"))
        if not dataset_infos:
            continue
        for info in dataset_infos:
            dataset_root = highest_unique_dataset_root(child, info)
            if dataset_root not in seen:
                seen.add(dataset_root)
                dataset_roots.append(dataset_root)
    return sorted(dataset_roots)


def export_dataset(root: Path, args: argparse.Namespace) -> dict:
    builder = resolve_builder(str(root))
    split_info = builder.info.splits[args.split]
    total_episodes = int(split_info.num_examples)
    vis_dir = root / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = vis_dir / "manifest.jsonl"

    if args.overwrite:
        for old_video in vis_dir.glob("episode_*.mp4"):
            old_video.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    if args.sample_every <= 0:
        raise ValueError("--sample-every must be positive")

    sampled_indices = list(range(0, total_episodes, args.sample_every))
    if args.max_episodes > 0:
        sampled_indices = sampled_indices[: args.max_episodes]

    print(
        f"[dataset] root={root} resolved={builder.info.name}:{builder.info.version} "
        f"sampled={len(sampled_indices)} total={total_episodes} stride={args.sample_every}",
        flush=True,
    )

    dataset = builder.as_dataset(
        split=args.split,
        shuffle_files=False,
        decoders=build_decoders(builder),
    )

    records = []
    for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
        if episode_index not in sampled_indices:
            continue
        episode_data = prepare_episode_video_data(
            episode=episode,
            episode_index=episode_index,
            preferred_camera_key=args.camera_key,
        )
        output_path = build_output_path(vis_dir, episode_index, episode_data["anomalies"])
        if output_path.exists() and not args.overwrite:
            print(f"[skip] root={root.name} episode={episode_index} existing={output_path.name}", flush=True)
            continue

        print(f"[export] root={root.name} episode={episode_index} -> {output_path.name}", flush=True)
        record = write_episode_video(episode_data, output_path, args.fps)
        records.append(record)
        if len(records) >= len(sampled_indices):
            break

    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "dataset_root": str(root),
        "resolved_name": f"{builder.info.name}:{builder.info.version}",
        "exported": len(records),
        "sample_every": args.sample_every,
        "sampled_episode_indices": sampled_indices[: len(records)],
        "vis_dir": str(vis_dir),
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    dataset_roots = discover_dataset_roots(root)
    print(f"[start] discovered={len(dataset_roots)} datasets under {root}", flush=True)
    for dataset_root in dataset_roots:
        print(f"[found] {dataset_root}", flush=True)

    summary = []
    for dataset_root in dataset_roots:
        try:
            summary.append(export_dataset(dataset_root, args))
        except Exception as exc:
            print(f"[error] root={dataset_root} error={exc}", flush=True)

    summary_path = root / "rlds_vis_export_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[finish] completed={len(summary)} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
