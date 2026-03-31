#!/usr/bin/env python3
"""Export RLDS dataset stats as a JSON file."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


DEFAULT_OUTPUT_NAME = "rlds_dataset_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RLDS dataset stats JSON with overall summary and per-episode records.",
    )
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--split", type=str, default="train", help="TFDS split to inspect.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON path. Defaults to <root>/rlds_dataset_stats.json.",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of episodes to inspect.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N episodes. Set <= 0 to disable.",
    )
    return parser.parse_args()


def resolve_builder(dataset_path: str) -> tfds.core.DatasetBuilder:
    path = Path(dataset_path)
    candidates: list[Path] = []

    if (path / "dataset_info.json").exists():
        candidates.append(path)
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if (child / "dataset_info.json").exists():
                candidates.append(child)
            else:
                for grandchild in sorted(child.iterdir()):
                    if grandchild.is_dir() and (grandchild / "dataset_info.json").exists():
                        candidates.append(grandchild)

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            ordered_candidates.append(candidate)
            seen.add(candidate)

    last_error: Exception | None = None
    for candidate in ordered_candidates:
        try:
            return tfds.builder_from_directory(str(candidate))
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"No TFDS dataset_info.json found under {dataset_path}")


def decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return decode_text(value.item())
        for item in value.reshape(-1):
            text = decode_text(item)
            if text:
                return text
        return ""
    if hasattr(value, "item"):
        try:
            return decode_text(value.item())
        except Exception:
            pass
    return str(value).strip()


def decode_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return decode_scalar(value.item())
        if value.size <= 16:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {key: decode_scalar(val) for key, val in value.items()}
    return value


def summarize_lengths(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {
            "min": 0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0,
        }

    arr = np.asarray(lengths, dtype=np.int32)
    return {
        "min": int(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(arr.mean()),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
    }


def extract_instruction_fields(step: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in step.items():
        if "instruction" not in key:
            continue
        fields[key] = decode_text(value)
    return fields


def choose_main_instruction(fields: dict[str, str]) -> str:
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        if fields.get(key):
            return fields[key]
    for value in fields.values():
        if value:
            return value
    return ""


def write_json_field(fobj, key: str, value: Any, trailing_comma: bool) -> None:
    fobj.write(f"  {json.dumps(key)}: ")
    json.dump(value, fobj, ensure_ascii=False)
    if trailing_comma:
        fobj.write(",\n")
    else:
        fobj.write("\n")


def export_stats(args: argparse.Namespace) -> Path:
    root = Path(args.root)
    output_path = Path(args.output_json) if args.output_json else root / DEFAULT_OUTPUT_NAME
    builder = resolve_builder(str(root))

    split_info = builder.info.splits[args.split]
    declared_episodes = int(split_info.num_examples)
    inspect_episodes = (
        min(declared_episodes, args.limit_episodes)
        if args.limit_episodes is not None
        else declared_episodes
    )

    dataset = builder.as_dataset(
        split=f"{args.split}[:{inspect_episodes}]",
        shuffle_files=False,
    )

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".episodes.tmp",
        prefix="rlds_stats_",
        dir=str(output_path.parent),
        delete=False,
    )

    episode_lengths: list[int] = []
    inspected_episodes = 0
    total_steps = 0
    empty_main_instruction_episodes = 0
    instruction_field_names: set[str] = set()

    try:
        first_record = True
        for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
            steps = list(episode["steps"])
            num_steps = len(steps)

            instruction_fields: dict[str, str] = {}
            for step in steps:
                for key, value in extract_instruction_fields(step).items():
                    instruction_field_names.add(key)
                    if key not in instruction_fields or not instruction_fields[key]:
                        if value:
                            instruction_fields[key] = value

            main_instruction = choose_main_instruction(instruction_fields)
            if not main_instruction:
                empty_main_instruction_episodes += 1

            metadata = decode_scalar(episode.get("episode_metadata", {}))
            record = {
                "episode_index": episode_index,
                "num_steps": num_steps,
                "main_instruction": main_instruction,
                "instruction_fields": instruction_fields,
                "episode_metadata": metadata,
            }

            if not first_record:
                temp_file.write(",\n")
            json.dump(record, temp_file, ensure_ascii=False)
            first_record = False

            inspected_episodes += 1
            total_steps += num_steps
            episode_lengths.append(num_steps)

            if args.progress_every > 0 and inspected_episodes % args.progress_every == 0:
                print(
                    f"[progress] root={root} episodes={inspected_episodes} steps={total_steps}",
                    flush=True,
                )
    finally:
        temp_file.close()

    summary = {
        "dataset_root": str(root),
        "resolved_name": f"{builder.info.name}:{builder.info.version}",
        "split": args.split,
        "declared_episodes": declared_episodes,
        "inspected_episodes": inspected_episodes,
        "total_steps": total_steps,
        "avg_steps_per_episode": (total_steps / inspected_episodes) if inspected_episodes else 0.0,
        "step_count_summary": summarize_lengths(episode_lengths),
        "empty_main_instruction_episodes": empty_main_instruction_episodes,
        "instruction_field_names": sorted(instruction_field_names),
    }

    temp_path = Path(temp_file.name)
    try:
        with output_path.open("w", encoding="utf-8") as out_f:
            out_f.write("{\n")
            summary_items = list(summary.items())
            for key, value in summary_items:
                write_json_field(out_f, key, value, trailing_comma=True)
            out_f.write('  "episodes": [\n')
            if temp_path.exists():
                out_f.write(temp_path.read_text(encoding="utf-8"))
            out_f.write("\n  ]\n}\n")
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(
        f"[finish] root={root} inspected={inspected_episodes} output={output_path}",
        flush=True,
    )
    return output_path


def main() -> None:
    args = parse_args()
    export_stats(args)


if __name__ == "__main__":
    main()
