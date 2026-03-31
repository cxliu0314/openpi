#!/usr/bin/env python3
"""Inspect DROID-style RLDS datasets for empty-instruction ratios and text distributions."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


INSTRUCTION_FIELDS = (
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
)

DEFAULT_DATASETS = (
    "/data/Embobrain/dataset/droid_rlds_cleaned",
    "/data/Embobrain/dataset/droid_rlds_split_padded",
    "/data/dataset/droid",
)

TOP_K = 20
FILLER_TOKENS = {
    "a",
    "an",
    "and",
    "arm",
    "arms",
    "by",
    "can",
    "do",
    "for",
    "from",
    "gripper",
    "hand",
    "into",
    "its",
    "left",
    "of",
    "on",
    "onto",
    "please",
    "right",
    "robot",
    "the",
    "their",
    "then",
    "this",
    "to",
    "toward",
    "up",
    "use",
    "using",
    "with",
    "your",
}
ACTION_ALIASES = {
    "bring": "move",
    "close": "push",
    "drop": "release",
    "flip": "flip",
    "grab": "grasp",
    "grasp": "grasp",
    "hold": "hold",
    "insert": "insert",
    "lift": "grasp",
    "move": "move",
    "open": "pull",
    "pick": "grasp",
    "pickup": "grasp",
    "place": "move",
    "plug": "insert",
    "position": "move",
    "press": "press",
    "pull": "pull",
    "push": "push",
    "put": "move",
    "reach": "reach",
    "release": "release",
    "reposition": "move",
    "rotate": "rotate",
    "set": "move",
    "slide": "move",
    "take": "grasp",
    "touch": "reach",
    "turn": "rotate",
    "twist": "rotate",
}
FUNCTION_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass
class EpisodeSummary:
    num_steps: int
    field_values: dict[str, str]
    field_inconsistent: dict[str, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check empty instruction ratios and text/category distributions for DROID RLDS datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset root path. Can be passed multiple times. Defaults to the three requested datasets.",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Only inspect the first N episodes of each dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="How many top items to print for category / verb / instruction counts.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the full report as JSON.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N episodes. Set <= 0 to disable.",
    )
    return parser.parse_args()


def decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if hasattr(value, "item"):
        try:
            return decode_text(value.item())
        except ValueError:
            pass
    return str(value).strip()


def safe_percent(numerator: int, denominator: int) -> float:
    return (100.0 * numerator / denominator) if denominator else 0.0


def summarize_lengths(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {
            "min": 0,
            "p25": 0,
            "median": 0,
            "mean": 0.0,
            "p75": 0,
            "p90": 0,
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


def compact_counter(counter: Counter[str], top_k: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(top_k)]


def extract_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def extract_leading_verb(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return "__empty__"

    func_match = FUNCTION_CALL_RE.match(normalized)
    if func_match:
        return func_match.group(1).lower()

    tokens = extract_tokens(normalized)
    for token in tokens:
        if token not in FILLER_TOKENS:
            return token
    return "__unknown__"


def extract_action_category(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return "__empty__"

    func_match = FUNCTION_CALL_RE.match(normalized)
    if func_match:
        return func_match.group(1).lower()

    for token in extract_tokens(normalized):
        if token in ACTION_ALIASES:
            return ACTION_ALIASES[token]

    return "__other__"


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
    ordered_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            ordered_candidates.append(candidate)
            seen.add(candidate)

    last_error: Exception | None = None
    for candidate in ordered_candidates:
        try:
            return tfds.builder_from_directory(str(candidate))
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_error = exc

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"Could not find a TFDS dataset directory under: {dataset_path}")


def build_decoders(builder: tfds.core.DatasetBuilder) -> dict[str, Any] | None:
    features = builder.info.features
    if "steps" not in features or "observation" not in features["steps"]:
        return None

    observation = features["steps"]["observation"]
    image_keys = [key for key in observation.keys() if "image" in key]
    if not image_keys:
        return None

    return {
        "steps": {
            "observation": {
                key: tfds.decode.SkipDecoding()
                for key in image_keys
            }
        }
    }


def summarize_episode(episode: dict[str, Any]) -> EpisodeSummary:
    num_steps = 0
    first_non_empty: dict[str, str] = {field: "" for field in INSTRUCTION_FIELDS}
    unique_values: dict[str, set[str]] = {field: set() for field in INSTRUCTION_FIELDS}

    steps_iterable = episode["steps"]
    if hasattr(steps_iterable, "as_numpy_iterator"):
        step_iterator = steps_iterable.as_numpy_iterator()
    else:
        step_iterator = iter(steps_iterable)

    for step in step_iterator:
        num_steps += 1
        for field in INSTRUCTION_FIELDS:
            if field not in step:
                continue
            value = decode_text(step[field])
            unique_values[field].add(value)
            if value and not first_non_empty[field]:
                first_non_empty[field] = value

    field_inconsistent = {
        field: len(unique_values[field]) > 1 for field in INSTRUCTION_FIELDS
    }
    return EpisodeSummary(
        num_steps=num_steps,
        field_values=first_non_empty,
        field_inconsistent=field_inconsistent,
    )


def inspect_dataset(
    dataset_path: str,
    limit_episodes: int | None,
    top_k: int,
    progress_every: int,
) -> dict[str, Any]:
    builder = resolve_builder(dataset_path)
    dataset_name = f"{builder.info.name}:{builder.info.version}"
    num_examples = builder.info.splits["train"].num_examples
    print(f"[start] dataset={dataset_path} resolved={dataset_name}", flush=True)

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    ds = builder.as_dataset(
        split="train",
        shuffle_files=False,
        decoders=build_decoders(builder),
    )

    total_episodes = 0
    total_steps = 0
    main_empty_episodes = 0
    all_empty_episodes = 0
    slot_empty_episodes = Counter()
    slot_inconsistent_episodes = Counter()
    non_empty_slot_count = Counter()
    step_lengths: list[int] = []
    category_counts = Counter()
    leading_verb_counts = Counter()
    exact_instruction_counts = Counter()

    iterable = ds.take(limit_episodes) if limit_episodes is not None else ds
    for episode in tfds.as_numpy(iterable):
        summary = summarize_episode(episode)
        total_episodes += 1
        total_steps += summary.num_steps
        step_lengths.append(summary.num_steps)

        if not summary.field_values["language_instruction"]:
            main_empty_episodes += 1

        empty_slot_count = 0
        for field in INSTRUCTION_FIELDS:
            if not summary.field_values[field]:
                slot_empty_episodes[field] += 1
                empty_slot_count += 1
            if summary.field_inconsistent[field]:
                slot_inconsistent_episodes[field] += 1
        non_empty_slot_count[len(INSTRUCTION_FIELDS) - empty_slot_count] += 1

        if empty_slot_count == len(INSTRUCTION_FIELDS):
            all_empty_episodes += 1

        primary_instruction = next(
            (summary.field_values[field] for field in INSTRUCTION_FIELDS if summary.field_values[field]),
            "",
        )
        if primary_instruction:
            exact_instruction_counts[primary_instruction] += 1
        category_counts[extract_action_category(primary_instruction)] += 1
        leading_verb_counts[extract_leading_verb(primary_instruction)] += 1

        if progress_every > 0 and total_episodes % progress_every == 0:
            print(
                f"[progress] dataset={dataset_path} episodes={total_episodes} steps={total_steps}",
                flush=True,
            )

    report = {
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "declared_train_episodes": int(num_examples),
        "inspected_episodes": total_episodes,
        "total_steps": total_steps,
        "avg_steps_per_episode": (total_steps / total_episodes) if total_episodes else 0.0,
        "step_length_summary": summarize_lengths(step_lengths),
        "empty_instruction_ratio": {
            "language_instruction": {
                "count": main_empty_episodes,
                "ratio_pct": safe_percent(main_empty_episodes, total_episodes),
            },
            "all_instruction_slots_empty": {
                "count": all_empty_episodes,
                "ratio_pct": safe_percent(all_empty_episodes, total_episodes),
            },
        },
        "slot_empty_ratio": {
            field: {
                "count": slot_empty_episodes[field],
                "ratio_pct": safe_percent(slot_empty_episodes[field], total_episodes),
            }
            for field in INSTRUCTION_FIELDS
        },
        "slot_inconsistent_ratio": {
            field: {
                "count": slot_inconsistent_episodes[field],
                "ratio_pct": safe_percent(slot_inconsistent_episodes[field], total_episodes),
            }
            for field in INSTRUCTION_FIELDS
        },
        "non_empty_instruction_slot_distribution": {
            str(k): v for k, v in sorted(non_empty_slot_count.items())
        },
        "primary_action_category_topk": compact_counter(category_counts, top_k),
        "leading_verb_topk": compact_counter(leading_verb_counts, top_k),
        "primary_instruction_topk": compact_counter(exact_instruction_counts, top_k),
    }
    return report


def print_report(report: dict[str, Any], top_k: int) -> None:
    print("=" * 100)
    print(f"Dataset: {report['dataset_path']}")
    print(f"Resolved builder: {report['dataset_name']}")
    print(
        f"Episodes inspected: {report['inspected_episodes']} / declared train episodes: {report['declared_train_episodes']}"
    )
    print(f"Total steps: {report['total_steps']}")
    print(f"Average steps / episode: {report['avg_steps_per_episode']:.2f}")

    length_summary = report["step_length_summary"]
    print(
        "Step length summary: "
        f"min={length_summary['min']}, "
        f"p25={length_summary['p25']:.1f}, "
        f"median={length_summary['median']:.1f}, "
        f"mean={length_summary['mean']:.2f}, "
        f"p75={length_summary['p75']:.1f}, "
        f"p90={length_summary['p90']:.1f}, "
        f"max={length_summary['max']}"
    )

    empty_ratio = report["empty_instruction_ratio"]
    print(
        "Empty ratio: "
        f"language_instruction={empty_ratio['language_instruction']['count']} "
        f"({empty_ratio['language_instruction']['ratio_pct']:.2f}%), "
        f"all 3 slots empty={empty_ratio['all_instruction_slots_empty']['count']} "
        f"({empty_ratio['all_instruction_slots_empty']['ratio_pct']:.2f}%)"
    )

    print("Per-slot empty ratio:")
    for field in INSTRUCTION_FIELDS:
        info = report["slot_empty_ratio"][field]
        print(f"  - {field}: {info['count']} ({info['ratio_pct']:.2f}%)")

    print("Per-slot inconsistent-across-steps ratio:")
    for field in INSTRUCTION_FIELDS:
        info = report["slot_inconsistent_ratio"][field]
        print(f"  - {field}: {info['count']} ({info['ratio_pct']:.2f}%)")

    print("Non-empty instruction slot distribution per trajectory:")
    for slot_count, count in report["non_empty_instruction_slot_distribution"].items():
        print(f"  - {slot_count} non-empty slots: {count}")

    print(f"Top {top_k} primary action categories:")
    for item in report["primary_action_category_topk"]:
        print(f"  - {item['key']}: {item['count']}")

    print(f"Top {top_k} leading verbs:")
    for item in report["leading_verb_topk"]:
        print(f"  - {item['key']}: {item['count']}")

    print(f"Top {top_k} exact primary instructions:")
    for item in report["primary_instruction_topk"]:
        print(f"  - {item['count']}: {item['key']}")


def main() -> None:
    args = parse_args()
    datasets = args.datasets or list(DEFAULT_DATASETS)

    reports = []
    for dataset_path in datasets:
        report = inspect_dataset(
            dataset_path,
            limit_episodes=args.limit_episodes,
            top_k=args.top_k,
            progress_every=args.progress_every,
        )
        reports.append(report)
        print_report(report, top_k=args.top_k)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        print("=" * 100)
        print(f"Saved JSON report to: {output_path}")


if __name__ == "__main__":
    main()
