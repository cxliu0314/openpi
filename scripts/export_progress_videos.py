#!/usr/bin/env python3
"""Export per-episode videos that visualize progress targets aligned with training semantics.

This script is intentionally lightweight at runtime: it mirrors the progress/action-chunk
construction used by the training loader for the pi05_libero family without importing the
full training stack.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "progress_videos"
DEFAULT_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)
DEFAULT_DATASET_ROOT_CANDIDATES = (
    Path("/data/HF_Cache_dataevo/lerobot"),
    Path("/data/conda/hug_data/lerobot"),
    Path.home() / ".cache" / "huggingface" / "lerobot",
)


@dataclass(frozen=True)
class ProgressStep:
    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    actions_is_pad: np.ndarray
    frame_index: int
    episode_index: int
    episode_len: int
    timestamp: float
    task_index: int
    global_index: int
    prompt: str
    progress_scalar: float
    progress_chunk: np.ndarray
    episode_hash: int
    formula_error: float


@dataclass(frozen=True)
class ProgressEpisode:
    episode_index: int
    prompt: str
    fps: int
    num_frames: int
    task_index: int
    steps: list[ProgressStep]
    scalar_formula_error_max: float
    first_frame_summary: str


@dataclass(frozen=True)
class ConfigSpec:
    repo_id: str
    prompt_from_task: bool
    action_horizon: int
    progress_target_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-episode progress videos using the same action chunk/progress construction as training.",
    )
    parser.add_argument("--config-name", type=str, required=True, help="Training config name, e.g. pi05_libero.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional explicit LeRobot dataset root. Defaults to auto-resolving from repo_id.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to artifacts/progress_videos/<config-name>.",
    )
    parser.add_argument("--num-videos", type=int, default=20, help="Number of episode videos to export.")
    parser.add_argument(
        "--episode-start",
        type=int,
        default=0,
        help="Start exporting from this episode index when --episode-indices is not set.",
    )
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Comma-separated explicit episode indices to export.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle episode order before selecting videos.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when --shuffle is enabled.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before writing new videos.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="FPS for exported videos. Defaults to the dataset FPS.",
    )
    return parser.parse_args()


def _parse_episode_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    indices = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        indices.append(int(chunk))
    return indices


def _resolve_dataset_root(repo_id: str, explicit_root: str | None = None) -> Path:
    if explicit_root is not None:
        root = Path(explicit_root).expanduser().resolve()
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Dataset root does not look like a LeRobot dataset: {root}")
        return root

    for candidate_base in DEFAULT_DATASET_ROOT_CANDIDATES:
        candidate = (candidate_base / repo_id).resolve()
        if (candidate / "meta" / "info.json").is_file():
            return candidate

    searched = ", ".join(str((base / repo_id).resolve()) for base in DEFAULT_DATASET_ROOT_CANDIDATES)
    raise FileNotFoundError(
        f"Could not resolve local dataset root for repo_id={repo_id!r}. "
        f"Tried: {searched}. Pass --dataset-root explicitly."
    )


def _decode_image(image_cell: dict) -> np.ndarray:
    image = Image.open(io.BytesIO(image_cell["bytes"])).convert("RGB")
    return np.asarray(image)


def _choose_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in DEFAULT_FONT_PATHS:
        path = Path(font_path)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _format_vector(values: np.ndarray, *, precision: int = 3) -> str:
    return np.array2string(
        np.asarray(values, dtype=np.float32),
        precision=precision,
        suppress_small=False,
        max_line_width=200,
    )


def _resolve_config_spec(config_name: str) -> ConfigSpec:
    if config_name.startswith("pi05_libero"):
        return ConfigSpec(
            repo_id="modified_libero_lerobot_split_padded/libero_10_no_noops",
            prompt_from_task=True,
            action_horizon=10,
            progress_target_mode="chunk",
        )
    raise ValueError(
        f"Unsupported config_name={config_name!r}. "
        "This exporter currently targets the pi05_libero family requested for this check."
    )


class LocalLeRobotMetadata:
    def __init__(self, root: Path):
        self.root = root
        with (root / "meta" / "info.json").open("r", encoding="utf-8") as f:
            self.info = json.load(f)
        self.tasks = self._load_jsonl_mapping(root / "meta" / "tasks.jsonl", "task_index", "task")
        self.episodes = self._load_jsonl_mapping(root / "meta" / "episodes.jsonl", "episode_index", None)

    @staticmethod
    def _load_jsonl_mapping(path: Path, key_name: str, value_name: str | None) -> dict:
        mapping = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                mapping[item[key_name]] = item if value_name is None else item[value_name]
        return mapping

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def chunks_size(self) -> int:
        return int(self.info["chunks_size"])

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    def get_data_file_path(self, ep_index: int) -> Path:
        rel = self.info["data_path"].format(
            episode_chunk=self.get_episode_chunk(ep_index),
            episode_index=ep_index,
        )
        return Path(rel)


def _compute_scalar_progress(frame_index: int, episode_len: int) -> float:
    denom = max(float(episode_len - 1), 1.0)
    return float(np.clip(frame_index / denom, 0.0, 1.0))


def _compute_chunk_progress(frame_index: int, episode_len: int, action_horizon: int) -> np.ndarray:
    offsets = np.arange(action_horizon, dtype=np.float32)
    denom = max(float(episode_len - 1), 1.0)
    return np.clip((frame_index + offsets) / denom, 0.0, 1.0).astype(np.float32)


class ProgressEpisodeDataset:
    """Episode-level loader that mirrors the training action/progress construction."""

    def __init__(
        self,
        config_name: str,
        *,
        dataset_root: str | None = None,
        episode_indices: list[int] | None = None,
        num_episodes: int = 20,
        episode_start: int = 0,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self._spec = _resolve_config_spec(config_name)
        self.config_name = config_name
        self.repo_id = self._spec.repo_id
        self.dataset_root = _resolve_dataset_root(self.repo_id, dataset_root)
        self.meta = LocalLeRobotMetadata(self.dataset_root)
        self.action_horizon = self._spec.action_horizon
        self.prompt_from_task = self._spec.prompt_from_task
        self.progress_target_mode = self._spec.progress_target_mode
        self.fps = self.meta.fps

        if episode_indices is None:
            all_indices = list(range(self.meta.total_episodes))
            if shuffle:
                random.Random(seed).shuffle(all_indices)
            start = max(episode_start, 0)
            self.episode_indices = all_indices[start : start + num_episodes]
        else:
            self.episode_indices = episode_indices[:num_episodes]

        if not self.episode_indices:
            raise ValueError("No episode indices selected for export.")

    def __len__(self) -> int:
        return len(self.episode_indices)

    def __getitem__(self, index: int) -> ProgressEpisode:
        episode_index = self.episode_indices[index]
        episode_path = self.dataset_root / self.meta.get_data_file_path(episode_index)
        table = pq.read_table(episode_path)
        rows = table.to_pylist()
        episode_len = len(rows)
        if episode_len == 0:
            raise ValueError(f"Episode {episode_index} is empty: {episode_path}")

        expected_len = int(self.meta.episodes[episode_index]["length"])
        if episode_len != expected_len:
            raise ValueError(
                f"Episode length mismatch for episode {episode_index}: parquet={episode_len}, meta={expected_len}"
            )

        actions_all = np.asarray([row["actions"] for row in rows], dtype=np.float32)
        steps: list[ProgressStep] = []
        scalar_formula_error_max = 0.0
        first_frame_summary = ""

        for local_idx, row in enumerate(rows):
            frame_index = int(row["frame_index"])
            if frame_index != local_idx:
                raise ValueError(
                    f"Episode {episode_index} frame_index mismatch at local_idx={local_idx}: frame_index={frame_index}"
                )

            offsets = np.arange(self.action_horizon, dtype=np.int64)
            unclipped_future = frame_index + offsets
            future_indices = np.clip(unclipped_future, 0, episode_len - 1)
            actions_chunk = actions_all[future_indices].astype(np.float32, copy=False)
            actions_is_pad = (unclipped_future >= episode_len).astype(bool)

            task_index = int(row["task_index"])
            prompt = self.meta.tasks[task_index] if self.prompt_from_task else self.meta.tasks[task_index]
            scalar_progress = _compute_scalar_progress(frame_index, episode_len)
            chunk_progress = _compute_chunk_progress(frame_index, episode_len, self.action_horizon)
            expected_scalar = _compute_scalar_progress(frame_index, episode_len)
            formula_error = abs(float(scalar_progress) - expected_scalar)
            scalar_formula_error_max = max(scalar_formula_error_max, formula_error)
            episode_hash = episode_index

            if local_idx == 0:
                first_frame_summary = (
                    f"episode_hash={episode_hash}\n"
                    f"frame_index={frame_index}\n"
                    f"episode_len={episode_len}\n"
                    f"prompt={prompt}\n"
                    f"state={_format_vector(np.asarray(row['state'], dtype=np.float32))}\n"
                    f"action_t0={_format_vector(actions_chunk[0])}\n"
                    f"future_progress={_format_vector(chunk_progress)}\n"
                    f"actions_is_pad={''.join('1' if v else '0' for v in actions_is_pad.tolist())}"
                )

            steps.append(
                ProgressStep(
                    image=_decode_image(row["image"]),
                    wrist_image=_decode_image(row["wrist_image"]),
                    state=np.asarray(row["state"], dtype=np.float32),
                    actions=np.asarray(actions_chunk, dtype=np.float32),
                    actions_is_pad=np.asarray(actions_is_pad, dtype=bool),
                    frame_index=frame_index,
                    episode_index=episode_index,
                    episode_len=episode_len,
                    timestamp=float(row["timestamp"]),
                    task_index=task_index,
                    global_index=int(row["index"]),
                    prompt=prompt,
                    progress_scalar=float(scalar_progress),
                    progress_chunk=np.asarray(chunk_progress, dtype=np.float32),
                    episode_hash=int(episode_hash),
                    formula_error=float(formula_error),
                )
            )

        return ProgressEpisode(
            episode_index=episode_index,
            prompt=steps[0].prompt,
            fps=self.fps,
            num_frames=episode_len,
            task_index=steps[0].task_index,
            steps=steps,
            scalar_formula_error_max=float(scalar_formula_error_max),
            first_frame_summary=first_frame_summary,
        )


class ProgressEpisodeDataLoader:
    """Simple iterable wrapper over ProgressEpisodeDataset."""

    def __init__(self, dataset: ProgressEpisodeDataset):
        self._dataset = dataset

    def __iter__(self) -> Iterable[ProgressEpisode]:
        for index in range(len(self._dataset)):
            yield self._dataset[index]

    def __len__(self) -> int:
        return len(self._dataset)


def create_progress_episode_loader(
    config_name: str,
    *,
    dataset_root: str | None = None,
    episode_indices: list[int] | None = None,
    num_episodes: int = 20,
    episode_start: int = 0,
    shuffle: bool = False,
    seed: int = 0,
) -> ProgressEpisodeDataLoader:
    dataset = ProgressEpisodeDataset(
        config_name,
        dataset_root=dataset_root,
        episode_indices=episode_indices,
        num_episodes=num_episodes,
        episode_start=episode_start,
        shuffle=shuffle,
        seed=seed,
    )
    return ProgressEpisodeDataLoader(dataset)


def _draw_progress_bar(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], progress: float) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=6, outline=(120, 120, 120), fill=(30, 30, 30), width=1)
    inner_width = max(int((x1 - x0 - 4) * float(np.clip(progress, 0.0, 1.0))), 0)
    if inner_width > 0:
        draw.rounded_rectangle((x0 + 2, y0 + 2, x0 + 2 + inner_width, y1 - 2), radius=5, fill=(76, 175, 80))


def _annotate_frame(
    episode: ProgressEpisode,
    step: ProgressStep,
    *,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> np.ndarray:
    image_size = (384, 384)
    panel_height = 256
    canvas_size = (image_size[0] * 2, image_size[1] + panel_height)

    base = Image.fromarray(step.image).resize(image_size, Image.Resampling.BILINEAR)
    wrist = Image.fromarray(step.wrist_image).resize(image_size, Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", canvas_size, color=(12, 12, 14))
    canvas.paste(base, (0, 0))
    canvas.paste(wrist, (image_size[0], 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, image_size[1], canvas_size[0], canvas_size[1]), fill=(18, 18, 22))
    draw.line((image_size[0], 0, image_size[0], image_size[1]), fill=(55, 55, 60), width=2)
    draw.rectangle((0, 0, 110, 28), fill=(0, 0, 0))
    draw.rectangle((image_size[0], 0, image_size[0] + 130, 28), fill=(0, 0, 0))
    draw.text((8, 6), "base view", font=body_font, fill=(255, 255, 255))
    draw.text((image_size[0] + 8, 6), "wrist view", font=body_font, fill=(255, 255, 255))

    title = f"{episode.prompt} | episode={episode.episode_index} | task_index={episode.task_index}"
    wrapped_title = textwrap.wrap(title, width=72)[:2]
    y = image_size[1] + 10
    for line in wrapped_title:
        draw.text((14, y), line, font=title_font, fill=(240, 240, 240))
        y += 26

    progress_label = (
        f"frame={step.frame_index:03d}/{step.episode_len - 1:03d} "
        f"ts={step.timestamp:6.3f}s "
        f"progress={step.progress_scalar:0.4f} "
        f"hash={step.episode_hash}"
    )
    draw.text((14, y), progress_label, font=body_font, fill=(220, 220, 220))
    y += 24
    _draw_progress_bar(draw, (14, y, canvas_size[0] - 14, y + 18), step.progress_scalar)
    y += 28

    last_valid_offset = int(np.where(~step.actions_is_pad)[0][-1]) if (~step.actions_is_pad).any() else 0
    pad_mask_str = "".join("1" if value else "0" for value in step.actions_is_pad.tolist())
    lines = [
        (
            f"config_progress_mode=chunk | action_horizon={episode.steps[0].progress_chunk.shape[0]} "
            f"chunk-preview "
            f"| scalar_formula_err={step.formula_error:.6f}"
        ),
        (
            f"future_progress={_format_vector(step.progress_chunk)}"
        ),
        (
            f"actions_is_pad={pad_mask_str} last_valid_offset={last_valid_offset} "
            f"tail_padded={bool(step.actions_is_pad.any())}"
        ),
        f"action[t+0]={_format_vector(step.actions[0])}",
        f"action[t+{last_valid_offset}]={_format_vector(step.actions[last_valid_offset])}",
        f"state={_format_vector(step.state)}",
        f"global_index={step.global_index}",
    ]

    for line in lines:
        wrapped_lines = textwrap.wrap(line, width=98, subsequent_indent="  ") or [line]
        for wrapped_line in wrapped_lines[:2]:
            draw.text((14, y), wrapped_line, font=body_font, fill=(205, 205, 205))
            y += 22

    return np.asarray(canvas)


def _episode_manifest_record(output_path: Path, episode: ProgressEpisode) -> dict:
    return {
        "episode_index": episode.episode_index,
        "prompt": episode.prompt,
        "task_index": episode.task_index,
        "num_frames": episode.num_frames,
        "fps": episode.fps,
        "output_path": str(output_path),
        "scalar_formula_error_max": episode.scalar_formula_error_max,
        "first_frame_summary": episode.first_frame_summary,
    }


def _export_episode_video(
    episode: ProgressEpisode,
    output_path: Path,
    *,
    video_fps: float,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=video_fps, codec="libx264") as writer:
        for step in episode.steps:
            writer.append_data(_annotate_frame(episode, step, title_font=title_font, body_font=body_font))


def main() -> None:
    args = parse_args()
    episode_indices = _parse_episode_indices(args.episode_indices)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / args.config_name

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = create_progress_episode_loader(
        args.config_name,
        dataset_root=args.dataset_root,
        episode_indices=episode_indices,
        num_episodes=args.num_videos,
        episode_start=args.episode_start,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    dataset = loader._dataset  # Internal access is fine for this debugging script.
    video_fps = float(args.video_fps or dataset.fps)
    title_font = _choose_font(20)
    body_font = _choose_font(16)

    manifest: list[dict] = []
    print(
        f"[start] config={args.config_name} repo_id={dataset.repo_id} "
        f"dataset_root={dataset.dataset_root} videos={len(loader)} output_dir={output_dir}",
        flush=True,
    )

    for episode in loader:
        output_path = output_dir / f"episode_{episode.episode_index:06d}.mp4"
        print(
            f"[export] episode={episode.episode_index} frames={episode.num_frames} "
            f"prompt={episode.prompt!r} -> {output_path.name}",
            flush=True,
        )
        _export_episode_video(
            episode,
            output_path,
            video_fps=video_fps,
            title_font=title_font,
            body_font=body_font,
        )
        manifest.append(_episode_manifest_record(output_path, episode))

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[done] exported={len(manifest)} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
