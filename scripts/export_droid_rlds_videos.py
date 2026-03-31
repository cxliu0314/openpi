#!/usr/bin/env python3
"""Minimal RLDS video export helpers used by export_all_rlds_dataset_vis.py."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import tensorflow_datasets as tfds


def resolve_builder(root: str):
    root_path = Path(root)
    version_infos = sorted(root_path.rglob("dataset_info.json"))
    if not version_infos:
        raise FileNotFoundError(f"No dataset_info.json found under {root_path}")
    version_dir = version_infos[0].parent
    return tfds.builder_from_directory(str(version_dir))


def build_decoders(builder):  # pylint: disable=unused-argument
    return None


def build_output_path(vis_dir: Path, episode_index: int, anomalies: list[str]) -> Path:
    suffix = "_anomaly" if anomalies else ""
    return vis_dir / f"episode_{episode_index:04d}{suffix}.mp4"


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_text(value.item())
        for item in value.reshape(-1):
            text = _decode_text(item)
            if text:
                return text
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip() if value is not None else ""


def _find_first_text(step_list: list[dict[str, Any]]) -> str:
    for step in step_list:
        for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
            if key not in step:
                continue
            text = _decode_text(step[key])
            if text:
                return text
    return ""


def _iter_image_candidates(node: Any, prefix: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            next_prefix = f"{prefix}/{key}" if prefix else key
            yield from _iter_image_candidates(value, next_prefix)
        return

    if isinstance(node, np.ndarray) and node.ndim >= 3 and node.shape[-1] in (1, 3, 4):
        yield prefix, node


def _choose_camera(observation: dict[str, Any], preferred_camera_key: str | None):
    candidates = list(_iter_image_candidates(observation))
    if not candidates:
        raise ValueError("No image observations found")

    if preferred_camera_key:
        for key, value in candidates:
            if key.endswith(preferred_camera_key) or preferred_camera_key in key.split("/"):
                return key, value

    priority = [
        "exterior_image_1_left",
        "image",
        "agentview_rgb",
        "rgb",
        "wrist_image_left",
        "wrist_image",
    ]
    for wanted in priority:
        for key, value in candidates:
            if key.endswith(wanted):
                return key, value

    return candidates[0]


def _get_by_path(node: dict[str, Any], key_path: str):
    value = node
    for part in key_path.split("/"):
        value = value[part]
    return value


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def prepare_episode_video_data(
    episode: dict[str, Any],
    episode_index: int,
    preferred_camera_key: str | None = None,
) -> dict[str, Any]:
    step_list = list(episode["steps"])
    if not step_list:
        raise ValueError("Episode has no steps")

    camera_key, _ = _choose_camera(step_list[0]["observation"], preferred_camera_key)
    frames = [_get_by_path(step["observation"], camera_key) for step in step_list]
    instruction = _find_first_text(step_list)
    anomalies = [] if instruction else ["missing_instruction"]

    processed_frames = [_to_uint8_rgb(frame) for frame in frames]
    return {
        "episode_index": episode_index,
        "camera_key": camera_key,
        "instruction": instruction,
        "frames": processed_frames,
        "anomalies": anomalies,
    }


def _annotate_frame(frame: np.ndarray, title: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    wrapped = textwrap.wrap(title, width=48)[:3]
    if wrapped:
        box_height = 8 + 18 * len(wrapped)
        draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0))
        for idx, line in enumerate(wrapped):
            draw.text((8, 4 + idx * 18), line, fill=(255, 255, 255))
    return np.asarray(image)


def write_episode_video(episode_data: dict[str, Any], output_path: Path, fps: float) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = episode_data["instruction"] or episode_data["camera_key"]

    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        for frame in episode_data["frames"]:
            writer.append_data(_annotate_frame(frame, title))

    return {
        "episode_index": episode_data["episode_index"],
        "camera_key": episode_data["camera_key"],
        "instruction": episode_data["instruction"],
        "num_frames": len(episode_data["frames"]),
        "output_path": str(output_path),
        "anomalies": episode_data["anomalies"],
    }
