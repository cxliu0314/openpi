"""Replay OpenPI training-pipeline trajectories through LIBERO for visual validation."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
import re
import sys
import traceback
from typing import Any, Iterator, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
LIBERO_ROOT = REPO_ROOT / "third_party" / "libero"
LIBERO_PACKAGE_ROOT = LIBERO_ROOT / "libero" / "libero"

for path in (REPO_ROOT, SRC_ROOT, LIBERO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@dataclass
class StepRecord:
    episode_key: str
    episode_index: int
    frame_index: int
    prompt: str
    source_image: Any
    replay_action: Any
    normalized_action: Any
    state: Any
    replay_action_chunk_shape: list[int]
    normalized_action_chunk_shape: list[int]


@dataclass
class EpisodeRecord:
    episode_key: str
    episode_index: int
    prompt: str
    source_images: Any
    replay_actions: Any
    normalized_actions: Any
    states: Any
    replay_action_chunk_shape: list[int]
    normalized_action_chunk_shape: list[int]


def parse_args() -> argparse.Namespace:
    default_output = REPO_ROOT / "rollouts" / f"openpi_training_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(
        description="Replay trajectories reconstructed from the OpenPI training pipeline.",
    )
    parser.add_argument("--config-name", required=True, help="OpenPI training config name.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Output directory.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of videos to save.")
    parser.add_argument(
        "--scan-steps",
        type=int,
        default=20000,
        help="Maximum number of training samples to scan before stopping.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum replay steps per episode. <=0 means full.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--resolution", type=int, default=256, help="LIBERO render resolution.")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS.")
    parser.add_argument("--env-horizon", type=int, default=10000, help="LIBERO env horizon.")
    parser.add_argument("--task-suite", default="libero_10", help="LIBERO task suite.")
    parser.add_argument("--fixed-task-id", type=int, default=0, help="Fallback task id for non-LIBERO sources.")
    parser.add_argument("--fixed-init-index", type=int, default=0, help="Fallback init-state index.")
    parser.add_argument(
        "--joint-replay-mode",
        choices=("controller", "direct_set"),
        default="controller",
        help="Only used when the reconstructed action is 8D joint-position action.",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Disable overlay text.")
    return parser.parse_args()


def configure_runtime(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / ".libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = output_dir / ".libero_datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "benchmark_root": LIBERO_PACKAGE_ROOT,
        "bddl_files": LIBERO_PACKAGE_ROOT / "bddl_files",
        "init_states": LIBERO_PACKAGE_ROOT / "init_files",
        "datasets": datasets_dir,
        "assets": LIBERO_PACKAGE_ROOT / "assets",
    }
    config_text = "".join(f"{key}: {value}\n" for key, value in config.items())
    config_path = config_dir / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return config_path


def configure_torch_serialization_compat() -> None:
    import numpy as np
    import torch

    try:
        torch.serialization.add_safe_globals(
            [
                np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                type(np.dtype(np.float32)),
                type(np.dtype(np.float64)),
                type(np.dtype(np.int64)),
            ]
        )
    except Exception:
        pass

    original_torch_load = torch.load

    def _compat_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load


def to_numpy(value: Any) -> Any:
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def tree_to_numpy(tree: Any) -> Any:
    import jax
    import numpy as np

    def _convert(x: Any) -> Any:
        try:
            return np.asarray(to_numpy(x))
        except Exception:
            return x

    return jax.tree.map(_convert, tree)


def decode_text(value: Any) -> str:
    if value is None:
        return ""
    array = to_numpy(value)
    try:
        flat = array.reshape(-1)
    except Exception:
        flat = [array]
    for item in flat:
        if isinstance(item, bytes):
            text = item.decode("utf-8", errors="replace")
        else:
            text = str(item)
        if text and text != "None":
            return text
    return ""


def scalar_int(value: Any) -> int:
    import numpy as np

    array = np.asarray(to_numpy(value))
    if array.size == 0:
        return 0
    return int(array.reshape(-1)[0])


def sanitize_filename(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unnamed")[:max_len]


def normalize_instruction(text: str) -> str:
    text = text.lower().replace(".", " ")
    return " ".join(text.split())


def decode_camera_frame(frame: Any) -> Any:
    from io import BytesIO

    import numpy as np

    if hasattr(frame, "numpy"):
        frame = frame.numpy()
    if isinstance(frame, np.ndarray) and frame.shape == ():
        frame = frame.item()
    if isinstance(frame, np.ndarray) and frame.dtype == object and frame.size == 1:
        frame = frame.reshape(()).item()

    if isinstance(frame, (bytes, bytearray, memoryview)):
        from PIL import Image

        image = Image.open(BytesIO(bytes(frame))).convert("RGB")
        return np.asarray(image)

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def annotate_image(image: Any, text: str) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if not text:
        return image

    pil_image = Image.fromarray(np.asarray(image))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    padding = 6
    max_width = pil_image.size[0] - 2 * padding

    lines: list[str] = []
    for raw_line in text.split("\n"):
        current: list[str] = []
        for word in raw_line.split():
            candidate = " ".join(current + [word])
            try:
                width = draw.textlength(candidate, font=font)
            except AttributeError:
                width = draw.textbbox((0, 0), candidate, font=font)[2]
            if width <= max_width or not current:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))

    bbox = font.getbbox("Ag")
    line_height = bbox[3] - bbox[1] + 2
    box_height = padding * 2 + line_height * len(lines)
    draw.rectangle([0, 0, pil_image.size[0], box_height], fill=(0, 0, 0))
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_height
    return np.asarray(pil_image)


def compose_comparison_frame(source_image: Any, replay_image: Any) -> Any:
    import numpy as np
    from PIL import Image

    replay = decode_camera_frame(replay_image)
    source = decode_camera_frame(source_image)
    source_pil = Image.fromarray(source).resize((replay.shape[1], replay.shape[0]))
    replay_pil = Image.fromarray(replay)
    canvas = Image.new("RGB", (replay.shape[1] * 2, replay.shape[0]))
    canvas.paste(source_pil, (0, 0))
    canvas.paste(replay_pil, (replay.shape[1], 0))
    return np.asarray(canvas)


def action_stats(actions: Any) -> dict[str, Any]:
    import numpy as np

    actions = np.asarray(actions)
    if actions.size == 0:
        return {"shape": list(actions.shape)}
    values = actions.reshape(-1, actions.shape[-1]) if actions.ndim > 1 else actions.reshape(1, -1)
    return {
        "shape": list(actions.shape),
        "min": np.nanmin(values, axis=0).astype(float).tolist(),
        "max": np.nanmax(values, axis=0).astype(float).tolist(),
        "mean": np.nanmean(values, axis=0).astype(float).tolist(),
    }


def make_overlay_text(
    config_name: str,
    source_kind: str,
    episode_index: int,
    step: int,
    total_steps: int,
    replay_mode: str,
    prompt: str,
) -> str:
    return (
        f"{config_name} [{source_kind}] ep={episode_index} step={step}/{total_steps} mode={replay_mode}\n"
        "left=training_image right=libero_replay\n"
        f"{prompt}"
    )


def write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def truncate_text(text: str, max_len: int = 110) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def extract_source_image(training_sample: dict[str, Any]) -> Any:
    image_dict = training_sample.get("image")
    if not isinstance(image_dict, dict) or not image_dict:
        raise ValueError("Training sample does not contain image dict.")
    for key in ("base_0_rgb", "cam_high", "image_primary", "left_wrist_0_rgb", "wrist_0_rgb"):
        if key in image_dict:
            return decode_camera_frame(image_dict[key])
    first_key = next(iter(image_dict))
    return decode_camera_frame(image_dict[first_key])


def extract_episode_key(sample: dict[str, Any], fallback_step: int) -> tuple[str, int]:
    if "step_id" in sample:
        step_id = decode_text(sample["step_id"])
        match = re.match(r"^(.*)--(\d+)$", step_id)
        if match is not None:
            return match.group(1), int(match.group(2))
        return step_id or f"step-{fallback_step}", scalar_int(sample.get("frame_index", 0))
    if "episode_index" in sample:
        episode_index = scalar_int(sample["episode_index"])
        return f"episode-{episode_index:06d}", scalar_int(sample.get("frame_index", 0))
    return f"step-{fallback_step}", fallback_step


def get_libero_image(obs: dict[str, Any]) -> Any:
    import numpy as np

    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


LIBERO_OSC_POSITION_SCALE = 0.05


def process_joint_position_action(action: Any, env: Any, max_delta: float = LIBERO_OSC_POSITION_SCALE) -> Any:
    import numpy as np

    raw = np.asarray(action, dtype=np.float32)
    target_joints = raw[:7]
    current_joints = np.asarray(env.env.robots[0]._joint_positions, dtype=np.float32)
    joint_delta_input = np.clip((target_joints - current_joints) / max_delta, -1.0, 1.0)
    gripper_input = np.clip(2.0 * raw[-1:] - 1.0, -1.0, 1.0)
    return np.concatenate((joint_delta_input, gripper_input), axis=-1).astype(np.float32)


def set_robot_joint_state_for_replay(env: Any, state: Any, fallback_action: Any) -> None:
    import numpy as np

    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] >= 7:
        initial_joints = state[:7]
    else:
        initial_joints = np.asarray(fallback_action[:7], dtype=np.float32)
    env.env.robots[0].set_robot_joint_positions(initial_joints)


def set_robot_joint_action_for_replay(env: Any, action: Any, set_gripper: bool = True) -> Any:
    import numpy as np

    raw = np.asarray(action, dtype=np.float32)
    robot = env.env.robots[0]
    robot.set_robot_joint_positions(raw[:7])

    gripper_qpos = None
    if set_gripper and raw.shape[0] >= 8 and getattr(robot, "has_gripper", False):
        indexes = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
        init_qpos = getattr(getattr(robot, "gripper", None), "init_qpos", None)
        if indexes is not None and init_qpos is not None:
            openness = float(np.clip(1.0 - raw[-1], 0.0, 1.0))
            gripper_qpos = openness * np.asarray(init_qpos, dtype=np.float32)
            robot.sim.data.qpos[indexes] = gripper_qpos
            robot.sim.forward()
    return gripper_qpos


def get_observations_after_direct_state_set(env: Any) -> Any:
    env.env.sim.forward()
    try:
        env._post_process()
    except Exception:
        pass
    try:
        env._update_observables(force=True)
    except Exception:
        pass
    try:
        return env.env._get_observations(force_update=True)
    except TypeError:
        return env.env._get_observations()


class TrainingPipelineProjector:
    def __init__(self, data_config: Any):
        from openpi import transforms as openpi_transforms

        self._standardize = openpi_transforms.compose(
            [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
        )
        self._normalize = openpi_transforms.Normalize(
            data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
        self._unnormalize = openpi_transforms.Unnormalize(
            data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
        self._model_inputs = list(data_config.model_transforms.inputs)
        self._replay_output = openpi_transforms.compose(
            [*data_config.model_transforms.outputs, *data_config.data_transforms.outputs]
        )

    def process_step(self, raw_sample: dict[str, Any], *, fallback_step: int) -> StepRecord:
        import numpy as np

        sample = tree_to_numpy(raw_sample)
        episode_key, frame_index = extract_episode_key(sample, fallback_step)
        episode_index = scalar_int(sample.get("episode_index", fallback_step))

        standardized = self._standardize(copy.deepcopy(sample))
        prompt = decode_text(standardized.get("prompt"))

        training_sample = self._normalize(copy.deepcopy(standardized))
        for transform in self._model_inputs:
            training_sample = transform(training_sample)

        source_image = extract_source_image(training_sample)
        normalized_actions = np.asarray(to_numpy(training_sample["actions"]), dtype=np.float32)

        replay_payload = self._unnormalize(copy.deepcopy(training_sample))
        replay_payload = self._replay_output(replay_payload)
        replay_actions = np.asarray(to_numpy(replay_payload["actions"]), dtype=np.float32)
        state = np.asarray(to_numpy(self._unnormalize(copy.deepcopy(training_sample))["state"]), dtype=np.float32)

        if replay_actions.ndim == 1:
            replay_actions = replay_actions[None, :]
        if normalized_actions.ndim == 1:
            normalized_actions = normalized_actions[None, :]

        return StepRecord(
            episode_key=episode_key,
            episode_index=episode_index,
            frame_index=frame_index,
            prompt=prompt,
            source_image=source_image,
            replay_action=np.asarray(replay_actions[0], dtype=np.float32),
            normalized_action=np.asarray(normalized_actions[0], dtype=np.float32),
            state=state,
            replay_action_chunk_shape=list(replay_actions.shape),
            normalized_action_chunk_shape=list(normalized_actions.shape),
        )


class LiberoReplayContext:
    def __init__(self, args: argparse.Namespace):
        from libero.libero import benchmark

        self.args = args
        self.task_suite = benchmark.get_benchmark(args.task_suite)()
        self.tasks = [self.task_suite.get_task(i) for i in range(self.task_suite.n_tasks)]
        self.description_to_task_id = {
            normalize_instruction(task.language): task_id for task_id, task in enumerate(self.tasks)
        }
        self.env_cache: dict[tuple[int, str], Any] = {}
        self.init_state_cache: dict[int, Any] = {}

    def choose_task(self, *, prompt: str, is_libero_source: bool) -> tuple[int, str, str]:
        if is_libero_source:
            normalized = normalize_instruction(prompt)
            task_id = self.description_to_task_id.get(normalized, self.args.fixed_task_id)
            replay_mode = (
                "instruction_matched" if normalized in self.description_to_task_id else "instruction_fallback_task0"
            )
        else:
            task_id = self.args.fixed_task_id
            replay_mode = "controller_sanity_only"
        task = self.tasks[task_id]
        return task_id, task.language, replay_mode

    def get_init_states(self, task_id: int) -> Any:
        if task_id not in self.init_state_cache:
            self.init_state_cache[task_id] = self.task_suite.get_task_init_states(task_id)
        return self.init_state_cache[task_id]

    def choose_init_index(self, *, task_id: int, episode_index: int, is_libero_source: bool) -> int:
        init_states = self.get_init_states(task_id)
        if is_libero_source:
            return episode_index % len(init_states)
        return self.args.fixed_init_index % len(init_states)

    def get_env(self, task_id: int, controller: str = "OSC_POSE") -> Any:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        cache_key = (task_id, controller)
        if cache_key not in self.env_cache:
            task = self.tasks[task_id]
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                controller=controller,
                camera_heights=self.args.resolution,
                camera_widths=self.args.resolution,
                horizon=self.args.env_horizon,
                ignore_done=True,
            )
            env.seed(self.args.seed)
            self.env_cache[cache_key] = env
        return self.env_cache[cache_key]

    def close(self) -> None:
        for env in self.env_cache.values():
            try:
                env.close()
            except Exception:
                pass
        self.env_cache.clear()


def finalize_episode(steps: list[StepRecord]) -> EpisodeRecord:
    import numpy as np

    prompt = next((step.prompt for step in steps if step.prompt), "")
    return EpisodeRecord(
        episode_key=steps[0].episode_key,
        episode_index=steps[0].episode_index,
        prompt=prompt,
        source_images=np.stack([decode_camera_frame(step.source_image) for step in steps], axis=0),
        replay_actions=np.stack([step.replay_action for step in steps], axis=0),
        normalized_actions=np.stack([step.normalized_action for step in steps], axis=0),
        states=np.stack([step.state for step in steps], axis=0),
        replay_action_chunk_shape=steps[0].replay_action_chunk_shape,
        normalized_action_chunk_shape=steps[0].normalized_action_chunk_shape,
    )


def iter_lerobot_steps(train_config: Any, data_config: Any, scan_steps: int) -> Iterator[dict[str, Any]]:
    from openpi.training import data_loader as openpi_data_loader

    dataset = openpi_data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    total = min(len(dataset), scan_steps)
    for index in range(total):
        yield dataset[index]


def iter_rlds_steps(train_config: Any, data_config: Any, scan_steps: int) -> Iterator[dict[str, Any]]:
    import jax

    from openpi.training import data_loader as openpi_data_loader

    dataset = openpi_data_loader.create_rlds_dataset(
        data_config,
        action_horizon=train_config.model.action_horizon,
        batch_size=1,
        shuffle=False,
    )
    for index, batch in enumerate(dataset):
        if index >= scan_steps:
            break
        sample = tree_to_numpy(batch)
        yield jax.tree.map(lambda x: x[0], sample)


def replay_source_only_episode(
    *,
    episode: EpisodeRecord,
    config_name: str,
    source_kind: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    actions = episode.replay_actions
    replay_actions = actions if args.max_steps <= 0 else actions[: args.max_steps]
    instruction_slug = sanitize_filename(episode.prompt or episode.episode_key, max_len=80)
    video_path = output_dir / (
        f"{episode.episode_index:06d}--steps={len(replay_actions):03d}--{instruction_slug}.mp4"
    )

    writer = imageio.get_writer(video_path, fps=args.fps)
    try:
        total_steps = int(replay_actions.shape[0])
        for step in range(total_steps + 1):
            image = episode.source_images[min(step, len(episode.source_images) - 1)]
            overlay = "" if args.no_overlay else make_overlay_text(
                config_name,
                source_kind,
                episode.episode_index,
                step,
                total_steps,
                "source_only",
                truncate_text(episode.prompt),
            )
            writer.append_data(annotate_image(image, overlay))
    finally:
        writer.close()

    return {
        "status": "ok",
        "episode_index": episode.episode_index,
        "episode_key": episode.episode_key,
        "prompt": episode.prompt,
        "video_path": str(video_path),
        "replay_mode": "source_only",
        "action_stats": action_stats(actions),
        "normalized_action_stats": action_stats(episode.normalized_actions),
        "replay_action_chunk_shape": episode.replay_action_chunk_shape,
        "normalized_action_chunk_shape": episode.normalized_action_chunk_shape,
    }


def replay_osc_episode(
    *,
    context: LiberoReplayContext,
    episode: EpisodeRecord,
    config_name: str,
    source_kind: str,
    is_libero_source: bool,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    actions = np.asarray(episode.replay_actions, dtype=np.float32)
    replay_actions = actions if args.max_steps <= 0 else actions[: args.max_steps]
    task_id, task_description, replay_mode = context.choose_task(
        prompt=episode.prompt,
        is_libero_source=is_libero_source,
    )
    init_index = context.choose_init_index(
        task_id=task_id,
        episode_index=episode.episode_index,
        is_libero_source=is_libero_source,
    )

    instruction_slug = sanitize_filename(episode.prompt or task_description, max_len=80)
    video_path = output_dir / (
        f"{episode.episode_index:06d}--steps={len(replay_actions):03d}--{instruction_slug}.mp4"
    )
    env = context.get_env(task_id, controller="OSC_POSE")
    env.reset()
    obs = env.set_init_state(context.get_init_states(task_id)[init_index])

    writer = imageio.get_writer(video_path, fps=args.fps)
    done_any = False
    last_reward: float | None = None
    last_info: dict[str, Any] | None = None
    try:
        overlay = "" if args.no_overlay else make_overlay_text(
            config_name,
            source_kind,
            episode.episode_index,
            0,
            len(replay_actions),
            replay_mode,
            truncate_text(episode.prompt),
        )
        writer.append_data(annotate_image(
            compose_comparison_frame(episode.source_images[0], get_libero_image(obs)),
            overlay,
        ))

        for step, action in enumerate(replay_actions, start=1):
            obs, reward, done, info = env.step(np.asarray(action, dtype=np.float32).tolist())
            done_any = bool(done_any or done)
            last_reward = float(reward)
            last_info = dict(info) if isinstance(info, dict) else {"info_type": type(info).__name__}
            overlay = "" if args.no_overlay else make_overlay_text(
                config_name,
                source_kind,
                episode.episode_index,
                step,
                len(replay_actions),
                replay_mode,
                truncate_text(episode.prompt),
            )
            source_frame = episode.source_images[min(step, len(episode.source_images) - 1)]
            writer.append_data(annotate_image(
                compose_comparison_frame(source_frame, get_libero_image(obs)),
                overlay,
            ))
    finally:
        writer.close()

    return {
        "status": "ok",
        "episode_index": episode.episode_index,
        "episode_key": episode.episode_key,
        "prompt": episode.prompt,
        "video_path": str(video_path),
        "replay_mode": replay_mode,
        "task_id": task_id,
        "task_description": task_description,
        "init_state_index": init_index,
        "done_any": done_any,
        "last_reward": last_reward,
        "last_info": last_info,
        "action_stats": action_stats(actions),
        "normalized_action_stats": action_stats(episode.normalized_actions),
        "replay_action_chunk_shape": episode.replay_action_chunk_shape,
        "normalized_action_chunk_shape": episode.normalized_action_chunk_shape,
    }


def replay_joint_episode(
    *,
    context: LiberoReplayContext,
    episode: EpisodeRecord,
    config_name: str,
    source_kind: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    actions = np.asarray(episode.replay_actions, dtype=np.float32)
    replay_actions = actions if args.max_steps <= 0 else actions[: args.max_steps]
    task_id, task_description, _ = context.choose_task(prompt=episode.prompt, is_libero_source=False)
    init_index = context.choose_init_index(
        task_id=task_id,
        episode_index=episode.episode_index,
        is_libero_source=False,
    )

    replay_mode = (
        "libero_joint_position_direct_set"
        if args.joint_replay_mode == "direct_set"
        else "libero_joint_position_controller"
    )
    instruction_slug = sanitize_filename(episode.prompt or task_description, max_len=80)
    video_path = output_dir / (
        f"{episode.episode_index:06d}--steps={len(replay_actions):03d}--{instruction_slug}.mp4"
    )

    env = context.get_env(task_id, controller="JOINT_POSITION")
    env.reset()
    env.set_init_state(context.get_init_states(task_id)[init_index])
    set_robot_joint_state_for_replay(env, episode.states[0], replay_actions[0])
    obs = get_observations_after_direct_state_set(env)

    writer = imageio.get_writer(video_path, fps=args.fps)
    done_any = False
    last_reward: float | None = None
    last_info: dict[str, Any] | None = None
    joint_error_norms: list[float] = []
    try:
        overlay = "" if args.no_overlay else make_overlay_text(
            config_name,
            source_kind,
            episode.episode_index,
            0,
            len(replay_actions),
            replay_mode,
            truncate_text(episode.prompt),
        )
        writer.append_data(annotate_image(
            compose_comparison_frame(episode.source_images[0], get_libero_image(obs)),
            overlay,
        ))

        for step, raw_action in enumerate(replay_actions, start=1):
            current_joints = np.asarray(env.env.robots[0]._joint_positions, dtype=np.float32)
            joint_error_norms.append(float(np.linalg.norm(np.asarray(raw_action[:7], dtype=np.float32) - current_joints)))
            if args.joint_replay_mode == "direct_set":
                set_robot_joint_action_for_replay(env, raw_action)
                obs = get_observations_after_direct_state_set(env)
                last_info = {"mode": "direct_set"}
            else:
                env_action = process_joint_position_action(raw_action, env)
                obs, reward, done, info = env.step(env_action.tolist())
                done_any = bool(done_any or done)
                last_reward = float(reward)
                last_info = dict(info) if isinstance(info, dict) else {"info_type": type(info).__name__}

            overlay = "" if args.no_overlay else make_overlay_text(
                config_name,
                source_kind,
                episode.episode_index,
                step,
                len(replay_actions),
                replay_mode,
                truncate_text(episode.prompt),
            )
            source_frame = episode.source_images[min(step, len(episode.source_images) - 1)]
            writer.append_data(annotate_image(
                compose_comparison_frame(source_frame, get_libero_image(obs)),
                overlay,
            ))
    finally:
        writer.close()

    return {
        "status": "ok",
        "episode_index": episode.episode_index,
        "episode_key": episode.episode_key,
        "prompt": episode.prompt,
        "video_path": str(video_path),
        "replay_mode": replay_mode,
        "task_id": task_id,
        "task_description": task_description,
        "init_state_index": init_index,
        "done_any": done_any,
        "last_reward": last_reward,
        "last_info": last_info,
        "joint_error_norm_min": min(joint_error_norms) if joint_error_norms else None,
        "joint_error_norm_max": max(joint_error_norms) if joint_error_norms else None,
        "joint_error_norm_mean": float(np.mean(joint_error_norms)) if joint_error_norms else None,
        "action_stats": action_stats(actions),
        "normalized_action_stats": action_stats(episode.normalized_actions),
        "replay_action_chunk_shape": episode.replay_action_chunk_shape,
        "normalized_action_chunk_shape": episode.normalized_action_chunk_shape,
    }


def make_failure_record(episode_index: Optional[int], stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "status": "error",
        "episode_index": episode_index,
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(limit=8),
    }


def detect_source_kind(data_factory: Any) -> tuple[str, bool]:
    from openpi.training import config as openpi_config
    from openpi.training import rlds_pi05_dataset

    if isinstance(data_factory, openpi_config.LeRobotLiberoDataConfig):
        return "libero_lerobot", True
    if isinstance(data_factory, openpi_config.RLDSPi05LiberoStyleDataConfig):
        if data_factory.adapter_kind == rlds_pi05_dataset.Pi05RLDSAdapterKind.LIBERO:
            return "libero_rlds_uni", True
        return "droid_rlds_uni", False
    if isinstance(data_factory, openpi_config.RLDSDroidDataConfig):
        return "droid_rlds_joint", False
    raise ValueError(
        "Unsupported data config for replay. "
        "Supported: LeRobotLiberoDataConfig, RLDSPi05LiberoStyleDataConfig, RLDSDroidDataConfig."
    )


def choose_iterator(train_config: Any, data_factory: Any, data_config: Any, scan_steps: int) -> Iterator[dict[str, Any]]:
    from openpi.training import config as openpi_config

    if isinstance(data_factory, openpi_config.LeRobotLiberoDataConfig):
        return iter_lerobot_steps(train_config, data_config, scan_steps)
    return iter_rlds_steps(train_config, data_config, scan_steps)


def write_summary(output_dir: Path, args: argparse.Namespace, config_name: str, records: list[dict[str, Any]]) -> Path:
    summary = {
        "status": "complete",
        "config_name": config_name,
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.jsonl"),
        "num_records": len(records),
        "num_success": sum(record.get("status") == "ok" for record in records),
        "num_errors": sum(record.get("status") == "error" for record in records),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    import numpy as np

    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    config_path = configure_runtime(args.output_dir)
    configure_torch_serialization_compat()

    random.seed(args.seed)
    np.random.seed(args.seed)

    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(args.config_name)
    data_factory = train_config.data
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    source_kind, is_libero_source = detect_source_kind(data_factory)
    iterator = choose_iterator(train_config, data_factory, data_config, args.scan_steps)
    projector = TrainingPipelineProjector(data_config)

    output_dir = args.output_dir / args.config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    records: list[dict[str, Any]] = []

    print(f"[replay] output_dir={output_dir}", flush=True)
    print(f"[replay] config={args.config_name}", flush=True)
    print(f"[replay] runtime_libero_config={config_path}", flush=True)
    print(f"[replay] source_kind={source_kind}", flush=True)

    context = LiberoReplayContext(args)
    current_steps: list[StepRecord] = []
    current_key: str | None = None
    completed = 0
    scanned = 0

    def _flush_episode(handle: Any) -> None:
        nonlocal current_steps, completed
        if not current_steps or completed >= args.num_episodes:
            current_steps = []
            return
        episode = finalize_episode(current_steps)
        try:
            action_dim = int(episode.replay_actions.shape[-1])
            if action_dim == 7:
                record = replay_osc_episode(
                    context=context,
                    episode=episode,
                    config_name=args.config_name,
                    source_kind=source_kind,
                    is_libero_source=is_libero_source,
                    output_dir=output_dir,
                    args=args,
                )
            elif action_dim == 8:
                record = replay_joint_episode(
                    context=context,
                    episode=episode,
                    config_name=args.config_name,
                    source_kind=source_kind,
                    output_dir=output_dir,
                    args=args,
                )
            else:
                record = replay_source_only_episode(
                    episode=episode,
                    config_name=args.config_name,
                    source_kind=source_kind,
                    output_dir=output_dir,
                    args=args,
                )
        except Exception as exc:
            record = make_failure_record(episode.episode_index, "replay", exc)

        record["config_name"] = args.config_name
        record["source_kind"] = source_kind
        record["episode_length"] = len(episode.replay_actions)
        write_jsonl(handle, record)
        records.append(record)
        if record.get("status") == "ok":
            completed += 1
            print(
                f"[replay] saved {completed}/{args.num_episodes}: {record.get('video_path', 'n/a')}",
                flush=True,
            )
        else:
            print(
                f"[replay] error episode={record.get('episode_index')}: {record.get('error_type')}: {record.get('error')}",
                flush=True,
            )
        current_steps = []

    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for raw_sample in iterator:
                if scanned >= args.scan_steps or completed >= args.num_episodes:
                    break
                scanned += 1
                try:
                    step = projector.process_step(raw_sample, fallback_step=scanned - 1)
                except Exception as exc:
                    record = make_failure_record(None, "project_step", exc)
                    record["config_name"] = args.config_name
                    record["source_kind"] = source_kind
                    write_jsonl(manifest, record)
                    records.append(record)
                    continue

                if current_key is None:
                    current_key = step.episode_key
                if current_steps and step.episode_key != current_key:
                    _flush_episode(manifest)
                    current_key = step.episode_key

                current_steps.append(step)

            if current_steps and completed < args.num_episodes:
                _flush_episode(manifest)
    finally:
        context.close()

    summary_path = write_summary(output_dir, args, args.config_name, records)
    print(f"[replay] manifest={manifest_path}", flush=True)
    print(f"[replay] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
