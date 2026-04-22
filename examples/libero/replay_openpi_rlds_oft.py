"""Replay OpenPI RLDS training pipelines inside the openvla-oft environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import random
import re
import sys
import traceback
from typing import Any, Iterator, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_ROOT = REPO_ROOT / "third_party" / "libero"
LIBERO_PACKAGE_ROOT = LIBERO_ROOT / "libero" / "libero"

for path in (REPO_ROOT, LIBERO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


LIBERO_OSC_POSITION_SCALE = 0.05
LIBERO_OSC_ROTATION_SCALE = 0.5
LIBERO_GRIPPER_QPOS_SCALE = 0.04


@dataclass(frozen=True)
class ReplaySpec:
    mode: str
    dataset_name: str
    version: str
    data_dir: str
    action_horizon: int
    is_libero_source: bool


@dataclass
class EpisodeRecord:
    mode: str
    episode_index: int
    prompt: str
    source_images: Any
    replay_actions: Any
    action_chunk_shape: list[int]
    state: Optional[Any] = None
    joint_state: Optional[Any] = None
    joint_actions: Optional[Any] = None


SPECS = {
    "unified_libero": ReplaySpec(
        mode="unified_libero",
        dataset_name="libero_10_no_noops",
        version="1.0.0",
        data_dir="/data/Embobrain/dataset_revised/modified_libero_rlds/libero_10_no_noops",
        action_horizon=10,
        is_libero_source=True,
    ),
    "unified_droid": ReplaySpec(
        mode="unified_droid",
        dataset_name="droid",
        version="0.0.1",
        data_dir="/data/Embobrain/dataset_revised/droid/droid_rlds_split_padded",
        action_horizon=10,
        is_libero_source=False,
    ),
    "droid_fallback_joint": ReplaySpec(
        mode="droid_fallback_joint",
        dataset_name="droid",
        version="0.0.1",
        data_dir="/data/Embobrain/dataset_revised/droid/droid_rlds_cleaned",
        action_horizon=16,
        is_libero_source=False,
    ),
}


def parse_args() -> argparse.Namespace:
    default_output = REPO_ROOT / "rollouts" / f"openpi_rlds_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(description="Replay OpenPI RLDS training samples in LIBERO.")
    parser.add_argument("--mode", choices=sorted(SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--scan-limit", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--env-horizon", type=int, default=10000)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--fixed-task-id", type=int, default=0)
    parser.add_argument("--fixed-init-index", type=int, default=0)
    parser.add_argument("--joint-replay-mode", choices=("controller", "direct_set"), default="controller")
    parser.add_argument(
        "--osc-replay-mode",
        choices=("open_loop", "closed_loop_joint_target", "teacher_forced_one_step"),
        default="open_loop",
    )
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def configure_runtime(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / ".libero_config"
    datasets_dir = output_dir / ".libero_datasets"
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "benchmark_root": LIBERO_PACKAGE_ROOT,
        "bddl_files": LIBERO_PACKAGE_ROOT / "bddl_files",
        "init_states": LIBERO_PACKAGE_ROOT / "init_files",
        "datasets": datasets_dir,
        "assets": LIBERO_PACKAGE_ROOT / "assets",
    }
    config_path = config_dir / "config.yaml"
    config_path.write_text("".join(f"{k}: {v}\n" for k, v in config.items()), encoding="utf-8")
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


def decode_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        flat = value.reshape(-1)
    except Exception:
        flat = [value]
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

    array = np.asarray(value)
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
    import numpy as np

    if hasattr(frame, "numpy"):
        frame = frame.numpy()
    if isinstance(frame, np.ndarray) and frame.shape == ():
        frame = frame.item()
    if isinstance(frame, np.ndarray) and frame.dtype == object and frame.size == 1:
        frame = frame.reshape(()).item()

    if isinstance(frame, (bytes, bytearray, memoryview)):
        from PIL import Image

        return np.asarray(Image.open(BytesIO(bytes(frame))).convert("RGB"))

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
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont

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
    mode: str,
    episode_index: int,
    step: int,
    total_steps: int,
    replay_mode: str,
    prompt: str,
) -> str:
    return (
        f"{mode} ep={episode_index} step={step}/{total_steps} mode={replay_mode}\n"
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


def clone_tree(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {key: clone_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [clone_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(child) for child in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def disable_tf_gpu() -> None:
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def resolve_tfds_data_dir(spec: ReplaySpec) -> str:
    path = Path(spec.data_dir)
    if path.name == spec.dataset_name and (path / spec.version).exists():
        return str(path.parent)
    return str(path)


def choose_first_nonempty_instruction(traj: dict[str, Any]) -> str:
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        text = decode_text(traj.get(key))
        if text:
            return text
    return ""


def droid_cartesian_to_libero_state(cartesian_position: Any, gripper_position: Any) -> Any:
    import numpy as np

    cartesian_position = np.asarray(cartesian_position, dtype=np.float32)
    gripper_position = np.asarray(gripper_position, dtype=np.float32)
    xyz = cartesian_position[:, :3]
    rotvec = cartesian_position[:, 3:6]
    openness = 1.0 - gripper_position
    gripper_qpos = LIBERO_GRIPPER_QPOS_SCALE * np.concatenate((openness, -openness), axis=-1)
    return np.concatenate((xyz, rotvec, gripper_qpos), axis=-1).astype(np.float32)


def droid_cartesian_to_libero_action(cartesian_position: Any, gripper_position: Any) -> Any:
    import numpy as np
    from scipy.spatial.transform import Rotation

    cartesian_position = np.asarray(cartesian_position, dtype=np.float32)
    gripper_position = np.asarray(gripper_position, dtype=np.float32)
    current_pose = cartesian_position
    next_pose = np.concatenate((cartesian_position[1:], cartesian_position[-1:]), axis=0)

    delta_pos = (next_pose[:, :3] - current_pose[:, :3]) / LIBERO_OSC_POSITION_SCALE

    current_rot = Rotation.from_rotvec(current_pose[:, 3:6]).as_matrix()
    next_rot = Rotation.from_rotvec(next_pose[:, 3:6]).as_matrix()
    delta_rot = next_rot @ np.transpose(current_rot, (0, 2, 1))
    delta_rot = Rotation.from_matrix(delta_rot).as_rotvec() / LIBERO_OSC_ROTATION_SCALE

    gripper = 2.0 * gripper_position.astype(np.float32) - 1.0
    return np.concatenate((np.clip(np.concatenate((delta_pos, delta_rot), axis=-1), -1.0, 1.0), gripper), axis=-1)


def normalize_gripper_action(action: Any, *, binarize: bool = True) -> Any:
    import numpy as np

    normalized = np.asarray(action, dtype=np.float32).copy()
    normalized[..., -1] = 2.0 * normalized[..., -1] - 1.0
    if binarize:
        normalized[..., -1] = np.sign(normalized[..., -1])
    return normalized


def invert_gripper_action(action: Any) -> Any:
    import numpy as np

    inverted = np.asarray(action, dtype=np.float32).copy()
    inverted[..., -1] *= -1.0
    return inverted


def process_dataset_action(action: Any) -> Any:
    return invert_gripper_action(normalize_gripper_action(action, binarize=True)).astype("float32")


def libero_eef_action_from_obs(current_obs: dict[str, Any], target_obs: dict[str, Any], gripper_open: float) -> Any:
    import numpy as np
    import robosuite.utils.transform_utils as T

    current_pos = np.asarray(current_obs["robot0_eef_pos"], dtype=np.float32)
    target_pos = np.asarray(target_obs["robot0_eef_pos"], dtype=np.float32)
    dpos = (target_pos - current_pos) / LIBERO_OSC_POSITION_SCALE

    current_quat = np.asarray(current_obs["robot0_eef_quat"], dtype=np.float32)
    target_quat = np.asarray(target_obs["robot0_eef_quat"], dtype=np.float32)
    delta_quat = T.quat_multiply(target_quat, T.quat_inverse(current_quat))
    if delta_quat[-1] < 0:
        delta_quat = -delta_quat
    drot = T.quat2axisangle(delta_quat) / LIBERO_OSC_ROTATION_SCALE

    eef_delta = np.clip(np.concatenate((dpos, drot), axis=-1), -1.0, 1.0)
    gripper = np.asarray([np.clip(gripper_open, 0.0, 1.0)], dtype=np.float32)
    return np.concatenate((eef_delta.astype(np.float32), gripper), axis=-1)


def libero_delta_action_from_obs(current_obs: dict[str, Any], target_obs: dict[str, Any], gripper_action: float) -> Any:
    import numpy as np
    import robosuite.utils.transform_utils as T

    current_pos = np.asarray(current_obs["robot0_eef_pos"], dtype=np.float32)
    target_pos = np.asarray(target_obs["robot0_eef_pos"], dtype=np.float32)
    dpos = (target_pos - current_pos) / LIBERO_OSC_POSITION_SCALE

    current_quat = np.asarray(current_obs["robot0_eef_quat"], dtype=np.float32)
    target_quat = np.asarray(target_obs["robot0_eef_quat"], dtype=np.float32)
    delta_quat = T.quat_multiply(target_quat, T.quat_inverse(current_quat))
    if delta_quat[-1] < 0:
        delta_quat = -delta_quat
    drot = T.quat2axisangle(delta_quat) / LIBERO_OSC_ROTATION_SCALE

    eef_delta = np.clip(np.concatenate((dpos, drot), axis=-1), -1.0, 1.0)
    gripper = np.asarray([np.clip(gripper_action, -1.0, 1.0)], dtype=np.float32)
    return np.concatenate((eef_delta.astype(np.float32), gripper), axis=-1)


def quaternion_axis_angle_error(current_quat: Any, target_quat: Any) -> float:
    import numpy as np
    import robosuite.utils.transform_utils as T

    current_quat = np.asarray(current_quat, dtype=np.float32)
    target_quat = np.asarray(target_quat, dtype=np.float32)
    delta_quat = T.quat_multiply(target_quat, T.quat_inverse(current_quat))
    if delta_quat[-1] < 0:
        delta_quat = -delta_quat
    return float(np.linalg.norm(T.quat2axisangle(delta_quat)))


def iter_raw_episodes(spec: ReplaySpec) -> Iterator[tuple[int, dict[str, Any]]]:
    import dlimp as dl
    import tensorflow_datasets as tfds

    disable_tf_gpu()
    builder = tfds.builder(spec.dataset_name, data_dir=resolve_tfds_data_dir(spec), version=spec.version)
    dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=1)
    iterator = dataset.iterator()
    for episode_index, traj in enumerate(iterator):
        yield episode_index, traj


def build_unified_libero_episode(episode_index: int, traj: dict[str, Any], spec: ReplaySpec) -> EpisodeRecord:
    import numpy as np

    actions = np.asarray(traj["action"], dtype=np.float32)
    images = np.asarray([decode_camera_frame(frame) for frame in traj["observation"]["image"]])
    prompt = decode_text(traj["language_instruction"])
    return EpisodeRecord(
        mode=spec.mode,
        episode_index=episode_index,
        prompt=prompt,
        source_images=images,
        replay_actions=actions,
        action_chunk_shape=[int(actions.shape[0]), spec.action_horizon, int(actions.shape[-1])],
        state=np.asarray(traj["observation"]["state"], dtype=np.float32),
    )


def build_unified_droid_episode(
    episode_index: int,
    traj: dict[str, Any],
    spec: ReplaySpec,
    *,
    seed: int,
) -> EpisodeRecord:
    import numpy as np

    episode_rng = random.Random(seed + episode_index)
    use_camera1 = episode_rng.random() > 0.5
    image_key = "exterior_image_1_left" if use_camera1 else "exterior_image_2_left"
    images = np.asarray([decode_camera_frame(frame) for frame in traj["observation"][image_key]])
    cartesian_position = np.asarray(traj["observation"]["cartesian_position"], dtype=np.float32)
    gripper_position = np.asarray(traj["action_dict"]["gripper_position"], dtype=np.float32)
    state_gripper = np.asarray(traj["observation"]["gripper_position"], dtype=np.float32)
    joint_actions = np.concatenate(
        (
            np.asarray(traj["action_dict"]["joint_position"], dtype=np.float32),
            np.asarray(traj["action_dict"]["gripper_position"], dtype=np.float32),
        ),
        axis=-1,
    )
    joint_state = np.concatenate(
        (
            np.asarray(traj["observation"]["joint_position"], dtype=np.float32),
            np.asarray(traj["observation"]["gripper_position"], dtype=np.float32),
        ),
        axis=-1,
    )
    prompt = choose_first_nonempty_instruction(traj)
    actions = droid_cartesian_to_libero_action(cartesian_position, gripper_position)
    state = droid_cartesian_to_libero_state(cartesian_position, state_gripper)
    return EpisodeRecord(
        mode=spec.mode,
        episode_index=episode_index,
        prompt=prompt,
        source_images=images,
        replay_actions=actions.astype(np.float32),
        action_chunk_shape=[int(actions.shape[0]), spec.action_horizon, int(actions.shape[-1])],
        state=state.astype(np.float32),
        joint_state=joint_state.astype(np.float32),
        joint_actions=joint_actions.astype(np.float32),
    )


def build_droid_joint_episode(
    episode_index: int,
    traj: dict[str, Any],
    spec: ReplaySpec,
    *,
    seed: int,
) -> EpisodeRecord:
    import numpy as np

    episode_rng = random.Random(seed + episode_index)
    use_camera1 = episode_rng.random() > 0.5
    image_key = "exterior_image_1_left" if use_camera1 else "exterior_image_2_left"
    images = np.asarray([decode_camera_frame(frame) for frame in traj["observation"][image_key]])
    instruction_candidates = [
        traj["language_instruction"],
        traj["language_instruction_2"],
        traj["language_instruction_3"],
    ]
    prompt = decode_text(episode_rng.choice(instruction_candidates))
    if not prompt:
        prompt = choose_first_nonempty_instruction(traj)
    actions = np.concatenate(
        (
            np.asarray(traj["action_dict"]["joint_position"], dtype=np.float32),
            np.asarray(traj["action_dict"]["gripper_position"], dtype=np.float32),
        ),
        axis=-1,
    )
    state = np.concatenate(
        (
            np.asarray(traj["observation"]["joint_position"], dtype=np.float32),
            np.asarray(traj["observation"]["gripper_position"], dtype=np.float32),
        ),
        axis=-1,
    )
    return EpisodeRecord(
        mode=spec.mode,
        episode_index=episode_index,
        prompt=prompt,
        source_images=images,
        replay_actions=actions,
        action_chunk_shape=[int(actions.shape[0]), spec.action_horizon, int(actions.shape[-1])],
        state=state,
    )


def build_episode(spec: ReplaySpec, episode_index: int, traj: dict[str, Any], *, seed: int) -> EpisodeRecord:
    if spec.mode == "unified_libero":
        return build_unified_libero_episode(episode_index, traj, spec)
    if spec.mode == "unified_droid":
        return build_unified_droid_episode(episode_index, traj, spec, seed=seed)
    if spec.mode == "droid_fallback_joint":
        return build_droid_joint_episode(episode_index, traj, spec, seed=seed)
    raise ValueError(f"Unsupported mode: {spec.mode}")


def get_libero_image(obs: dict[str, Any]) -> Any:
    import numpy as np

    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def set_robot_joint_state_for_replay(env: Any, state: Any, fallback_action: Any) -> None:
    import numpy as np

    state = np.asarray(state, dtype=np.float32)
    if state.ndim > 1:
        state = state[0]
    initial_joints = state[:7] if state.shape[-1] >= 7 else np.asarray(fallback_action[:7], dtype=np.float32)
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


def set_robot_full_state_for_replay(env: Any, state: Any) -> Any:
    set_robot_joint_action_for_replay(env, state, set_gripper=True)
    return get_observations_after_direct_state_set(env)


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


def process_joint_position_action(action: Any, env: Any, max_delta: float = LIBERO_OSC_POSITION_SCALE) -> Any:
    import numpy as np

    raw = np.asarray(action, dtype=np.float32)
    target_joints = raw[:7]
    current_joints = np.asarray(env.env.robots[0]._joint_positions, dtype=np.float32)
    joint_delta_input = np.clip((target_joints - current_joints) / max_delta, -1.0, 1.0)
    gripper_input = np.clip(2.0 * raw[-1:] - 1.0, -1.0, 1.0)
    return np.concatenate((joint_delta_input, gripper_input), axis=-1).astype(np.float32)


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

    def get_init_states(self, task_id: int) -> Any:
        if task_id not in self.init_state_cache:
            self.init_state_cache[task_id] = self.task_suite.get_task_init_states(task_id)
        return self.init_state_cache[task_id]

    def choose_task(self, episode: EpisodeRecord, is_libero_source: bool) -> tuple[int, str, str]:
        if is_libero_source:
            normalized = normalize_instruction(episode.prompt)
            task_id = self.description_to_task_id.get(normalized, self.args.fixed_task_id)
            replay_mode = "instruction_matched" if normalized in self.description_to_task_id else "instruction_fallback_task0"
        else:
            task_id = self.args.fixed_task_id
            replay_mode = "controller_sanity_only"
        return task_id, self.tasks[task_id].language, replay_mode

    def choose_init_index(self, task_id: int, episode: EpisodeRecord, is_libero_source: bool) -> int:
        init_states = self.get_init_states(task_id)
        if is_libero_source:
            return episode.episode_index % len(init_states)
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


def replay_osc_episode(
    context: LiberoReplayContext,
    episode: EpisodeRecord,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    is_libero_source: bool,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    actions = np.asarray(episode.replay_actions, dtype=np.float32)
    replay_actions = actions if args.max_steps <= 0 else actions[: args.max_steps]
    task_id, task_description, replay_mode = context.choose_task(episode, is_libero_source)
    init_index = context.choose_init_index(task_id, episode, is_libero_source)
    instruction_slug = sanitize_filename(episode.prompt or task_description, max_len=80)
    video_path = output_dir / f"{episode.episode_index:06d}--steps={len(replay_actions):03d}--{instruction_slug}.mp4"

    env = context.get_env(task_id, controller="OSC_POSE")
    env.reset()
    obs = env.set_init_state(context.get_init_states(task_id)[init_index])

    writer = imageio.get_writer(video_path, fps=args.fps)
    done_any = False
    last_reward = None
    last_info = None
    try:
        overlay = "" if args.no_overlay else make_overlay_text(
            episode.mode,
            episode.episode_index,
            0,
            len(replay_actions),
            replay_mode,
            truncate_text(episode.prompt),
        )
        writer.append_data(annotate_image(compose_comparison_frame(episode.source_images[0], get_libero_image(obs)), overlay))
        for step, action in enumerate(replay_actions, start=1):
            obs, reward, done, info = env.step(np.asarray(action, dtype=np.float32).tolist())
            done_any = bool(done_any or done)
            last_reward = float(reward)
            last_info = dict(info) if isinstance(info, dict) else {"info_type": type(info).__name__}
            overlay = "" if args.no_overlay else make_overlay_text(
                episode.mode,
                episode.episode_index,
                step,
                len(replay_actions),
                replay_mode,
                truncate_text(episode.prompt),
            )
            source_frame = episode.source_images[min(step, len(episode.source_images) - 1)]
            writer.append_data(annotate_image(compose_comparison_frame(source_frame, get_libero_image(obs)), overlay))
    finally:
        writer.close()

    return {
        "status": "ok",
        "mode": episode.mode,
        "episode_index": episode.episode_index,
        "prompt": episode.prompt,
        "video_path": str(video_path),
        "task_id": task_id,
        "task_description": task_description,
        "init_state_index": init_index,
        "replay_mode": replay_mode,
        "done_any": done_any,
        "last_reward": last_reward,
        "last_info": last_info,
        "action_stats": action_stats(actions),
        "action_chunk_shape": episode.action_chunk_shape,
    }


def replay_osc_episode_closed_loop_joint_target(
    context: LiberoReplayContext,
    episode: EpisodeRecord,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    if episode.joint_actions is None:
        raise ValueError("Closed-loop joint-target replay requires joint_actions.")

    joint_actions = np.asarray(episode.joint_actions, dtype=np.float32)
    replay_actions = joint_actions if args.max_steps <= 0 else joint_actions[: args.max_steps]
    task_id, task_description, _ = context.choose_task(episode, False)
    init_index = context.choose_init_index(task_id, episode, False)
    replay_mode = "osc_closed_loop_joint_target"
    instruction_slug = sanitize_filename(episode.prompt or task_description, max_len=80)
    video_path = output_dir / f"{episode.episode_index:06d}--steps={len(replay_actions):03d}--{instruction_slug}.mp4"

    joint_env = context.get_env(task_id, controller="JOINT_POSITION")
    eef_env = context.get_env(task_id, controller="OSC_POSE")

    joint_env.reset()
    joint_env.set_init_state(context.get_init_states(task_id)[init_index])
    set_robot_joint_state_for_replay(joint_env, episode.joint_state, replay_actions[0])
    joint_obs = get_observations_after_direct_state_set(joint_env)

    eef_env.reset()
    eef_env.set_init_state(context.get_init_states(task_id)[init_index])
    set_robot_joint_state_for_replay(eef_env, episode.joint_state, replay_actions[0])
    eef_obs = get_observations_after_direct_state_set(eef_env)

    writer = imageio.get_writer(video_path, fps=args.fps)
    tracking_error_norms: list[float] = []
    generated_actions: list[Any] = []
    done_any = False
    last_reward = None
    last_info = None
    try:
        overlay = "" if args.no_overlay else make_overlay_text(
            episode.mode,
            episode.episode_index,
            0,
            len(replay_actions),
            replay_mode,
            truncate_text(episode.prompt),
        )
        writer.append_data(annotate_image(compose_comparison_frame(episode.source_images[0], get_libero_image(eef_obs)), overlay))
        for step, raw_joint_action in enumerate(replay_actions, start=1):
            set_robot_joint_action_for_replay(joint_env, raw_joint_action)
            joint_obs = get_observations_after_direct_state_set(joint_env)

            raw_eef_action = libero_eef_action_from_obs(
                eef_obs,
                joint_obs,
                gripper_open=1.0 - float(raw_joint_action[-1]),
            )
            env_action = process_dataset_action(raw_eef_action)
            generated_actions.append(env_action)

            eef_obs, reward, done, info = eef_env.step(np.asarray(env_action, dtype=np.float32).tolist())
            done_any = bool(done_any or done)
            last_reward = float(reward)
            last_info = dict(info) if isinstance(info, dict) else {"info_type": type(info).__name__}
            tracking_error_norms.append(
                float(
                    np.linalg.norm(
                        np.asarray(joint_obs["robot0_eef_pos"], dtype=np.float32)
                        - np.asarray(eef_obs["robot0_eef_pos"], dtype=np.float32)
                    )
                )
            )

            overlay = "" if args.no_overlay else make_overlay_text(
                episode.mode,
                episode.episode_index,
                step,
                len(replay_actions),
                replay_mode,
                truncate_text(episode.prompt),
            )
            source_frame = episode.source_images[min(step, len(episode.source_images) - 1)]
            writer.append_data(annotate_image(compose_comparison_frame(source_frame, get_libero_image(eef_obs)), overlay))
    finally:
        writer.close()

    generated_action_array = np.stack(generated_actions, axis=0) if generated_actions else np.empty((0, 7), dtype=np.float32)
    return {
        "status": "ok",
        "mode": episode.mode,
        "episode_index": episode.episode_index,
        "prompt": episode.prompt,
        "video_path": str(video_path),
        "task_id": task_id,
        "task_description": task_description,
        "init_state_index": init_index,
        "replay_mode": replay_mode,
        "done_any": done_any,
        "last_reward": last_reward,
        "last_info": last_info,
        "tracking_error_norm_min": min(tracking_error_norms) if tracking_error_norms else None,
        "tracking_error_norm_max": max(tracking_error_norms) if tracking_error_norms else None,
        "tracking_error_norm_mean": float(np.mean(tracking_error_norms)) if tracking_error_norms else None,
        "action_stats": action_stats(generated_action_array),
        "action_chunk_shape": episode.action_chunk_shape,
    }


def replay_osc_episode_teacher_forced_one_step(
    context: LiberoReplayContext,
    episode: EpisodeRecord,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    if episode.joint_state is None:
        raise ValueError("Teacher-forced one-step validation requires joint_state.")

    loader_actions = np.asarray(episode.replay_actions, dtype=np.float32)
    joint_states = np.asarray(episode.joint_state, dtype=np.float32)
    total_steps = min(int(loader_actions.shape[0]), max(int(joint_states.shape[0]) - 1, 0))
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    if total_steps <= 0:
        raise ValueError("Teacher-forced one-step validation requires at least two joint states.")

    task_id, task_description, _ = context.choose_task(episode, False)
    init_index = context.choose_init_index(task_id, episode, False)
    init_state = context.get_init_states(task_id)[init_index]
    replay_mode = "osc_teacher_forced_one_step"
    instruction_slug = sanitize_filename(episode.prompt or task_description, max_len=80)
    video_path = output_dir / f"{episode.episode_index:06d}--steps={total_steps:03d}--{instruction_slug}.mp4"

    ref_env = context.get_env(task_id, controller="JOINT_POSITION")
    osc_env = context.get_env(task_id, controller="OSC_POSE")

    writer = imageio.get_writer(video_path, fps=args.fps)
    reference_actions: list[Any] = []
    action_arm_error_norms: list[float] = []
    position_error_norms: list[float] = []
    rotation_error_rads: list[float] = []
    done_any = False
    last_reward = None
    last_info = None
    try:
        osc_env.reset()
        osc_env.set_init_state(init_state)
        osc_obs = set_robot_full_state_for_replay(osc_env, joint_states[0])
        overlay = "" if args.no_overlay else make_overlay_text(
            episode.mode,
            episode.episode_index,
            0,
            total_steps,
            replay_mode,
            truncate_text(episode.prompt),
        )
        writer.append_data(annotate_image(compose_comparison_frame(episode.source_images[0], get_libero_image(osc_obs)), overlay))

        for step_index in range(total_steps):
            current_state = joint_states[step_index]
            target_state = joint_states[step_index + 1]
            loader_action = loader_actions[step_index]

            ref_env.reset()
            ref_env.set_init_state(init_state)
            ref_current_obs = clone_tree(set_robot_full_state_for_replay(ref_env, current_state))
            ref_target_obs = clone_tree(set_robot_full_state_for_replay(ref_env, target_state))

            reference_action = libero_delta_action_from_obs(
                ref_current_obs,
                ref_target_obs,
                gripper_action=float(loader_action[-1]),
            )
            reference_actions.append(reference_action)
            action_arm_error_norms.append(float(np.linalg.norm(loader_action[:6] - reference_action[:6])))

            osc_env.reset()
            osc_env.set_init_state(init_state)
            set_robot_full_state_for_replay(osc_env, current_state)
            osc_next_obs, reward, done, info = osc_env.step(loader_action.tolist())
            done_any = bool(done_any or done)
            last_reward = float(reward)
            last_info = dict(info) if isinstance(info, dict) else {"info_type": type(info).__name__}

            position_error_norms.append(
                float(
                    np.linalg.norm(
                        np.asarray(osc_next_obs["robot0_eef_pos"], dtype=np.float32)
                        - np.asarray(ref_target_obs["robot0_eef_pos"], dtype=np.float32)
                    )
                )
            )
            rotation_error_rads.append(
                quaternion_axis_angle_error(
                    osc_next_obs["robot0_eef_quat"],
                    ref_target_obs["robot0_eef_quat"],
                )
            )

            overlay = "" if args.no_overlay else make_overlay_text(
                episode.mode,
                episode.episode_index,
                step_index + 1,
                total_steps,
                replay_mode,
                truncate_text(episode.prompt),
            )
            source_frame = episode.source_images[min(step_index + 1, len(episode.source_images) - 1)]
            writer.append_data(annotate_image(compose_comparison_frame(source_frame, get_libero_image(osc_next_obs)), overlay))
    finally:
        writer.close()

    reference_action_array = (
        np.stack(reference_actions, axis=0) if reference_actions else np.empty((0, 7), dtype=np.float32)
    )
    return {
        "status": "ok",
        "mode": episode.mode,
        "episode_index": episode.episode_index,
        "prompt": episode.prompt,
        "video_path": str(video_path),
        "task_id": task_id,
        "task_description": task_description,
        "init_state_index": init_index,
        "replay_mode": replay_mode,
        "done_any": done_any,
        "last_reward": last_reward,
        "last_info": last_info,
        "steps_validated": total_steps,
        "loader_vs_reference_action_arm_l2_min": min(action_arm_error_norms) if action_arm_error_norms else None,
        "loader_vs_reference_action_arm_l2_max": max(action_arm_error_norms) if action_arm_error_norms else None,
        "loader_vs_reference_action_arm_l2_mean": (
            float(np.mean(action_arm_error_norms)) if action_arm_error_norms else None
        ),
        "teacher_forced_pos_error_m_min": min(position_error_norms) if position_error_norms else None,
        "teacher_forced_pos_error_m_max": max(position_error_norms) if position_error_norms else None,
        "teacher_forced_pos_error_m_mean": float(np.mean(position_error_norms)) if position_error_norms else None,
        "teacher_forced_rot_error_rad_min": min(rotation_error_rads) if rotation_error_rads else None,
        "teacher_forced_rot_error_rad_max": max(rotation_error_rads) if rotation_error_rads else None,
        "teacher_forced_rot_error_rad_mean": float(np.mean(rotation_error_rads)) if rotation_error_rads else None,
        "teacher_forced_rot_error_deg_mean": (
            float(np.degrees(np.mean(rotation_error_rads))) if rotation_error_rads else None
        ),
        "teacher_forced_pos_error_lt_2cm_rate": (
            float(np.mean(np.asarray(position_error_norms) < 0.02)) if position_error_norms else None
        ),
        "teacher_forced_rot_error_lt_15deg_rate": (
            float(np.mean(np.asarray(rotation_error_rads) < np.deg2rad(15.0))) if rotation_error_rads else None
        ),
        "reference_action_stats": action_stats(reference_action_array),
        "loader_action_stats": action_stats(loader_actions[:total_steps]),
        "action_chunk_shape": episode.action_chunk_shape,
    }


def replay_joint_episode(
    context: LiberoReplayContext,
    episode: EpisodeRecord,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    actions = np.asarray(episode.replay_actions, dtype=np.float32)
    replay_actions = actions if args.max_steps <= 0 else actions[: args.max_steps]
    task_id, task_description, _ = context.choose_task(episode, False)
    init_index = context.choose_init_index(task_id, episode, False)
    replay_mode = (
        "libero_joint_position_direct_set" if args.joint_replay_mode == "direct_set" else "libero_joint_position_controller"
    )
    instruction_slug = sanitize_filename(episode.prompt or task_description, max_len=80)
    video_path = output_dir / f"{episode.episode_index:06d}--steps={len(replay_actions):03d}--{instruction_slug}.mp4"

    env = context.get_env(task_id, controller="JOINT_POSITION")
    env.reset()
    env.set_init_state(context.get_init_states(task_id)[init_index])
    set_robot_joint_state_for_replay(env, episode.state, replay_actions[0])
    obs = get_observations_after_direct_state_set(env)

    writer = imageio.get_writer(video_path, fps=args.fps)
    joint_error_norms: list[float] = []
    done_any = False
    last_reward = None
    last_info = None
    try:
        overlay = "" if args.no_overlay else make_overlay_text(
            episode.mode,
            episode.episode_index,
            0,
            len(replay_actions),
            replay_mode,
            truncate_text(episode.prompt),
        )
        writer.append_data(annotate_image(compose_comparison_frame(episode.source_images[0], get_libero_image(obs)), overlay))
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
                episode.mode,
                episode.episode_index,
                step,
                len(replay_actions),
                replay_mode,
                truncate_text(episode.prompt),
            )
            source_frame = episode.source_images[min(step, len(episode.source_images) - 1)]
            writer.append_data(annotate_image(compose_comparison_frame(source_frame, get_libero_image(obs)), overlay))
    finally:
        writer.close()

    return {
        "status": "ok",
        "mode": episode.mode,
        "episode_index": episode.episode_index,
        "prompt": episode.prompt,
        "video_path": str(video_path),
        "task_id": task_id,
        "task_description": task_description,
        "init_state_index": init_index,
        "replay_mode": replay_mode,
        "done_any": done_any,
        "last_reward": last_reward,
        "last_info": last_info,
        "joint_error_norm_min": min(joint_error_norms) if joint_error_norms else None,
        "joint_error_norm_max": max(joint_error_norms) if joint_error_norms else None,
        "joint_error_norm_mean": float(np.mean(joint_error_norms)) if joint_error_norms else None,
        "action_stats": action_stats(actions),
        "action_chunk_shape": episode.action_chunk_shape,
    }


def make_failure_record(mode: str, episode_index: Optional[int], stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "status": "error",
        "mode": mode,
        "episode_index": episode_index,
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(limit=8),
    }


def write_summary(output_dir: Path, args: argparse.Namespace, records: list[dict[str, Any]]) -> Path:
    numeric_metric_keys = (
        "loader_vs_reference_action_arm_l2_mean",
        "teacher_forced_pos_error_m_mean",
        "teacher_forced_rot_error_rad_mean",
        "teacher_forced_rot_error_deg_mean",
        "teacher_forced_pos_error_lt_2cm_rate",
        "teacher_forced_rot_error_lt_15deg_rate",
        "tracking_error_norm_mean",
        "joint_error_norm_mean",
    )
    ok_records = [record for record in records if record.get("status") == "ok"]
    aggregated_metrics: dict[str, float] = {}
    if ok_records:
        import numpy as np

        for key in numeric_metric_keys:
            values = [float(record[key]) for record in ok_records if record.get(key) is not None]
            if values:
                aggregated_metrics[key] = float(np.mean(values))

    summary = {
        "status": "complete",
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.jsonl"),
        "num_records": len(records),
        "num_success": sum(record.get("status") == "ok" for record in records),
        "num_errors": sum(record.get("status") == "error" for record in records),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "aggregated_metrics": aggregated_metrics,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    spec = SPECS[args.mode]
    args.output_dir = args.output_dir.expanduser().resolve()
    config_path = configure_runtime(args.output_dir)
    configure_torch_serialization_compat()
    random.seed(args.seed)

    output_dir = args.output_dir / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    completed = 0
    scanned = 0

    print(f"[replay] output_dir={output_dir}", flush=True)
    print(f"[replay] runtime_libero_config={config_path}", flush=True)
    print(f"[replay] mode={args.mode}", flush=True)

    context = LiberoReplayContext(args)
    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for episode_index, traj in iter_raw_episodes(spec):
                if completed >= args.num_episodes or scanned >= args.scan_limit:
                    break
                scanned += 1
                try:
                    episode = build_episode(spec, episode_index, traj, seed=args.seed)
                    if episode.replay_actions.shape[-1] == 8:
                        record = replay_joint_episode(context, episode, output_dir, args)
                    elif args.mode == "unified_droid" and args.osc_replay_mode == "teacher_forced_one_step":
                        record = replay_osc_episode_teacher_forced_one_step(
                            context,
                            episode,
                            output_dir,
                            args,
                        )
                    elif args.mode == "unified_droid" and args.osc_replay_mode == "closed_loop_joint_target":
                        record = replay_osc_episode_closed_loop_joint_target(
                            context,
                            episode,
                            output_dir,
                            args,
                        )
                    else:
                        record = replay_osc_episode(
                            context,
                            episode,
                            output_dir,
                            args,
                            is_libero_source=spec.is_libero_source,
                        )
                except Exception as exc:
                    record = make_failure_record(args.mode, episode_index, "build_or_replay", exc)

                write_jsonl(manifest, record)
                records.append(record)
                if record.get("status") == "ok":
                    completed += 1
                    print(f"[replay] saved {completed}/{args.num_episodes}: {record['video_path']}", flush=True)
                else:
                    print(
                        f"[replay] error episode={episode_index}: {record['error_type']}: {record['error']}",
                        flush=True,
                    )
    finally:
        context.close()

    summary_path = write_summary(output_dir, args, records)
    print(f"[replay] manifest={manifest_path}", flush=True)
    print(f"[replay] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
