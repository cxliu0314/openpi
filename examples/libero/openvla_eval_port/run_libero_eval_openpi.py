"""
OpenPI LIBERO evaluator with OpenVLA-style control logic.

This script is isolated from the existing `examples/libero/main.py` flow.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
from enum import Enum
from typing import Dict, List, Optional, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from examples.libero.openvla_eval_port.libero_replanner import (
    get_default_recovery_plan,
    get_recovery_plan,
)
from examples.libero.openvla_eval_port.libero_subtask import SubtaskStatus, SubtaskTracker

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 2000,  # extended to support replan-heavy runs
    TaskSuite.LIBERO_90: 400,
}


@dataclasses.dataclass
class Args:
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    num_open_loop_steps: int = 8

    # LIBERO eval
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL.value
    num_steps_wait: int = 20
    num_trials_per_task: int = 10
    initial_states_path: str = "DEFAULT"
    env_img_res: int = LIBERO_ENV_RESOLUTION
    seed: int = 7

    # logging / video
    run_id_note: Optional[str] = None
    local_log_dir: str = "examples/libero/logs"
    rollout_dir_name: str = "openvla_eval_port"
    save_video: bool = True

    # subtask control
    use_subtasks: bool = True
    subtask_hold_steps: int = 1
    subtask_debug: bool = False
    subtask_switch_mode: str = "env"  # "env" or "progress"
    progress_threshold: float = 0.95
    progress_rollback_enabled: bool = True
    progress_thresholds_by_primitive: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "reach": 0.9,
            "grasp": 0.6,
            "release": 0.6,
            "move": 0.9,
            "rotate": 0.8,
            "push": 0.9,
            "flip": 0.9,
            "insert": 0.9,
            "press": 0.9,
        }
    )
    progress_key: str = "progress"

    # replanning
    use_llm_replan: bool = False
    llm_replan_model: str = "gemini-2.5-flash"

    # action post-processing (disabled by default for OpenPI)
    normalize_gripper: bool = False
    invert_gripper: bool = False


def _setup_logging(args: Args) -> Tuple[logging.Logger, pathlib.Path]:
    run_id = f"EVAL-{args.task_suite_name}-openpi-port"
    if args.run_id_note:
        run_id += f"--{args.run_id_note}"
    log_dir = pathlib.Path(args.local_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.txt"
    logger = logging.getLogger("openvla_eval_port")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_path))
    return logger, log_path


def _validate_args(args: Args) -> None:
    valid = {suite.value for suite in TaskSuite}
    if args.task_suite_name not in valid:
        raise ValueError(f"Invalid task suite {args.task_suite_name}. Must be one of {sorted(valid)}")
    if args.subtask_switch_mode not in {"env", "progress"}:
        raise ValueError("subtask_switch_mode must be 'env' or 'progress'")
    if args.subtask_switch_mode == "progress" and not args.use_subtasks:
        raise ValueError("progress mode requires use_subtasks=True")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _prepare_observation(obs: Dict, resize_size: int) -> Tuple[Dict, np.ndarray]:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }
    return element, img


def _normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    out = np.array(action, dtype=np.float32).copy()
    grip = out[-1]
    if binarize:
        grip = 1.0 if grip >= 0.5 else 0.0
    out[-1] = 2.0 * grip - 1.0
    return out


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    out = np.array(action, dtype=np.float32).copy()
    out[-1] = -out[-1]
    return out


def _process_action(action: np.ndarray, args: Args) -> np.ndarray:
    out = np.array(action, dtype=np.float32)
    if args.normalize_gripper:
        out = _normalize_gripper_action(out, binarize=True)
    if args.invert_gripper:
        out = _invert_gripper_action(out)
    return out


def _get_init_action_sequence(num_move_steps: int = 10, step_size: float = 0.8, release_steps: int = 3):
    actions = []
    for _ in range(release_steps):
        actions.append(np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32))
    for _ in range(num_move_steps):
        actions.append(np.array([0, 0, step_size, 0, 0, 0, 1.0], dtype=np.float32))
    return actions


def _get_hold_action():
    return np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def _extract_progress(infer_result: Dict, args: Args) -> Optional[float]:
    if args.progress_key in infer_result:
        return float(infer_result[args.progress_key])
    if "progress_pred" in infer_result:
        return float(infer_result["progress_pred"])
    return None


def _run_episode(
    args: Args,
    client: _websocket_client_policy.WebsocketClientPolicy,
    env,
    task_name: str,
    task_description: str,
    initial_state: np.ndarray,
    logger: logging.Logger,
) -> Tuple[bool, List[np.ndarray], List[str], SubtaskStatus, List[float]]:
    env.reset()
    obs = env.set_init_state(initial_state)

    action_queue = collections.deque(maxlen=args.num_open_loop_steps)
    primitive_action_queue = collections.deque()
    replay_images: List[np.ndarray] = []
    replay_texts: List[str] = []
    progress_predictions: List[float] = []

    subtask_tracker = SubtaskTracker(
        env,
        task_name,
        task_description,
        enabled=args.use_subtasks,
        hold_steps=args.subtask_hold_steps,
        debug=args.subtask_debug,
        debug_log=logger.info if args.subtask_debug else None,
    )

    subtask_progress_history: Dict[int, List[float]] = {}
    subtask_max_progress: Dict[int, float] = {}
    subtask_start_progress: Dict[int, float] = {}
    recent_progress_window = collections.deque(maxlen=5)
    consecutive_above_threshold = 0
    current_progress: Optional[float] = None

    replan_cooldown = 0
    REPLAN_COOLDOWN_STEPS = 15
    total_replan_count = 0
    current_replan_status: Optional[str] = None
    progress_check_cooldown = 0

    success = False
    t = 0
    max_steps = TASK_MAX_STEPS[TaskSuite(args.task_suite_name)]

    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            if done:
                success = True
                break
            continue

        element, frame = _prepare_observation(obs, args.resize_size)
        replay_images.append(frame)

        if args.subtask_switch_mode == "env":
            subtask_tracker.update(obs, step_idx=t)
            if subtask_tracker.advanced:
                action_queue.clear()
                consecutive_above_threshold = 0
                recent_progress_window.clear()

        current_instruction = subtask_tracker.current_instruction()
        status = subtask_tracker.status()
        overlay = f"step {t} | {current_instruction} ({status.completed}/{status.total})"
        if current_progress is not None:
            overlay += f" | progress: {current_progress:.3f}"
        if current_replan_status is not None:
            overlay += f" | [LLM REPLAN #{total_replan_count}: {current_replan_status}]"
        if replan_cooldown > 0:
            overlay += f" | [cooldown: {replan_cooldown}]"
        replay_texts.append(overlay)

        current_subtask_idx = subtask_tracker.index
        current_primitive = None
        if current_subtask_idx < len(subtask_tracker.plan):
            current_subtask = subtask_tracker.plan[current_subtask_idx]
            current_primitive = current_subtask.get("primitive") or current_subtask.get("rule")

        if current_primitive in {"init", "hold"}:
            if current_primitive == "init":
                if len(primitive_action_queue) == 0:
                    primitive_action_queue.extend(_get_init_action_sequence())
                    logger.info("[INIT] generated %d hardcoded actions", len(primitive_action_queue))
                action = primitive_action_queue.popleft()
                action = _process_action(action, args)
                obs, _, done, _ = env.step(action.tolist())
                t += 1
                subtask_tracker.update(obs, step_idx=t)
                if subtask_tracker.advanced:
                    primitive_action_queue.clear()
                    progress_check_cooldown = 3
                elif len(primitive_action_queue) == 0:
                    subtask_tracker.advance_to_next(step_idx=t)
                    progress_check_cooldown = 3
                if done:
                    success = True
                    break
                continue

            action = _process_action(_get_hold_action(), args)
            subtask_tracker.update(obs, step_idx=t)
            obs, _, done, _ = env.step(action.tolist())
            t += 1
            if done:
                success = True
                break
            continue

        if len(action_queue) == 0:
            infer_input = dict(element)
            infer_input["prompt"] = str(current_instruction)
            result = client.infer(infer_input)
            actions = result["actions"]

            progress_pred = _extract_progress(result, args)
            if progress_pred is not None:
                current_progress = progress_pred
                progress_predictions.append(progress_pred)
            elif args.subtask_switch_mode == "progress":
                raise RuntimeError(
                    f"Progress mode requires `{args.progress_key}` (or `progress_pred`) in server response."
                )

            if (
                args.subtask_switch_mode == "progress"
                and args.use_subtasks
                and progress_pred is not None
            ):
                idx = subtask_tracker.index
                primitive = None
                subtask = None
                if idx < len(subtask_tracker.plan):
                    subtask = subtask_tracker.plan[idx]
                    primitive = subtask.get("primitive") or subtask.get("rule")

                if primitive in {"init", "hold"}:
                    subtask_tracker.update(obs, step_idx=t)
                    if subtask_tracker.advanced:
                        action_queue.clear()
                        consecutive_above_threshold = 0
                        recent_progress_window.clear()
                    current_progress = None
                else:
                    if idx not in subtask_progress_history:
                        subtask_progress_history[idx] = []
                        subtask_max_progress[idx] = 0.0
                        subtask_start_progress[idx] = progress_pred
                    subtask_progress_history[idx].append(progress_pred)
                    subtask_max_progress[idx] = max(subtask_max_progress[idx], progress_pred)
                    recent_progress_window.append(progress_pred)

                    threshold = args.progress_thresholds_by_primitive.get(
                        primitive, args.progress_threshold
                    )
                    if progress_pred >= threshold:
                        consecutive_above_threshold += 1
                        if consecutive_above_threshold >= 2 and subtask_tracker.advance_to_next(step_idx=t):
                            action_queue.clear()
                            consecutive_above_threshold = 0
                            recent_progress_window.clear()
                    else:
                        consecutive_above_threshold = 0

                    if replan_cooldown > 0:
                        replan_cooldown -= 1
                        if replan_cooldown == 10:
                            current_replan_status = None

                    if progress_check_cooldown > 0:
                        progress_check_cooldown -= 1
                    elif args.progress_rollback_enabled:
                        history = subtask_progress_history.get(idx, [])
                        if history:
                            max_progress = subtask_max_progress.get(idx, 0.0)
                            start_progress = subtask_start_progress.get(idx, 0.0)
                            progress_drop = max_progress - progress_pred
                            progress_gain = progress_pred - start_progress
                            is_dropped = progress_drop > 0.03
                            is_no_growth = len(history) >= 10 and progress_gain < 0.03
                            has_problem = is_dropped or is_no_growth
                        else:
                            has_problem = False

                        if replan_cooldown == 0 and has_problem and idx > 0:
                            total_replan_count += 1
                            current_replan_status = "calling"
                            if args.use_llm_replan:
                                previous_subtask = subtask_tracker.get_previous_subtask()
                                current_subtask = subtask_tracker.get_current_subtask()
                                recovery_plan, ok = get_recovery_plan(
                                    replay_images=replay_images,
                                    previous_subtask=previous_subtask,
                                    current_subtask=current_subtask,
                                    task_description=task_description,
                                    available_objects=None,
                                    model=args.llm_replan_model,
                                    log_fn=logger.info,
                                )
                                if ok and recovery_plan:
                                    subtask_tracker.inject_recovery_plan(recovery_plan, step_idx=t)
                                    current_replan_status = "injected"
                                else:
                                    fallback = get_default_recovery_plan(current_subtask)
                                    subtask_tracker.inject_recovery_plan(fallback, step_idx=t)
                                    current_replan_status = "failed"
                                replan_cooldown = REPLAN_COOLDOWN_STEPS
                                action_queue.clear()
                                recent_progress_window.clear()
                                consecutive_above_threshold = 0
                                subtask_progress_history.pop(idx, None)
                                subtask_max_progress.pop(idx, None)
                                subtask_start_progress.pop(idx, None)
                            elif subtask_tracker.rollback_to_previous(step_idx=t):
                                action_queue.clear()
                                recent_progress_window.clear()
                                consecutive_above_threshold = 0
                                subtask_progress_history.pop(idx, None)
                                subtask_max_progress.pop(idx, None)
                                subtask_start_progress.pop(idx, None)

            action_queue.extend(actions[: args.num_open_loop_steps])

        action = _process_action(np.asarray(action_queue.popleft()), args)
        obs, _, done, _ = env.step(action.tolist())
        t += 1
        if done:
            success = True
            break

    return success, replay_images, replay_texts, subtask_tracker.status(), progress_predictions


def _load_initial_states(args: Args, task_suite, task_id: int):
    initial_states = task_suite.get_task_init_states(task_id)
    if args.initial_states_path == "DEFAULT":
        return initial_states, None
    with open(args.initial_states_path, "r", encoding="utf-8") as f:
        all_initial_states = json.load(f)
    return initial_states, all_initial_states


def _save_video(
    frames: List[np.ndarray],
    video_path: pathlib.Path,
) -> None:
    if not frames:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(video_path, [np.asarray(x) for x in frames], fps=10)


def eval_libero(args: Args) -> float:
    _validate_args(args)
    np.random.seed(args.seed)
    logger, log_path = _setup_logging(args)
    logger.info("Logging to %s", log_path)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    logger.info("Task suite: %s", args.task_suite_name)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    rollout_dir = pathlib.Path("rollouts") / args.rollout_dir_name
    rollout_dir.mkdir(parents=True, exist_ok=True)

    total_episodes, total_successes = 0, 0
    task_ids = list(range(task_suite.n_tasks))
    if args.task_suite_name == TaskSuite.LIBERO_10.value:
        task_ids = [8, 7, 4, 0, 1, 2, 3, 5, 6, 9]

    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        task_name = task.name
        initial_states, all_initial_states = _load_initial_states(args, task_suite, task_id)
        env, task_description = _get_libero_env(task, args.env_img_res, args.seed)

        task_episodes, task_successes = 0, 0
        subtask_total, subtask_completed, subtask_episodes, subtask_full_success = 0, 0, 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logger.info("\nTask: %s", task_description)
            if args.initial_states_path == "DEFAULT":
                initial_state = initial_states[episode_idx]
            else:
                task_key = task_description.replace(" ", "_")
                ep_key = f"demo_{episode_idx}"
                if not all_initial_states[task_key][ep_key]["success"]:
                    logger.info("Skipping task %d episode %d due to failed expert demo", task_id, episode_idx)
                    continue
                initial_state = np.array(all_initial_states[task_key][ep_key]["initial_state"])

            success, replay_images, _, subtask_status, _ = _run_episode(
                args=args,
                client=client,
                env=env,
                task_name=task_name,
                task_description=task_description,
                initial_state=initial_state,
                logger=logger,
            )

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            if args.save_video:
                suffix = "success" if success else "failure"
                seg = task_description.replace(" ", "_")
                _save_video(replay_images, rollout_dir / f"rollout_{total_episodes:05d}_{seg}_{suffix}.mp4")

            if subtask_status.total > 0:
                subtask_episodes += 1
                subtask_total += subtask_status.total
                subtask_completed += subtask_status.completed
                if subtask_status.all_done:
                    subtask_full_success += 1
                logger.info(
                    "Subtasks: %d/%d (index=%d)",
                    subtask_status.completed,
                    subtask_status.total,
                    subtask_status.index,
                )

            logger.info("Success: %s", success)
            logger.info("# episodes completed: %d", total_episodes)
            logger.info("# successes: %d (%.1f%%)", total_successes, 100.0 * total_successes / total_episodes)

        task_sr = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        total_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        logger.info("Current task success rate: %.4f", task_sr)
        logger.info("Current total success rate: %.4f", total_sr)
        if subtask_total > 0:
            logger.info("Subtask completion ratio: %.4f", float(subtask_completed) / float(subtask_total))
            logger.info("Subtask full success rate: %.4f", float(subtask_full_success) / float(subtask_episodes))

    final_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    logger.info("Final results:")
    logger.info("Total episodes: %d", total_episodes)
    logger.info("Total successes: %d", total_successes)
    logger.info("Overall success rate: %.4f (%.1f%%)", final_sr, final_sr * 100.0)
    return final_sr


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    eval_libero(tyro.cli(Args))

