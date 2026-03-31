"""
Mappings from LIBERO task names to subtask plans and rule-based subtask tracking.

Isolated OpenPI port of the OpenVLA evaluation subtask engine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import robosuite.utils.transform_utils as T

from libero.libero.envs.predicates import eval_predicate_fn

# Libero-10 plans (same structure used by OpenVLA eval flow).
SUBTASK_PLANS: Dict[str, List[Dict[str, Any]]] = {
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": [
        {"primitive": "reach", "args": ["alphabet_soup_1"], "instruction": "reach('alphabet soup')"},
        {"primitive": "grasp", "args": ["alphabet_soup_1"], "instruction": "grasp('alphabet soup')"},
        {"primitive": "move", "args": ["basket_1"], "kwargs": {"pos_offset": [0, 0, [0, 0.2]]}, "instruction": "move('in', 'basket')"},
        {"primitive": "release", "args": ["alphabet_soup_1"], "instruction": "release('alphabet soup')"},
        {"primitive": "reach", "args": ["tomato_sauce_1"], "instruction": "reach('tomato sauce')"},
        {"primitive": "grasp", "args": ["tomato_sauce_1"], "instruction": "grasp('tomato sauce')"},
        {"primitive": "move", "args": ["basket_1"], "kwargs": {"pos_offset": [0, 0, [0, 0.2]]}, "instruction": "move('in', 'basket')"},
        {"primitive": "release", "args": ["tomato_sauce_1"], "instruction": "release('tomato sauce')"},
    ],
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": [
        {"primitive": "reach", "args": ["cream_cheese_1"], "instruction": "reach('cream cheese box')"},
        {"primitive": "grasp", "args": ["cream_cheese_1"], "instruction": "grasp('cream cheese box')"},
        {"primitive": "move", "args": ["basket_1"], "kwargs": {"pos_offset": [-0.05, 0, [0, 0.2]]}, "instruction": "move('in', 'basket')"},
        {"primitive": "release", "args": ["cream_cheese_1"], "instruction": "release('cream cheese box')"},
        {"primitive": "reach", "args": ["butter_1"], "instruction": "reach('butter')"},
        {"primitive": "grasp", "args": ["butter_1"], "instruction": "grasp('butter')"},
        {"primitive": "move", "args": ["basket_1"], "kwargs": {"pos_offset": [0.05, 0, [0, 0.2]]}, "instruction": "move('in', 'basket')"},
        {"primitive": "release", "args": ["butter_1"], "instruction": "release('butter')"},
    ],
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": [
        {"primitive": "reach", "args": ["flat_stove_1"], "instruction": "reach('knob')"},
        {"primitive": "grasp", "args": ["flat_stove_1"], "instruction": "grasp('knob')"},
        {"primitive": "rotate", "args": ["flat_stove_1"], "kwargs": {"angle": 0.5}, "instruction": "rotate('knob', 'clockwise')"},
        {"primitive": "release", "args": ["flat_stove_1"], "instruction": "release('knob')"},
        {"primitive": "reach", "args": ["moka_pot_1"], "instruction": "reach('moka pot')"},
        {"primitive": "grasp", "args": ["moka_pot_1"], "instruction": "grasp('moka pot')"},
        {"primitive": "move", "args": ["flat_stove_1"], "kwargs": {"pos_offset": [0, 0, [0, 0.1]]}, "instruction": "move('on', 'stove')"},
        {"primitive": "release", "args": ["moka_pot_1"], "instruction": "release('moka pot')"},
    ],
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": [
        {"primitive": "reach", "args": ["akita_black_bowl_1"], "instruction": "reach('black bowl')"},
        {"primitive": "grasp", "args": ["akita_black_bowl_1"], "instruction": "grasp('black bowl')"},
        {"primitive": "move", "args": ["white_cabinet_1"], "kwargs": {"pos_offset": [0, -0.15, 0]}, "instruction": "move('in', 'bottom drawer')"},
        {"primitive": "release", "args": ["akita_black_bowl_1"], "instruction": "release('black bowl')"},
        {"primitive": "reach", "args": ["white_cabinet_1"], "kwargs": {"pos_offset": [0, -0.3, 0]}, "instruction": "reach('bottom drawer')"},
        {"primitive": "push", "args": ["white_cabinet_1"], "kwargs": {"dist": 0.2}, "instruction": "push('bottom drawer', 'closed')"},
    ],
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": [
        {"primitive": "reach", "args": ["porcelain_mug_1"], "kwargs": {"threshold": 0.12}, "instruction": "reach('white mug')"},
        {"primitive": "grasp", "args": ["porcelain_mug_1"], "instruction": "grasp('white mug')"},
        {"primitive": "move", "args": ["plate_1"], "instruction": "move('on', 'left plate')"},
        {"primitive": "release", "args": ["porcelain_mug_1"], "instruction": "release('white mug')"},
        {"primitive": "reach", "args": ["white_yellow_mug_1"], "kwargs": {"threshold": 0.12}, "instruction": "reach('yellow and white mug')"},
        {"primitive": "grasp", "args": ["white_yellow_mug_1"], "instruction": "grasp('yellow and white mug')"},
        {"primitive": "move", "args": ["plate_2"], "instruction": "move('on', 'right plate')"},
        {"primitive": "release", "args": ["white_yellow_mug_1"], "instruction": "release('yellow and white mug')"},
    ],
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": [
        {"primitive": "reach", "args": ["black_book_1"], "kwargs": {"threshold": 0.12}, "instruction": "reach('the book')"},
        {"primitive": "grasp", "args": ["black_book_1"], "instruction": "grasp('the book')"},
        {"primitive": "move", "args": ["desk_caddy_1"], "kwargs": {"pos_offset": [0.02, 0, [0.05, 0.2]]}, "instruction": "move('in_the_back_compartment', 'the caddy')"},
        {"primitive": "release", "args": ["black_book_1"], "instruction": "release('the book')"},
    ],
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": [
        {"primitive": "reach", "args": ["porcelain_mug_1"], "kwargs": {"threshold": 0.10}, "instruction": "reach('white mug')"},
        {"primitive": "grasp", "args": ["porcelain_mug_1"], "instruction": "grasp('white mug')"},
        {"primitive": "move", "args": ["plate_1"], "instruction": "move('on', 'plate')"},
        {"primitive": "release", "args": ["porcelain_mug_1"], "instruction": "release('white mug')"},
        {"primitive": "reach", "args": ["chocolate_pudding_1"], "instruction": "reach('chocolate pudding')"},
        {"primitive": "grasp", "args": ["chocolate_pudding_1"], "instruction": "grasp('chocolate pudding')"},
        {"primitive": "move", "args": ["plate_1"], "kwargs": {"pos_offset": [0.15, 0, 0]}, "instruction": "move('to the right of', 'plate')"},
        {"primitive": "release", "args": ["chocolate_pudding_1"], "instruction": "release('chocolate pudding')"},
    ],
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": [
        {"primitive": "reach", "args": ["alphabet_soup_1"], "instruction": "reach('alphabet soup')"},
        {"primitive": "grasp", "args": ["alphabet_soup_1"], "instruction": "grasp('alphabet soup')"},
        {"primitive": "move", "args": ["basket_1"], "kwargs": {"pos_offset": [0, 0, [0, 0.2]]}, "instruction": "move('in', 'basket')"},
        {"primitive": "release", "args": ["alphabet_soup_1"], "instruction": "release('alphabet soup')"},
        {"primitive": "reach", "args": ["cream_cheese_1"], "instruction": "reach('cream cheese box')"},
        {"primitive": "grasp", "args": ["cream_cheese_1"], "instruction": "grasp('cream cheese box')"},
        {"primitive": "move", "args": ["basket_1"], "kwargs": {"pos_offset": [0, 0, [0, 0.2]]}, "instruction": "move('in', 'basket')"},
        {"primitive": "release", "args": ["cream_cheese_1"], "instruction": "release('cream cheese box')"},
    ],
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": [
        {"primitive": "reach", "args": ["moka_pot_2"], "kwargs": {"threshold": 0.11}, "instruction": "reach('left moka pot')"},
        {"primitive": "grasp", "args": ["moka_pot_2"], "instruction": "grasp('left moka pot')"},
        {"primitive": "move", "args": ["flat_stove_1"], "kwargs": {"pos_offset": [[0.1, 0.2], [-0.1, 0.2], 0.1]}, "instruction": "move('on', 'stove')"},
        {"primitive": "release", "args": ["moka_pot_2"], "instruction": "release('left moka pot')"},
        {"primitive": "reach", "args": ["moka_pot_1"], "kwargs": {"threshold": 0.10}, "instruction": "reach('right moka pot')"},
        {"primitive": "grasp", "args": ["moka_pot_1"], "instruction": "grasp('right moka pot')"},
        {"primitive": "move", "args": ["flat_stove_1"], "kwargs": {"pos_offset": [[0.1, 0.2], [-0.1, 0.2], 0.1]}, "instruction": "move('on', 'stove')"},
        {"primitive": "release", "args": ["moka_pot_1"], "instruction": "release('right moka pot')"},
    ],
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": [
        {"primitive": "reach", "args": ["white_yellow_mug_1"], "instruction": "reach('yellow and white mug')"},
        {"primitive": "grasp", "args": ["white_yellow_mug_1"], "instruction": "grasp('yellow and white mug')"},
        {"primitive": "move", "args": ["microwave_1"], "kwargs": {"pos_offset": [0, -0.05, 0]}, "instruction": "move('in', 'microwave')"},
        {"primitive": "release", "args": ["white_yellow_mug_1"], "instruction": "release('yellow and white mug')"},
        {"primitive": "reach", "args": ["microwave_1"], "instruction": "reach('microwave door')"},
        {"primitive": "push", "args": ["microwave_1"], "kwargs": {"dist": 0.3}, "instruction": "push('microwave door', 'closed')"},
    ],
}

TASK_PARAMS = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": {"knob_name": "flat_stove_1_button"},
}

DEFAULT_THRESHOLDS = {"reach": 0.08, "move": 0.05}
DEFAULT_GRIPPER_OPEN = 0.01
DEFAULT_GRIPPER_CLOSED = 0.04
DEFAULT_RELEASE_OPEN_DELTA = 0.001
DEFAULT_RELEASE_FULL_OPEN = 0.39
DEFAULT_AXIS_TOLERANCE = 0.04
DEFAULT_ROTATE_THRESHOLD = 0.2
DEFAULT_INSERT_THRESHOLD = 0.04
DEFAULT_PUSH_DISTANCE = 0.05


def _unwrap_env(env):
    return env.env if hasattr(env, "env") else env


def _get_object_body_id(env, object_name):
    if env is None or object_name is None:
        return None
    if hasattr(env, "obj_body_id") and object_name in env.obj_body_id:
        return env.obj_body_id[object_name]
    try:
        return env.sim.model.body_name2id(object_name)
    except Exception:  # noqa: BLE001
        return None


def _get_geom_ids_for_body_id(env, body_id):
    if env is None or body_id is None:
        return []
    sim = env.sim
    return [i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == body_id]


def _check_contact_ids(env, geom_ids1, geom_ids2):
    if env is None:
        return False
    sim = env.sim
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        if (contact.geom1 in geom_ids1 and contact.geom2 in geom_ids2) or (
            contact.geom1 in geom_ids2 and contact.geom2 in geom_ids1
        ):
            return True
    return False


def _contact_geoms_between_sets(env, geom_ids1, geom_ids2):
    if env is None:
        return set()
    sim = env.sim
    contacts = set()
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        if contact.geom1 in geom_ids1 and contact.geom2 in geom_ids2:
            contacts.add(contact.geom1)
        elif contact.geom2 in geom_ids1 and contact.geom1 in geom_ids2:
            contacts.add(contact.geom2)
    return contacts


def _get_object_state(env, obj_name):
    if env is None or obj_name is None:
        return None
    return getattr(env, "object_states_dict", {}).get(obj_name)


def _get_object_pos(env, obj_name):
    body_id = _get_object_body_id(env, obj_name)
    if body_id is None:
        return None
    return np.array(env.sim.data.body_xpos[body_id], dtype=float)


def _get_object_quat(env, obj_name):
    body_id = _get_object_body_id(env, obj_name)
    if body_id is None:
        return None
    return np.array(env.sim.data.body_xquat[body_id], dtype=float)


def _get_eef_pos(obs):
    if obs is None or "robot0_eef_pos" not in obs:
        return None
    return np.array(obs["robot0_eef_pos"], dtype=float)


def _get_gripper_qpos(obs):
    if obs is None or "robot0_gripper_qpos" not in obs:
        return None
    qpos = np.array(obs["robot0_gripper_qpos"], dtype=float)
    if qpos.size == 0:
        return None
    return float(np.mean(np.abs(qpos)))


def _get_gripper_qpos_from_env(env):
    if env is None or not hasattr(env, "robots"):
        return None
    robot = env.robots[0]
    gripper = getattr(robot, "gripper", None)
    if gripper is None or not hasattr(gripper, "get_joint_positions"):
        return None
    qpos = gripper.get_joint_positions()
    return float(np.mean(np.abs(qpos)))


def _get_current_gripper_qpos(env, obs):
    gripper_qpos = _get_gripper_qpos_from_env(env)
    return _get_gripper_qpos(obs) if gripper_qpos is None else gripper_qpos


def _get_gripper_geoms(env):
    if env is None or not hasattr(env, "robots"):
        return []
    robot = env.robots[0]
    sim = env.sim
    prefix = getattr(robot.robot_model, "naming_prefix", "")
    geom_ids = []
    for name in robot.gripper.contact_geoms:
        try:
            geom_ids.append(sim.model.geom_name2id(name))
            continue
        except Exception:  # noqa: BLE001
            pass
        if prefix:
            try:
                geom_ids.append(sim.model.geom_name2id(prefix + name))
            except Exception:  # noqa: BLE001
                pass
    return geom_ids


def _get_gripper_finger_geom_ids(env):
    if env is None or not hasattr(env, "robots"):
        return [], []
    robot = env.robots[0]
    sim = env.sim
    gripper = robot.gripper
    try:
        important = gripper.important_geoms
        left_names = important.get("left_fingerpad") or important.get("left_finger") or []
        right_names = important.get("right_fingerpad") or important.get("right_finger") or []
    except Exception:  # noqa: BLE001
        left_names, right_names = [], []
    left = []
    right = []
    for name in left_names:
        try:
            left.append(sim.model.geom_name2id(name))
        except Exception:  # noqa: BLE001
            pass
    for name in right_names:
        try:
            right.append(sim.model.geom_name2id(name))
        except Exception:  # noqa: BLE001
            pass
    return left, right


def _resolve_obj_name(tracker, obj_name, kwargs):
    if kwargs.get("obj_name"):
        return kwargs["obj_name"]
    if tracker is None or obj_name is None:
        return obj_name
    if obj_name == "flat_stove_1":
        knob_name = tracker.task_params.get("knob_name")
        if knob_name:
            return knob_name
    return obj_name


def _get_subtask_kwargs(subtask):
    return subtask.get("kwargs") or {}


def _evaluate_predicates(env, predicates):
    if not predicates:
        return False
    predicate_list = predicates if isinstance(predicates[0], (list, tuple)) else [predicates]
    for predicate in predicate_list:
        if not predicate:
            return False
        fn_name = predicate[0]
        args = []
        for obj_name in predicate[1:]:
            obj_state = _get_object_state(env, obj_name)
            if obj_state is None:
                return False
            args.append(obj_state)
        if not eval_predicate_fn(fn_name, *args):
            return False
    return True


def _rule_reach(env, obs, subtask, tracker):
    kwargs = _get_subtask_kwargs(subtask)
    obj_name = _resolve_obj_name(tracker, subtask.get("args", [None])[0], kwargs)
    obj_pos = _get_object_pos(env, obj_name)
    eef_pos = _get_eef_pos(obs)
    if obj_pos is None or eef_pos is None:
        return False
    target = obj_pos + np.array(kwargs.get("pos_offset", [0.0, 0.0, 0.0]), dtype=float)
    threshold = float(kwargs.get("threshold", DEFAULT_THRESHOLDS["reach"]))
    return float(np.linalg.norm(eef_pos - target)) < threshold


def _rule_grasp(env, obs, subtask, tracker):
    kwargs = _get_subtask_kwargs(subtask)
    obj_name = _resolve_obj_name(tracker, subtask.get("args", [None])[0], kwargs)
    if obj_name is None:
        return False
    close_thresh = float(kwargs.get("gripper_close_threshold", tracker.gripper_closed_thresh))
    avg_qpos = _get_current_gripper_qpos(env, obs)
    gripper_ok = avg_qpos is None or avg_qpos < close_thresh
    body_id = _get_object_body_id(env, obj_name)
    if body_id is None:
        return False
    obj_geoms = _get_geom_ids_for_body_id(env, body_id)
    if not obj_geoms:
        return False
    left, right = _get_gripper_finger_geom_ids(env)
    if left and right:
        return gripper_ok and _check_contact_ids(env, left, obj_geoms) and _check_contact_ids(env, right, obj_geoms)
    gripper_geoms = _get_gripper_geoms(env)
    if not gripper_geoms:
        return False
    return gripper_ok and len(_contact_geoms_between_sets(env, gripper_geoms, obj_geoms)) >= 2


def _rule_release(env, obs, subtask, tracker):
    kwargs = _get_subtask_kwargs(subtask)
    gripper_qpos = _get_current_gripper_qpos(env, obs)
    if gripper_qpos is None:
        return False
    start_qpos = tracker.release_start_qpos.get(tracker.index)
    if start_qpos is None:
        tracker.release_start_qpos[tracker.index] = gripper_qpos
        start_qpos = gripper_qpos
    open_delta = float(kwargs.get("release_open_delta", DEFAULT_RELEASE_OPEN_DELTA))
    full_open = float(kwargs.get("release_full_open_threshold", DEFAULT_RELEASE_FULL_OPEN))
    return (gripper_qpos - start_qpos) >= open_delta or gripper_qpos >= full_open


def _rule_move(env, obs, subtask, tracker):
    kwargs = _get_subtask_kwargs(subtask)
    obj_name = subtask.get("object") or tracker.active_object
    target_name = subtask.get("args", [None])[0]
    if obj_name is None or target_name is None:
        return False
    obj_pos = _get_object_pos(env, obj_name)
    target_base = _get_object_pos(env, target_name)
    if obj_pos is None or target_base is None:
        return False
    pos_offset = list(kwargs.get("pos_offset", [0.0, 0.0, 0.0]))
    while len(pos_offset) < 3:
        pos_offset.append(0.0)
    axis_tol = float(kwargs.get("axis_tolerance", DEFAULT_AXIS_TOLERANCE))
    for i in range(3):
        off = pos_offset[i]
        if isinstance(off, (list, tuple)) and len(off) == 2:
            low = target_base[i] + min(off)
            high = target_base[i] + max(off)
            if not (low <= float(obj_pos[i]) <= high):
                return False
        else:
            tgt = target_base[i] + float(off)
            if abs(float(obj_pos[i]) - float(tgt)) >= axis_tol:
                return False
    return True


def _rule_rotate(env, obs, subtask, tracker):
    _ = obs
    if _evaluate_predicates(env, subtask.get("predicate") or subtask.get("predicates") or []):
        return True
    kwargs = _get_subtask_kwargs(subtask)
    obj_name = subtask.get("args", [None])[0]
    obj_state = _get_object_state(env, obj_name)
    if obj_state is not None:
        state = kwargs.get("state")
        if state == "open" and hasattr(obj_state, "is_open"):
            try:
                return obj_state.is_open()
            except Exception:  # noqa: BLE001
                pass
        if state == "close" and hasattr(obj_state, "is_close"):
            try:
                return obj_state.is_close()
            except Exception:  # noqa: BLE001
                pass
        try:
            if hasattr(obj_state, "turn_on") and obj_state.turn_on():
                return True
        except Exception:  # noqa: BLE001
            pass
    init_quat = tracker.initial_quats.get(tracker.index)
    curr_quat = _get_object_quat(env, obj_name)
    if init_quat is None or curr_quat is None:
        return False
    q_diff = T.quat_multiply(curr_quat, T.quat_inverse(init_quat))
    angle = 2 * np.arccos(float(np.clip(q_diff[0], -1.0, 1.0)))
    target = float(kwargs.get("angle", 0.0))
    threshold = float(kwargs.get("threshold", DEFAULT_ROTATE_THRESHOLD))
    return abs(angle - target) < threshold


def _rule_push(env, obs, subtask, tracker):
    _ = obs
    kwargs = _get_subtask_kwargs(subtask)
    obj_name = subtask.get("args", [None])[0]
    if obj_name is None:
        return False
    obj_body_id = _get_object_body_id(env, obj_name)
    if obj_body_id is None:
        return False
    obj_geoms = _get_geom_ids_for_body_id(env, obj_body_id)
    grip_geoms = _get_gripper_geoms(env)
    if not obj_geoms or not grip_geoms:
        return False
    sim = env.sim
    contact_positions = []
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        if (contact.geom1 in grip_geoms and contact.geom2 in obj_geoms) or (
            contact.geom1 in obj_geoms and contact.geom2 in grip_geoms
        ):
            contact_positions.append(contact.pos.copy())
    if not contact_positions:
        return False
    current_pos = np.mean(np.asarray(contact_positions), axis=0)
    if tracker.push_start_pos.get(tracker.index) is None:
        tracker.push_start_pos[tracker.index] = current_pos
        return False
    dist = float(np.linalg.norm(current_pos - tracker.push_start_pos[tracker.index]))
    threshold = float(kwargs.get("dist", kwargs.get("distance", kwargs.get("threshold", DEFAULT_PUSH_DISTANCE))))
    return dist > threshold


def _rule_flip(env, obs, subtask, tracker):
    _ = obs
    kwargs = _get_subtask_kwargs(subtask)
    obj_name = subtask.get("args", [None])[0]
    init_quat = tracker.initial_quats.get(tracker.index)
    curr_quat = _get_object_quat(env, obj_name)
    if init_quat is None or curr_quat is None:
        return False
    q_diff = T.quat_multiply(curr_quat, T.quat_inverse(init_quat))
    rot_vec = T.quat2axisangle(q_diff)
    measured_angle = float(np.linalg.norm(rot_vec))
    if measured_angle < 1e-3:
        return False
    measured_axis = rot_vec / measured_angle
    axis = np.array(kwargs.get("axis", [0.0, 0.0, 1.0]), dtype=float)
    axis = axis / np.linalg.norm(axis)
    axis_match = abs(float(np.dot(measured_axis, axis))) > 0.9
    target = float(kwargs.get("angle", 0.0))
    threshold = float(kwargs.get("threshold", DEFAULT_ROTATE_THRESHOLD))
    return axis_match and abs(measured_angle - target) < threshold


def _rule_insert(env, obs, subtask, tracker):
    _ = obs, tracker
    args = subtask.get("args", [])
    if len(args) < 2:
        return False
    pos1 = _get_object_pos(env, args[0])
    pos2 = _get_object_pos(env, args[1])
    if pos1 is None or pos2 is None:
        return False
    threshold = float(_get_subtask_kwargs(subtask).get("threshold", DEFAULT_INSERT_THRESHOLD))
    return float(np.linalg.norm(pos1[:2] - pos2[:2])) < threshold


def _rule_press(env, obs, subtask, tracker):
    _ = obs, tracker
    obj_name = subtask.get("args", [None])[0]
    body_id = _get_object_body_id(env, obj_name)
    if body_id is None:
        return False
    sim = env.sim
    for jid in [j for j in range(sim.model.njnt) if sim.model.jnt_bodyid[j] == body_id]:
        qpos = sim.data.qpos[sim.model.jnt_qposadr[jid]]
        if abs(float(qpos)) > 0.01:
            return True
    return False


def _rule_contact_between(env, obs, subtask, tracker):
    _ = obs, tracker
    args = subtask.get("args", [])
    if len(args) < 2:
        return False
    body1 = _get_object_body_id(env, args[0])
    body2 = _get_object_body_id(env, args[1])
    if body1 is None or body2 is None:
        return False
    return _check_contact_ids(env, _get_geom_ids_for_body_id(env, body1), _get_geom_ids_for_body_id(env, body2))


def _rule_turn_on(env, obs, subtask, tracker):
    _ = obs, tracker
    obj_state = _get_object_state(env, subtask.get("args", [None])[0])
    if obj_state is None:
        return False
    try:
        return bool(obj_state.turn_on())
    except Exception:  # noqa: BLE001
        return False


def _rule_open(env, obs, subtask, tracker):
    _ = obs, tracker
    obj_name = subtask.get("args", [None])[0]
    try:
        if hasattr(env, "check_predicate") and env.check_predicate("Open", [obj_name]):
            return True
    except Exception:  # noqa: BLE001
        pass
    obj_state = _get_object_state(env, obj_name)
    if obj_state is None:
        return True
    if hasattr(obj_state, "is_open"):
        try:
            return bool(obj_state.is_open())
        except Exception:  # noqa: BLE001
            return True
    return True


def _rule_predicate(env, obs, subtask, tracker):
    _ = obs, tracker
    predicates = subtask.get("predicate") or subtask.get("predicates") or []
    return _evaluate_predicates(env, predicates)


def _rule_init(env, obs, subtask, tracker):
    eef_pos = _get_eef_pos(obs)
    if eef_pos is None:
        return False
    gripper_qpos = _get_current_gripper_qpos(env, obs)
    gripper_ok = gripper_qpos is None or gripper_qpos >= tracker.gripper_open_thresh
    target_gain = float(_get_subtask_kwargs(subtask).get("target_height_gain", 0.05))
    if tracker.index not in tracker._init_start_height:
        tracker._init_start_height[tracker.index] = float(eef_pos[2])
    gained = float(eef_pos[2] - tracker._init_start_height[tracker.index])
    return gained >= target_gain and gripper_ok


def _rule_hold(env, obs, subtask, tracker):
    _ = env, obs
    duration = int(_get_subtask_kwargs(subtask).get("duration", 5))
    tracker.hold_count += 1
    return tracker.hold_count >= duration


SUBTASK_RULES: Dict[str, Callable[[Any, Any, Dict[str, Any], "SubtaskTracker"], bool]] = {
    "reach": _rule_reach,
    "grasp": _rule_grasp,
    "release": _rule_release,
    "move": _rule_move,
    "rotate": _rule_rotate,
    "push": _rule_push,
    "flip": _rule_flip,
    "insert": _rule_insert,
    "press": _rule_press,
    "contact": _rule_contact_between,
    "turn_on": _rule_turn_on,
    "open": _rule_open,
    "hold": _rule_hold,
    "init": _rule_init,
    "predicate": _rule_predicate,
}


def evaluate_subtask(env, obs, subtask, tracker):
    rule_name = subtask.get("rule") or subtask.get("primitive")
    rule_fn = SUBTASK_RULES.get(rule_name)
    return False if rule_fn is None else bool(rule_fn(env, obs, subtask, tracker))


@dataclass
class SubtaskStatus:
    total: int
    completed: int
    index: int
    all_done: bool


class SubtaskTracker:
    def __init__(
        self,
        env,
        task_name,
        fallback_instruction,
        enabled=True,
        hold_steps=1,
        thresholds=None,
        gripper_open_thresh=DEFAULT_GRIPPER_OPEN,
        gripper_closed_thresh=DEFAULT_GRIPPER_CLOSED,
        debug=False,
        debug_log=None,
    ):
        self.env = _unwrap_env(env)
        self.task_name = task_name
        self.plan = SUBTASK_PLANS.get(task_name, [])
        self.enabled = bool(enabled and self.plan)
        self.fallback_instruction = fallback_instruction
        self.task_params = TASK_PARAMS.get(task_name, {})
        self.index = 0
        self.completed = [False for _ in self.plan]
        self.active_object = None
        self.advanced = False
        self.last_completed = None
        self.hold_steps = max(1, int(hold_steps))
        self.hold_count = 0
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.gripper_open_thresh = float(gripper_open_thresh)
        self.gripper_closed_thresh = float(gripper_closed_thresh)
        self.history: List[Dict[str, Any]] = []
        self.initial_quats = {}
        self.push_start_pos = {}
        self.release_start_qpos = {}
        self.grasp_qpos_hist = {}
        self._init_start_height = {}
        self._prev_index = None
        self.debug = bool(debug)
        self.debug_log = debug_log
        self._debug_state = {}

        # Replanning state.
        self._in_recovery_mode = False
        self._recovery_plan: List[Dict[str, Any]] = []
        self._recovery_start_index = 0
        self._waiting_for_replan = False

    def current_instruction(self):
        if not self.enabled or self.index >= len(self.plan):
            return self.fallback_instruction
        return self.plan[self.index].get("instruction", self.fallback_instruction)

    def all_done(self):
        return self.enabled and self.index >= len(self.plan)

    def _maybe_set_initial_quat(self, subtask):
        primitive = subtask.get("primitive") or subtask.get("rule")
        if primitive not in ("rotate", "flip"):
            return
        if self.index in self.initial_quats:
            return
        obj_name = subtask.get("args", [None])[0]
        initial_quat = _get_object_quat(self.env, obj_name)
        if initial_quat is not None:
            self.initial_quats[self.index] = initial_quat

    def debug_kv(self, rule_name, **kvs):
        if not self.debug:
            return
        if self._debug_state.get(rule_name) == kvs:
            return
        self._debug_state[rule_name] = kvs
        parts = []
        for key, value in kvs.items():
            parts.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
        message = f"[subtask_debug] {rule_name}: " + ", ".join(parts)
        if self.debug_log is not None:
            self.debug_log(message)
        else:
            print(message)

    def update(self, obs=None, step_idx=None):
        self.advanced = False
        self.last_completed = None
        if not self.enabled or self.index >= len(self.plan):
            return
        if self._prev_index != self.index:
            self._prev_index = self.index
            self.push_start_pos.pop(self.index, None)

        subtask = self.plan[self.index]
        self._maybe_set_initial_quat(subtask)
        if evaluate_subtask(self.env, obs, subtask, self):
            self.hold_count += 1
            if self.hold_count >= self.hold_steps:
                self._complete_subtask(subtask, step_idx=step_idx)
        else:
            self.hold_count = 0

    def _complete_subtask(self, subtask, step_idx=None):
        self.completed[self.index] = True
        self.advanced = True
        self.last_completed = subtask
        primitive = subtask.get("primitive")
        args = subtask.get("args") or []
        if primitive == "grasp" and args:
            self.active_object = args[0]
        if primitive == "release":
            self.active_object = None
        self.history.append({"index": self.index, "primitive": primitive, "args": args, "step": step_idx})
        self.index += 1
        self.hold_count = 0

    def advance_to_next(self, step_idx=None):
        if not self.enabled or self.index >= len(self.plan):
            return False
        self._complete_subtask(self.plan[self.index], step_idx=step_idx)
        return True

    def rollback_to_previous(self, step_idx=None):
        if not self.enabled or self.index <= 0:
            return False
        self.index -= 1
        self.advanced = True
        if self.index < len(self.completed):
            self.completed[self.index] = False
        if self.index < len(self.plan):
            self.last_completed = {"instruction": self.plan[self.index].get("instruction", ""), "rollback": True}
        if self.index > 0 and (self.index - 1) < len(self.plan):
            prev = self.plan[self.index - 1]
            if prev.get("primitive") == "grasp":
                args = prev.get("args") or []
                if args:
                    self.active_object = args[0]
        self.history.append({"index": self.index, "event": "rollback", "step": step_idx})
        self.hold_count = 0
        return True

    def status(self):
        return SubtaskStatus(
            total=len(self.plan),
            completed=sum(1 for done in self.completed if done),
            index=self.index,
            all_done=self.all_done(),
        )

    def get_previous_subtask(self) -> Optional[Dict]:
        if self.index <= 0:
            return None
        return self.plan[self.index - 1]

    def get_current_subtask(self) -> Optional[Dict]:
        if self.index >= len(self.plan):
            return None
        return self.plan[self.index]

    def enter_waiting_for_replan(self):
        self._waiting_for_replan = True

    def is_waiting_for_replan(self) -> bool:
        return self._waiting_for_replan

    def inject_recovery_plan(self, recovery_subtasks: List[Dict], step_idx=None):
        if not recovery_subtasks:
            return False
        self._waiting_for_replan = False
        self._in_recovery_mode = True
        self._recovery_start_index = self.index
        self._recovery_plan = list(recovery_subtasks)

        remaining_plan = self.plan[self.index :]
        self.plan = self.plan[: self.index] + recovery_subtasks + remaining_plan
        self.completed = self.completed[: self.index] + [False] * len(recovery_subtasks) + [False] * len(remaining_plan)

        self.history.append(
            {
                "index": self.index,
                "event": "recovery_injected",
                "recovery_count": len(recovery_subtasks),
                "step": step_idx,
            }
        )
        self.hold_count = 0
        self.advanced = True
        return True

