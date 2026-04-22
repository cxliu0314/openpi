"""Unified RLDS loader for pi0.5 LIBERO-style training."""

from collections.abc import Sequence
import dataclasses
from enum import Enum
import json
import logging
import numpy as np
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class RLDSDatasetSpec:
    name: str
    version: str
    weight: float = 1.0


class Pi05RLDSAdapterKind(str, Enum):
    LIBERO = "libero"
    DROID_OBS_EEF_DELTA = "droid_obs_eef_delta"


LIBERO_GRIPPER_QPOS_SCALE = 0.04


def _load_total_steps_from_stats(data_dir: str, dataset_name: str) -> int | None:
    candidates = [
        Path(data_dir) / "rlds_dataset_stats.json",
        Path(data_dir) / dataset_name / "rlds_dataset_stats.json",
    ]
    for stats_path in candidates:
        if not stats_path.exists():
            continue
        try:
            with stats_path.open("r") as f:
                stats = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        total_steps = stats.get("total_steps")
        if total_steps is not None:
            return int(total_steps)
    return None


class Pi05RldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDatasetSpec],
        *,
        adapter_kind: Pi05RLDSAdapterKind | str,
        shuffle: bool = True,
        action_chunk_size: int = 10,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
    ):
        import dlimp as dl
        from scipy.spatial.transform import Rotation
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        if not datasets:
            raise ValueError("At least one RLDS dataset must be provided.")

        adapter_kind = Pi05RLDSAdapterKind(adapter_kind)
        weight_sum = sum(dataset.weight for dataset in datasets)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Dataset weights must sum to 1.0, got {weight_sum}.")

        resolved_parallel_reads = tf.data.AUTOTUNE if num_parallel_reads == -1 else num_parallel_reads
        resolved_parallel_calls = tf.data.AUTOTUNE if num_parallel_calls == -1 else num_parallel_calls

        def _resolve_rlds_read_settings(builder, *, dataset_name: str) -> tuple[bool, int]:
            dataset_shuffle = shuffle
            dataset_parallel_reads = resolved_parallel_reads
            if getattr(builder.info, "disable_shuffling", False):
                if dataset_shuffle:
                    logging.warning(
                        "Dataset `%s` is ordered (`disable_shuffling=True`); forcing shuffle=False for TFDS reads.",
                        dataset_name,
                    )
                    dataset_shuffle = False
                dataset_parallel_reads = 1
            return dataset_shuffle, dataset_parallel_reads

        def _decode_image(image):
            if image.dtype == tf.string:
                image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
            else:
                image = tf.cast(image, tf.uint8)
            image.set_shape([None, None, None])
            return image

        def _numpy_euler_to_rotmat(euler):
            return Rotation.from_euler("xyz", euler).as_matrix().astype(np.float32)

        def _numpy_rotmat_to_axis_angle(rot_mat):
            return Rotation.from_matrix(rot_mat).as_rotvec().astype(np.float32)

        def _euler_to_rotation_matrix(euler):
            euler = tf.cast(euler, tf.float32)
            rot_mat = tf.numpy_function(_numpy_euler_to_rotmat, [euler], tf.float32)
            rot_mat.set_shape(euler.shape[:-1].concatenate([3, 3]))
            return rot_mat

        def _rotation_matrix_to_axis_angle(rot_mat):
            rot_mat = tf.cast(rot_mat, tf.float32)
            axis_angle = tf.numpy_function(_numpy_rotmat_to_axis_angle, [rot_mat], tf.float32)
            axis_angle.set_shape(rot_mat.shape[:-2].concatenate([3]))
            return axis_angle

        def _droid_cartesian_position_to_libero_eef_state(cartesian_position):
            cartesian_position = tf.cast(cartesian_position, tf.float32)
            xyz = cartesian_position[:, :3]
            rotvec = _rotation_matrix_to_axis_angle(_euler_to_rotation_matrix(cartesian_position[:, 3:6]))
            return tf.concat((xyz, rotvec), axis=-1)

        def _droid_cartesian_position_to_libero_delta_action(cartesian_position):
            cartesian_position = tf.cast(cartesian_position, tf.float32)
            current_cartesian_position = cartesian_position
            next_cartesian_position = tf.concat((cartesian_position[1:], cartesian_position[-1:]), axis=0)

            delta_pos = (next_cartesian_position[:, :3] - current_cartesian_position[:, :3]) / 0.05

            current_rot = _euler_to_rotation_matrix(current_cartesian_position[:, 3:6])
            next_rot = _euler_to_rotation_matrix(next_cartesian_position[:, 3:6])
            delta_rot = next_rot @ tf.linalg.matrix_transpose(current_rot)
            delta_rot = _rotation_matrix_to_axis_angle(delta_rot) / 0.5
            return tf.clip_by_value(tf.concat((delta_pos, delta_rot), axis=-1), -1.0, 1.0)

        def _droid_gripper_to_libero_gripper_qpos(gripper_position):
            openness = 1.0 - tf.cast(gripper_position, tf.float32)
            return tf.concat(
                (
                    LIBERO_GRIPPER_QPOS_SCALE * openness,
                    -LIBERO_GRIPPER_QPOS_SCALE * openness,
                ),
                axis=-1,
            )

        def _choose_first_nonempty_instruction(traj):
            instruction = _trajectory_scalar(traj["language_instruction"])
            for key in ("language_instruction_2", "language_instruction_3"):
                candidate = _trajectory_scalar(traj[key])
                instruction = tf.where(tf.strings.length(instruction) > 0, instruction, candidate)
            return instruction

        def _repeat_scalar(value, traj_len):
            return tf.repeat(tf.expand_dims(value, axis=0), traj_len, axis=0)

        def _trajectory_scalar(value):
            return tf.reshape(value, [-1])[0]

        def _restructure_episode(episode_index, traj):
            episode_index = tf.cast(episode_index, tf.int64)

            if adapter_kind == Pi05RLDSAdapterKind.LIBERO:
                actions = tf.cast(traj["action"], tf.float32)
                state = tf.cast(traj["observation"]["state"], tf.float32)
                image = traj["observation"]["image"]
                wrist_image = traj["observation"]["wrist_image"]
                prompt = _trajectory_scalar(traj["language_instruction"])
                step_id = None
            elif adapter_kind == Pi05RLDSAdapterKind.DROID_OBS_EEF_DELTA:
                cartesian_position = tf.cast(traj["observation"]["cartesian_position"], tf.float32)
                gripper_position = tf.cast(traj["observation"]["gripper_position"], tf.float32)
                state = tf.concat(
                    (
                        _droid_cartesian_position_to_libero_eef_state(cartesian_position),
                        _droid_gripper_to_libero_gripper_qpos(gripper_position),
                    ),
                    axis=-1,
                )
                actions = tf.concat(
                    (
                        _droid_cartesian_position_to_libero_delta_action(cartesian_position),
                        2.0 * tf.cast(traj["action_dict"]["gripper_position"], tf.float32) - 1.0,
                    ),
                    axis=-1,
                )
                image = tf.cond(
                    tf.random.uniform(shape=[]) > 0.5,
                    lambda: traj["observation"]["exterior_image_1_left"],
                    lambda: traj["observation"]["exterior_image_2_left"],
                )
                wrist_image = traj["observation"]["wrist_image_left"]
                prompt = _choose_first_nonempty_instruction(traj)
                metadata = traj["traj_metadata"]["episode_metadata"]
            else:
                raise ValueError(f"Unsupported adapter kind: {adapter_kind}")

            traj_len = tf.shape(actions)[0]
            frame_index = tf.range(traj_len, dtype=tf.int32)
            episode_len = tf.fill([traj_len], tf.cast(traj_len, tf.int32))
            episode_index = tf.fill([traj_len], episode_index)

            result = {
                "image": image,
                "wrist_image": wrist_image,
                "state": state,
                "actions": actions,
                "prompt": _repeat_scalar(prompt, traj_len),
                "frame_index": frame_index,
                "episode_index": episode_index,
                "episode_len": episode_len,
            }

            if adapter_kind == Pi05RLDSAdapterKind.DROID_OBS_EEF_DELTA:
                recording_folderpath = _trajectory_scalar(metadata["recording_folderpath"])
                file_path = _trajectory_scalar(metadata["file_path"])
                segment_index = tf.strings.as_string(_trajectory_scalar(metadata["segment_index"]))
                result["step_id"] = (
                    recording_folderpath
                    + "--"
                    + file_path
                    + "--"
                    + segment_index
                    + "--"
                    + tf.strings.as_string(frame_index)
                )

            return result

        def _chunk_actions(traj):
            traj_len = tf.shape(traj["actions"])[0]
            action_chunk_indices = (
                tf.range(action_chunk_size, dtype=tf.int32)[None, :]
                + tf.range(traj_len, dtype=tf.int32)[:, None]
            )
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        def _decode_frame(frame):
            frame["image"] = _decode_image(frame["image"])
            frame["wrist_image"] = _decode_image(frame["wrist_image"])
            frame["prompt"] = tf.reshape(frame["prompt"], [-1])[0]
            if "step_id" in frame:
                frame["step_id"] = tf.reshape(frame["step_id"], [-1])[0]
            return frame

        def _prepare_single_dataset(dataset_cfg: RLDSDatasetSpec):
            builder = tfds.builder(dataset_cfg.name, data_dir=data_dir, version=dataset_cfg.version)
            dataset_shuffle, dataset_parallel_reads = _resolve_rlds_read_settings(
                builder, dataset_name=f"{dataset_cfg.name}:{dataset_cfg.version}"
            )
            dataset = dl.DLataset.from_rlds(
                builder,
                split="train",
                shuffle=dataset_shuffle,
                num_parallel_reads=dataset_parallel_reads,
            )
            dataset = dataset.repeat()
            dataset = dataset.enumerate()
            dataset = dataset.map(_restructure_episode, num_parallel_calls=resolved_parallel_calls)
            dataset = dataset.map(_chunk_actions, num_parallel_calls=resolved_parallel_calls)
            dataset = dataset.unbatch()
            dataset = dataset.map(_decode_frame, num_parallel_calls=resolved_parallel_calls)

            total_steps = _load_total_steps_from_stats(data_dir, dataset_cfg.name)
            if total_steps is None:
                num_episodes = int(builder.info.splits["train"].num_examples)
                total_steps = max(num_episodes * 64, batch_size)
            return dataset, total_steps

        logging.info("Preparing %d unified RLDS dataset(s)...", len(datasets))
        for dataset in datasets:
            logging.info("    %s:%s weight=%.2f", dataset.name, dataset.version, dataset.weight)

        prepared = [_prepare_single_dataset(dataset_cfg) for dataset_cfg in datasets]
        all_datasets = [dataset for dataset, _ in prepared]
        total_steps = sum(total for _, total in prepared)

        final_dataset = tf.data.Dataset.sample_from_datasets(
            all_datasets,
            weights=[dataset.weight for dataset in datasets],
        )
        if shuffle:
            final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        final_dataset = final_dataset.batch(batch_size)
        final_dataset = final_dataset.prefetch(tf.data.AUTOTUNE)

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.adapter_kind = adapter_kind
        self._length = max(total_steps, batch_size)

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        return self._length
