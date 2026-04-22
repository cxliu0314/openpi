"""
将 RoboTwin Franka 单臂 RLDS（如 franka_move_can_pot）转为 LeRobot 格式。

仿照 ``convert_libero_data_to_lerobot.py``；RLDS 步内字段见
``dataset/.../franka_move_can_pot/1.3.0/features.json``（4 路相机、state/action 各 8 维）。

Usage:
  uv run examples/libero/convert_robotwin_data_to_lerobot.py

  # 自定义路径 / 只转 train
  uv run examples/libero/convert_robotwin_data_to_lerobot.py \\
    --rlds-dir /data/Embobrain/dataset/franka_robotwin_data0331_split/franka_move_can_pot \\
    --repo-id franka_move_can_pot \\
    --splits train

默认输出目录：``/data/Embobrain/RoboTwin/.cache/lerobot/<repo_id>/``（可用 ``--lerobot-cache`` 覆盖）。

依赖：``uv pip install tensorflow tensorflow_datasets``（与 Libero 转换脚本相同）。
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _decode_task(language_instruction: bytes | str | np.ndarray) -> str:
    if isinstance(language_instruction, bytes):
        return language_instruction.decode()
    if isinstance(language_instruction, str):
        return language_instruction
    arr = np.asarray(language_instruction)
    if arr.dtype == object:
        x = arr.flat[0]
        return x.decode() if isinstance(x, bytes) else str(x)
    return str(arr.item()) if arr.size else ""


def main(
    rlds_dir: str = "/data/Embobrain/dataset/franka_robotwin_data0331_split/franka_move_can_pot",
    repo_id: str = "franka_move_can_pot",
    lerobot_cache: str = "/data/Embobrain/RoboTwin/.cache/lerobot",
    splits: tuple[str, ...] = ("train", "val"),
    *,
    fps: int = 10,
    push_to_hub: bool = False,
) -> None:
    """将 ``<parent>/franka_move_can_pot`` 下的 RLDS 写入 LeRobot。

    ``rlds_dir`` 须为数据集根目录（其下含 ``1.x.x/`` 与 TFRecord），
    ``tfds.load`` 使用 ``data_dir=<parent>``、``name=<basename(rlds_dir)>``。
    """
    tf.config.set_visible_devices([], "GPU")

    rlds_path = pathlib.Path(rlds_dir).resolve()
    data_dir = str(rlds_path.parent)
    dataset_name = rlds_path.name

    cache_root = pathlib.Path(lerobot_cache).resolve()
    # LeRobot 的 ``root`` 是**单个数据集目录**（``.../lerobot/<repo_id>``），不是 ``lerobot`` 缓存根。
    output_path = cache_root / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        root=str(output_path),
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "left_wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "low_cam_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for split in splits:
        raw = tfds.load(dataset_name, data_dir=data_dir, split=split)
        for episode in raw:
            for step in episode["steps"].as_numpy_iterator():
                obs = step["observation"]
                dataset.add_frame(
                    {
                        "image": obs["image"],
                        "left_wrist_image": obs["left_wrist_image"],
                        "right_wrist_image": obs["right_wrist_image"],
                        "low_cam_image": obs["low_cam_image"],
                        "state": obs["state"],
                        "actions": step["action"],
                        "task": _decode_task(step["language_instruction"]),
                    }
                )
            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["robotwin", "franka", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
