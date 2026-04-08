"""Convert LIBERO fewshot RLDS train/val splits to local LeRobot repos.

Example:
    uv run scripts/convert_fewshot_rlds_to_lerobot.py --data-kind original --shot 10 --overwrite
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import shutil
import time

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro


DATASET_NAME = "libero_10_no_noops"
DATA_KINDS = ("original", "split_padded")
SHOTS = (10, 20, 30, 50, 100)
SPLITS = ("train", "val")


def _as_tuple(value: str | Iterable[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _decode_task(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "decode"):
        return value.decode()
    return str(value)


def _create_dataset(output_repo_id: str, *, overwrite: bool, image_writer_processes: int) -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / output_repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
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
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=image_writer_processes,
    )


def _convert_split(
    *,
    input_dir: Path,
    output_repo_id: str,
    split: str,
    overwrite: bool,
    image_writer_processes: int,
) -> None:
    dataset_info = input_dir / DATASET_NAME / "1.0.0" / "dataset_info.json"
    if not dataset_info.exists():
        raise FileNotFoundError(f"Missing RLDS dataset_info.json: {dataset_info}")

    print(f"[INFO] Loading {DATASET_NAME}/{split} from {input_dir}")
    raw_dataset = tfds.load(DATASET_NAME, data_dir=str(input_dir), split=split)
    raw_dataset = raw_dataset.prefetch(buffer_size=50)

    print(f"[INFO] Writing LeRobot repo: {HF_LEROBOT_HOME / output_repo_id}")
    dataset = _create_dataset(output_repo_id, overwrite=overwrite, image_writer_processes=image_writer_processes)

    episode_count = 0
    step_count = 0
    start_time = time.time()
    for episode in raw_dataset:
        for step in episode["steps"].as_numpy_iterator():
            dataset.add_frame(
                {
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state": step["observation"]["state"],
                    "actions": step["action"],
                    "task": _decode_task(step["language_instruction"]),
                }
            )
            step_count += 1

        dataset.save_episode()
        episode_count += 1

    elapsed = time.time() - start_time
    print(
        f"[DONE] {output_repo_id}: {episode_count} episodes, {step_count} steps, "
        f"{elapsed:.1f}s elapsed"
    )


def main(
    rlds_root: str = "/data/Embobrain/dataset_revised/libero_fewshot_trainval_2x",
    output_prefix: str = "libero_fewshot_lerobot_trainval_2x",
    data_kind: str | None = None,
    shot: int | None = None,
    overwrite: bool = False,
    image_writer_processes: int = 32,
) -> None:
    data_kinds = _as_tuple(data_kind, DATA_KINDS)
    shots = (shot,) if shot is not None else SHOTS

    for current_kind in data_kinds:
        if current_kind not in DATA_KINDS:
            raise ValueError(f"Unknown data_kind={current_kind}. Expected one of {DATA_KINDS}.")
        for current_shot in shots:
            if current_shot not in SHOTS:
                raise ValueError(f"Unknown shot={current_shot}. Expected one of {SHOTS}.")
            input_dir = Path(rlds_root) / f"{current_kind}_n{current_shot}"
            for split in SPLITS:
                output_repo_id = f"{output_prefix}/{current_kind}_n{current_shot}_{split}/{DATASET_NAME}"
                _convert_split(
                    input_dir=input_dir,
                    output_repo_id=output_repo_id,
                    split=split,
                    overwrite=overwrite,
                    image_writer_processes=image_writer_processes,
                )


if __name__ == "__main__":
    tyro.cli(main)
