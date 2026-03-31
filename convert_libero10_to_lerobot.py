"""
Convert LIBERO-10 RLDS dataset to LeRobot format.

Usage:
python convert_libero10_to_lerobot.py

This script will:
1. Load RLDS data from: /data/Embobrain/dataset/libero_10_rlds_split_padded
2. Convert to LeRobot format
3. Save to: modified_libero_lerobot_split_padded/libero_10_no_noops

To push to HuggingFace Hub after conversion:
python convert_libero10_to_lerobot.py --push_to_hub --repo_id YOUR_USERNAME/libero_10_no_noops
"""

import os
import shutil
import time
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro


def main(
    data_dir: str = "/data/Embobrain/dataset/libero_10_rlds_split_padded",
    output_dir: str = "modified_libero_lerobot_split_padded/libero_10_no_noops",
    repo_id: str = "zxlei/libero_10_no_noops",
    push_to_hub: bool = False,
):
    """
    Convert LIBERO-10 RLDS dataset to LeRobot format.
    
    Args:
        data_dir: Path to RLDS dataset directory
        output_dir: Output directory name (relative to HF_LEROBOT_HOME)
        repo_id: HuggingFace Hub repository ID (format: username/dataset_name)
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """
    
    print(f"[INFO] Input RLDS directory: {data_dir}")
    print(f"[INFO] Output directory: {HF_LEROBOT_HOME}/{output_dir}")
    print(f"[INFO] Repository ID: {repo_id}")
    
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / output_dir
    if output_path.exists():
        print(f"[WARNING] Output directory exists, removing: {output_path}")
        shutil.rmtree(output_path)

    # Create LeRobot dataset
    print("[INFO] Creating LeRobot dataset structure...")
    dataset = LeRobotDataset.create(
        repo_id=output_dir,  # This will be the local directory name
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
        image_writer_processes=80,  # 96核服务器：使用80个进程 = 83% CPU利用率
    )

    # Load RLDS dataset with optimizations
    print(f"[INFO] Loading RLDS dataset: libero_10_no_noops")
    try:
        raw_dataset = tfds.load(
            "libero_10_no_noops", 
            data_dir=data_dir, 
            split="train"
        )
        # Prefetch data to speed up iteration
        raw_dataset = raw_dataset.prefetch(buffer_size=50)
    except Exception as e:
        print(f"[ERROR] Failed to load RLDS dataset: {e}")
        print(f"[INFO] Make sure the dataset exists at: {data_dir}/libero_10_no_noops")
        return

    # Convert episodes
    print("[INFO] Converting episodes to LeRobot format...")
    print("[INFO] Using high parallelism: 4 threads × 80 processes = 320 concurrent tasks")
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
                    "task": step["language_instruction"].decode(),
                }
            )
            step_count += 1
        
        dataset.save_episode()
        episode_count += 1
        
        if episode_count % 5 == 0:
            elapsed = time.time() - start_time
            rate = episode_count / elapsed
            print(f"[INFO] Processed {episode_count} episodes, {step_count} steps | Rate: {rate:.2f} eps/s | Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"[SUCCESS] Conversion complete!")
    print(f"[INFO] Total episodes: {episode_count}")
    print(f"[INFO] Total steps: {step_count}")
    print(f"[INFO] Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"[INFO] Average rate: {episode_count/elapsed:.2f} episodes/s")
    print(f"[INFO] Dataset saved to: {output_path}")

    # Push to HuggingFace Hub
    if push_to_hub:
        print(f"[INFO] Pushing dataset to HuggingFace Hub: {repo_id}")
        print("[INFO] This may take a while...")
        
        try:
            dataset.push_to_hub(
                repo_id=repo_id,
                tags=["libero", "panda", "rlds", "manipulation"],
                private=False,
                push_videos=True,
                license="apache-2.0",
            )
            print(f"[SUCCESS] Dataset pushed to: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"[ERROR] Failed to push to hub: {e}")
            print("[INFO] Make sure you are logged in: huggingface-cli login")


if __name__ == "__main__":
    tyro.cli(main)
