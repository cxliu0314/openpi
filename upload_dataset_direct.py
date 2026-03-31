#!/usr/bin/env python3
"""
直接上传 LeRobot 数据集到 HuggingFace Hub
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo
import tyro


def main(
    dataset_path: str = "/home/batchcom/.cache/huggingface/lerobot/modified_libero_lerobot_split_padded/libero_10_no_noops",
    repo_id: str = "cxliu0314/libero_10_no_noops",
    private: bool = False,
):
    """
    直接上传数据集文件夹到 HuggingFace Hub
    
    Args:
        dataset_path: 本地数据集路径
        repo_id: HuggingFace Hub 仓库 ID (格式: username/dataset_name)
        private: 是否设为私有仓库
    """
    
    dataset_path = Path(dataset_path)
    
    print(f"[INFO] 数据集路径: {dataset_path}")
    print(f"[INFO] 目标仓库: {repo_id}")
    print(f"[INFO] 可见性: {'私有' if private else '公开'}")
    print("")
    
    # 检查数据集是否存在
    if not dataset_path.exists():
        print(f"[ERROR] 数据集不存在: {dataset_path}")
        return
    
    # 创建 API 客户端
    api = HfApi()
    
    # 创建仓库 (如果不存在)
    print("[INFO] 创建/检查仓库...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"[INFO] ✅ 仓库已就绪")
    except Exception as e:
        print(f"[ERROR] 创建仓库失败: {e}")
        return
    
    # 上传整个文件夹
    print(f"[INFO] 开始上传大型数据集 (~9.9GB)...")
    print(f"[INFO] 使用 upload_large_folder 方法...")
    print(f"[INFO] 这可能需要较长时间，请耐心等待...")
    print("")
    
    try:
        api.upload_large_folder(
            folder_path=str(dataset_path),
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("")
        print(f"[SUCCESS] ✅ 上传成功!")
        print(f"[INFO] 数据集链接: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"[ERROR] ❌ 上传失败: {e}")
        print("[INFO] 请检查:")
        print("  1. 网络连接 (代理是否正常)")
        print("  2. HuggingFace token 权限 (需要 write 权限)")
        print("  3. 仓库名称是否正确")


if __name__ == "__main__":
    tyro.cli(main)
