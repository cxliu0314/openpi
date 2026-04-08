import argparse
import dataclasses
import pathlib
import sys

import flax.nnx as nnx

from openpi.models import pi0 as pi0_model
from openpi.models import pi0_config
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as train_config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train only chunk progress heads on top of an existing PI0.5 checkpoint.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--init-ckpt", required=True, help="Path to checkpoint params dir, e.g. .../29999/params")
    parser.add_argument("--num-train-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fsdp-devices", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--val-num-batches", type=int, default=10)
    parser.add_argument("--val-split-ratio", type=float, default=0.1)
    parser.add_argument("--progress-loss-weight", type=float, default=0.1)
    parser.add_argument("--progress-readout-mode", choices=("chunk_prefix",), default="chunk_prefix")
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--output-config-name")
    parser.add_argument("--peak-lr", type=float, default=2e-4)
    parser.add_argument("--decay-lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--decay-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")
    return parser.parse_args()


def _progress_head_desc(prefix_dim: int) -> str:
    return (
        f"pooled_prefix_dim={prefix_dim}, adapter={prefix_dim}->{pi0_model.CHUNK_PROGRESS_ADAPTER_DIM}, "
        f"shared_mlp={pi0_model.CHUNK_PROGRESS_ADAPTER_DIM}->{pi0_model.CHUNK_PROGRESS_HIDDEN_DIM}"
        f"->{pi0_model.CHUNK_PROGRESS_MID_DIM}->1"
    )


def main() -> None:
    args = parse_args()

    base_cfg = train_config.get_config(args.config_name)
    if not isinstance(base_cfg.model, pi0_config.Pi0Config):
        raise ValueError(f"Config {args.config_name} is not Pi0Config.")

    model_cfg = dataclasses.replace(base_cfg.model, enable_progress_head=True)

    progress_head_regex = nnx_utils.PathRegex(".*progress_chunk.*")
    freeze_all_except_progress_chunk = nnx.Not(progress_head_regex)

    cfg = dataclasses.replace(
        base_cfg,
        name=args.output_config_name or base_cfg.name,
        exp_name=args.exp_name,
        model=model_cfg,
        weight_loader=weight_loaders.CheckpointWeightLoader(args.init_ckpt),
        freeze_filter=freeze_all_except_progress_chunk,
        checkpoint_base_dir=args.checkpoint_base_dir,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        val_num_batches=args.val_num_batches,
        val_split_ratio=args.val_split_ratio,
        enable_progress_loss=True,
        progress_loss_weight=args.progress_loss_weight,
        progress_target_mode="chunk",
        progress_readout_mode=args.progress_readout_mode,
        use_val_set=True,
        wandb_enabled=args.wandb_enabled,
        resume=args.resume,
        overwrite=args.overwrite,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=args.decay_steps or args.num_train_steps,
            decay_lr=args.decay_lr,
        ),
    )

    print("[ChunkProgressHeadOnly] Starting with config:")
    import openpi

    print(f"  openpi_module={openpi.__file__}")
    print(f"  config_name={args.config_name}")
    print(f"  exp_name={args.exp_name}")
    print(f"  init_ckpt={args.init_ckpt}")
    print(f"  checkpoint_dir={cfg.checkpoint_dir}")
    print(f"  trainable_filter=Param AND {progress_head_regex}")
    print(f"  progress_target_mode={cfg.progress_target_mode}")
    print(f"  progress_readout_mode={cfg.progress_readout_mode}")
    print("  progress_input=pooled_prefix_hidden(image+prompt)+chunk_step_embedding")
    prefix_dim = pi0_model._gemma.get_config(model_cfg.paligemma_variant).width
    print(f"  progress_head_arch={_progress_head_desc(prefix_dim)}")
    print("  progress_target_rule=(frame_index + offset) / max(episode_len - 1, 1), clipped to [0, 1]")
    print(f"  progress_loss_weight={cfg.progress_loss_weight}")
    print(
        f"  lr_schedule=warmup_cosine(warmup_steps={args.warmup_steps}, peak_lr={args.peak_lr}, "
        f"decay_steps={args.decay_steps or args.num_train_steps}, decay_lr={args.decay_lr})"
    )

    train.main(cfg)


if __name__ == "__main__":
    main()
