#!/usr/bin/env python3
"""
LoRA-aware JAX -> PyTorch converter for OpenPI PI0/PI0.5 checkpoints.

This script intentionally does NOT modify the original converter. It adds a
pre-merge stage for LoRA weights stored in JAX checkpoints (lora_a/lora_b),
then reuses the original slicing logic to export a standard PyTorch checkpoint.
"""

import json
import os
import pathlib
import shutil
from typing import Literal

import numpy as np
import openpi.models.gemma
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.training.config as _config
import safetensors
import torch
import tyro

# Reuse core conversion utilities from the original script without editing it.
import convert_jax_model_to_pytorch as base_converter


def _get_lora_scaling(model_variant: str, group: str) -> float:
    """Return LoRA scaling for a given Gemma variant/group ('attn' or 'ffn')."""
    gemma_cfg = openpi.models.gemma.get_config(model_variant)
    lora_cfg = None if gemma_cfg.lora_configs is None else gemma_cfg.lora_configs.get(group)
    if lora_cfg is None:
        return 1.0
    return float(lora_cfg.scaling_value)


def _merge_lora_pair(
    flat_params: dict[str, np.ndarray],
    weight_key: str,
    lora_a_key: str,
    lora_b_key: str,
    scaling: float,
) -> bool:
    """Merge one LoRA pair into a base weight: W <- W + (A @ B) * scaling."""
    if weight_key not in flat_params and (lora_a_key in flat_params or lora_b_key in flat_params):
        raise KeyError(f"Found LoRA keys but missing base weight: {weight_key}")

    if lora_a_key not in flat_params and lora_b_key not in flat_params:
        return False
    if lora_a_key not in flat_params or lora_b_key not in flat_params:
        raise KeyError(f"Incomplete LoRA pair: ({lora_a_key}, {lora_b_key})")

    weight = np.asarray(flat_params[weight_key])
    lora_a = np.asarray(flat_params[lora_a_key])
    lora_b = np.asarray(flat_params[lora_b_key])

    # LoRA params in OpenPI follow [..., in_dim, rank] and [..., rank, out_dim].
    delta = np.einsum("...ir,...rj->...ij", lora_a, lora_b)
    if delta.shape != weight.shape:
        raise ValueError(
            f"LoRA merge shape mismatch for {weight_key}: delta={delta.shape}, weight={weight.shape}"
        )

    flat_params[weight_key] = weight + delta * scaling
    flat_params.pop(lora_a_key)
    flat_params.pop(lora_b_key)
    return True


def _merge_all_lora(flat_params: dict[str, np.ndarray], model_config) -> dict[str, object]:
    """Merge all known LoRA tensors in OpenPI Gemma blocks."""
    base_attn_scale = _get_lora_scaling(model_config.paligemma_variant, "attn")
    base_ffn_scale = _get_lora_scaling(model_config.paligemma_variant, "ffn")
    expert_attn_scale = _get_lora_scaling(model_config.action_expert_variant, "attn")
    expert_ffn_scale = _get_lora_scaling(model_config.action_expert_variant, "ffn")

    merge_specs = [
        # Base (PaliGemma language branch)
        ("llm/layers/attn/attn_vec_einsum/w", "llm/layers/attn/attn_vec_einsum/lora_a", "llm/layers/attn/attn_vec_einsum/lora_b", base_attn_scale),
        ("llm/layers/attn/kv_einsum/w", "llm/layers/attn/kv_einsum/lora_a", "llm/layers/attn/kv_einsum/lora_b", base_attn_scale),
        ("llm/layers/attn/q_einsum/w", "llm/layers/attn/q_einsum/lora_a", "llm/layers/attn/q_einsum/lora_b", base_attn_scale),
        ("llm/layers/mlp/gating_einsum", "llm/layers/mlp/gating_einsum_lora_a", "llm/layers/mlp/gating_einsum_lora_b", base_ffn_scale),
        ("llm/layers/mlp/linear", "llm/layers/mlp/linear_lora_a", "llm/layers/mlp/linear_lora_b", base_ffn_scale),
        # Expert branch (suffix _1)
        ("llm/layers/attn/attn_vec_einsum_1/w", "llm/layers/attn/attn_vec_einsum_1/lora_a", "llm/layers/attn/attn_vec_einsum_1/lora_b", expert_attn_scale),
        ("llm/layers/attn/kv_einsum_1/w", "llm/layers/attn/kv_einsum_1/lora_a", "llm/layers/attn/kv_einsum_1/lora_b", expert_attn_scale),
        ("llm/layers/attn/q_einsum_1/w", "llm/layers/attn/q_einsum_1/lora_a", "llm/layers/attn/q_einsum_1/lora_b", expert_attn_scale),
        ("llm/layers/mlp_1/gating_einsum", "llm/layers/mlp_1/gating_einsum_lora_a", "llm/layers/mlp_1/gating_einsum_lora_b", expert_ffn_scale),
        ("llm/layers/mlp_1/linear", "llm/layers/mlp_1/linear_lora_a", "llm/layers/mlp_1/linear_lora_b", expert_ffn_scale),
    ]

    merged_count = 0
    for w_key, a_key, b_key, scale in merge_specs:
        if _merge_lora_pair(flat_params, w_key, a_key, b_key, scale):
            merged_count += 1

    leftover_lora = sorted([k for k in flat_params if "lora" in k.lower()])
    return {
        "merged_pairs": merged_count,
        "leftover_lora_keys": leftover_lora,
        "base_attn_scale": base_attn_scale,
        "base_ffn_scale": base_ffn_scale,
        "expert_attn_scale": expert_attn_scale,
        "expert_ffn_scale": expert_ffn_scale,
    }


def _build_projection_params(initial_params: dict, pi05: bool) -> dict[str, torch.Tensor]:
    if pi05:
        proj_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    else:
        proj_keys = ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]

    projection_params = {}
    for key in proj_keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params

        projection_params[f"{key}.weight"] = torch.from_numpy(np.array(weight)).T
        projection_params[f"{key}.bias"] = torch.from_numpy(np.array(bias))
    return projection_params


def _copy_norm_assets_to_output_root(checkpoint_dir: str, output_path: str) -> None:
    """Copy normalization assets so output_path/<asset_id>/norm_stats.json exists."""
    src_candidates = [
        pathlib.Path(checkpoint_dir) / "assets",
        pathlib.Path(checkpoint_dir).parent / "assets",
    ]
    copied = False
    for src in src_candidates:
        if not src.exists() or not src.is_dir():
            continue
        children = list(src.iterdir())
        if not children:
            continue
        for child in children:
            dst = pathlib.Path(output_path) / child.name
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            if child.is_dir():
                shutil.copytree(child, dst)
            else:
                shutil.copy2(child, dst)
        print(f"[LoRA-aware] Copied norm assets from: {src}")
        copied = True
        break

    if not copied:
        print(
            "[LoRA-aware] Warning: no assets directory found under checkpoint dir; "
            "norm stats may be missing for downstream eval."
        )


def convert_pi0_checkpoint_lora_aware(
    checkpoint_dir: str,
    precision: str,
    output_path: str,
    model_config: openpi.models.pi0_config.Pi0Config,
) -> None:
    print(f"[LoRA-aware] Converting from: {checkpoint_dir}")
    print(f"[LoRA-aware] Output path: {output_path}")
    print(f"[LoRA-aware] Model config: {model_config}")

    initial_params = base_converter.slice_initial_orbax_checkpoint(
        checkpoint_dir=checkpoint_dir, restore_precision="float32"
    )

    paligemma_flat = dict(initial_params["paligemma_params"])
    merge_stats = _merge_all_lora(paligemma_flat, model_config)
    print(
        "[LoRA-aware] Merged LoRA pairs:",
        merge_stats["merged_pairs"],
    )
    if merge_stats["leftover_lora_keys"]:
        raise ValueError(
            "Unmerged LoRA keys remain after merge stage. "
            f"Examples: {merge_stats['leftover_lora_keys'][:10]}"
        )

    projection_params = _build_projection_params(initial_params, model_config.pi05)

    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    paligemma_config = PaliGemmaConfig()
    action_expert_config = openpi.models.gemma.get_config("gemma_300m")

    paligemma_params, expert_params = base_converter.slice_paligemma_state_dict(
        paligemma_flat, paligemma_config
    )
    gemma_params = base_converter.slice_gemma_state_dict(
        expert_params,
        action_expert_config,
        num_expert=1,
        checkpoint_dir=checkpoint_dir,
        pi05=model_config.pi05,
    )

    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    all_params = {**paligemma_params, **gemma_params, **projection_params}
    msg = pi0_model.load_state_dict(all_params, strict=False)

    unexpected_lora = [k for k in msg.unexpected_keys if "lora" in k.lower()]
    if unexpected_lora:
        raise ValueError(
            "LoRA keys are still unexpected after merge. "
            f"Examples: {unexpected_lora[:10]}"
        )

    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)
    elif precision == "float16":
        pi0_model = pi0_model.to(torch.float16)
    else:
        raise ValueError(f"Invalid precision: {precision}")

    os.makedirs(output_path, exist_ok=True)
    safetensors.torch.save_model(pi0_model, os.path.join(output_path, "model.safetensors"))
    _copy_norm_assets_to_output_root(checkpoint_dir, output_path)

    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": precision,
        "lora_merged_pairs": merge_stats["merged_pairs"],
        "lora_scales": {
            "base_attn": merge_stats["base_attn_scale"],
            "base_ffn": merge_stats["base_ffn_scale"],
            "expert_attn": merge_stats["expert_attn_scale"],
            "expert_ffn": merge_stats["expert_ffn_scale"],
        },
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("[LoRA-aware] Conversion completed.")
    print("[LoRA-aware] missing_keys:", len(msg.missing_keys), msg.missing_keys)
    print("[LoRA-aware] unexpected_keys:", len(msg.unexpected_keys), msg.unexpected_keys[:10])
    print("[LoRA-aware] Saved:", output_path)


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    inspect_only: bool = False,
):
    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")

    if inspect_only:
        base_converter.load_jax_model_and_print_keys(checkpoint_dir)
        return

    if not output_path:
        raise ValueError("--output_path is required unless --inspect_only is set.")

    convert_pi0_checkpoint_lora_aware(
        checkpoint_dir=checkpoint_dir,
        precision=precision,
        output_path=output_path,
        model_config=model_config,
    )


if __name__ == "__main__":
    tyro.cli(main)
