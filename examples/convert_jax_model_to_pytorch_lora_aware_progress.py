#!/usr/bin/env python3
"""
LoRA-aware + progress-head-aware JAX -> PyTorch converter for OpenPI PI0/PI0.5.

This script intentionally does NOT modify the original converter. It adds:
1. LoRA merge (lora_a/lora_b -> base weights)
2. Optional progress-head extraction and standalone export (`progress_head.pt`)
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


def _extract_linear_params(module_params: dict, prefix: str) -> dict[str, torch.Tensor]:
    kernel_params = module_params["kernel"]
    bias_params = module_params["bias"]

    if isinstance(kernel_params, dict):
        weight = kernel_params["value"]
        bias = bias_params["value"]
    else:
        weight = kernel_params
        bias = bias_params

    return {
        f"{prefix}.weight": torch.from_numpy(np.array(weight)).T,
        f"{prefix}.bias": torch.from_numpy(np.array(bias)),
    }


def _extract_chunk_progress_module_state(
    projection_params: dict,
    module_key: str,
) -> tuple[str, dict[str, torch.Tensor]] | None:
    """
    Extract ChunkProgressHead-like params:
      input_adapter.{weight,bias}
      shared_head.fc{1,2,3}.{weight,bias}
    """
    if module_key not in projection_params:
        return None
    module_params = projection_params[module_key]
    if "input_adapter" not in module_params or "shared_head" not in module_params:
        raise ValueError(
            f"Unsupported chunk progress module `{module_key}`. "
            f"Expected keys input_adapter/shared_head, got: {sorted(module_params.keys())}"
        )

    state_dict = {}
    state_dict.update(_extract_linear_params(module_params["input_adapter"], "input_adapter"))
    shared = module_params["shared_head"]
    for layer_name in ("fc1", "fc2", "fc3"):
        if layer_name not in shared:
            raise ValueError(
                f"Unsupported shared_head in `{module_key}`. Missing `{layer_name}`. "
                f"Got keys: {sorted(shared.keys())}"
            )
        state_dict.update(_extract_linear_params(shared[layer_name], f"shared_head.{layer_name}"))
    return "chunk_mlp", state_dict


def _infer_progress_variant(checkpoint_dir: str, projection_params: dict, progress_variant: str) -> str:
    if progress_variant != "auto":
        return progress_variant

    exp_name = os.path.basename(os.path.dirname(checkpoint_dir.rstrip("/"))).lower()
    if "chunk_progress_prefix" in exp_name:
        return "chunk_prefix"

    if "progress_chunk_prefix_out_proj" in projection_params:
        return "chunk_prefix"
    return "chunk_prefix"


def _extract_progress_payload(
    initial_params: dict,
    checkpoint_dir: str,
    progress_variant: str,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, object] | None]:
    """
    Returns:
      base_progress_params: no longer used; kept as None for converter compatibility
      progress_payload: standalone progress_head.pt payload with metadata + state_dict
    """
    projection_params = initial_params["projection_params"]
    base_progress_params = None

    selected_variant = _infer_progress_variant(checkpoint_dir, projection_params, progress_variant)
    variant_map = {
        "chunk_prefix": ("progress_chunk_prefix_out_proj", "chunk_prefix", "chunk"),
    }
    module_key, readout_mode, module_type = variant_map[selected_variant]

    if module_type == "chunk":
        extracted = _extract_chunk_progress_module_state(projection_params, module_key)
    else:
        raise ValueError(f"Unsupported progress module type: {module_type}")
    if extracted is None:
        return base_progress_params, None
    head_type, state_dict = extracted

    payload = {
        "format": "openpi_progress_head_v2",
        "head_type": head_type,
        "readout_mode": readout_mode,
        "source_module": module_key,
        "state_dict": state_dict,
    }
    # Attach dimensions for easier downstream validation/debug.
    first_weight = next((v for k, v in state_dict.items() if k.endswith("weight")), None)
    if first_weight is not None:
        payload["first_weight_shape"] = list(first_weight.shape)
    return base_progress_params, payload


def _save_progress_head_checkpoint(
    progress_payload: dict[str, object],
    output_path: str,
    progress_head_filename: str,
) -> str:
    """Save standalone progress head checkpoint with metadata."""
    state_dict = progress_payload.get("state_dict", {})
    if not isinstance(state_dict, dict):
        raise ValueError("progress_payload.state_dict must be a dict")

    cpu_state_dict = {}
    for key, value in state_dict.items():
        cpu_state_dict[key] = value.detach().cpu() if torch.is_tensor(value) else torch.tensor(value)

    to_save = dict(progress_payload)
    to_save["state_dict"] = cpu_state_dict
    progress_path = os.path.join(output_path, progress_head_filename)
    torch.save(to_save, progress_path)
    return progress_path


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
    progress_head_filename: str = "progress_head.pt",
    progress_variant: Literal["auto", "chunk_prefix"] = "auto",
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
    progress_params, progress_payload = _extract_progress_payload(
        initial_params, checkpoint_dir=checkpoint_dir, progress_variant=progress_variant
    )
    if progress_params is None:
        print("[LoRA-aware+Progress] No chunk-prefix progress params found for base model load.")
    else:
        print("[LoRA-aware+Progress] Found chunk-prefix progress params for base model load.")
    if progress_payload is None:
        print("[LoRA-aware+Progress] No standalone progress payload extracted.")
    else:
        print(
            "[LoRA-aware+Progress] Selected standalone progress head:",
            f"source={progress_payload.get('source_module')},",
            f"type={progress_payload.get('head_type')},",
            f"readout={progress_payload.get('readout_mode')}",
        )

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
    supports_progress_module = (
        hasattr(pi0_model, "progress_chunk_prefix_out_proj")
        and getattr(pi0_model, "progress_chunk_prefix_out_proj") is not None
    )
    if progress_params is not None and supports_progress_module:
        # Also load into full model when model architecture includes the chunk-prefix progress head.
        all_params.update(progress_params)
    elif progress_params is not None:
        print(
            "[LoRA-aware+Progress] PI0Pytorch in current env has no chunk-prefix progress module; "
            "keep progress head as standalone checkpoint only."
        )
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
    progress_head_path = None
    if progress_payload is not None:
        progress_head_path = _save_progress_head_checkpoint(
            progress_payload, output_path, progress_head_filename
        )
        print(f"[LoRA-aware+Progress] Saved progress head: {progress_head_path}")
    _copy_norm_assets_to_output_root(checkpoint_dir, output_path)

    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": precision,
        "lora_merged_pairs": merge_stats["merged_pairs"],
        "has_progress_head": progress_payload is not None,
        "progress_head_file": progress_head_filename if progress_payload is not None else None,
        "progress_head_type": progress_payload.get("head_type") if progress_payload is not None else None,
        "progress_head_readout_mode": progress_payload.get("readout_mode") if progress_payload is not None else None,
        "progress_head_source_module": progress_payload.get("source_module") if progress_payload is not None else None,
        "progress_variant_arg": progress_variant,
        "progress_head_loaded_into_model": bool(
            progress_params is not None and supports_progress_module
        ),
        "lora_scales": {
            "base_attn": merge_stats["base_attn_scale"],
            "base_ffn": merge_stats["base_ffn_scale"],
            "expert_attn": merge_stats["expert_attn_scale"],
            "expert_ffn": merge_stats["expert_ffn_scale"],
        },
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("[LoRA-aware+Progress] Conversion completed.")
    print("[LoRA-aware+Progress] missing_keys:", len(msg.missing_keys), msg.missing_keys)
    print("[LoRA-aware+Progress] unexpected_keys:", len(msg.unexpected_keys), msg.unexpected_keys[:10])
    print("[LoRA-aware+Progress] Saved:", output_path)


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    progress_head_filename: str = "progress_head.pt",
    progress_variant: Literal["auto", "chunk_prefix"] = "auto",
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
        progress_head_filename=progress_head_filename,
        progress_variant=progress_variant,
    )


if __name__ == "__main__":
    tyro.cli(main)
