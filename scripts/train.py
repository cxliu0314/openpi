import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        wandb_id_file = ckpt_dir / "wandb_id.txt"
        if wandb_id_file.exists():
            run_id = wandb_id_file.read_text().strip()
            wandb.init(id=run_id, resume="must", project=config.project_name)
        else:
            # Checkpoint created without wandb — start a fresh tracked run
            wandb.init(
                name=config.exp_name,
                config=dataclasses.asdict(config),
                project=config.project_name,
            )
            wandb_id_file.write_text(wandb.run.id)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


CURRENT_RELATIVE_PROGRESS_MODES = (
    "chunk_current_relative_flat",
    "chunk_current_relative_multilayer",
)


def _compute_action_and_progress_with_mode(
    model: _model.BaseModel,
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    train: bool,
):
    method_name = {
        "prefix": "compute_action_and_progress",
        "low_noise_action": "compute_action_and_progress_low_noise",
        "chunk_prefix": "compute_action_and_progress_chunk_prefix",
        "chunk_prefix_large": "compute_action_and_progress_chunk_prefix_large",
        "chunk_low_noise_action": "compute_action_and_progress_chunk_low_noise",
        "chunk_hybrid_concat": "compute_action_and_progress_chunk_hybrid",
        "chunk_self_action": "compute_action_and_progress_chunk_self_action",
        "chunk_hybrid_self_action": "compute_action_and_progress_chunk_hybrid_self_action",
        "chunk_hybrid_self_action_large": "compute_action_and_progress_chunk_hybrid_self_action_large",
        "chunk_multilayer_self_action": "compute_action_and_progress_chunk_multilayer_self_action",
        "chunk_current_relative_flat": "compute_action_and_progress_chunk_current_relative_flat",
        "chunk_current_relative_multilayer": "compute_action_and_progress_chunk_current_relative_multilayer",
    }.get(config.progress_readout_mode, "compute_action_and_progress")
    if hasattr(model, method_name):
        return getattr(model, method_name)(rng, observation, actions, train=train)
    if hasattr(model, "compute_action_and_progress"):
        return model.compute_action_and_progress(rng, observation, actions, train=train)
    return model.compute_loss(rng, observation, actions, train=train), None


def _compute_progress_metrics(
    config: _config.TrainConfig,
    progress_pred: at.Array | None,
    progress_target: at.Array,
    mask: at.Array,
    *,
    masked_mean,
    prefix: str,
) -> tuple[at.Array, dict[str, at.Array]]:
    zero = jnp.array(0.0, dtype=progress_target.dtype)
    metrics = {
        f"{prefix}/current_progress_loss": zero,
        f"{prefix}/relative_progress_loss": zero,
        f"{prefix}/current_pred_mean": zero,
        f"{prefix}/current_target_mean": zero,
        f"{prefix}/relative_pred_mean": zero,
        f"{prefix}/relative_target_mean": zero,
        f"{prefix}/progress_pred_last_mean": zero,
        f"{prefix}/progress_target_last_mean": zero,
    }
    if not config.enable_progress_loss or progress_pred is None:
        return zero, metrics

    progress_target = jnp.reshape(progress_target, progress_pred.shape)
    if config.progress_readout_mode in CURRENT_RELATIVE_PROGRESS_MODES:
        current_pred = progress_pred[..., 0]
        current_target = progress_target[..., 0]
        current_loss_per_sample = jnp.square(current_pred - current_target)
        current_loss = masked_mean(current_loss_per_sample, mask)

        if progress_pred.shape[-1] > 1:
            relative_pred = progress_pred[..., 1:] - progress_pred[..., :1]
            relative_target = progress_target[..., 1:] - progress_target[..., :1]
            relative_error = jnp.square(relative_pred - relative_target)
            relative_loss_per_sample = jnp.mean(relative_error, axis=-1)
            relative_pred_mean_per_sample = jnp.mean(relative_pred, axis=-1)
            relative_target_mean_per_sample = jnp.mean(relative_target, axis=-1)
            progress_pred_last = progress_pred[..., -1]
            progress_target_last = progress_target[..., -1]
        else:
            relative_loss_per_sample = jnp.zeros_like(current_loss_per_sample)
            relative_pred_mean_per_sample = jnp.zeros_like(current_pred)
            relative_target_mean_per_sample = jnp.zeros_like(current_target)
            progress_pred_last = current_pred
            progress_target_last = current_target

        relative_loss = masked_mean(relative_loss_per_sample, mask)
        progress_loss = current_loss + config.progress_relative_loss_weight * relative_loss
        metrics.update(
            {
                f"{prefix}/current_progress_loss": current_loss,
                f"{prefix}/relative_progress_loss": relative_loss,
                f"{prefix}/current_pred_mean": masked_mean(current_pred, mask),
                f"{prefix}/current_target_mean": masked_mean(current_target, mask),
                f"{prefix}/relative_pred_mean": masked_mean(relative_pred_mean_per_sample, mask),
                f"{prefix}/relative_target_mean": masked_mean(relative_target_mean_per_sample, mask),
                f"{prefix}/progress_pred_last_mean": masked_mean(progress_pred_last, mask),
                f"{prefix}/progress_target_last_mean": masked_mean(progress_target_last, mask),
            }
        )
        return progress_loss, metrics

    progress_error = jnp.square(progress_pred - progress_target)
    if progress_error.ndim > 1:
        progress_loss_per_sample = jnp.mean(progress_error, axis=tuple(range(1, progress_error.ndim)))
    else:
        progress_loss_per_sample = progress_error
    progress_loss = masked_mean(progress_loss_per_sample, mask)
    metrics[f"{prefix}/progress_pred_last_mean"] = masked_mean(progress_pred[..., -1], mask)
    metrics[f"{prefix}/progress_target_last_mean"] = masked_mean(progress_target[..., -1], mask)
    return progress_loss, metrics


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, at.Array, at.Array],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def split_masks(episode_hash: at.Array) -> tuple[at.Array, at.Array]:
        if config.use_val_set:
            bucket = 1000
            threshold = int(config.val_split_ratio * bucket)
            val_mask = jnp.mod(jnp.abs(episode_hash.astype(jnp.int32)), bucket) < threshold
        else:
            val_mask = jnp.zeros_like(episode_hash, dtype=jnp.bool_)
        return ~val_mask, val_mask

    def masked_mean(values: at.Array, mask: at.Array) -> at.Array:
        mask_f = mask.astype(values.dtype)
        denom = jnp.maximum(jnp.sum(mask_f), 1.0)
        return jnp.sum(values * mask_f) / denom

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        progress_target: at.Array,
        episode_hash: at.Array,
    ):
        train_mask, _ = split_masks(episode_hash)

        chunked_loss, progress_pred = _compute_action_and_progress_with_mode(
            model, config, rng, observation, actions, train=True
        )

        action_loss_per_sample = jnp.mean(chunked_loss, axis=-1)
        action_loss = masked_mean(action_loss_per_sample, train_mask)

        progress_loss, progress_metrics = _compute_progress_metrics(
            config,
            progress_pred,
            progress_target,
            train_mask,
            masked_mean=masked_mean,
            prefix="train",
        )

        total_loss = action_loss + config.progress_loss_weight * progress_loss
        aux = {
            "loss": total_loss,
            "train/action_loss": action_loss,
            "train/progress_loss": progress_loss,
            "train/total_loss": total_loss,
            **progress_metrics,
        }
        return total_loss, aux

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, progress_target, episode_hash = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True, argnums=diff_state)(
        model,
        train_rng,
        observation,
        actions,
        progress_target,
        episode_hash,
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "train/action_loss": aux["train/action_loss"],
        "train/progress_loss": aux["train/progress_loss"],
        "train/total_loss": aux["train/total_loss"],
        "train/current_progress_loss": aux["train/current_progress_loss"],
        "train/relative_progress_loss": aux["train/relative_progress_loss"],
        "train/current_pred_mean": aux["train/current_pred_mean"],
        "train/current_target_mean": aux["train/current_target_mean"],
        "train/relative_pred_mean": aux["train/relative_pred_mean"],
        "train/relative_target_mean": aux["train/relative_target_mean"],
        "train/progress_pred_last_mean": aux["train/progress_pred_last_mean"],
        "train/progress_target_last_mean": aux["train/progress_target_last_mean"],
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, at.Array, at.Array],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    observation, actions, progress_target, episode_hash = batch

    bucket = 1000
    threshold = int(config.val_split_ratio * bucket)
    val_mask = jnp.mod(jnp.abs(episode_hash.astype(jnp.int32)), bucket) < threshold

    def masked_mean(values: at.Array, mask: at.Array) -> at.Array:
        mask_f = mask.astype(values.dtype)
        denom = jnp.maximum(jnp.sum(mask_f), 1.0)
        return jnp.sum(values * mask_f) / denom

    eval_rng = jax.random.key(0)
    chunked_loss, progress_pred = _compute_action_and_progress_with_mode(
        model, config, eval_rng, observation, actions, train=False
    )

    action_loss_per_sample = jnp.mean(chunked_loss, axis=-1)
    action_loss = masked_mean(action_loss_per_sample, val_mask)

    progress_loss, progress_metrics = _compute_progress_metrics(
        config,
        progress_pred,
        progress_target,
        val_mask,
        masked_mean=masked_mean,
        prefix="val",
    )

    total_loss = action_loss + config.progress_loss_weight * progress_loss
    return {
        "val/action_loss": action_loss,
        "val/progress_loss": progress_loss,
        "val/total_loss": total_loss,
        **progress_metrics,
    }


def _log_model_batch_sample(step: int, batch: tuple[_model.Observation, _model.Actions, at.Array, at.Array]) -> None:
    observation, actions, progress_target, episode_hash = batch
    state_np = np.asarray(jax.device_get(observation.state))
    actions_np = np.asarray(jax.device_get(actions))
    progress_target_np = np.asarray(jax.device_get(progress_target))
    episode_hash_np = np.asarray(jax.device_get(episode_hash)).reshape(-1)

    lines = [
        f"episode_hash={int(episode_hash_np[0])}",
        f"state[:8]={np.array2string(state_np[0, : min(8, state_np.shape[-1])], precision=4, suppress_small=False)}",
    ]

    action_preview = actions_np[0, : min(3, actions_np.shape[1]), : min(6, actions_np.shape[2])]
    lines.append(
        f"actions[:{action_preview.shape[0]}, :{action_preview.shape[1]}]="
        + np.array2string(action_preview, precision=4, suppress_small=False)
    )

    if progress_target_np.ndim == 1:
        lines.append(f"progress_target={float(progress_target_np[0]):.4f}")
    else:
        progress_preview = progress_target_np[0]
        lines.append(
            f"progress_target[:{progress_preview.shape[0]}]="
            + np.array2string(progress_preview, precision=4, suppress_small=False)
        )
        progress_diffs = np.diff(progress_target_np, axis=-1)
        lines.append(
            f"progress_target_monotonic_violations={int((progress_diffs < -1e-6).sum())}, "
            f"progress_target_max_jump={float(np.abs(progress_diffs).max()):.4f}"
        )

    if observation.tokenized_prompt is not None:
        tokenized_prompt_np = np.asarray(jax.device_get(observation.tokenized_prompt))
        prompt_mask_np = (
            np.asarray(jax.device_get(observation.tokenized_prompt_mask))
            if observation.tokenized_prompt_mask is not None
            else None
        )
        prompt_valid = int(prompt_mask_np[0].sum()) if prompt_mask_np is not None else int(tokenized_prompt_np.shape[-1])
        prompt_preview = tokenized_prompt_np[0, : min(24, tokenized_prompt_np.shape[-1])]
        lines.append(
            f"tokenized_prompt_valid={prompt_valid}, tokenized_prompt[:{len(prompt_preview)}]="
            + np.array2string(prompt_preview, separator=", ")
        )

    for name, image in observation.images.items():
        image_np = np.asarray(jax.device_get(image))[0]
        mask_value = (
            bool(np.asarray(jax.device_get(observation.image_masks[name])).reshape(-1)[0])
            if name in observation.image_masks
            else True
        )
        lines.append(
            f"image[{name}]: mask={mask_value}, mean={float(image_np.mean()):.4f}, std={float(image_np.std()):.4f}, "
            f"min={float(image_np.min()):.4f}, max={float(image_np.max()):.4f}"
        )

    logging.info("model_batch_sample[step=%d]\n%s", step, "\n".join(f"  {line}" for line in lines))


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from the first local shard on process 0 to avoid cross-device gathers.
    if config.wandb_enabled and jax.process_index() == 0:
        local_images = {}
        for name, img in batch[0].images.items():
            local_img = img
            if hasattr(local_img, "addressable_shards") and local_img.addressable_shards:
                local_img = local_img.addressable_shards[0].data
            local_images[name] = np.asarray(jax.device_get(local_img))

        if local_images:
            local_batch_size = min(img.shape[0] for img in local_images.values())
            images_to_log = [
                wandb.Image(np.concatenate([img[i] for img in local_images.values()], axis=1))
                for i in range(min(5, local_batch_size))
            ]
            wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    peval_step = jax.jit(
        functools.partial(eval_step, config),
        in_shardings=(train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    def _create_val_iter():
        return iter(
            _data_loader.create_data_loader(
                config,
                sharding=data_sharding,
                shuffle=False,
                num_batches=config.val_num_batches if config.use_val_set else None,
            )
        )

    val_iter = _create_val_iter()

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        if step < 8:
            progress_target_np = np.asarray(jax.device_get(batch[2]))
            progress_sample0 = float(progress_target_np.reshape(-1)[0]) if progress_target_np.size > 0 else float("nan")
            logging.info(
                "train_progress_target_debug[%d]: shape=%s, sample0=%.4f",
                step,
                tuple(progress_target_np.shape),
                progress_sample0,
            )

        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            _log_model_batch_sample(step, batch)
            progress_target_np = np.asarray(jax.device_get(batch[2]))
            if progress_target_np.size > 0:
                progress_sample0_str = f"{float(progress_target_np.reshape(-1)[0]):.4f}"
                reduced_info["train/progress_target_min"] = float(progress_target_np.min())
                reduced_info["train/progress_target_max"] = float(progress_target_np.max())
                reduced_info["train/progress_target_mean"] = float(progress_target_np.mean())
            else:
                progress_sample0_str = "nan"
            if config.use_val_set and step > 0 and step % config.val_interval == 0:
                val_infos = []
                for _ in range(max(1, config.val_num_batches)):
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = _create_val_iter()
                        val_batch = next(val_iter)
                    with sharding.set_mesh(mesh):
                        val_info = peval_step(train_state, val_batch)
                    val_infos.append(val_info)
                stacked_val_infos = common_utils.stack_forest(val_infos)
                reduced_val_info = jax.device_get(jax.tree.map(jnp.mean, stacked_val_infos))
                reduced_info.update(reduced_val_info)
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}, progress_target_sample0={progress_sample0_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
