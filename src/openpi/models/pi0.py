import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import progress_current_relative as _progress_current_relative
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

CHUNK_PROGRESS_ADAPTER_DIM = 1024
CHUNK_PROGRESS_HIDDEN_DIM = 512
CHUNK_PROGRESS_MID_DIM = 128
CHUNK_PROGRESS_LARGE_ADAPTER_DIM = 1536
CHUNK_PROGRESS_LARGE_HIDDEN_DIM = 1024
CHUNK_PROGRESS_LARGE_MID_DIM = 256
CHUNK_MULTILAYER_PREFIX_PROJ_DIM = 128
CHUNK_MULTILAYER_PREFIX_FUSE_DIM = 256
PROGRESS_SAMPLER_NUM_STEPS = 10


class ProgressHead(nnx.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int | None = None, mid_dim: int | None = None, rngs: nnx.Rngs):
        hidden_dim = min(input_dim, max(input_dim // 2, 256)) if hidden_dim is None else hidden_dim
        mid_dim = min(hidden_dim, max(hidden_dim // 4, 64)) if mid_dim is None else mid_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mid_dim = mid_dim
        self.fc1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, mid_dim, rngs=rngs)
        self.fc3 = nnx.Linear(mid_dim, 1, rngs=rngs)

    def predict_progress(self, hidden_states: at.Float[at.Array, "*b emb"]) -> at.Float[at.Array, "*b"]:
        x = self.fc1(hidden_states)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        x = jax.nn.relu(x)
        x = self.fc3(x)
        return jax.nn.sigmoid(x[..., 0])


class ChunkProgressHead(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        adapter_dim: int = CHUNK_PROGRESS_ADAPTER_DIM,
        hidden_dim: int = CHUNK_PROGRESS_HIDDEN_DIM,
        mid_dim: int = CHUNK_PROGRESS_MID_DIM,
        rngs: nnx.Rngs,
    ):
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        self.input_adapter = nnx.Linear(input_dim, adapter_dim, rngs=rngs)
        self.shared_head = ProgressHead(adapter_dim, hidden_dim=hidden_dim, mid_dim=mid_dim, rngs=rngs)

    def adapt_inputs(self, hidden_states: at.Float[at.Array, "*b emb"]) -> at.Float[at.Array, "*b adapted"]:
        return self.input_adapter(hidden_states)

    def predict_from_adapted(self, adapted_hidden_states: at.Float[at.Array, "*b adapted"]) -> at.Float[at.Array, "*b"]:
        return self.shared_head.predict_progress(adapted_hidden_states)

    def predict_progress(self, hidden_states: at.Float[at.Array, "*b emb"]) -> at.Float[at.Array, "*b"]:
        return self.predict_from_adapted(self.adapt_inputs(hidden_states))


class ChunkMultilayerHybridProgressHead(nnx.Module):
    def __init__(
        self,
        prefix_input_dim: int,
        action_input_dim: int,
        *,
        prefix_proj_dim: int = CHUNK_MULTILAYER_PREFIX_PROJ_DIM,
        prefix_fuse_dim: int = CHUNK_MULTILAYER_PREFIX_FUSE_DIM,
        adapter_dim: int = CHUNK_PROGRESS_ADAPTER_DIM,
        hidden_dim: int = CHUNK_PROGRESS_HIDDEN_DIM,
        mid_dim: int = CHUNK_PROGRESS_MID_DIM,
        rngs: nnx.Rngs,
    ):
        self.prefix_proj_dim = prefix_proj_dim
        self.prefix_fuse_dim = prefix_fuse_dim
        self.prefix_proj_0 = nnx.Linear(prefix_input_dim, prefix_proj_dim, rngs=rngs)
        self.prefix_proj_1 = nnx.Linear(prefix_input_dim, prefix_proj_dim, rngs=rngs)
        self.prefix_proj_2 = nnx.Linear(prefix_input_dim, prefix_proj_dim, rngs=rngs)
        self.prefix_fuse = nnx.Linear(prefix_proj_dim * 3, prefix_fuse_dim, rngs=rngs)
        self.chunk_head = ChunkProgressHead(
            prefix_fuse_dim + action_input_dim,
            adapter_dim=adapter_dim,
            hidden_dim=hidden_dim,
            mid_dim=mid_dim,
            rngs=rngs,
        )

    def _fuse_prefix(
        self,
        prefix_layer_ctxs: tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b d"], at.Float[at.Array, "b d"]],
    ) -> at.Float[at.Array, "b d"]:
        prefix_feats = [
            nnx.swish(self.prefix_proj_0(prefix_layer_ctxs[0])),
            nnx.swish(self.prefix_proj_1(prefix_layer_ctxs[1])),
            nnx.swish(self.prefix_proj_2(prefix_layer_ctxs[2])),
        ]
        return nnx.swish(self.prefix_fuse(jnp.concatenate(prefix_feats, axis=-1)))

    def predict_progress(
        self,
        prefix_layer_ctxs: tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b d"], at.Float[at.Array, "b d"]],
        action_tokens: at.Float[at.Array, "b h d"],
    ) -> at.Float[at.Array, "b h"]:
        fused_prefix = self._fuse_prefix(prefix_layer_ctxs)
        prefix_chunk = jnp.broadcast_to(
            fused_prefix[:, None, :],
            (fused_prefix.shape[0], action_tokens.shape[1], fused_prefix.shape[-1]),
        )
        hybrid_features = jnp.concatenate([prefix_chunk, action_tokens], axis=-1)
        return self.chunk_head.predict_progress(hybrid_features)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.enable_progress_head = bool(config.enable_progress_head)
        self.progress_out_proj = ProgressHead(paligemma_config.width, rngs=rngs) if self.enable_progress_head else None
        self.progress_out_proj_action = (
            ProgressHead(action_expert_config.width, rngs=rngs) if self.enable_progress_head else None
        )
        self.progress_chunk_prefix_out_proj = (
            ChunkProgressHead(paligemma_config.width, rngs=rngs) if self.enable_progress_head else None
        )
        self.progress_chunk_action_out_proj = (
            ChunkProgressHead(action_expert_config.width, rngs=rngs) if self.enable_progress_head else None
        )
        self.progress_chunk_hybrid_out_proj = (
            ChunkProgressHead(paligemma_config.width + action_expert_config.width, rngs=rngs)
            if self.enable_progress_head
            else None
        )
        self.progress_chunk_prefix_large_out_proj = (
            ChunkProgressHead(
                paligemma_config.width,
                adapter_dim=CHUNK_PROGRESS_LARGE_ADAPTER_DIM,
                hidden_dim=CHUNK_PROGRESS_LARGE_HIDDEN_DIM,
                mid_dim=CHUNK_PROGRESS_LARGE_MID_DIM,
                rngs=rngs,
            )
            if self.enable_progress_head
            else None
        )
        self.progress_chunk_hybrid_large_out_proj = (
            ChunkProgressHead(
                paligemma_config.width + action_expert_config.width,
                adapter_dim=CHUNK_PROGRESS_LARGE_ADAPTER_DIM,
                hidden_dim=CHUNK_PROGRESS_LARGE_HIDDEN_DIM,
                mid_dim=CHUNK_PROGRESS_LARGE_MID_DIM,
                rngs=rngs,
            )
            if self.enable_progress_head
            else None
        )
        self.progress_chunk_multilayer_hybrid_out_proj = (
            ChunkMultilayerHybridProgressHead(
                paligemma_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
            if self.enable_progress_head
            else None
        )
        self.progress_chunk_current_relative_flat_out_proj = (
            _progress_current_relative.CurrentRelativeProgressHeadFlat(
                paligemma_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
            if self.enable_progress_head
            else None
        )
        self.progress_chunk_current_relative_multilayer_out_proj = (
            _progress_current_relative.CurrentRelativeProgressHeadMultilayer(
                paligemma_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
            if self.enable_progress_head
            else None
        )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @at.typecheck
    def _pool_prefix_features(
        self, prefix_hidden_states: at.Float[at.Array, "b s emb"], prefix_mask: at.Bool[at.Array, "b s"]
    ) -> at.Float[at.Array, "b emb"]:
        weights = prefix_mask.astype(jnp.float32)[..., None]
        denom = jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        pooled = jnp.sum(prefix_hidden_states.astype(jnp.float32) * weights, axis=1) / denom
        return pooled.astype(prefix_hidden_states.dtype)

    def _chunk_step_embedding(self, dtype: jnp.dtype) -> at.Float[at.Array, "ah adapted"]:
        if self.action_horizon == 1:
            positions = jnp.zeros((1,), dtype=jnp.float32)
        else:
            positions = jnp.linspace(0.0, 1.0, self.action_horizon, dtype=jnp.float32)
        return posemb_sincos(
            positions,
            CHUNK_PROGRESS_ADAPTER_DIM,
            min_period=4e-3,
            max_period=4.0,
        ).astype(dtype)

    def _prepare_training_inputs(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[
        _model.Observation,
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, " b"],
    ]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        return observation, u_t, x_t, time

    def _compute_chunked_action_loss(
        self, action_tokens: at.Float[at.Array, "*b ah emb"], u_t: at.Float[at.Array, "*b ah ad"]
    ) -> at.Float[at.Array, "*b ah"]:
        v_t = self.action_out_proj(action_tokens)
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _encode_low_noise_action_tokens(
        self,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        progress_t: float,
    ) -> at.Float[at.Array, "*b ah action_emb"]:
        batch_shape = actions.shape[:-2]
        progress_time = jnp.full(batch_shape, progress_t, dtype=actions.dtype)
        progress_time_expanded = progress_time[..., None, None]
        clean_action_tokens = (1 - progress_time_expanded) * actions
        _progress_prefix_out, _progress_prefix_mask, progress_action_tokens = self._encode_suffix_tokens(
            observation, clean_action_tokens, progress_time
        )
        return progress_action_tokens

    @at.typecheck
    def _encode_suffix_tokens(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Float[at.Array, "*b ah action_emb"],
    ]:
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, noisy_actions, timestep)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        assert prefix_out is not None
        action_tokens = suffix_out[:, -self.action_horizon :]
        return prefix_out, prefix_mask, action_tokens

    @at.typecheck
    def _encode_suffix_tokens_with_pooled_prefix_captures(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        capture_layer_indices: tuple[int, ...],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Float[at.Array, "*b ah action_emb"],
        tuple[at.Float[at.Array, "b emb"], ...],
    ]:
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, noisy_actions, timestep)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), prefix_pooled_captures, _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            prefix_mask=prefix_mask,
            capture_layer_indices=capture_layer_indices,
            method="forward_with_pooled_prefix_captures",
        )
        assert prefix_out is not None
        action_tokens = suffix_out[:, -self.action_horizon :]
        return prefix_out, prefix_mask, action_tokens, tuple(prefix_pooled_captures)

    def _sample_progress_action_tokens(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = PROGRESS_SAMPLER_NUM_STEPS,
    ) -> at.Float[at.Array, "b h action_emb"]:
        batch_size = observation.state.shape[0]
        dt = -1.0 / num_steps
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)

        sample_times = 1.0 + dt * jnp.arange(num_steps, dtype=jnp.float32)

        def step(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.full((batch_size,), time, dtype=jnp.float32)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            action_tokens = suffix_out[:, -self.action_horizon :]
            v_t = self.action_out_proj(action_tokens)
            return x_t + dt * v_t, action_tokens

        _sampled_actions, sampled_action_tokens = jax.lax.scan(step, noise, sample_times)
        return sampled_action_tokens[-1]

    def _forward_action_tokens(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, "*b ah ad"],
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Float[at.Array, "*b ah emb"],
    ]:
        observation, u_t, x_t, time = self._prepare_training_inputs(rng, observation, actions, train=train)
        prefix_out, prefix_mask, action_tokens = self._encode_suffix_tokens(observation, x_t, time)
        return u_t, x_t, prefix_out, prefix_mask, action_tokens

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        u_t, _x_t, _prefix_out, _prefix_mask, action_tokens = self._forward_action_tokens(
            rng, observation, actions, train=train
        )
        return self._compute_chunked_action_loss(action_tokens, u_t)

    @at.typecheck
    def compute_action_and_progress(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b"] | None]:
        u_t, _x_t, prefix_out, prefix_mask, action_tokens = self._forward_action_tokens(
            rng, observation, actions, train=train
        )
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_out_proj is None:
            return chunked_loss, None
        progress_feature = self._pool_prefix_features(prefix_out, prefix_mask)
        progress_pred = self.progress_out_proj.predict_progress(progress_feature)
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_low_noise(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        progress_t: float = 0.001,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b"] | None]:
        observation, u_t, x_t, time = self._prepare_training_inputs(rng, observation, actions, train=train)
        _prefix_out, _prefix_mask, action_tokens = self._encode_suffix_tokens(observation, x_t, time)
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_out_proj_action is None:
            return chunked_loss, None
        progress_action_tokens = self._encode_low_noise_action_tokens(observation, actions, progress_t=progress_t)
        progress_feature = progress_action_tokens[..., 0, :]
        progress_pred = self.progress_out_proj_action.predict_progress(progress_feature)
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_chunk_prefix(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"] | None]:
        u_t, _x_t, prefix_out, prefix_mask, action_tokens = self._forward_action_tokens(
            rng, observation, actions, train=train
        )
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_chunk_prefix_out_proj is None:
            return chunked_loss, None
        pooled_prefix = self._pool_prefix_features(prefix_out, prefix_mask)
        adapted_prefix = self.progress_chunk_prefix_out_proj.adapt_inputs(pooled_prefix)
        step_embedding = self._chunk_step_embedding(adapted_prefix.dtype)
        chunk_features = adapted_prefix[:, None, :] + step_embedding[None, :, :]
        progress_pred = self.progress_chunk_prefix_out_proj.predict_from_adapted(chunk_features)
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_chunk_prefix_large(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"] | None]:
        u_t, _x_t, prefix_out, prefix_mask, action_tokens = self._forward_action_tokens(
            rng, observation, actions, train=train
        )
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_chunk_prefix_large_out_proj is None:
            return chunked_loss, None
        pooled_prefix = self._pool_prefix_features(prefix_out, prefix_mask)
        adapted_prefix = self.progress_chunk_prefix_large_out_proj.adapt_inputs(pooled_prefix)
        step_embedding = posemb_sincos(
            jnp.linspace(0.0, 1.0, self.action_horizon, dtype=jnp.float32)
            if self.action_horizon > 1
            else jnp.zeros((1,), dtype=jnp.float32),
            adapted_prefix.shape[-1],
            min_period=4e-3,
            max_period=4.0,
        ).astype(adapted_prefix.dtype)
        chunk_features = adapted_prefix[:, None, :] + step_embedding[None, :, :]
        progress_pred = self.progress_chunk_prefix_large_out_proj.predict_from_adapted(chunk_features)
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_chunk_low_noise(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        progress_t: float = 0.001,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"] | None]:
        observation, u_t, x_t, time = self._prepare_training_inputs(rng, observation, actions, train=train)
        _prefix_out, _prefix_mask, action_tokens = self._encode_suffix_tokens(observation, x_t, time)
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_chunk_action_out_proj is None:
            return chunked_loss, None
        progress_action_tokens = self._encode_low_noise_action_tokens(observation, actions, progress_t=progress_t)
        progress_pred = self.progress_chunk_action_out_proj.predict_progress(progress_action_tokens)
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_chunk_hybrid(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        progress_t: float = 0.001,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"] | None]:
        observation, u_t, x_t, time = self._prepare_training_inputs(rng, observation, actions, train=train)
        prefix_out, prefix_mask, action_tokens = self._encode_suffix_tokens(observation, x_t, time)
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_chunk_hybrid_out_proj is None:
            return chunked_loss, None
        pooled_prefix = self._pool_prefix_features(prefix_out, prefix_mask)
        prefix_chunk = jnp.broadcast_to(
            pooled_prefix[:, None, :],
            (pooled_prefix.shape[0], self.action_horizon, pooled_prefix.shape[-1]),
        )
        progress_action_tokens = self._encode_low_noise_action_tokens(observation, actions, progress_t=progress_t)
        hybrid_features = jnp.concatenate([prefix_chunk, progress_action_tokens], axis=-1)
        progress_pred = self.progress_chunk_hybrid_out_proj.predict_progress(hybrid_features)
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_chunk_self_action(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"] | None]:
        train_rng, progress_rng = jax.random.split(rng)
        observation, u_t, x_t, time = self._prepare_training_inputs(train_rng, observation, actions, train=train)
        _prefix_out, _prefix_mask, action_tokens = self._encode_suffix_tokens(observation, x_t, time)
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_chunk_action_out_proj is None:
            return chunked_loss, None
        progress_action_tokens = jax.lax.stop_gradient(self._sample_progress_action_tokens(progress_rng, observation))
        progress_pred = self.progress_chunk_action_out_proj.predict_progress(progress_action_tokens)
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_chunk_hybrid_self_action(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"] | None]:
        train_rng, progress_rng = jax.random.split(rng)
        observation, u_t, x_t, time = self._prepare_training_inputs(train_rng, observation, actions, train=train)
        prefix_out, prefix_mask, action_tokens = self._encode_suffix_tokens(observation, x_t, time)
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_chunk_hybrid_out_proj is None:
            return chunked_loss, None
        pooled_prefix = self._pool_prefix_features(prefix_out, prefix_mask)
        prefix_chunk = jnp.broadcast_to(
            pooled_prefix[:, None, :],
            (pooled_prefix.shape[0], self.action_horizon, pooled_prefix.shape[-1]),
        )
        progress_action_tokens = jax.lax.stop_gradient(self._sample_progress_action_tokens(progress_rng, observation))
        hybrid_features = jnp.concatenate([prefix_chunk, progress_action_tokens], axis=-1)
        progress_pred = self.progress_chunk_hybrid_out_proj.predict_progress(hybrid_features)
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_chunk_hybrid_self_action_large(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"] | None]:
        train_rng, progress_rng = jax.random.split(rng)
        observation, u_t, x_t, time = self._prepare_training_inputs(train_rng, observation, actions, train=train)
        prefix_out, prefix_mask, action_tokens = self._encode_suffix_tokens(observation, x_t, time)
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_chunk_hybrid_large_out_proj is None:
            return chunked_loss, None
        pooled_prefix = self._pool_prefix_features(prefix_out, prefix_mask)
        prefix_chunk = jnp.broadcast_to(
            pooled_prefix[:, None, :],
            (pooled_prefix.shape[0], self.action_horizon, pooled_prefix.shape[-1]),
        )
        progress_action_tokens = jax.lax.stop_gradient(self._sample_progress_action_tokens(progress_rng, observation))
        hybrid_features = jnp.concatenate([prefix_chunk, progress_action_tokens], axis=-1)
        progress_pred = self.progress_chunk_hybrid_large_out_proj.predict_progress(hybrid_features)
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_chunk_multilayer_self_action(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"] | None]:
        train_rng, progress_rng = jax.random.split(rng)
        observation, u_t, x_t, time = self._prepare_training_inputs(train_rng, observation, actions, train=train)
        prefix_out, prefix_mask, action_tokens, prefix_pooled_captures = self._encode_suffix_tokens_with_pooled_prefix_captures(
            observation,
            x_t,
            time,
            capture_layer_indices=_progress_current_relative.PREFIX_CAPTURE_LAYERS,
        )
        del prefix_out, prefix_mask
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_chunk_multilayer_hybrid_out_proj is None:
            return chunked_loss, None
        if len(prefix_pooled_captures) != len(_progress_current_relative.PREFIX_CAPTURE_LAYERS):
            raise ValueError(
                f"Expected {len(_progress_current_relative.PREFIX_CAPTURE_LAYERS)} prefix captures, got {len(prefix_pooled_captures)}"
            )
        progress_action_tokens = jax.lax.stop_gradient(self._sample_progress_action_tokens(progress_rng, observation))
        progress_pred = self.progress_chunk_multilayer_hybrid_out_proj.predict_progress(
            tuple(prefix_pooled_captures),
            progress_action_tokens,
        )
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_chunk_current_relative_flat(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        progress_t: float = 0.001,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"] | None]:
        observation, u_t, x_t, time = self._prepare_training_inputs(rng, observation, actions, train=train)
        prefix_out, prefix_mask, action_tokens = self._encode_suffix_tokens(observation, x_t, time)
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_chunk_current_relative_flat_out_proj is None:
            return chunked_loss, None
        pooled_prefix = self._pool_prefix_features(prefix_out, prefix_mask)
        progress_action_tokens = self._encode_low_noise_action_tokens(observation, actions, progress_t=progress_t)
        progress_pred = self.progress_chunk_current_relative_flat_out_proj.predict_progress(
            pooled_prefix,
            progress_action_tokens,
        )
        return chunked_loss, progress_pred

    @at.typecheck
    def compute_action_and_progress_chunk_current_relative_multilayer(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        progress_t: float = 0.001,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b ah"] | None]:
        observation, u_t, x_t, time = self._prepare_training_inputs(rng, observation, actions, train=train)
        prefix_out, prefix_mask, action_tokens, prefix_pooled_captures = self._encode_suffix_tokens_with_pooled_prefix_captures(
            observation,
            x_t,
            time,
            capture_layer_indices=_progress_current_relative.PREFIX_CAPTURE_LAYERS,
        )
        chunked_loss = self._compute_chunked_action_loss(action_tokens, u_t)
        if not self.enable_progress_head or self.progress_chunk_current_relative_multilayer_out_proj is None:
            return chunked_loss, None
        progress_action_tokens = self._encode_low_noise_action_tokens(observation, actions, progress_t=progress_t)
        if len(prefix_pooled_captures) != len(_progress_current_relative.PREFIX_CAPTURE_LAYERS):
            raise ValueError(
                f"Expected {len(_progress_current_relative.PREFIX_CAPTURE_LAYERS)} prefix captures, got {len(prefix_pooled_captures)}"
            )
        progress_pred = self.progress_chunk_current_relative_multilayer_out_proj.predict_progress(
            tuple(prefix_pooled_captures),
            progress_action_tokens,
        )
        return chunked_loss, progress_pred

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
