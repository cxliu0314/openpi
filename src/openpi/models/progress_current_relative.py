import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at

ACTION_SUMMARY_TOKEN_DIM = 128
ACTION_SUMMARY_DIM = 128
CURRENT_FLAT_PREFIX_DIM = 192
CURRENT_MULTILAYER_PREFIX_DIM = 96
CURRENT_CONTEXT_DIM = 128
CURRENT_FUSE_HIDDEN_DIM = 256
MULTILAYER_WEIGHT_HIDDEN_DIM = 64
RELATIVE_CURRENT_EMBED_DIM = 16
RELATIVE_STEP_EMBED_DIM = 16
RELATIVE_FUSE_HIDDEN_DIM = 192
RELATIVE_FUSE_MID_DIM = 96
PREFIX_CAPTURE_LAYERS = (5, 11, 17)


def _sincos_step_embedding(num_steps: int, embedding_dim: int, dtype: jnp.dtype) -> at.Float[at.Array, "s e"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    if num_steps == 0:
        return jnp.zeros((0, embedding_dim), dtype=dtype)
    if num_steps == 1:
        positions = jnp.zeros((1,), dtype=jnp.float32)
    else:
        positions = jnp.linspace(0.0, 1.0, num_steps, dtype=jnp.float32)
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    sinusoid_input = positions[:, None] * (1.0 / period[None, :]) * 2.0 * jnp.pi
    emb = jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
    return emb.astype(dtype)


def _assemble_progress(current_pred: at.Array, relative_pred: at.Array) -> at.Array:
    if relative_pred.shape[-1] == 0:
        return current_pred[:, None]
    return jnp.concatenate([current_pred[:, None], current_pred[:, None] + relative_pred], axis=-1)


class ActionSummaryEncoder(nnx.Module):
    def __init__(self, action_input_dim: int, *, rngs: nnx.Rngs):
        self.action_input_dim = action_input_dim
        self.token_adapter = nnx.Linear(action_input_dim, ACTION_SUMMARY_TOKEN_DIM, rngs=rngs)
        self.summary_proj = nnx.Linear(ACTION_SUMMARY_TOKEN_DIM * 3, ACTION_SUMMARY_DIM, rngs=rngs)

    def encode(
        self, action_tokens: at.Float[at.Array, "b h d"]
    ) -> tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b h d"]]:
        adapted_tokens = nnx.swish(self.token_adapter(action_tokens))
        mean_token = jnp.mean(adapted_tokens, axis=1)
        first_token = adapted_tokens[:, 0, :]
        last_token = adapted_tokens[:, -1, :]
        summary_inputs = jnp.concatenate([mean_token, first_token, last_token], axis=-1)
        action_summary = nnx.swish(self.summary_proj(summary_inputs))
        return action_summary, adapted_tokens


class CurrentProgressHeadFlat(nnx.Module):
    def __init__(self, prefix_input_dim: int, *, rngs: nnx.Rngs):
        self.prefix_proj = nnx.Linear(prefix_input_dim, CURRENT_FLAT_PREFIX_DIM, rngs=rngs)
        self.fuse_fc1 = nnx.Linear(CURRENT_FLAT_PREFIX_DIM + ACTION_SUMMARY_DIM, CURRENT_FUSE_HIDDEN_DIM, rngs=rngs)
        self.fuse_fc2 = nnx.Linear(CURRENT_FUSE_HIDDEN_DIM, CURRENT_CONTEXT_DIM, rngs=rngs)
        self.out_proj = nnx.Linear(CURRENT_CONTEXT_DIM, 1, rngs=rngs)

    def predict(
        self, prefix_ctx: at.Float[at.Array, "b d"], action_summary: at.Float[at.Array, "b d"]
    ) -> tuple[at.Float[at.Array, "b"], at.Float[at.Array, "b d"]]:
        prefix_feature = nnx.swish(self.prefix_proj(prefix_ctx))
        fused = jnp.concatenate([prefix_feature, action_summary], axis=-1)
        hidden = nnx.swish(self.fuse_fc1(fused))
        current_ctx = nnx.swish(self.fuse_fc2(hidden))
        current_pred = jax.nn.sigmoid(self.out_proj(current_ctx)[..., 0])
        return current_pred, current_ctx


class CurrentProgressHeadMultilayer(nnx.Module):
    def __init__(self, prefix_input_dim: int, *, rngs: nnx.Rngs):
        self.prefix_proj_0 = nnx.Linear(prefix_input_dim, CURRENT_MULTILAYER_PREFIX_DIM, rngs=rngs)
        self.prefix_proj_1 = nnx.Linear(prefix_input_dim, CURRENT_MULTILAYER_PREFIX_DIM, rngs=rngs)
        self.prefix_proj_2 = nnx.Linear(prefix_input_dim, CURRENT_MULTILAYER_PREFIX_DIM, rngs=rngs)
        self.layer_weight_fc1 = nnx.Linear(ACTION_SUMMARY_DIM, MULTILAYER_WEIGHT_HIDDEN_DIM, rngs=rngs)
        self.layer_weight_fc2 = nnx.Linear(MULTILAYER_WEIGHT_HIDDEN_DIM, 3, rngs=rngs)
        self.fuse_fc1 = nnx.Linear(CURRENT_MULTILAYER_PREFIX_DIM + ACTION_SUMMARY_DIM, CURRENT_CONTEXT_DIM, rngs=rngs)
        self.out_proj = nnx.Linear(CURRENT_CONTEXT_DIM, 1, rngs=rngs)

    def predict(
        self,
        prefix_layer_ctxs: tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b d"], at.Float[at.Array, "b d"]],
        action_summary: at.Float[at.Array, "b d"],
    ) -> tuple[at.Float[at.Array, "b"], at.Float[at.Array, "b d"]]:
        prefix_feats = [
            nnx.swish(self.prefix_proj_0(prefix_layer_ctxs[0])),
            nnx.swish(self.prefix_proj_1(prefix_layer_ctxs[1])),
            nnx.swish(self.prefix_proj_2(prefix_layer_ctxs[2])),
        ]
        layer_logits = self.layer_weight_fc2(nnx.swish(self.layer_weight_fc1(action_summary)))
        layer_weights = jax.nn.softmax(layer_logits, axis=-1)
        fused_prefix = sum(prefix_feats[i] * layer_weights[:, i : i + 1] for i in range(3))
        fused = jnp.concatenate([fused_prefix, action_summary], axis=-1)
        current_ctx = nnx.swish(self.fuse_fc1(fused))
        current_pred = jax.nn.sigmoid(self.out_proj(current_ctx)[..., 0])
        return current_pred, current_ctx


class RelativeDeltaHead(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        input_dim = ACTION_SUMMARY_TOKEN_DIM + CURRENT_CONTEXT_DIM + RELATIVE_CURRENT_EMBED_DIM + RELATIVE_STEP_EMBED_DIM
        self.current_pred_proj = nnx.Linear(1, RELATIVE_CURRENT_EMBED_DIM, rngs=rngs)
        self.fuse_fc1 = nnx.Linear(input_dim, RELATIVE_FUSE_HIDDEN_DIM, rngs=rngs)
        self.fuse_fc2 = nnx.Linear(RELATIVE_FUSE_HIDDEN_DIM, RELATIVE_FUSE_MID_DIM, rngs=rngs)
        self.out_proj = nnx.Linear(RELATIVE_FUSE_MID_DIM, 1, rngs=rngs)

    def predict(
        self,
        future_tokens: at.Float[at.Array, "b h d"],
        current_ctx: at.Float[at.Array, "b d"],
        current_pred: at.Float[at.Array, "b"],
    ) -> at.Float[at.Array, "b h"]:
        if future_tokens.shape[1] == 0:
            return jnp.zeros((future_tokens.shape[0], 0), dtype=future_tokens.dtype)
        step_emb = _sincos_step_embedding(future_tokens.shape[1], RELATIVE_STEP_EMBED_DIM, future_tokens.dtype)
        current_ctx_broadcast = jnp.broadcast_to(current_ctx[:, None, :], (future_tokens.shape[0], future_tokens.shape[1], current_ctx.shape[-1]))
        current_emb = nnx.swish(self.current_pred_proj(current_pred[:, None]))[:, None, :]
        current_emb = jnp.broadcast_to(current_emb, (future_tokens.shape[0], future_tokens.shape[1], current_emb.shape[-1]))
        step_emb = jnp.broadcast_to(step_emb[None, :, :], (future_tokens.shape[0], future_tokens.shape[1], step_emb.shape[-1]))
        fused = jnp.concatenate([future_tokens, current_ctx_broadcast, current_emb, step_emb], axis=-1)
        hidden = nnx.swish(self.fuse_fc1(fused))
        hidden = nnx.swish(self.fuse_fc2(hidden))
        relative_pred = jax.nn.sigmoid(self.out_proj(hidden)[..., 0])
        return relative_pred * (1.0 - current_pred)[:, None]


class CurrentRelativeProgressHeadFlat(nnx.Module):
    def __init__(self, prefix_input_dim: int, action_input_dim: int, *, rngs: nnx.Rngs):
        self.action_summary_encoder = ActionSummaryEncoder(action_input_dim, rngs=rngs)
        self.current_head = CurrentProgressHeadFlat(prefix_input_dim, rngs=rngs)
        self.relative_head = RelativeDeltaHead(rngs=rngs)

    def predict_outputs(
        self, prefix_ctx: at.Float[at.Array, "b d"], action_tokens: at.Float[at.Array, "b h d"]
    ) -> tuple[at.Float[at.Array, "b"], at.Float[at.Array, "b h"]]:
        action_summary, adapted_tokens = self.action_summary_encoder.encode(action_tokens)
        current_pred, current_ctx = self.current_head.predict(prefix_ctx, action_summary)
        relative_pred = self.relative_head.predict(adapted_tokens[:, 1:, :], current_ctx, current_pred)
        return current_pred, relative_pred

    def predict_progress(
        self, prefix_ctx: at.Float[at.Array, "b d"], action_tokens: at.Float[at.Array, "b h d"]
    ) -> at.Float[at.Array, "b h"]:
        current_pred, relative_pred = self.predict_outputs(prefix_ctx, action_tokens)
        return _assemble_progress(current_pred, relative_pred)


class CurrentRelativeProgressHeadMultilayer(nnx.Module):
    def __init__(self, prefix_input_dim: int, action_input_dim: int, *, rngs: nnx.Rngs):
        self.action_summary_encoder = ActionSummaryEncoder(action_input_dim, rngs=rngs)
        self.current_head = CurrentProgressHeadMultilayer(prefix_input_dim, rngs=rngs)
        self.relative_head = RelativeDeltaHead(rngs=rngs)

    def predict_outputs(
        self,
        prefix_layer_ctxs: tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b d"], at.Float[at.Array, "b d"]],
        action_tokens: at.Float[at.Array, "b h d"],
    ) -> tuple[at.Float[at.Array, "b"], at.Float[at.Array, "b h"]]:
        action_summary, adapted_tokens = self.action_summary_encoder.encode(action_tokens)
        current_pred, current_ctx = self.current_head.predict(prefix_layer_ctxs, action_summary)
        relative_pred = self.relative_head.predict(adapted_tokens[:, 1:, :], current_ctx, current_pred)
        return current_pred, relative_pred

    def predict_progress(
        self,
        prefix_layer_ctxs: tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b d"], at.Float[at.Array, "b d"]],
        action_tokens: at.Float[at.Array, "b h d"],
    ) -> at.Float[at.Array, "b h"]:
        current_pred, relative_pred = self.predict_outputs(prefix_layer_ctxs, action_tokens)
        return _assemble_progress(current_pred, relative_pred)
