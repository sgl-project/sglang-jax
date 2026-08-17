"""Shared utilities for speculative decoding."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


@dataclass(frozen=True)
class SimulatedAcceptanceConfig:
    accept_len: float
    method: str
    token_mode: str
    enabled: bool


def _load_config() -> SimulatedAcceptanceConfig:
    raw_len = os.environ.get("SIMULATE_ACC_LEN")

    try:
        accept_len = float(raw_len) if raw_len is not None else -1.0
    except ValueError as exc:
        raise ValueError(f"SIMULATE_ACC_LEN must be a float, got {raw_len!r}.") from exc

    method = os.environ.get("SIMULATE_ACC_METHOD", "match-expected")
    token_mode = os.environ.get("SIMULATE_ACC_TOKEN_MODE", "fixed")
    enabled = accept_len > 0
    if enabled and method not in ("match-expected", "multinomial"):
        raise ValueError(
            f"Invalid SIMULATE_ACC_METHOD {method!r}; expected 'match-expected' or 'multinomial'."
        )
    if enabled and token_mode not in ("fixed", "real-draft-token"):
        raise ValueError(
            f"Invalid SIMULATE_ACC_TOKEN_MODE {token_mode!r}; expected "
            "'fixed' or 'real-draft-token'."
        )

    return SimulatedAcceptanceConfig(
        accept_len=accept_len,
        method=method,
        token_mode=token_mode,
        enabled=enabled,
    )


SIMULATED_ACCEPTANCE_CONFIG = _load_config()


def _sample_accept_len(rng: jax.Array, max_len: int) -> jax.Array:
    config = SIMULATED_ACCEPTANCE_CONFIG
    clamped_len = max(1.0, min(float(max_len), config.accept_len))

    if config.method == "multinomial":
        sampled = jnp.rint(
            jax.random.normal(rng, shape=(), dtype=jnp.float32) + clamped_len
        ).astype(jnp.int32)
        return jnp.clip(sampled, 1, max_len)

    lower = math.floor(clamped_len)
    upper = min(lower + 1, max_len)
    if lower == upper:
        return jnp.asarray(lower, dtype=jnp.int32)

    use_upper = jax.random.uniform(rng, shape=(), dtype=jnp.float32) < (clamped_len - lower)
    return jnp.where(use_upper, upper, lower).astype(jnp.int32)


def apply_simulated_acceptance(
    *,
    accept_index: jax.Array,
    predict: jax.Array,
    accept_lens: jax.Array,
    candidates: jax.Array,
    target_predict: jax.Array | None,
    valid_mask: jax.Array,
    spec_steps: int,
    topk: int,
    rng: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Replace real verify results with a fixed-shape simulated result."""
    config = SIMULATED_ACCEPTANCE_CONFIG
    if not config.enabled:
        return accept_index, predict, accept_lens

    if config.token_mode == "real-draft-token" and topk != 1:
        raise ValueError(
            "SIMULATE_ACC_TOKEN_MODE='real-draft-token' requires speculative_eagle_topk=1."
        )
    if rng is None:
        raise ValueError("An RNG key is required when simulated acceptance is enabled.")

    width = spec_steps + 1
    if accept_index.ndim != 2 or accept_index.shape[1] != width:
        raise ValueError(
            f"accept_index must have shape (batch_size, {width}), got {accept_index.shape}."
        )

    simulated_len = _sample_accept_len(rng, width)
    offsets = jnp.arange(width, dtype=jnp.int32)
    accepted_mask = offsets[None, :] < simulated_len
    valid_mask = valid_mask.astype(jnp.bool_)

    base = accept_index[:, :1]
    simulated_index = jnp.where(accepted_mask, base + offsets[None, :], -1)
    simulated_index = jnp.where(valid_mask[:, None], simulated_index, -1)
    simulated_lens = jnp.where(valid_mask, simulated_len, 0).astype(jnp.int32)

    if config.token_mode == "fixed":
        return simulated_index, jnp.full_like(predict, 32), simulated_lens

    if target_predict is None:
        raise ValueError(
            "target_predict is required when SIMULATE_ACC_TOKEN_MODE='real-draft-token'."
        )
    if candidates.shape[1] < width or target_predict.shape[1] < width:
        raise ValueError(
            f"candidates and target_predict must contain at least {width} tokens per request."
        )

    draft_columns = jnp.minimum(offsets + 1, candidates.shape[1] - 1)
    draft_values = jnp.take(candidates, draft_columns, axis=1)
    target_values = target_predict[:, :width]
    simulated_values = jnp.where(
        offsets[None, :] == simulated_len - 1,
        target_values,
        draft_values,
    ).astype(predict.dtype)

    write_mask = accepted_mask & valid_mask[:, None]
    scatter_index = jnp.where(write_mask, simulated_index, predict.size)
    predict = predict.at[scatter_index.reshape(-1)].set(
        simulated_values.reshape(-1),
        mode="drop",
        out_sharding=jax.typeof(predict).sharding,
    )
    return simulated_index, predict, simulated_lens


class GreedyChainVerifyOutput(NamedTuple):
    """Common result of topk=1 linear-chain greedy verification."""

    target_predict: jax.Array
    accepted_children: jax.Array
    accepted_draft_lens: jax.Array
    accept_lens: jax.Array
    next_verified_id: jax.Array


def greedy_chain_verify(
    draft_tokens: jax.Array,
    target_logits: jax.Array,
    *,
    draft_width: int,
    valid_mask: jax.Array | None = None,
) -> GreedyChainVerifyOutput:
    """Verify a topk=1 draft chain against target-model logits.

    ``draft_tokens`` contains the verified root in column zero. ``accept_lens``
    includes that root, while ``accepted_draft_lens`` counts only accepted
    children. Invalid padding rows produce zero lengths and next token.
    """
    target_predict = jnp.argmax(target_logits, axis=-1).astype(jnp.int32)
    draft_2d = draft_tokens.reshape((-1, int(draft_width))).astype(jnp.int32)
    target_predict_2d = target_predict.reshape(draft_2d.shape).astype(jnp.int32)

    predict_sharding = jax.typeof(target_predict).sharding
    mesh = predict_sharding.mesh if isinstance(predict_sharding, NamedSharding) else None
    if mesh is not None and mesh.empty:
        mesh = None
    if mesh is not None:
        data_2d = NamedSharding(mesh, P("data", None))
        draft_2d = jax.sharding.reshard(draft_2d, data_2d)
        target_predict_2d = jax.sharding.reshard(target_predict_2d, data_2d)

    child_matches = draft_2d[:, 1:] == target_predict_2d[:, :-1]
    accepted_children = jnp.cumprod(child_matches.astype(jnp.int32), axis=1).astype(jnp.bool_)
    if valid_mask is not None:
        accepted_children = jnp.where(
            valid_mask.reshape((-1, 1)),
            accepted_children,
            False,
        )

    accepted_draft_lens = jnp.sum(accepted_children.astype(jnp.int32), axis=1)
    accept_lens = accepted_draft_lens + 1
    if valid_mask is not None:
        accept_lens = jnp.where(valid_mask, accept_lens, 0)

    if mesh is None:
        next_verified_id = jnp.take_along_axis(
            target_predict_2d,
            accepted_draft_lens[:, None],
            axis=1,
        ).reshape(-1)
    else:

        def _select_local_next_id(local_predict, local_accept_len):
            return jnp.take_along_axis(
                local_predict,
                local_accept_len[:, None],
                axis=1,
            ).reshape(-1)

        next_verified_id = jax.shard_map(
            _select_local_next_id,
            mesh=mesh,
            in_specs=(P("data", None), P("data")),
            out_specs=P("data"),
        )(target_predict_2d, accepted_draft_lens)

    if valid_mask is not None:
        next_verified_id = jnp.where(valid_mask, next_verified_id, 0)

    return GreedyChainVerifyOutput(
        target_predict=target_predict_2d.reshape(-1),
        accepted_children=accepted_children,
        accepted_draft_lens=accepted_draft_lens.astype(jnp.int32),
        accept_lens=accept_lens.astype(jnp.int32),
        next_verified_id=next_verified_id.astype(jnp.int32),
    )
