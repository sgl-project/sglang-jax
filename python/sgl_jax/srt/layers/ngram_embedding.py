"""Qwen4Exp N-gram embedding (``ple``). Single layer, ``ple_layer_ids: [2]`` (1-based).

Released checkpoint: HS=2560, HC=4, ngram_size=3, heads_per_ngram=8 ->
HEADS=16, ple_embed_dim=2560, dim per head=160, C=HC*HS=10240,
conv kernel=4, dilation=ngram_size=3, conv state len=(4-1)*3=9.

    host   input_ids [T]      -> ids [T, 16]       -> gather -> E [T, 2560]
    device E [T, 2560]        -> K [T, 10240], V [T, 2560]
           g = Gate(Norm(K), Norm(R))              -> [T, HC]
           U = g * V                               -> [T, 10240]
           delta = U + SiLU(DWConv(Norm(U)))       -> [T, 10240]

The hash runs on the host in numpy because it needs full 64-bit products.
See ngram_table.py.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.ngram_conv import ngram_conv_prefill, ngram_conv_update
from sgl_jax.srt.layers.hyperconnection import GroupedGemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.utils.profiling_utils import named_scope

# --- hash addressing (host, int64) -----------------------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    """Deterministic Miller-Rabin, standard 64-bit witness set."""
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent, shifts = value - 1, 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


@dataclass(frozen=True)
class NGramHashParams:
    """Hash addressing for one PLE layer. NGramTable checks it against the
    copies shipped in the checkpoint."""

    multipliers: np.ndarray  # [ngram_size]  int64
    sizes: np.ndarray  # [HEADS]       int64, distinct primes > ngram_vocab_size_base
    offsets: np.ndarray  # [HEADS]       int64, exclusive prefix sum of sizes
    total_vocab_size: int
    heads_per_ngram: int
    ngram_size: int
    eos_token_id: int

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_context_len(self) -> int:
        return self.ngram_size - 1


def build_hash_params(
    *,
    ngram_size: int,
    heads_per_ngram: int,
    vocab_size: int,
    ngram_vocab_size_base: int,
    eos_token_id: int,
    ple_dense_layer_id: int = 0,
    seed: int = 1234,
) -> NGramHashParams:
    """``ple_dense_layer_id`` indexes into ``ple_layer_ids`` (0 here), not layers."""
    if ngram_size < 2:
        raise ValueError(f"ngram_size must be >= 2, got {ngram_size}")
    if heads_per_ngram <= 0:
        raise ValueError(f"heads_per_ngram must be positive, got {heads_per_ngram}")

    max_multiplier = ((1 << 63) - 1) // vocab_size
    half_bound = max(1, max_multiplier // 2)
    base_seed = seed + _PLE_LAYER_PRIME * ple_dense_layer_id
    multipliers = [  # odd, so the token's low bit survives into the hash
        2 * (_splitmix64(base_seed + _SPLITMIX_GAMMA * (i + 1)) % half_bound) + 1
        for i in range(ngram_size)
    ]

    ngram_heads = (ngram_size - 1) * heads_per_ngram
    sizes: list[int] = []
    offsets: list[int] = []
    offset = 0
    for local_head in range(ngram_heads):
        global_head = ple_dense_layer_id * ngram_heads + local_head
        size = _nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
        sizes.append(size)
        offsets.append(offset)
        offset += size

    return NGramHashParams(
        multipliers=np.array(multipliers, dtype=np.int64),
        sizes=np.array(sizes, dtype=np.int64),
        offsets=np.array(offsets, dtype=np.int64),
        total_vocab_size=offset,
        heads_per_ngram=heads_per_ngram,
        ngram_size=ngram_size,
        eos_token_id=eos_token_id,
    )


def ngram_context_row(
    fill_ids,  # request's full token stream, prompt + output
    chunk_start: int,  # extend: extend_prefix_lens[b]; decode: seq_lens[b] - 1
    ctx_len: int,  # ngram_size - 1
    eos_token_id: int,
) -> np.ndarray:  # [ctx_len] int32
    """The ctx_len tokens before chunk_start, oldest first, EOS-padded."""
    if ctx_len <= 0:
        return np.zeros(0, np.int32)
    if chunk_start < 0:
        raise ValueError(f"chunk_start must be non-negative, got {chunk_start}")
    lo = chunk_start - ctx_len
    have = np.asarray(fill_ids[max(lo, 0) : chunk_start], np.int32)  # [min(ctx_len, chunk_start)]
    if lo >= 0:
        return have
    return np.concatenate([np.full(-lo, eos_token_id, np.int32), have])  # [ctx_len]


def ngram_context_row_split(
    prompt_ids,  # req.origin_input_ids
    output_ids,  # req.output_ids
    chunk_start: int,  # extend: prefix_lens[b]; decode: seq_lens[b] - 1
    ctx_len: int,  # ngram_size - 1
    eos_token_id: int,
) -> np.ndarray:  # [ctx_len] int32
    """``ngram_context_row`` over ``prompt_ids + output_ids`` without concatenating
    them; it runs per request per decode step."""
    if ctx_len <= 0:
        return np.zeros(0, np.int32)
    n_prompt = len(prompt_ids)
    total = n_prompt + len(output_ids)
    if chunk_start > total:
        raise ValueError(
            f"chunk_start {chunk_start} is past the request's {total} known tokens; "
            "the token stream is stale (overlap scheduling defers output_ids)"
        )
    lo = max(chunk_start - ctx_len, 0)
    if chunk_start <= n_prompt:
        window = prompt_ids[lo:chunk_start]
    elif lo >= n_prompt:
        window = output_ids[lo - n_prompt : chunk_start - n_prompt]
    else:
        window = list(prompt_ids[lo:]) + list(output_ids[: chunk_start - n_prompt])
    return ngram_context_row(window, len(window), ctx_len, eos_token_id)


def compute_ngram_ids(
    input_ids: np.ndarray,  # [T]       packed across requests
    cu_seqlens: np.ndarray,  # [B+1]
    context: np.ndarray,  # [B, ctx_len]  tokens before each chunk, oldest first
    params: NGramHashParams,
) -> np.ndarray:  # [T, HEADS] int32
    """Hash each token's n-grams into table row ids.

    EOS is a barrier: positions older than an EOS read as EOS, so an n-gram
    never spans two documents. Ids stay under 2^31, so int32 is enough. The
    reduce runs in uint64; both operands are non-negative because
    ``build_hash_params`` keeps token * multiplier under 2^63.
    """
    input_ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)  # [T]
    cu_seqlens = np.asarray(cu_seqlens, dtype=np.int64)  # [B+1]
    context = np.asarray(context, dtype=np.int64)  # [B, ctx_len]
    ctx_len = params.ngram_context_len
    num_tokens = input_ids.shape[0]  # T
    num_reqs = cu_seqlens.shape[0] - 1  # B
    per_order = params.heads_per_ngram  # HEADS / ctx_len
    if context.shape != (num_reqs, ctx_len):
        raise ValueError(f"context must be [{num_reqs}, {ctx_len}], got {context.shape}")

    sizes = params.sizes.reshape(ctx_len, per_order).astype(np.uint64)  # [ctx_len, hpn]
    offsets = params.offsets.reshape(ctx_len, per_order).astype(np.uint64)

    # Decode: every token is at chunk position 0, so each lookback is a column of `context`.
    decode = num_tokens == num_reqs and bool((np.diff(cu_seqlens) == 1).all())
    if not decode:
        t_idx = np.arange(num_tokens, dtype=np.int64)  # [T]
        req = np.clip(np.searchsorted(cu_seqlens, t_idx, side="right") - 1, 0, num_reqs - 1)
        chunk_pos = t_idx - cu_seqlens[req]  # [T]  position within the request's chunk

    rolling = input_ids * params.multipliers[0]  # [T]  prefix-XOR accumulator
    rolling_u = rolling.view(np.uint64)  # [T]  same buffer, unsigned reduce
    ids = np.empty((num_tokens, ctx_len, per_order), dtype=np.int32)
    residues = np.empty((num_tokens, per_order), dtype=np.uint64)
    crossed = np.zeros(num_tokens, dtype=bool)  # [T]
    for shift in range(1, ctx_len + 1):
        if decode:
            token = context[:, ctx_len - shift].copy()  # [T]  copy: written below
        else:
            step_token = input_ids[np.clip(t_idx - shift, 0, num_tokens - 1)]  # [T]
            ctx_col = np.clip(ctx_len - shift + chunk_pos, 0, ctx_len - 1)  # [T]
            token = np.where(chunk_pos >= shift, step_token, context[req, ctx_col])  # [T]
        np.copyto(token, params.eos_token_id, where=crossed)
        crossed |= token == params.eos_token_id  # [T]
        rolling ^= token * params.multipliers[shift]  # [T]
        # rolling is now the order-(shift+1) hash, reduced by heads [shift-1].
        np.mod(rolling_u[:, None], sizes[shift - 1][None, :], out=residues)  # [T, hpn]
        residues += offsets[shift - 1][None, :]
        ids[:, shift - 1, :] = residues
    return ids.reshape(num_tokens, ctx_len * per_order)  # [T, HEADS]


# --- device layer ----------------------------------------------------------


class NGramEmbedding(nnx.Module):
    """Gate and dilated short conv; returns the PLE delta. The host lookup result
    comes in as ``ple_embeddings``.

    Checkpoint weights::

        ple.key_proj.weight     [HC*HS, ple_embed_dim]   10240 x 2560
        ple.value_proj.weight   [HS, ple_embed_dim]       2560 x 2560
        ple.norm_key.weight     [HC*HS]                  grouped by HS
        ple.norm_query.weight   [HC*HS]
        ple.norm_conv.weight    [HC*HS]
        ple.conv1d.weight       [HC*HS, 1, kernel]       depthwise

    Conv state is [slots, HC*HS, (kernel-1)*ngram_size], a pool entry separate
    from GDN's conv state on the same layer.
    """

    def __init__(
        self,
        config,
        mesh: jax.sharding.Mesh,
        params_dtype: jnp.dtype = jnp.bfloat16,
        scope_name: str = "ple",
    ):
        hidden_size = int(config.hidden_size)
        hc_count = int(config.hc_count)
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hyper_hidden_size = hidden_size * hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.dilation
        self.params_dtype = params_dtype
        self.norm_eps = float(config.rms_norm_eps)
        self.mesh = mesh
        self.name = scope_name

        # Replicated: sharding the projections would save a few hundred MB but add an all-reduce.
        self.key_proj = LinearBase(
            input_size=int(config.ple_embed_dim),
            output_size=self.hyper_hidden_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=params_dtype,
            kernel_axes=(None, None),
            scope_name="key_proj",
        )
        self.value_proj = LinearBase(
            input_size=int(config.ple_embed_dim),
            output_size=hidden_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=params_dtype,
            kernel_axes=(None, None),
            scope_name="value_proj",
        )

        eps = float(config.rms_norm_eps)
        self.norm_key = GroupedGemmaRMSNorm(
            self.hyper_hidden_size, epsilon=eps, group_size=hidden_size
        )
        self.norm_query = GroupedGemmaRMSNorm(
            self.hyper_hidden_size, epsilon=eps, group_size=hidden_size
        )
        self.norm_conv = GroupedGemmaRMSNorm(
            self.hyper_hidden_size, epsilon=eps, group_size=hidden_size
        )
        # Channel-sharded like the pool's conv state; depthwise, so no collective.
        self.conv1d_weight = nnx.Param(
            jnp.zeros(
                (self.hyper_hidden_size, self.conv_kernel_size),
                dtype=params_dtype,
                out_sharding=P("tensor", None),
            )
        )

    @named_scope
    def gate(self, hyper_input: jax.Array, ple_embeddings: jax.Array) -> jax.Array:
        """-> U [T, HC*HS]: one scalar gate per (token, stream) times a value
        vector shared by all streams."""
        if hyper_input.shape[-1] != self.hyper_hidden_size:
            raise ValueError(
                f"hyper_input last dim must be {self.hyper_hidden_size}, "
                f"got {hyper_input.shape[-1]}"
            )
        key, _ = self.key_proj(ple_embeddings)  # [T, HC*HS]
        value, _ = self.value_proj(ple_embeddings)  # [T, HS]

        def _streams(x):
            return x.reshape(*x.shape[:-1], self.hc_count, self.hidden_size).astype(jnp.float32)

        k_n = _streams(self.norm_key(key))  # [T, HC, HS] f32
        q_n = _streams(self.norm_query(hyper_input))  # [T, HC, HS] f32
        dot = jnp.sum(k_n * q_n, axis=-1) / math.sqrt(self.hidden_size)  # [T, HC]
        # The sqrt and its 1e-6 floor are the trained gate, not a guard.
        gate = jax.nn.sigmoid(jnp.sign(dot) * jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6)))  # [T, HC]
        gated = gate[..., None] * value[..., None, :].astype(jnp.float32)  # [T, HC, HS]
        return gated.reshape(*hyper_input.shape).astype(hyper_input.dtype)  # [T, HC*HS]

    def _conv_weight(self, dtype: jnp.dtype) -> jax.Array:
        return jnp.asarray(self.conv1d_weight, dtype)

    def _to_conv_layout(self, x: jax.Array) -> jax.Array:
        """Replicated -> channel-sharded, matching the pool's conv state."""
        return jax.sharding.reshard(x, jax.sharding.NamedSharding(self.mesh, P("data", "tensor")))

    def _shard_mapped(self, local_fn, extra_in_specs=()):
        """Per-shard conv, as gdn_backend does: XLA cannot shard the slot-indexed
        conv_state once tokens are on `data`."""
        return jax.shard_map(
            local_fn,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor"),  # activations
                P("data", "tensor", None),  # conv_state
                P("tensor", None),  # conv weight
                P("data"),  # state_indices
                P("data"),  # has_initial_state
                *extra_in_specs,
            ),
            out_specs=(P("data", "tensor"), P("data", "tensor", None)),
            check_vma=False,
        )

    @named_scope
    def forward_extend(
        self,
        hyper_input: jax.Array,  # [T, HC*HS]
        ple_embeddings: jax.Array,  # [T, ple_embed_dim]
        conv_state: jax.Array,  # [num_slots, HC*HS, (kernel-1)*dilation]
        state_indices: jax.Array,  # [B]
        cu_seqlens: jax.Array,  # [B+1]
        has_initial_state: jax.Array,  # [B] bool
        track_indices: jax.Array | None = None,  # [B] req → track slot (None = OFF)
        track_mask: jax.Array | None = None,  # [B] bool boundary mask
        *,
        use_pallas: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """PLE delta and state; use_pallas is a static, opt-in JIT argument."""
        if use_pallas:
            return self._fused_forward(
                hyper_input,
                ple_embeddings,
                conv_state,
                state_indices,
                has_initial_state,
                cu_seqlens,
                track_indices=track_indices,
                track_mask=track_mask,
            )
        gated = self.gate(hyper_input, ple_embeddings)

        def _local(
            x_l, state_l, weight_l, indices_l, init_l, cu_l, track_indices_l=None, track_mask_l=None
        ):
            y, new_state = ngram_conv_prefill(
                x=x_l.T,  # [T, C] -> [C, T], the kernels are channel-first
                weight=weight_l,
                cu_seqlens=cu_l,
                conv_state=state_l,
                state_indices=indices_l,
                has_initial_state=init_l,
                dilation=self.dilation,
                track_indices=track_indices_l,
                track_mask=track_mask_l,
            )
            return y.T, new_state  # [T, C], [num_slots, C, S]

        extra_in_specs = [P("data")]  # cu_seqlens
        args = [
            self._to_conv_layout(self.norm_conv(gated)),
            conv_state,
            self._conv_weight(gated.dtype),
            state_indices,
            has_initial_state,
            cu_seqlens,
        ]
        if track_indices is not None:
            extra_in_specs += [P("data"), P("data")]  # track_indices, track_mask
            args += [track_indices, track_mask]
        conv_out, new_conv_state = self._shard_mapped(_local, extra_in_specs)(*args)
        return gated + conv_out, new_conv_state  # [T, HC*HS]

    @named_scope
    def forward_decode(
        self,
        hyper_input: jax.Array,  # [B, HC*HS]
        ple_embeddings: jax.Array,  # [B, ple_embed_dim]
        conv_state: jax.Array,  # [num_slots, HC*HS, (kernel-1)*dilation]
        state_indices: jax.Array,  # [B]
        has_initial_state: jax.Array,  # [B] bool
        track_indices: jax.Array | None = None,  # [B] req → track slot (None = OFF)
        track_mask: jax.Array | None = None,  # [B] bool boundary mask
        *,
        use_pallas: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """PLE delta and state; use_pallas is a static, opt-in JIT argument."""
        if use_pallas:
            return self._fused_forward(
                hyper_input,
                ple_embeddings,
                conv_state,
                state_indices,
                has_initial_state,
                track_indices=track_indices,
                track_mask=track_mask,
            )
        gated = self.gate(hyper_input, ple_embeddings)

        def _local(
            x_l, state_l, weight_l, indices_l, init_l, track_indices_l=None, track_mask_l=None
        ):
            return ngram_conv_update(
                x=x_l,
                conv_state=state_l,
                state_indices=indices_l,
                weight=weight_l,
                has_initial_state=init_l,
                dilation=self.dilation,
                track_indices=track_indices_l,
                track_mask=track_mask_l,
            )

        extra_in_specs = []
        args = [
            self._to_conv_layout(self.norm_conv(gated)),
            conv_state,
            self._conv_weight(gated.dtype),
            state_indices,
            has_initial_state,
        ]
        if track_indices is not None:
            extra_in_specs += [P("data"), P("data")]  # track_indices, track_mask
            args += [track_indices, track_mask]
        conv_out, new_conv_state = self._shard_mapped(_local, extra_in_specs)(*args)
        return gated + conv_out, new_conv_state  # [B, HC*HS]

    def _fused_forward(
        self,
        hyper_input,
        ple_embeddings,
        conv_state,
        state_indices,
        has_init,
        cu=None,
        track_indices=None,
        track_mask=None,
    ):
        from sgl_jax.srt.kernels.ngram_fused import (
            ngram_decode_pallas,
            ngram_extend_pallas,
        )

        if self.hc_count % self.mesh.shape["tensor"]:
            raise ValueError("Fused PLE tensor shards must own complete RMSNorm groups")
        if self.mesh.shape["data"] != 1:
            raise NotImplementedError("Opt-in Pallas PLE currently supports one data shard")
        if hyper_input.shape[-1] != self.hyper_hidden_size:
            raise ValueError(f"hyper_input last dim must be {self.hyper_hidden_size}")
        # Keep the key in FP32 across the custom call; rounding it early can flip tiny gate dots.
        key = jax.lax.dot_general(
            ple_embeddings,
            jnp.asarray(self.key_proj.weight),
            (((ple_embeddings.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            out_sharding=jax.sharding.NamedSharding(self.mesh, P("data", None)),
        )
        value, _ = self.value_proj(ple_embeddings)
        local_fn = functools.partial(
            ngram_decode_pallas if cu is None else ngram_extend_pallas,
            hidden_size=self.hidden_size,
            dilation=self.dilation,
            eps=self.norm_eps,
        )

        def norm_weight(param):
            return jax.sharding.reshard(
                jnp.asarray(param), jax.sharding.NamedSharding(self.mesh, P("tensor"))
            )

        in_specs = [
            P("data", "tensor"),
            P("data", "tensor"),
            P("data", None),
            P("tensor"),
            P("tensor"),
            P("tensor"),
            P("tensor", None),
            P("data", "tensor", None),
            P("data"),
            P("data"),
        ]
        args = [
            self._to_conv_layout(key),
            self._to_conv_layout(hyper_input),
            value,
            norm_weight(self.norm_key.weight),
            norm_weight(self.norm_query.weight),
            norm_weight(self.norm_conv.weight),
            self._conv_weight(hyper_input.dtype),
            conv_state,
            state_indices,
            has_init,
        ]
        if cu is not None:
            in_specs += [P("data")]  # cu_seqlens
            args += [cu]
        if track_indices is not None:
            in_specs += [P("data"), P("data")]  # track_indices, track_mask
            args += [track_indices, track_mask]

        return jax.shard_map(
            local_fn,
            mesh=self.mesh,
            in_specs=tuple(in_specs),
            out_specs=(P("data", "tensor"), P("data", "tensor", None)),
            check_vma=False,
        )(*args)


__all__ = [
    "NGramEmbedding",
    "NGramHashParams",
    "build_hash_params",
    "compute_ngram_ids",
    "ngram_context_row",
    "ngram_context_row_split",
]
