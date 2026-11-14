import copy
import logging
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import (
    Embed,
    ParallelLMHead,
    RotaryEmbedding,
    apply_rotary_emb,
    get_rope,
)
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase, MergedColumnParallelLinear
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.profiling_utils import named_scope

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


# Token-parallel prefill: EXTEND/MIXED batches with at least this many (padded)
# tokens run every dense projection token-parallel on replicated weights and
# keep only the attention core head-parallel (two all-to-alls per layer instead
# of two [T, hidden] all-reduces). Smaller batches (decode, short extends) keep
# the tensor-parallel program, whose weight traffic is 1/tp per device: below
# ~T/tp = 256 local rows the replicated-weight dots turn HBM-bound.
TOKEN_PARALLEL_MIN_TOKENS = 2048


class TokenParallelWeight(nnx.Variable):
    """Fully replicated copy of a tensor-parallel weight.

    Only read by the token-parallel prefill program; the tensor-parallel
    (decode / short extend) program keeps using the sharded originals.
    """


@partial(jax.jit, static_argnames=("mesh",))
def _replicate_concat(*xs: jax.Array, mesh: jax.sharding.Mesh) -> jax.Array:
    replicated = NamedSharding(mesh, P())
    parts = [jax.sharding.reshard(x, replicated) for x in xs]
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)


def _replicated(*xs: jax.Array, mesh: jax.sharding.Mesh) -> jax.Array:
    """Fully replicated copy (on-device all-gather); several inputs are concatenated
    along the last axis inside the same program."""
    return _replicate_concat(*xs, mesh=mesh)


def _token_sharding(mesh: jax.sharding.Mesh, ndim: int) -> NamedSharding:
    return NamedSharding(mesh, P("tensor", *([None] * (ndim - 1))))


def _qkv_tokens_to_heads(
    q: jax.Array, k: jax.Array, v: jax.Array, mesh: jax.sharding.Mesh
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One all-to-all: token-sharded [T, H, D] q/k/v -> head-sharded [T, H, D].

    Each device packs, per destination device, its q heads followed by its k and
    v heads, so a single all-to-all delivers every device the full token range
    for exactly the heads it owns in the tensor-parallel layout (q heads
    [i*Hq/tp, (i+1)*Hq/tp), kv heads [i*Hkv/tp, (i+1)*Hkv/tp)).
    """
    tp = mesh.shape["tensor"]
    hq, hk, hv = q.shape[1], k.shape[1], v.shape[1]
    gq, gk, gv = hq // tp, hk // tp, hv // tp
    head_dim = q.shape[2]

    def _a2a(ql, kl, vl):
        tl = ql.shape[0]
        # Destination-major packing [tp, T/tp, gq+gk+gv, D] so the all-to-all
        # splits and concatenates along the major dimension (contiguous
        # per-destination chunks, no layout transposes around the collective).
        packed = jnp.concatenate(
            [
                ql.reshape(tl, tp, gq, head_dim),
                kl.reshape(tl, tp, gk, head_dim),
                vl.reshape(tl, tp, gv, head_dim),
            ],
            axis=2,
        )
        packed = jnp.swapaxes(packed, 0, 1)
        out = jax.lax.all_to_all(packed, "tensor", split_axis=0, concat_axis=0, tiled=True)
        # [tp (source = token block), T/tp, g, D] -> [T, g, D] in global token order.
        out = out.reshape(tl * tp, gq + gk + gv, head_dim)
        return out[:, :gq], out[:, gq : gq + gk], out[:, gq + gk :]

    tok = P("tensor", None, None)
    head = P("data", "tensor", None)
    tok_sharding = NamedSharding(mesh, tok)
    q, k, v = (jax.sharding.reshard(x, tok_sharding) for x in (q, k, v))
    return jax.shard_map(
        _a2a,
        mesh=mesh,
        in_specs=(tok, tok, tok),
        out_specs=(head, head, head),
        check_vma=False,
    )(q, k, v)


def _heads_to_tokens(x: jax.Array, mesh: jax.sharding.Mesh) -> jax.Array:
    """One all-to-all: head-sharded [T, H*D] -> token-sharded [T, H*D]."""
    tp = mesh.shape["tensor"]

    def _a2a(xl):
        num_tokens, local_cols = xl.shape
        tl = num_tokens // tp
        # Token blocks are already major: split/concat along dim 0, then
        # interleave the per-source head blocks locally.
        out = jax.lax.all_to_all(
            xl.reshape(tp, tl, local_cols), "tensor", split_axis=0, concat_axis=0, tiled=True
        )
        return jnp.swapaxes(out, 0, 1).reshape(tl, tp * local_cols)

    x = jax.sharding.reshard(x, NamedSharding(mesh, P("data", "tensor")))
    return jax.shard_map(
        _a2a,
        mesh=mesh,
        in_specs=P("data", "tensor"),
        out_specs=P("tensor", None),
        check_vma=False,
    )(x)


def _select_token_rows(x: jax.Array, indices: jax.Array, mesh: jax.sharding.Mesh) -> jax.Array:
    """Exact rows ``x[indices]`` of a token-sharded [T, H] array, replicated.

    Each device contributes the rows it owns and zeros elsewhere, so the psum
    over the tensor axis adds exactly one non-zero term per element.
    """

    def _select(xl, idx):
        tl = xl.shape[0]
        local = idx - jax.lax.axis_index("tensor") * tl
        owned = (local >= 0) & (local < tl)
        rows = jnp.take(xl, jnp.clip(local, 0, tl - 1), axis=0)
        rows = jnp.where(owned[:, None], rows, jnp.zeros((), rows.dtype))
        return jax.lax.psum(rows, "tensor")

    x = jax.sharding.reshard(x, NamedSharding(mesh, P("tensor", None)))
    indices = jax.sharding.reshard(indices, NamedSharding(mesh, P("data")))
    return jax.shard_map(
        _select,
        mesh=mesh,
        in_specs=(P("tensor", None), P("data")),
        out_specs=P("data", None),
        check_vma=False,
    )(x, indices)


class QWen3Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = None,
        layer_id: int = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        assert num_heads % num_kv_heads == 0
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.mesh = mesh

        self.q_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm"
        )
        self.k_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm"
        )

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    @named_scope
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
        cos_sin: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k, cos_sin=cos_sin)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        output, _ = self.o_proj(attn_output, out_sharding=out_sharding)
        return output, kv_fused

    @named_scope
    def forward_fused_rmsnorm_qkv(
        self,
        hidden_states: jax.Array,
        input_layernorm: RMSNorm,
        positions: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
        cos_sin: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Input RMSNorm followed by the q/k/v projections as one direct dot.

        When q/k/v_proj are plain ``LinearBase`` layers, every token count and
        layer uses one ``MergedColumnParallelLinear.shard_local_merged_column_dot``,
        which concatenates and splits each device's own q/k/v columns, so no
        weight data moves between devices. Other projections (quantized, LoRA)
        go through their own ``__call__``.
        """
        normed_states = input_layernorm(hidden_states)
        if (
            type(self.q_proj) is LinearBase
            and type(self.k_proj) is LinearBase
            and type(self.v_proj) is LinearBase
        ):
            w_q = getattr(self.q_proj.weight, "value", self.q_proj.weight)
            w_k = getattr(self.k_proj.weight, "value", self.k_proj.weight)
            w_v = getattr(self.v_proj.weight, "value", self.v_proj.weight)
            q, k, v = MergedColumnParallelLinear.shard_local_merged_column_dot(
                normed_states,
                (w_q, w_k, w_v),
                (w_q.shape[-1], w_k.shape[-1], w_v.shape[-1]),
                self.mesh,
                preferred_element_type=jnp.float32,
                out_dtype=normed_states.dtype,
            )
            # The merged dot does not go through LinearBase.__call__, so add
            # the attention_bias here.
            if getattr(self.q_proj, "bias", None) is not None:
                q = q + self.q_proj.bias.value
            if getattr(self.k_proj, "bias", None) is not None:
                k = k + self.k_proj.bias.value
            if getattr(self.v_proj, "bias", None) is not None:
                v = v + self.v_proj.bias.value
        else:
            q, _ = self.q_proj(normed_states)
            k, _ = self.k_proj(normed_states)
            v, _ = self.v_proj(normed_states)

        head_sharding = NamedSharding(self.mesh, P("data", "tensor", None))
        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=head_sharding,
        )
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=head_sharding,
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=head_sharding,
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k, cos_sin=cos_sin)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        output, _ = self.o_proj(attn_output, out_sharding=out_sharding)
        return output, kv_fused

    def supports_token_parallel(self) -> bool:
        tp = self.mesh.shape.get("tensor", 1)
        return (
            type(self.q_proj) is LinearBase
            and type(self.k_proj) is LinearBase
            and type(self.v_proj) is LinearBase
            and type(self.o_proj) is LinearBase
            and all(
                getattr(proj, "bias", None) is None
                for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj)
            )
            and type(self.rotary_emb) is RotaryEmbedding
            and self.rotary_emb.rotary_dim == self.rotary_emb.head_size
            and tp > 1
            and self.q_head_num % tp == 0
            and self.kv_head_num % tp == 0
        )

    def init_token_parallel_weights(self) -> None:
        self.tp_w_qkv = TokenParallelWeight(
            _replicated(
                self.q_proj.weight.value,
                self.k_proj.weight.value,
                self.v_proj.weight.value,
                mesh=self.mesh,
            )
        )
        self.tp_w_o = TokenParallelWeight(_replicated(self.o_proj.weight.value, mesh=self.mesh))

    @named_scope
    def forward_token_parallel(
        self,
        normed_states: jax.Array,
        cos_sin: tuple[jax.Array, jax.Array],
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        """Attention block for token-sharded [T, hidden] activations.

        Q/K/V and output projections are shard-local dots over the full
        contraction against replicated weights; q/k RMSNorm and RoPE run on the
        local tokens for all heads; one all-to-all hands every device the heads
        it owns for all tokens (the exact tensor-parallel layout the attention
        kernel and the head-sharded KV cache expect) and a second all-to-all
        returns the attention output to the token-sharded layout.
        """
        num_tokens = normed_states.shape[0]
        tok2 = _token_sharding(self.mesh, 2)
        tok3 = _token_sharding(self.mesh, 3)
        qkv = jax.lax.dot_general(
            normed_states,
            self.tp_w_qkv.value,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
            out_sharding=tok2,
        ).astype(normed_states.dtype)
        q = qkv[:, : self.q_size].reshape(
            num_tokens, self.q_head_num, self.head_dim, out_sharding=tok3
        )
        k = qkv[:, self.q_size : self.q_size + self.kv_size].reshape(
            num_tokens, self.kv_head_num, self.head_dim, out_sharding=tok3
        )
        v = qkv[:, self.q_size + self.kv_size :].reshape(
            num_tokens, self.kv_head_num, self.head_dim, out_sharding=tok3
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin, self.rotary_emb.is_neox_style)
        k = apply_rotary_emb(k, cos, sin, self.rotary_emb.is_neox_style)

        q, k, v = _qkv_tokens_to_heads(q, k, v, self.mesh)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        attn_output = _heads_to_tokens(attn_output, self.mesh)

        output = jax.lax.dot_general(
            attn_output,
            self.tp_w_o.value,
            (((1,), (0,)), ((), ())),
            preferred_element_type=self.o_proj.params_dtype,
            out_sharding=tok2,
        )
        return output, kv_fused


class Qwen3MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id
        self.mesh = mesh
        self.intermediate_size = intermediate_size

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

        self.act_fn = jax.nn.silu

    @named_scope
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ):
        if type(self.gate_proj) is LinearBase and type(self.up_proj) is LinearBase:
            w_g = getattr(self.gate_proj.weight, "value", self.gate_proj.weight)
            w_u = getattr(self.up_proj.weight, "value", self.up_proj.weight)
            a1, a2 = MergedColumnParallelLinear.shard_local_merged_column_dot(
                hidden_states,
                (w_g, w_u),
                (w_g.shape[-1], w_u.shape[-1]),
                self.mesh,
                preferred_element_type=jnp.float32,
                out_dtype=hidden_states.dtype,
            )
        else:
            a1, _ = self.gate_proj(hidden_states)
            a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel, out_sharding=out_sharding)
        return output

    def supports_token_parallel(self) -> bool:
        return (
            type(self.gate_proj) is LinearBase
            and type(self.up_proj) is LinearBase
            and type(self.down_proj) is LinearBase
            and all(
                getattr(proj, "bias", None) is None
                for proj in (self.gate_proj, self.up_proj, self.down_proj)
            )
        )

    def init_token_parallel_weights(self) -> None:
        mesh = self.mesh
        self.tp_w_gate = TokenParallelWeight(_replicated(self.gate_proj.weight.value, mesh=mesh))
        self.tp_w_up = TokenParallelWeight(_replicated(self.up_proj.weight.value, mesh=mesh))
        self.tp_w_down = TokenParallelWeight(_replicated(self.down_proj.weight.value, mesh=mesh))

    @named_scope
    def forward_token_parallel(self, hidden_states: jax.Array) -> jax.Array:
        """MLP on token-sharded [T, hidden] activations with replicated weights.

        Same dot forms as ``__call__`` (f32-accumulated gate/up cast to the
        activation dtype, down projection in ``params_dtype``), but every
        contraction is shard-local over the full dimension, so no all-reduce.
        """
        tok = _token_sharding(self.mesh, 2)
        dims = (((1,), (0,)), ((), ()))
        a1 = jax.lax.dot_general(
            hidden_states,
            self.tp_w_gate.value,
            dims,
            preferred_element_type=jnp.float32,
            out_sharding=tok,
        ).astype(hidden_states.dtype)
        a2 = jax.lax.dot_general(
            hidden_states,
            self.tp_w_up.value,
            dims,
            preferred_element_type=jnp.float32,
            out_sharding=tok,
        ).astype(hidden_states.dtype)
        intermediate = a2 * self.act_fn(a1)
        return jax.lax.dot_general(
            intermediate,
            self.tp_w_down.value,
            dims,
            preferred_element_type=self.down_proj.params_dtype,
            out_sharding=tok,
        )


class QWen3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_parameters["rope_theta"]
        rope_scaling = config.rope_parameters
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = QWen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            layer_id=layer_id,
            attention_bias=config.attention_bias,
            dtype=dtype,
            mesh=mesh,
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        cos_sin: tuple[jax.Array, jax.Array] | None = None,
    ):
        layer_callback_flag = []
        if residual is None:
            residual = hidden_states
        else:
            hidden_states += residual
            residual = hidden_states

        hidden_states, kv_fused = self.self_attn.forward_fused_rmsnorm_qkv(
            hidden_states=residual,
            input_layernorm=self.input_layernorm,
            positions=positions,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            cos_sin=cos_sin,
        )

        layer_norm_callback_flag = precision_tracer.jit_pure_callback_record(
            self.input_layernorm(residual),
            "input_layernorm_output",
            "INPUT_LAYERNORM",
            self.layer_id,
        )
        layer_callback_flag.append(layer_norm_callback_flag)

        attn_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "self_attn_output", "SELF_ATTN", self.layer_id
        )
        layer_callback_flag.append(attn_callback_flag)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        mlp_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "mlp_output", "MLP", self.layer_id
        )
        layer_callback_flag.append(mlp_callback_flag)

        return hidden_states, residual, kv_fused, layer_callback_flag

    def supports_token_parallel(self) -> bool:
        return self.self_attn.supports_token_parallel() and self.mlp.supports_token_parallel()

    def init_token_parallel_weights(self) -> None:
        self.self_attn.init_token_parallel_weights()
        self.mlp.init_token_parallel_weights()

    def forward_token_parallel(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None,
        cos_sin: tuple[jax.Array, jax.Array],
    ):
        """Decoder layer on token-sharded activations (same op order as __call__)."""
        if residual is None:
            residual = hidden_states
        else:
            hidden_states += residual
            residual = hidden_states

        normed_states = self.input_layernorm(residual)
        hidden_states, kv_fused = self.self_attn.forward_token_parallel(
            normed_states, cos_sin, forward_batch, token_to_kv_pool
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.forward_token_parallel(hidden_states)
        return hidden_states, residual, kv_fused


class QWen3Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        embed_sharding = (
            ("tensor", None) if getattr(config, "tie_word_embeddings", False) else (None, None)
        )
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=embed_sharding,
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                QWen3DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.rotary_emb = self.layers[0].self_attn.rotary_emb if len(self.layers) > 0 else None

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="norm",
        )
        # For EAGLE3 support
        self.layers_to_capture = []
        self.mesh = mesh
        # Set by Qwen3ForCausalLM.init_token_parallel_weights() once replicated
        # weights exist; gates the token-parallel prefill program.
        self.token_parallel_ready = False

    def supports_token_parallel(self) -> bool:
        return (
            len(self.layers) > 0
            and self.mesh.shape.get("data", 1) == 1
            and type(self.embed_tokens) is Embed
            and all(layer.supports_token_parallel() for layer in self.layers)
        )

    def init_token_parallel_weights(self) -> None:
        self.tp_embedding = TokenParallelWeight(
            _replicated(self.embed_tokens.embedding.value, mesh=self.mesh)
        )
        for layer in self.layers:
            layer.init_token_parallel_weights()
        self.token_parallel_ready = True

    def _use_token_parallel(self, forward_batch: ForwardBatch, positions: jax.Array) -> bool:
        if not self.token_parallel_ready:
            return False
        num_tokens = forward_batch.input_ids.shape[0]
        tp = self.mesh.shape["tensor"]
        return (
            forward_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
            and forward_batch.spec_info is None
            and forward_batch.input_embedding is None
            and forward_batch.deepstack_visual_embedding is None
            and not self.layers_to_capture
            and not precision_tracer._enable_precision_tracer
            and positions.ndim == 1
            and forward_batch.input_ids.ndim == 1
            and num_tokens >= TOKEN_PARALLEL_MIN_TOKENS
            and num_tokens % tp == 0
        )

    def _forward_token_parallel(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        positions: jax.Array,
        token_sharded_output: bool = False,
    ):
        """Prefill forward with the residual stream sharded over tokens.

        Replaces the two [T, hidden] all-reduces per layer of the
        tensor-parallel program with two small all-to-alls around the
        attention core (see QWen3Attention.forward_token_parallel); the
        embedding lookup, norms, residual adds and all dense projections run
        on the local T/tp tokens.
        """
        tok1 = _token_sharding(self.mesh, 1)
        tok2 = _token_sharding(self.mesh, 2)
        input_ids = jax.sharding.reshard(forward_batch.input_ids, tok1)
        hidden_states = self.tp_embedding.value.at[input_ids].get(out_sharding=tok2)

        rotary_emb = self.layers[0].self_attn.rotary_emb
        cos, sin = rotary_emb._compute_cos_sin(jax.sharding.reshard(positions, tok1))
        cos_sin = (cos.astype(rotary_emb.dtype), sin.astype(rotary_emb.dtype))

        residual = None
        layers_kv_fused = []
        layers_callback_flag = []
        for layer in self.layers:
            hidden_states, residual, kv_fused = layer.forward_token_parallel(
                hidden_states, forward_batch, token_to_kv_pool, residual, cos_sin
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend([None, None, None])

        hidden_states += residual
        hidden_states = self.norm(hidden_states)
        if not token_sharded_output:
            # Downstream consumers expect the token axis replicated across the
            # tensor axis.
            hidden_states = jax.sharding.reshard(
                hidden_states, NamedSharding(self.mesh, P("data", None))
            )
        layers_callback_flag.append(None)
        return hidden_states, [], layers_kv_fused, layers_callback_flag

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        positions: jax.Array | None = None,
        *,
        token_sharded_output: bool = False,
    ):
        positions = forward_batch.positions if positions is None else positions
        if getattr(self, "token_parallel_ready", False) and self._use_token_parallel(
            forward_batch, positions
        ):
            return self._forward_token_parallel(
                forward_batch, token_to_kv_pool, positions, token_sharded_output
            )
        residual = None
        hidden_states = (
            self.embed_tokens(forward_batch.input_ids)
            if forward_batch.input_embedding is None
            else forward_batch.input_embedding
        )
        layers_kv_fused = []
        layers_callback_flag = []
        aux_hidden_states = []
        cos_sin = self.rotary_emb.get_cos_sin(positions) if self.rotary_emb is not None else None
        # When connecting the vision head, even without deepstack, it should be padded with 0, which avoids doubling the EXTEND compilation.
        deepstack = forward_batch.deepstack_visual_embedding
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            hidden_states, residual, kv_fused, callback_flag = layer(
                positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                cos_sin=cos_sin,
            )
            if deepstack is not None and layer_id < deepstack.shape[0]:
                hidden_states = jax.lax.cond(
                    forward_batch.apply_for_deepstack,
                    lambda values: values[0] + values[1].astype(values[0].dtype),
                    lambda values: values[0],
                    (hidden_states, deepstack[layer_id]),
                )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "transformer_output", "TRANSFORMER"
        )
        layers_callback_flag.append(callback_flag)
        return hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag


class Qwen3ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("QWen3ForCausalLMModel config dtype: %s", self.dtype)
        self.model = QWen3Model(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                mesh=mesh,
                enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
            )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            mesh=self.mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_qwen3_weight_mappings()

        loader.load(weight_mappings)
        logger.info("Qwen3 weights loaded successfully!")
        if not getattr(model_config, "_abstract_mode", False):
            self.init_token_parallel_weights()

    def init_token_parallel_weights(self) -> bool:
        """Create the replicated weights used by the token-parallel prefill path.

        Costs one extra full (unsharded) copy of the dense decoder weights and
        the input embedding per device, so it is only enabled when that copy
        fits comfortably in the free device memory measured before the KV cache
        is sized.
        """
        from sgl_jax.srt.environ import envs
        from sgl_jax.srt.eplb.expert_location import get_global_server_args

        if not envs.SGLANG_JAX_TOKEN_PARALLEL_PREFILL.get():
            return False
        server_args = get_global_server_args()
        if server_args is not None and (
            getattr(server_args, "enable_lora", False)
            or getattr(server_args, "enable_static_lora", False)
            or getattr(server_args, "speculative_algorithm", None)
            or getattr(server_args, "aot_model_dir", None)
        ):
            return False
        if not self.model.supports_token_parallel():
            return False

        def _nbytes(x) -> int:
            return int(x.size) * jnp.dtype(x.dtype).itemsize

        weights = [self.model.embed_tokens.embedding.value]
        for layer in self.model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            for proj in (
                attn.q_proj,
                attn.k_proj,
                attn.v_proj,
                attn.o_proj,
                mlp.gate_proj,
                mlp.up_proj,
                mlp.down_proj,
            ):
                weights.append(proj.weight.value)
        # Deferred/shared (still abstract) weights cannot be replicated yet.
        if not all(isinstance(w, jax.Array) for w in weights):
            return False
        need = sum(_nbytes(w) for w in weights)
        del weights
        try:
            stats = jax.local_devices()[0].memory_stats()
        except Exception:  # pragma: no cover - backend without memory stats
            stats = None
        if stats and "bytes_limit" in stats:
            free = int(stats["bytes_limit"]) - int(stats.get("bytes_in_use", 0))
            if need > 0.3 * free:
                logger.info(
                    "Token-parallel prefill disabled: replicated weights need %.1f GB per "
                    "device, free %.1f GB",
                    need / (1 << 30),
                    free / (1 << 30),
                )
                return False

        self.model.init_token_parallel_weights()
        logger.info(
            "Token-parallel prefill enabled for extend batches >= %d tokens "
            "(replicated weights: %.2f GB per device)",
            TOKEN_PARALLEL_MIN_TOKENS,
            need / (1 << 30),
        )
        return True

    def _create_qwen3_weight_mappings(self) -> dict:
        embed_sharding = (
            ("tensor", None) if getattr(self.config, "tie_word_embeddings", False) else (None, None)
        )
        mappings = {
            "model.embed_tokens.weight": WeightSpec(
                target_path="model.embed_tokens.embedding",
                sharding=embed_sharding,
                transpose=False,
            ),
            "model.norm.weight": WeightSpec(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = self.lm_head.weight_mapping("lm_head.embedding")

        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightSpec(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightSpec(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.q_norm.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.q_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.k_norm.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.k_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        if getattr(self.config, "attention_bias", False):
            bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightSpec(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightSpec(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.bias": WeightSpec(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.o_proj.bias": WeightSpec(
                    target_path=f"{target_prefix}.self_attn.o_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
            }
            mappings.update(bias_mappings)

        return mappings

    def get_embed_and_head(self):
        return (
            self.model.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        """Set word embedding and LM Head weights.

        Args:
            embed_weight: Embedding matrix with shape [vocab_size, hidden_size].
            head_weight:  LM Head matrix with shape [vocab_size, hidden_size].
        """
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
            if self.model.token_parallel_ready:
                self.model.tp_embedding.value = _replicated(embed_weight, mesh=self.mesh)

        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None):

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: list[int]):
        if layer_ids is None:
            raise ValueError("DFLASH requires explicit target layer ids.")
        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def _token_parallel_selects_rows(
        self, forward_batch: ForwardBatch, logits_metadata: LogitsMetadata
    ) -> bool:
        """True when the logits processor only reads ``logits_indices`` rows."""
        mode = logits_metadata.forward_mode
        capture = logits_metadata.capture_hidden_mode
        return (
            self.model._use_token_parallel(forward_batch, forward_batch.positions)
            and mode.is_extend()
            and not mode.is_draft_extend()
            and not mode.is_target_verify()
            and not logits_metadata.extend_return_logprob
            and (capture is None or not capture.need_capture())
            and logits_metadata.logits_indices is not None
            and not self.capture_aux_hidden_states
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        # Token-parallel prefill whose logits only need the sampled rows: keep
        # the final hidden states token-sharded and gather just those rows
        # instead of all-gathering the full [T, hidden] activations.
        select_rows = self._token_parallel_selects_rows(forward_batch, logits_metadata)
        hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, kv_pool, token_sharded_output=select_rows
        )
        if select_rows:
            hidden_states = _select_token_rows(
                hidden_states, logits_metadata.logits_indices, self.mesh
            )
            logits_metadata = copy.copy(logits_metadata)
            logits_metadata.logits_indices = jax.sharding.reshard(
                jnp.arange(hidden_states.shape[0], dtype=jnp.int32),
                NamedSharding(self.mesh, P("data")),
            )

        if not self.capture_aux_hidden_states:
            aux_hidden_states = None
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(
                hidden_states, self.lm_head, logits_metadata, aux_hidden_states=aux_hidden_states
            )
        else:
            output = self.logits_processor(
                hidden_states,
                self.model.embed_tokens,
                logits_metadata,
                aux_hidden_states=aux_hidden_states,
            )

        return output, {"token_to_kv_pool": layers_kv_fused}, layers_callback_flag, None


EntryClass = Qwen3ForCausalLM
