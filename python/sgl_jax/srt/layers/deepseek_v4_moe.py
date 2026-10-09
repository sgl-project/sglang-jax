"""DeepSeek V4 MoE block for the EPMoE backend.

The caller supplies flattened token IDs alongside the collapsed [T, H] mHC
stream. This block does not own scheduler metadata or full-model loading.
"""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.deepseek_v4 import hash_moe_layer_flags
from sgl_jax.srt.kernels.sparse_core.moe_permute import moe_sc_permute_enabled_by_env
from sgl_jax.srt.layers.activation import silu_and_mul_with_clamp
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
from sgl_jax.srt.layers.moe import EPMoE

_W8A8_DENSE = os.environ.get("DSV4_W8A8_DENSE", "1") == "1"
_ACT_ROWS = os.environ.get("SGL_JAX_MOE_ACT_ROWS", "1") == "1"
_W8A8_DENSE_NAMES_ENV = os.environ.get(
    "DSV4_W8A8_DENSE_NAMES", "wq_a,wkv,wo_a,wo_b,indexer_wq_b,gate_proj,up_proj,down_proj"
)
_W8A8_DENSE_NAMES = (
    frozenset(x.strip() for x in _W8A8_DENSE_NAMES_ENV.split(",") if x.strip())
    if _W8A8_DENSE_NAMES_ENV
    else None
)


class DeepseekV4EPMoE(EPMoE):
    """V4 activation semantics over the shared expert-parallel execution path."""

    def __init__(self, *args, swiglu_limit: float | None = None, **kwargs):
        if kwargs.get("activation", "silu") != "silu":
            raise ValueError("V4 experts require silu activation")
        if swiglu_limit is not None and (not math.isfinite(swiglu_limit) or swiglu_limit <= 0):
            raise ValueError("swiglu_limit requires a finite positive limit")
        super().__init__(*args, **kwargs)
        self.swiglu_limit = swiglu_limit

    def _apply_activation(self, layer_w0, layer_w1):
        if self.swiglu_limit is None:
            return super()._apply_activation(layer_w0, layer_w1)
        return silu_and_mul_with_clamp(layer_w0, layer_w1, self.swiglu_limit)

    def _apply_activation_with_context(self, layer_w0, layer_w1, *, local_range=None):
        if self.swiglu_limit is None or not _ACT_ROWS or local_range is None:
            return self._apply_activation(layer_w0, layer_w1)
        from sgl_jax.srt.kernels.dsv4.moe_act import silu_mul_rows

        return silu_mul_rows(
            layer_w0,
            layer_w1,
            local_range[0],
            local_range[1],
            limit=self.swiglu_limit,
            interpret=os.environ.get("PALLAS_INTERPRET", "0") == "1",
        )


class DeepseekV4SharedMLP(nnx.Module):
    def __init__(
        self, hidden_size, intermediate_size, mesh, dtype, swiglu_limit, *, quantized=False
    ):
        self.swiglu_limit = swiglu_limit
        for name in ("gate_proj", "up_proj", "down_proj"):
            down = name == "down_proj"
            input_size = intermediate_size if down else hidden_size
            output_size = hidden_size if down else intermediate_size
            axes = ("tensor", None) if down else (None, "tensor")
            if quantized:
                if input_size % 128 or output_size % 128:
                    raise ValueError(
                        f"static V4 FP8 linear {name} requires dimensions divisible by 128"
                    )
                if axes[0] is not None and (input_size // 128) % mesh.shape[axes[0]]:
                    axes = (None, axes[1])
                linear = QuantizedLinear(
                    weight_q=jnp.zeros(
                        (output_size, input_size),
                        jnp.float8_e4m3fn,
                        out_sharding=P(axes[1], axes[0]),
                    ),
                    weight_scale=jnp.zeros(
                        (input_size // 128, 1, output_size),
                        jnp.float32,
                        out_sharding=P(axes[0], None, axes[1]),
                    ),
                    bias=None,
                    activation_dtype=(
                        jnp.float8_e4m3fn
                        if _W8A8_DENSE and (_W8A8_DENSE_NAMES is None or name in _W8A8_DENSE_NAMES)
                        else None
                    ),
                    mesh=mesh,
                    kernel_axes=axes,
                    params_dtype=dtype,
                    weight_block_size=(128, 128),
                    allow_narrow_n_blockwise=True,
                    scope_name=name,
                )
            else:
                linear = LinearBase(
                    input_size=input_size,
                    output_size=output_size,
                    kernel_axes=axes,
                    use_bias=False,
                    params_dtype=dtype,
                    mesh=mesh,
                    scope_name=name,
                )
            setattr(self, name, linear)

    def __call__(self, hidden_states):
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        activated = (
            jax.nn.silu(gate) * up
            if self.swiglu_limit is None
            else silu_and_mul_with_clamp(gate, up, self.swiglu_limit)
        )
        output, _ = self.down_proj(activated)
        return output


class DeepseekV4MoE(nnx.Module):
    """Hash layers and learned routing share scoring, normalization and experts.

    ``load_hash_table`` must be called with the checkpoint's ``gate.tid2eid``
    before real inference; the deterministic initial table is for dummy loads.
    ``route`` exposes weights/IDs for routing diagnostics without running GMM.
    """

    def __init__(self, config, mesh, layer_id, dtype=jnp.bfloat16, *, backend="auto"):
        self.mesh = mesh
        if backend not in ("auto", "epmoe", "reference"):
            raise ValueError(f"unknown V4 MoE backend {backend!r}")
        from sgl_jax.srt.utils.jax_utils import is_tpu_runtime

        self.backend = (
            ("epmoe" if is_tpu_runtime(mesh) else "reference") if backend == "auto" else backend
        )
        self.layer_id = layer_id
        quant = getattr(config, "quantization_config", None)
        self.static_fp8 = quant is not None and getattr(quant, "is_static_checkpoint", False)
        self.dtype = dtype
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.vocab_size = config.vocab_size
        if self.num_experts < 1 or getattr(config, "n_shared_experts", 1) != 1:
            raise ValueError("Flash V4 requires routed experts and one shared expert")
        if getattr(config, "scoring_func", "sqrtsoftplus") != "sqrtsoftplus":
            raise ValueError("Flash V4 requires sqrtsoftplus routing")
        if getattr(config, "topk_method", "noaux_tc") != "noaux_tc":
            raise ValueError("Flash V4 requires noaux_tc expert selection")
        if getattr(config, "expert_dtype", None) == "fp4" and (
            config.hidden_size % 32 or config.moe_intermediate_size % 32
        ):
            raise ValueError("MXFP4 expert K dimensions must be multiples of 32")
        if not 0 <= layer_id < config.num_hidden_layers:
            raise ValueError("layer_id outside the V4 backbone")
        self.is_hash_layer = hash_moe_layer_flags(config)[layer_id]
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, n_routed_experts]")
        if self.vocab_size <= 0 or layer_id < 0:
            raise ValueError("vocab_size must be positive and layer_id nonnegative")
        self.gate = GateLogit(
            self.hidden_size,
            self.num_experts,
            weight_dtype=jnp.float32,
            enable_expert_bias=not self.is_hash_layer,
            score_func=getattr(config, "scoring_func", "sqrtsoftplus"),
        )
        if self.is_hash_layer:
            table = (np.arange(self.vocab_size)[:, None] + np.arange(self.top_k)) % self.num_experts
            self.gate.tid2eid = nnx.Param(
                jax.device_put(table.astype(np.int32), NamedSharding(mesh, P(None, None)))
            )
            self._hash_table_loaded = False
        self.topk = TopK(
            topk=self.top_k,
            renormalize=config.norm_topk_prob,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            routed_scaling_factor=config.routed_scaling_factor,
            layer_id=layer_id,
            mesh=mesh,
        )
        self.experts = DeepseekV4EPMoE(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.top_k,
            intermediate_dim=config.moe_intermediate_size,
            mesh=mesh,
            ep_size=getattr(config, "ep_size", 1),
            moe_dp_size=getattr(config, "moe_dp_size", 1),
            weight_dtype=dtype,
            dtype=dtype,
            layer_id=layer_id,
            quantization_config=getattr(config, "quantization_config", None),
            swiglu_limit=config.swiglu_limit,
            use_sc_permute=moe_sc_permute_enabled_by_env("true"),
            sort_free_permute=True,
        )
        if getattr(config, "expert_dtype", None) == "fp4":
            if self.experts.replicate_experts:
                raise ValueError("V4 resident FP8 experts require moe_dp_size=1")
            self.experts.quantized_dtype = jnp.float8_e4m3fn
            self.experts.weight_block_size = None
            with jax.sharding.use_abstract_mesh(self.experts.updated_mesh):
                for name in ("wi_0", "wi_1", "wo"):
                    old = getattr(self.experts, name).value
                    spec = jax.typeof(old).sharding.spec
                    setattr(
                        self.experts,
                        name,
                        nnx.Param(jnp.zeros(old.shape, jnp.float8_e4m3fn, out_sharding=spec)),
                    )
                    setattr(
                        self.experts,
                        name + "_scale",
                        nnx.data(
                            nnx.Param(
                                jnp.ones(
                                    (old.shape[0], 1, 1, old.shape[2]),
                                    jnp.float32,
                                    out_sharding=P(spec[0], None, None, spec[2]),
                                )
                            )
                        ),
                    )
        if getattr(config, "n_shared_experts", 0):
            self.shared_experts = DeepseekV4SharedMLP(
                self.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                mesh,
                dtype,
                config.swiglu_limit,
                quantized=self.static_fp8,
            )
        else:
            self.shared_experts = None

    def load_hash_table(self, table):
        """Load a host checkpoint tensor without floating-point dtype conversion."""
        if not self.is_hash_layer:
            raise ValueError("only hash layers have gate.tid2eid")
        table = np.asarray(table)
        if table.shape != (self.vocab_size, self.top_k):
            raise ValueError("gate.tid2eid must have shape [vocab_size, top_k]")
        if not np.issubdtype(table.dtype, np.integer):
            raise ValueError("gate.tid2eid must contain integer expert IDs")
        if np.any(table < 0) or np.any(table >= self.num_experts):
            raise ValueError("gate.tid2eid expert IDs are outside the logical expert range")
        self.gate.tid2eid.value = jax.device_put(
            table.astype(np.int32), NamedSharding(self.mesh, P(None, None))
        )
        self._hash_table_loaded = True

    def load_owned_weights(
        self, assigned_info, *, expert_format=None, row_chunk_size=128, weight_source=None
    ):
        """Load only this layer's E-owned inventory and report consumed source keys."""
        from sgl_jax.srt.layers.deepseek_v4_moe_loader import load_moe_weights

        return load_moe_weights(
            self,
            assigned_info,
            expert_format=expert_format,
            row_chunk_size=row_chunk_size,
            weight_source=weight_source,
        )

    def route(
        self,
        hidden_states,
        input_ids=None,
        *,
        token_valid_mask=None,
        dispatch_info=None,
        routing_sharding=None,
        return_logical_ids=False,
    ):
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if hidden_states.dtype != self.dtype:
            raise ValueError(f"hidden_states must have model activation dtype {self.dtype}")
        if self.experts.replicate_experts and dispatch_info is not None:
            raise ValueError("replicated experts do not support EPLB dispatch metadata")
        routing_sharding = routing_sharding or NamedSharding(self.mesh, P("data", None))
        if (
            not isinstance(routing_sharding, NamedSharding)
            or routing_sharding.mesh != self.mesh
            or len(routing_sharding.spec) != 2
            or routing_sharding.spec[1] is not None
        ):
            raise ValueError("routing sharding must partition tokens only")
        hidden_states = jax.sharding.reshard(hidden_states, routing_sharding)
        token_sharding = NamedSharding(self.mesh, P(routing_sharding.spec[0]))
        tokens = hidden_states.shape[0]
        valid = jnp.ones((tokens,), dtype=jnp.bool_)
        if token_valid_mask is not None:
            if token_valid_mask.shape != (tokens,):
                raise ValueError("token_valid_mask must have shape [tokens]")
            if token_valid_mask.dtype != jnp.bool_:
                raise ValueError("token_valid_mask must be boolean")
            valid = jax.sharding.reshard(token_valid_mask.astype(jnp.bool_), token_sharding)
        selected = None
        if self.is_hash_layer:
            if not self._hash_table_loaded:
                raise ValueError("hash routing requires checkpoint gate.tid2eid")
            if input_ids is None or input_ids.shape != (tokens,):
                raise ValueError("hash routing requires input_ids with shape [tokens]")
            if not jnp.issubdtype(input_ids.dtype, jnp.integer):
                raise ValueError("input_ids must be integers")
            input_ids = jax.sharding.reshard(input_ids, token_sharding)
            valid = valid & (input_ids >= 0) & (input_ids < self.vocab_size)
            # Padding must never produce negative IDs in EPMoE's bincount/permutation.
            safe_ids = jnp.where(valid, input_ids, 0)
            selected = self.gate.tid2eid.value.at[safe_ids].get(out_sharding=routing_sharding)
        scores = self.gate(jnp.where(valid[:, None], hidden_states, 0))
        routed = self.topk(
            scores,
            None if self.is_hash_layer else self.gate.bias.value,
            dispatch_info,
            routing_sharding,
            selected_experts=selected,
            return_logical_ids=return_logical_ids,
        )
        if return_logical_ids:
            weights, ids, logical_ids = routed
            return jnp.where(valid[:, None], weights, 0), ids, logical_ids
        weights, ids = routed
        return jnp.where(valid[:, None], weights, 0), ids

    def _reference_experts(self, hidden_states, weights, ids, out_sharding):
        """Unfused dense numerical baseline over the resident EPMoE layout."""
        x = jax.sharding.reshard(
            hidden_states.astype(jnp.float32), NamedSharding(self.mesh, P("data", None))
        )
        projections = []
        for name in ("wi_0", "wi_1", "wo"):
            expert_weights = getattr(self.experts, name).value
            expert_weights = jax.sharding.reshard(
                expert_weights, NamedSharding(self.mesh, P(None, None, None))
            )
            selected = (
                expert_weights.at[ids]
                .get(out_sharding=NamedSharding(self.mesh, P("data", None, None, None)))
                .astype(jnp.float32)
            )
            scale_param = getattr(self.experts, name + "_scale", None)
            if scale_param is not None:
                scale_values = jax.sharding.reshard(
                    scale_param.value, NamedSharding(self.mesh, P(None, None, None, None))
                )
                per_channel = jnp.squeeze(scale_values, axis=(1, 2))
                selected_scale = per_channel.at[ids].get(
                    out_sharding=NamedSharding(self.mesh, P("data", None, None))
                )
                selected = selected * selected_scale[:, :, None, :]
            projections.append(selected)
        w1, w3, w2 = projections
        gate = jnp.einsum("th,tqhn->tqn", x, w1)
        up = jnp.einsum("th,tqhn->tqn", x, w3)
        if self.experts.swiglu_limit is None:
            activated = jax.nn.silu(gate) * up
        else:
            activated = silu_and_mul_with_clamp(gate, up, self.experts.swiglu_limit)
        routed = jnp.einsum("tqi,tqih->tqh", activated, w2)
        output = jnp.sum(routed * weights[:, :, None], axis=1).astype(hidden_states.dtype)
        return jax.sharding.reshard(output, out_sharding) if out_sharding is not None else output

    def _reference_static_shared(self, hidden_states):
        """Unfused static shared-expert oracle from resident FP8 weights and scales."""
        x = hidden_states.astype(jnp.float32)

        def project(linear, activation):
            weight = jax.sharding.reshard(
                linear.weight_q.value, NamedSharding(self.mesh, P(None, None))
            ).astype(jnp.float32)
            scale = jax.sharding.reshard(
                linear.weight_scale.value, NamedSharding(self.mesh, P(None, None, None))
            )
            expanded = jnp.repeat(scale[:, 0, :], 128, axis=0)[: weight.shape[1], :]
            return activation @ (weight.T * expanded)

        gate = project(self.shared_experts.gate_proj, x)
        up = project(self.shared_experts.up_proj, x)
        activated = (
            jax.nn.silu(gate) * up
            if self.shared_experts.swiglu_limit is None
            else silu_and_mul_with_clamp(gate, up, self.shared_experts.swiglu_limit)
        )
        return project(self.shared_experts.down_proj, activated).astype(hidden_states.dtype)

    def __call__(
        self,
        hidden_states,
        input_ids=None,
        *,
        token_valid_mask=None,
        dispatch_info=None,
        out_sharding=None,
        output_sharding=None,
        return_expert_ids=False,
    ):
        if output_sharding is None:
            output_sharding = out_sharding
        if output_sharding is not None and (
            not isinstance(output_sharding, NamedSharding)
            or output_sharding.mesh != self.mesh
            or len(output_sharding.spec) != 2
            or output_sharding.spec[1] is not None
        ):
            raise ValueError("output sharding must partition tokens only")
        routed = self.route(
            hidden_states,
            input_ids,
            token_valid_mask=token_valid_mask,
            dispatch_info=dispatch_info,
            routing_sharding=out_sharding,
            return_logical_ids=return_expert_ids,
        )
        if return_expert_ids:
            weights, ids, logical_ids = routed
        else:
            weights, ids = routed
        # Zero invalid activations too: zero routing weight alone cannot mask NaN.
        valid = jnp.ones(hidden_states.shape[:1], dtype=jnp.bool_)
        if token_valid_mask is not None:
            valid = valid & token_valid_mask.astype(jnp.bool_)
        if self.is_hash_layer:
            valid = valid & (input_ids >= 0) & (input_ids < self.vocab_size)
        hidden_states = jnp.where(valid[:, None], hidden_states, 0)
        if self.backend == "reference":
            output = self._reference_experts(hidden_states, weights, ids, output_sharding)
        else:
            output = self.experts(hidden_states, weights, ids, out_sharding=output_sharding)
        if self.shared_experts is not None:
            # routed_scaling_factor applies only to routed weights, exactly once.
            shared_input = (
                jax.sharding.reshard(hidden_states, NamedSharding(self.mesh, P("data", None)))
                if self.static_fp8
                else hidden_states
            )
            shared = (
                self._reference_static_shared(shared_input)
                if self.backend == "reference" and self.static_fp8
                else self.shared_experts(shared_input)
            )
            if output_sharding is not None:
                shared = jax.sharding.reshard(shared, output_sharding)
            output = output + shared
        if output_sharding is not None:
            valid = jax.sharding.reshard(
                valid, NamedSharding(self.mesh, P(output_sharding.spec[0]))
            )
        output = jnp.where(valid[:, None], output, 0).astype(hidden_states.dtype)
        return (output, logical_ids) if return_expert_ids else output
