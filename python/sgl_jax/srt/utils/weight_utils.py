import copy
import glob
import json
import logging
import math
import os
import pickle
import re
import struct
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx
from jax.experimental import multihost_utils

if TYPE_CHECKING:
    from sgl_jax.srt.layers.linear import LinearBase
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from tqdm import tqdm

from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

if not hasattr(np, "float8_e4m3fn"):
    np.float8_e4m3fn = ml_dtypes.float8_e4m3fn
if not hasattr(np, "float8_e5m2"):
    np.float8_e5m2 = ml_dtypes.float8_e5m2


def _view_as_fp8_if_needed(data: np.ndarray, target_dtype: jnp.dtype) -> np.ndarray:
    if data.dtype == np.uint8:
        if target_dtype == jnp.float8_e4m3fn:
            return data.view(ml_dtypes.float8_e4m3fn)
        elif target_dtype == jnp.float8_e5m2:
            return data.view(ml_dtypes.float8_e5m2)
    return data


@dataclass
class WeightMapping:
    target_path: str | list[str]
    sharding: tuple | None = None
    transpose: bool = False
    transpose_axes: tuple[int, ...] | None = (
        None  # For multi-dimensional transpose (e.g., conv weights)
    )
    reshape: tuple | None = None
    repeat: tuple[int, int] | None = None
    head_dim_padding: bool = False
    kv_head_padding: bool = False
    concat_axis: int | None = None
    is_eagle3: bool = False
    physical_to_logical_map: np.ndarray | None = None

    def __post_init__(self):
        if self.sharding is None:
            self.sharding = self._infer_default_sharding()

    def _infer_default_sharding(self) -> tuple:
        path = self.target_path[0] if isinstance(self.target_path, list) else self.target_path

        if any(pattern in path for pattern in ["embedding", "lm_head"]):
            return (None, None)
        elif any(
            pattern in path
            for pattern in [
                "q_proj",
                "k_proj",
                "v_proj",
                "w1",
                "w3",
                "w2",
                "gate_proj",
                "up_proj",
            ]
        ):
            return (None, "tensor")
        elif any(pattern in path for pattern in ["c_proj", "o_proj", "down_proj"]):
            return ("tensor", None)
        elif "bias" in path or "weight" in path:
            return (None,)
        else:
            return (None,)


class SequentialSafetensorManager:
    """
    Manages open file handles during a weight loading session to prevent
    repeated opening/parsing of safetensors headers.
    """

    def __init__(self):
        self.handles = {}

    def get_handle(self, filename):
        if filename not in self.handles:
            # Keep the file open. framework="np" is crucial for JAX interop.
            # device="cpu" ensures we don't accidentally alloc on GPU/TPU here.
            self.handles[filename] = safe_open(filename, framework="np", device="cpu")
        return self.handles[filename]

    def close_all(self):
        # safe_open objects don't strictly require close() as they rely on RAII/GC,
        # but clearing references ensures we don't hold descriptors.
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()


class WeightLoader:
    def __init__(
        self,
        model: nnx.Module,
        model_config: ModelConfig,
        mesh: Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.model = model
        self.model_config = model_config
        self.mesh = mesh
        self.dtype = dtype
        self.dummy_mode = getattr(model_config, "_dummy_mode", False)
        if hasattr(model_config, "num_attention_heads"):
            self.num_heads = model_config.num_attention_heads
            # Use original count for replication logic
            self.num_kv_heads = model_config.get_total_num_kv_heads()
            self.hidden_size = model_config.hidden_size
            self.head_dim_original = getattr(
                model_config, "head_dim", self.hidden_size // self.num_heads
            )

            self.head_dim_pad = (self.head_dim_original + 127) // 128 * 128 - self.head_dim_original
            self.head_dim = self.head_dim_original
            self.v_head_dim = getattr(model_config, "v_head_dim", self.head_dim_original)
        if hasattr(self.mesh, "shape") and "tensor" in self.mesh.shape:
            self.sharding_size = self.mesh.shape["tensor"]
        else:
            self.sharding_size = 1

        if hasattr(model_config, "ep_size") and model_config.ep_size > 1:
            world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
            tp_size = world_size // model_config.ep_size
            ep_size = model_config.ep_size
            abstract_mesh = self.mesh.abstract_mesh
            self.moe_abstract_mesh = abstract_mesh.update(
                axis_sizes=(ep_size, tp_size), axis_names=("expert", "tensor")
            )
        else:
            self.moe_abstract_mesh = None

    # ------------------------------------------------------------------
    # Quant config helpers
    # ------------------------------------------------------------------

    @property
    def is_static_quant(self) -> bool:
        """Check if the model uses a static FP8 checkpoint."""
        quant_cfg = getattr(self.model_config, "quantization_config", None)
        return quant_cfg is not None and quant_cfg.is_static_checkpoint

    def is_quant_ignored(self, hf_path: str) -> bool:
        """Check if a HuggingFace weight path is in the quantization ignored_layers list."""
        quant_cfg = getattr(self.model_config, "quantization_config", None)
        if quant_cfg is None or not quant_cfg.is_static_checkpoint:
            return True
        ignored = quant_cfg.ignored_layers or []
        return any(hf_path == ig or hf_path.endswith(f".{ig}") for ig in ignored)

    # ------------------------------------------------------------------
    # Post-load hooks: dequant FP8 → BF16, KV head replication
    # ------------------------------------------------------------------

    @staticmethod
    def create_bf16_linear(
        weight: jax.Array, kernel_axes, mesh, use_bias=False, bias=None
    ) -> "LinearBase":
        """Create a bf16 LinearBase from a weight array [in, out]."""
        from sgl_jax.srt.layers.linear import LinearBase

        in_features, out_features = weight.shape
        with jax.set_mesh(mesh):
            new_linear = LinearBase(
                input_size=in_features,
                output_size=out_features,
                kernel_axes=kernel_axes,
                use_bias=use_bias,
                params_dtype=jnp.bfloat16,
                mesh=mesh,
            )
            new_linear.weight = nnx.Param(weight)
            if bias is not None:
                new_linear.bias = nnx.Param(bias.astype(jnp.bfloat16))
        return new_linear

    def dequant_fp8_linear(self, ql, head_dim: int | None = None) -> "LinearBase":
        """Dequantize a single QuantizedLinear → bf16 LinearBase.

        Handles 3D block-quant scales, 1D per-channel scales, kv_head_padding,
        and per-head block quant (when head_dim % block_size != 0).

        Args:
            ql: QuantizedLinear module with weight_q and weight_scale.
            head_dim: If set, enables per-head block quant handling.
        """
        weight_q = ql.weight_q.value
        weight_scale = ql.weight_scale.value

        if weight_scale.ndim == 3:
            weight_bf16 = self._block_dequant(weight_q, weight_scale, head_dim=head_dim)
        elif weight_scale.ndim == 1:
            out_dim = weight_scale.shape[0]
            if weight_q.shape[1] == out_dim:
                weight_bf16 = (weight_q.astype(jnp.float32) * weight_scale[None, :]).astype(
                    jnp.bfloat16
                )
            else:
                weight_bf16 = (
                    jnp.transpose(weight_q).astype(jnp.float32) * weight_scale[None, :]
                ).astype(jnp.bfloat16)
        else:
            raise ValueError(f"Unexpected weight_scale ndim={weight_scale.ndim}")

        bias = ql.bias.value if ql.bias is not None else None
        return self.create_bf16_linear(
            weight_bf16, ql.kernel_axes, ql.mesh, ql.bias is not None, bias
        )

    @staticmethod
    def _block_dequant(
        weight_q: jax.Array, weight_scale: jax.Array, head_dim: int | None = None
    ) -> jax.Array:
        """Block-dequantize weight_q using 3D scale [in_blocks, 1, out_dim].

        Returns bf16 weight in model layout [in_dim, out_dim].
        Handles kv_head_padding (scale tiling) and per-head block quant.
        """
        import math

        in_blocks = weight_scale.shape[0]
        out_dim = weight_scale.shape[2]
        dim0, dim1 = weight_q.shape

        if dim1 == out_dim:
            pass  # Model layout [in, out], no kv-padding
        elif dim0 == out_dim:
            weight_q = jnp.transpose(weight_q)  # HF layout [out, in]
        elif head_dim is not None and dim1 != out_dim and dim0 != out_dim:
            # Per-head block quant: scale was expanded for wrong n_out.
            if dim0 % in_blocks == 0:
                actual_out = dim1  # model layout [in, out]
            else:
                weight_q = jnp.transpose(weight_q)
                actual_out = dim0  # was HF layout [out, in]

            block_size = 128
            blocks_per_head = math.ceil(head_dim / block_size)
            num_heads = actual_out // head_dim

            gather_idx = jnp.array(
                [
                    ((j // head_dim) * blocks_per_head + (j % head_dim) // block_size) * block_size
                    for j in range(actual_out)
                ]
            )
            weight_scale = weight_scale[:, :, gather_idx]
            out_dim = actual_out

            logger.info(
                "Per-head block dequant: %d heads x head_dim=%d, %d blocks/head, "
                "remapped scale to (%s)",
                num_heads,
                head_dim,
                blocks_per_head,
                weight_scale.shape,
            )
        elif dim1 > out_dim and dim1 % out_dim == 0:
            # Model layout [in, kv_padded_out] — tile scale
            kv_replicas = dim1 // out_dim
            weight_scale = jnp.tile(weight_scale, (1, 1, kv_replicas))
            out_dim = dim1
        elif dim0 > out_dim and dim0 % out_dim == 0:
            # HF layout [kv_padded_out, in] — transpose and tile scale
            kv_replicas = dim0 // out_dim
            weight_q = jnp.transpose(weight_q)
            weight_scale = jnp.tile(weight_scale, (1, 1, kv_replicas))
            out_dim = weight_q.shape[1]
        else:
            raise ValueError(
                f"Cannot match weight_q shape {weight_q.shape} with scale out_dim={out_dim}"
            )

        in_dim = weight_q.shape[0]
        block_k = in_dim // in_blocks
        weight_f = weight_q.astype(jnp.float32).reshape(in_blocks, block_k, out_dim)
        weight_bf16 = (weight_f * weight_scale).reshape(in_dim, out_dim).astype(jnp.bfloat16)
        return weight_bf16

    def dequant_fp8_layers(
        self,
        layers: list,
        specs: list[tuple[str, int | None]],
        *,
        layer_filter=None,
    ):
        """Dequantize specified QuantizedLinear projections → bf16 LinearBase.

        Args:
            layers: model.layers list.
            specs: list of (dotted_path, head_dim) tuples, e.g.
                [("self_attn.q_proj", 192), ("self_attn.v_proj", 128)].
            layer_filter: optional callable(layer_idx, layer) -> bool.
        """
        from sgl_jax.srt.layers.linear import QuantizedLinear

        for layer_idx, layer in enumerate(layers):
            if layer_filter is not None and not layer_filter(layer_idx, layer):
                continue
            for proj_path, hd in specs:
                parts = proj_path.split(".")
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr_name = parts[-1]
                proj = getattr(parent, attr_name)
                if isinstance(proj, QuantizedLinear):
                    setattr(parent, attr_name, self.dequant_fp8_linear(proj, head_dim=hd))
                    logger.info("Dequantized layer %d %s -> bf16", layer_idx, proj_path)

        logger.info("FP8 -> BF16 dequantization complete.")

    def replicate_kv_heads(
        self,
        layers: list,
        specs: list[tuple[str, int]],
        target_kv_heads_fn,
    ):
        """Replicate KV heads for TP alignment.

        Args:
            layers: model.layers list.
            specs: list of (dotted_path, head_dim) tuples, e.g.
                [("self_attn.k_proj", 192), ("self_attn.v_proj", 128)].
            target_kv_heads_fn: callable(attn_module) -> int, returns expected
                TP-aligned KV head count.
        """
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            target_kv_heads = target_kv_heads_fn(attn)

            for proj_path, hd in specs:
                parts = proj_path.split(".")
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr_name = parts[-1]
                proj = getattr(parent, attr_name)
                w = proj.weight.value
                expected_size = target_kv_heads * hd
                actual_size = w.shape[1]

                if actual_size == expected_size:
                    continue

                if actual_size > 0 and expected_size % actual_size == 0:
                    num_replicas = expected_size // actual_size
                    orig_kv = actual_size // hd

                    logger.info(
                        "KV head replication: layer %d %s %d->%d heads (%d->%d)",
                        layer_idx,
                        proj_path,
                        orig_kv,
                        target_kv_heads,
                        actual_size,
                        expected_size,
                    )

                    w_full = jax.device_put(w, NamedSharding(self.mesh, P()))
                    w_3d = w_full.reshape(w.shape[0], orig_kv, hd)
                    w_rep = jnp.repeat(w_3d, num_replicas, axis=1)
                    w_new = w_rep.reshape(w.shape[0], expected_size)
                    w_new = jax.device_put(w_new, NamedSharding(self.mesh, P(None, "tensor")))

                    setattr(
                        parent,
                        attr_name,
                        self.create_bf16_linear(w_new, proj.kernel_axes, self.mesh),
                    )

    @staticmethod
    def _uniform_block_dequant(weight, scale, block_size):
        """Uniform block dequant for weight[out_dim, in_dim] * scale[out_blocks, in_blocks]."""
        out_dim, in_dim = weight.shape
        out_blocks = scale.shape[0]
        padded_out = out_blocks * block_size
        in_blocks = scale.shape[1]
        if padded_out > out_dim:
            weight = jnp.pad(weight, ((0, padded_out - out_dim), (0, 0)))
        w_4d = weight.astype(jnp.float32).reshape(out_blocks, block_size, in_blocks, block_size)
        s_4d = scale[:, None, :, None]
        result = (w_4d * s_4d).reshape(padded_out, in_dim)[:out_dim, :].astype(jnp.bfloat16)
        return result

    def dequant_fused_kv(
        self,
        kv_buffers: dict[int, dict],
        layers: list,
        config,
    ):
        """Dequantize FP8 K+V weights with per-layer quantization scheme detection.

        Handles two schemes:
        - Per-head fused: K+V quantized as fused [K(head_dim), V(v_head_dim)] per head.
          Block boundaries cross K/V boundary, so they must be fused for correct dequant.
        - Uniform: K and V quantized independently across the whole tensor.

        Args:
            kv_buffers: dict mapping layer_idx -> {k_weight, k_scale, v_weight, v_scale}
            layers: model.layers list
            config: model config with head_dim, v_head_dim
        """
        import math

        from jax.sharding import NamedSharding

        if not kv_buffers:
            return

        head_dim = config.head_dim
        v_head_dim = getattr(config, "v_head_dim", head_dim)
        quant_cfg = getattr(config, "quantization_config", None)
        block_size = int(quant_cfg.weight_block_size[0]) if quant_cfg else 128

        fused_dim = head_dim + v_head_dim
        blocks_per_head = math.ceil(fused_dim / block_size)
        padded_dim = blocks_per_head * block_size
        k_blocks_per_head = math.ceil(head_dim / block_size)
        v_blocks_per_head = blocks_per_head - k_blocks_per_head

        tp_sharding = NamedSharding(self.mesh, P(None, "tensor"))

        for layer_idx in sorted(kv_buffers.keys()):
            buf = kv_buffers[layer_idx]
            k_weight = buf["k_weight"]
            k_scale = buf["k_scale"]
            v_weight = buf["v_weight"]
            v_scale = buf["v_scale"]

            in_dim = k_weight.shape[1]
            in_blocks = in_dim // block_size

            num_kv_heads = k_weight.shape[0] // head_dim
            k_scale_blocks = k_scale.shape[0]

            expected_per_head = num_kv_heads * k_blocks_per_head
            expected_uniform = math.ceil(num_kv_heads * head_dim / block_size)
            is_per_head = (
                k_scale_blocks == expected_per_head and expected_per_head != expected_uniform
            )

            if is_per_head:
                k_w = k_weight.reshape(num_kv_heads, head_dim, in_dim)
                v_w = v_weight.reshape(num_kv_heads, v_head_dim, in_dim)
                k_s = k_scale.reshape(num_kv_heads, k_blocks_per_head, in_blocks)
                v_s = v_scale.reshape(num_kv_heads, v_blocks_per_head, in_blocks)
                fused_w = jnp.concatenate([k_w, v_w], axis=1)
                fused_s = jnp.concatenate([k_s, v_s], axis=1)
                if fused_dim < padded_dim:
                    fused_w = jnp.pad(fused_w, ((0, 0), (0, padded_dim - fused_dim), (0, 0)))
                fused_5d = fused_w.astype(jnp.float32).reshape(
                    num_kv_heads, blocks_per_head, block_size, in_blocks, block_size
                )
                scale_5d = fused_s[:, :, None, :, None]
                dequanted = (
                    (fused_5d * scale_5d)
                    .reshape(num_kv_heads, padded_dim, in_dim)[:, :fused_dim, :]
                    .astype(jnp.bfloat16)
                )
                k_bf16 = dequanted[:, :head_dim, :].reshape(num_kv_heads * head_dim, in_dim)
                v_bf16 = dequanted[:, head_dim:, :].reshape(num_kv_heads * v_head_dim, in_dim)
            else:
                k_bf16 = self._uniform_block_dequant(k_weight, k_scale, block_size)
                v_bf16 = self._uniform_block_dequant(v_weight, v_scale, block_size)

            k_bf16 = jax.device_put(jnp.transpose(k_bf16), tp_sharding)
            v_bf16 = jax.device_put(jnp.transpose(v_bf16), tp_sharding)

            attn = layers[layer_idx].self_attn
            attn.k_proj = self.create_bf16_linear(k_bf16, (None, "tensor"), self.mesh)
            attn.v_proj = self.create_bf16_linear(v_bf16, (None, "tensor"), self.mesh)

            if layer_idx % 10 == 0 or layer_idx == 0:
                logger.info(
                    "Layer %d KV dequant: %s, heads=%d, K=%s V=%s",
                    layer_idx,
                    "per-head" if is_per_head else "uniform",
                    num_kv_heads,
                    k_bf16.shape,
                    v_bf16.shape,
                )

        kv_buffers.clear()
        logger.info("FP8 KV dequantization complete for all layers.")

    def dequant_fused_qkv(
        self,
        fused_qkv_buffers: dict[int, dict],
        layers: list,
        config,
    ):
        """Dequantize per-shard-interleaved fused QKV FP8 weights.

        The FP8 checkpoint was quantized per-TP-shard and concatenated:
          weight: [shard0_QKV | shard1_QKV | ... | shardN_QKV], shape [total_qkv, hidden]
          scale:  [shard0_scale | shard1_scale | ... | shardN_scale], shape [total_blocks, in_blocks]

        Each shard's QKV dim may not be a multiple of block_size, so scale blocks
        within a shard can span K/V boundaries. We must dequantize per-shard first,
        then extract Q/K/V from each shard.

        All dequant math runs on CPU (numpy) to avoid TPU OOM.  Only the final
        bf16 result is ``device_put`` to TPU with TP sharding.

        Args:
            fused_qkv_buffers: dict mapping layer_idx -> {"weight": np.ndarray, "scale": np.ndarray}
            layers: model.layers list
            config: model config with head_dim, v_head_dim, num_attention_heads, etc.
        """
        if not fused_qkv_buffers:
            return

        from jax.sharding import NamedSharding

        head_dim = config.head_dim
        v_head_dim = getattr(config, "v_head_dim", head_dim)
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        quant_cfg = getattr(config, "quantization_config", None)
        block_size = int(quant_cfg.weight_block_size[0]) if quant_cfg else 128

        tp_sharding = NamedSharding(self.mesh, P(None, "tensor"))

        for layer_idx in sorted(fused_qkv_buffers.keys()):
            buf = fused_qkv_buffers[layer_idx]
            fused_weight = buf["weight"]  # numpy, [total_qkv, hidden], FP8
            fused_scale = buf["scale"]  # numpy, [total_blocks, in_blocks], f32

            in_dim = fused_weight.shape[1]  # hidden_size

            # Infer n_shards: find TP size where per-shard blocking matches scale shape
            n_shards, orig_kv_heads = self._infer_qkv_shards(
                fused_weight.shape[0],
                fused_scale.shape[0],
                num_heads,
                num_kv_heads,
                head_dim,
                v_head_dim,
                block_size,
            )

            per_shard_q = (num_heads // n_shards) * head_dim
            per_shard_k = (orig_kv_heads // n_shards) * head_dim
            per_shard_v = (orig_kv_heads // n_shards) * v_head_dim
            per_shard_total = per_shard_q + per_shard_k + per_shard_v
            per_shard_blocks = math.ceil(per_shard_total / block_size)
            padded_rows = per_shard_blocks * block_size
            in_blocks = in_dim // block_size

            if layer_idx % 10 == 0:
                logger.info(
                    "Layer %d: dequant fused QKV FP8 (CPU), n_shards=%d, "
                    "per_shard=%d (Q=%d K=%d V=%d), blocks=%d",
                    layer_idx,
                    n_shards,
                    per_shard_total,
                    per_shard_q,
                    per_shard_k,
                    per_shard_v,
                    per_shard_blocks,
                )

            q_parts, k_parts, v_parts = [], [], []

            for shard_idx in range(n_shards):
                # Extract this shard's weight and scale (numpy slicing)
                w_start = shard_idx * per_shard_total
                shard_w = fused_weight[w_start : w_start + per_shard_total, :]

                s_start = shard_idx * per_shard_blocks
                shard_s = fused_scale[s_start : s_start + per_shard_blocks, :]

                # Pad weight rows to block boundary for dequant
                if per_shard_total < padded_rows:
                    shard_w = np.pad(shard_w, ((0, padded_rows - per_shard_total), (0, 0)))

                # Block dequantize on CPU: [padded_rows, in_dim] × [blocks, in_blocks]
                shard_f = shard_w.astype(np.float32).reshape(
                    per_shard_blocks, block_size, in_blocks, block_size
                )
                shard_s_4d = shard_s[:, None, :, None]
                shard_dequant = (shard_f * shard_s_4d).reshape(padded_rows, in_dim)[
                    :per_shard_total, :
                ]

                # Split into Q, K, V and transpose to model layout [in, out_shard].
                # .T is O(1) numpy view; .copy() releases the shard_dequant reference.
                q_parts.append(shard_dequant[:per_shard_q, :].T.copy())
                k_parts.append(shard_dequant[per_shard_q : per_shard_q + per_shard_k, :].T.copy())
                v_parts.append(shard_dequant[per_shard_q + per_shard_k :, :].T.copy())

            # Free CPU buffers for this layer
            del buf["weight"], buf["scale"]

            # Concat on CPU, convert to bf16, shard to TPU via callback.
            # Uses make_array_from_callback to avoid the allgather that
            # device_put triggers in multi-host (assert_equal OOM).
            # Process Q/K/V sequentially to limit peak CPU memory.
            attn = layers[layer_idx].self_attn
            for proj_name, parts in [
                ("q_proj", q_parts),
                ("k_proj", k_parts),
                ("v_proj", v_parts),
            ]:
                merged = np.ascontiguousarray(
                    np.concatenate(parts, axis=1).astype(ml_dtypes.bfloat16)
                )
                del parts[:]
                # Bind merged via default arg to avoid late-binding closure issue.
                weight = jax.make_array_from_callback(
                    merged.shape,
                    tp_sharding,
                    lambda idx, m=merged: jnp.array(m[idx]),
                )
                del merged
                setattr(
                    attn,
                    proj_name,
                    self.create_bf16_linear(weight, (None, "tensor"), self.mesh),
                )

            if layer_idx == 0:
                q_w = attn.q_proj.weight.value
                k_w = attn.k_proj.weight.value
                v_w = attn.v_proj.weight.value
                logger.info(
                    "Layer 0 dequant result: Q=%s K=%s V=%s",
                    q_w.shape,
                    k_w.shape,
                    v_w.shape,
                )

        # Clean up buffers
        fused_qkv_buffers.clear()
        logger.info("Fused QKV FP8 dequantization complete (CPU numpy path).")

    @staticmethod
    def _infer_qkv_shards(
        total_out_dim: int,
        total_scale_blocks: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        block_size: int,
    ) -> tuple[int, int]:
        """Infer the number of TP shards used during FP8 quantization.

        The config's num_kv_heads may be GQA-replicated (e.g. 32 instead of
        the original 8), so we also try divisors of num_kv_heads to find the
        original value used during per-shard quantization.

        Returns (tp_shards, original_num_kv_heads).
        """
        # Collect candidate kv_heads values: config value and its divisors
        kv_candidates = []
        for d in range(1, num_kv_heads + 1):
            if num_kv_heads % d == 0:
                kv_candidates.append(d)
        # Try config value first, then smaller divisors (descending)
        kv_candidates = sorted(set(kv_candidates), reverse=True)

        # TP candidates: all divisors of num_heads (covers any quantization-time TP)
        tp_candidates = sorted(d for d in range(1, num_heads + 1) if num_heads % d == 0)

        for orig_kv in kv_candidates:
            for tp in tp_candidates:
                if orig_kv % tp != 0:
                    continue
                per_shard = (
                    (num_heads // tp) * head_dim
                    + (orig_kv // tp) * head_dim
                    + (orig_kv // tp) * v_head_dim
                )
                per_shard_blocks = math.ceil(per_shard / block_size)
                if per_shard_blocks * tp == total_scale_blocks and per_shard * tp == total_out_dim:
                    if orig_kv != num_kv_heads:
                        logger.info(
                            "Inferred original num_kv_heads=%d (config has %d), tp=%d",
                            orig_kv,
                            num_kv_heads,
                            tp,
                        )
                    return tp, orig_kv
        raise ValueError(
            f"Cannot infer QKV shard count: out_dim={total_out_dim}, "
            f"scale_blocks={total_scale_blocks}, num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, "
            f"v_head_dim={v_head_dim}, block_size={block_size}"
        )

    def _normalize_physical_to_logical_map(
        self,
        physical_to_logical_map: np.ndarray | None,
        num_logical_experts: int,
        context: str,
    ) -> np.ndarray | None:
        if physical_to_logical_map is None:
            return None

        map_np = np.asarray(physical_to_logical_map, dtype=np.int64)
        if map_np.ndim != 1:
            raise ValueError(
                f"{context}: expected 1D physical_to_logical_map, got shape={map_np.shape}"
            )
        if map_np.size == 0:
            raise ValueError(f"{context}: physical_to_logical_map is empty")

        min_idx = int(np.min(map_np))
        max_idx = int(np.max(map_np))
        if min_idx < 0 or max_idx >= num_logical_experts:
            raise ValueError(
                f"{context}: invalid physical_to_logical_map range [{min_idx}, {max_idx}] "
                f"for num_logical_experts={num_logical_experts}"
            )

        quant_cfg = getattr(self.model_config, "quantization_config", None)
        is_static_quant = quant_cfg is not None and quant_cfg.is_static_checkpoint
        log_fn = logger.info if is_static_quant else logger.debug
        sample = map_np[: min(10, map_np.size)].tolist()
        log_fn(
            "%s: p2l_map physical=%d logical=%d unique=%d sample=%s",
            context,
            map_np.size,
            num_logical_experts,
            np.unique(map_np).size,
            sample,
        )
        return map_np

    def _maybe_convert_epmoe_scale_for_kernel(
        self,
        weight: jax.Array,
        model_param: nnx.Variable,
        target_path: str,
    ) -> jax.Array:
        """Convert offline EPMoE/FusedEPMoE scales into kernel-ready 4D layout.

        Offline checkpoints may store MoE scales in one of several compact
        layouts, for example:

        - per-channel: ``[E, out_dim]``
        - block-channel: ``[E, out_dim, k_blocks]`` or ``[E, k_blocks, out_dim]``
        - 2D block quant: ``[E, out_blocks, k_blocks]``

        The runtime GMM kernel consumes ``[E, k_blocks, 1, out_dim]``.
        The FusedEPMoE kernel consumes ``[E, k_blocks, 1, out_groups_padded]``.

        This helper performs the cheap layout conversion during weight loading
        so the forward path does not need to reinterpret checkpoint tensors.
        """
        # Match both EPMoE (wi_0_scale etc.) and FusedEPMoE (w1_scale etc.)
        if not target_path.endswith(
            ("wi_0_scale", "wi_1_scale", "wo_scale", "w1_scale", "w2_scale", "w3_scale")
        ):
            return weight

        if weight.ndim == 4 or model_param.value.ndim != 4:
            return weight

        param_shape = model_param.value.shape
        num_experts = param_shape[0]

        # --- FusedEPMoE legacy 2D block-wise placeholder ---
        # Older placeholders may use (E, K_groups, N_groups, 1).
        if param_shape[3] == 1 and param_shape[2] > 1 and weight.ndim == 3:
            return weight[..., None]

        # --- EPMoE / GMM path (also FusedEPMoE 1D sub-channel) ---
        if param_shape[2] != 1:
            return weight
        num_experts, k_blocks, _, out_dim = param_shape
        if weight.ndim == 2 and weight.shape == (num_experts, out_dim):
            return weight[:, None, None, :]

        if weight.ndim != 3:
            return weight

        quant_cfg = getattr(self.model_config, "quantization_config", None)
        weight_block_size = getattr(quant_cfg, "weight_block_size", None)
        block_size_out = None
        if isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2:
            block_size_out = int(weight_block_size[0])

        is_fused_scale = target_path.endswith(("w1_scale", "w2_scale", "w3_scale"))
        if is_fused_scale and block_size_out is not None and block_size_out > 0:
            expected_out_blocks = (out_dim + block_size_out - 1) // block_size_out
            if weight.shape == (num_experts, k_blocks, expected_out_blocks):
                logger.info(
                    "Expanding fused MoE 2D scale %s from %s to fast kernel layout %s",
                    target_path,
                    weight.shape,
                    model_param.value.shape,
                )
                weight = jnp.repeat(weight, block_size_out, axis=2)[..., :out_dim]
                return weight[:, :, None, :]
            if weight.shape == (num_experts, expected_out_blocks, k_blocks):
                logger.info(
                    "Transposing+expanding fused MoE 2D scale %s from %s to fast kernel layout %s",
                    target_path,
                    weight.shape,
                    model_param.value.shape,
                )
                weight = jnp.transpose(weight, (0, 2, 1))
                weight = jnp.repeat(weight, block_size_out, axis=2)[..., :out_dim]
                return weight[:, :, None, :]

        if weight.shape == (num_experts, out_dim, k_blocks):
            return jnp.expand_dims(jnp.transpose(weight, (0, 2, 1)), axis=2)

        if weight.shape == (num_experts, k_blocks, out_dim):
            return weight[:, :, None, :]

        # FusedEPMoE 2D block-quant checkpoints (e.g., MiMo) often store scales
        # compactly as [E, K_blocks, N_blocks] or [E, N_blocks, K_blocks], while
        # the fused kernel expects [E, K_blocks, 1, out_groups_padded].
        if is_fused_scale and weight.ndim == 3:
            if weight.shape[0] == num_experts and weight.shape[1] == k_blocks:
                n_groups = weight.shape[2]
                if n_groups <= out_dim:
                    if n_groups < out_dim:
                        logger.info(
                            "Padding fused MoE scale %s from %s to kernel layout %s",
                            target_path,
                            weight.shape,
                            model_param.value.shape,
                        )
                        weight = jnp.pad(weight, ((0, 0), (0, 0), (0, out_dim - n_groups)))
                    return weight[:, :, None, :]
            if weight.shape[0] == num_experts and weight.shape[2] == k_blocks:
                n_groups = weight.shape[1]
                if n_groups <= out_dim:
                    logger.info(
                        "Transposing fused MoE scale %s from %s to kernel layout %s",
                        target_path,
                        weight.shape,
                        model_param.value.shape,
                    )
                    weight = jnp.transpose(weight, (0, 2, 1))
                    if n_groups < out_dim:
                        weight = jnp.pad(weight, ((0, 0), (0, 0), (0, out_dim - n_groups)))
                    return weight[:, :, None, :]

        if not (isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2):
            return weight

        block_size_out = int(weight_block_size[0])
        if block_size_out <= 0:
            return weight

        expected_out_blocks = (out_dim + block_size_out - 1) // block_size_out
        if weight.shape != (num_experts, expected_out_blocks, k_blocks):
            return weight

        logger.info(
            "Converting offline EPMoE scale %s from shape %s to GMM layout %s",
            target_path,
            weight.shape,
            model_param.value.shape,
        )
        out_block_ids = np.arange(out_dim, dtype=np.int32) // block_size_out
        scale_per_out = jnp.take(weight, jnp.asarray(out_block_ids), axis=1)
        return jnp.expand_dims(jnp.transpose(scale_per_out, (0, 2, 1)), axis=2)

    def _maybe_expand_linear_block_scale(
        self,
        weight: jax.Array,
        model_param: nnx.Variable,
        target_path: str,
    ) -> jax.Array:
        """Expand 2D block-quant scale [out_blocks, in_blocks] to 3D [in_blocks, 1, n_out] at load time."""
        if not target_path.endswith("weight_scale"):
            return weight

        # Only convert when checkpoint has 2D scale and model expects 3D.
        if weight.ndim != 2 or model_param.value.ndim != 3:
            return weight

        # Model param shape: [in_blocks, 1, n_out]
        if model_param.value.shape[1] != 1:
            return weight

        quant_cfg = getattr(self.model_config, "quantization_config", None)
        weight_block_size = getattr(quant_cfg, "weight_block_size", None)
        if not (isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2):
            return weight

        block_size_out = int(weight_block_size[0])
        if block_size_out <= 0:
            return weight

        from sgl_jax.srt.kernels.quantized_matmul.blockwise_utils import (
            expand_block_scale,
        )

        n_out = int(model_param.value.shape[2])
        logger.info(
            "Expanding linear block-quant scale %s from %s to kernel-ready layout [%d, 1, %d]",
            target_path,
            weight.shape,
            weight.shape[1],
            n_out,
        )
        return expand_block_scale(weight, n_out, block_size_out)

    def _scan_weight_info(self) -> dict[str, list[dict]]:
        """
        Scan all safetensors files to build a mapping from HF key to file info.
        """
        # 1. Host 0 does the heavy lifting (Scanning)
        if jax.process_index() == 0:
            model_path = self.model_config.model_path
            weights_files = glob.glob(os.path.join(model_path, "*.safetensors"))

            if len(weights_files) == 0:
                raise RuntimeError(f"Cannot find any *.safetensors files in {model_path}")

            weights_files.sort()
            weight_info = {}

            logger.info(
                "Scanning metadata for %s model files (single host only)...", len(weights_files)
            )
            # Use tqdm only on master to avoid log spam
            iterator = tqdm(weights_files, desc="Scanning Metadata", unit="file")

            for st_file in iterator:
                # Parse safetensors header for byte offsets (for bulk MoE reads)
                with open(st_file, "rb") as raw_f:
                    header_size = struct.unpack("<Q", raw_f.read(8))[0]
                    raw_header = json.loads(raw_f.read(header_size))
                data_section_offset = 8 + header_size

                with safe_open(st_file, framework="flax", device="cpu") as f:
                    for key in f.keys():  # noqa: SIM118
                        slice_info = f.get_slice(key)
                        info = {
                            "file": st_file,
                            "shape": tuple(slice_info.get_shape()),
                            "dtype": slice_info.get_dtype(),
                        }
                        # Add byte offset info for direct reads
                        if key in raw_header:
                            offsets = raw_header[key].get("data_offsets")
                            if offsets:
                                info["byte_offset"] = data_section_offset + offsets[0]
                                info["byte_size"] = offsets[1] - offsets[0]
                        if key not in weight_info:
                            weight_info[key] = []
                        weight_info[key].append(info)

            # Serialize the result
            serialized_data = pickle.dumps(weight_info)
            # Convert to uint8 array for JAX broadcasting
            data_np = np.frombuffer(serialized_data, dtype=np.uint8)
            data_len = np.array(len(data_np), dtype=np.int32)
        else:
            # Other hosts just wait
            logger.info("Waiting for metadata broadcast from other host...")
            data_len = np.array(0, dtype=np.int32)
            data_np = None

        # 2. Broadcast the length of the data first
        data_len = multihost_utils.broadcast_one_to_all(
            data_len, is_source=(jax.process_index() == 0)
        )

        # 3. Prepare buffer on receivers
        if jax.process_index() != 0:
            data_np = np.empty(data_len.item(), dtype=np.uint8)

        # 4. Broadcast the actual serialized data
        synced_data = multihost_utils.broadcast_one_to_all(
            data_np, is_source=(jax.process_index() == 0)
        )

        # 5. Deserialize
        synced_bytes = np.array(synced_data).tobytes()
        weight_info = pickle.loads(synced_bytes)

        if jax.process_index() != 0:
            logger.info("Metadata received. Total keys: %s", len(weight_info))

        return weight_info

    def _create_lazy_tensors(
        self,
        hf_key: str,
        infos: list[dict],
        file_manager: SequentialSafetensorManager,
        target_sharding: jax.sharding.NamedSharding = None,
    ) -> list[jax.Array]:
        """
        Create a list of JAX arrays that lazy load data from safetensors via callback.
        Supports 'Global Loading' via target_sharding to avoid redundant I/O.
        """
        lazy_arrays = []

        for info in infos:
            shape = info["shape"]
            st_dtype = info["dtype"]

            dtype_map = {
                "BF16": jnp.bfloat16,
                "F16": jnp.float16,
                "F32": jnp.float32,
                "I64": jnp.int64,
                "I32": jnp.int32,
                "BOOL": jnp.bool_,
                "F8_E4M3": jnp.float8_e4m3fn,
                "F8_E5M2": jnp.float8_e5m2,
            }
            target_dtype = dtype_map.get(st_dtype, jnp.float32)

            filename = info["file"]

            if target_sharding is not None:
                # Load only what this host needs (Global Loading)
                sharding = target_sharding
            else:
                # Fallback: Load full tensor on every host (Replicated)
                sharding = jax.sharding.NamedSharding(self.mesh, P())

            def _make_load_slice(fname=filename, fm=file_manager, target_dtype=target_dtype):
                def _load_slice(index):
                    f = fm.get_handle(fname)
                    data = f.get_slice(hf_key)[index]
                    return _view_as_fp8_if_needed(data, target_dtype)

                return _load_slice

            lazy_array = jax.make_array_from_callback(shape, sharding, _make_load_slice()).astype(
                target_dtype
            )

            lazy_arrays.append(lazy_array)

        return lazy_arrays

    def _create_split_lazy_tensor(
        self,
        hf_key: str,
        infos: list[dict],
        file_manager: SequentialSafetensorManager,
        concat_axis: int,
        target_sharding: jax.sharding.NamedSharding = None,
    ) -> jax.Array:
        """
        Lazy loader for TP-Split weights (e.g., Grok Attention/MLP).
        Instead of loading ALL shards on EVERY host, it calculates overlap
        and only reads the specific file(s) containing the requested slice.
        """
        # 1. Build the "Map": Calculate start/end offsets for each file
        # Sort by filename to ensure correct order (part-00001, part-00002...)
        sorted_infos = sorted(infos, key=lambda x: x["file"])

        cumulative_start = 0
        file_intervals = []  # List of (start, end, info)

        # Assume all shards have same shape except on concat_axis
        base_shape = list(sorted_infos[0]["shape"])

        for info in sorted_infos:
            shape = info["shape"]
            length = shape[concat_axis]
            start = cumulative_start
            end = start + length
            file_intervals.append((start, end, info))
            cumulative_start = end

        # 2. Determine Global Shape
        global_shape = list(base_shape)
        global_shape[concat_axis] = cumulative_start
        global_shape = tuple(global_shape)

        st_dtype = sorted_infos[0]["dtype"]
        dtype_map = {
            "BF16": jnp.bfloat16,
            "F16": jnp.float16,
            "F32": jnp.float32,
            "I64": jnp.int64,
            "I32": jnp.int32,
            "BOOL": jnp.bool_,
            "F8_E4M3": jnp.float8_e4m3fn,
            "F8_E5M2": jnp.float8_e5m2,
        }
        target_dtype = dtype_map.get(st_dtype, jnp.float32)

        if target_sharding is None:
            sharding = jax.sharding.NamedSharding(self.mesh, P())
        else:
            sharding = target_sharding

        # 3. Define Smart Stitching Callback
        def _smart_load_slice(index):
            # index is the slice required by JAX.
            # We need to intersect this slice with the physical file intervals.
            slice_on_axis = index[concat_axis]

            # Normalize slice
            req_start, req_stop, req_step = slice_on_axis.indices(global_shape[concat_axis])
            assert req_step == 1, "Strided access not supported in split loader yet"

            collected_chunks = []

            for f_start, f_end, info in file_intervals:
                # Calculate Intersection: [req_start, req_stop) AND [f_start, f_end)
                intersect_start = max(req_start, f_start)
                intersect_end = min(req_stop, f_end)

                if intersect_start < intersect_end:
                    local_start = intersect_start - f_start
                    local_end = intersect_end - f_start

                    # Construct read index for this file
                    file_read_index = list(index)
                    file_read_index[concat_axis] = slice(local_start, local_end)
                    file_read_index = tuple(file_read_index)

                    # Read directly
                    f = file_manager.get_handle(info["file"])
                    chunk = f.get_slice(hf_key)[file_read_index]
                    collected_chunks.append(chunk)

            if not collected_chunks:
                return np.zeros((0,) * len(global_shape), dtype=target_dtype)

            if len(collected_chunks) == 1:
                # Perfect match (1-to-1 mapping), no copy needed
                result = collected_chunks[0]
            else:
                # Cross-file boundary (rare if TP matches), needs stitching
                result = np.concatenate(collected_chunks, axis=concat_axis)
            return _view_as_fp8_if_needed(result, target_dtype)

        return jax.make_array_from_callback(global_shape, sharding, _smart_load_slice).astype(
            target_dtype
        )

    def _create_stacked_split_moe_lazy_tensor(
        self,
        expected_hf_keys: list[str],
        weight_infos: dict[str, list[dict]],
        file_manager: SequentialSafetensorManager,
        concat_axis: int,
        do_transpose: bool = False,
        target_sharding: jax.sharding.NamedSharding = None,
        physical_to_logical_map: np.ndarray | None = None,
    ) -> jax.Array:
        """
        Lazy loader for TP-Split MOE weights (e.g., Grok MOE).
        """
        num_logical_experts = len(expected_hf_keys)
        physical_to_logical_map = self._normalize_physical_to_logical_map(
            physical_to_logical_map=physical_to_logical_map,
            num_logical_experts=num_logical_experts,
            context="split_moe_loader",
        )
        num_physical_experts = (
            len(physical_to_logical_map)
            if physical_to_logical_map is not None
            else num_logical_experts
        )

        # 1. Build file intervals for each expert
        expert_file_intervals = []
        expert_global_shapes = []

        first_hf_key = expected_hf_keys[0]
        first_infos = weight_infos[first_hf_key]
        sorted_first_infos = sorted(first_infos, key=lambda x: x["file"])

        st_dtype = sorted_first_infos[0]["dtype"]
        dtype_map = {
            "BF16": jnp.bfloat16,
            "F16": jnp.float16,
            "F32": jnp.float32,
            "I64": jnp.int64,
            "I32": jnp.int32,
            "BOOL": jnp.bool_,
            "F8_E4M3": jnp.float8_e4m3fn,
            "F8_E5M2": jnp.float8_e5m2,
        }
        target_dtype = dtype_map.get(st_dtype, jnp.float32)

        for hf_key in expected_hf_keys:
            infos = weight_infos[hf_key]
            sorted_infos = sorted(infos, key=lambda x: x["file"])
            cumulative_start = 0
            file_intervals = []
            base_shape = list(sorted_infos[0]["shape"])
            for info in sorted_infos:
                shape = info["shape"]
                length = shape[concat_axis]
                start = cumulative_start
                end = start + length
                file_intervals.append((start, end, info))
                cumulative_start = end
            global_shape = list(base_shape)
            global_shape[concat_axis] = cumulative_start
            expert_file_intervals.append(file_intervals)
            expert_global_shapes.append(tuple(global_shape))

        single_expert_shape = expert_global_shapes[0]
        stacked_shape = (num_physical_experts, *single_expert_shape)
        sharding = target_sharding or jax.sharding.NamedSharding(self.mesh, P())

        def _load_single_expert_slice(expert_idx, inner_index):
            hf_key = expected_hf_keys[expert_idx]
            file_intervals = expert_file_intervals[expert_idx]
            expert_shape = expert_global_shapes[expert_idx]
            slice_on_axis = inner_index[concat_axis]
            req_start, req_stop, req_step = slice_on_axis.indices(expert_shape[concat_axis])
            assert req_step == 1
            collected_chunks = []
            for f_start, f_end, info in file_intervals:
                intersect_start = max(req_start, f_start)
                intersect_end = min(req_stop, f_end)
                if intersect_start < intersect_end:
                    file_read_index = list(inner_index)
                    file_read_index[concat_axis] = slice(
                        intersect_start - f_start, intersect_end - f_start
                    )
                    f = file_manager.get_handle(info["file"])
                    chunk = f.get_slice(hf_key)[tuple(file_read_index)]
                    collected_chunks.append(chunk)
            if not collected_chunks:
                return np.zeros((0,) * len(expert_shape), dtype=target_dtype)
            if len(collected_chunks) > 1:
                result = np.concatenate(collected_chunks, axis=concat_axis)
            else:
                result = collected_chunks[0]
            result = _view_as_fp8_if_needed(result, target_dtype)
            return result

        MAX_WORKERS = 128

        def _load_stacked_slice(index):
            expert_slice = index[0]
            inner_slice = index[1:]
            start, stop, step = expert_slice.indices(num_physical_experts)
            physical_indices = list(range(start, stop, step))
            if not physical_indices:
                return np.zeros((0, *[1] * len(inner_slice)), dtype=target_dtype)

            if physical_to_logical_map is not None:
                logical_indices = [int(physical_to_logical_map[p]) for p in physical_indices]
                if physical_indices[0] == 0:
                    sample_size = min(10, len(physical_indices))
                    sample_map = {
                        p: logical_indices[i] for i, p in enumerate(physical_indices[:sample_size])
                    }
                    logger.debug("Cloning split-experts map (sample): %s", sample_map)
            else:
                logical_indices = physical_indices

            # Build task list: (logical_idx, list of physical positions that need it)
            logical_to_positions = {}
            for phys_pos, log_idx in enumerate(logical_indices):
                if log_idx not in logical_to_positions:
                    logical_to_positions[log_idx] = []
                logical_to_positions[log_idx].append(phys_pos)

            # Pre-load first expert to determine shape
            first_log_idx = logical_indices[0]
            first_data = _load_single_expert_slice(first_log_idx, tuple(inner_slice))

            out_array = np.empty((len(physical_indices), *first_data.shape), dtype=target_dtype)
            for pos in logical_to_positions[first_log_idx]:
                out_array[pos] = first_data

            # Load remaining unique experts in parallel and fill positions
            remaining_logical = [
                log_idx for log_idx in logical_to_positions if log_idx != first_log_idx
            ]

            def load_and_fill_expert(log_idx):
                data = _load_single_expert_slice(log_idx, tuple(inner_slice))
                for pos in logical_to_positions[log_idx]:
                    out_array[pos] = data

            if remaining_logical:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    list(executor.map(load_and_fill_expert, remaining_logical))

            return out_array

        result = jax.make_array_from_callback(stacked_shape, sharding, _load_stacked_slice)
        if result.dtype != target_dtype:
            result = result.astype(target_dtype)
        if do_transpose and result.ndim >= 3:
            result = jnp.transpose(result, (0, 2, 1))
        return result

    def _create_stacked_moe_lazy_tensor(
        self,
        expected_hf_keys: list[str],
        weight_info: dict,
        file_manager: SequentialSafetensorManager,
        do_transpose: bool = False,
        target_sharding: jax.sharding.NamedSharding = None,
        physical_to_logical_map: np.ndarray | None = None,
    ) -> jax.Array:
        first_key = expected_hf_keys[0]
        info = weight_info[first_key][0]
        single_expert_shape = info["shape"]
        st_dtype = info["dtype"]

        dtype_map = {
            "BF16": jnp.bfloat16,
            "F16": jnp.float16,
            "F32": jnp.float32,
            "I64": jnp.int64,
            "I32": jnp.int32,
            "BOOL": jnp.bool_,
            "F8_E4M3": jnp.float8_e4m3fn,
            "F8_E5M2": jnp.float8_e5m2,
        }
        target_dtype = dtype_map.get(st_dtype, jnp.float32)

        num_logical_experts = len(expected_hf_keys)
        physical_to_logical_map = self._normalize_physical_to_logical_map(
            physical_to_logical_map=physical_to_logical_map,
            num_logical_experts=num_logical_experts,
            context="moe_loader",
        )
        if physical_to_logical_map is not None:
            num_physical_experts = len(physical_to_logical_map)
        else:
            num_physical_experts = num_logical_experts

        # Detect whether we can defer transpose to TPU instead of doing it
        # on CPU during loading. This is possible when do_transpose=True but
        # the weight dimensions (all dims after expert dim) are unsharded,
        # meaning each callback loads the full tensor. Deferring avoids the
        # costly strided memcpy from np.transpose on non-contiguous views.
        defer_transpose = False
        if do_transpose and len(single_expert_shape) >= 2 and target_sharding is not None:
            spec = target_sharding.spec
            # spec[0] is expert dim sharding, spec[1:] are weight dims
            weight_dims_unsharded = all(s is None for s in spec[1:])
            if weight_dims_unsharded:
                defer_transpose = True
                logger.info(
                    "MoE defer_transpose=True: will load in HF layout and "
                    "transpose on TPU (shape=%s)",
                    single_expert_shape,
                )

        # Check if bulk raw byte reading is possible (byte offsets available
        # from scan phase). This eliminates per-expert safetensors API calls
        # and reduces GCSFuse round-trips by reading contiguous byte ranges.
        # Skip bulk_read for small tensors (e.g. scales) — safetensors API
        # with cached handles is faster than raw open/seek/read on GCSFuse.
        _expert_elems = 1
        for d in single_expert_shape:
            _expert_elems *= d
        _expert_bytes_est = _expert_elems * (1 if st_dtype.startswith("F8_") else 4)
        _BULK_READ_MIN_BYTES = 1024 * 1024  # 1 MB per expert
        bulk_read = (
            defer_transpose
            and _expert_bytes_est >= _BULK_READ_MIN_BYTES
            and all(
                "byte_offset" in weight_info[expected_hf_keys[i]][0]
                for i in range(min(2, num_logical_experts))
            )
        )
        if bulk_read:
            logger.info(
                "MoE bulk_read=True: using raw byte reads for %d experts",
                num_logical_experts,
            )

        # Map safetensors dtype strings to numpy dtypes for raw reads
        _st_to_np_dtype = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.dtype("V2"),  # 2-byte view, reinterpret later
            "I64": np.int64,
            "I32": np.int32,
            "BOOL": np.bool_,
            "F8_E4M3": np.uint8,
            "F8_E5M2": np.uint8,
        }
        np_read_dtype = _st_to_np_dtype.get(st_dtype, np.uint8)

        if do_transpose and not defer_transpose and len(single_expert_shape) >= 2:
            final_single_shape = list(single_expert_shape)
            final_single_shape[-1], final_single_shape[-2] = (
                final_single_shape[-2],
                final_single_shape[-1],
            )
            final_single_shape = tuple(final_single_shape)
        else:
            final_single_shape = single_expert_shape

        stacked_shape = (num_physical_experts, *final_single_shape)
        sharding = target_sharding or jax.sharding.NamedSharding(self.mesh, P())

        LOAD_WORKERS = int(os.environ.get("SGLANG_MOE_LOAD_WORKERS", "16"))

        _callback_times = []
        _detailed_logged = False

        def _load_stacked_slice(index):
            nonlocal _detailed_logged
            _cb_start = time.monotonic()
            expert_slice = index[0]
            inner_slice = index[1:]

            start, stop, step = expert_slice.indices(num_physical_experts)
            physical_indices = list(range(start, stop, step))
            sliced_num_physical = len(physical_indices)

            if sliced_num_physical == 0:
                return np.zeros((0, *[1] * len(inner_slice)), dtype=target_dtype)

            if physical_to_logical_map is not None:
                logical_indices_to_load = [
                    int(physical_to_logical_map[p]) for p in physical_indices
                ]
                if physical_indices[0] == 0:
                    sample_size = min(10, len(physical_indices))
                    sample_map = {
                        p: logical_indices_to_load[i]
                        for i, p in enumerate(physical_indices[:sample_size])
                    }
                    logger.debug("Cloning experts map (sample): %s", sample_map)
            else:
                logical_indices_to_load = physical_indices

            # --- Load first expert with detailed timing ---
            first_log_idx = logical_indices_to_load[0]
            first_hf_key = expected_hf_keys[first_log_idx]
            first_fname = weight_info[first_hf_key][0]["file"]

            _t0 = time.monotonic()
            first_f = file_manager.get_handle(first_fname)
            _t_handle = time.monotonic() - _t0

            if not do_transpose or defer_transpose:
                _t1 = time.monotonic()
                first_chunk = first_f.get_slice(first_hf_key)[inner_slice]
                _t_getslice = time.monotonic() - _t1
                _t2 = time.monotonic()
                first_chunk = _view_as_fp8_if_needed(first_chunk, target_dtype)
                _t_fp8 = time.monotonic() - _t2
                _t_transpose = 0.0
            else:
                _t1 = time.monotonic()
                data = first_f.get_slice(first_hf_key)[:]
                _t_getslice = time.monotonic() - _t1
                _t2 = time.monotonic()
                data = _view_as_fp8_if_needed(data, target_dtype)
                _t_fp8 = time.monotonic() - _t2
                _t3 = time.monotonic()
                first_chunk = np.transpose(data)[inner_slice]
                _t_transpose = time.monotonic() - _t3

            # Log detailed timing for first callback only
            if not _detailed_logged:
                _detailed_logged = True
                expert_bytes = first_chunk.nbytes
                logger.debug(
                    "MoE callback[0] detail: experts_in_shard=%d "
                    "get_handle=%.4fs get_slice=%.4fs fp8_view=%.4fs "
                    "transpose=%.4fs expert_bytes=%d file=%s",
                    sliced_num_physical,
                    _t_handle,
                    _t_getslice,
                    _t_fp8,
                    _t_transpose,
                    expert_bytes,
                    os.path.basename(first_fname),
                )

            out_shape = (sliced_num_physical, *first_chunk.shape)
            out_array = np.empty(out_shape, dtype=first_chunk.dtype)
            out_array[0] = first_chunk

            logical_to_positions = {}
            for phys_pos in range(1, sliced_num_physical):
                log_idx = logical_indices_to_load[phys_pos]
                if log_idx == first_log_idx:
                    out_array[phys_pos] = first_chunk
                else:
                    logical_to_positions.setdefault(log_idx, []).append(phys_pos)

            # Per-thread timing accumulators
            _thread_stats = {
                "get_handle": [],
                "get_slice": [],
                "fp8": [],
                "transpose": [],
                "copy": [],
            }
            _thread_files = []

            def load_and_fill_expert(args):
                l_idx, positions = args
                hf_k = expected_hf_keys[l_idx]
                fname = weight_info[hf_k][0]["file"]

                t_a = time.monotonic()
                f = file_manager.get_handle(fname)
                t_b = time.monotonic()
                _thread_stats["get_handle"].append(t_b - t_a)

                if not do_transpose or defer_transpose:
                    chunk = f.get_slice(hf_k)[inner_slice]
                    t_c = time.monotonic()
                    _thread_stats["get_slice"].append(t_c - t_b)
                    chunk = _view_as_fp8_if_needed(chunk, target_dtype)
                    t_d = time.monotonic()
                    _thread_stats["fp8"].append(t_d - t_c)
                    _thread_stats["transpose"].append(0.0)
                else:
                    data = f.get_slice(hf_k)[:]
                    t_c = time.monotonic()
                    _thread_stats["get_slice"].append(t_c - t_b)
                    data = _view_as_fp8_if_needed(data, target_dtype)
                    t_d = time.monotonic()
                    _thread_stats["fp8"].append(t_d - t_c)
                    chunk = np.transpose(data)[inner_slice]
                    t_e = time.monotonic()
                    _thread_stats["transpose"].append(t_e - t_d)

                t_copy_start = time.monotonic()
                for pos in positions:
                    out_array[pos] = chunk
                _thread_stats["copy"].append(time.monotonic() - t_copy_start)
                _thread_files.append(os.path.basename(fname))

            if logical_to_positions:
                tasks = list(logical_to_positions.items())
                with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as executor:
                    list(executor.map(load_and_fill_expert, tasks))

            _callback_times.append(time.monotonic() - _cb_start)

            # Log aggregated thread stats for first callback
            if len(_callback_times) == 1 and _thread_stats["get_slice"]:
                n = len(_thread_stats["get_slice"])
                unique_files = len(set(_thread_files))
                logger.debug(
                    "MoE callback[0] threads: n=%d unique_files=%d "
                    "get_handle=%.4fs get_slice(sum=%.2fs avg=%.4fs max=%.4fs) "
                    "fp8=%.4fs transpose(sum=%.2fs avg=%.4fs) copy=%.4fs",
                    n,
                    unique_files,
                    sum(_thread_stats["get_handle"]),
                    sum(_thread_stats["get_slice"]),
                    sum(_thread_stats["get_slice"]) / n,
                    max(_thread_stats["get_slice"]),
                    sum(_thread_stats["fp8"]),
                    sum(_thread_stats["transpose"]),
                    sum(_thread_stats["transpose"]) / n,
                    sum(_thread_stats["copy"]),
                )

            return out_array

        t0 = time.monotonic()
        if bulk_read:
            # Host-level bulk loading: read all experts for local devices at
            # once, then device_put in parallel. This replaces serial
            # make_array_from_callback with a single bulk I/O + parallel
            # device transfers.
            local_devices = sharding.mesh.local_devices
            n_local = len(local_devices)

            # Determine which physical experts each local device needs
            device_assignments = sharding.devices_indices_map(stacked_shape)
            local_assignments = []
            for dev in local_devices:
                idx = device_assignments[dev]
                expert_slice = idx[0]
                s, e, st = expert_slice.indices(num_physical_experts)
                local_assignments.append(list(range(s, e, st)))

            # Collect ALL unique logical expert indices needed by this host
            all_logical = set()
            for phys_indices in local_assignments:
                for p in phys_indices:
                    if physical_to_logical_map is not None:
                        all_logical.add(int(physical_to_logical_map[p]))
                    else:
                        all_logical.add(p)

            # Group by file for bulk reading
            file_groups = {}
            for log_idx in all_logical:
                hf_key = expected_hf_keys[log_idx]
                info = weight_info[hf_key][0]
                fname = info["file"]
                byte_offset = info["byte_offset"]
                file_groups.setdefault(fname, []).append((log_idx, byte_offset, hf_key))

            # Compute per-expert byte size
            expert_nbytes = 1
            for d in single_expert_shape:
                expert_nbytes *= d
            elem_size = 1 if np_read_dtype == np.uint8 else np.dtype(np_read_dtype).itemsize
            expert_nbytes *= elem_size

            # Read all expert data from files (parallel across files)
            expert_data_map = {}  # log_idx -> np.ndarray

            def _bulk_read_file(fname, entries):
                entries.sort(key=lambda e: e[1])
                min_off = entries[0][1]
                max_end = entries[-1][1] + expert_nbytes
                rng = max_end - min_off
                actual = len(entries) * expert_nbytes
                result = {}
                if rng <= actual * 2 and len(entries) > 1:
                    with open(fname, "rb") as f:
                        f.seek(min_off)
                        bulk = f.read(rng)
                    for log_idx, byte_off, hf_key in entries:
                        local_off = byte_off - min_off
                        arr = np.frombuffer(
                            bulk,
                            dtype=np_read_dtype,
                            count=expert_nbytes // elem_size,
                            offset=local_off,
                        ).reshape(single_expert_shape)
                        result[log_idx] = arr.copy()
                else:
                    with open(fname, "rb") as f:
                        for log_idx, byte_off, hf_key in entries:
                            f.seek(byte_off)
                            raw = f.read(expert_nbytes)
                            arr = np.frombuffer(
                                raw,
                                dtype=np_read_dtype,
                            ).reshape(single_expert_shape)
                            result[log_idx] = arr
                return result

            t_io_start = time.monotonic()
            if len(file_groups) > 1:
                with ThreadPoolExecutor(max_workers=len(file_groups)) as ex:
                    futs = {
                        ex.submit(_bulk_read_file, fn, ents): fn for fn, ents in file_groups.items()
                    }
                    for fut in futs:
                        expert_data_map.update(fut.result())
            else:
                for fn, ents in file_groups.items():
                    expert_data_map.update(_bulk_read_file(fn, ents))
            t_io = time.monotonic() - t_io_start
            total_read_mb = sum(v.nbytes for v in expert_data_map.values()) / 1e6

            # Assemble per-device arrays and device_put (parallel)
            t_assemble_start = time.monotonic()
            n_experts_per_shard = len(local_assignments[0])

            def _build_and_put(dev_idx):
                dev = local_devices[dev_idx]
                phys_indices = local_assignments[dev_idx]
                shard = np.empty(
                    (n_experts_per_shard, *single_expert_shape),
                    dtype=np_read_dtype,
                )
                for pos, p in enumerate(phys_indices):
                    log_idx = (
                        int(physical_to_logical_map[p])
                        if physical_to_logical_map is not None
                        else p
                    )
                    shard[pos] = expert_data_map[log_idx]
                shard = _view_as_fp8_if_needed(shard, target_dtype)
                return jax.device_put(shard, dev)

            with ThreadPoolExecutor(max_workers=n_local) as ex:
                per_device_arrays = list(ex.map(_build_and_put, range(n_local)))
            t_assemble = time.monotonic() - t_assemble_start

            result = jax.make_array_from_single_device_arrays(
                stacked_shape,
                sharding,
                per_device_arrays,
            )
            t_callback = time.monotonic() - t0

            logger.debug(
                "MoE host-bulk load: shape=%s dtype=%s "
                "io=%.2fs (%.1f MB, %.1f MB/s) "
                "assemble+put=%.2fs total=%.2fs "
                "experts_read=%d files=%d devices=%d",
                stacked_shape,
                target_dtype,
                t_io,
                total_read_mb,
                total_read_mb / t_io if t_io > 0 else 0,
                t_assemble,
                t_callback,
                len(expert_data_map),
                len(file_groups),
                n_local,
            )
        else:
            # Pre-warm safetensors file handles to avoid cold-start latency
            # during serial callbacks. This is especially important for small
            # tensors (scales) where GCSFuse file-open latency dominates.
            unique_files = set()
            for i in range(num_logical_experts):
                hf_key = expected_hf_keys[i]
                unique_files.add(weight_info[hf_key][0]["file"])
            uncached = [fn for fn in unique_files if fn not in file_manager.handles]
            if uncached:

                def _prewarm(fn):
                    return fn, safe_open(fn, framework="np", device="cpu")

                with ThreadPoolExecutor(max_workers=min(len(uncached), 16)) as ex:
                    for fn, handle in ex.map(_prewarm, uncached):
                        file_manager.handles[fn] = handle

            callback_fn = _load_stacked_slice
            result = jax.make_array_from_callback(
                stacked_shape,
                sharding,
                callback_fn,
            )
            t_callback = time.monotonic() - t0
        t1 = time.monotonic()
        if result.dtype != target_dtype:
            result = result.astype(target_dtype)
        t_astype = time.monotonic() - t1
        # Deferred transpose: data was loaded in HF layout [experts, out, in],
        # now transpose to kernel layout [experts, in, out] on TPU.
        t2 = time.monotonic()
        if defer_transpose and result.ndim >= 3:
            result = jnp.transpose(result, (0, 2, 1))
        t_defer = time.monotonic() - t2
        if _callback_times:
            defer_msg = f" defer_transpose={t_defer:.3f}s" if defer_transpose else ""
            logger.debug(
                "MoE tensor load: shape=%s dtype=%s callbacks=%d "
                "callback_total=%.2fs (min=%.3fs max=%.3fs) "
                "make_array=%.2fs astype=%.2fs workers=%d%s",
                stacked_shape,
                target_dtype,
                len(_callback_times),
                sum(_callback_times),
                min(_callback_times),
                max(_callback_times),
                t_callback,
                t_astype,
                LOAD_WORKERS,
                defer_msg,
            )
        return result

    def load_weights_from_safetensors(
        self,
        weight_mappings: Mapping[str, str | list[str] | WeightMapping],
        safetensors_partition=1,
        dummy=False,
    ):
        """Load weights using JAX lazy evaluation and parallel I/O."""
        params = nnx.state(self.model)

        if dummy or self.dummy_mode:
            self._load_dummy_weights(params, weight_mappings)
            return

        # 1. Build index
        weight_info = self._scan_weight_info()

        regular_mappings = {}
        moe_mappings = {}

        for key, mapping in weight_mappings.items():
            if "*" not in key:
                if key.startswith("__MOE_EXPERTS__"):
                    moe_mappings[key] = mapping
                else:
                    regular_mappings[key] = mapping
            else:
                key_as_regex = re.escape(key).replace(r"\*", r"(.*?)")
                for weight_info_key, _ in weight_info.items():
                    match = re.search(key_as_regex, weight_info_key)
                    if match:
                        matched_parts = match.groups()

                        if isinstance(mapping, str):
                            format_template = mapping.replace("*", "{}")
                            replaced_mapping = format_template.format(*matched_parts)
                        elif isinstance(mapping, list):
                            format_template = mapping[0].replace("*", "{}")
                            replaced_str = format_template.format(*matched_parts)
                            replaced_mapping = [replaced_str, *mapping[1:]]
                        elif isinstance(mapping, tuple):
                            format_template = mapping[0].replace("*", "{}")
                            replaced_str = format_template.format(*matched_parts)
                            replaced_mapping = (replaced_str, *mapping[1:])
                        elif isinstance(mapping, WeightMapping):
                            format_template = mapping.target_path.replace("*", "{}")
                            replaced_path = format_template.format(*matched_parts)
                            replaced_mapping = copy.copy(mapping)
                            replaced_mapping.target_path = replaced_path
                        else:
                            replaced_mapping = mapping

                        if key.startswith("__MOE_EXPERTS__"):
                            moe_mappings[weight_info_key] = replaced_mapping
                        else:
                            regular_mappings[weight_info_key] = replaced_mapping

        logger.info("Starting parallel weight loading via JAX Lazy Loader...")
        quant_cfg = getattr(self.model_config, "quantization_config", None)
        is_static_quant = quant_cfg is not None and quant_cfg.is_static_checkpoint

        with SequentialSafetensorManager() as file_manager:
            # 2. Process Regular Weights (Lazy Pull)
            for hf_key, mapping in tqdm(regular_mappings.items(), desc="Loading Regular Weights"):
                if hf_key not in weight_info:
                    if hf_key == "d2t":
                        logger.warning("Weight %s not found in safetensors index.", hf_key)
                        continue
                    if self._is_excluded_layer_weight(hf_key):
                        logger.debug("Skipping excluded layer weight: %s", hf_key)
                        continue
                    else:
                        logger.warning("No file found for weight: %s", hf_key)
                        continue

                infos = weight_info[hf_key]

                if isinstance(mapping, str | list):
                    mapping = WeightMapping(target_path=mapping)

                is_split_weight = len(infos) > 1 and mapping.concat_axis is not None

                can_optimize = (
                    isinstance(mapping.target_path, str)
                    and not mapping.target_path.startswith("__FUSED_QKV_")
                    and not mapping.target_path.startswith("__KV_")
                    and mapping.reshape is None
                    and mapping.repeat is None  # Check repeat here too!
                    and not mapping.kv_head_padding
                    and not mapping.head_dim_padding
                    and mapping.sharding is not None
                    and hf_key != "d2t"
                )

                if can_optimize:
                    try:
                        if mapping.transpose and len(mapping.sharding) == 2:
                            # Swap: (dim0, dim1) -> (dim1, dim0)
                            sharding_tuple = mapping.sharding[::-1]
                        else:
                            sharding_tuple = mapping.sharding

                        spec = P(*sharding_tuple)
                        final_sharding = jax.sharding.NamedSharding(self.mesh, spec)

                        lazy_weight = None

                        if is_split_weight:
                            lazy_weight = self._create_split_lazy_tensor(
                                hf_key,
                                infos,
                                file_manager,
                                concat_axis=mapping.concat_axis,
                                target_sharding=final_sharding,
                            )
                        else:
                            lazy_arrays = self._create_lazy_tensors(
                                hf_key,
                                infos,
                                file_manager,
                                target_sharding=final_sharding,
                            )
                            lazy_weight = lazy_arrays[0]

                        # Handle multi-dimensional transpose (transpose_axes) or 2D transpose
                        if mapping.transpose_axes is not None:
                            lazy_weight = jnp.transpose(lazy_weight, mapping.transpose_axes)
                        elif mapping.transpose:
                            lazy_weight = jnp.transpose(lazy_weight, (1, 0))

                        if "lm_head" in hf_key and hasattr(
                            self.model_config.hf_config, "output_multiplier_scale"
                        ):
                            lazy_weight = (
                                lazy_weight.astype(jnp.float32)
                                * self.model_config.hf_config.output_multiplier_scale
                            )

                        target_path = mapping.target_path
                        model_param = self._get_param(params, target_path)

                        # Expand 2D block-quant scale to 3D kernel-ready layout.
                        lazy_weight = self._maybe_expand_linear_block_scale(
                            lazy_weight, model_param, target_path
                        )

                        if lazy_weight.dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
                            model_param.value = lazy_weight
                        else:
                            model_param.value = lazy_weight.astype(model_param.value.dtype)

                        mode_str = "Split-Stitch" if is_split_weight else "Direct"
                        logger.debug(
                            "Fast Loading %s -> %s (%s), shape: %s",
                            hf_key,
                            target_path,
                            mode_str,
                            lazy_weight.shape,
                        )
                        continue

                    except Exception as e:
                        logger.warning(
                            "Fast load failed for %s, falling back to slow path. Error: %s",
                            hf_key,
                            str(e),
                        )
                lazy_arrays = self._create_lazy_tensors(
                    hf_key,
                    infos,
                    file_manager,
                    target_sharding=None,
                )

                if len(lazy_arrays) > 1 and mapping.concat_axis is not None:
                    lazy_weight = jnp.concatenate(lazy_arrays, axis=mapping.concat_axis)
                else:
                    lazy_weight = lazy_arrays[0]

                if hf_key == "d2t":
                    base = jnp.arange(lazy_weight.shape[0], dtype=lazy_weight.dtype)
                    hot_ids = (lazy_weight + base).astype(jnp.int32)
                    params["hot_token_ids"].value = hot_ids
                    continue

                self._process_and_assign_weight(params, hf_key, lazy_weight, mapping)

            # 3. Process MoE Weights (Lazy Pull)
            for moe_key, mapping in tqdm(moe_mappings.items(), desc="Loading MoE Weights"):
                expected_hf_keys = mapping.target_path[1:]

                group_complete = True
                is_tp_split = False

                # Validation pass
                for hf_key in expected_hf_keys:
                    if hf_key not in weight_info:
                        if self._is_excluded_layer_weight(hf_key):
                            logger.debug("Skipping excluded MoE expert weight: %s", hf_key)
                        else:
                            logger.warning("MoE expert weight %s not found.", hf_key)
                            raise ValueError(f"MoE expert weight {hf_key} not found.")
                        group_complete = False
                        break

                    infos = weight_info[hf_key]

                    # Check for TP split (Grok style)
                    if mapping.concat_axis is not None:
                        if len(infos) > 1:
                            is_tp_split = True

                        if len(infos) < safetensors_partition:
                            logger.warning(
                                "Incomplete shards for %s: expected %s, found %s",
                                hf_key,
                                safetensors_partition,
                                len(infos),
                            )
                            group_complete = False
                            break

                if not group_complete:
                    continue

                # OPTIMIZATION: Use Stacked Loader if no TP split
                if not is_tp_split and mapping.concat_axis is None:
                    # 1. Pre-construct target sharding
                    if "expert" in mapping.sharding:
                        ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
                        world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get(
                            "tensor", 1
                        )
                        tp_size = world_size // ep_size

                        devices = self.mesh.devices.flatten()
                        # Construct MoE specific mesh
                        moe_mesh = jax.sharding.Mesh(
                            devices.reshape(ep_size, tp_size),
                            axis_names=("expert", "tensor"),
                            axis_types=(
                                jax.sharding.AxisType.Explicit,
                                jax.sharding.AxisType.Explicit,
                            ),
                        )
                        final_sharding = jax.sharding.NamedSharding(moe_mesh, P(*mapping.sharding))
                    else:
                        # Standard Sharding
                        final_sharding = jax.sharding.NamedSharding(self.mesh, P(*mapping.sharding))

                    # 2. Call creator
                    _t_load_start = time.monotonic()
                    stacked_weight = self._create_stacked_moe_lazy_tensor(
                        expected_hf_keys,
                        weight_info,
                        file_manager,
                        do_transpose=mapping.transpose,  # CPU transpose
                        target_sharding=final_sharding,  # Global loading
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )
                    _t_load = time.monotonic() - _t_load_start
                    loaded_shape = stacked_weight.shape

                    if mapping.reshape is not None:
                        stacked_weight = jnp.reshape(stacked_weight, mapping.reshape)

                    if mapping.repeat is not None:
                        axis, times = mapping.repeat
                        stacked_weight = jnp.repeat(stacked_weight, times, axis=axis)

                    # 3. Direct assignment
                    target_path = mapping.target_path[0]
                    model_param = self._get_param(params, target_path)
                    stacked_weight = self._maybe_convert_epmoe_scale_for_kernel(
                        stacked_weight,
                        model_param,
                        target_path,
                    )

                    if is_static_quant and moe_key.endswith("_scale"):
                        logger.debug(
                            "MoE scale debug group=%s target=%s loaded_shape=%s final_shape=%s "
                            "param_shape=%s reshape=%s repeat=%s sharding=%s",
                            moe_key,
                            target_path,
                            loaded_shape,
                            stacked_weight.shape,
                            model_param.value.shape,
                            mapping.reshape,
                            mapping.repeat,
                            mapping.sharding,
                        )

                    try:
                        _t_assign_start = time.monotonic()
                        if stacked_weight.dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
                            model_param.value = stacked_weight
                        else:
                            model_param.value = stacked_weight.astype(model_param.value.dtype)
                        _t_assign = time.monotonic() - _t_assign_start
                        logger.debug(
                            "MoE group %s: load=%.2fs assign=%.2fs total=%.2fs "
                            "shape=%s sharding=%s",
                            moe_key,
                            _t_load,
                            _t_assign,
                            _t_load + _t_assign,
                            loaded_shape,
                            mapping.sharding,
                        )
                    except Exception as e:
                        logger.error(
                            "Failed MoE assign group=%s target=%s loaded_shape=%s final_shape=%s "
                            "param_shape=%s reshape=%s repeat=%s sharding=%s err=%s",
                            moe_key,
                            target_path,
                            loaded_shape,
                            stacked_weight.shape,
                            model_param.value.shape,
                            mapping.reshape,
                            mapping.repeat,
                            mapping.sharding,
                            str(e),
                        )
                        raise

                    if mapping.physical_to_logical_map is not None:
                        num_logical = len(expected_hf_keys)
                        num_physical = len(mapping.physical_to_logical_map)
                        logger.info(
                            "Assigned MoE group %s with redundant experts: %d logical -> %d physical, shape: %s",
                            moe_key,
                            num_logical,
                            num_physical,
                            stacked_weight.shape,
                        )
                    else:
                        logger.info(
                            "Assigned MoE group %s, shape: %s",
                            moe_key,
                            stacked_weight.shape,
                        )
                else:
                    ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
                    if "expert" in mapping.sharding:
                        world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get(
                            "tensor", 1
                        )
                        tp_size = world_size // ep_size
                        devices = self.mesh.devices.flatten()
                        moe_mesh = jax.sharding.Mesh(
                            devices.reshape(ep_size, tp_size),
                            axis_names=("expert", "tensor"),
                            axis_types=(
                                jax.sharding.AxisType.Explicit,
                                jax.sharding.AxisType.Explicit,
                            ),
                        )
                        # Use regular mesh for loading individual expert weights (TP sharding only)
                        final_sharding = jax.sharding.NamedSharding(moe_mesh, P(*mapping.sharding))
                    else:
                        final_sharding = jax.sharding.NamedSharding(self.mesh, P(*mapping.sharding))

                    expert_weights = self._create_stacked_split_moe_lazy_tensor(
                        expected_hf_keys,
                        weight_info,
                        file_manager,
                        concat_axis=mapping.concat_axis,
                        do_transpose=mapping.transpose,
                        target_sharding=final_sharding,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )
                    loaded_shape = expert_weights.shape

                    if mapping.reshape is not None:
                        expert_weights = jnp.reshape(expert_weights, mapping.reshape)

                    if mapping.repeat is not None:
                        axis, times = mapping.repeat
                        expert_weights = jnp.repeat(expert_weights, times, axis=axis)

                    target_path = mapping.target_path[0]
                    model_param = self._get_param(params, target_path)
                    expert_weights = self._maybe_convert_epmoe_scale_for_kernel(
                        expert_weights,
                        model_param,
                        target_path,
                    )

                    if is_static_quant and moe_key.endswith("_scale"):
                        logger.debug(
                            "Split-MoE scale debug group=%s target=%s loaded_shape=%s final_shape=%s "
                            "param_shape=%s reshape=%s repeat=%s sharding=%s",
                            moe_key,
                            target_path,
                            loaded_shape,
                            expert_weights.shape,
                            model_param.value.shape,
                            mapping.reshape,
                            mapping.repeat,
                            mapping.sharding,
                        )

                    try:
                        if expert_weights.dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
                            model_param.value = expert_weights
                        else:
                            model_param.value = expert_weights.astype(model_param.value.dtype)
                    except Exception as e:
                        logger.error(
                            "Failed Split-MoE assign group=%s target=%s loaded_shape=%s final_shape=%s "
                            "param_shape=%s reshape=%s repeat=%s sharding=%s err=%s",
                            moe_key,
                            target_path,
                            loaded_shape,
                            expert_weights.shape,
                            model_param.value.shape,
                            mapping.reshape,
                            mapping.repeat,
                            mapping.sharding,
                            str(e),
                        )
                        raise

                    logger.info(
                        "Assigned MoE group %s (Grok Split-Stitch), shape: %s",
                        moe_key,
                        expert_weights.shape,
                    )

        nnx.update(self.model, params)
        logger.info("All weights loaded successfully.")

    def _load_dummy_weights(
        self,
        params: nnx.State,
        weight_mappings: dict[str, str | list[str] | WeightMapping],
        seed: int = 1234,
    ):
        logger.info("Generating dummy weights with proper sharding from weight mappings")
        regular_mappings = {}
        moe_mappings = {}

        for hf_key, mapping in weight_mappings.items():
            if hf_key.startswith("__MOE_EXPERTS__"):
                moe_mappings[hf_key] = mapping
            else:
                regular_mappings[hf_key] = mapping

        for hf_key, mapping in regular_mappings.items():

            if isinstance(mapping, str | list):
                mapping = WeightMapping(target_path=mapping)

            target_path = (
                mapping.target_path
                if isinstance(mapping.target_path, str)
                else mapping.target_path[0]
            )

            try:
                model_param = self._get_param(params, target_path)
            except (KeyError, AttributeError, ValueError):
                logger.debug("Skip dummy weight for %s (parameter not found)", target_path)
                continue

            shape = model_param.value.shape
            dtype = model_param.value.dtype

            sharding_spec = P(*mapping.sharding) if mapping.sharding else P()
            sharding = jax.sharding.NamedSharding(self.mesh, sharding_spec)

            def make_shard(indices, shape=shape, dtype=dtype):
                shard_shape = []
                for dim_size, idx in zip(shape, indices):
                    if isinstance(idx, slice):
                        start, stop, step = idx.indices(dim_size)
                        assert step == 1, f"Non-unit step not supported: {idx}"
                        shard_shape.append(stop - start)
                    else:
                        shard_shape.append(1)
                shard_shape = tuple(shard_shape)

                rng = np.random.default_rng(seed)
                if jnp.issubdtype(dtype, jnp.floating):
                    if dtype == jnp.bfloat16:
                        gen_dtype = np.float32
                    else:
                        gen_dtype = {
                            jnp.float16: np.float16,
                            jnp.float32: np.float32,
                            jnp.float64: np.float64,
                        }.get(dtype, np.float32)
                    arr_np = rng.uniform(-1e-3, 1e-3, size=shard_shape).astype(gen_dtype)
                    return jnp.asarray(arr_np, dtype=dtype)
                else:
                    return jnp.zeros(shard_shape, dtype=dtype)

            dummy_weight = jax.make_array_from_callback(shape, sharding, make_shard)
            model_param.value = dummy_weight
            logger.debug(
                "Generated dummy weight for %s, shape=%s, sharding=%s",
                target_path,
                shape,
                sharding_spec,
            )

        for moe_key, mapping in moe_mappings.items():
            if isinstance(mapping, str | list):
                mapping = WeightMapping(target_path=mapping)

            target_path = mapping.target_path[0]

            try:
                model_param = self._get_param(params, target_path)
            except (KeyError, AttributeError, ValueError):
                logger.debug("Skip dummy MOE weight for %s (parameter not found)", target_path)
                continue

            full_shape = model_param.value.shape
            num_experts = full_shape[0]
            expert_weight_shape = full_shape[1:]
            dtype = model_param.value.dtype

            collected_weights = []
            for expert_idx in range(num_experts):
                if mapping.sharding and "expert" in mapping.sharding:
                    expert_sharding_tuple = tuple(s for s in mapping.sharding if s != "expert")
                else:
                    expert_sharding_tuple = mapping.sharding

                expert_sharding_spec = P(*expert_sharding_tuple) if expert_sharding_tuple else P()
                expert_sharding = jax.sharding.NamedSharding(self.mesh, expert_sharding_spec)

                def make_expert_shard(
                    indices, weight_shape=expert_weight_shape, weight_dtype=dtype, idx=expert_idx
                ):
                    shard_shape = []
                    for dim_size, idx_val in zip(weight_shape, indices):
                        if isinstance(idx_val, slice):
                            start, stop, step = idx_val.indices(dim_size)
                            assert step == 1, f"Non-unit step not supported: {idx_val}"
                            shard_shape.append(stop - start)
                        else:
                            shard_shape.append(1)
                    shard_shape = tuple(shard_shape)

                    rng = np.random.default_rng(seed + idx)
                    if jnp.issubdtype(weight_dtype, jnp.floating):
                        gen_dtype = np.float32 if weight_dtype == jnp.bfloat16 else weight_dtype
                        arr_np = rng.uniform(-1e-3, 1e-3, size=shard_shape).astype(gen_dtype)
                        return jnp.asarray(arr_np, dtype=weight_dtype)
                    else:
                        return jnp.zeros(shard_shape, dtype=weight_dtype)

                expert_weight = jax.make_array_from_callback(
                    expert_weight_shape, expert_sharding, make_expert_shard
                )
                collected_weights.append(expert_weight)

            stacked_weight = jnp.stack(collected_weights, axis=0)

            if mapping.sharding and "expert" in mapping.sharding:
                ep_size = getattr(self.model_config.hf_config, "ep_size", 1)
                if ep_size > 1:
                    world_size = self.mesh.shape.get("data", 1) * self.mesh.shape.get("tensor", 1)
                    tp_size = world_size // ep_size

                    devices = self.mesh.devices.flatten()
                    moe_mesh = jax.sharding.Mesh(
                        devices.reshape(ep_size, tp_size), axis_names=("expert", "tensor")
                    )
                    final_sharding_spec = P(*mapping.sharding)
                    final_sharding = jax.sharding.NamedSharding(moe_mesh, final_sharding_spec)
                else:
                    final_sharding_spec = P(*mapping.sharding)
                    final_sharding = jax.sharding.NamedSharding(self.mesh, final_sharding_spec)
            else:
                final_sharding_spec = P(*mapping.sharding) if mapping.sharding else P()
                final_sharding = jax.sharding.NamedSharding(self.mesh, final_sharding_spec)

            sharded_weight = jax.device_put(stacked_weight, final_sharding)
            model_param.value = sharded_weight.astype(dtype)

            logger.debug(
                "Generated dummy MOE weight for %s, shape=%s, num_experts=%s, sharding=%s",
                target_path,
                full_shape,
                num_experts,
                mapping.sharding,
            )

        nnx.update(self.model, params)
        logger.info("Dummy weights generated successfully!")

    def _process_and_assign_weight(
        self,
        params: nnx.State,
        hf_key: str,
        hf_weight: jax.Array,
        mapping: WeightMapping,
    ):
        processed_weight = hf_weight

        # Handle multi-dimensional transpose (transpose_axes) or 2D transpose
        if mapping.transpose_axes is not None and not hf_key.endswith(".bias"):
            processed_weight = jnp.transpose(processed_weight, mapping.transpose_axes)
        elif mapping.transpose and not hf_key.endswith(".bias"):
            processed_weight = jnp.transpose(processed_weight, (1, 0))

        if isinstance(mapping.target_path, list):
            self._handle_split_weight(params, hf_key, processed_weight, mapping)
        else:
            self._handle_single_weight(params, hf_key, processed_weight, mapping)

    def _handle_single_weight(
        self, params: nnx.State, hf_key: str, weight: jax.Array, mapping: WeightMapping
    ):
        jax_path = mapping.target_path
        processed_weight = weight

        # Handle fused KV buffer storage (used by MiMo-V2-Flash per-head dequant).
        # target_path like "__KV_K_WEIGHT__42" stores into model._kv_buffers.
        if jax_path.startswith("__KV_"):
            if hasattr(self.model, "_kv_buffers"):
                layer_idx = int(jax_path.rsplit("__", 1)[-1])
                buf = self.model._kv_buffers.setdefault(layer_idx, {})
                if "K_WEIGHT" in jax_path:
                    buf["k_weight"] = processed_weight
                elif "K_SCALE" in jax_path:
                    buf["k_scale"] = processed_weight
                elif "V_WEIGHT" in jax_path:
                    buf["v_weight"] = processed_weight
                elif "V_SCALE" in jax_path:
                    buf["v_scale"] = processed_weight
                logger.info(
                    "Stored KV buffer %s for layer %d, shape=%s",
                    jax_path.split("__")[2],
                    layer_idx,
                    processed_weight.shape,
                )
            return

        # Handle fused QKV buffer storage (used by MiMo-V2-Pro per-shard dequant).
        # target_path like "__FUSED_QKV_WEIGHT__42" stores into model._fused_qkv_buffers.
        if jax_path.startswith("__FUSED_QKV_"):
            if hasattr(self.model, "_fused_qkv_buffers"):
                is_scale = "SCALE" in jax_path
                layer_idx = int(jax_path.rsplit("__", 1)[-1])
                buf = self.model._fused_qkv_buffers.setdefault(layer_idx, {})
                np_weight = np.asarray(processed_weight)
                buf["scale" if is_scale else "weight"] = np_weight
                logger.info(
                    "Stored fused QKV %s for layer %d, shape=%s (CPU numpy)",
                    "scale" if is_scale else "weight",
                    layer_idx,
                    np_weight.shape,
                )
            return

        # Apply output_multiplier_scale to lm_head weights (matching PyTorch implementation)
        if "lm_head" in hf_key and hasattr(self.model_config.hf_config, "output_multiplier_scale"):
            logger.info(
                "Applying output_multiplier_scale (%.2f) to %s",
                self.model_config.hf_config.output_multiplier_scale,
                hf_key,
            )
            processed_weight = processed_weight.astype(jnp.float32)
            processed_weight = (
                processed_weight * self.model_config.hf_config.output_multiplier_scale
            )

        if mapping.reshape is not None:
            processed_weight = jnp.reshape(processed_weight, mapping.reshape)
        if mapping.repeat is not None:
            axis, times = mapping.repeat
            processed_weight = jnp.repeat(processed_weight, times, axis=axis)
        if mapping.kv_head_padding:
            processed_weight = self._apply_kv_head_padding(processed_weight, hf_key)

        sharded_weight = self._shard_weight(processed_weight, mapping.sharding)

        try:
            model_param = self._get_param(params, jax_path)

            # Expand 2D block-quant scale to 3D kernel-ready layout.
            sharded_weight = self._maybe_expand_linear_block_scale(
                sharded_weight, model_param, jax_path
            )

            logger.debug(
                "Loading %s -> %s, shape: %s, transpose: %s",
                hf_key,
                jax_path,
                processed_weight.shape,
                mapping.transpose,
            )
            if sharded_weight.dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
                model_param.value = sharded_weight
            else:
                model_param.value = sharded_weight.astype(model_param.value.dtype)
        except Exception as e:
            logger.error("Failed to load %s -> %s: %s", hf_key, jax_path, str(e))
            raise

    def _handle_split_weight(
        self, params: nnx.State, hf_key: str, weight: jax.Array, mapping: WeightMapping
    ):
        self._split_qkv_weight(params, hf_key, weight, mapping)

    def _split_qkv_weight(
        self, params: nnx.State, hf_key: str, weight: jax.Array, mapping: WeightMapping
    ):
        jax_paths = mapping.target_path

        v_head_dim = getattr(self, "v_head_dim", self.head_dim_original)
        v_head_dim_pad = (v_head_dim + 127) // 128 * 128 - v_head_dim
        v_head_dim_padded = v_head_dim + v_head_dim_pad

        if hf_key.endswith(".bias"):
            q_dim = self.num_heads * self.head_dim_original
            k_dim = self.num_kv_heads * self.head_dim_original
            v_dim = self.num_kv_heads * v_head_dim

            q_bias = weight[:q_dim]
            k_bias = weight[q_dim : q_dim + k_dim]
            v_bias = weight[q_dim + k_dim : q_dim + k_dim + v_dim]

            if mapping.head_dim_padding and self.head_dim_pad > 0:
                q_bias = jnp.reshape(q_bias, (self.num_heads, self.head_dim_original))
                q_bias = jnp.pad(q_bias, ((0, 0), (0, self.head_dim_pad)))
                q_bias = jnp.reshape(q_bias, (self.num_heads * self.head_dim,))

                k_bias = jnp.reshape(k_bias, (self.num_kv_heads, self.head_dim_original))
                k_bias = jnp.pad(k_bias, ((0, 0), (0, self.head_dim_pad)))
                k_bias = jnp.reshape(k_bias, (self.num_kv_heads * self.head_dim,))

            if mapping.head_dim_padding and v_head_dim_pad > 0:
                v_bias = jnp.reshape(v_bias, (self.num_kv_heads, v_head_dim))
                v_bias = jnp.pad(v_bias, ((0, 0), (0, v_head_dim_pad)))
                v_bias = jnp.reshape(v_bias, (self.num_kv_heads * v_head_dim_padded,))

            splits = [q_bias, k_bias, v_bias]
        elif "scale" in hf_key and weight.ndim == 2:
            # Block-quant scale: split along block dimension, not element dimension.
            # The fused QKV scale has shape [total_blocks, in_blocks] where blocks
            # are computed per Q/K/V segment independently.
            import math

            quant_cfg = getattr(self.model_config, "quantization_config", None)
            block_size = int(quant_cfg.weight_block_size[0]) if quant_cfg else 128

            q_dim = self.num_heads * self.head_dim_original
            k_dim = self.num_kv_heads * self.head_dim_original

            q_blocks = math.ceil(q_dim / block_size)
            k_blocks = math.ceil(k_dim / block_size)
            # V gets remaining blocks (may include padding to head_dim_original)
            v_blocks = weight.shape[0] - q_blocks - k_blocks

            logger.info(
                "Splitting QKV scale %s shape=%s into Q=%d K=%d V=%d blocks",
                hf_key,
                weight.shape,
                q_blocks,
                k_blocks,
                v_blocks,
            )

            q_scale = weight[:q_blocks, :]
            k_scale = weight[q_blocks : q_blocks + k_blocks, :]
            v_scale = weight[q_blocks + k_blocks :, :]

            splits = [q_scale, k_scale, v_scale]
        else:
            q_dim = self.num_heads * self.head_dim_original
            k_dim = self.num_kv_heads * self.head_dim_original
            v_dim = self.num_kv_heads * v_head_dim

            if mapping.transpose:
                q_weight = weight[:, :q_dim]
                k_weight = weight[:, q_dim : q_dim + k_dim]
                v_weight = weight[:, q_dim + k_dim : q_dim + k_dim + v_dim]
            else:
                q_weight = weight[:q_dim, :]
                k_weight = weight[q_dim : q_dim + k_dim, :]
                v_weight = weight[q_dim + k_dim : q_dim + k_dim + v_dim, :]

            if mapping.head_dim_padding and self.head_dim_pad > 0:
                if mapping.transpose:
                    q_weight = jnp.reshape(
                        q_weight,
                        (self.hidden_size, self.num_heads, self.head_dim_original),
                    )
                    q_weight = jnp.pad(q_weight, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                    q_weight = jnp.reshape(
                        q_weight, (self.hidden_size, self.num_heads * self.head_dim)
                    )

                    k_weight = jnp.reshape(
                        k_weight,
                        (self.hidden_size, self.num_kv_heads, self.head_dim_original),
                    )
                    k_weight = jnp.pad(k_weight, ((0, 0), (0, 0), (0, self.head_dim_pad)))
                    k_weight = jnp.reshape(
                        k_weight, (self.hidden_size, self.num_kv_heads * self.head_dim)
                    )
                else:
                    q_weight = jnp.reshape(
                        q_weight,
                        (self.num_heads, self.head_dim_original, self.hidden_size),
                    )
                    q_weight = jnp.pad(q_weight, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                    q_weight = jnp.reshape(
                        q_weight, (self.num_heads * self.head_dim, self.hidden_size)
                    )

                    k_weight = jnp.reshape(
                        k_weight,
                        (self.num_kv_heads, self.head_dim_original, self.hidden_size),
                    )
                    k_weight = jnp.pad(k_weight, ((0, 0), (0, self.head_dim_pad), (0, 0)))
                    k_weight = jnp.reshape(
                        k_weight, (self.num_kv_heads * self.head_dim, self.hidden_size)
                    )

            if mapping.head_dim_padding and v_head_dim_pad > 0:
                if mapping.transpose:
                    v_weight = jnp.reshape(
                        v_weight,
                        (self.hidden_size, self.num_kv_heads, v_head_dim),
                    )
                    v_weight = jnp.pad(v_weight, ((0, 0), (0, 0), (0, v_head_dim_pad)))
                    v_weight = jnp.reshape(
                        v_weight, (self.hidden_size, self.num_kv_heads * v_head_dim_padded)
                    )
                else:
                    v_weight = jnp.reshape(
                        v_weight,
                        (self.num_kv_heads, v_head_dim, self.hidden_size),
                    )
                    v_weight = jnp.pad(v_weight, ((0, 0), (0, v_head_dim_pad), (0, 0)))
                    v_weight = jnp.reshape(
                        v_weight, (self.num_kv_heads * v_head_dim_padded, self.hidden_size)
                    )

            splits = [q_weight, k_weight, v_weight]

        for split_weight, jax_path in zip(splits, jax_paths):
            processed_weight = split_weight

            if mapping.kv_head_padding and ("k_proj" in jax_path or "v_proj" in jax_path):
                processed_weight = self._apply_kv_head_padding(processed_weight, jax_path)

            sharded_weight = self._shard_weight(processed_weight, mapping.sharding)

            model_param = self._get_param(params, jax_path)

            # Expand 2D block-quant scale to 3D kernel-ready layout.
            sharded_weight = self._maybe_expand_linear_block_scale(
                sharded_weight, model_param, jax_path
            )

            if sharded_weight.dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]:
                model_param.value = sharded_weight
            else:
                model_param.value = sharded_weight.astype(model_param.value.dtype)

            logger.debug("Split %s -> %s, shape: %s", hf_key, jax_path, processed_weight.shape)

    def _shard_weight(
        self, weight: jax.Array, sharding_spec: tuple, mesh: jax.sharding.Mesh = None
    ) -> jax.Array:
        if mesh is None:
            mesh = self.mesh
        target_sharding = jax.sharding.NamedSharding(mesh, P(*sharding_spec))
        # Since 'weight' is already a Lazy JAX Array (backed by a callback),
        # using device_put here is necessary when we are NOT using the "Global Loading"
        # optimization path. It will trigger the slice/distribute logic lazily.
        # However, for the optimized path, we skip this method entirely.
        return jax.device_put(weight, target_sharding)

    def _get_param(self, params: nnx.State, path: str) -> nnx.State:
        keys = path.split(".")
        current_level = params

        for key in keys:
            if key.isdigit():
                current_level = current_level[int(key)]
            else:
                if hasattr(current_level, "__contains__") and key in current_level:
                    current_level = current_level[key]
                elif hasattr(current_level, key):
                    current_level = getattr(current_level, key)
                else:
                    raise ValueError(f"{path} is not a valid param path")

        return current_level

    def _apply_kv_head_padding(self, weight: jax.Array, hf_key: str) -> jax.Array:
        """Apply KV head padding/replication when tp_size > total_kv_heads.

        Handles:
        1. Bias/Scale (1D or 2D with shape[0]=heads) -> Pad Axis 0
        2. Standard Weight (2D with shape[1]=heads*dim) -> Pad Axis 1
        3. Static Quant Weight (2D with shape[0]=heads*dim) -> Pad Axis 0
        """
        if not (
            any(proj in hf_key for proj in ["k_proj", "v_proj"])
            and self.model_config.needs_kv_head_replication(self.sharding_size)
        ):
            return weight

        total_kv_heads = self.model_config.get_total_num_kv_heads()
        num_replicas = self.model_config.get_num_kv_head_replicas(self.sharding_size)
        padding_strategy = self.model_config.get_kv_padding_strategy()

        target_axis = -1
        step_size = -1

        dim0 = weight.shape[0]
        if dim0 == total_kv_heads:
            target_axis = 0
            step_size = 1
        elif dim0 == total_kv_heads * self.head_dim:
            target_axis = 0
            step_size = self.head_dim

        if target_axis == -1 and weight.ndim > 1:
            dim1 = weight.shape[1]
            if dim1 == total_kv_heads * self.head_dim:
                target_axis = 1
                step_size = self.head_dim

        if target_axis == -1:
            return weight

        if padding_strategy == "replicate":
            replicated_parts = []

            for original_head_id in range(total_kv_heads):
                start = original_head_id * step_size
                end = (original_head_id + 1) * step_size

                part = weight[start:end] if target_axis == 0 else weight[:, start:end]

                for _ in range(num_replicas):
                    replicated_parts.append(part)

            weight = jnp.concatenate(replicated_parts, axis=target_axis)

        elif padding_strategy == "zero":
            target_heads_total = total_kv_heads * num_replicas

            if step_size == 1:
                target_len = target_heads_total
            else:
                target_len = target_heads_total * self.head_dim

            current_len = weight.shape[target_axis]
            padding_len = target_len - current_len

            if padding_len > 0:
                pad_shape = list(weight.shape)
                pad_shape[target_axis] = padding_len

                padding = jnp.zeros(tuple(pad_shape), dtype=weight.dtype)
                weight = jnp.concatenate([weight, padding], axis=target_axis)

        return weight

    def _is_excluded_layer_weight(self, hf_key: str) -> bool:
        if not hf_key.startswith("model.layers."):
            return False

        parts = hf_key.split(".")
        if len(parts) < 3 or not parts[2].isdigit():
            return False

        layer_num = int(parts[2])
        return layer_num >= self.model_config.num_hidden_layers
