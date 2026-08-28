"""Assemble K3's per-expert MXFP4 MoE weights into the stacked arrays EPMoE expects.

K3 ships each expert separately, packed::

    ...block_sparse_moe.experts.<e>.w1.weight_packed   uint8  [inter, hidden/2]   gate
    ...block_sparse_moe.experts.<e>.w3.weight_packed   uint8  [inter, hidden/2]   up
    ...block_sparse_moe.experts.<e>.w2.weight_packed   uint8  [hidden, inter/2]   down
    ...<same>.weight_scale                              uint8  e8m0, one per 32

EPMoE wants three stacked bf16 arrays::

    wi_0 [E, hidden, inter]   gate     <- w1
    wi_1 [E, hidden, inter]   up       <- w3
    wo   [E, inter, hidden]   down     <- w2

Two transforms are therefore needed per expert, and both are easy to get subtly wrong:
**dequantize** (unpack fp4 pairs, decode e8m0 exponents, apply per-32 group scale) and
**transpose** (the checkpoint stores ``[out, in]``, EPMoE wants ``[in, out]``).

> Memory note: dequantizing to bf16 at load is correct but 4x the checkpoint size. That is fine
> for a truncated bring-up config; the full 93-layer model at 1.42 TiB would become ~5.6 TB and
> needs a native-fp4 path instead. This module deliberately does the simple thing and says so.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.quantization.mxfp4 import (
    e8m0_to_fp32,
    MXFP4_GROUP_SIZE,
    dequantize_tensor_from_mxfp4_packed,
)

# checkpoint projection name -> EPMoE parameter name
EXPERT_PROJ_TO_EPMOE = {"w1": "wi_0", "w3": "wi_1", "w2": "wo"}


def dequant_expert_weight(
    packed: jax.Array,
    scale: jax.Array,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Dequantize one expert projection and transpose to EPMoE's ``[in, out]``.

    The quantized axis is the LAST axis of the stored ``[out, in]`` tensor (the input dim), which
    is why dequant happens BEFORE the transpose. Transposing first would apply the per-32 group
    scale along the output dim and silently corrupt every expert.
    """
    if packed.dtype != jnp.uint8:
        raise TypeError(f"expected uint8 packed weight, got {packed.dtype}")
    if scale.dtype != jnp.uint8:
        raise TypeError(f"expected uint8 e8m0 scale, got {scale.dtype}")
    expected_groups = (packed.shape[-1] * 2) // MXFP4_GROUP_SIZE
    if scale.shape[-1] != expected_groups:
        raise ValueError(
            f"scale groups {scale.shape[-1]} != expected {expected_groups} "
            f"for packed {packed.shape} at group_size {MXFP4_GROUP_SIZE}"
        )
    w = dequantize_tensor_from_mxfp4_packed(packed, scale, axis=-1, out_dtype=out_dtype)
    return jnp.swapaxes(w, -1, -2)


def stack_experts(
    per_expert: dict[int, jax.Array],
    num_experts: int,
) -> jax.Array:
    """Stack ``{expert_idx: [in, out]}`` into ``[E, in, out]``, erroring on gaps.

    A missing expert would otherwise stack short or silently reorder; both produce a model that
    loads and routes tokens to the wrong weights.
    """
    missing = [e for e in range(num_experts) if e not in per_expert]
    if missing:
        raise KeyError(f"missing {len(missing)} experts, first few: {missing[:5]}")
    return jnp.stack([per_expert[e] for e in range(num_experts)], axis=0)


def build_epmoe_weights(
    tensors: dict[str, jax.Array],
    layer_idx: int,
    num_experts: int,
    prefix: str = "language_model.model.layers",
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, jax.Array]:
    """Build ``{wi_0, wi_1, wo}`` for one layer from a dict of checkpoint tensors."""
    out: dict[str, jax.Array] = {}
    for proj, target in EXPERT_PROJ_TO_EPMOE.items():
        per_expert = {}
        for e in range(num_experts):
            base = f"{prefix}.{layer_idx}.block_sparse_moe.experts.{e}.{proj}"
            pk, sk = f"{base}.weight_packed", f"{base}.weight_scale"
            if pk not in tensors:
                continue
            if sk not in tensors:
                raise KeyError(f"{pk} present but {sk} missing -- dequant would mis-scale")
            per_expert[e] = dequant_expert_weight(tensors[pk], tensors[sk], out_dtype)
        if per_expert:
            out[target] = stack_experts(per_expert, num_experts)
    return out


# ==============================================================================================
# fp4-NATIVE path -- what the full 93-layer model needs
# ==============================================================================================
# Dequantizing to bf16 (above) is fine for a truncated bring-up and impossible at full depth:
# K3's 2.723 T routed-expert params cost 5,072 GiB as bf16 versus 1,347 GiB kept as fp4 --
# 26.8 chips of weights instead of 7.1, at v7x's measured 189.5 GiB/chip.
#
# The functions below keep the weights fp4 all the way into the kernel, matching what the
# reference MXFP4 dequant path in the K3 MoE. They widen the released e2m1 expert weights and
# their e8m0 block scales to bf16 for the standard grouped matmul.


def unpack_fp4_to_e2m1(packed: jax.Array) -> jax.Array:
    """``[..., N, K/2]`` uint8 -> native ``float4_e2m1fn`` ``[..., K, N]`` (K-major).

    Two transforms in one, both done ONCE at load so the forward hands native fp4 straight to the
    kernel with no per-forward unpack or transpose:

    * ``bitcast_convert_type`` to a 4-bit type appends a trailing axis of 2 (the two values per
      byte); folding it into the last dim restores ``[..., N, K]``.
    * ``swapaxes`` puts the contracting axis first, because gmm_v2 wants
      ``[size_group, size_k, size_n]``.

    The result stays 4 bits per value -- there is no widening anywhere in here, which is the whole
    point.
    """
    if packed.dtype != jnp.uint8:
        raise TypeError(f"expected uint8 packed weight, got {packed.dtype}")
    fp4 = jax.lax.bitcast_convert_type(packed, jnp.float4_e2m1fn)
    fp4 = fp4.reshape(*packed.shape[:-1], -1)  # [..., N, K]
    return jnp.swapaxes(fp4, -1, -2)  # [..., K, N]


def e8m0_scale_to_kernel_layout(scale: jax.Array) -> jax.Array:
    """e8m0 uint8 ``[..., N, num_blocks]`` -> fp32 ``[..., num_blocks, 1, N]``.

    gmm_v2 reads block scales as ``(size_group, num_k_blocks, 1, size_n)`` and validates that
    shape, so the moved axis is not cosmetic. The e8m0 decode must be exponent arithmetic
    (``ldexp``), never a cast -- see :func:`e8m0_to_fp32`.
    """
    fp32 = e8m0_to_fp32(scale)
    return jnp.expand_dims(jnp.moveaxis(fp32, -1, -2), axis=-2)


def build_fp4_expert_weights(
    tensors: dict[str, jax.Array],
    layer_idx: int,
    num_experts: int,
    prefix: str = "language_model.model.layers",
) -> dict[str, tuple[jax.Array, jax.Array]]:
    """Build ``{wi_0|wi_1|wo: (fp4_kmajor [E, K, N], fp32_scale [E, blocks, 1, N])}``.

    Same expert-completeness guarantee as :func:`build_epmoe_weights` -- a missing expert raises
    rather than stacking short, which would load cleanly and route tokens to the wrong weights.
    """
    out: dict[str, tuple[jax.Array, jax.Array]] = {}
    for proj, target in EXPERT_PROJ_TO_EPMOE.items():
        w_per_expert, s_per_expert = {}, {}
        for e in range(num_experts):
            base = f"{prefix}.{layer_idx}.block_sparse_moe.experts.{e}.{proj}"
            pk, sk = f"{base}.weight_packed", f"{base}.weight_scale"
            if pk not in tensors:
                continue
            if sk not in tensors:
                raise KeyError(f"{pk} present but {sk} missing -- the kernel would mis-scale")
            packed, scale = tensors[pk], tensors[sk]
            expected_groups = (packed.shape[-1] * 2) // MXFP4_GROUP_SIZE
            if scale.shape[-1] != expected_groups:
                raise ValueError(
                    f"scale groups {scale.shape[-1]} != expected {expected_groups} "
                    f"for packed {packed.shape} at group_size {MXFP4_GROUP_SIZE}"
                )
            w_per_expert[e] = unpack_fp4_to_e2m1(packed)
            s_per_expert[e] = e8m0_scale_to_kernel_layout(scale)
        if w_per_expert:
            out[target] = (
                stack_experts(w_per_expert, num_experts),
                stack_experts(s_per_expert, num_experts),
            )
    return out
