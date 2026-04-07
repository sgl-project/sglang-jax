"""MiMo-V2-Flash specific weight loading utilities.

Extracted from weight_utils.py and model_config.py to isolate model-specific
complexity (per-layer KV heads, split K/V head dims, misaligned block scales)
from the shared weight loading infrastructure.
"""

import logging
import math
import re

import jax.numpy as jnp

from sgl_jax.srt.kernels.quantized_matmul.blockwise_utils import expand_block_scale

logger = logging.getLogger(__name__)


def mimo_apply_kv_head_padding(
    weight: "jax.Array",
    hf_key: str,
    model_config,
    sharding_size: int,
    head_dim: int,
) -> "jax.Array":
    """MiMo-V2-Flash version of KV head padding/replication.

    Handles MiMo-specific features:
    - Per-layer KV head counts (SWA layers vs normal layers)
    - Separate K/V head dimensions (head_dim=192 for K, v_head_dim=128 for V)
    - Misaligned block-quant scales (head_dim=192, block_size=128 -> 1.5 blocks/head)
    - Per-head tile-based scale replication for block-quant weights
    """
    if not any(proj in hf_key for proj in ["k_proj", "v_proj"]):
        return weight

    layer_idx = None
    match = re.search(r"(?:^|[./])layers[.\[](\d+)[\].]", hf_key)
    if match is not None:
        layer_idx = int(match.group(1))

    original_kv_heads, total_kv_heads = _get_kv_head_counts_for_layer(
        model_config, layer_idx, sharding_size
    )

    if sharding_size <= original_kv_heads:
        return weight

    padding_strategy = model_config.get_kv_padding_strategy()

    target_axis = -1
    step_size = -1

    # Use v_head_dim for v_proj, head_dim for k_proj
    proj_head_dim = (
        getattr(model_config, "v_head_dim", head_dim)
        if "v_proj" in hf_key
        else head_dim
    )

    dim0 = weight.shape[0]

    # Resolve actual quantization block_size_out from model config.
    _quant_block_size_out = 0
    _quant_cfg = getattr(model_config, "quantization_config", None)
    _wbs = getattr(_quant_cfg, "weight_block_size", None)
    if isinstance(_wbs, (list, tuple)) and len(_wbs) == 2:
        _quant_block_size_out = int(_wbs[0])

    # Detect block-wise quant scales (e.g. static fp8 scale_inv with shape [n_blocks, k_blocks]).
    # For k_proj/v_proj scales, the KV-head axis is axis 0, grouped by blocks-per-head.
    # Only use this fast path when blocks align with heads (proj_head_dim % block_size == 0).
    # When they don't align (e.g. k_proj head_dim=192, block_size=128), fall through
    # to the per-channel expansion fallback below.
    if target_axis == -1 and hf_key.endswith("weight_scale_inv") and weight.ndim == 2:
        if original_kv_heads > 0 and dim0 % original_kv_heads == 0:
            blocks_align = (
                _quant_block_size_out > 0
                and proj_head_dim % _quant_block_size_out == 0
            )
            if blocks_align:
                target_axis = 0
                step_size = dim0 // original_kv_heads

    # Detect common layouts
    if target_axis == -1 and dim0 % proj_head_dim == 0:
        target_axis = 0
        step_size = proj_head_dim
    if target_axis == -1 and dim0 == total_kv_heads:
        target_axis = 0
        step_size = 1

    if target_axis == -1 and weight.ndim > 1:
        dim1 = weight.shape[1]
        if dim1 % proj_head_dim == 0:
            target_axis = 1
            step_size = proj_head_dim
        elif dim1 == total_kv_heads * proj_head_dim:
            target_axis = 1
            step_size = proj_head_dim

    if target_axis == -1:
        # Special case: block-wise KV head scales where block boundaries don't align
        # with head boundaries (e.g. head_dim=192, block_size=128, so 1.5 blocks/head).
        # Expand compact [out_blocks, in_blocks] to per-channel
        # [original_kv_heads * proj_head_dim, in_blocks], then fall through
        # to normal head-replication.
        if hf_key.endswith("weight_scale_inv") and weight.ndim == 2:
            original_weight_out = original_kv_heads * proj_head_dim
            if _quant_block_size_out > 0:
                orig_block_size = _quant_block_size_out
            else:
                orig_block_size = math.ceil(original_weight_out / weight.shape[0])
            if orig_block_size > 0:
                weight = jnp.repeat(weight, orig_block_size, axis=0)[:original_weight_out, :]
                target_axis = 0
                step_size = proj_head_dim
            else:
                return weight
        else:
            return weight

    # If weight has fewer heads than expected (e.g., orig 4 -> target 8), replicate.
    actual_heads = weight.shape[target_axis] // step_size
    if (
        padding_strategy == "replicate"
        and actual_heads < total_kv_heads
        and actual_heads == original_kv_heads
    ):
        if (
            hf_key.endswith("weight_scale_inv")
            and weight.ndim == 2
            and target_axis == 0
            and total_kv_heads % actual_heads == 0
        ):
            # Static block scales: tile-per-head to match repeat-per-head replication.
            reps = total_kv_heads // actual_heads
            head_parts = []
            for h in range(actual_heads):
                head_scale = weight[h * step_size:(h + 1) * step_size, :]
                head_parts.append(jnp.tile(head_scale, (reps, 1)))
            out = jnp.concatenate(head_parts, axis=0)
            return out
        # Repeat-per-head: each head replicated reps times consecutively
        # e.g. 4 heads -> 16: [h0,h0,h0,h0, h1,h1,h1,h1, ...]
        reps = total_kv_heads // actual_heads
        parts = []
        for head_idx in range(actual_heads):
            start = head_idx * step_size
            end = (head_idx + 1) * step_size
            head_slice = weight[start:end] if target_axis == 0 else weight[:, start:end]
            for _ in range(reps):
                parts.append(head_slice)
        weight = jnp.concatenate(parts, axis=target_axis)
        return weight

    # If tensor already has enough heads, skip legacy replication path.
    if actual_heads >= total_kv_heads:
        return weight

    if padding_strategy == "replicate":
        replicated_parts = []
        target_heads_total = total_kv_heads
        reps_per_head = (target_heads_total + actual_heads - 1) // actual_heads

        for original_head_id in range(actual_heads):
            start = original_head_id * step_size
            end = (original_head_id + 1) * step_size
            part = weight[start:end] if target_axis == 0 else weight[:, start:end]
            for _ in range(reps_per_head):
                replicated_parts.append(part)

        weight = jnp.concatenate(replicated_parts, axis=target_axis)
        target_len = target_heads_total * step_size
        if weight.shape[target_axis] > target_len:
            if target_axis == 0:
                weight = weight[:target_len]
            else:
                weight = weight[:, :target_len]
    elif padding_strategy == "zero":
        target_heads_total = total_kv_heads

        if step_size == 1:
            target_len = target_heads_total
        else:
            target_len = target_heads_total * head_dim

        current_len = weight.shape[target_axis]
        padding_len = target_len - current_len

        if padding_len > 0:
            pad_shape = list(weight.shape)
            pad_shape[target_axis] = padding_len
            padding = jnp.zeros(tuple(pad_shape), dtype=weight.dtype)
            weight = jnp.concatenate([weight, padding], axis=target_axis)

    return weight


def mimo_expand_linear_block_scale(
    scale: "jax.Array",
    model_param,
    jax_path: str,
    model_config,
) -> "jax.Array":
    """Expand a 2D block-quant scale to 3D kernel-ready layout.

    Handles per-head block boundaries where head_dim doesn't divide evenly
    into block_size (e.g., head_dim=192, block_size=128 -> 2 blocks per head
    with the second block only covering 64 channels).
    """
    target_shape = getattr(model_param.value, "shape", ())
    if scale.ndim != 2 or len(target_shape) != 3:
        return scale

    n_out = target_shape[2]
    out_blocks = scale.shape[0]

    # Per-channel scale (e.g., from kv_head_padding misaligned-block expansion).
    if out_blocks >= n_out:
        return jnp.transpose(scale[:n_out, :], (1, 0))[:, None, :]

    # Get actual block_size from quantization config.
    quant_config = getattr(model_config, "quantization_config", None)
    block_size_out = 128  # default
    if quant_config and hasattr(quant_config, "weight_block_size"):
        wbs = quant_config.weight_block_size
        if wbs is not None:
            block_size_out = int(wbs[0])

    expected_uniform_blocks = math.ceil(n_out / block_size_out)

    if out_blocks <= expected_uniform_blocks:
        inferred = math.ceil(n_out / out_blocks)
        return expand_block_scale(scale, n_out, inferred)

    # Per-head blocks: more blocks than uniform would give.
    # Build explicit channel-to-block mapping respecting head boundaries.
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        hf_text = getattr(model_config, "hf_text_config", None)
        if hf_text:
            head_dim = getattr(hf_text, "head_dim", None)
    if head_dim is None:
        inferred = math.ceil(n_out / out_blocks)
        return expand_block_scale(scale, n_out, inferred)

    blocks_per_head = math.ceil(head_dim / block_size_out)
    channel_map = []
    for h in range(out_blocks // blocks_per_head):
        for b in range(blocks_per_head):
            block_idx = h * blocks_per_head + b
            remaining = head_dim - b * block_size_out
            channels = min(block_size_out, remaining)
            channel_map.extend([block_idx] * channels)
    channel_to_block = jnp.asarray(channel_map[:n_out], dtype=jnp.int32)
    return expand_block_scale(scale, n_out, block_size_out, channel_to_block=channel_to_block)


def _get_kv_head_counts_for_layer(
    model_config,
    layer_idx: int | None,
    tensor_parallel_size: int | None,
) -> tuple[int, int]:
    """Return (original_kv_heads, total_kv_heads_after_tp) for a given layer.

    Handles MiMo's hybrid attention: SWA layers may use a different KV head
    count than normal layers, determined by ``hybrid_layer_pattern``.
    """
    from sgl_jax.srt.utils.jax_utils import get_num_kv_heads_by_tp

    is_swa_layer = False
    hf_text_config = model_config.hf_text_config

    if layer_idx is not None:
        hybrid_layer_pattern = getattr(hf_text_config, "hybrid_layer_pattern", None)
        if hybrid_layer_pattern is not None and 0 <= layer_idx < len(hybrid_layer_pattern):
            is_swa_layer = hybrid_layer_pattern[layer_idx] == 1
        else:
            hybrid_pattern = getattr(hf_text_config, "hybrid_pattern", None)
            if hybrid_pattern and 0 <= layer_idx < len(hybrid_pattern):
                is_swa_layer = hybrid_pattern[layer_idx] == "swa"

    if is_swa_layer and hasattr(hf_text_config, "swa_num_key_value_heads"):
        original_kv_heads = getattr(
            model_config,
            "_original_swa_num_key_value_heads",
            getattr(hf_text_config, "swa_num_key_value_heads"),
        )
        current_total_kv_heads = getattr(hf_text_config, "swa_num_key_value_heads")
    else:
        original_kv_heads = getattr(
            model_config,
            "_original_num_key_value_heads",
            model_config.get_total_num_kv_heads(),
        )
        current_total_kv_heads = getattr(
            hf_text_config,
            "num_key_value_heads",
            model_config.num_key_value_heads,
        )

    if tensor_parallel_size is None:
        return int(original_kv_heads), int(current_total_kv_heads)

    kv_heads_per_device = get_num_kv_heads_by_tp(int(original_kv_heads), tensor_parallel_size)
    return int(original_kv_heads), int(kv_heads_per_device * tensor_parallel_size)
