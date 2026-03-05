# SPDX-License-Identifier: Apache-2.0
"""Quantization utilities."""

import logging
import re
from typing import Any, List

import jax
import jax.numpy as jnp
from flax import nnx

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "int8": jnp.int8,
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "float8_e5m2": jnp.float8_e5m2,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
}


def apply_linear_quantization(
    obj: Any,
    quant_config: Any,
    linear_rules: List[dict],
    is_static_input: bool = False,
) -> Any:
    """
    Recursively apply quantization to linear layers based on rules.
    """
    from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear

    if not linear_rules:
        return obj

    # Compile regex patterns from rules
    compiled_rules = []
    for rule in linear_rules:
        pattern = re.compile(rule["module_path"])
        weight_dtype_str = rule.get("weight_dtype")
        activation_dtype_str = rule.get("activation_dtype")

        # Convert string dtypes to jnp dtypes
        weight_dtype = DTYPE_MAP.get(weight_dtype_str)
        activation_dtype = DTYPE_MAP.get(activation_dtype_str)

        if weight_dtype is None:
            raise ValueError(f"weight_dtype is required in rule but got: {weight_dtype_str}")

        compiled_rules.append(
            {
                "pattern": pattern,
                "weight_dtype": weight_dtype,
                "activation_dtype": activation_dtype,
            }
        )

    ignored_layers = getattr(quant_config, "ignored_layers", None) or []

    def _find_matching_rule(path: str):
        """Find the first rule that matches the given module path."""
        for rule in compiled_rules:
            if rule["pattern"].match(path):
                return rule
        return None

    def _replace_linear_recursive(obj, path: str = "", visited: set | None = None):
        """Recursively walk the model and replace LinearBase with QuantizedLinear."""
        if visited is None:
            visited = set()

        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        # Try to iterate through attributes
        if hasattr(obj, "__dict__"):
            for attr_name, attr_value in list(obj.__dict__.items()):
                child_path = f"{path}/{attr_name}" if path else attr_name

                if isinstance(attr_value, LinearBase):
                    # Check if this path matches any rule
                    dot_path = child_path.replace("/", ".")
                    if any(dot_path.endswith(ignored) or ignored in dot_path for ignored in ignored_layers):
                        logger.info("Skipping %s - in ignored_layers", child_path)
                        continue
                    if "self_attn.o_proj" in dot_path and ignored_layers:
                        logger.info("Skipping %s - explicit o_proj ignore", child_path)
                        continue

                    rule = _find_matching_rule(child_path)
                    if rule is not None:
                        logger.debug(
                            "Quantizing %s with weight_dtype=%s, activation_dtype=%s",
                            child_path,
                            rule["weight_dtype"],
                            rule["activation_dtype"],
                        )
                        # Convert to QuantizedLinear
                        quantized_linear = QuantizedLinear.from_linear(
                            attr_value,
                            weight_dtype=rule["weight_dtype"],
                            activation_dtype=rule["activation_dtype"],
                            is_static_input=is_static_input,
                            weight_block_size=getattr(quant_config, "weight_block_size", None),
                        )
                        # Replace the attribute and free old weights
                        setattr(obj, attr_name, quantized_linear)
                        del attr_value
                    else:
                        _replace_linear_recursive(attr_value, child_path, visited)
                elif isinstance(attr_value, nnx.Module):
                    _replace_linear_recursive(attr_value, child_path, visited)
                elif isinstance(attr_value, list):
                    for idx, item in enumerate(attr_value):
                        if isinstance(item, nnx.Module):
                            _replace_linear_recursive(item, f"{child_path}[{idx}]", visited)

    _replace_linear_recursive(obj)
    return obj


def apply_moe_quantization(
    model: Any,
    quantization_config: Any,
    is_static: bool = False,
) -> Any:
    """
    Recursively apply quantization to MoE layers.

    Args:
        model: The model to apply quantization to
        quantization_config: The quantization configuration
        is_static: Whether weights are already quantized

    Returns:
        The modified model with quantized MoE layers
    """
    if quantization_config is None:
        return model

    # Import here to avoid circular imports
    from sgl_jax.srt.layers.moe import EPMoE

    def _quantize_moe_recursive(obj, visited: set | None = None):
        if visited is None:
            visited = set()

        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, EPMoE):
            obj.quantize_weights(is_static=is_static)
            return

        if hasattr(obj, "__dict__"):
            for attr_value in obj.__dict__.values():
                if isinstance(attr_value, nnx.Module):
                    _quantize_moe_recursive(attr_value, visited)
                elif isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, nnx.Module):
                            _quantize_moe_recursive(item, visited)

    _quantize_moe_recursive(model)
    return model


def adapt_fused_moe_static_block_quant_for_kernel(
    model: nnx.Module,
    *,
    target_subc_quant_wsz: int = 256,
) -> nnx.Module:
    """Adapt static fused-MoE block quant weights/scales before fused kernel execution.

    This is a front-end compatibility step for static checkpoints whose fused MoE
    subchannel block size is smaller than the fused kernel's supported size.
    """
    # Import here to avoid circular imports
    from sgl_jax.srt.layers.moe import FusedEPMoE

    adapted_count = 0

    def _adapt_recursive(obj, path: str = "", visited=None):
        nonlocal adapted_count
        if visited is None:
            visited = set()

        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, FusedEPMoE):
            if obj.prepare_static_block_quant_for_fused_kernel(
                target_subc_quant_wsz=target_subc_quant_wsz
            ):
                adapted_count += 1
                logger.info(
                    "Adapted static fused MoE at %s to subc=%s for fused kernel",
                    path or getattr(obj, "name", type(obj).__name__),
                    target_subc_quant_wsz,
                )
            return

        if hasattr(obj, "__dict__"):
            for attr_name, attr_value in obj.__dict__.items():
                child_path = f"{path}/{attr_name}" if path else attr_name
                if isinstance(attr_value, nnx.Module):
                    _adapt_recursive(attr_value, child_path, visited)
                elif isinstance(attr_value, list):
                    for idx, item in enumerate(attr_value):
                        if isinstance(item, nnx.Module):
                            item_path = f"{child_path}[{idx}]"
                            _adapt_recursive(item, item_path, visited)

    _adapt_recursive(model)
    if adapted_count:
        logger.info(
            "Completed static fused MoE block-quant kernel adaptation on %d layer(s)",
            adapted_count,
        )
    return model


def quantize_tensor_simple(
    x: jax.Array, dtype: jnp.dtype, dim: int = -1, out_dtype: jnp.dtype = jnp.float32
):
    """
    Quantize a tensor to a lower precision using absolute maximum scaling.
    """
    if dtype == jnp.int8:
        min_val, max_val = -128, 127
    else:
        # For float8 or others
        finfo = jnp.finfo(dtype)
        min_val, max_val = finfo.min, finfo.max

    x_abs_max = jnp.max(jnp.abs(x), axis=dim, keepdims=True)
    scale = x_abs_max / max_val
    # Guard all-zero slices to avoid 0/0 -> NaN.
    scale_safe = scale + (scale == 0).astype(scale.dtype)
    x_q = jnp.clip(x / scale_safe, min_val, max_val).astype(dtype)
    return x_q, scale.astype(out_dtype)


def quantize_tensor(
    dtype: jnp.dtype,
    tensor: jax.Array,
    axis: int | tuple[int, ...] | None = None,
    block_size: int | tuple[int, ...] | None = None,
    pad_tensor: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """
    Quantize a tensor using absolute maximum scaling.

    Args:
        dtype: The target dtype for quantization
        tensor: The input tensor to quantize
        axis: The axis or axes to reduce along to find the maximum
        block_size: The block size for block-wise quantization
        pad_tensor: Whether to pad the tensor if its size is not divisible by block_size

    Returns:
        A tuple of (quantized_tensor, scale)
    """
    if dtype == jnp.int8:
        min_val, max_val = -128, 127
    else:
        finfo = jnp.finfo(dtype)
        min_val, max_val = finfo.min, finfo.max

    if block_size is not None:
        # Block-wise quantization
        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(block_size, int):
            block_size = (block_size,)

        if len(axis) != len(block_size):
            raise ValueError("axis and block_size must have the same length")

        # Reshape to block format
        new_shape = list(tensor.shape)
        for ax, bs in zip(axis, block_size):
            if pad_tensor and new_shape[ax] % bs != 0:
                pad_width = [(0, 0)] * tensor.ndim
                pad_width[ax] = (0, bs - (new_shape[ax] % bs))
                tensor = jnp.pad(tensor, pad_width)
                new_shape[ax] = tensor.shape[ax]

            if new_shape[ax] % bs != 0:
                raise ValueError(f"Dimension {ax} size {new_shape[ax]} not divisible by block_size {bs}")

        # Complex reshape logic for block quant
        reshape_dims = []
        reduction_axes = []
        curr_axis = 0
        for i in range(tensor.ndim):
            if i in axis:
                idx = axis.index(i)
                bs = block_size[idx]
                reshape_dims.append(tensor.shape[i] // bs)
                reshape_dims.append(bs)
                reduction_axes.append(curr_axis + 1)
                curr_axis += 2
            else:
                reshape_dims.append(tensor.shape[i])
                curr_axis += 1

        reshaped_tensor = tensor.reshape(reshape_dims)
        abs_max = jnp.max(jnp.abs(reshaped_tensor), axis=tuple(reduction_axes), keepdims=True)
        scale = abs_max / max_val
        scale_safe = scale + (scale == 0).astype(scale.dtype)
        tensor_q = jnp.clip(reshaped_tensor / scale_safe, min_val, max_val).astype(dtype)

        # Reshape back quantized tensor
        tensor_q = tensor_q.reshape(tensor.shape)
        # Squeeze scale reduction dimensions
        scale = jnp.squeeze(scale, axis=tuple(reduction_axes))
        return tensor_q, scale
    else:
        # Per-channel or per-tensor quantization
        abs_max = jnp.max(jnp.abs(tensor), axis=axis, keepdims=True)
        scale = abs_max / max_val
        scale_safe = scale + (scale == 0).astype(scale.dtype)
        tensor_q = jnp.clip(tensor / scale_safe, min_val, max_val).astype(dtype)
        if axis is not None:
            scale = jnp.squeeze(scale, axis=axis)
        return tensor_q, scale


def dequantize_tensor(
    tensor_q: jax.Array,
    scale: jax.Array,
    axis: int | tuple[int, ...] | None = None,
    out_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """
    Dequantize a tensor.
    """
    if axis is not None:
        if isinstance(axis, int):
            axis = (axis,)
        scale = jnp.expand_dims(scale, axis=axis)

    return (tensor_q.astype(out_dtype) * scale.astype(out_dtype)).astype(out_dtype)
