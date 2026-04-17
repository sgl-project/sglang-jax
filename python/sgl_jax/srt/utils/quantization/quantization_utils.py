# Quantization utilities for sglang-jax

import itertools
import logging
import re

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.configs.quantization_config import (
    DTYPE_MAP,
    normalize_weight_block_size,
)

logger = logging.getLogger(__name__)


def _get_block_reshape_sharding(
    tensor: jax.Array,
    quantized_axes: list[int],
) -> NamedSharding | None:
    """Extend sharding specs for block reshapes.

    ``quantize_tensor()`` reshapes a quantized axis into ``(num_blocks, block)``
    before reducing over the new block axis. When the input already has explicit
    sharding, we need a matching sharding spec for the reshaped tensor:

    - keep the original sharding on the new ``num_blocks`` axis
    - mark the inner ``block`` axis as replicated
    """
    input_sharding = getattr(tensor, "sharding", None)
    if not isinstance(input_sharding, NamedSharding):
        return None

    quantized_axis_set = set(quantized_axes)
    blocked_spec = []
    for axis_idx, axis_spec in enumerate(input_sharding.spec):
        if axis_idx in quantized_axis_set:
            blocked_spec.extend([axis_spec, None])
        else:
            blocked_spec.append(axis_spec)

    return NamedSharding(input_sharding.mesh, P(*blocked_spec))


def _get_safe_block_quant_input_sharding(
    tensor: jax.Array,
    quantized_axes: list[int],
) -> NamedSharding | None:
    """Drop sharding on axes that cannot be safely split for block quant.

    Some explicit shardings become invalid once an axis is reshaped into
    ``(num_blocks, block)``. In those cases we temporarily make the quantized
    axis replicated, perform the block quantization reshape/reduction, and let
    callers restore a suitable sharding afterwards.
    """
    input_sharding = getattr(tensor, "sharding", None)
    if not isinstance(input_sharding, NamedSharding):
        return None

    adjusted_spec = list(input_sharding.spec)
    changed = False
    for axis_idx in quantized_axes:
        if axis_idx < len(adjusted_spec) and adjusted_spec[axis_idx] is not None:
            adjusted_spec[axis_idx] = None
            changed = True

    if not changed:
        return None

    return NamedSharding(input_sharding.mesh, P(*adjusted_spec))


def apply_linear_quantization(
    model_config: ModelConfig, model: nnx.Module, is_static_input: bool = False
) -> nnx.Module:
    """Apply quantization to linear layers based on regex rules.

    This walks through the model and replaces LinearBase layers with QuantizedLinear
    layers based on the regex rules in the quantization config.

    Uses per-channel weight quantization and dynamic per-token activation quantization.

    Args:
        model_config: Model configuration with quantization config
        model: The model to quantize

    Returns:
        The model with LinearBase layers replaced by QuantizedLinear layers
    """
    # Import here to avoid circular imports
    from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear

    quant_config = model_config.quantization_config
    if quant_config is None:
        raise ValueError(
            "apply_linear_quantization called but model_config.quantization_config is None. "
            "Ensure --quantization-config-path is set."
        )

    linear_rules = quant_config.get_linear_rules()
    if not linear_rules:
        raise ValueError(
            "No linear rules found in quantization config. "
            "Check your quantization config YAML file."
        )

    # Compile regex patterns from rules
    compiled_rules = []
    for rule in linear_rules:
        pattern = re.compile(rule["module_path"])
        # Accept both sglang-jax style and Qwix-style field names.
        weight_dtype_str = rule.get("weight_dtype", rule.get("weight_qtype"))
        activation_dtype_str = rule.get("activation_dtype", rule.get("act_qtype"))
        weight_block_size = (
            rule["weight_block_size"]
            if "weight_block_size" in rule
            else getattr(quant_config, "weight_block_size", None)
        )
        weight_block_size = normalize_weight_block_size(weight_block_size)

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
                "weight_block_size": weight_block_size,
            }
        )

    ignored_layers = quant_config.ignored_layers or []

    # Normalize ignored layer patterns: convert HF dot-index (layers.0.) to
    # bracket-index (layers[0].) since the model walk uses bracket notation.
    normalized_ignored = []
    for ig in ignored_layers:
        normalized_ignored.append(re.sub(r"\.(\d+)\.", r"[\1].", ig))
    ignored_layers = normalized_ignored

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
                    if any(
                        dot_path == ignored or dot_path.endswith(f".{ignored}")
                        for ignored in ignored_layers
                    ):
                        logger.info("Skipping %s - in ignored_layers", dot_path)
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
                            weight_block_size=rule["weight_block_size"],
                        )
                        # Replace the attribute and free old weights
                        setattr(obj, attr_name, quantized_linear)
                        del attr_value
                    else:
                        logger.info("Skipping %s - no matching rule", child_path)

                elif isinstance(attr_value, nnx.Module):
                    _replace_linear_recursive(attr_value, child_path, visited)

                elif isinstance(attr_value, list):
                    for idx, item in enumerate(attr_value):
                        if isinstance(item, nnx.Module):
                            item_path = f"{child_path}[{idx}]"
                            _replace_linear_recursive(item, item_path, visited)

    logger.info("Applying quantization to linear layers...")
    _replace_linear_recursive(model)
    logger.info("Quantization complete.")

    return model


def apply_moe_quantization(
    model_config: ModelConfig, model: nnx.Module, is_static_input: bool = False
) -> nnx.Module:
    """
    Quantize MoE weights in-place.

    This walks through the model and calls quantize_weights() on each EPMoE module,
    which quantizes wi_0, wi_1, wo weights and stores the scales as separate parameters.

    Uses the unified QuantizationConfig from model_config.quantization_config.
    """
    # Import here to avoid circular imports
    from sgl_jax.srt.layers.moe import EPMoE, FusedEPMoE

    quant_config = model_config.quantization_config
    if quant_config is None:
        return model

    if not quant_config.has_moe_quantization():
        return model

    # Walk through the model and quantize all EPMoE/FusedEPMoE modules
    # Models with MoE typically have: model.model.layers[i].block_sparse_moe.experts
    # or similar structure. We recursively search for EPMoE/FusedEPMoE instances.
    def _quantize_moe_recursive(obj, path: str = "", visited=None):
        if visited is None:
            visited = set()

        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, (EPMoE, FusedEPMoE)):
            log_path = path or obj.name
            logger.debug("Quantizing MoE weights path=%s", log_path)
            obj.quantize_weights(is_static=is_static_input)
            return

        # Try to iterate through attributes
        if hasattr(obj, "__dict__"):
            for attr_name, attr_value in obj.__dict__.items():
                child_path = f"{path}/{attr_name}" if path else attr_name
                if isinstance(attr_value, nnx.Module):
                    _quantize_moe_recursive(attr_value, child_path, visited)
                elif isinstance(attr_value, list):
                    for idx, item in enumerate(attr_value):
                        if isinstance(item, nnx.Module):
                            item_path = f"{child_path}[{idx}]"
                            _quantize_moe_recursive(item, item_path, visited)

    _quantize_moe_recursive(model)
    return model


def quantize_tensor_simple(
    x: jax.Array, dtype: jnp.dtype, dim: int = -1, out_dtype: jnp.dtype = jnp.float32
):
    """Simple per-token quantization for activations."""
    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = int(dtype_info.max)
        min_val = int(dtype_info.min)
    else:
        dtype_info = jnp.finfo(dtype)
        max_val = float(dtype_info.max)
        min_val = float(dtype_info.min)

    x_abs_max = jnp.max(jnp.abs(x), axis=dim, keepdims=True)

    scale = x_abs_max / max_val
    # Guard all-zero slices to avoid 0/0 -> NaN.
    scale_safe = scale + (scale == 0).astype(scale.dtype)
    x_q = jnp.clip(x / scale_safe, min_val, max_val).astype(dtype)
    return x_q, scale.astype(out_dtype)


def quantize_tensor(
    dtype: jnp.dtype,
    tensor: jax.Array,
    axis: int | tuple | None = -1,
    block_size: int | None = None,
    pad_tensor: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Quantize tensor.

    Args:
        dtype: dtype to perform quantization.
        tensor: Unquantized tensor
        axis: Axis to perform quantization. None denotes per-tensor.
        block_size: Specify block quantization size.
        pad_tensor: Whether to pad the axis along block size.

    Returns:
        Tensor quantized to dtype.
    """
    if axis is None:
        # Perform per-tensor quantization.
        axis = [i for i in range(tensor.ndim)]
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor.shape
    original_input_sharding = getattr(tensor, "sharding", None)
    mask = None

    if block_size is not None:
        if isinstance(block_size, int):
            block_size = [block_size] * len(axis)

        blocked_shape = [[i] for i in orig_shape]
        pad_width = [[0, 0] for _ in range(tensor.ndim)]
        has_padding = False
        for i, block in zip(axis, block_size):
            num_blocks = (tensor.shape[i] + block - 1) // block
            padding_size = num_blocks * block - tensor.shape[i]
            if padding_size and not pad_tensor:
                raise ValueError(
                    f"Unable to perform block quantization. axis={i} of "
                    f"{tensor.shape=} is not divisible by {block=}"
                )

            # Pad the tensor to align with block size.
            pad_width[i][1] = padding_size
            has_padding = has_padding or padding_size != 0

            blocked_shape[i] = (num_blocks, block)

        # In order to avoid padded values affecting scale value, we pad it
        # using edge value of the tensor.
        if pad_tensor and has_padding:
            mask = jnp.ones_like(tensor, jnp.int32)
            tensor = jnp.pad(tensor, pad_width, "edge")
            mask = jnp.pad(mask, pad_width)

        orig_shape = tensor.shape
        # Convert all axis into positive values.
        quantized_axes = sorted([i % tensor.ndim for i in axis])
        safe_input_sharding = _get_safe_block_quant_input_sharding(tensor, quantized_axes)
        if safe_input_sharding is not None:
            tensor = jax.sharding.reshard(tensor, safe_input_sharding)
            if mask is not None:
                mask = jax.sharding.reshard(mask, safe_input_sharding)

        # Shift axis by 1 since its original position is now occupied by
        # num_blocks dim. Also, if n axes before an axis was also quantized,
        # shift its position by n.
        axis = [1 + n + i for n, i in enumerate(quantized_axes)]

        blocked_out_sharding = _get_block_reshape_sharding(tensor, quantized_axes)

        # Flatten list of lists that contains (num_blocks, block).
        blocked_shape = list(itertools.chain(*blocked_shape))
        if blocked_out_sharding is not None:
            tensor = jax.lax.reshape(tensor, blocked_shape, out_sharding=blocked_out_sharding)
        else:
            tensor = tensor.reshape(blocked_shape)

    dtype_info = jnp.iinfo(dtype) if jnp.issubdtype(dtype, jnp.integer) else jnp.finfo(dtype)

    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    abs_max = jnp.max(jnp.abs(tensor.astype(jnp.float32)), axis=axis, keepdims=True)
    scale = abs_max / dtype_max

    # Guard all-zero blocks/tensors: scale==0 would produce 0/0 -> NaN.
    scale_safe = scale + (scale == 0).astype(scale.dtype)
    tensor_q = jnp.clip(tensor / scale_safe, dtype_min, dtype_max)
    if block_size is not None and isinstance(original_input_sharding, NamedSharding):
        tensor_q = jax.lax.reshape(tensor_q, orig_shape, out_sharding=original_input_sharding)
    else:
        tensor_q = tensor_q.reshape(orig_shape)
    tensor_q = tensor_q.astype(dtype)

    # To avoid padded values affecting output of quantized matmul, we mask them
    # out with 0s.
    if mask is not None:
        tensor_q = jnp.where(mask.astype(jnp.bool_), tensor_q, 0)

    scale = jnp.squeeze(scale, axis).astype(jnp.float32)

    return tensor_q, scale


def dequantize_tensor(
    tensor_q: jax.Array,
    scale: jax.Array,
    axis: int | None | tuple = -1,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Dequantize a quantized tensor

    Args:
        tensor_q: Quantized tensor.
        scale: Quantization scale.
        axis: The axis tensor was quantized. None denotes per-tensor.
        out_dtype: Dtype of the output.

    Returns:
        Dequantized tensor_q.
    """
    if axis is None:
        # Perform per-tensor quantization.
        axis = [i for i in range(tensor_q.ndim)]
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor_q.shape
    if tensor_q.ndim == scale.ndim:
        # Indicates the tensor was block quantized.
        blocked_shape = [[i] for i in orig_shape]
        for i in axis:
            num_blocks = scale.shape[i]
            if tensor_q.shape[i] % num_blocks:
                raise ValueError(
                    f"Unable to perform block dequantization. axis={i} of "
                    f"{tensor_q.shape=} is not divisible by {num_blocks=}",
                )
            block_size = tensor_q.shape[i] // num_blocks

            blocked_shape[i] = (num_blocks, block_size)

        # Convert all axis into positive values.
        axis = sorted([(i + tensor_q.ndim) % tensor_q.ndim for i in axis])
        # Shift axis by 1 since its original position is now occupied by
        # num_blocks dim. Also, if n axes before an axis was also quantized,
        # shift its position by n.
        axis = [1 + n + i for n, i in enumerate(axis)]

        # Flatten list of lists that contains (num_blocks, block).
        blocked_shape = list(itertools.chain(*blocked_shape))
        tensor_q = tensor_q.reshape(blocked_shape)

    scale = jnp.expand_dims(scale, axis)

    tensor = (tensor_q.astype(jnp.float32) * scale).astype(out_dtype)

    return tensor.reshape(orig_shape)
