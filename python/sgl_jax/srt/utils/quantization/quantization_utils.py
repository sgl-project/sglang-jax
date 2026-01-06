# Adapted from https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/models/jax/utils/qwix/qwix_utils.py

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import qwix
import yaml
from flax import nnx
from jax.sharding import auto_axes
import itertools
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo

DEFAULT_NUM_PAGES = 100

def parse_qwix_config_to_rules(qwix_config: list[dict]) -> list[qwix.QuantizationRule]:
    """
    Parse a list of dictionaries containing Qwix quantization rules into a list of QuantizationRule objects.

    Args:
        qwix_config: a dictionary containing the Qwix quantization rules

    Returns:
        a list of QuantizationRule objects
    """
    rules = []
    for rule in qwix_config:
        rules.append(qwix.QuantizationRule(**rule))

    return rules


def qwix_quantize_nnx_model(
    model: nnx.Module,
    qwix_config: list[dict],
    forward_batch: ForwardBatch,
    token_to_kv_pool: MHATokenToKVPool,
    logits_metadata: LogitsMetadata,
) -> nnx.Module:
    """
    Quantizes a Flax NNX model using Qwix.

    Args:
        model: the model to quantize
        qwix_config: a list of dictionaries, where each dictionary corresponds to a Qwix quantization rule
            For example:
            [
                {
                    "module_path": ".*attn.*",
                    "weight_qtype": "int8",
                },
                {
                    "module_path": ".*mlp.*",
                    "weight_qtype": "int8",
                    "act_qtype": "int8",
                    "tile_size": None,
                },
            ]

    Returns:
        model: the quantized model
    """

    qwix_rules = parse_qwix_config_to_rules(qwix_config)
    model_input = {
        "forward_batch": forward_batch,
        "token_to_kv_pool": token_to_kv_pool,
        "logits_metadata": logits_metadata,
    }
    model = qwix.quantize_model(model, qwix.PtqProvider(qwix_rules), **model_input)
    return model


def apply_qwix_quantization(
    model_config: ModelConfig, model: nnx.Module, model_runner
) -> nnx.Module:
    """
    Will apply quantization if a valid quantization config with Qwix rules is provided.  See README
    for more details on Qwix.

    Uses the unified QuantizationConfig from model_config.quantization_config.
    """
    # Get qwix rules from the unified QuantizationConfig
    quant_config = model_config.quantization_config
    if quant_config is None:
        raise ValueError(
            "apply_qwix_quantization called but model_config.quantization_config is None. "
            "Ensure --quantization-config-path is set."
        )

    qwix_config = quant_config.get_qwix_rules()
    if not qwix_config:
        raise ValueError(
            "No qwix rules found in quantization config. "
            "Check your quantization config YAML file."
        )

    # prepare batch input
    forward_batch, token_to_kv_pool, logits_metadata = prepare_inputs_for_quantization(
        model_config, model_runner
    )

    qwix_quantize_nnx_model_with_config_and_attn_backend = functools.partial(
        qwix_quantize_nnx_model,
        qwix_config=qwix_config,
    )
    with jax.set_mesh(model_runner.mesh):
        print("model_runner.mesh: ", model_runner.mesh)
        model = nnx.jit(
            qwix_quantize_nnx_model_with_config_and_attn_backend,
            donate_argnames=("model",),  # donate the model to the jitted function to save memory
        )(
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            logits_metadata=logits_metadata,
            model=model,
        )
    return model


def apply_moe_quantization(
    model_config: ModelConfig, model: nnx.Module
) -> nnx.Module:
    """
    Quantize MoE weights in-place. Call this after apply_qwix_quantization.
    
    This walks through the model and calls quantize_weights() on each EPMoE module,
    which quantizes wi_0, wi_1, wo weights and stores the scales as separate parameters.
    
    Uses the unified QuantizationConfig from model_config.quantization_config.
    """
    # Import here to avoid circular imports
    from sgl_jax.srt.layers.moe import EPMoE
    
    quant_config = model_config.quantization_config
    if quant_config is None:
        return model
    
    if not quant_config.has_moe_quantization():
        return model
    
    # Walk through the model and quantize all EPMoE modules
    # Models with MoE typically have: model.model.layers[i].block_sparse_moe.experts
    # or similar structure. We recursively search for EPMoE instances.
    def _quantize_moe_recursive(obj, visited=None):
        if visited is None:
            visited = set()
        
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        
        if isinstance(obj, EPMoE):
            obj.quantize_weights()
            return
        
        # Try to iterate through attributes
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                if isinstance(attr_value, nnx.Module):
                    _quantize_moe_recursive(attr_value, visited)
                elif isinstance(attr_value, (list, tuple)):
                    for item in attr_value:
                        if isinstance(item, nnx.Module):
                            _quantize_moe_recursive(item, visited)
    
    _quantize_moe_recursive(model)
    return model


def prepare_inputs_for_quantization(
    model_config: ModelConfig, model_runner
) -> tuple[ForwardBatch, MHATokenToKVPool, LogitsMetadata]:
    bs = 1
    max_seq_len = 4096
    num_tokens = model_config.vocab_size
    num_kv_heads = model_config.get_total_num_kv_heads_with_replication(
        model_runner.mesh.shape["tensor"]
    )
    head_dim = model_config.head_dim
    vocab_size = model_config.vocab_size
    layer_num = model_config.num_hidden_layers
    kv_cache_dtype = None
    if model_runner.server_args.kv_cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    elif model_runner.server_args.kv_cache_dtype == "bf16":
        kv_cache_dtype = jnp.bfloat16

    model_worker_batch = generate_mock_model_worker_batch(
        bs,
        num_tokens,
        ForwardMode.DECODE,
        vocab_size,
        max_seq_len,
        model_runner.page_size,
    )
    token_to_kv_pool = MHATokenToKVPool(
        size=model_runner.page_size * DEFAULT_NUM_PAGES,
        page_size=model_runner.page_size,
        dtype=kv_cache_dtype,
        head_num=num_kv_heads,
        head_dim=head_dim,
        layer_num=layer_num,
        mesh=model_runner.mesh,
    )
    logits_metadata = LogitsMetadata.from_model_worker_batch(model_worker_batch, model_runner.mesh)
    model_runner.attn_backend.forward_metadata = model_runner.attn_backend.get_forward_metadata(
        model_worker_batch
    )
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)

    return forward_batch, token_to_kv_pool, logits_metadata

@functools.partial(jax.jit, static_argnums=(1,2))
def quantize_tensor_simple(x: jax.Array, dtype: jnp.dtype, dim: int = -1, out_sharding = None):
    
    @auto_axes
    def _quantize_tensor(x: jax.Array):
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
        x_q = jnp.clip(x / scale, min_val, max_val).astype(dtype)
        return x_q, scale.astype(jnp.float32)

    return _quantize_tensor(x, out_sharding=out_sharding)


def quantize_tensor(
    dtype: jnp.dtype,
    tensor: jax.Array,
    axis: int | tuple | None = -1,
    block_size: int | None = None,
    pad_tensor: bool = False,
    out_sharding: P = None,
) -> tuple[jax.Array, jax.Array]:
    """Quantize tensor.

    Args:
        dtype: dtype to perform quantization.
        tensor: Unquantized tensor
        axis: Axis to perform quantization. None denotes per-tensor.
        block_size: Specify block quantization size.
        pad_tensor: Whether to pad the axis along block size.
        out_sharding: Optional output sharding specification for the quantized tensor.
                      The scale sharding is automatically derived by removing the
                      squeezed axes from this spec.

    Returns:
        Tuple of (quantized tensor, scale).
    """
    # Normalize axis outside the inner function to avoid closure reassignment issues
    if axis is None:
        axis_normalized = list(range(tensor.ndim))
    elif isinstance(axis, int):
        axis_normalized = [axis]
    else:
        axis_normalized = list(axis)
    
    # Convert negative axes to positive
    axis_normalized = [a if a >= 0 else tensor.ndim + a for a in axis_normalized]

    # Normalize block_size
    if block_size is not None:
        if isinstance(block_size, int):
            block_size_normalized = [block_size] * len(axis_normalized)
        else:
            block_size_normalized = list(block_size)
    else:
        block_size_normalized = None

    # Get dtype info
    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
    else:
        dtype_info = jnp.finfo(dtype)
    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    # Derive scale sharding by removing the squeezed axes from tensor sharding
    if out_sharding is not None:
        tensor_sharding = out_sharding
        # Scale has axes squeezed out - remove those from the sharding spec
        scale_spec_list = [s for i, s in enumerate(out_sharding) if i not in axis_normalized]
        scale_sharding = P(*scale_spec_list)
        combined_sharding = (tensor_sharding, scale_sharding)
    else:
        combined_sharding = None

    @auto_axes
    def _quantize_tensor(x: jax.Array):
        orig_shape = x.shape
        mask = jnp.ones_like(x, jnp.int32)
        quant_axis = axis_normalized

        if block_size_normalized is not None:
            blocked_shape = [[i] for i in orig_shape]
            pad_width = [[0, 0] for _ in range(x.ndim)]
            for i, block in zip(quant_axis, block_size_normalized):
                num_blocks = (x.shape[i] + block - 1) // block
                padding_size = num_blocks * block - x.shape[i]
                if padding_size and not pad_tensor:
                    raise ValueError(
                        f"Unable to perform block quantization. axis={i} of "
                        f"{x.shape=} is not divisible by {block=}")

                # Pad the tensor to align with block size.
                pad_width[i][1] = padding_size
                blocked_shape[i] = (num_blocks, block)

            # In order to avoid padded values affecting scale value, we pad it
            # using edge value of the tensor.
            x = jnp.pad(x, pad_width, "edge")
            mask = jnp.pad(mask, pad_width)

            padded_shape = x.shape
            # Convert all axis into positive values.
            quant_axis = sorted([i % x.ndim for i in quant_axis])
            # Shift axis by 1 since its original position is now occupied by
            # num_blocks dim. Also, if n axes before an axis was also quantized,
            # shift its position by n.
            quant_axis = [1 + n + i for n, i in enumerate(quant_axis)]

            # Flatten list of lists that contains (num_blocks, block).
            blocked_shape = list(itertools.chain(*blocked_shape))
            x = x.reshape(blocked_shape)
        else:
            padded_shape = orig_shape

        abs_max = jnp.max(jnp.abs(x), axis=quant_axis, keepdims=True)
        scale = abs_max / dtype_max

        x_q = jnp.clip(x / scale, dtype_min, dtype_max)
        x_q = x_q.reshape(padded_shape)
        x_q = x_q.astype(dtype)

        # To avoid padded values affecting output of quantized matmul, we mask them
        # out with 0s.
        x_q = jnp.where(mask, x_q, 0)

        scale = jnp.squeeze(scale, quant_axis).astype(jnp.float32)
        return x_q, scale

    return _quantize_tensor(tensor, out_sharding=combined_sharding)

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
                    f"{tensor_q.shape=} is not divisible by {num_blocks=}", )
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


def generate_mock_model_worker_batch(
    bs: int,
    num_tokens: int,
    mode: ForwardMode,
    vocab_size: int,
    max_seq_len: int,
    page_size: int = 1,
    do_penalties: bool = False,
    speculative_algotithm=None,
) -> ModelWorkerBatch:

    # calculate the page size
    max_cache_loc_size = (bs * max_seq_len + page_size - 1) // page_size * page_size
    valid_input_ids = np.array([1] * bs, dtype=jnp.int32)
    invalid_input_ids = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
    valid_out_cache_loc = np.arange(bs, dtype=jnp.int32)
    invalid_out_cache_loc = np.array([-1] * (num_tokens - bs), dtype=jnp.int32)
    valid_positions = np.array([0] * bs, dtype=jnp.int32)
    invalid_positions = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
    invalid_cache_loc_size = max_cache_loc_size - bs
    if invalid_cache_loc_size < 0:
        raise ValueError(f"padding cache_loc_size {invalid_cache_loc_size} < 0!")

    valid_cache_loc = np.arange(bs)
    invalid_cache_loc = np.array([0] * (invalid_cache_loc_size), dtype=jnp.int32)

    return ModelWorkerBatch(
        bid=1,
        forward_mode=mode,
        input_ids=np.concat([valid_input_ids, invalid_input_ids], axis=0),
        real_input_ids_len=len(valid_input_ids),
        real_bs=bs,
        req_pool_indices=np.arange(bs, dtype=np.int32),
        seq_lens=np.array([1] * bs, dtype=np.int32),
        out_cache_loc=np.concat([valid_out_cache_loc, invalid_out_cache_loc], axis=0),
        return_logprob=False,
        return_output_logprob_only=True,
        sampling_info=(
            SamplingBatchInfo.generate_for_precompile(bs, vocab_size)
            if speculative_algotithm is None
            else SamplingBatchInfo.generate_for_precompile_all_greedy(bs, vocab_size)
        ),
        extend_input_logprob_token_ids=None,
        positions=np.concat([valid_positions, invalid_positions], axis=0),
        extend_start_loc=np.arange(bs, dtype=np.int64),
        cache_loc=np.concat([valid_cache_loc, invalid_cache_loc], axis=0),
        extend_prefix_lens=(np.array([0] * bs) if mode == ForwardMode.EXTEND else None),
        extend_seq_lens=np.array([1] * bs) if mode == ForwardMode.EXTEND else None,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        spec_algorithm=speculative_algotithm,
    )
