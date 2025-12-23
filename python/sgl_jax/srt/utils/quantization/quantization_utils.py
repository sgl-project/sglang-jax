# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import qwix
import yaml
from flax import nnx
from jax.sharding import Mesh

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
# from sgl_jax.srt.model_executor.model_runner import ModelRunner

QUANTIZATION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs")
DEFAULT_PAGE_SIZE = 1
DEFAULT_MAX_NUM_TOKENS = 2000


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
    model_worker_batch: ModelWorkerBatch,
    model_runner,
    attn_backend,
    kv_head_num: int,
    head_dim: int,
    layer_num: int,
    kv_cache_dtype: jnp.dtype,
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
    token_to_kv_pool = MHATokenToKVPool(
        size=DEFAULT_MAX_NUM_TOKENS,
        page_size=model_runner.page_size,
        dtype=kv_cache_dtype,
        head_num=kv_head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        mesh=model_runner.mesh,
    )
    qwix_rules = parse_qwix_config_to_rules(qwix_config)
    logits_metadata = LogitsMetadata.from_model_worker_batch(model_worker_batch, model_runner.mesh)
    attn_backend.forward_metadata = attn_backend.get_forward_metadata(model_worker_batch)
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    model_input = {
        "forward_batch": forward_batch,
        "token_to_kv_pool": token_to_kv_pool,
        "logits_metadata": logits_metadata,
    }
    model = qwix.quantize_model(model, qwix.PtqProvider(qwix_rules), **model_input)
    return model


def quantization_config_file_path_to_dict(quantization_config_file_path: str) -> dict:
    """
    Converts a quantization config YAML file path to a dictionary.

    The expected format of the quantization config YAML file is as follows:
    ```yaml
        qwix:
            # optional, defaults to False if not specified
            use_abstract_model: True
            rules:
                # NOTE: each entry corresponds to a qwix.QuantizationRule
                - module_path: '.*attn.*'
                weight_qtype: 'int8'
                - module_path: '.*'
                weight_qtype: 'int8'
                act_qtype: 'int8'
    ```

    Args:
        quantization_config_file_path: the path to the quantization config YAML file

    Returns:
        a dictionary containing the quantization config
    """
    all_entries = os.listdir(QUANTIZATION_CONFIG_PATH)
    for filename in all_entries:
        if filename == quantization_config_file_path:
            path = os.path.join(QUANTIZATION_CONFIG_PATH, filename)
            with open(path) as f:
                return yaml.safe_load(f)
    raise ValueError(
        f"Could not find quantization config file with name '{quantization_config_file_path}' in 'sgl_jax/srt/utils/quantization/configs."
    )


def apply_qwix_quantization(model_config: ModelConfig, model: nnx.Module, model_runner) -> nnx.Module:
    """
    Will apply quantization if a valid quantization config with Qwix rules is provided.  See README
    for more details on Qwix.
    """
    from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention

    qwix_config_dict = quantization_config_file_path_to_dict(
        os.path.join(model_config.quantization_config_path)
    )
    qwix_config = qwix_config_dict.get("qwix").get("rules")

    # prepare batch input
    bs = 1
    max_seq_len = 4096
    page_size = 1
    num_tokens = model_config.vocab_size
    num_attn_heads = model_config.num_attention_heads
    num_kv_heads = model_config.get_total_num_kv_heads_with_replication(model_runner.mesh.shape["tensor"])
    head_dim = model_config.head_dim
    vocab_size = model_config.vocab_size
    layer_num = model_config.num_hidden_layers
    kv_cache_dtype = None
    if model_runner.server_args.kv_cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    elif model_runner.server_args.kv_cache_dtype == "bf16":
        kv_cache_dtype = jnp.bfloat16

    model_worker_batch = generate_mock_model_worker_batch(
        bs, num_tokens, ForwardMode.DECODE, vocab_size, max_seq_len
    )
    attn_backend = FlashAttention(
        num_attn_heads=num_attn_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        mesh=model_runner.mesh,
    )

    qwix_quantize_nnx_model_with_config_and_attn_backend = functools.partial(
        qwix_quantize_nnx_model, qwix_config=qwix_config, model_worker_batch=model_worker_batch, model_runner=model_runner,
    )
    with jax.set_mesh(model_runner.mesh):
        model = nnx.jit(
            qwix_quantize_nnx_model_with_config_and_attn_backend,
            static_argnames=(
                "kv_head_num",
                "head_dim",
                "layer_num",
                "kv_cache_dtype",
            ),
            donate_argnames=("model",),  # donate the model to the jitted function to save memory
        )(
            kv_head_num=num_kv_heads,
            head_dim=(head_dim + 127) // 128 * 128,
            layer_num=layer_num,
            kv_cache_dtype=kv_cache_dtype,
            model=model,
            attn_backend=attn_backend,
        )
    return model


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
