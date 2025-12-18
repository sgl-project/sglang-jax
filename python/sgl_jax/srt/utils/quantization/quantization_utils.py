# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import os
from typing import TYPE_CHECKING, Callable, List

import jax
import jax.numpy as jnp
import qwix
import qwix.pallas as qpl
import yaml
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from qwix._src.core.qarray import QArray
from qwix._src.providers import ptq
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
)
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sgl_jax.srt.configs.model_config import ModelConfig
import numpy as np
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool



QUANTIZATION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs")


def parse_qwix_config_to_rules(
        qwix_config: List[dict]) -> List[qwix.QuantizationRule]:
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


def qwix_quantize_nnx_model(qwix_config: List[dict],
                            model_worker_batch: ModelWorkerBatch,
                            attn_backend,
                            mesh: Mesh,
                            kv_head_num: int,
                            head_dim: int,
                            layer_num: int,
                            model: nnx.Module = None,
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
        rng: the random number generator to use
        mesh: the mesh to use
        num_hidden_layers: the number of hidden layers in the model
        kv_cache_page_size: the page size of the kv cache
        kv_cache_num_kv_heads: the number of kv heads
        head_size: the head size of the kv cache
        kv_cache_dtype: the dtype of the kv cache

    Returns:
        model: the quantized model
    """
    token_to_kv_pool = MHATokenToKVPool(size=2000, page_size=1, dtype=jnp.bfloat16, 
                                        head_num=kv_head_num,
                                        head_dim=head_dim, layer_num=layer_num,
                                        mesh=mesh)
    qwix_rules = parse_qwix_config_to_rules(qwix_config)
    logits_metadata = LogitsMetadata.from_model_worker_batch(model_worker_batch, mesh)
    attn_backend.forward_metadata = attn_backend.get_forward_metadata(model_worker_batch)
    forward_batch = ForwardBatch.init_from_batch(model_worker_batch, mesh, attn_backend)
    model_input = {
        "forward_batch": forward_batch,
        "token_to_kv_pool": token_to_kv_pool,
        "logits_metadata": logits_metadata,
    }
    model = qwix.quantize_model(model, qwix.PtqProvider(qwix_rules),
                                    **model_input)
    return model


def quantization_config_file_path_to_dict(
        quantization_config_file_path: str) -> dict:
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
            with open(path, "r") as f:
                return yaml.safe_load(f)
    raise ValueError(
        f"Could not find quantization config file with name '{quantization_config_file_path}' in 'sgl_jax/srt/utils/quantization/configs."
    )


def apply_qwix_quantization(
        model_config: ModelConfig, model: nnx.Module, mesh: Mesh) -> nnx.Module:
        from sgl_jax.srt.layers.attention.flashattention_backend import (
                    FlashAttention,
)
        qwix_config_dict = quantization_config_file_path_to_dict(
                os.path.join(model_config.quantization_config_path)
        )
        qwix_config = qwix_config_dict.get("qwix").get("rules")
        
        bs = 1
        max_seq_len = 4096
        page_size = 1
        num_tokens = model_config.vocab_size
        num_attn_heads = model_config.num_attention_heads
        num_kv_heads = model_config.get_total_num_kv_heads_with_replication(mesh.shape["tensor"])
        head_dim = model_config.head_dim
        vocab_size = model_config.vocab_size
        layer_num = model_config.num_hidden_layers
        
        
        model_worker_batch = generate_mock_model_worker_batch(bs, num_tokens, ForwardMode.DECODE, vocab_size, max_seq_len)
        # prepare attn_backend
        attn_backend = FlashAttention(
            num_attn_heads=num_attn_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            mesh=mesh,
        )
        
        qwix_quantize_nnx_model_with_config_and_attn_backend = functools.partial(
            qwix_quantize_nnx_model, qwix_config=qwix_config, model_worker_batch=model_worker_batch)
        with jax.set_mesh(mesh):
            model = nnx.jit(
                    qwix_quantize_nnx_model_with_config_and_attn_backend,
                    static_argnames=(
                        "mesh",
                        "kv_head_num",
                        "head_dim",
                        "layer_num",
                    ))(
                    attn_backend=attn_backend,
                    mesh=mesh,
                    kv_head_num=num_kv_heads,
                    head_dim=(head_dim + 127) // 128 * 128,
                    layer_num=layer_num,
                    model=model)
        return model


def manually_quantize_qwix_weight(weight: jax.Array, qtype: jnp.dtype,
                                  channelwise_axes: List[int],
                                  tiled_axes: dict,
                                  calibration_method: str) -> QArray:
    """
    Manually quantizes a weight tensor using Qwix.  Only needed for the SparseMatmul DeepSeek case right now, since
    otherwise, Qwix will handle this automatically (through our application of `qwix.quantize_model`).
    """
    # TODO (jacobplatin): clean this up; this is needed because of issues with Qwix quantizing the `shard_map` in SpraseMatmul
    how_to_quantize = ptq.qarray.HowToQuantize(
        qtype=qtype,
        channelwise_axes=channelwise_axes,
        tiled_axes=tiled_axes,
        calibration_method=calibration_method)

    return ptq.create_quantized_param(weight, how_to_quantize)


def manually_quantize_qwix_activation(inputs: jax.Array, rule_name: str,
                                      qtype: jnp.dtype,
                                      channelwise_axes: List[int],
                                      tiled_axes: dict,
                                      calibration_method: str) -> QArray:
    """
    Manually quantizes an activation tensor using Qwix.  Needed for the SparseMatmul
    DeepSeek MoE case currently.

    Args:
        inputs: The activation tensor to quantize.
        rule_name: The name of the quantization rule to use.
        qtype: The quantization type.
        channelwise_axes: The channelwise axes to quantize.
        tiled_axes: The tiled axes to quantize.
        calibration_method: The calibration method to use.

    Returns:
        The quantized activation tensor.
    """
    rule = qpl.get_current_rule(rule_name)
    lhs_how = ptq.qarray.HowToQuantize(qtype=qtype,
                                       channelwise_axes=channelwise_axes,
                                       tiled_axes=tiled_axes,
                                       calibration_method=calibration_method)
    # This is needed because we aren't passing `act_name` right now
    assert not rule.act_static_scale, "Static scale not supported right now"

    # channelwise_axes should be set to (a subset of) non-contraction axes. e.g.
    # for ragged_dot [m, k] x [g, k, n], they are [0] and [0, 2]
    # TODO (jacobplatin): add support for `act_name`
    return ptq.quantize_act(inputs, lhs_how, rule, "")


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
                else SamplingBatchInfo.generate_for_precompile_all_greedy(
                    bs, vocab_size
                )
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
