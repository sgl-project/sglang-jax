"""Abstract Qwen3 decode inputs for the initial offline export implementation.

Like MaxText train_compile, construct shaped state using the real model and
weight mappings, then compile the serving function without executing it.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import Qwen3Config

from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.model_executor.model_forward import make_jitted_run_model
from sgl_jax.srt.models.qwen3 import Qwen3ForCausalLM
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def build_mesh(options):
    if options.target == "cpu":
        devices = jax.devices("cpu")
    else:
        from jax.experimental.topologies import get_topology_desc

        # MaxText accelerator_to_spec_map uses these libtpu topology names and
        # host bounds. User-facing names such as v6e-1 are not topology strings.
        topology_name, host_bounds = {
            "v6e-1": ("v6e:1x1", (1, 1, 1)),
            "v6e-4": ("v6e:2x2", (2, 2, 1)),
        }[options.topology]
        devices = get_topology_desc(
            platform="tpu",
            topology_name=topology_name,
            chip_config_name="default",
            chips_per_host_bounds=host_bounds,
            num_slices=1,
            wrap=(False, False, False),
        ).devices
    if len(devices) != options.tp_size:
        raise ValueError(f"Target has {len(devices)} devices, but tp_size={options.tp_size}")
    return create_device_mesh([1, options.tp_size], [1, 1], devices=devices, is_full_topology=True)


def load_config(options):
    if options.model_config:
        config = Qwen3Config.from_json_file(options.model_config)
    else:
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
    if config.model_type != "qwen3" or getattr(config, "quantization_config", None):
        raise ValueError("This PoC supports unquantized Qwen3 only")
    if getattr(config, "use_sliding_window", False):
        raise ValueError("Sliding-window Qwen3 is outside this PoC")
    for value in (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size,
        config.intermediate_size,
    ):
        if value % options.tp_size:
            raise ValueError("Model dimensions must be divisible by tp_size (no head replication)")
    if config.head_dim != 128:
        raise ValueError("This PoC requires head_dim=128 to match the serving KV layout")
    if options.context_length > config.max_position_embeddings:
        raise ValueError("context_length exceeds max_position_embeddings")
    return config


def build_inputs(options, mesh):
    config = load_config(options)

    def init_model():
        model = Qwen3ForCausalLM(config, mesh, dtype=jnp.bfloat16)
        # Trace the existing dummy loader: applies real weight mappings without
        # materializing zero weights. The resulting weights remain JIT inputs.
        model.load_weights(SimpleNamespace(_dummy_mode=True))
        return model

    def bind_sharding(value):
        if not isinstance(value, jax.ShapeDtypeStruct):
            return value
        spec = getattr(value.sharding, "spec", P())
        return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=NamedSharding(mesh, spec))

    with jax.set_mesh(mesh):
        model = nnx.eval_shape(init_model)
        model_def, model_state = nnx.split(model)
        model_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)
        model_leaves = [bind_sharding(x) for x in model_leaves]
        pool = MHATokenToKVPool(
            size=options.kv_capacity,
            page_size=options.page_size,
            dtype=jnp.bfloat16,
            head_num=config.num_key_value_heads,
            head_dim=config.head_dim,
            layer_num=config.num_hidden_layers,
            mesh=mesh,
            abstract=True,
        )

    def vector(length):
        return jax.ShapeDtypeStruct((length,), jnp.int32, sharding=NamedSharding(mesh, P("data")))

    backend = NativeAttention(config.num_attention_heads, config.num_key_value_heads, mesh)
    batch_size = options.batch_size
    padded_context = (
        (options.context_length + options.page_size - 1) // options.page_size * options.page_size
    )
    batch = ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=vector(batch_size),
        req_pool_indices=vector(batch_size),
        seq_lens=vector(batch_size),
        out_cache_loc=vector(batch_size),
        positions=vector(batch_size),
        attn_backend=backend,
        cache_loc=vector(batch_size * padded_context),
        spec_algorithm=SpeculativeAlgorithm.NONE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    logits = LogitsMetadata(
        forward_mode=ForwardMode.DECODE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        logits_indices=vector(batch_size),
    )
    args = (
        model_def,
        model_state_def,
        model_leaves,
        batch,
        MemoryPools(token_to_kv_pool=pool),
        logits,
    )
    return make_jitted_run_model(backend), args, config
