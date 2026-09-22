"""Abstract model state and decode inputs for offline IR export.

Like MaxText train_compile, construct shaped state using the real model and
weight mappings, then compile the serving function without executing it.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig, Qwen3Config

from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, MHATokenToKVPool, SWAKVPool
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

        # Follow MaxText's compile-only topology and host-bounds approach.
        # User-facing suffixes count JAX devices: v7x exposes two per chip.
        topology_name, host_bounds = {
            "v6e-1": ("v6e:1x1", (1, 1, 1)),
            "v6e-4": ("v6e:2x2", (2, 2, 1)),
            "v6e-8": ("v6e:2x4", (2, 2, 1)),
            "v6e-16": ("v6e:4x4", (2, 2, 1)),
            "v6e-32": ("v6e:4x8", (2, 2, 1)),
            "v6e-64": ("v6e:8x8", (2, 2, 1)),
            "v7x-8": ("TPU7x:2x2x1", (2, 2, 1)),
            "v7x-16": ("TPU7x:2x2x2", (2, 2, 1)),
            "v7x-32": ("TPU7x:2x2x4", (2, 2, 1)),
            "v7x-64": ("TPU7x:2x4x4", (2, 2, 1)),
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
    return create_device_mesh(
        [options.dp_size, options.tp_size // options.dp_size],
        [1, 1],
        devices=devices,
        is_full_topology=True,
    )


def load_config(options):
    if options.model_config:
        raw = json.loads(Path(options.model_config).read_text())
        config_cls = {
            "qwen3": Qwen3Config,
            "mimo_v2_flash": PretrainedConfig,
            "kimi_linear": KimiLinearConfig,
        }.get(raw.get("model_type"))
        if config_cls is None:
            raise ValueError(f"No offline input builder for model_type={raw.get('model_type')!r}")
        config = config_cls.from_dict(raw)
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
    if getattr(config, "quantization_config", None) and not options.bf16_model:
        raise ValueError(
            "Quantized checkpoints are unsupported; --bf16-model explicitly exports a synthetic BF16 variant"
        )
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    attention_tp = options.tp_size // options.dp_size
    if options.recurrent_capacity is not None and config.model_type != "kimi_linear":
        raise ValueError("recurrent_capacity requires a linear-attention model")
    if config.model_type == "kimi_linear":
        from sgl_jax.srt.model_executor.aot_kimi_inputs import validate_config

        validate_config(config, options)
    elif config.model_type == "mimo_v2_flash":
        if options.attention_backend != "fa" or options.moe_backend != "fused_v2":
            raise ValueError("MiMo requires --attention-backend=fa --moe-backend=fused_v2")
        if options.ep_size != options.tp_size or config.n_routed_experts % options.ep_size:
            raise ValueError("fused_v2 requires ep_size=tp_size and experts divisible by ep_size")
        if options.batch_size % options.ep_size:
            raise ValueError("fused_v2 requires batch_size divisible by ep_size")
        for name in ("hybrid_layer_pattern", "moe_layer_freq"):
            if len(getattr(config, name)) != config.num_hidden_layers:
                raise ValueError(f"{name} must describe every layer")
        for name in ("swa_num_attention_heads", "swa_num_key_value_heads"):
            if getattr(config, name) % attention_tp:
                raise ValueError(f"{name} must be divisible by attention TP")
        config.moe_backend = options.moe_backend
        config.ep_size = options.ep_size
    elif (
        options.moe_backend
        or options.ep_size != 1
        or options.attention_backend != "native"
        or options.dp_size != 1
    ):
        raise ValueError("Qwen3 export requires native attention, DP=1, EP=1 and no MoE backend")
    if getattr(config, "use_sliding_window", False):
        raise ValueError("Offline export does not yet support sliding-window Qwen3")
    for value in (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size,
        config.intermediate_size,
    ):
        if value % attention_tp:
            raise ValueError(
                "Model dimensions must be divisible by attention TP (no head replication)"
            )
    if config.model_type == "qwen3" and config.head_dim != 128:
        raise ValueError("Qwen3 export requires head_dim=128 to match the serving KV layout")
    max_context = getattr(config, "max_position_embeddings", None) or getattr(
        config, "model_max_length", None
    )
    if max_context is not None and options.context_length > max_context:
        raise ValueError("context_length exceeds the model's context limit")
    return config


def build_inputs(options, mesh):
    config = load_config(options)

    def init_model():
        if config.model_type == "mimo_v2_flash":
            from sgl_jax.srt.models.mimo_v2_flash import MiMoV2FlashForCausalLM

            model_cls = MiMoV2FlashForCausalLM
        elif config.model_type == "kimi_linear":
            from sgl_jax.srt.models.kimi_linear import KimiLinearForCausalLM

            model_cls = KimiLinearForCausalLM
        else:
            model_cls = Qwen3ForCausalLM
        model = model_cls(config, mesh, dtype=jnp.bfloat16)
        # Trace the existing dummy loader: applies real weight mappings without
        # materializing zero weights. The resulting weights remain JIT inputs.
        with tempfile.TemporaryDirectory() as empty_checkpoint:
            model.load_weights(
                SimpleNamespace(
                    _dummy_mode=True,
                    model_path=empty_checkpoint,
                    quantization_config=None,
                    hf_config=config,
                    ep_size=options.ep_size,
                )
            )
        return model

    with jax.set_mesh(mesh):
        model = nnx.eval_shape(init_model)
        mappings = (
            model._create_weight_mappings()
            if config.model_type != "qwen3"
            else model._create_qwen3_weight_mappings()
        )
        parameter_specs = {
            (m.target_path if isinstance(m.target_path, str) else m.target_path[0]): P(*m.sharding)
            for m in mappings.values()
        }
        parameter_meshes = {}
        if config.model_type == "kimi_linear":
            from sgl_jax.srt.model_executor.aot_kimi_inputs import bind_parameter_specs

            parameter_meshes = bind_parameter_specs(model, config, parameter_specs)
        model_def, model_state = nnx.split(model)
        leaves_with_paths, model_state_def = jax.tree_util.tree_flatten_with_path(model_state)
        model_leaves = []
        for path, value in leaves_with_paths:
            # Nested dummy-loader JITs may expose replicated *tracer* shardings
            # despite explicit output shardings. As in MaxText, bind abstract
            # shapes to independently derived parameter shardings; never infer
            # the checkpoint mapping from eval_shape's output placement.
            name = ".".join(str(key.key) for key in path[:-1])
            if name not in parameter_specs:
                raise ValueError(f"No offline weight sharding mapping for {name}")
            model_leaves.append(
                jax.ShapeDtypeStruct(
                    value.shape,
                    value.dtype,
                    sharding=NamedSharding(parameter_meshes.get(name, mesh), parameter_specs[name]),
                )
            )
        pool_kwargs = dict(
            size=options.kv_capacity,
            page_size=options.page_size,
            dtype=jnp.bfloat16,
            head_num=config.num_key_value_heads,
            head_dim=(config.head_dim + 127) // 128 * 128,
            mesh=mesh,
            dp_size=options.dp_size,
            abstract=True,
        )
        if config.model_type == "kimi_linear":
            from sgl_jax.srt.model_executor.aot_kimi_inputs import build_resources

            backend, memory_pools = build_resources(config, options, mesh)
        elif config.model_type == "mimo_v2_flash":
            pool = SWAKVPool(
                **pool_kwargs,
                size_swa=options.kv_capacity,
                swa_attention_layer_ids=[
                    i for i, swa in enumerate(config.hybrid_layer_pattern) if swa
                ],
                full_attention_layer_ids=[
                    i for i, swa in enumerate(config.hybrid_layer_pattern) if not swa
                ],
                swa_head_num=config.swa_num_key_value_heads,
                swa_head_dim=(config.swa_head_dim + 127) // 128 * 128,
            )
        else:
            pool = MHATokenToKVPool(**pool_kwargs, layer_num=config.num_hidden_layers)
        if config.model_type != "kimi_linear":
            memory_pools = MemoryPools(token_to_kv_pool=pool)

    def vector(length):
        return jax.ShapeDtypeStruct((length,), jnp.int32, sharding=NamedSharding(mesh, P("data")))

    batch_size = options.batch_size
    padded_context = (
        (options.context_length + options.page_size - 1) // options.page_size * options.page_size
    )
    if config.model_type == "kimi_linear":
        recurrent_indices = backend.linear_attn_backend.forward_metadata.recurrent_indices
    elif options.attention_backend == "fa":
        from sgl_jax.srt.layers.attention.flashattention_backend import (
            FlashAttention,
            FlashAttentionMetadata,
        )

        backend = FlashAttention(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            page_size=options.page_size,
            mesh=mesh,
        )
        # Same per-DP shapes as FlashAttention.get_forward_metadata(DECODE).
        # Values remain runtime inputs; no token/page IDs are baked into IR.
        backend.forward_metadata = FlashAttentionMetadata(
            cu_q_lens=vector(batch_size + options.dp_size),
            cu_kv_lens=vector(batch_size + options.dp_size),
            page_indices=vector(batch_size * padded_context // options.page_size),
            swa_page_indices=vector(batch_size * padded_context // options.page_size),
            seq_lens=vector(batch_size),
            distribution=vector(3 * options.dp_size),
        )
        recurrent_indices = None
    else:
        backend = NativeAttention(config.num_attention_heads, config.num_key_value_heads, mesh)
        recurrent_indices = None
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
        recurrent_indices=recurrent_indices,
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
        memory_pools,
        logits,
    )
    return make_jitted_run_model(backend), args, config
