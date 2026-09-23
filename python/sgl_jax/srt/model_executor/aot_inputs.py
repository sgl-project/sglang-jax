"""Build offline inputs through the serving config, model registry and loaders."""

import json
import tempfile
from pathlib import Path

import jax
from flax import nnx
from jax.sharding import NamedSharding
from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_executor.aot_resources import build_resources
from sgl_jax.srt.model_executor.aot_workloads import InputContext, get_input_builder
from sgl_jax.srt.model_executor.model_forward import make_jitted_run_model
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def build_mesh(options):
    if options.target == "cpu":
        devices = jax.devices("cpu")
    else:
        from jax.experimental.topologies import get_topology_desc

        topology_name, host_bounds = options.topology_name, options.host_bounds
        if options.topology:
            # Follow MaxText's compile-only topology and host-bounds approach.
            # Preset suffixes count JAX devices: v7x exposes two per chip.
            topology_name, preset_host_bounds = {
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
            host_bounds = host_bounds or preset_host_bounds
        devices = get_topology_desc(
            platform="tpu",
            topology_name=topology_name,
            chip_config_name="default",
            chips_per_host_bounds=tuple(host_bounds),
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


def load_config(options, model_path, *, is_draft=False):
    if options.model_config:
        raw = json.loads(Path(options.model_config).read_text())
    else:
        # A small example for the no-argument smoke command, not a model selector.
        raw = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 256,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 256,
            "tie_word_embeddings": False,
        }
    if raw.get("quantization_config") and not options.bf16_model:
        raise ValueError(
            "Use --bf16-model to explicitly export a synthetic BF16 variant of a quantized config"
        )
    raw.pop("quantization_config", None)
    # ModelConfig imports the serving custom config registrations. Unknown HF
    # model_type values can still name a registered SGLang architecture, whose
    # constructor consumes the fields of a generic PretrainedConfig.
    kind = raw.get("model_type")
    try:
        config_cls = CONFIG_MAPPING[kind]
    except KeyError:
        config_cls = PretrainedConfig
    hf_config = config_cls.from_dict(raw)
    model_config = ModelConfig(
        model_path=model_path,
        hf_config=hf_config,
        dtype="bfloat16",
        is_draft_model=is_draft,
        moe_backend=options.moe_backend or "epmoe",
    )
    if is_draft:
        count = getattr(hf_config, "num_nextn_predict_layers", None)
        if count is not None and options.mtp_layer_idx >= count:
            raise ValueError("mtp_layer_idx exceeds num_nextn_predict_layers")
        model_config.hf_config.mtp_layer_idx = options.mtp_layer_idx
    model_config.ep_size = options.ep_size
    model_config.hf_config.ep_size = options.ep_size
    model_config.hf_config.moe_backend = model_config.moe_backend.value
    attention_tp = options.tp_size // options.dp_size
    model_config.validate_tensor_parallel_config(attention_tp)
    model_config.configure_for_tensor_parallel(attention_tp)
    if options.context_length > model_config.context_len:
        raise ValueError("context_length exceeds the model's context limit")
    if options.moe_backend == "fused_v2" and options.ep_size != options.tp_size:
        raise ValueError("fused_v2 requires ep_size=tp_size")
    model_config._abstract_mode = True
    return model_config


def _bind_concrete_sharding(value, mesh):
    # Explicit dummy allocations preserve placement through post-load reshapes,
    # splits and expert-mesh transforms. Bind their abstract meshes to the target
    # devices without maintaining a second copy of each model's weight mappings.
    sharding = value.sharding
    if not isinstance(sharding, NamedSharding):
        raise ValueError("Abstract model state has no named sharding")
    layout = sharding.mesh
    concrete = jax.sharding.Mesh(
        mesh.devices.flatten().reshape(tuple(layout.shape.values())),
        layout.axis_names,
        axis_types=layout.axis_types,
    )
    return jax.ShapeDtypeStruct(
        value.shape, value.dtype, sharding=NamedSharding(concrete, sharding.spec)
    )


def build_inputs(options, mesh):
    builder = get_input_builder(options)
    with tempfile.TemporaryDirectory() as empty_checkpoint, jax.set_mesh(mesh):
        is_draft = builder.model_role == "draft"
        model_config = load_config(options, empty_checkpoint, is_draft=is_draft)
        loader = get_model_loader(LoadConfig(load_format="dummy"), mesh)
        model = loader.load_model(model_config=model_config)
        if is_draft:
            # Match the serving draft worker: shared embedding/head layouts come
            # from the target loader, never from an AOT-specific parameter table.
            target_config = load_config(options, empty_checkpoint)
            target = loader.load_model(model_config=target_config)
            if not hasattr(model, "set_embed_and_head") or not hasattr(
                target, "get_embed_and_head"
            ):
                raise ValueError("Draft export requires the serving embed/head sharing interface")
            model.set_embed_and_head(*target.get_embed_and_head())
        model_def, model_state = nnx.split(model)
        model_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)
        model_leaves = [_bind_concrete_sharding(value, mesh) for value in model_leaves]
        backend, memory_pools = build_resources(model_config, model, options, mesh)

    inputs = builder.build(InputContext(model_config, mesh, backend, memory_pools))
    backend.forward_metadata = inputs.attention_metadata
    args = (model_def, model_state_def, model_leaves, inputs.batch, memory_pools, inputs.logits)
    return make_jitted_run_model(backend), args, model_config.hf_config, builder.spec
