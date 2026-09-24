"""Compile the serving model bucket plan without weights, devices, or an HTTP server."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb.expert_location import set_global_server_args
from sgl_jax.srt.model_executor.aot_inputs import (
    AbstractModel,
    AbstractSampler,
    build_mesh,
)
from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import compilation_target

logger = logging.getLogger(__name__)


def export_server(server_args):
    """Resolve ServerArgs once and compile the same model shapes as online warmup."""
    server_args.check_server_args()
    if server_args.aot_model_dir or server_args.disable_precompile:
        raise ValueError(
            "--save-aot cannot be combined with --aot-model-dir or --disable-precompile"
        )
    if server_args.device != "tpu":
        raise ValueError("--save-aot uses a CPU host and requires --device tpu as its target")
    if not (server_args.aot_topology or server_args.aot_topology_name):
        raise ValueError("--save-aot requires --aot-topology or --aot-topology-name")
    if server_args.aot_topology_name and not server_args.aot_host_bounds:
        raise ValueError("--aot-topology-name requires --aot-host-bounds X Y Z")
    if server_args.aot_host_bounds and any(n <= 0 for n in server_args.aot_host_bounds):
        raise ValueError("--aot-host-bounds must contain positive chip counts")
    if not server_args.max_total_tokens or server_args.max_total_tokens < server_args.page_size:
        raise ValueError(
            "CPU AOT needs --max-total-tokens (per DP rank, at least one page); "
            "available TPU HBM cannot be queried on the compilation host"
        )
    if server_args.tp_size % server_args.dp_size:
        raise ValueError("tp_size must be divisible by dp_size")
    if server_args.speculative_algorithm:
        raise ValueError(
            "Automatic AOT export needs a speculative serving warmup adapter; "
            "use sgl_jax.compile for individual draft/verify forwards"
        )
    if (
        server_args.enable_lora
        or server_args.enable_static_lora
        or server_args.enable_recurrent_extra_buffer
    ):
        raise ValueError(
            "Automatic AOT export needs input builders for LoRA/recurrent extra buffers"
        )
    if server_args.ep_dispatch_algorithm:
        raise ValueError("CPU AOT cannot initialize runtime expert placement metadata")
    if server_args.multimodal:
        raise ValueError("Automatic AOT export needs serving vision input builders")
    if jax.default_backend() != "cpu":
        raise ValueError("Start --save-aot in a fresh CPU process (JAX_PLATFORMS=cpu)")
    output = Path(server_args.save_aot)
    if output.exists() and any(output.iterdir()):
        raise ValueError("AOT output directory must be new or empty")
    config = ModelConfig.from_server_args(server_args)
    if config.is_multimodal:
        raise ValueError("Automatic AOT export needs serving vision input builders")
    config.configure_for_serving(server_args)
    set_global_server_args(server_args)
    options = SimpleNamespace(
        target="tpu",
        topology=server_args.aot_topology,
        topology_name=server_args.aot_topology_name,
        host_bounds=server_args.aot_host_bounds,
        tp_size=server_args.tp_size,
        dp_size=server_args.dp_size,
        ep_size=server_args.ep_size,
        attention_backend=server_args.attention_backend,
        moe_backend=config.moe_backend.value,
        context_length=config.context_len,
        page_size=server_args.page_size,
        kv_capacity=server_args.max_total_tokens,
        recurrent_capacity=server_args.max_recurrent_state_size,
        batch_size=server_args.max_running_requests or server_args.dp_size,
        mtp_layer_idx=0,
        draft_token_num=None,
        compiler_options={},
    )
    mesh = build_mesh(options)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "buckets": [], "sampling": []}
    manifest_path = output / "serving.json"
    try:
        with compilation_target(mesh), jax.set_mesh(mesh):
            model = AbstractModel(config, options, mesh, server_args=server_args)
            resources = model.resources
            max_running = CompilationManager.resolve_max_running_requests(
                server_args,
                config.context_len,
                resources.attn_backend,
                resources.max_total_num_tokens,
                resources.max_num_reqs,
                config.moe_backend.value,
            )
            max_bs, max_tokens = CompilationManager.get_max_padded_size(server_args, max_running)
            max_req_len = min(
                config.context_len - 1, resources.max_total_num_tokens // server_args.dp_size - 1
            )
            if max_req_len <= 5:
                raise ValueError("Memory pool size is too small for serving")
            manager = CompilationManager(
                server_args,
                max_bs,
                max_tokens,
                server_args.dp_size,
                server_args.tp_size,
                server_args.page_size,
                max_req_len,
                config.vocab_size,
                max_total_num_tokens=(
                    resources.max_total_num_tokens if server_args.attention_backend == "tt" else 0
                ),
                moe_backend=config.moe_backend.value,
            )
            manifest.update(
                token_buckets=manager.token_buckets,
                batch_buckets=manager.bs_buckets,
                max_total_num_tokens=resources.max_total_num_tokens,
                max_total_tokens=server_args.max_total_tokens,
                max_recurrent_state_size=server_args.max_recurrent_state_size,
                max_running_requests=max_running,
                max_req_len=max_req_len,
            )
            sampler_options = getattr(resources.attn_backend, "sampler_compiler_options", None)
            sampler = AbstractSampler(mesh, server_args.random_seed, sampler_options)
            for mode, workload in ((ForwardMode.EXTEND, "prefill"), (ForwardMode.DECODE, "decode")):
                for bs, tokens, cache_loc in manager.iter_model_shapes(mode):
                    options.workload = workload
                    options.batch_size = bs
                    options.num_tokens = tokens if mode.is_extend() else None
                    options.chunked_prefill_size = (
                        max_tokens // server_args.dp_size if mode.is_extend() else None
                    )
                    options.cache_loc_size = cache_loc
                    fn, args, _, _ = model.build_inputs(options)
                    compiler_options = CompilationManager.compiler_options(
                        args[3].attn_backend, args[3]
                    )
                    directory = output / f"{workload}-bs{bs}-tokens{tokens}"
                    directory.mkdir()
                    logger.info("[aot-model] compiling %s", directory.name)
                    lowered = fn.lower(*args)
                    compiled = CompilationManager.get_executable(
                        lowered, mesh, compiler_options, output=directory
                    )
                    if mode.is_decode():
                        logits = _abstract_outputs(lowered, compiled)[0]
                        _export_sampling(
                            sampler, logits, manager, mesh, sampler_options, output, manifest
                        )
                    manifest["buckets"].append(
                        {
                            "workload": workload,
                            "batch_size": bs,
                            "num_tokens": tokens,
                            "directory": directory.name,
                        }
                    )
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"status": "complete", "output": str(output), "buckets": len(manifest["buckets"])}
        ),
        flush=True,
    )


def _abstract_outputs(lowered, compiled):
    return jax.tree.map(
        lambda value, sharding: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding),
        lowered.out_info,
        compiled.output_shardings,
    )


def _export_sampling(sampler, logits, manager, mesh, compiler_options, output, manifest):
    from sgl_jax.srt.layers.sampler import jitted_compute_logprobs

    bs = logits.next_token_logits.shape[0]
    batch = manager._make_dummy_batch(bs, bs, ForwardMode.DECODE, bs)
    # ModelRunner.sample runs outside the explicit model mesh. Match that
    # context; input arrays still carry their concrete TPU shardings.
    with jax.set_mesh(None):
        for seeded in (False, True):
            batch.sampling_info.sampling_seeds = np.zeros(bs, dtype=np.int32) if seeded else None
            fn, args = sampler.build_inputs(logits, batch)
            directory = output / f"sampler-bs{bs}-{'seeded' if seeded else 'unseeded'}"
            directory.mkdir()
            logger.info("[aot-model] compiling %s", directory.name)
            lowered = fn.lower(*args)
            compiled = CompilationManager.get_executable(
                lowered, mesh, compiler_options, output=directory
            )
            manifest["sampling"].append({"batch_size": bs, "directory": directory.name})
            if not seeded:
                (next_tokens, logprobs, _), _ = _abstract_outputs(lowered, compiled)
                directory = output / f"compute-logprobs-bs{bs}"
                directory.mkdir()
                CompilationManager.get_executable(
                    jitted_compute_logprobs.lower(mesh, logprobs, next_tokens),
                    mesh,
                    output=directory,
                )
                manifest["sampling"].append({"batch_size": bs, "directory": directory.name})
