"""Lower the shared serving forward and retain compiler artifacts, without running it."""

import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import jax

from sgl_jax.srt.model_executor.aot_dispatch import (
    decode_no_sc_gather_compiler_options_fn,
)
from sgl_jax.srt.model_executor.aot_inputs import build_inputs, build_mesh
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import compilation_target, is_tpu_runtime


def _versions():
    result = {}
    for name in ("jax", "jaxlib", "flax", "libtpu", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _source():
    root = Path(__file__).resolve().parents[4]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    # Include uncommitted/new Python files too: HEAD alone does not identify the export source.
    files = {}
    for path in sorted((root / "python/sgl_jax").rglob("*.py")):
        files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"revision": revision, "python_source_sha256": digest}


def _array_signature(tree):
    result = []
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if hasattr(value, "shape"):
            result.append(
                {
                    "path": jax.tree_util.keystr(path),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sharding": str(getattr(value, "sharding", None)),
                }
            )
    return result


def _compiler_options(options, batch):
    # Reuse serving backend options and the per-forward decode workaround. CLI
    # options override these defaults and the manifest records the final values.
    result = dict(getattr(batch.attn_backend, "compiler_options", None) or {})
    if is_tpu_runtime() and get_bool_env_var("SGLANG_JAX_ENABLE_KERNEL_LOG_RECORDER"):
        result["xla_tpu_enable_log_recorder"] = "true"
    decode_options = decode_no_sc_gather_compiler_options_fn()
    if decode_options is not None:
        result.update(decode_options((batch,)) or {})
    result.update(options.compiler_options)
    return result


def _custom_calls(hlo):
    """Inventory actual custom-call instructions, not kernel names elsewhere in IR."""
    calls = []
    for line in hlo.splitlines():
        target = re.search(r'custom_call_target=("(?:[^"\\]|\\.)*")', line)
        if target is None:
            continue
        instruction = re.match(r"\s*(?:ROOT )?(%\S+)\s*=", line)
        op_name = re.search(r'op_name=("(?:[^"\\]|\\.)*")', line)
        calls.append(
            {
                "instruction": instruction.group(1) if instruction else None,
                "target": json.loads(target.group(1)),
                "op_name": json.loads(op_name.group(1)) if op_name else None,
            }
        )
    return calls


def export(options):
    output = options.output
    manifest = {
        "schema_version": 1,
        "status": "running",
        "scope": f"Synthetic model forward, {options.workload}, no checkpoint loading or execution",
        "executed": False,
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(options).items()
        },
        "versions": _versions(),
        "source": _source(),
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "XLA_FLAGS",
                "LIBTPU_INIT_ARGS",
                "JAX_PLATFORMS",
                "PALLAS_INTERPRET",
                "SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD",
                "SGLANG_JAX_ENABLE_KERNEL_LOG_RECORDER",
            )
        },
        "stages": {},
    }

    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    started = time.monotonic()
    try:
        # A cache hit may skip codegen and therefore omit LLO dumps.
        jax.config.update("jax_enable_compilation_cache", False)
        mesh = build_mesh(options)
        with compilation_target(mesh):
            fn, args, config, workload = build_inputs(options, mesh)
            manifest["model_config"] = config.to_dict()
            manifest["model_config"]["model_type"] = config.model_type
            quantization = getattr(config, "quantization_config", None)
            manifest["quantization"] = quantization.to_dict() if quantization is not None else None
            manifest["parameter_dtypes"] = {}
            for value in args[2]:
                dtype = str(value.dtype)
                stats = manifest["parameter_dtypes"].setdefault(
                    dtype, {"tensors": 0, "elements": 0, "bytes": 0}
                )
                elements = math.prod(value.shape)
                stats["tensors"] += 1
                stats["elements"] += elements
                stats["bytes"] += elements * value.dtype.itemsize
            if options.model_config:
                original = Path(options.model_config).read_bytes()
                (output / "source_config.json").write_bytes(original)
                manifest["source_config_sha256"] = hashlib.sha256(original).hexdigest()
            manifest["workload"] = {
                **asdict(workload),
                "requests_per_dp": workload.request_count // workload.dp_size,
                "input_tokens_per_dp": workload.input_token_count // workload.dp_size,
                "model_class": config.architectures[0] if config.architectures else None,
                "model_type": config.model_type,
                "num_hidden_layers": config.num_hidden_layers,
                "attention_backend": options.attention_backend,
                "moe_backend": options.moe_backend,
                "attention_tp_size": options.tp_size // options.dp_size,
                "ep_size": options.ep_size,
                "weights": "synthetic",
            }
            manifest["input_signature"] = _array_signature(args[2:])
            manifest["target_devices"] = [str(device) for device in mesh.devices.flat]
            manifest["mesh"] = dict(mesh.shape)
            manifest["mesh_device_ids"] = [int(d.id) for d in mesh.devices.flat]
            manifest["target_topology"] = [
                {
                    "id": int(d.id),
                    "device_kind": d.device_kind,
                    "process_index": d.process_index,
                    "coords": getattr(d, "coords", None),
                    "core_on_chip": getattr(d, "core_on_chip", None),
                    "slice_index": getattr(d, "slice_index", None),
                }
                for d in mesh.devices.flat
            ]
            manifest["static_args"] = {
                "forward_mode": args[3].forward_mode.name,
                "capture_hidden_mode": args[3].capture_hidden_mode.name,
                "spec_algorithm": args[3].spec_algorithm.name,
            }
            manifest["donate_argnames"] = ["memory_pools"]
            compiler_options = _compiler_options(options, args[3])
            manifest["compiler_options"] = compiler_options
            lowered = fn.lower(*args)
            manifest["output_signature"] = _array_signature(lowered.out_info)
            (output / "stablehlo.mlir").write_text(str(lowered.compiler_ir(dialect="stablehlo")))
            manifest["stages"]["stablehlo"] = "complete"
            save_manifest()
            if options.stage == "compiled":
                compiled = lowered.compile(compiler_options=compiler_options or None)
                if options.save_executable:
                    from sgl_jax.srt.model_executor.aot_executable import save_executable

                    manifest["executable"] = save_executable(
                        compiled, lowered, mesh, compiler_options, output
                    )
                    manifest["stages"]["executable"] = "complete"
                hlo = compiled.as_text()
                if not hlo or "HloModule" not in hlo:
                    raise RuntimeError("Backend did not expose optimized HLO text")
                (output / "optimized_hlo.txt").write_text(hlo)
                manifest["custom_calls"] = _custom_calls(hlo)
                manifest["stages"]["optimized_hlo"] = "complete"
                if options.dump_llo:
                    # libtpu 0.0.46.1 writes LLO pass snapshots as
                    # <timestamp>-<kernel>-<pass-id>-original/post-<pass>.txt.
                    # Reports (bufferinfo, allocinfo, fingerprints) alone do not
                    # establish that LLO was dumped.
                    llo_files = [
                        p
                        for p in (output / "llo").rglob("*")
                        if p.is_file()
                        and p.stat().st_size
                        and (
                            ".llo" in p.name
                            or re.search(r"-\d+-(?:original|post-.+)\.txt$", p.name)
                        )
                    ]
                    if not llo_files:
                        manifest["stages"]["llo"] = "missing"
                        raise RuntimeError(
                            "Compilation finished but requested LLO files were not generated"
                        )
                    manifest["stages"]["llo"] = "complete"
                    manifest["llo_snapshot_count"] = len(llo_files)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["artifacts"] = [
            {
                "path": str(p.relative_to(output)),
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        ]
        save_manifest()
        print(
            json.dumps(
                {"status": manifest["status"], "output": str(output), "stages": manifest["stages"]}
            ),
            flush=True,
        )
