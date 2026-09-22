"""Lower the shared serving forward and retain compiler artifacts, without running it."""

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

import jax

from sgl_jax.srt.model_executor.aot_inputs import build_inputs, build_mesh


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


def _input_signature(args):
    result = []
    for path, value in jax.tree_util.tree_flatten_with_path(args[2:])[0]:
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


def export(options):
    output = options.output
    manifest = {
        "schema_version": 1,
        "status": "running",
        "scope": "Synthetic BF16 model forward, decode, no checkpoint loading or execution",
        "executed": False,
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(options).items()
        },
        "versions": _versions(),
        "source": _source(),
        "environment": {
            key: os.environ.get(key, "")
            for key in ("XLA_FLAGS", "LIBTPU_INIT_ARGS", "JAX_PLATFORMS")
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
        fn, args, config = build_inputs(options, mesh)
        manifest["model_config"] = config.to_dict()
        manifest["model_config"]["model_type"] = config.model_type
        if options.model_config:
            original = Path(options.model_config).read_bytes()
            (output / "source_config.json").write_bytes(original)
            manifest["source_config_sha256"] = hashlib.sha256(original).hexdigest()
        manifest["workload"] = {
            "model_type": config.model_type,
            "num_hidden_layers": config.num_hidden_layers,
            "attention_backend": options.attention_backend,
            "moe_backend": options.moe_backend,
            "attention_tp_size": options.tp_size // options.dp_size,
            "ep_size": options.ep_size,
            "weights": "synthetic_bfloat16",
        }
        manifest["input_signature"] = _input_signature(args)
        manifest["target_devices"] = [str(device) for device in mesh.devices.flat]
        manifest["mesh"] = dict(mesh.shape)
        manifest["static_args"] = {
            "forward_mode": "DECODE",
            "capture_hidden_mode": "NULL",
            "spec_algorithm": "NONE",
        }
        manifest["donate_argnames"] = ["memory_pools"]
        with jax.set_mesh(mesh):
            lowered = fn.lower(*args)
        (output / "stablehlo.mlir").write_text(str(lowered.compiler_ir(dialect="stablehlo")))
        manifest["stages"]["stablehlo"] = "complete"
        save_manifest()
        if options.stage == "compiled":
            compiled = lowered.compile(compiler_options=options.compiler_options or None)
            hlo = compiled.as_text()
            if not hlo or "HloModule" not in hlo:
                raise RuntimeError("Backend did not expose optimized HLO text")
            (output / "optimized_hlo.txt").write_text(hlo)
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
                    and (".llo" in p.name or re.search(r"-\d+-(?:original|post-.+)\.txt$", p.name))
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
