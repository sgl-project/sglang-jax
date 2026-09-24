"""Load one dedicated TPU scheduler, run the destructive transfer probe, then exit.

No HTTP listener or scheduler event loop is started. --check-config only checks
the frozen JSON and does not import JAX, load a tokenizer/model, or contact TPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def validate_config(config):
    targets = {
        "google/gemma-4-31B-it": (1, 1, 128),
        "XiaomiMiMo/MiMo-V2.5": (2, 8, 256),
    }
    model = config.get("model_path")
    if model not in targets:
        raise ValueError("use one of the two acceptance model repository IDs")
    dp, ep, page = targets[model]
    required = {
        "device": "tpu",
        "tp_size": 8,
        "dp_size": dp,
        "ep_size": ep,
        "page_size": page,
        "enable_unified_radix_tree": True,
        "hicache_storage": "none",
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    if config.get("nnodes", 1) != 1 or config.get("node_rank", 0) != 0:
        raise ValueError("single-host node_rank=0 only")
    if config.get("tokenizer_path", model) != model:
        raise ValueError("tokenizer must use the pinned model repository")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", config.get("revision", "")):
        raise ValueError("revision must be an exact 40-character model/tokenizer commit")
    if config.get("hicache_transfer_backend") not in {"jax", "raiden"}:
        raise ValueError("choose an explicit jax or raiden backend")
    if config.get("hicache_write_policy") not in {"write_through", "write_back"}:
        raise ValueError("choose an explicit write policy")
    if dp == 2 and config.get("swa_full_tokens_ratio") != 0.25:
        raise ValueError("MiMo acceptance requires swa_full_tokens_ratio=0.25")
    return config


def run_loaded_probe(config, output, page_count):
    # Match launch_server's native-before-JAX import order, including when the
    # backend is supplied in a JSON rather than directly in sys.argv.
    from sgl_jax.raiden import preload_raiden_if_requested

    preload_raiden_if_requested(["--hicache-transfer-backend", config["hicache_transfer_backend"]])
    import jax

    if jax.default_backend() != "tpu":
        raise RuntimeError("real TPU required; no model will be loaded on this backend")
    from sgl_jax.srt.entrypoints.engine import _set_envs_and_config
    from sgl_jax.srt.managers.scheduler import Scheduler
    from sgl_jax.srt.server_args import PortArgs, ServerArgs
    from sgl_jax.srt.utils import configure_logger, prepare_model_and_tokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "srt"))
    from hybrid_hicache_transfer_probe import run_transfer_probe

    args = ServerArgs(**config)
    args.check_server_args()
    configure_logger(args)
    _set_envs_and_config(args)
    args.model_path, args.tokenizer_path = prepare_model_and_tokenizer(
        args.model_path, args.tokenizer_path
    )
    # The same constructor used by run_scheduler_process loads the model and
    # connects its allocator/cache. The calling thread retains sole ownership.
    scheduler = Scheduler(args, PortArgs.init_new(args))
    original_error = None
    try:
        allocator = scheduler.token_to_kv_pool_allocator
        output.with_suffix(".runtime.json").write_text(
            json.dumps(
                {
                    "devices": [
                        {
                            "id": device.id,
                            "kind": device.device_kind,
                            "process_index": device.process_index,
                        }
                        for device in jax.devices()
                    ],
                    "window": scheduler.tree_cache.sliding_window_size,
                    "full_capacity_tokens_per_rank": allocator.full_attn_allocator.size_per_rank,
                    "swa_capacity_tokens_per_rank": allocator.swa_attn_allocator.size_per_rank,
                    "tp_size": args.tp_size,
                    "dp_size": args.dp_size,
                    "ep_size": args.ep_size,
                },
                indent=2,
            )
            + "\n"
        )
        expected_window = 1024 if config["dp_size"] == 1 else 128
        if scheduler.tree_cache.sliding_window_size != expected_window:
            raise RuntimeError("loaded model SWA window differs from the acceptance target")
        return run_transfer_probe(
            scheduler, destructive_opt_in=True, report_path=output, page_count=page_count
        )
    except BaseException as exc:
        original_error = exc
        raise
    finally:
        cleanup_errors = []
        for controller in scheduler.tree_cache.hicache_controllers.values():
            try:
                controller.shutdown()
            except Exception as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if original_error is not None:
                for exc in cleanup_errors:
                    original_error.add_note(f"probe controller shutdown also failed: {exc!r}")
            else:
                raise cleanup_errors[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-args", type=Path, required=True, help="Frozen ServerArgs JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-count", type=int, default=2)
    parser.add_argument("--destructive-opt-in", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    config_bytes = args.server_args.read_bytes()
    config = validate_config(json.loads(config_bytes))
    if args.page_count < 2:
        parser.error("acceptance requires at least two pages")
    if args.check_config:
        print("Configuration checked; no model or hardware was accessed.")
        return
    if not args.destructive_opt_in:
        parser.error("--destructive-opt-in is required before loading the dedicated model")
    if args.output.exists():
        parser.error("output exists; choose a fresh run path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "server_args": config,
        "server_args_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[1] / "srt" / "hybrid_hicache_transfer_probe.py"
            ).read_bytes()
        ).hexdigest(),
        "page_count": args.page_count,
    }
    args.output.with_suffix(".launch.json").write_text(json.dumps(provenance, indent=2) + "\n")
    run_loaded_probe(config, args.output, args.page_count)


if __name__ == "__main__":
    main()
