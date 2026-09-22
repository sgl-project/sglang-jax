"""Offline IR export. Keep imports stdlib-only until dump flags are configured."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="BF16 decode AOT export PoC (Qwen3/native or MiMo-V2-Flash/fa/fused_v2)"
    )
    parser.add_argument("--model-config", help="Local config.json; omitted: built-in tiny Qwen3")
    parser.add_argument("--target", choices=("cpu", "tpu"), default="tpu")
    parser.add_argument(
        "--topology",
        choices=tuple(f"v6e-{n}" for n in (1, 4, 8, 16, 32, 64))
        + tuple(f"v7x-{n}" for n in (8, 16, 32, 64)),
        help="Target TPU topology; the suffix counts JAX devices, not chips",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Total devices, as in serving")
    parser.add_argument(
        "--dp-size", type=int, default=1, help="Attention DP; attention TP=tp_size/dp_size"
    )
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--attention-backend", choices=("native", "fa"), default="native")
    parser.add_argument("--moe-backend", choices=("fused_v2",))
    parser.add_argument(
        "--bf16-model",
        action="store_true",
        help="Explicitly replace checkpoint quantization with synthetic BF16 weights",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument(
        "--kv-capacity", type=int, default=128, help="KV token capacity, excluding padding"
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--stage", choices=("stablehlo", "compiled"), default="compiled")
    parser.add_argument("--dump-llo", action="store_true", help="Require TPU LLO text artifacts")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--compiler-option", action="append", default=[], metavar="NAME=JSON_VALUE")
    options = parser.parse_args()
    for key in (
        "tp_size",
        "dp_size",
        "ep_size",
        "batch_size",
        "context_length",
        "kv_capacity",
        "page_size",
    ):
        if getattr(options, key) <= 0:
            parser.error(f"{key} must be positive")
    if options.tp_size % options.dp_size or options.batch_size % options.dp_size:
        parser.error("tp_size and batch_size must be divisible by dp_size")
    if options.kv_capacity % (options.page_size * options.dp_size):
        parser.error("kv_capacity must be divisible by page_size * dp_size")
    if options.attention_backend == "fa" and options.target != "tpu":
        parser.error("fa requires --target=tpu")
    padded_context = -(-options.context_length // options.page_size) * options.page_size
    if options.batch_size * padded_context > options.kv_capacity:
        parser.error("kv_capacity must cover every request's page-aligned context")
    if options.target == "tpu" and not options.topology:
        parser.error("--topology is required for TPU cross-compilation")
    if options.target == "cpu" and options.topology:
        parser.error("--topology is only valid with --target=tpu")
    if options.dump_llo and (options.target != "tpu" or options.stage != "compiled"):
        parser.error("--dump-llo requires --target=tpu --stage=compiled")
    compiler_options = {}
    for item in options.compiler_option:
        try:
            name, value = item.split("=", 1)
            value = json.loads(value)
            if not name.startswith("xla_") or not isinstance(value, (str, int, float, bool)):
                raise ValueError("expected an xla_* option with a scalar JSON value")
            if "dump" in name:
                raise ValueError("dump options are managed by the exporter")
            compiler_options[name] = value
        except (ValueError, TypeError) as error:
            parser.error(f"Invalid compiler option {item!r}: {error}")
    options.compiler_options = compiler_options
    return options


def configure_dump_environment(options):
    if "jax" in sys.modules:
        raise RuntimeError("Run python -m sgl_jax.compile in a fresh process before importing JAX")
    options.output = options.output.resolve()
    options.output.mkdir(parents=True, exist_ok=True)
    if any(options.output.iterdir()):
        raise ValueError("Output directory must be empty; old dumps cannot count as new artifacts")
    for key in ("XLA_FLAGS", "LIBTPU_INIT_ARGS"):
        if any("dump" in flag for flag in shlex.split(os.environ.get(key, ""))):
            raise ValueError(
                f"Remove existing dump flags from {key}; this command owns its dump paths"
            )
    # Compilation targets come from get_topology_desc. The host needs no TPU.
    os.environ["JAX_PLATFORMS"] = "cpu"
    if options.stage == "compiled":
        flags = [
            f"--xla_dump_to={options.output / 'xla_dump'}",
            "--xla_dump_hlo_as_text=true",
            "--xla_dump_hlo_as_proto=true",
            "--xla_dump_hlo_module_re=jit_jitted_run_model",
        ]
        os.environ["XLA_FLAGS"] = " ".join(
            filter(None, (os.environ.get("XLA_FLAGS", ""), shlex.join(flags)))
        )
    if options.dump_llo:
        flags = [
            f"--xla_jf_dump_to={options.output / 'llo'}",
            "--xla_jf_dump_llo_text=true",
        ]
        os.environ["LIBTPU_INIT_ARGS"] = " ".join(
            filter(None, (os.environ.get("LIBTPU_INIT_ARGS", ""), shlex.join(flags)))
        )


def main():
    options = parse_args()
    configure_dump_environment(options)
    from sgl_jax.srt.model_executor.aot_export import export

    export(options)


if __name__ == "__main__":
    main()
