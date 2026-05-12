"""Dump LLO IR for small_dma and batch_dma kernels to inspect DMA instructions.

Sets XLA/libtpu dump flags before importing JAX, then compiles both kernels
for a single configuration. LLO dumps land in /tmp/ir_dumps/llo/.

Usage:
    python -m benchmark.moe.dump_dma_llo [--num-tokens 4] [--output-dir /tmp/ir_dumps]
"""
import os
import sys
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="/tmp/ir_dumps")
    args = parser.parse_args()

    ir_root = args.output_dir
    hlo_dir = os.path.join(ir_root, "hlo")
    llo_dir = os.path.join(ir_root, "llo")
    mosaic_dir = os.path.join(ir_root, "mosaic")

    if os.path.exists(ir_root):
        shutil.rmtree(ir_root)
    for d in [hlo_dir, llo_dir, mosaic_dir]:
        os.makedirs(d, exist_ok=True)

    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_dump_hlo_as_text --xla_dump_to={hlo_dir}"
    )
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "")
        + f" --xla_jf_dump_to={llo_dir}"
        + " --xla_jf_dump_hlo_text=true"
        + " --xla_jf_dump_llo_text=true"
        + " --xla_jf_emit_annotations=true"
        + f" --xla_mosaic_dump_to={mosaic_dir}"
        + " --xla_mosaic_enable_llo_source_annotations=true"
    )

    import jax
    import jax.numpy as jnp

    sys.path.insert(0, ".")
    from benchmark.moe.bench_dma_size import build_local_benchmark

    for mode in ["small", "batch"]:
        print(f"\n=== Compiling {mode}_dma kernel (num_tokens={args.num_tokens}) ===")
        run_fn, src, dst = build_local_benchmark(
            args.num_tokens, args.hidden_size, args.num_repeats, mode
        )
        out = run_fn(src, dst)
        jax.block_until_ready(out)
        print(f"  {mode} kernel executed successfully")

    print(f"\n=== IR dumps written to {ir_root} ===")
    for subdir in ["hlo", "llo", "mosaic"]:
        path = os.path.join(ir_root, subdir)
        files = os.listdir(path) if os.path.exists(path) else []
        print(f"  {subdir}/: {len(files)} files")

    print("\nTo copy to GCS for local analysis:")
    print(f"  cp -r {ir_root} /models/ir_dumps_dma_bench/")
    print("  # then locally:")
    print("  gcloud storage cp -r gs://inference-model-storage-poc-tpu-hns/ir_dumps_dma_bench/ .")


if __name__ == "__main__":
    main()
