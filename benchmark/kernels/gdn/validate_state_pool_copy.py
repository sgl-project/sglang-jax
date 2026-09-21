"""Real-TPU, donated-pool A/B validation for issue #1667.

Export the untouched adapter with ``git show HEAD:python/sgl_jax/srt/kernels/gdn/
 fused_chunk_parallel_adapter.py > /tmp/baseline_adapter.py`` (join the path).
Run with PYTHONPATH=python and explicit model head counts. Pool capacities exclude
one dummy slot per DP rank: capacity 1024 means 4 * 257 physical pool slots.
This checks single-layer equivalence; it does not establish full-model peak HBM.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-k-heads", type=int, required=True)
    parser.add_argument("--num-v-heads", type=int, required=True)
    parser.add_argument("--head-k-dim", type=int, default=128)
    parser.add_argument("--head-v-dim", type=int, default=128)
    parser.add_argument("--conv-kernel-size", type=int, default=4)
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--pool-capacities", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[192, 384, 768])
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1667)
    return parser.parse_args()


def run(args, report):
    # Keep --help usable without JAX installed, and never execute the kernel on CPU.
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    import sgl_jax.srt.kernels.gdn.fused_chunk_parallel_adapter as candidate

    devices = jax.devices()
    if len(devices) != args.dp * args.tp or any(d.platform != "tpu" for d in devices):
        raise RuntimeError(f"Requires exactly {args.dp * args.tp} TPU devices; got {devices}")
    if jax.process_count() != 1:
        raise RuntimeError("This runner requires a single host with all devices addressable")
    if len(args.pool_capacities) != len(args.batch_sizes):
        raise ValueError("Pool capacities and batch sizes must have equal lengths")
    if args.num_k_heads % args.tp or args.num_v_heads % args.tp:
        raise ValueError("Global head counts must be divisible by TP")
    if args.tokens % args.dp or min(args.samples, args.steps) < 1 or args.warmup < 0:
        raise ValueError("Tokens must divide DP; samples/steps must be positive; warmup >= 0")
    spec = importlib.util.spec_from_file_location("baseline_adapter", args.baseline_adapter)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    mesh = Mesh(np.asarray(devices).reshape(args.dp, args.tp), ("data", "tensor"))
    report["runtime"] = {
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "libtpu": version("libtpu"),
        "baseline_sha256": hashlib.sha256(args.baseline_adapter.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
        "devices": [str(d) for d in devices],
        "device_kind": devices[0].device_kind,
        "mesh": dict(mesh.shape),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "comparison": "exact finite equality; no tolerance",
        "tracking": False,
    }
    rng = np.random.default_rng(args.seed)
    dim = 2 * args.num_k_heads * args.head_k_dim + args.num_v_heads * args.head_v_dim
    specs = (
        P("data", "tensor", None),
        P("data", "tensor", None, None),
        P("data", "tensor"),
        P("data", "tensor"),
        P("data", "tensor"),
        P("tensor", None),
        P("tensor"),
        P("tensor"),
        P("data"),
        P("data"),
        P("data"),
        P("data"),
    )
    shardings = tuple(NamedSharding(mesh, p) for p in specs)

    def place(values, start=0):
        # Separate host copies are transferred for A and B, never recycled after donation.
        return tuple(
            jax.device_put(np.array(v, copy=True), s) for v, s in zip(values, shardings[start:])
        )

    def random(shape, scale, bf16=False):
        value = rng.standard_normal(shape, dtype=np.float32)
        value *= scale
        return value.astype(jnp.bfloat16) if bf16 else value

    def function(module):
        def execute(conv, recurrent, qkv, b, a, weight, a_log, bias, cu, indices, initial, lengths):
            backend = SimpleNamespace(
                mesh=mesh,
                num_k_heads=args.num_k_heads,
                num_v_heads=args.num_v_heads,
                head_k_dim=args.head_k_dim,
                head_v_dim=args.head_v_dim,
                conv_kernel_size=args.conv_kernel_size,
                forward_metadata=SimpleNamespace(
                    cu_q_lens=cu,
                    recurrent_indices=indices,
                    has_initial_state=initial,
                    recurrent_track_indices=None,
                    recurrent_track_mask=None,
                ),
            )
            return module.fused_chunk_parallel_prefill(
                backend, qkv, conv, recurrent, b, a, weight, a_log, bias, lengths
            )

        return jax.jit(execute, donate_argnums=(0, 1), in_shardings=shardings)

    def compare(left, right, label):
        # Transfer one local shard at a time, avoiding two full host recurrent pools.
        for a, b in zip(left.addressable_shards, right.addressable_shards):
            if a.index != b.index:
                raise AssertionError(f"{label}: shard index mismatch")
            x, y = np.asarray(a.data), np.asarray(b.data)
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise AssertionError(f"{label}: non-finite values")
            if not np.array_equal(x, y):
                max_error = max(
                    float(np.max(np.abs(u.astype(np.float32) - v.astype(np.float32))))
                    for u, v in zip(x, y)
                )
                raise AssertionError(
                    f"{label}: nonzero absolute error {max_error}, shard {a.index}"
                )

    def metadata(batch, case, step=0):
        n = batch // args.dp
        if n < 5:
            raise ValueError("Need at least 5 padded requests per DP shard")
        lengths = np.ones(n, dtype=np.int32)
        lengths[-2:] = 0  # Both fresh and resumed empty requests.
        lengths[0] += args.tokens // args.dp - int(lengths.sum())
        if lengths[0] < 1:
            raise ValueError("Tokens per rank must cover the nonempty padded requests")
        slots = np.arange(1, n + 1, dtype=np.int32)
        initial = np.zeros(n, dtype=np.bool_)
        if case == "resumed" or step:
            initial[:] = True
        elif case in ("mixed", "dummy", "empty"):
            initial[1::2] = True
        if case == "dummy":
            # One positive-length dummy plus a repeated empty dummy avoids an
            # unrelated concurrent-write race between two active dummy requests.
            slots[1] = slots[-1] = 0
        if case == "empty":
            lengths[:] = 0
        if step:
            initial[0] = False  # Slot reuse/reset within a continuing trajectory.
        cu = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
        total = lengths + initial.astype(np.int32) * (17 + step * 13)
        if case == "mixed":
            # Explicit reset must win over a stale total length on a reused slot.
            total[0] = lengths[0] + 17
        return tuple(np.tile(v, args.dp) for v in (cu, slots, initial, total))

    for capacity, batch in zip(args.pool_capacities, args.batch_sizes):
        if capacity % args.dp or batch % args.dp or batch > capacity:
            raise ValueError("Capacity/batch must divide DP and batch must fit capacity")
        slots = capacity + args.dp
        row = {
            "capacity": capacity,
            "pool_slots": slots,
            "batch": batch,
            "variants": {},
            "checks": [],
        }
        report["runs"].append(row)
        folder = args.output_dir / f"pool{capacity}_bs{batch}"
        folder.mkdir(parents=True, exist_ok=True)
        host_pools = (
            random((slots, dim, args.conv_kernel_size - 1), 0.03, True),
            random((slots, args.num_v_heads, args.head_k_dim, args.head_v_dim), 0.01),
        )
        host_fixed = (
            random((args.tokens, dim), 0.1, True),
            random((args.tokens, args.num_v_heads), 0.1, True),
            random((args.tokens, args.num_v_heads), 0.1, True),
            random((dim, args.conv_kernel_size), 0.03, True),
            rng.uniform(-1.0, 0.0, args.num_v_heads).astype(np.float32),
            rng.uniform(-0.5, 0.5, args.num_v_heads).astype(np.float32),
        )
        fixed = place(host_fixed, 2)
        meta = place(metadata(batch, "mixed"), 8)
        executables = {}
        for name, module in (("baseline", baseline), ("candidate", candidate)):
            abstract = tuple(
                jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=s)
                for v, s in zip(host_pools, shardings)
            )
            start = time.perf_counter()
            executable = function(module).lower(*abstract, *fixed, *meta).compile()
            stats = executable.memory_analysis()
            memory = {
                field: int(getattr(stats, field))
                for field in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                    "temp_size_in_bytes",
                    "generated_code_size_in_bytes",
                )
            }
            memory["estimated_total_bytes"] = (
                memory["argument_size_in_bytes"]
                + memory["output_size_in_bytes"]
                - memory["alias_size_in_bytes"]
                + memory["temp_size_in_bytes"]
            )
            row["variants"][name] = {
                "compile_seconds": time.perf_counter() - start,
                "memory_analysis": memory,
            }
            hlo = executable.as_text()
            pool_shape = (
                f"f32[{capacity // args.dp + 1},{args.num_v_heads // args.tp},"
                f"{args.head_k_dim},{args.head_v_dim}]"
            )
            row["variants"][name]["full_recurrent_pool_shape"] = pool_shape
            row["variants"][name]["full_recurrent_pool_copy_lines"] = [
                line.strip()
                for line in hlo.splitlines()
                if pool_shape in line and re.search(r"\bcopy\(", line)
            ]
            (folder / f"{name}.optimized_hlo.txt").write_text(hlo)
            (folder / f"{name}.memory.json").write_text(json.dumps(memory, indent=2))
            executables[name] = executable
            print(json.dumps({"capacity": capacity, "variant": name, "memory": memory}), flush=True)

        for case in ("new", "resumed", "mixed", "dummy", "empty"):
            pools = {name: place(host_pools) for name in executables}
            for step in range(args.steps):
                host_meta = metadata(batch, case, step)
                meta = place(host_meta, 8)
                results = {}
                for name, executable in executables.items():
                    results[name] = jax.block_until_ready(executable(*pools[name], *fixed, *meta))
                    pools[name] = results[name][1:]
                for index, label in enumerate(("output", "conv_pool", "recurrent_pool")):
                    compare(
                        results["baseline"][index],
                        results["candidate"][index],
                        f"{case}/{step}/{label}",
                    )
                # Independently assert every untouched slot, including each DP dummy.
                local_slots = capacity // args.dp + 1
                local_indices = host_meta[1].reshape(args.dp, -1)
                for name in executables:
                    for leaf, host in zip(pools[name], host_pools):
                        for shard in leaf.addressable_shards:
                            rank = shard.index[0].start // local_slots
                            active = set(local_indices[rank]) - {0}
                            unchanged = [s for s in range(local_slots) if s not in active]
                            actual = np.asarray(shard.data)[unchanged]
                            expected = host[shard.index][unchanged]
                            if not np.array_equal(actual, expected):
                                raise AssertionError(
                                    f"{name}/{case}/{step}: inactive or dummy slot changed"
                                )
                row["checks"].append({"case": case, "step": step, "max_abs_error": 0.0})
                print(f"PASS pool={capacity} bs={batch} case={case} step={step} exact", flush=True)
            del pools, results

        meta = place(metadata(batch, "mixed", step=1), 8)
        final_results = {}
        for name, executable in executables.items():
            pools = place(host_pools)
            jax.block_until_ready((pools, fixed, meta))
            samples = []
            for iteration in range(args.warmup + args.samples):
                start = time.perf_counter()
                result = jax.block_until_ready(executable(*pools, *fixed, *meta))
                elapsed = (time.perf_counter() - start) * 1000
                pools = result[1:]
                if iteration >= args.warmup:
                    samples.append(elapsed)
            final_results[name] = result
            row["variants"][name]["timing_ms"] = {
                "samples": samples,
                "median": float(np.median(samples)),
                "p10": float(np.percentile(samples, 10)),
                "p90": float(np.percentile(samples, 90)),
            }
        for a, b in zip(final_results["baseline"], final_results["candidate"]):
            compare(a, b, "timing trajectory final")
        row["timing_trajectory_exact"] = True
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2, default=str))
        del host_pools, final_results, result, pools, executables, executable
        jax.clear_caches()

    # Finish every numerical and timing trajectory before enforcing the memory
    # objective, so a copy regression still leaves all correctness evidence.
    failed_capacities = [
        row["capacity"]
        for row in report["runs"]
        if row["variants"]["candidate"]["full_recurrent_pool_copy_lines"]
    ]
    report["full_recurrent_pool_copy_gate"] = {
        "passed": not failed_capacities,
        "failed_capacities": failed_capacities,
        "scope": "candidate full recurrent pool copy instructions in compiled optimized HLO",
    }
    if failed_capacities:
        raise AssertionError(
            f"Candidate still copies a full recurrent state pool at capacities {failed_capacities}"
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args), "runs": [], "status": "running"}
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
