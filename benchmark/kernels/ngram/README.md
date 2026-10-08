# N-gram PLE benchmarks

Benchmarks and probes for the Qwen4Exp N-gram embedding (PLE): the host hash
(`compute_ngram_ids`), the host table gather, and the device layer. Measured on
v6e-1 (TPU v6 lite x1, EPYC 9B14 44 vCPU), jax 0.11.1 / libtpu 0.0.46.1.

## Why the hash and gather stay on the host

- The table is 320,001,446 x 160 bf16 = 95.4 GiB, more than v6e HBM (31.24 GiB),
  so it lives in host RAM.
- XLA:TPU cannot gather from host memory: a `pinned_host` table is rejected at
  trace time and a host-only gather aborts in the compiler (`probe_host_gather.py`).
- Pallas TPU has no int64 on this stack. With x64 on it raises; with x64 off it
  silently narrows to int32 and returns wrong ids (`probe_pallas_i64.py`).
- An exact two-uint32-limb Pallas hash works (`bench_ngram_hash_limb.py`), but
  the ids go back to the host table anyway. It is not wired into the scheduler.

| Hash | CPU NumPy | TPU + ids D2H | H2D + TPU + D2H |
|---|---:|---:|---:|
| decode B=256 | 0.051 ms | 0.210 ms | 0.382 ms |
| prefill T=8192 | 0.915 ms | 0.575 ms | 0.743 ms |

## Host hash

`compute_ngram_ids` mixes in `[T]` (one prefix XOR yields every n-gram order),
reduces in uint64, and has a decode fast path. Against the earlier `[T, HEADS]`
version (`bench_ngram_hash.py`):

| Case | Base | Now | Speedup |
|---|---:|---:|---:|
| decode B=256 | 0.100 ms | 0.048 ms | 2.09x |
| prefill T=8192 | 1.215 ms | 0.910 ms | 1.33x |

The host gather is about 2.3 ms at T=8192 with 32 threads.

## Device layer

`use_pallas=True` on `forward_extend` / `forward_decode` fuses the norms, gate,
dilated conv, SiLU and state writeback into one Pallas call (`bench_ngram_fused.py`).
It matches the reference within `atol=rtol=0.025` (tolerance, not bitwise parity)
and is tested at TP=1 only.

| Workload | Slots | Reference | Fused | Speedup |
|---|---:|---:|---:|---:|
| decode B=256 | 1024 | 1.943 ms | 1.408 ms | 1.38x |
| decode B=256 | 4096 | 4.131 ms | 3.618 ms | 1.14x |
| prefill T=2048 | 1024 | 4.365 ms | 2.803 ms | 1.56x |
| prefill T=8192 | 1024 | 20.668 ms | 8.129 ms | 2.54x |

The speedup shrinks with pool size because two pool-sized layout copies remain
around the fused call (`--hlo` shows them).

## Running

From `python/`, on a TPU host:

```bash
B=../benchmark/kernels/ngram
python $B/bench_ngram_hash.py <base-rev-or-path> [--ablation]
python $B/bench_ngram_hash_limb.py --tokens 8192 --batch 4
python $B/bench_ngram_device.py [--hlo]
python $B/bench_ngram_fused.py --mode extend --tokens 8192 --batch 4 --slots 1024 [--hlo]
python $B/probe_pallas_i64.py [--x64]   # compares compiled results with NumPy
python $B/probe_pallas_mod_cost.py
python $B/probe_host_gather.py          # aborts the process on purpose
python -m pytest -q $B/test_hash_limb.py
```

Run the TPU benchmarks one at a time. Re-run `probe_pallas_i64.py` after a
jax/libtpu upgrade: if `--x64` passes with `numeric match: YES`, native int64 works.
