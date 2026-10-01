# N-gram PLE: can the hash be a kernel, and where does the time go

For the 2026-09-23 opt-in Pallas fusion, exact limb hash, corrected int64
conclusions and overlap scope, see [FUSION_REVIEW.md](FUSION_REVIEW.md).
The measurements below are the earlier 2026-09-20 investigation.

`compute_ngram_ids` turns each token's n-grams into PLE table row ids
(`python/sgl_jax/srt/layers/ngram_embedding.py`, called from
`ScheduleBatch._merge_ngram_ple`). vLLM and SGLang both do this in one fused
kernel. This note records whether we can, what we did instead, and what the
device side actually costs.

Date: 2026-09-20. Branch `p3-ngram`. Baseline revision `a1c7923`.
Everything measured on `v6e-1` (`t1v-n-1a3e89da-w-0`): TPU v6 lite x1, AMD EPYC
9B14 44 vCPU, 172 GB, jax 0.11.1 / libtpu 0.0.46.1, `~/venv-sgl`.

## Prior art

**vLLM** (`vllm/models/deepseek_v41/common/engram.py`) has `_hash_ids_kernel`,
a Triton kernel keyed on `(token block, layer)` that does the whole thing: an
inline binary search for the request id, the per-shift lookback gather, the
dead-token barrier, the rolling mix, and the per-head reduce.

```python
for shift in tl.static_range(MAX_NGRAM):
    ...
    rolling ^= value * multiplier          # [BLOCK_T]  — one value per token
    if shift > 0:
        prime  = tl.load(primes  + param_offset)
        offset = tl.load(offsets + param_offset)
        hashed = rolling[:, None] % prime[None, :] + offset[None, :]
        tl.store(output + out_offset, hashed, ...)
```

`rolling` is **one value per token**; the head axis appears only in the final
`% prime + offset`. vLLM also has `_fused_engram_post_wkv_kernel` (gate + norm +
residual) and `_engram_lookup_kernel` (FP8 row gather).

**SGLang** reports the same two fusions for DeepSeek-V4.1 Flash: "n-gram hashing
is computed in one kernel", plus a fused gate/RMSNorm/depthwise-conv/residual
kernel, quoted at 186.6 → 203.3 tok/s in plain decode bundled with other
small-projection work. No isolated hash number published.

Sources: [vLLM engram API docs](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/common/engram/),
[SGLang DeepSeek-V4.1 Flash kernel optimization](https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization),
[LMSYS day-0 DeepSeek-V4.1 post](https://www.lmsys.org/blog/2026-09-10-deepseek-v41).

## Native int64 is blocked; limb emulation works

**2026-09-23 correction:** native int64 is blocked, but exact two-uint32-limb
hashing works. `bench_ngram_hash_limb.py` measures the full hash plus ids D2H;
the host consumer is a communication cost, not a correctness blocker. The
historical 16-nibble probe below is not a lower bound on emulation cost.

### Blocker 1: Pallas TPU has no working int64

The hash is irreducibly 64-bit. `build_hash_params` bounds the multipliers only
by `((1 << 63) - 1) // vocab_size`, so they reach ~6e13 and `token * multiplier`
runs to just under 2^63. XOR does not commute with the modulus, so that wide
value has to be materialized before the reduce — you cannot fold it into 32 bits
early.

`probe_pallas_i64.py` tests int64 both ways. Passing int64 **through** the
`pallas_call` boundary fails in XLA:

```
UNIMPLEMENTED: While rewriting computation to not contain X{64|128} element
types, XLA encountered an HLO for which this rewriting is not implemented:
%pallas_call
```

That one is arguably self-inflicted — the hash only needs int32 at the boundary
(token ids in, row ids out, both < 2^31) and int64 as an *intermediate*. So
the same probe also tests exactly that: int32 in, int32 out, int64 only
inside, and it checks the result against numpy rather than only checking that it
compiles. Both the pinned and the latest release behave identically:

| | jax 0.11.1 / libtpu 0.0.46.1 | jax 0.11.2 / libtpu 0.0.48 |
|---|---|---|
| x64 **on** | `NotImplementedError: 64-bit types are not supported` | identical |
| x64 **off** | compiles, runs, **wrong numbers** | identical |

The x64-off case is the trap. Every `astype(jnp.int64)` is silently narrowed to
int32 (JAX emits a `UserWarning`, the kernel runs anyway):

```
got  [14627850   718262 14156773  4985197  9088704]
want [18244588 10866603  5536579 17319997  4381943]
```

A probe that only asserts "it compiled" concludes the opposite of the truth.

The Pallas TPU docs list `jnp.int*` (all precisions except int4) as supported;
on this stack Mosaic lowering does not implement int64. Re-run
`probe_pallas_i64.py` after any jax/libtpu bump — if x64-on turns green *and*
prints `numeric match: YES`, this blocker is gone.

### What 32-bit emulation would cost

Carrying the 64-bit value as limbs makes the reduce the expensive part. `x mod p`
with `p ~ 2e7 < 2^25` needs progressive reduction. The original probe used
16 nibble steps, each with an int32 modulo. This was unnecessarily narrow:
6-bit steps fit signed int32 and 7-bit steps fit uint32. The new prototype
also replaces integer modulo with an FP32 quotient estimate plus exact
integer corrections. Historical `probe_pallas_mod_cost.py`, at `[8192, 16]`:

| | ms |
|---|---|
| Pallas, 16-nibble emulated reduce | **0.865** |
| XLA, same computation | 0.411 |
| one native int32 mod (for scale) | 0.123 |
| **host numpy, the entire hash** | **0.910** |

That particular emulated reduce **alone** eats 95% of the whole host hash, before the 64-bit
multiply emulation (~20–30 int32 ops per term, two terms), before the position
machinery, and before the round trip below. Hand-written Pallas is also 2x
slower than just letting XLA do it.

### Communication cost: the ids are consumed on the host

Independent of int64. The hash lives where the table lives. vLLM's engram table
is on the accelerator (`ParallelEngramEmbedding`, TP-sharded, FP8 rows), so
their hash is a device kernel feeding a device gather. Ours is
320,001,446 x 160 bf16 = **95.4 GiB** against **31.24 GiB** of v6e HBM per chip,
so it sits in host RAM, the gather is `np.take`, and the only consumer of the
hash output is that numpy call. A device kernel would mean H2D(`input_ids`) →
hash → D2H(`ids`, 0.5 MiB at T=8192) → host gather → H2D(rows, 40 MiB): a device
round trip in the middle to hand indices back to the side that owns the table.

The intermediate option — table in host memory, device gathers from it — does
not exist on XLA:TPU. `probe_host_gather.py`, all three placements:

| table | indices | result |
|---|---|---|
| device | device | OK |
| `pinned_host` | device | `ValueError: memory_space of all inputs passed to gather must be the same` |
| host | host | **process abort** — `F lowering_util.cc:2168] Check failed: source->memory_space() != MemorySpace::kHost` |

Mixed memory spaces are rejected when tracing; host-only fails a compiler CHECK
and takes the process with it.

This flips if the table ever fits in HBM — 95.4 GiB needs 4 v6e chips of pure
HBM, realistically 8–16 once weights and KV are in. On a slice that big you
shard the table across HBM, hash on device, gather on device, and the host drops
out of the path. That is when vLLM's kernel shape becomes the right thing to
copy wholesale.

## What we did instead

Not a kernel — the same *shape* as vLLM's kernel, applied to the numpy the host
is stuck with. Three changes, in `compute_ngram_ids`.

**1. Mix in `[T]`, not `[T, HEADS]`.** The old loop carried a `[T, 16]` int64
accumulator and masked each term with `order > shift`:

```python
mixed = np.empty((num_tokens, heads), dtype=np.int64)            # [T, 16]
mixed[:] = (input_ids * params.multipliers[0])[:, None]
for shift in range(1, ctx_len + 1):
    ...
    mixed ^= np.where(order[None, :] > shift, term[:, None], 0)  # [T, 16]
ids = mixed % params.sizes[None, :] + params.offsets[None, :]
```

`order[h] = h // heads_per_ngram + 2`, so heads 0–7 are order 2 and 8–15 are
order 3. Expanding the mask: `shift=1` is true for every head, `shift=2` only
for the order-3 heads. So order-2's hash is `t0^t1` and order-3's is `t0^t1^t2`
— the accumulator holds 2 distinct values in 16 columns, and order *o*'s hash is
order *o-1*'s XORed with one more term. That is a **prefix XOR over `[T]`**, and
after the `shift`-th term the accumulator already *is* the order-`(shift+1)`
hash, so it can be written out and carried forward. The head axis survives only
in the reduce, where it is unavoidable (16 distinct primes). 8x less int64
traffic through the loop, and the `order` mask and `np.where` are gone.

**2. Reduce in uint64.** Multipliers are bounded so `token * multiplier < 2^63`;
non-negative XOR stays non-negative, so numpy's floor-mod sign fixup is dead
work. `rolling.view(np.uint64)` aliases the same buffer, so `rolling ^=` keeps
both in sync with no copy.

**3. A decode fast path.** With one token per request every `chunk_pos` is 0, so
`ctx_col` collapses to the constant `ctx_len - shift` and `in_chunk` is always
false: `searchsorted`, two `clip`s, the fancy-index gather and the `where` all
become one column slice. Guarded on `T == B` **and**
`all(diff(cu_seqlens) == 1)`, because `T == B` alone can still be ragged —
`cu_seqlens = [0, 0, 1, 4, 4]` is 4 tokens and 4 requests. The `.copy()` on that
slice is load-bearing: the next line writes into it via `np.copyto`.

## Results — host hash

`python benchmark/kernels/ngram/bench_ngram_hash.py [base-rev-or-path]` loads the
pure-numpy hash section out of both revisions, checks them against each other and
against the independent Python reference from the test suite, then times them.
Median of 200 reps after 20 warm-up. Released Qwen4Exp shape: `ngram_size=3`,
`heads_per_ngram=8` → 16 heads, 16 primes above 20,000,000, 320,001,446 rows.

| case | tokens | reqs | base (ms) | fused (ms) | speedup |
|---|---|---|---|---|---|
| decode B=1 | 1 | 1 | 0.0610 | 0.0276 | **2.21x** |
| decode B=64 | 64 | 64 | 0.0715 | 0.0325 | **2.20x** |
| decode B=256 | 256 | 256 | 0.1001 | 0.0479 | **2.09x** |
| decode B=512 | 512 | 512 | 0.1396 | 0.0698 | **2.00x** |
| prefill T=512 | 512 | 2 | 0.1344 | 0.1120 | **1.20x** |
| prefill T=2048 | 2048 | 4 | 0.3596 | 0.2728 | **1.32x** |
| prefill T=8192 | 8192 | 4 | 1.2154 | 0.9104 | **1.33x** |
| prefill T=32768 | 32768 | 8 | 4.6444 | 3.4433 | **1.35x** |

Per change — `python benchmark/kernels/ngram/bench_ngram_hash.py --ablation`:

| case | base | + `[T]` prefix XOR | + uint64 reduce | + decode fast path |
|---|---|---|---|---|
| decode B=256 | 0.1069 ms | 0.0875 ms (1.22x) | — | 0.0476 ms (2.25x) |
| prefill T=2048 | 0.3566 ms | 0.2755 ms (1.29x) | 0.2722 ms (1.31x) | no effect |
| prefill T=8192 | 1.2071 ms | 0.9250 ms (1.31x) | 0.9199 ms (1.31x) | no effect |
| prefill T=32768 | 4.6075 ms | 3.4746 ms (1.33x) | 3.4314 ms (1.34x) | no effect |

**The uint64 reduce is architecture-dependent.** On Zen 4 it buys nothing —
the integer divide path costs the same signed or unsigned. The same ablation on
an Apple M1 Pro goes 0.638 → 0.534 ms (1.20x) and moves the end-to-end prefill
ratio from 1.31x to 1.58x. Kept because it is free on the target host and real
elsewhere, but on v6e hosts the prefill win is change 1 alone.

At B=1 the fused version costs 0.028 ms, almost entirely numpy dispatch — ~15
ufunc calls. That is the floor without leaving numpy.

Decode saves ~52 µs per step on a path that runs every step. Prefill saves
~0.31 ms at T=8192 against the ~2.3 ms the row gather takes at that shape.

Tests, on the chip: `63 passed, 1 skipped, 14 subtests passed`
(`test_ngram_embedding.py`, `test_ngram_table.py`, `test_recurrent_short_conv.py`).

## Results — device forward

`python benchmark/kernels/ngram/bench_ngram_device.py`. Released shape,
activations and conv state bf16, conv state donated the way `model_runner`
donates `memory_pools`. "post-gate" is everything after the gate: `norm_conv`,
the dilated conv, the state write-back, and the two residual adds.

| case | slots | pool MiB | full ms | gate ms | post-gate ms | post-gate % |
|---|---|---|---|---|---|---|
| decode B=256 | 256 | 45 | 1.502 | 0.230 | 1.272 | 85% |
| decode B=256 | 1024 | 180 | 1.929 | 0.222 | 1.708 | 89% |
| decode B=256 | 4096 | 720 | 4.128 | 0.226 | 3.902 | 95% |
| decode B=512 | 1024 | 180 | 3.582 | 0.307 | 3.274 | 91% |
| decode B=512 | 4096 | 720 | 5.776 | 0.311 | 5.465 | 95% |
| extend T=2048 | 1024 | 180 | 4.355 | 1.237 | 3.118 | 72% |
| extend T=8192 | 256 | 45 | 20.135 | 6.030 | 14.105 | 70% |
| extend T=8192 | 1024 | 180 | 20.620 | 6.007 | 14.613 | 71% |
| extend T=8192 | 4096 | 720 | 22.705 | 6.048 | 16.657 | 73% |

**Historical hypothesis, superseded by the opt-in fusion results linked above.** The
gate is 5–11% of decode, and XLA already folds the elementwise chain into its
surrounding fusions — 30 fusion instructions for the whole decode forward, 0
separate dots. And the cost tracks `num_slots`, not the batch: at fixed B=256 it
goes 1.27 → 1.71 → 3.90 ms as the pool grows 45 → 180 → 720 MiB, while the batch
touches 256 slots throughout.

### Shared GDN follow-up

The shared GDN dilation change and its state-writeback benchmark are excluded
from this PLE PR for separate API review, GDN regression testing, and performance
validation. PLE's reference convolution now lives in `kernels/ngram_conv.py`;
the opt-in fused path is in `kernels/ngram_fused.py`.

## Reproducing

The VM checkout is rsync'd rather than cloned, so pass a path to the base file
instead of a git rev.

```bash
rsync -az benchmark/kernels/ngram/ v6e-1:~/sglang-jax/benchmark/kernels/ngram/
rsync -az python/sgl_jax/srt/layers/ngram_embedding.py \
    v6e-1:~/sglang-jax/python/sgl_jax/srt/layers/ngram_embedding.py
rsync -az python/sgl_jax/srt/kernels/ngram_conv.py python/sgl_jax/srt/kernels/ngram_fused.py \
    v6e-1:~/sglang-jax/python/sgl_jax/srt/kernels/
git show a1c7923:python/sgl_jax/srt/layers/ngram_embedding.py \
    | ssh v6e-1 'cat > ~/ngram_embedding_base.py'

R="cd ~/sglang-jax/python && ~/venv-sgl/bin/python -u ../benchmark/kernels/ngram"
ssh v6e-1 "$R/bench_ngram_hash.py ~/ngram_embedding_base.py"
ssh v6e-1 "$R/bench_ngram_hash.py --ablation ~/ngram_embedding_base.py"
ssh v6e-1 "$R/bench_ngram_device.py"
ssh v6e-1 "$R/probe_pallas_i64.py"
ssh v6e-1 "$R/probe_pallas_i64.py --x64"     # both runs compare every compiled result with NumPy
ssh v6e-1 "$R/probe_host_gather.py"          # aborts the process on purpose
```

The device benchmarks hold the TPU — one at a time, and `-u` because Python
block-buffers stdout over ssh. To re-test int64 on a newer jax, build a throwaway
venv (`uv venv ~/venv-probe && uv pip install --python ~/venv-probe/bin/python
"jax[tpu]==<ver>"`) and run `probe_pallas_i64.py` under it; leave `venv-sgl`
alone.
