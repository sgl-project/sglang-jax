# MiMo-V2-Flash Benchmark Report (2026-04-02)

## Setup

- **Hardware**: TPU v6e-16 (4 pods × 4 chips)
- **Model**: MiMo-V2-Flash (256-expert MoE, FP8)
- **Config**: TP=16, EP=16, page_size=128, mem_fraction_static=0.80
- **SWA**: swa_full_tokens_ratio=0.15, disable_radix_cache
- **Commits**: `a85c1ba7` (force_dequant), `5f15edae` (schedule_policy fix)

## 4K Prefill (input=4096, output=1)

| BS  | Input tok/s | TTFT (ms) | Duration (s) |
|-----|-------------|-----------|---------------|
| 1   | 6,863       | 594       | 0.60          |
| 4   | 6,616       | 2,002     | 2.48          |
| 8   | 7,328       | 3,042     | 4.47          |
| 16  | 7,755       | 5,045     | 8.45          |
| 24  | 7,915       | 7,033     | 12.42         |
| 32  | 7,987       | 9,035     | 16.41         |
| 48  | 8,068       | 13,006    | 24.37         |
| 64  | 8,108       | 16,969    | 32.33         |
| 96  | 8,141       | 24,925    | 48.30         |
| 128 | 8,163       | 32,860    | 64.22         |

**Peak**: ~8,163 tok/s @ bs=128 (near compute-bound limit)

## 4K Decode (input=4096, output=1024)

| BS  | Output tok/s | Input tok/s | ITL (ms) | TTFT (ms) | Duration (s) |
|-----|-------------|-------------|----------|-----------|---------------|
| 1   | 78.7        | 315         | 12.07    | 593       | 13.0          |
| 4   | 93.5        | 374         | 15.66    | 2,000     | 43.8          |
| 8   | 161.3       | 645         | 20.45    | 3,039     | 50.8          |
| 16  | 260.6       | 1,042       | 27.27    | 14,372    | 62.9          |
| 24  | 252.2       | 1,009       | 31.58    | 20,345    | 97.4          |
| 32  | **417.3**   | 1,669       | 31.20    | 22,602    | 78.5          |
| 48  | 412.6       | 1,650       | 31.68    | 40,075    | 119.1         |
| 64  | 412.3       | 1,649       | 31.76    | 57,638    | 159.0         |
| 96  | 413.5       | 1,654       | 31.73    | 92,674    | 237.7         |
| 128 | FAILED      | -           | -        | -         | -             |

**Peak**: **417.3 tok/s** @ bs=32 (compute-bound, plateau at bs≥32)
**ITL**: ~31.7ms stable at bs≥24 (compute-bound)

## 16K Prefill (input=16384, output=1, context-length=32768)

| BS  | Input tok/s | TTFT (ms) | Duration (s) |
|-----|-------------|-----------|---------------|
| 1   | 4,806       | 3,406     | 3.41          |
| 4   | 4,849       | 8,455     | 13.52         |
| 8   | 4,853       | 15,194    | 27.01         |
| 16  | 4,858       | 28,678    | 53.96         |
| 24  | 4,861       | 42,154    | 80.90         |
| 32  | 4,861       | 55,631    | 107.86        |

**Peak**: ~4,861 tok/s @ bs≥24 (saturated)

## 16K Decode (input=16384, output=1024, context-length=32768)

| BS  | Output tok/s | Input tok/s | ITL (ms) | TTFT (ms) | Duration (s) |
|-----|-------------|-------------|----------|-----------|---------------|
| 1   | 62.2        | 995         | 12.71    | 3,394     | 16.5          |
| 4   | 59.3        | 948         | 14.72    | 28,393    | 69.1          |
| 8   | 93.5        | 1,496       | 14.71    | 37,979    | 87.6          |
| 16  | 93.5        | 1,495       | 14.74    | 81,734    | 175.3         |
| 24  | 93.6        | 1,497       | 14.71    | 125,549   | 262.7         |
| 32  | 93.8        | 1,500       | 14.67    | 168,929   | 349.5         |

**Peak**: ~93.8 tok/s @ bs=32 (very low - context-length increase severely limits batch)

## Key Observations

1. **4K Prefill**: Near compute-bound at ~8,100 tok/s, scales well with batch size
2. **4K Decode**: Peak 417 tok/s at bs=32, plateau after that (matches E2E bench_serving baseline)
3. **16K Prefill**: Drops to ~4,860 tok/s (60% of 4K), 4x longer per-request
4. **16K Decode**: Only ~93 tok/s peak - **context-length=32768 drastically reduces concurrency**, limiting throughput
5. **ITL**: 4K decode ~31.7ms, 16K decode ~14.7ms (fewer concurrent requests → lower per-step latency)
6. **OOM**: 4K decode bs=128 causes server crash

## Mixed-Chunk Experiment

Tested `--enable-mixed-chunk` with context-length=16384 (4K decode, rate=3, 64 prompts):

| Metric | Without mixed-chunk | With mixed-chunk | Change |
|--------|:---:|:---:|:---:|
| Output tok/s | **410** | **210** | **-49%** |
| ITL median | 31.7ms | 57.0ms | +80% |
| Concurrency | ~18 | 41.5 | +2.3x |

**Result**: Mixed-chunk mode **hurts performance** significantly. Merging prefill+decode tokens into the same forward step doubles per-step compute, degrading ITL and overall throughput. Not suitable for this compute-bound MoE model.

## MMLU-Pro Eval

Running with context-length=65536, temperature=0.6, max_tokens=32000, concurrency=2.
493 questions, estimated 4-5 hours. Results pending.

## Summary

| Scenario | Peak tok/s | Config |
|----------|:---------:|:------:|
| 4K Prefill | 8,163 (input) | bs=128, ctx=16384 |
| 4K Decode | 417 (output) | bs=32, ctx=16384 |
| 16K Prefill | 4,861 (input) | bs=32, ctx=32768 |
| 16K Decode | 94 (output) | bs=32, ctx=32768 |
