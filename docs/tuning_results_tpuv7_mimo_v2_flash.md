# MiMoV2Flash Fused MoE Tuning Results - TPU v7

Tuned on TPU v7 (2x4 topology, 8 devices, ep_size=8)
Model: 256 experts, top_k=8, hidden_size=4096, intermediate_size=2048
No shared expert, no grouped topk
VMEM budget: 64MB, headroom_ratio=0.90
Date: 2026-04-15

## BF16 Results (sparse_hotspot, hotspot_ratio=1, hotspot_count=48)

| tokens | best_ms | config (bt, bf, bd1, bd2, bts, btc, bfc, bd1c, bd2c, bse) |
|--------|---------|-----------------------------------------------------------|
| 16     | 0.328   | (2, 2048, 1024, 1024, 8, 8, 2048, 1024, 1024, 2048)      |
| 32     | 0.341   | (4, 2048, 1024, 1024, 8, 8, 2048, 1024, 1024, 2048)      |
| 64     | 0.378   | (8, 2048, 1024, 1024, 16, 16, 2048, 1024, 1024, 2048)    |
| 128    | 0.433   | (16, 2048, 1024, 1024, 32, 32, 2048, 1024, 1024, 2048)   |
| 256    | 0.578   | (32, 2048, 1024, 1024, 64, 64, 2048, 1024, 1024, 2048)   |
| 512    | 0.887   | (64, 2048, 1024, 1024, 128, 128, 2048, 1024, 1024, 2048) |
| 1024   | 1.795   | (128, 2048, 512, 512, 256, 256, 2048, 512, 512, 2048)    |
| 2048   | 3.282   | (256, 1024, 1024, 1024, 128, 128, 1024, 1024, 1024, 1024)|
| 4096   | 6.956   | (256, 1024, 1024, 1024, 128, 128, 1024, 1024, 1024, 1024)|
| 8192   | 13.653  | (256, 1024, 1024, 1024, 128, 128, 1024, 1024, 1024, 1024)|
| 16384  | 27.051  | (256, 1024, 1024, 1024, 128, 128, 1024, 1024, 1024, 1024)|
| 32768  | 52.730  | (256, 1024, 1024, 1024, 128, 128, 1024, 1024, 1024, 1024)|

## FP8 Config A Results (sparse_hotspot, hotspot_ratio=1, hotspot_count=48)

| tokens | best_ms | config (bt, bf, bd1, bd2, bts, btc, bfc, bd1c, bd2c, bse) |
|--------|---------|-----------------------------------------------------------|
| 16     | 0.219   | (2, 2048, 2048, 2048, 8, 8, 2048, 2048, 2048, 2048)      |
| 32     | 0.232   | (4, 2048, 2048, 2048, 8, 8, 2048, 2048, 2048, 2048)      |
| 64     | 0.262   | (8, 2048, 2048, 2048, 16, 16, 2048, 2048, 2048, 2048)    |
| 128    | 0.322   | (16, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 256    | 0.496   | (32, 2048, 2048, 2048, 64, 64, 2048, 2048, 2048, 2048)   |
| 512    | 0.855   | (64, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 1024   | 1.694   | (128, 2048, 1024, 1024, 64, 64, 2048, 1024, 1024, 2048)  |
| 2048   | 3.548   | (64, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 4096   | 7.076   | (64, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 8192   | 13.986  | (64, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 16384  | 27.751  | (128, 2048, 1024, 1024, 64, 64, 2048, 1024, 1024, 2048)  |
| 32768  | 54.316  | (128, 2048, 1024, 1024, 64, 64, 2048, 1024, 1024, 2048)  |

## FP8 Config B Results (zipf, zipf_s=1.2)

| tokens | best_ms | config (bt, bf, bd1, bd2, bts, btc, bfc, bd1c, bd2c, bse) |
|--------|---------|-----------------------------------------------------------|
| 16     | 0.167   | (2, 1024, 4096, 4096, 16, 16, 1024, 4096, 4096, 1024)    |
| 32     | 0.233   | (4, 2048, 2048, 2048, 16, 16, 2048, 2048, 2048, 2048)    |
| 64     | 0.316   | (8, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)    |
| 128    | 0.449   | (16, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 256    | 0.711   | (32, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 512    | 1.157   | (64, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 1024   | 2.182   | (64, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 2048   | 4.196   | (64, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 4096   | 8.084   | (64, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 8192   | 15.982  | (64, 2048, 2048, 2048, 32, 32, 2048, 2048, 2048, 2048)   |
| 16384  | 32.419  | (128, 2048, 1024, 1024, 64, 64, 2048, 1024, 1024, 2048)  |
| 32768  | 63.359  | (128, 2048, 1024, 1024, 128, 64, 2048, 1024, 1024, 2048) |

## FP8 Comparison (sparse_hotspot vs zipf)

| tokens | hotspot (ms) | zipf (ms) | winner       |
|--------|-------------|-----------|--------------|
| 16     | 0.219       | 0.167     | zipf         |
| 32     | 0.232       | 0.233     | ~same        |
| 64     | 0.262       | 0.316     | hotspot      |
| 128    | 0.322       | 0.449     | hotspot      |
| 256    | 0.496       | 0.711     | hotspot      |
| 512    | 0.855       | 1.157     | hotspot      |
| 1024   | 1.694       | 2.182     | hotspot      |
| 2048   | 3.548       | 4.196     | hotspot      |
| 4096   | 7.076       | 8.084     | hotspot      |
| 8192   | 13.986      | 15.982    | hotspot      |
| 16384  | 27.751      | 32.419    | hotspot      |
| 32768  | 54.316      | 63.359    | hotspot      |

Note: The latency comparison above is NOT directly comparable since the configs were
tuned under different routing distributions. The user should run both config sets under
the actual workload to determine which performs better in practice.

## Serving Benchmark Results

Server config: tp_size=8, dp_size=2, ep_size=8, moe_backend=epmoe, context_length=262144,
chunked_prefill_size=2048, page_size=256, dtype=bfloat16, mem_fraction_static=0.95,
swa_full_tokens_ratio=0.2, max_running_requests=128, dp_schedule_policy=round_robin

Workload: random, input_len=16384, output_len=1024, range_ratio=1.0, seed=12345

### Config A (sparse_hotspot FP8) — 64 Concurrency

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    100.0
Max request concurrency:                 64
Successful requests:                     640
Benchmark duration (s):                  1407.04
Total input tokens:                      10485760
Total input text tokens:                 10485760
Total generated tokens:                  655360
Total generated tokens (retokenized):    655376
Request throughput (req/s):              0.45
Input token throughput (tok/s):          7452.35
Output token throughput (tok/s):         465.77
Peak output token throughput (tok/s):    1920.00
Peak concurrent requests:                128
Total token throughput (tok/s):          7918.12
Concurrency:                             63.97
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   140647.69
Median E2E Latency (ms):                 140706.60
P90 E2E Latency (ms):                    142347.84
P99 E2E Latency (ms):                    143087.15
---------------Time to First Token----------------
Mean TTFT (ms):                          50545.93
Median TTFT (ms):                        50632.62
P99 TTFT (ms):                           100268.85
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          88.08
Median TPOT (ms):                        88.25
P99 TPOT (ms):                           136.17
---------------Inter-Token Latency----------------
Mean ITL (ms):                           88.08
Median ITL (ms):                         36.82
P95 ITL (ms):                            41.13
P99 ITL (ms):                            48.48
Max ITL (ms):                            100724.46
==================================================
```

### Config A (sparse_hotspot FP8) — 128 Concurrency

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    100.0
Max request concurrency:                 128
Successful requests:                     1280
Benchmark duration (s):                  2763.55
Total input tokens:                      20971520
Total input text tokens:                 20971520
Total generated tokens:                  1310720
Total generated tokens (retokenized):    1310832
Request throughput (req/s):              0.46
Input token throughput (tok/s):          7588.61
Output token throughput (tok/s):         474.29
Peak output token throughput (tok/s):    2048.00
Peak concurrent requests:                256
Total token throughput (tok/s):          8062.90
Concurrency:                             127.95
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   276254.73
Median E2E Latency (ms):                 276097.19
P90 E2E Latency (ms):                    278405.52
P99 E2E Latency (ms):                    278579.65
---------------Time to First Token----------------
Mean TTFT (ms):                          101421.88
Median TTFT (ms):                        101618.05
P99 TTFT (ms):                           201590.94
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          170.90
Median TPOT (ms):                        171.15
P99 TPOT (ms):                           267.27
---------------Inter-Token Latency----------------
Mean ITL (ms):                           170.90
Median ITL (ms):                         69.32
P95 ITL (ms):                            79.28
P99 ITL (ms):                            88.32
Max ITL (ms):                            203324.42
==================================================
```

### Config A Summary

| Metric                    | 64 concurrency | 128 concurrency |
|---------------------------|----------------|-----------------|
| Output throughput (tok/s) | 465.77         | 474.29          |
| Peak output (tok/s)       | 1920.00        | 2048.00         |
| Total throughput (tok/s)  | 7918.12        | 8062.90         |
| Request throughput (req/s)| 0.45           | 0.46            |
| Mean TPOT (ms)            | 88.08          | 170.90          |
| Median ITL (ms)           | 36.82          | 69.32           |
| Mean TTFT (ms)            | 50545.93       | 101421.88       |
| Mean E2E (ms)             | 140647.69      | 276254.73       |

Note: 128 concurrency achieves slightly higher throughput but at ~2x latency.
Also fixed: `np.empty` → `np.zeros` in `schedule_batch.py` to resolve SWA IndexError.

### Config B (zipf FP8) — 64 Concurrency

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    100.0
Max request concurrency:                 64
Successful requests:                     640
Benchmark duration (s):                  1415.23
Total input tokens:                      10485760
Total input text tokens:                 10485760
Total generated tokens:                  655360
Total generated tokens (retokenized):    655584
Request throughput (req/s):              0.45
Input token throughput (tok/s):          7409.22
Output token throughput (tok/s):         463.08
Peak output token throughput (tok/s):    1856.00
Peak concurrent requests:                128
Total token throughput (tok/s):          7872.30
Concurrency:                             63.97
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   141467.69
Median E2E Latency (ms):                 141405.74
P90 E2E Latency (ms):                    142186.08
P99 E2E Latency (ms):                    142401.71
---------------Time to First Token----------------
Mean TTFT (ms):                          50688.08
Median TTFT (ms):                        50722.28
P99 TTFT (ms):                           100539.44
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          88.74
Median TPOT (ms):                        88.87
P99 TPOT (ms):                           136.85
---------------Inter-Token Latency----------------
Mean ITL (ms):                           88.74
Median ITL (ms):                         37.34
P95 ITL (ms):                            41.04
P99 ITL (ms):                            48.02
Max ITL (ms):                            100647.77
==================================================
```

### Config B (zipf FP8) — 128 Concurrency

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    100.0
Max request concurrency:                 128
Successful requests:                     1280
Benchmark duration (s):                  2760.69
Total input tokens:                      20971520
Total input text tokens:                 20971520
Total generated tokens:                  1310720
Total generated tokens (retokenized):    1311142
Request throughput (req/s):              0.46
Input token throughput (tok/s):          7596.49
Output token throughput (tok/s):         474.78
Peak output token throughput (tok/s):    2162.00
Peak concurrent requests:                256
Total token throughput (tok/s):          8071.27
Concurrency:                             127.95
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   275964.49
Median E2E Latency (ms):                 275843.00
P90 E2E Latency (ms):                    277000.88
P99 E2E Latency (ms):                    278603.46
---------------Time to First Token----------------
Mean TTFT (ms):                          101933.41
Median TTFT (ms):                        101849.90
P99 TTFT (ms):                           202392.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          170.12
Median TPOT (ms):                        170.36
P99 TPOT (ms):                           266.45
---------------Inter-Token Latency----------------
Mean ITL (ms):                           170.12
Median ITL (ms):                         67.40
P95 ITL (ms):                            83.69
P99 ITL (ms):                            93.96
Max ITL (ms):                            203191.12
==================================================
```

### Config A vs Config B Comparison

#### 64 Concurrency

| Metric                    | Config A (hotspot) | Config B (zipf) | Winner   |
|---------------------------|--------------------|-----------------|----------|
| Output throughput (tok/s) | 465.77             | 463.08          | A (+0.6%)|
| Peak output (tok/s)       | 1920.00            | 1856.00         | A (+3.4%)|
| Total throughput (tok/s)  | 7918.12            | 7872.30         | A (+0.6%)|
| Mean TPOT (ms)            | 88.08              | 88.74           | A (-0.7%)|
| Median ITL (ms)           | 36.82              | 37.34           | A (-1.4%)|
| P99 ITL (ms)              | 48.48              | 48.02           | B (-1.0%)|
| Mean TTFT (ms)            | 50545.93           | 50688.08        | A (-0.3%)|
| Mean E2E (ms)             | 140647.69          | 141467.69       | A (-0.6%)|

#### 128 Concurrency

| Metric                    | Config A (hotspot) | Config B (zipf) | Winner   |
|---------------------------|--------------------|-----------------|----------|
| Output throughput (tok/s) | 474.29             | 474.78          | B (+0.1%)|
| Peak output (tok/s)       | 2048.00            | 2162.00         | B (+5.6%)|
| Total throughput (tok/s)  | 8062.90            | 8071.27         | B (+0.1%)|
| Mean TPOT (ms)            | 170.90             | 170.12          | B (-0.5%)|
| Median ITL (ms)           | 69.32              | 67.40           | B (-2.8%)|
| P99 ITL (ms)              | 88.32              | 93.96           | A (-6.0%)|
| Mean TTFT (ms)            | 101421.88          | 101933.41       | A (-0.5%)|
| Mean E2E (ms)             | 276254.73          | 275964.49       | B (-0.1%)|

### Conclusion

Config A and Config B perform nearly identically under real serving workload (random
input_len=16384, output_len=1024). Differences are within noise margin (<1% on most
throughput/latency metrics).

- At 64 concurrency: Config A has a slight edge in throughput and latency.
- At 128 concurrency: Config B has a slight edge in throughput; Config A has better P99 ITL.
- The differences are negligible (<1-3%) and not statistically significant from a single run.

**Recommendation**: Keep Config A (sparse_hotspot) as the default, since it performs
marginally better at the more common 64-concurrency operating point and has more
consistent P99 tail latency at high concurrency.
