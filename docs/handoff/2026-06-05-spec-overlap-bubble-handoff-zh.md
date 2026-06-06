# Spec Decode Scheduler Overlap 交接文档

更新时间：2026-06-07 00:00 CST

## 当前跟进状态（2026-06-07）

### 2026-06-07 待执行：bench_serving 16k input / 1k output 吞吐

用户要求：

- 开始 bench_serving 前，先把当前可用代码和交接进展推送到远端。
- 重新部署 server，不复用当前占 TPU 的旧 server。
- 使用 bench_serving 接口测试：
  - input len: `16384`
  - output len: `1024`
  - batch/concurrency: `32`、`64`、`128`
- 启动参数需要适配该测试规模，重点是 `context-length >= 17408`、`max-prefill-tokens` 足够容纳 16k 输入、`max-running-requests >= 128`，并保留 overlap/spec decode 相关优化环境变量。
- 测试仍在四个 `perf-16-*` rank pod 上执行，部署前后必须确认四个 pod 代码一致。

本节开始前已完成的当前进展：

- KV cache 分配/释放 lifecycle 已对齐上游方向，32 并发 SWA OOM/Empty reply crash 已修复。
- `get_eagle_multi_step_metadata()` 已恢复 d52/upstream 的 draft `page_indices` 固定 `16384` 容量，accept-rate 回退主因已修复。
- 当前 overlap+same-batch chain xprof 中 `broadcast -> verify` device gap 平均约 `7.23 us`。
- GSM8K evalscope 全量 `1319` 条结果 `AverageAccuracy=0.953`。

待填 bench_serving 结果：

```text
server run_id:
remote branch/commit:
runtime hash check:

bsz=32:
bsz=64:
bsz=128:
```

启动/调试记录：

- 已先推送 `origin/dev/spec-overlap-bubble-followup-codex`，commit `d27127f10917a43a7548a6e3c9525856914cb818`。
- 初次长上下文 server `bench16k1k_004426` 使用：
  - `--context-length 18432`
  - `--max-prefill-tokens 16384`
  - `--chunked-prefill-size 4096`
  - `--max-running-requests 128`
  - `--precompile-bs-paddings 32 64 128`
  - `--precompile-token-paddings 4096 8192 16384`
  - `SGL_JAX_ENABLE_SAME_BATCH_SPEC_CHAIN=1`
- 初次 `bench_serving` smoke：`num_prompts=4`、`random_input_len=16384`、`random_output_len=16`、`max_concurrency=4` 触发 crash。
- crash 根因：
  - chunked prefill 第二块调用 `SWARadixCache.cache_unfinished_req()`。
  - `req.swa_evicted_seqlen=4096` 时，insert 会把 full prefix 前半段作为 SWA tombstone 保留。
  - public `match_prefix()` 会按 SWA window 过滤 tombstone path，只返回约 4096 个 SWA-safe indices。
  - `cache_unfinished_req()` 需要的是 full prefix writeback；使用 filtered match 导致 `new_prefix_len=8192` 但 `len(new_indices)=4096`，触发 assert：
    `python/sgl_jax/srt/mem_cache/swa_radix_cache.py:483`。
- 修复：
  - 新增内部 `_match_full_prefix()`，只供 `cache_unfinished_req()` 在 insert 后回写 full prefix path。
  - 保持 public `match_prefix()` SWA-filtered 语义不变。
- pod rank0 CPU focused 验证：
  - `test_cache_unfinished_req_writeback_includes_swa_tombstone_prefix`: `1 passed`
  - `test_cache_unfinished_req_writeback_range` + 新测试：`2 passed`

## 当前跟进状态（2026-06-06）

### 2026-06-06 后续更新：accept-rate 回退根因已定位并修复

本轮继续在当前 worktree：

```text
/Users/niu/code/sglang-jax/.worktrees/spec-overlap-bubble-followup-codex
```

新增关键修复：

- 将 `FlashAttention.get_eagle_multi_step_metadata()` 的 draft multi-step `page_indices` 输出容量恢复为 d52/upstream 口径的固定 `16384`。
  - 之前 current 把该容量改成了 `full_size`，与 d52 和现有 DP shape invariant 不一致。
  - 该 metadata 负责 prefill 后 MTP draft 的 KV page 布局；容量/DP 分段布局错误会污染 draft topk，导致 first verify 开始接受率回退。
  - 修复后不改 KV lifecycle、same-batch chain、device-frontier/JIT overlap 逻辑。

四个 perf-16 rank 已同步并确认关键 runtime 文件 hash 一致，其中：

```text
flashattention_backend.py  c8434bf4b87ddac751f19fd913936571932eb16609436f38e4bfafa1640e7390
```

同口径 d52 baseline：

```text
d52 baseline no-overlap running32 accept_ratio_mean ~= 0.8350
```

修复后 pod 验证：

```text
monolithic/no-overlap:
  run_id: accept_current_monolithic_targetpadding_clean_213814
  tag: gsm8k8x4_current_monolithic_targetpadding_retry_134530
  ok=32 curl_bad=0 json_bad=0
  finish_counts={'stop': 32}
  running32 accept_ratio_mean=0.8430
  running32 accept_len_mean=3.3741
  running32 throughput_mean=2963.75 token/s

split/no-overlap:
  run_id: accept_current_split_targetpadding_214727
  tag: gsm8k8x4_current_split_targetpadding_135247
  ok=32 curl_bad=0 json_bad=0
  finish_counts={'stop': 32}
  running32 accept_ratio_mean=0.8444
  running32 accept_len_mean=3.3800
  running32 throughput_mean=2933.12 token/s

overlap + same-batch chain:
  run_id: overlap_chain_targetpadding_215442
  tag: gsm8k8x4_overlap_chain_targetpadding_135952
  ok=32 curl_bad=0 json_bad=0
  finish_counts={'stop': 32}
  running32 accept_ratio_mean=0.8260
  running32 accept_len_mean=3.3044
  running32 throughput_mean=1599.84 token/s

overlap + same-batch chain warm2:
  tag: gsm8k8x4_overlap_chain_targetpadding_warm2_140242
  elapsed=6s
  ok=32 curl_bad=0 json_bad=0
  finish_counts={'stop': 32}

overlap + same-batch chain profile trigger:
  tag: gsm8k8x4_overlap_chain_targetpadding_profile_140539
  elapsed=17s
  ok=32 curl_bad=0 json_bad=0
  finish_counts={'stop': 32}
```

四 rank 错误 grep 为空，无 `Traceback/OOM/Empty reply`。

pod 上 focused CPU 测试：

```text
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_dp_shapes.py::test_draft_page_indices_dp_segmented -q

6 passed
```

当前修复后 xprof：

```text
remote:
/tmp/profile_overlap_chain_targetpadding_215442/decode6_after_decode32_targetpadding/plugins/profile/2026_06_06_14_05_45/perf-16-0.trace.json.gz

local:
profiles/overlap_chain_targetpadding_215442/extracted/profile_overlap_chain_targetpadding_215442/decode6_after_decode32_targetpadding/plugins/profile/2026_06_06_14_05_45/perf-16-0.trace.json.gz

archive:
profiles/overlap_chain_targetpadding_215442/profile_overlap_chain_targetpadding_215442.tar.gz
sha256: 0bb73e90cdfaba204e8426324931e80d413f07992cfcb1c663b8b9cfbae895c6
```

xprof gap 结论：

```text
device items: 24
jit_fused_draft_extend: n=6, mean=1310.96 us
jit_broadcast_in_dim: n=6, median=11.38 us, mean=69.65 us, max=256.16 us
jit_fused_greedy_verify: n=12, mean=28003.05 us

draft -> broadcast device gap: n=6, mean=560.39 us, median=550.26 us, max=654.02 us
broadcast -> verify device gap: n=6, mean=7.23 us, median=7.24 us, max=7.28 us
verify -> verify device gap: n=11, mean=1086.45 us, median=1880.66 us, max=2264.81 us

trace contains forward_batch_speculative_chained_verify_phase annotations,
so same-batch chain path was hit.
```

GSM8K evalscope 全量：

```text
run_id: overlap_chain_targetpadding_215442
log: /tmp/evalscope_gsm8k_overlap_chain_targetpadding_215442.log
remote output: /tmp/sglang-jax/python/outputs/20260606_140928
dataset: gsm8k
metric: AverageAccuracy
num: 1319
score: 0.953
```

evalscope 后再次 grep 四 rank 错误日志为空，server 仍存活。

当前结论：

- accept-rate 回退主因已确认并修复；no-overlap split/monolithic 均恢复到 d52 baseline 附近。
- 32 并发 crash 修复仍成立；overlap+chain 路径多轮 32 并发、profile-trigger 请求和 GSM8K 全量 eval 后均无 crash，全部 `stop`。
- chain 路径 running32 accept 仍有正常波动，本轮 `0.8260`，低于 no-overlap `0.8444` 但已通过 GSM8K 全量正确性验证。
- 修复后的当前代码已重新下载 xprof；`broadcast -> verify` device gap 平均约 `7.23 us`，达到几十 us 以内目标。

### 2026-06-06 后续更新：prefill seq_len 修复后 32 并发、chain、xprof 均通过

本轮在当前 worktree：

```text
/Users/niu/code/sglang-jax/.worktrees/spec-overlap-bubble-followup-codex
```

新增关键修复：

- overlap + spec prefill 路径在 append prefill sampled token 到 `req.output_ids` 后，同步推进 scheduler-visible `info.seq_lens` / `seq_lens_sum`。
  - 根因证据：debug 对比显示 overlap 首轮 spec decode 的 `seq_len` 比 no-overlap 少 1，但 `kv_committed_len` 一样；token 轨迹随后分叉并导致大量 `finish_reason=length`。
  - 修复后 4 请求 debug 中，overlap 前 5 次 verify 的 `seq_len/kv/tokens` 与 no-overlap 对齐，输出 hash 对齐，全部 `stop`。
- spec decode result processing 中按 `accepted_len - 1` 补齐 `req.kv_committed_len`，保持 `prepare_for_decode()` 预提交 1 个 token、result 阶段补齐剩余 accepted draft tokens 的语义。

四个 perf-16 rank 当前关键文件 hash 一致：

```text
scheduler.py                         b632b24deed2d0b104767c50f44de1a3c3b6cdb10ea118f1ee89272e86b88ca6
scheduler_output_processor_mixin.py  4b41956d51c5f10963f1bee16d492ec591b09f8071c2ca7b7835e246ade6807f
tp_worker_overlap_thread.py          96ff3c14bb14046d72002d8f3c624bf569f68cfd443cc24d0138d311ae5b1ba3
schedule_batch.py                    78d740fd08b1815dcc7b69aa16a5283d5ada211e578c7674766431172aa4f25c
eagle_util.py                        b4691f9061fa056d563c2bb0534a054650b3c4c1b57636b2ba601cb1f55327a1
```

no-chain overlap 32 并发最终复测：

```text
run_id: overlap_nochain_prefillfix_final_210316
tag: gsm8k8x4_overlap_nochain_prefillfix_final_130829
ok=32 curl_bad=0 json_bad=0 err_files=0
finish_counts={'stop': 32}
running32 accept_ratio_mean=0.8442
running32 accept_len_mean=3.3769
running32 throughput_mean=3307.68 token/s
四 rank alive，错误 grep 为空
```

chain overlap 32 并发与稳定性复测：

```text
run_id: overlap_chain_prefillfix_211147

single profile-trigger batch:
tag: gsm8k8x4_overlap_chain_profile_132017
ok=32 curl_bad=0 json_bad=0 err_files=0
finish_counts={'stop': 32}
running32 accept_ratio_mean=0.8360
running32 accept_len_mean=3.3468
running32 throughput_mean=1539.11 token/s

warm final r1:
tag: gsm8k8x4_overlap_chain_final_r1_132606
ok=32 curl_bad=0 json_bad=0 err_files=0
finish_counts={'stop': 32}
running32 accept_ratio_mean=0.8404
running32 accept_len_mean=3.3600
running32 throughput_mean=1562.64 token/s

warm final r2:
tag: gsm8k8x4_overlap_chain_final_r2_132613
ok=32 curl_bad=0 json_bad=0 err_files=0
finish_counts={'stop': 32}
running32 accept_ratio_mean=0.8437
running32 accept_len_mean=3.3752
running32 throughput_mean=1571.69 token/s

warm final r3:
tag: gsm8k8x4_overlap_chain_final_r3_132619
ok=32 curl_bad=0 json_bad=0 err_files=0
finish_counts={'stop': 32}
running32 accept_ratio_mean=0.8276
running32 accept_len_mean=3.3088
running32 throughput_mean=1593.56 token/s
```

四 rank 最终错误 grep 为空：

```text
Traceback|RuntimeError|Exception|fatal|F0606|E0606|Scheduler hit|ModelWorkerClient hit|
Received sigquit|ValueError|AssertionError|OOM|Out of memory|memory leak|
Watchdog timeout|Empty reply
```

三轮 warm final decode log 均未出现 cache miss 字样。

当前 xprof：

```text
profiles/overlap_chain_prefillfix_211147_gsm8k8x4_profile_132017/extracted/profile_overlap_chain_prefillfix_211147/gsm8k8x4_overlap_chain_profile_132017_decode6_after_decode32/plugins/profile/2026_06_06_13_20_24/perf-16-0.trace.json.gz
sha256: fe78d4008ebe4fd91d6a83ea17004cb7864b60a8d58bc72d9429b1c74d2c7fec

archive:
profiles/overlap_chain_prefillfix_211147_gsm8k8x4_profile_132017/profile_overlap_chain_prefillfix_211147.tar.gz
sha256: b6bfae6a1f7779491c7034797351ca6aabdfbfa2c507b06f3dc7dae90b8a807d
```

xprof gap 结论：

```text
JIT broadcast -> fused_greedy_verify gap:
  median about 7.2 us
  p90 about 24.8 us
  max about 62.5 us

JIT draft -> broadcast gap:
  mean about 0.58 ms

trace contains forward_batch_speculative_chained_verify_phase 112-117,
so same-batch chain path was hit.
```

当前剩余 TODO：

- chain 路径 device idle 已达几十 us，但 rank0 decode log 的 `running32 throughput_mean` 约 `1.5-1.6k token/s`，低于 no-chain 约 `3.3k token/s`；同时 wall-clock 32 请求约 6-7 秒，说明日志吞吐口径和 end-to-end 口径需要进一步解释或修正。
- accept rate 已恢复到 baseline 附近，但 chain 下仍有波动：`running32 accept_ratio_mean` 约 `0.8276-0.8437`，需要保留为最终 evalscope/GSM8K 全量验收关注项。
- 最终合入前建议再跑一次更正式的 GSM8K/evalscope 或项目标准吞吐脚本。

### 2026-06-06 关键更新：device-frontier early enqueue 已把 device idle 压到 us 级

当前最新本地 worktree：

```text
/Users/niu/code/sglang-jax/.worktrees/spec-overlap-bubble-followup-codex
```

当前最新 pod run：

```text
current_chain_device_frontier_161511
```

四个 perf-16 rank 已同步关键文件并校验 hash 一致：

```text
flashattention_backend.py      62fc302357351819f85f4b62d0aa41771ffc6b791e83dfcf5ff92bb29a6e1ff7
tp_worker_overlap_thread.py    bf48543c27fc920bd1ad438cf4db6bf49e1947c939e3af65bb14b5783c4e2836
draft_extend_fused.py          c4c8e4a2384c615f9cb7c02da93f277408b332bf01e14b6a531bf626695752f2
scheduler.py                   b632b24deed2d0b104767c50f44de1a3c3b6cdb10ea118f1ee89272e86b88ca6
schedule_batch.py              c049f9dfc22c461483e3b89e69fa3cec5626fea9bdf6d69c7a8f2034b555c47b
```

本轮实现要点：

- 回退了负收益的 Phase-A prepared-launch 复用尝试。该尝试的 run `current_chain_phasea_prepare_reuse_155553` 虽然不 crash，但 `broadcast_to_verify` 从约 `2.5 ms` 退化到约 `2.9 ms`。
- 新增 device-frontier same-batch chain candidate：当 Phase-B pending 只有 device `new_seq_lens` 时，不再因为缺 host seq_lens 跳过；用 reserved `verify_write_lens` 推导 host upper-bound frontier 只用于 metadata/cache_loc shape，真实 `ForwardBatch.seq_lens` 仍走 device `new_seq_lens`。
- 对 device-frontier candidate，commit guard 不再要求 host `seq_lens` 完全相等，但仍要求 `req_pool_indices`、`allocate_lens`、`verify_write_lens` 匹配，并继续延迟提交 target KV pool updates，保留之前的输出污染/crash 修复。

32 并发 GSM8K 8 条样本 x4 结果：

```text
run_id: current_chain_device_frontier_161511

first batch:
  tag: gsm8k8x4_chain_device_frontier_082030
  ok=32 curl_bad=0 json_bad=0 err_files=0
  running32 accept_ratio_mean=0.8271
  running32 accept_len_mean=3.3073
  running32 throughput_mean=1624.57 token/s
  output_unique=32

profile batch:
  tag: gsm8k8x4_chain_device_frontier_profile_082342
  ok=32 curl_bad=0 json_bad=0 err_files=0
  running32 accept_ratio_mean=0.8135
  running32 accept_len_mean=3.2513
  running32 throughput_mean=1511.53 token/s
  output_unique=32
```

四 rank 错误 grep 为空：

```text
Traceback|RuntimeError|Exception|fatal|F0606|E0606|Scheduler hit|ModelWorkerClient hit|
Received sigquit|ValueError|AssertionError|memory leak|OOM|OutOfMemory|
_concat_spec_info_per_rank|Array has been deleted|Empty reply
```

最新 xprof：

```text
profiles/current_chain_device_frontier_161511/perf-16-0.trace.json.gz
sha256: 737b0c28473c946e8fa963f923b0f88004c5fb048366a33d88ae16b7aac04630
remote source:
/tmp/profile_current_chain_device_frontier_161511/decode6_after_decode32_device_frontier/plugins/profile/2026_06_06_08_23_49/perf-16-0.trace.json.gz
```

device gap 结论：

```text
verify_to_verify:
  mean about 7.6 us

draft_to_broadcast:
  mean about 0.59-0.60 ms

broadcast_to_verify:
  TPU threads mostly about 7.2 us
  worst observed spikes about 52-59 us on two threads
```

这说明原先 Phase-B 后 host submit 暴露出来的 ms 级 device idle 已被 device-frontier early enqueue 隐藏掉，当前稳态 device idle 已达到几十 us 目标。

仍未完成：

- accept-rate 仍有回退/波动。d52 GSM8K baseline running32 约 `0.8450` 或旧批次 `0.8308`；当前 device-frontier 两轮为 `0.8271` / `0.8135`，不能视为完全恢复。
- cache miss / reserve miss 回退仍需保留待办。
- host `submit_fused_greedy_verify_jit` 在 xprof 中变成 `~9-20 ms`，但已经提前排队并和 device work 重叠；后续可作为 host overhead 优化，不再是 device idle blocker。
- 还需要更长 32 并发稳定性、最终 GSM8K/evalscope 和吞吐验收。

当前开发 worktree：

```bash
/Users/niu/code/sglang-jax/.worktrees/spec-overlap-bubble-followup-codex
```

当前分支：

```bash
dev/spec-overlap-bubble-followup-codex
```

pod 测试固定使用 4 个 rank：

```bash
perf-16-0-jgb5c
perf-16-1-zhn6d
perf-16-2-hs7kc
perf-16-3-vqw2x
```

注意：pod 上代码经常和本地 HEAD 不一致。测试前必须把当前改动同步到 4 个 pod，并用 `sha256sum` 确认相关文件四个 pod 完全一致。

### Baseline 回归标准

已按用户指定使用 `sgl-project` 的 `epic/mtp-phase2-performance` commit：

```bash
d52a68b350b27ced3c3ed43a597032cef63b7387
```

作为 baseline。该 commit 在 `--disable-overlap-schedule` 下 32 并发 GSM8K first 8 x4：

- run id: `accept_d52_nooverlap_recheck_005817`
- 请求结果：`ok 32 bad 0`
- rank 日志无 `Traceback/OOM/Empty reply`
- all decode: `accept_len_mean=3.5216`, `accept_ratio_mean=0.8804`
- running32: `accept_len_mean=3.3356`, `accept_ratio_mean=0.8350`

这个值作为当前 accept-rate 回归标准。

### 当前代码已验证结果

当前 split/no-overlap 路径：

- run id: `accept_current_nooverlap_norefresh_013843`
- 请求结果：`ok 32 bad 0`
- health: `200`
- rank 日志无 `Traceback/OOM/Empty reply`
- all decode: `accept_len_mean=1.2578`, `accept_ratio_mean=0.3148`
- running32: `accept_len_mean=1.2062`, `accept_ratio_mean=0.3031`

结论：32 并发 crash 已修，但 accept-rate 明显低于 d52 baseline。

当前临时单体 fused decode 诊断路径：

- 开关：`SGL_JAX_SPEC_DECODE_MONOLITHIC=1`
- run id: `accept_current_nooverlap_monolithic2_015455`
- 请求结果：`ok 32 bad 0`
- health: `200`
- rank 日志无 `Traceback/OOM/Empty reply`
- all decode: `accept_len_mean=1.6062`, `accept_ratio_mean=0.4022`
- running32: `accept_len_mean=2.5833`, `accept_ratio_mean=0.6450`

结论：split phase 是接受率回退的主要来源之一，但不是全部根因；单体路径也没有恢复到 d52 的 `~0.835`。

### 2026-06-06 first-verify 对照诊断

本轮在 pod 上做了 current 与 d52 的同口径 GSM8K first 8 x4 32 并发对照，均为 `--disable-overlap-schedule`，并确保每次测试前四个 pod 代码哈希一致。

current 诊断过程：

- `debug_current_firstverify_count_032658`
  - current 源码四 pod 一致，`SGL_JAX_SPEC_DECODE_MONOLITHIC=1`
  - 请求结果：`ok 32 bad 0`, health `200`
  - all decode: `accept_len_mean=1.5336`, `accept_ratio_mean=0.3837`
  - running32: `accept_len_mean=2.7283`, `accept_ratio_mean=0.6800`
  - 发现：first-verify dump 没打出，原因是 hook 只在 split helper 中，monolithic 主体没调用。
- `debug_current_route_033616`
  - 增加临时 route debug 后确认 runtime 确实走 fused route：
    `can_use_fused=True`, `is_all_greedy=True`, `has_fused_state=True`, `is_precompile_dummy=False`
  - 请求结果：`ok 32 bad 0`, health `200`
  - all decode: `accept_len_mean=1.6710`, `accept_ratio_mean=0.4179`
  - running32: `accept_len_mean=2.3200`, `accept_ratio_mean=0.5780`
- `debug_current_firstverify_bs32_034505`
  - 把 first-verify hook 放到 monolithic 主体，并设置 `SGL_JAX_SPEC_DEBUG_FIRST_VERIFY_MIN_BS=32`
  - 请求结果：`ok 32 bad 0`, health `200`
  - all decode: `accept_len_mean=1.5556`, `accept_ratio_mean=0.3891`
  - running32: `accept_len_mean=2.9920`, `accept_ratio_mean=0.7480`
  - batch32 first-verify dump 头部：
    - `seq_lens`: `[82, 68, 91, 91, 83, 83, 82, 61, 137, 83, 82, 137]`
    - `req_pool_indices`: `[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`
    - `out_cache_loc`: `[210, 211, 212, 213, 324, 325, 326, 327, 475, 476, 477, 478]`
    - `previous_verified_id`: `[8420, 8420, 8420, 8420, 8420, 8420, 8420, 28084, 8420, 8420, 8420, 8420]`
    - `previous_token_list` head: `[374, 279, 3019, 374, 279, 3019, 374, 279, 3019, 374, 279, 3019]`
    - `positions/draft_token/draft_positions/retrive_index` 为跨 host non-addressable JAX Array，当前 dump helper 不能直接 `device_get`，需要下一步用 `process_allgather(..., tiled=True)` 补齐。

d52 baseline 对照：

- 临时源码树：`/tmp/sglang-jax-d52-diag`
- 基于 commit：`d52a68b350b27ced3c3ed43a597032cef63b7387`
- 只加 env-gated first-verify dump 和 `is_precompile_dummy=True`，避免 precompile 抢占 dump；测试后已把 pod 源码恢复 current。
- run id: `debug_d52_firstverify_bs32_035459`
- 请求结果：`ok 32 bad 0`, health `200`
- all decode: `accept_len_mean=3.5406`, `accept_ratio_mean=0.8850`
- running32: `accept_len_mean=3.2267`, `accept_ratio_mean=0.8050`
- batch32 first-verify dump 头部：
  - `seq_lens`: `[96, 52, 91, 96, 61, 91, 68, 83, 82, 82, 83, 137]`
  - `req_pool_indices`: `[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`
  - `out_cache_loc`: `[224, 225, 226, 227, 308, 309, 310, 311, 411, 412, 413, 414]`
  - `previous_verified_id`: `[8420, 2132, 8420, 8420, 28084, 8420, 8420, 8420, 8420, 8420, 8420, 1249]`
  - `previous_token_list` head: `[374, 279, 3019, 4990, 3070, 17, 374, 279, 3019, 374, 279, 3019]`
  - `positions/draft_token/draft_positions/retrive_index` 在 d52 为 host-addressable，头部全 0。

结论：

- current 32 并发 crash 仍未复现，三轮 current + 一轮 d52 均 `ok 32 bad 0`。
- accept-rate 回退仍成立：d52 `running32 accept_ratio_mean=0.8050`，current 最新 `0.7480`，而 current 全量平均仍只有 `0.3891`。
- first-verify 输入已经出现差异，但当前对照还不能直接归因，因为并发请求的完成/入队顺序可能不同；需要下一步把 dump 扩展为 per-request prompt/index 标识，并对 non-addressable JAX Array 用 allgather 打印 `draft_token/retrive_index/positions`。
- 临时 `SPEC_DEBUG_ROUTE/FIRST_VERIFY` 打印代码已从 current 分支移除；pod 当前已恢复 current 源码且无 server 进程。

本轮新增单变量诊断（均使用 `SGL_JAX_SPEC_DECODE_MONOLITHIC=1` + `--disable-overlap-schedule`，同一 GSM8K first 8 x4 32 并发）：

- `accept_current_monolithic_compactalloc_022837`
  - 目的：把 monolithic verify 的 `allocate_lens` 改回 d52 风格 compact 语义。
  - 结果：`ok 32 bad 0`, health `200`, 四 rank 无 crash/OOM。
  - all decode: `accept_len_mean=1.5673`, `accept_ratio_mean=0.3921`
  - running32: `accept_len_mean=2.1673`, `accept_ratio_mean=0.5432`
  - 结论：不是主因，已回滚。
- `accept_current_monolithic_nocacheowner_023738`
  - 目的：monolithic target logits metadata 不使用 cached `cache_owner=target_mr`，贴近 d52。
  - 结果：`ok 32 bad 0`, health `200`, 四 rank 无 crash/OOM。
  - all decode: `accept_len_mean=1.5354`, `accept_ratio_mean=0.3841`
  - running32: `accept_len_mean=2.4883`, `accept_ratio_mean=0.6250`
  - 结论：不是主因，已恢复 `cache_owner=target_mr`。
- `accept_current_monolithic_noseqlensoverride_024644`
  - 目的：monolithic 下 `ForwardBatch.seq_lens` 忽略 `target_verify_seq_lens_device`，恢复 d52 的 `batch.seq_lens` 来源。
  - 结果：`ok 32 bad 0`, health `200`, 四 rank 无 crash/OOM。
  - all decode: `accept_len_mean=1.4655`, `accept_ratio_mean=0.3663`
  - running32: `accept_len_mean=1.9753`, `accept_ratio_mean=0.4940`
  - 结论：不是主因；该开关不应保留为修复。

### 本轮新增修复/诊断

- 修复 precompile dummy/all-padding `accept_length=0` 触发 materialize assert 的 crash。
- 修复 `EagleDraftInput._ensure_host` 对 non-addressable `jax.Array` 直接 `np.asarray()` 的 crash，改用 `process_allgather(..., tiled=True)`。
- 修复 `EagleDraftWorker.copy_model_worker_batch_to_cpu` 同类 non-addressable `jax.Array` crash。
- 增加临时诊断开关 `SGL_JAX_SPEC_DECODE_MONOLITHIC=1`，默认关闭，只用于定位 split vs monolithic 接受率差异。
- 已尝试移除 target verify dynamic metadata refresh；接受率未恢复，说明它不是主要根因。
- 已尝试 monolithic compact `allocate_lens`、移除 monolithic target logits metadata cache owner、禁用 monolithic `target_verify_seq_lens_device` override；接受率均未恢复到 d52 baseline。

### 当前待办

- accept-rate 回退必须继续修：目标是同一 GSM8K 32 并发下 running32 accept-ratio 回到 d52 baseline `~0.835`。
- cache miss 回退已观察到，先记录为待办；当前优先级低于 crash 和 accept-rate。
- 需要继续对比 d52 monolithic 和当前 monolithic 路径，重点转向 prefill 后 draft state 和 first verify 输入：
  - `previous_verified_id`
  - `previous_token_list/topk_index`
  - prefill 后 `hidden_states`
  - first decode 的 `draft_token/retrive_index`
  - first target verify 的 `input_ids/positions/out_cache_loc/req_pool_indices`
  - per-request prompt/index 标识，用来消除并发入队顺序差异
  - 对 non-addressable JAX Array 使用 `process_allgather(..., tiled=True)` 后再 dump
  已排除的方向：monolithic compact `allocate_lens`、target logits metadata cache owner、`target_verify_seq_lens_device` override。
- `test_spec_dp_shapes.py` 当前 67 项 focused run 中有 15 个 DP shape/fake Req 失败，需要单独修或更新测试；此前较窄的 overlap/materialization guard 能过，但这组失败不能忽略。
- 最终性能验收前必须关闭 `SGL_JAX_SPEC_DECODE_MONOLITHIC`，恢复 overlap/split 优化路径，并重新跑 xprof。

## 目标

最终目标是让 spec decode 的 scheduler overlap 达到接近 non-spec overlap 的效果：

- 32 并发长 greedy 请求下，batch 间旧的约 `30 ms` bubble 被完全消除或只剩真实 device 依赖。
- 吞吐有明显提升。
- 接受率不因实现改动下降；如果请求完全一致，可以用相同请求的 accept-len / accept-ratio 做直接对比。
- 最终阶段 GSM8K 正确性不下降。
- KV allocator leak 可以先记录、最后修；不要把 leak/OOM 误判成 overlap 失败。

当前阶段的实际目标：先集中把剩余 `3-4 ms` verify-to-verify gap 继续压掉。

## 当前代码位置

本地 worktree：

```bash
/Users/niu/code/sglang-jax/.worktrees/fused-greedy-spec-decode-step3
```

当前分支：

```bash
dev/fused-greedy-device-chain-verify-inputs
```

当前 HEAD：

```bash
aac7a6ed16481422708e55596548b94cc12f83fe
```

HEAD commit：

```bash
aac7a6ed chore(spec): simplify fused spec decode route
```

remote 状态：当前分支 HEAD 与 `origin/dev/fused-greedy-device-chain-verify-inputs` 一致。

当前 worktree 仍有未提交改动：

```bash
M python/sgl_jax/srt/speculative/draft_extend_fused.py
M python/sgl_jax/srt/speculative/eagle_util.py
M python/sgl_jax/test/speculative/test_eagle_utils.py
```

当前未提交 diff 的含义：

- `draft_extend_fused.py`
  - padding seq 的 accept length 改为 0。
  - `_greedy_prepare_draft_inputs` 对 `accept_length=0` 做 safe clip，避免负 index。
  - `_materialize_fused_greedy_batch_output_for_scheduler` 里 `new_seq_lens` 改为从原始 seq lens 计算，而不是从 verify-expanded seq lens 直接加 accept lens。
- `eagle_util.py`
  - 删除旧的 `GreedySampleDeviceOutputs` / `greedy_sample_device_outputs` helper。
- `test_eagle_utils.py`
  - 增加 fused chain greedy/topk1 linear reference 测试。
  - 增加 padding accept length 为 0 的测试。
  - 增加 materialize 阶段 `new_seq_lens` 使用 original seq lens 的测试。

## Pod 信息

```bash
PODS=(
  perf-16-0-jgb5c
  perf-16-1-zhn6d
  perf-16-2-hs7kc
  perf-16-3-vqw2x
)
CONTAINER=jax-tpu
WORKDIR=/tmp/sglang-jax
PYWORKDIR=/tmp/sglang-jax/python
PORT=30271
```

rank0 登录：

```bash
kubectl exec -it perf-16-0-jgb5c -c jax-tpu -- bash
cd /tmp/sglang-jax/python
```

注意：

- 单个 pod 无法跑 TPU e2e；要么 pod 上 CPU 小测，要么 4 个 rank 一起跑。
- 本地不要跑 TPU 测试。
- `zsh` 数组是 1-indexed，批量 pod 操作建议全部包在 `bash -lc '...'` 里。

## 同步代码到 Pod

同步当前改动文件到 4 个 pod：

```bash
cd /Users/niu/code/sglang-jax/.worktrees/fused-greedy-spec-decode-step3

bash -lc '
PODS=(perf-16-0-jgb5c perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x)
FILES=(
  python/sgl_jax/srt/speculative/draft_extend_fused.py
  python/sgl_jax/srt/speculative/eagle_util.py
  python/sgl_jax/test/speculative/test_eagle_utils.py
)
for pod in "${PODS[@]}"; do
  echo "sync:$pod"
  COPYFILE_DISABLE=1 tar cf - "${FILES[@]}" |
    kubectl exec -i "$pod" -c jax-tpu -- tar xf - -C /tmp/sglang-jax/
done
'
```

如果后续改到 scheduler / worker overlap 相关文件，至少同步这些：

```bash
FILES=(
  python/sgl_jax/srt/layers/attention/flashattention_backend.py
  python/sgl_jax/srt/managers/schedule_batch.py
  python/sgl_jax/srt/managers/scheduler.py
  python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py
  python/sgl_jax/srt/managers/tp_worker_overlap_thread.py
  python/sgl_jax/srt/server_args.py
  python/sgl_jax/srt/speculative/base_worker.py
  python/sgl_jax/srt/speculative/draft_extend_fused.py
  python/sgl_jax/srt/speculative/eagle_draft_worker.py
  python/sgl_jax/srt/speculative/eagle_util.py
  python/sgl_jax/srt/speculative/eagle_worker.py
  python/sgl_jax/srt/speculative/multi_layer_draft_worker.py
)
```

## CPU 快速测试

pod rank0 上跑 focused tests：

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc '
cd /tmp/sglang-jax/python &&
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_eagle_utils.py -q
'
```

如果改到 overlap split tests，并且对应测试文件存在：

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc '
cd /tmp/sglang-jax/python &&
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_spec_overlap_split.py -q
'
```

## 清理 Server

使用 `[s]gl_jax.launch_server` 避免 `pkill -f` 匹配到当前 shell 命令自己：

```bash
bash -lc '
PODS=(perf-16-0-jgb5c perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x)
for pod in "${PODS[@]}"; do
  echo "kill:$pod"
  kubectl exec "$pod" -c jax-tpu -- bash -lc "
    pkill -9 -f \"[s]gl_jax.launch_server\" || true
    pkill -9 -f \"[u]v run python -m sgl_jax.launch_server\" || true
  "
done
'
```

确认无残留：

```bash
bash -lc '
PODS=(perf-16-0-jgb5c perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x)
for pod in "${PODS[@]}"; do
  echo "$pod"
  kubectl exec "$pod" -c jax-tpu -- bash -lc \
    "ps -ef | grep sgl_jax.launch_server | grep -v grep || true"
done
'
```

## 4-rank Server 启动方法

重要参数：

- 使用原有 spec flags，不新增 flag。
- log interval 参数名是 `--decode-log-interval=1`。
- 不要写成其他不存在的名字。
- 之前 `--disable-overlap-schedule` 是 runbook 里的旧模板；当前要验证 scheduler overlap 时，不要带这个参数。

推荐启动方式：用 `setsid ... < /dev/null > log 2>&1 &`，保证 `kubectl exec` 立即返回，避免只启动 rank0/rank1 后卡住。

```bash
bash -lc '
RUN_ID=20260605_spec_overlap_handoff
PODS=(perf-16-0-jgb5c perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x)
echo "$RUN_ID" > /tmp/current_specdecode_route_run_id
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "echo $RUN_ID > /tmp/current_specdecode_route_run_id"

for i in 0 1 2 3; do
  pod=${PODS[$i]}
  echo "start:$pod rank:$i"
  kubectl exec "$pod" -c jax-tpu -- bash -lc "
cd /tmp/sglang-jax/python &&
setsid /opt/venv/bin/uv run python -m sgl_jax.launch_server \
  --model-path /data/pc \
  --trust-remote-code \
  --speculative-algorithm NEXTN \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --tp-size 16 \
  --dp-size 4 \
  --ep-size 16 \
  --moe-backend epmoe \
  --host 0.0.0.0 \
  --port 30271 \
  --page-size 64 \
  --context-length 4096 \
  --max-prefill-tokens 4096 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --swa-full-tokens-ratio 0.5 \
  --max-running-requests 64 \
  --attention-backend fa \
  --decode-log-interval=1 \
  --precompile-bs-paddings 32 64 \
  --precompile-token-paddings 256 512 \
  --nnodes 4 \
  --node-rank $i \
  --dist-init-addr perf-16-0.perf-16-headless-svc:5000 \
  > /tmp/sglang_${RUN_ID}_rank${i}.log 2>&1 < /dev/null &
"
done
'
```

看 ready：

```bash
RUN_ID=$(cat /tmp/current_specdecode_route_run_id)
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "
tail -f /tmp/sglang_${RUN_ID}_rank0.log
"
```

错误扫描：

```bash
RUN_ID=$(cat /tmp/current_specdecode_route_run_id)
bash -lc '
PODS=(perf-16-0-jgb5c perf-16-1-zhn6d perf-16-2-hs7kc perf-16-3-vqw2x)
for pod in "${PODS[@]}"; do
  echo "$pod"
  kubectl exec "$pod" -c jax-tpu -- bash -lc "
    grep -E \"Traceback|RuntimeError|Exception|fatal|F0605|E0605|Scheduler hit|ModelWorkerClient hit|Received sigquit|ValueError|AssertionError|memory leak\" \
      /tmp/sglang_${RUN_ID}_rank*.log | grep -v server_args || true
  "
done
'
```

## 32 并发 Curl 测试方法

目的：不用 bench serving，直接构造 32 个并发 greedy 请求。必须设置 `temperature=0`，确保走 all-greedy 路径。

在 rank0 pod 上执行：

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc '
set -euo pipefail
cd /tmp/sglang-jax/python
RUN_ID=$(cat /tmp/current_specdecode_route_run_id 2>/dev/null || date +%Y%m%d_%H%M%S)
PROMPT=$(python3 - <<PY
print(" ".join(["Please solve this arithmetic problem step by step."] * 8))
PY
)
cat > /tmp/decode32_body_${RUN_ID}.json <<PY
{
  "model": "/data/pc",
  "messages": [
    {"role": "user", "content": "$PROMPT"}
  ],
  "temperature": 0,
  "max_tokens": 512
}
PY

rm -f /tmp/curl_decode32_${RUN_ID}_*.out
for i in $(seq 0 31); do
  curl -sS --max-time 600 \
    http://127.0.0.1:30271/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @/tmp/decode32_body_${RUN_ID}.json \
    > /tmp/curl_decode32_${RUN_ID}_${i}.out &
done
wait
echo "curl decode32 done: /tmp/curl_decode32_${RUN_ID}_*.out"
'
```

查看稳态 decode 日志：

```bash
RUN_ID=$(cat /tmp/current_specdecode_route_run_id)
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "
grep 'Decode batch' /tmp/sglang_${RUN_ID}_rank0.log | tail -30
"
```

关注字段：

- `accept-len`
- `accept-ratio`
- `gen throughput (token/s)`
- `#running-req`

不要用第一个 decode batch 判断吞吐；要看 `#running-req: 32` 后、没有 prefill/compile/profile stop 干扰的稳态 decode batch。

## Profiling 方法

当前代码支持 `/start_profile` endpoint，也可以用 `python -m sgl_jax.profiler`。

手动时序：

1. server ready。
2. 发 32 并发长请求。
3. 看到 rank0 日志第一次出现 `Decode batch. #running-req: 32` 后，再开始 profile。
4. profile 5-6 个 decode step。
5. 压缩并下载 `/tmp/profile_${RUN_ID}`。

启动 profile：

```bash
RUN_ID=$(cat /tmp/current_specdecode_route_run_id)
PROFILE_DIR=/tmp/profile_${RUN_ID}/decode6_after_decode32

kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "
curl -sS -X POST http://127.0.0.1:30271/start_profile \
  -H 'Content-Type: application/json' \
  -d '{
    \"output_dir\": \"${PROFILE_DIR}\",
    \"num_steps\": \"6\",
    \"activities\": [\"CPU\", \"GPU\"],
    \"host_tracer_level\": 2,
    \"python_tracer_level\": 1
  }'
echo
"
```

也可以用模块：

```bash
RUN_ID=$(cat /tmp/current_specdecode_route_run_id)
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "
cd /tmp/sglang-jax/python &&
PYTHONPATH=/tmp/sglang-jax/python /opt/venv/bin/python -m sgl_jax.profiler \
  --url http://127.0.0.1:30271 \
  --output-dir /tmp/profile_${RUN_ID} \
  --profile-name decode6_after_decode32 \
  --num-steps 6 \
  --host-tracer-level 2 \
  --python-tracer-level 1
"
```

如果要自动等到 `Decode batch. #running-req: 32` 再触发 profile：

```bash
RUN_ID=$(cat /tmp/current_specdecode_route_run_id)
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "
set -euo pipefail
LOG=/tmp/sglang_${RUN_ID}_rank0.log
PROFILE_DIR=/tmp/profile_${RUN_ID}/decode6_after_decode32
old_count=\$(grep -c 'Decode batch. #running-req: 32' \"\$LOG\" || true)
echo waiting_decode32_from_count=\$old_count
while true; do
  new_count=\$(grep -c 'Decode batch. #running-req: 32' \"\$LOG\" || true)
  if [ \"\$new_count\" -gt \"\$old_count\" ]; then
    break
  fi
  sleep 0.2
done
curl -sS -X POST http://127.0.0.1:30271/start_profile \
  -H 'Content-Type: application/json' \
  -d \"{\\\"output_dir\\\":\\\"\$PROFILE_DIR\\\",\\\"num_steps\\\":\\\"6\\\",\\\"activities\\\":[\\\"CPU\\\",\\\"GPU\\\"],\\\"host_tracer_level\\\":2,\\\"python_tracer_level\\\":1}\"
echo
"
```

## 下载 Profile

pod 上压缩：

```bash
RUN_ID=$(cat /tmp/current_specdecode_route_run_id)
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "
cd /tmp &&
tar -czf profile_${RUN_ID}.tar.gz profile_${RUN_ID}
ls -lh /tmp/profile_${RUN_ID}.tar.gz
"
```

推荐用 httpserver 下载，`kubectl cp` 大文件有时会 EOF：

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc '
cd /tmp &&
pkill -f "[p]ython3 -m http.server 18080" || true
setsid python3 -m http.server 18080 > /tmp/profile_http_18080.log 2>&1 < /dev/null &
sleep 1
ps -ef | grep "python3 -m http.server 18080" | grep -v grep
'
```

本地开 port-forward：

```bash
kubectl port-forward pod/perf-16-0-jgb5c 18080:18080
```

本地下载：

```bash
RUN_ID=20260605_spec_overlap_handoff
cd /Users/niu/code/sglang-jax/.worktrees/fused-greedy-spec-decode-step3
curl --fail --location --retry 3 \
  -O http://127.0.0.1:18080/profile_${RUN_ID}.tar.gz
gzip -t profile_${RUN_ID}.tar.gz
mkdir -p profile_${RUN_ID}
tar -xzf profile_${RUN_ID}.tar.gz -C profile_${RUN_ID}
```

停止 httpserver：

```bash
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc '
pkill -f "[p]ython3 -m http.server 18080" || true
'
```

停止本地 port-forward：

```bash
ps -ef | grep 'kubectl port-forward pod/perf-16-0-jgb5c 18080:18080' | grep -v grep
kill <pid>
```

## 当前最好 Profile 证据

最好的一次已下载到本地：

```bash
/Users/niu/code/sglang-jax/.worktrees/fused-greedy-spec-decode-step3/profile_20260605_114500_logits_meta_cache_decode6_after_decode32_12.tar.gz
```

解压目录：

```bash
/Users/niu/code/sglang-jax/.worktrees/fused-greedy-spec-decode-step3/profile_20260605_114500_logits_meta_cache_decode6_after_decode32_12
```

关键 trace：

```bash
/Users/niu/code/sglang-jax/.worktrees/fused-greedy-spec-decode-step3/profile_20260605_114500_logits_meta_cache_decode6_after_decode32_12/decode6_after_decode32_12/plugins/profile/2026_06_05_03_46_52/perf-16-0.trace.json.gz
```

同目录还有：

```bash
perf-16-0.xplane.pb
jit_fused_greedy_verify(...).hlo_proto.pb
jit_fused_draft_extend(...).hlo_proto.pb
```

这次 profile 的 TPU0 `jit_fused_greedy_verify` gaps：

```text
[4.032, 3.855, 3.083, 3.931, 3.698, 3.069] ms
```

gap breakdown：

- 每个 gap 开头有真实 device work：
  - `jit_fused_draft_extend`: 约 `1.31 ms`
  - `jit_gather + tiny broadcast`: 约 `0.5-0.6 ms`
- 真正 idle tail：约 `1.16-2.06 ms`
- idle 的直接原因不是 scheduler `queue.get` 了，而是下一轮 chained verify 的 host/JAX submit 路径还在 critical path：
  - `_build_same_batch_spec_chain_candidate_batch`: 约 `0.32-0.40 ms`
  - `get_eagle_forward_metadata`: 约 `0.46-0.66 ms`
  - `_forward_batch_init_new_preserve_device`: 约 `0.40-0.48 ms`
  - `PjitFunction(fused_greedy_verify)`: 约 `1.38-1.89 ms`

对应代码边界：

```bash
python/sgl_jax/srt/managers/tp_worker_overlap_thread.py
  _stash_same_batch_spec_chain_candidate
  _stash_prebuilt_same_batch_spec_chain_candidate

python/sgl_jax/srt/speculative/draft_extend_fused.py
  _forward_batch_init_new_preserve_device

python/sgl_jax/srt/layers/attention/flashattention_backend.py
  get_eagle_forward_metadata
```

稳态吞吐参考：

- 使用 `20260605_114500_logits_meta_cache` 的 rank0 log。
- 取 `#running-req: 32` 且进入 `#full token: 8128` plateau 后，排除明显 profile/请求结束低值。
- 平均吞吐约 `1609.8 token/s`。
- median 约 `1511.1 token/s`。
- 平均 `accept-len` 约 `1.42`。
- 平均 `accept-ratio` 约 `0.355`。

## Trace Gap 分析脚本

本地分析 downloaded trace：

```bash
cd /Users/niu/code/sglang-jax/.worktrees/fused-greedy-spec-decode-step3
python3 - <<'PY'
import gzip, json

trace = "profile_20260605_114500_logits_meta_cache_decode6_after_decode32_12/decode6_after_decode32_12/plugins/profile/2026_06_05_03_46_52/perf-16-0.trace.json.gz"
with gzip.open(trace, "rt") as f:
    data = json.load(f)

proc = {}
thread = {}
for e in data["traceEvents"]:
    if e.get("ph") == "M" and e.get("name") == "process_name":
        proc[e.get("pid")] = e.get("args", {}).get("name")
    if e.get("ph") == "M" and e.get("name") == "thread_name":
        thread[(e.get("pid"), e.get("tid"))] = e.get("args", {}).get("name")

base = min(e.get("ts", 0) for e in data["traceEvents"] if "ts" in e)
events = []
for e in data["traceEvents"]:
    if e.get("ph") != "X" or "ts" not in e or "dur" not in e:
        continue
    pid = e.get("pid")
    tid = e.get("tid")
    s = (e["ts"] - base) / 1000
    en = (e["ts"] + e["dur"] - base) / 1000
    events.append(
        {
            "name": e.get("name", ""),
            "start": s,
            "end": en,
            "dur": e["dur"] / 1000,
            "pname": proc.get(pid, str(pid)),
            "tname": thread.get((pid, tid), str(tid)),
        }
    )

verify = [
    e
    for e in events
    if e["pname"] == "/device:TPU:0" and "jit_fused_greedy_verify" in e["name"]
]

def overlap(e, a, b):
    return max(0, min(e["end"], b) - max(e["start"], a))

for i in range(len(verify) - 1):
    a = verify[i]["end"]
    b = verify[i + 1]["start"]
    dev = [
        e
        for e in events
        if e["pname"] == "/device:TPU:0"
        and overlap(e, a, b) > 0.005
        and "jit_fused_greedy_verify" not in e["name"]
    ]
    intervals = sorted((max(e["start"], a), min(e["end"], b), e["name"]) for e in dev)
    merged = []
    for s, en, _ in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, en])
        else:
            merged[-1][1] = max(merged[-1][1], en)
    busy = sum(en - s for s, en in merged)
    last = max((en for _, en, _ in intervals), default=a)
    draft = sum(overlap(e, a, b) for e in dev if "jit_fused_draft_extend" in e["name"])
    gather = sum(overlap(e, a, b) for e in dev if "jit_gather" in e["name"])
    print(
        i,
        "gap_ms",
        round(b - a, 3),
        "device_busy_ms",
        round(busy, 3),
        "idle_tail_ms",
        round(b - last, 3),
        "draft_ms",
        round(draft, 3),
        "gather_ms",
        round(gather, 3),
    )
PY
```

## 当前进度

### 2026-06-05 follow-up 更新

基于新 worktree `dev/spec-overlap-bubble-followup-codex` 的 KV cache lifecycle 对齐实现，当前阶段先验证 crash 修复，性能回退项先记录为待办。

已验证：

- 4-rank server run id: `deviceidle_kvslack4_active_224459`。
- 代码已同步到 4 个 rank pod: `perf-16-0-jgb5c`, `perf-16-1-zhn6d`, `perf-16-2-hs7kc`, `perf-16-3-vqw2x`。
- 带 profile 的 32 并发 `max_tokens=512` 请求：`ok=32 bad=0 err_files=0`。
- 追加 3 轮 32 并发 `max_tokens=512` 请求，三轮均为 `ok=32 bad=0 err_files=0`。
- 4 个 rank 日志未见 `Traceback` / `RuntimeError` / `OOM` / `Received sigquit` / `Empty reply` / `Scheduler hit an exception`。
- rank0 收尾 decode 日志中 `#swa token` 回到 0。

本地已下载 profile：

```text
profiles/deviceidle_kvslack4_active_224459/profile_deviceidle_kvslack4_active_224459/decode6_after_decode32_kvslack4/plugins/profile/2026_06_05_14_52_18/perf-16-0.trace.json.gz
```

轻量 xprof 观察：

- `jit_broadcast_in_dim -> jit_fused_greedy_verify` gap 前两轮约 `7-31 us`，最后一轮约 `3.1 ms`，最后一轮更像 profile/请求收尾。
- `same_batch_chain_peek_device_reserved_suffix` 只有 2 次，说明 same-batch chain/cache 命中仍有回退。
- rank0 日志中仍有大量 `same_batch_chain_peek_skip reason=device_reserve/reserve`，例如 `min_slack=-1..-4`。

新增/更新待办：

- 当前实现引入/暴露新的 same-batch chain cache miss / reserve miss。先记录，暂不修；后续需要重新分析 reserve frontier 与 chain preview 的一致性。
- accept-rate 回退已用相同 GSM8K 8 条样本 x4 的 32 并发请求确认：
  - baseline：`sgl-project/epic/mtp-phase2-performance` commit `d52a68b350b27ced3c3ed43a597032cef63b7387`，RUN_ID `accept_d52_nooverlap_232826`。该提交启动 spec decode 必须加 `--disable-overlap-schedule`，所以只作为接受率 baseline，不作为 overlap/bubble 性能 baseline。结果 `ok=32 bad=0`，all decode accept-ratio mean `0.8912`，accept-len mean `3.5629`；`#running-req: 32` accept-ratio mean `0.8308`，accept-len mean `3.3181`。
  - page-align 前当前 overlap：RUN_ID `accept_current_overlap_233823`，`ok=32 bad=0`，all decode accept-ratio mean `0.3724`，accept-len mean `1.4882`。该 run 没有 `#running-req: 32` 稳态区间，主区间在 `29/28/26/23/22/20`。
  - page-align 后当前 overlap：RUN_ID `accept_current_pagealign_234810`，`ok=32 bad=0`，all decode accept-ratio mean `0.3593`，accept-len mean `1.4367`；`#running-req: 32` accept-ratio mean `0.3063`，accept-len mean `1.2229`。
  - 当前 no-overlap crash 诊断：RUN_ID `accept_current_nooverlap_diag_235736` / `accept_current_nooverlap_hostfix_000541` 均在首批请求后 crash，栈在 `ScheduleBatch._split_spec_info_per_rank -> flat._ensure_host()`，先后暴露了 sharded slice 和 non-addressable global `jax.Array` host 化问题。
  - 当前 no-overlap crash 修复后：RUN_ID `accept_current_nooverlap_allgather_002847`，启动命令使用 `/opt/venv/bin/python` + `PYTHONPATH=/tmp/sglang-jax/python`，并带 `--disable-overlap-schedule`。GSM8K 8 条样本 x4 的 32 并发结果 `ok=32 bad=0`，无 `Traceback/RuntimeError/OOM/Received sigquit/ShardingTypeError`，server 请求后仍存活。
    - all decode accept-ratio mean `0.2899`，accept-len mean `1.1622`。
    - `#running-req: 32` accept-ratio mean `0.3400`，accept-len mean `1.3550`。
  - 结论：page allocator reserve ceil-align 能修 reserve slack/cache miss 方向的问题；`EagleDraftInput._ensure_host()` 需要同时 hostify scheduler metadata 字段，并对 non-fully-addressable JAX arrays 使用 `process_allgather(..., tiled=True)`，可修 no-overlap crash。但接受率在 no-overlap 下仍显著低于 d52 baseline，因此接受率回退不只是 overlap fastpath 问题，而是在当前 shared fused split/no-overlap 路径里也存在，需要作为独立高优先级问题继续查。

已完成/已验证：

- 不新增 spec-overlap flag；沿用现有 spec/server args。
- `--decode-log-interval=1` 用于逐 decode batch 打日志。
- verify/cache miss 级别的大 bubble 已经消掉；最新 profile 没有之前那种 25s cache miss。
- 原始约 `30 ms` batch 间 bubble 已经缩小到 `3-4 ms` verify-to-verify gap。
- 剩余 gap 中约 `1.8-1.9 ms` 是真实 device work，不应算 idle。
- 当前主要 idle 是 PhaseB 后下一轮 verify enqueue 的 host/JAX submit 边界。
- 当前最好 profile 已下载并解压，路径见上。

注意：

- 之前出现过 `token_to_kv_pool_allocator memory leak detected`，这是单独问题，当前 bubble 阶段先记录、最后修。
- 32 个 `max_tokens=512` 长请求跑到后段也可能触发 SWA token 接近耗尽 / Prefill OOM，这不等价于 overlap 失败。

## 剩余问题

1. 剩余 `3-4 ms` verify-to-verify gap 没完全消除。

   其中真正 idle tail 约 `1.16-2.06 ms`，主要来自 PhaseB 后下一轮 verify 的 host-side enqueue 仍然在 critical path。

2. `get_eagle_forward_metadata` / `_forward_batch_init_new_preserve_device` / `PjitFunction(fused_greedy_verify)` 仍然发生在下一轮 verify launch 前。

   这些动作需要尽量提前到 PhaseB device work 运行期间，或者变成 device dependency relay。

3. KV allocator leak 尚未修。

   不要在 bubble-focused profile 阶段把 leak 作为 overlap 失败；但最终必须修。

4. 当前 follow-up 版本存在 same-batch chain cache miss / reserve miss 回退。

   证据是 rank0 日志中大量 `same_batch_chain_peek_skip reason=device_reserve/reserve`，以及 profile 中 `same_batch_chain_peek_device_reserved_suffix` 次数偏低。当前先优先证明不 crash，后续再修性能。

5. accept-rate 已确认整体回退。

   相同 GSM8K 8 条样本 x4 的 32 并发请求下，d52 baseline `accept_d52_nooverlap_232826` all decode accept-ratio mean `0.8912`、`#running-req: 32` mean `0.8308`；当前 page-align 后 overlap `accept_current_pagealign_234810` all decode accept-ratio mean `0.3593`、`#running-req: 32` mean `0.3063`。该问题不是 page allocator reserve ceil-align 能解决的，需要单独对比 d52 的 draft/verify 输入、accept length 计算、topk/hidden state relay、以及 overlap 下 accepted token 写回。

6. 最终正确性验收还没做。

   开发阶段可以先不跑 evalscope；最终必须跑 GSM8K，并确认不回退。

## 建议的剩余实现计划

### Step 1: 稳定当前未提交 fused greedy 修复

目标：

- padding seq 的 accept length 为 0。
- `new_seq_lens` 使用 original seq lens 计算。
- CPU focused tests 通过。

验收：

```bash
PYTHONPATH=/tmp/sglang-jax/python JAX_PLATFORMS=cpu \
/opt/venv/bin/uv run --active pytest \
  sgl_jax/test/speculative/test_eagle_utils.py -q
```

### Step 2: 把 next verify candidate prebuild 前移

目标：

- 在拿到 PhaseA verify result 后，先 prebuild same-batch chained verify candidate。
- 不等 PhaseB Python result 完整 materialize 后才 build。
- PhaseB pending 只负责补齐 `topk_index` / `topk_p` / `verified_id` / `hidden_states` / `previous_token_list` 等真实 device fields。

预期收益：

- 吃掉 `_build_same_batch_spec_chain_candidate_batch` 的 `~0.3-0.4 ms`。

验收：

- CPU 单测覆盖 prebuild 在 PhaseB dispatch 前发生。
- 4-rank profile 的 gap 比当前 `[4.032, 3.855, 3.083, 3.931, 3.698, 3.069] ms` 有下降，且无 cache miss 大 stall。

### Step 3: 提前准备 verify metadata / ForwardBatch

目标：

- 对 same-batch chain，尽量在 PhaseA 之后、PhaseB device work 运行期间准备：
  - `get_eagle_forward_metadata`
  - `ForwardBatch` 中不依赖 PhaseB result 的 host/device fields
  - out_cache_loc / req_pool_indices / seq_lens 这类 stable metadata
- PhaseB 完成后只接入必要 device handles，再 submit next verify。

预期收益：

- 压缩 `get_eagle_forward_metadata ~0.46-0.66 ms`。
- 压缩 `_forward_batch_init_new_preserve_device ~0.40-0.48 ms`。

验收：

- 4-rank profile 里 `get_eagle_forward_metadata` / `_forward_batch_init_new_preserve_device` 不再处于 PhaseB 后的 idle critical tail，或者耗时明显下降。
- 稳态吞吐高于当前保守值 `~1.6k token/s`。

### Step 4: Device/FutureMap relay

目标：

- 不等 Python 完整 `padded_next_draft_input`，让下一轮 verify input 持有 PhaseB device future/dependency。
- CPU 先把 launch 结构搭好，真实数据依赖交给 JAX/PJRT 排序。

预期收益：

- 消掉最后 `1-2 ms` idle tail。

验收：

- xprof 中 `jit_fused_draft_extend -> next jit_fused_greedy_verify` 基本贴合。
- gap 中只剩真实 useful device work，没有明显 host submit idle。
- 吞吐有明显提升。

### Step 5: 修 KV allocator leak

目标：

- 修掉 `token_to_kv_pool_allocator memory leak detected`。
- 长请求不会在 profile 后段因为 allocator accounting 错误崩。

验收：

- 32 并发 `max_tokens=512` greedy 请求跑完整不触发 allocator leak。
- full/swa expected/available/evict/protected accounting 一致。

### Step 6: 最终验收

必须记录：

- 当前 commit / run id。
- 32 并发长 greedy 请求稳态吞吐。
- `accept-len` / `accept-ratio`。
- 5-step steady decode profile。
- GSM8K evalscope 结果。

最终标准：

- 吞吐明显提升。
- 接受率不下降，至少在相同 deterministic 请求下不下降。
- GSM8K 不下降。
- 无 cache miss 大 stall。
- 无 allocator leak / OOM 尾部崩溃。

## Evalscope 最终命令

开发阶段可以先跳过；最终跑 GSM8K：

```bash
RUN_ID=$(cat /tmp/current_specdecode_route_run_id)
kubectl exec perf-16-0-jgb5c -c jax-tpu -- bash -lc "
cd /tmp/sglang-jax/python &&
/opt/venv/bin/evalscope eval \
  --model /data/pc \
  --api-url http://127.0.0.1:30271/v1/chat/completions \
  --api-key EMPTY \
  --eval-type service \
  --datasets gsm8k \
  --dataset-args '{\"gsm8k\":{\"dataset_id\":\"/tmp/gsm8k_eval_data\"}}' \
  --eval-batch-size 64 \
  --generation-config '{\"temperature\":0,\"max_tokens\":1024}' \
  2>&1 | tee /tmp/evalscope_gsm8k_${RUN_ID}.log
"
```

## 给新线程的第一步建议

1. 从当前 worktree 开始，不要从主仓库 main 重新找。
2. 先跑 `git status --short`，确认仍是上面 3 个 modified tracked 文件。
3. 同步当前改动到 pod，跑 `test_eagle_utils.py` CPU 测试。
4. 重启 4-rank server，不带 `--disable-overlap-schedule`。
5. 用 32 并发 curl 请求复现当前稳态吞吐和 profile。
6. 如果 profile 仍是 `3-4 ms` gap，优先做 Step 2 / Step 3。
