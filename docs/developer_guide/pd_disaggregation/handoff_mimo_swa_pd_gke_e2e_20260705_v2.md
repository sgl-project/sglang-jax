# MiMo-V2-Flash SWA PD GKE E2E 交接文档 v2

快照时间：2026-07-05 08:49:33 CST。GKE pod 内日志时间通常是 UTC。

本文档是 `handoff_mimo_swa_pd_gke_e2e_20260705.md` 的 v2 交接版本。旧文档记录的是更早的 bring-up 状态，里面的 HEAD、job 状态和测试进展已经过时；后续 agent 请优先以本文档为准。

## 目标

当前 epic 目标：

- 让 MiMo-V2-Flash 在 GKE 上支持 PD disaggregation + SWA KV + Raiden data plane。
- 功能正确优先；当前是 epic 分支，不需要为不相关模块做保守兼容。
- 功能正确后，围绕不同 batch size / concurrency 优化 TTFT、IOT/ITL/TPOT 和吞吐。
- 重要稳定节点需要 commit & push。

当前结论：

- 功能正确性已经基本跑通。
- 主要性能问题不是吞吐不够，而是高并发下 TTFT 和 P99 TTFT 很差，IOT/ITL 也随并发明显恶化。
- 下一阶段不建议继续盲跑大范围 benchmark；应该先做短输出、小矩阵、细粒度时间戳 profiling，定位 latency 瓶颈。

## 当前代码状态

仓库：

```bash
cd /Users/jiongxuan/workspace/sglang-jax
git branch --show-current
# epic/mimo-pd-disggragation

git rev-parse HEAD
# dfe33c59de3229709a62bd0aea0bfc4d18f5f0c0
```

远端分支已 push 到同一个 HEAD：

```text
dfe33c59d fix(pd/swa): allocate decode pages on request dp rank
```

最近关键 commits：

```text
dfe33c59d fix(pd/swa): allocate decode pages on request dp rank
c2bb48312 chore(pd/gke): add request time stats toggle
2188e9f07 fix(pd/swa): select raiden endpoint shards by dp rank
612cab3c9 fix(pd/swa): globalize raiden page ids by dp segment
78f3b007a fix(pd/swa): keep raiden page ids local under dp
ecdfbc34b fix(pd/swa): namespace raiden page ids by dp rank
```

注意：`612cab3c9` 在历史里，但不是当前最终方案；它曾导致 prefill 侧 `Copy range exceeds source device buffer size`。当前 HEAD `dfe33c59d` 已经改成 decode 侧按 request DP rank 分配/释放 page，并通过后续 GKE 验证。

当前 HEAD 修改文件：

```text
python/sgl_jax/srt/disaggregation/decode.py
python/sgl_jax/test/test_pd_swa_basic.py
```

核心实现点：

- `python/sgl_jax/srt/disaggregation/decode.py`
  - 增加 `_req_dp_rank(req)` 辅助逻辑。
  - `_admit_decode_prealloc` 在 `available_size` 和 `alloc` 时使用 request 的 DP rank。
  - decode KV indices 的 release 路径也传入 request DP rank，避免跨 DP rank 释放/分配混乱。
- `python/sgl_jax/test/test_pd_swa_basic.py`
  - 新增 `TestDecodeDPAllocation::test_admit_decode_prealloc_allocates_request_dp_rank`
  - 新增 `TestDecodeDPAllocation::test_admit_one_raiden_builds_dp_rank_swa_local_pages`

本地工作区是脏的，有大量既有未跟踪 docs/scripts。不要 `git reset --hard`，也不要批量清理。后续提交时只 stage 自己本轮相关文件。

当前 `git status --short --branch` 里可见：

```text
## epic/mimo-pd-disggragation
 M scripts/disaggregation/gke/pd_singlehost_eval_job.yaml
?? docs/developer_guide/pd_disaggregation/handoff_mimo_swa_pd_gke_e2e_20260705.md
?? docs/developer_guide/pd_disaggregation/...
?? scripts/disaggregation/gke/pd_bench_serving_job.yaml
...
```

## 已验证的本地测试

CPU 单测已在当前修复上通过：

```bash
cd /Users/jiongxuan/workspace/sglang-jax

USE_DEVICE_TYPE=cpu .venv/bin/python -m pytest \
  python/sgl_jax/test/test_pd_swa_basic.py::TestDecodeDPAllocation -q
# 2 passed

USE_DEVICE_TYPE=cpu .venv/bin/python -m pytest \
  python/sgl_jax/test/test_pd_swa_basic.py -q
# 23 passed

USE_DEVICE_TYPE=cpu .venv/bin/python -m pytest \
  python/sgl_jax/test/mem_cache/test_swa_allocator.py -q
# 31 passed
```

如后续改动 PD router，也建议补跑：

```bash
USE_DEVICE_TYPE=cpu .venv/bin/python -m pytest \
  test/srt/disaggregation/test_pd_router.py -q
```

## GKE / 机器状态

本地使用 kubectl/gcloud 前先设置：

```bash
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
```

当前仍存在的 GKE job：

```text
job.batch/mimo-swa-pd-dp2-cap32   Running   0/2   age=68m
```

当前 pods：

```text
pod/mimo-swa-pd-dp2-cap32-0-rzrfb   Running   2/2   restarts=0   IP=10.125.144.25   node=gke-tpu-744dcbc9-xhpf
pod/mimo-swa-pd-dp2-cap32-1-5rm4v   Running   2/2   restarts=0   IP=10.125.131.26   node=gke-tpu-3f263395-s243
```

角色：

- index 0：bootstrap + prefill。
- index 1：decode + router + eval/bench driver。

当前 cap32 pod1 状态：

```text
decode health = 200
router health = 200
bench_serving 进程 = 无
error counts = 0
```

pod1 日志文件：

```text
/tmp/e2e_logs/decode_server.log
/tmp/e2e_logs/router.log
/tmp/e2e_logs/gsm8k_stdout.log
/tmp/e2e_logs/gsm8k_result.jsonl
/tmp/e2e_logs/gsm8k_outputs.txt
/tmp/e2e_logs/bench_16384_4096_cap32_c128.log
```

注意：CAP=32 的 c128 benchmark 被人为中断，只能作为 partial signal，不是完整 benchmark 结果。中断后已向 decode 发过一次：

```bash
curl -X POST http://localhost:10001/abort_request \
  -H "Content-Type: application/json" \
  -d '{"rid":"","abort_all":true}'
```

之后确认无 `sgl_jax.bench_serving` 进程，decode/router health 仍为 200。

如果需要节省资源，可以删除当前 CAP=32 job：

```bash
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
kubectl delete job mimo-swa-pd-dp2-cap32
kubectl delete svc mimo-swa-pd-dp2-cap32-headless-svc 2>/dev/null || true
```

本文档创建时没有删除该 job，以便后续 agent 还能查看现场日志。

## Correctness / Eval 结果

### CAP=16 基线 job

旧 job `mimo-swa-pd-dp2-fix` 已删除；benchmark summary 已保存到本地：

```text
/tmp/mimo-swa-pd-baseline-dfe33c59d/bench_16384_4096_c32.jsonl
/tmp/mimo-swa-pd-baseline-dfe33c59d/bench_16384_4096_c64.jsonl
/tmp/mimo-swa-pd-baseline-dfe33c59d/bench_16384_4096_c128.jsonl
```

GSM8K：

```text
q32/p32:  accuracy=0.938, invalid=0.000, latency=39.527s, output_throughput=579.249 tok/s
q128/p128: accuracy=0.898, invalid=0.000, latency=106.396s, output_throughput=935.195 tok/s
```

错误计数为 0，没有 Traceback、Copy range、RESOURCE_EXHAUSTED、receiver failure 或 5xx。

### CAP=32 A/B job

当前 job：`mimo-swa-pd-dp2-cap32`

GSM8K q32/p32 已通过：

```json
{"task":"gsm8k","backend":"sgl-jax","latency":39.558,"accuracy":0.875,"num_requests":32,"other":{"num_questions":32,"parallel":32}}
```

CAP=32 的 q32 correctness 可接受，但不比 CAP=16 更好：

- CAP=16 q32 output throughput: `579 tok/s`
- CAP=32 q32 output throughput: `562 tok/s`

## Benchmark 基线

基线为 CAP=16，random dataset，16k input / 4k output，3 waves：

```text
--random-input-len 16384
--random-output-len 4096
--num-prompts concurrency*3
--max-concurrency concurrency
--ignore_eos true
```

结果：

| concurrency | completed | duration | output throughput | peak output | mean TTFT | median TTFT | P99 TTFT | mean TPOT/ITL | P99 TPOT | P99 ITL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 96/96 | 382.80s | 1027.22 tok/s | 1344 tok/s | 13.19s | 9.45s | 43.80s | 27.04ms | 29.36ms | 56.19ms |
| 64 | 192/192 | 582.75s | 1349.51 tok/s | 1963 tok/s | 25.78s | 18.68s | 84.80s | 39.73ms | 45.60ms | 122.29ms |
| 128 | 384/384 | 913.25s | 1722.26 tok/s | 3048 tok/s | 53.13s | 39.27s | 170.68s | 59.14ms | 71.50ms | 214.48ms |

结论：

- 吞吐随并发上升，并不是当前最坏指标。
- 最差的是高并发 TTFT 和 P99 TTFT：c128 P99 TTFT 已到 `170.68s`。
- 第二差的是高并发 IOT/ITL/TPOT：c128 mean ITL `59.14ms`，P99 ITL `214.48ms`。

## 当前 latency 瓶颈线索

已有 `PD-TIME-STATS` 分解字段：

```text
bootstrap
prealloc_wait
kv_wait
total
```

观察到的模式：

- `kv_wait` 大体稳定在 `2-3s`。
- `prealloc_wait` 随并发明显变长。
- CAP=16 c128 后段日志中 `prealloc_wait` 约 `25-34s`，`kv_wait` 约 `2.1-2.6s`。
- CAP=32 partial c128 中出现大量 `prealloc_wait=135-167s`，后续也有 `50s -> 28s` 级别的 prealloc wait。
- CAP=32 partial c128 的 client 侧第一个完成请求约在 `296.91s` 后才出现。
- CAP=32 partial c128 decode batch 内部瞬时 gen throughput 可到 `~2.9k tok/s`，说明 raw decode 吞吐并不差；first-token 前的 admission/prealloc 排队更可疑。

CAP=32 partial 的关键日志片段：

```text
1/384 [04:56<31:35:17, 296.91s/it]
PD-TIME-STATS ... prealloc_wait=135238.4ms kv_wait=2474.2ms total=137712.6ms
PD-TIME-STATS ... prealloc_wait=167336.3ms kv_wait=2431.7ms total=169768.1ms
Decode batch ... #running-req: 128 ... gen throughput (token/s): ~2900
```

代码层面还有一个重要线索：PD disaggregation 当前强制关闭 overlap schedule。

相关代码：

```text
python/sgl_jax/srt/managers/scheduler.py
```

初始化处：

```python
self.enable_overlap = not server_args.disable_overlap_schedule
if server_args.disaggregation_mode != "null":
    logger.info("PD disaggregation mode enabled, disabling overlap schedule")
    self.enable_overlap = False
```

dispatch 处：

```python
if mode == "prefill":
    scheduler.event_loop_normal_disagg_prefill()
elif mode == "decode":
    scheduler.event_loop_normal_disagg_decode()
elif scheduler.enable_overlap:
    scheduler.event_loop_overlap()
else:
    scheduler.event_loop_normal()
```

因此，后续如果要优化 overlap，不能只删一行；需要 PD-aware overlap loop，至少要保证：

- decode 模式仍持续调用 `process_decode_queue()`。
- prealloc -> transfer -> ready 的状态推进不被 overlap result queue 破坏。
- abort/release/resource accounting 在 overlap 下仍正确。

## 不建议继续的方向

不建议下一步继续做完整 `c32/c64/c128 * 16k/4k * 多 CAP` 大矩阵试错，原因：

- 每轮成本太高。
- 当前最差指标已经很明确是 TTFT/P99 TTFT，而不是总吞吐。
- 只调 `disaggregation_max_inflight_transfers` 信号不佳：CAP=32 correctness 过了，但 c128 partial 的 prealloc wait 更糟。
- full benchmark 的最终 summary 太晚，不能高效定位 queue 卡在哪一层。

## 建议的下一步

推荐改成两阶段。

### 阶段 A：低成本定位

目标：证明 TTFT 主要卡在 router、decode waiting queue、prealloc queue、allocator capacity、transfer cap，还是 PD normal event loop。

建议 workload：

```text
input_len=16384
output_len=256 或 512
concurrency=16/32/64
num_prompts=concurrency
warmup=0
```

这类短输出可以快速得到 TTFT，不需要等待 4k decode 完整跑完。

建议补的时间戳：

- router 收到请求。
- router 发给 decode。
- decode tokenizer/scheduler intake。
- 请求进入 `disagg_prealloc_queue`。
- `_admit_decode_prealloc` 尝试 admission 的时间。
- admission 成功，进入 transfer queue。
- Raiden receiver success。
- `_enqueue_for_decode`。
- first token。

已有 `prealloc_entry -> transfer_entry -> first_token` 不足以判断：

- 请求是在 router 前排队，还是 decode intake 后排队。
- `_admit_decode_prealloc` 是被 `max_inflight` 卡住，还是被 allocator available/headroom 卡住。
- `event_loop_normal_disagg_decode` 是否因为正在长 decode batch 而没有及时 pump `process_decode_queue()`。

### 阶段 B：基于证据做一个优化

若阶段 A 证明瓶颈在 admission/prealloc：

- 优先加 admission debug counter，而不是直接调 CAP。
- 对 `_admit_decode_prealloc` 记录 defer reason：
  - `max_inflight`
  - `allocator_capacity`
  - `prefill_metadata_missing`
  - `alloc_none`
- 同时记录：
  - `n_running`
  - `n_transfer`
  - `admitted`
  - `available_size(dp_rank)`
  - `page_aligned`
  - `reserved`
  - `max_inflight`

若阶段 A 证明 normal PD event loop pump 不及时：

- 设计 PD-aware decode overlap。
- 先写 CPU/unit test 验证 dispatch 行为和 `process_decode_queue()` 调用约束。
- 再改 scheduler，不要直接启用普通 overlap loop。

## 复现命令

### 重新部署 CAP=16 correctness job

```bash
cd /Users/jiongxuan/workspace/sglang-jax
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

JOB_NAME=mimo-swa-pd-dp2-fix \
BRANCH=epic/mimo-pd-disggragation \
EPHEMERAL_STORAGE=38Gi \
DP_SIZE=2 \
CAP=16 \
SKIP_GCSFUSE_WARMUP=1 \
ENABLE_REQUEST_TIME_STATS_LOGGING=1 \
RUN_GSM8K=1 \
GSM_Q=32 \
GSM_PAR=32 \
GSM_MAXTOK=2048 \
GSM_MIN_ACC=0.80 \
RUN_LONG_BENCH=0 \
RUN_MMLU_PRO=0 \
scripts/disaggregation/gke/deploy_mimo_swa_pd_e2e.sh
```

### 重新部署 CAP=32 A/B job

```bash
cd /Users/jiongxuan/workspace/sglang-jax
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

JOB_NAME=mimo-swa-pd-dp2-cap32 \
BRANCH=epic/mimo-pd-disggragation \
EPHEMERAL_STORAGE=38Gi \
DP_SIZE=2 \
CAP=32 \
SKIP_GCSFUSE_WARMUP=1 \
ENABLE_REQUEST_TIME_STATS_LOGGING=1 \
RUN_GSM8K=1 \
GSM_Q=32 \
GSM_PAR=32 \
GSM_MAXTOK=2048 \
GSM_MIN_ACC=0.80 \
RUN_LONG_BENCH=0 \
RUN_MMLU_PRO=0 \
scripts/disaggregation/gke/deploy_mimo_swa_pd_e2e.sh
```

### 监控当前 job

```bash
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

kubectl get jobs,pods -o wide | rg "mimo-swa-pd|NAME"

kubectl exec mimo-swa-pd-dp2-cap32-1-5rm4v -c jax-tpu -- bash -lc '
set +e
printf "decode="; curl -sf -o /dev/null -w "%{http_code}" http://localhost:10001/health; printf "\n"
printf "router="; curl -sf -o /dev/null -w "%{http_code}" http://localhost:30000/health; printf "\n"
find /tmp/e2e_logs -maxdepth 1 -type f -printf "%f %s bytes\n" | sort
grep -cE "Traceback|SIGABRT|Copy range|RESOURCE_EXHAUSTED|failed_recving|receiver_terminal_failed| 5[0-9][0-9] " \
  /tmp/e2e_logs/decode_server.log \
  /tmp/e2e_logs/router.log \
  /tmp/e2e_logs/gsm8k_stdout.log \
  /tmp/e2e_logs/bench_*.log 2>/dev/null || true
'
```

### 跑 GSM8K q128

在 decode/router pod 内执行：

```bash
kubectl exec mimo-swa-pd-dp2-cap32-1-5rm4v -c jax-tpu -- bash -lc '
set -e
cd /tmp/sglang-jax
timeout 7200 /opt/venv/bin/python benchmark/gsm8k/bench_sglang_jax.py \
  --base-url http://localhost:30000 \
  --num-questions 128 \
  --num-shots 5 \
  --max-new-tokens 2048 \
  --parallel 128 \
  --enable-thinking \
  --tokenizer-path /models/MiMo-V2-Flash \
  --result-file /tmp/e2e_logs/gsm8k_q128_result.jsonl \
  --output-file /tmp/e2e_logs/gsm8k_q128_outputs.txt \
  2>&1 | tee /tmp/e2e_logs/gsm8k_q128_stdout.log
'
```

### 跑完整 long benchmark

仅在需要重新建立完整基线时跑。成本较高，不建议作为下一步定位手段。

```bash
kubectl exec mimo-swa-pd-dp2-cap32-1-5rm4v -c jax-tpu -- bash -lc '
set -e
cd /tmp/sglang-jax
for C in 32 64 128; do
  N=$((C * 3))
  timeout 21600 /opt/venv/bin/python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --base-url http://localhost:30000 \
    --model /models/MiMo-V2-Flash \
    --tokenizer /models/MiMo-V2-Flash \
    --dataset-name random \
    --random-input-len 16384 \
    --random-output-len 4096 \
    --random-range-ratio 1.0 \
    --num-prompts "$N" \
    --max-concurrency "$C" \
    --warmup-requests 0 \
    --extra-request-body "{\"sampling_params\":{\"temperature\":0.1,\"top_p\":0.95,\"max_new_tokens\":4096,\"ignore_eos\":true}}" \
    --output-file /tmp/e2e_logs/bench_16384_4096_c${C}.jsonl \
    2>&1 | tee /tmp/e2e_logs/bench_16384_4096_c${C}.log
done
'
```

### 推荐的短测 profiling 命令

这是下一阶段更推荐的命令形态：

```bash
kubectl exec mimo-swa-pd-dp2-cap32-1-5rm4v -c jax-tpu -- bash -lc '
set -e
cd /tmp/sglang-jax
for C in 16 32 64; do
  N=$C
  timeout 3600 /opt/venv/bin/python -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --base-url http://localhost:30000 \
    --model /models/MiMo-V2-Flash \
    --tokenizer /models/MiMo-V2-Flash \
    --dataset-name random \
    --random-input-len 16384 \
    --random-output-len 512 \
    --random-range-ratio 1.0 \
    --num-prompts "$N" \
    --max-concurrency "$C" \
    --warmup-requests 0 \
    --extra-request-body "{\"sampling_params\":{\"temperature\":0.1,\"top_p\":0.95,\"max_new_tokens\":512,\"ignore_eos\":true}}" \
    --output-file /tmp/e2e_logs/diag_16384_512_c${C}.jsonl \
    2>&1 | tee /tmp/e2e_logs/diag_16384_512_c${C}.log
done
'
```

抽取关键 benchmark 字段：

```bash
for f in /tmp/mimo-swa-pd-baseline-dfe33c59d/bench_16384_4096_c*.jsonl; do
  jq -r '{
    file: input_filename,
    max_concurrency,
    completed,
    duration,
    output_throughput,
    max_output_tokens_per_s,
    mean_ttft_ms,
    median_ttft_ms,
    p99_ttft_ms,
    mean_tpot_ms,
    p99_tpot_ms,
    mean_itl_ms,
    p95_itl_ms,
    p99_itl_ms
  } | @json' "$f"
done
```

抽取 PD time stats：

```bash
kubectl exec mimo-swa-pd-dp2-cap32-1-5rm4v -c jax-tpu -- bash -lc '
grep "PD-TIME-STATS" /tmp/e2e_logs/decode_server.log | tail -n 100
'
```

## 手动停止负载 / 清理现场

如果 bench client 还在跑：

```bash
kubectl exec mimo-swa-pd-dp2-cap32-1-5rm4v -c jax-tpu -- bash -lc '
set +e
pkill -f "sgl_jax.bench_serving" || true
ps -ef | grep -E "sgl_jax\\.bench_serving|bench_" | grep -v grep || true
'
```

如果需要 abort decode 侧所有请求：

```bash
kubectl exec mimo-swa-pd-dp2-cap32-1-5rm4v -c jax-tpu -- bash -lc '
curl -sS -X POST http://localhost:10001/abort_request \
  -H "Content-Type: application/json" \
  -d "{\"rid\":\"\",\"abort_all\":true}" \
  -w "\nHTTP=%{http_code}\n"
'
```

如果要删除 job：

```bash
kubectl delete job mimo-swa-pd-dp2-cap32
kubectl delete svc mimo-swa-pd-dp2-cap32-headless-svc 2>/dev/null || true
```

## 后续 agent 注意事项

- 不要把 CAP=32 partial c128 当完整 benchmark 结果；它只说明 CAP=32 没有明显改善 TTFT，且 prealloc wait 有恶化信号。
- 当前 job 的 `/tmp/e2e_logs/bench_16384_4096_cap32_c128.log` 是中断日志，没有 JSON summary。
- 后续若要对比性能，优先复用 CAP=16 完整基线文件：
  `/tmp/mimo-swa-pd-baseline-dfe33c59d/*.jsonl`
- 如果做代码改动，必须先写 CPU regression/unit test 或至少新增可复现 profiling harness；不要直接上大 GKE benchmark 验证猜想。
- 若进入 PD overlap 改造，先读完整 `event_loop_normal_disagg_decode`、`event_loop_overlap`、`process_decode_queue`、abort/release 路径，再动 scheduler。
- 当前功能正确性不要轻易破坏；每次性能改动后至少跑：
  - `test_pd_swa_basic.py`
  - `test_swa_allocator.py`
  - GSM8K q32
  - 一个短输出 TTFT profiling case

