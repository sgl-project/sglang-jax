# Spec Decode Scheduler Overlap 交接文档

更新时间：2026-06-05 16:16 CST

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

4. 最终正确性验收还没做。

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
