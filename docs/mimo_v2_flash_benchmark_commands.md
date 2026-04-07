# MiMo-V2-Flash Benchmark & Eval 命令手册

TPU v6e-16 (4 pods), TP=16, EP=16

## 集群信息

```
KUBECTL="/opt/homebrew/share/google-cloud-sdk/bin/kubectl"
PODS=("mimo-v6e16-0-5c794" "mimo-v6e16-1-nn5v7" "mimo-v6e16-2-p7zcz" "mimo-v6e16-3-w88cs")
PYTHON="/workspace/sgl-jax/.venv/bin/python"
```

---

## 1. Server 启动

### 基础配置 (context-length=16384)

每个 pod 上的 `/tmp/start_server.sh`：

```bash
#!/bin/bash
set -euo pipefail
export PATH="/root/.local/bin:$PATH"

RANK="${1:?Usage: start_server.sh <rank>}"

cd /workspace/sgl-jax
nohup .venv/bin/python -u -m sgl_jax.launch_server \
    --model-path /models/MiMo-V2-Flash \
    --trust-remote-code \
    --tp-size 16 --ep-size 16 \
    --moe-backend epmoe \
    --nnodes 4 --node-rank "$RANK" \
    --dist-init-addr mimo-v6e16-0.mimo-v6e16-svc:10011 \
    --host 0.0.0.0 --port 30271 \
    --page-size 128 \
    --context-length 16384 \
    --disable-radix-cache \
    --chunked-prefill-size 16384 \
    --dtype bfloat16 \
    --mem-fraction-static 0.80 \
    --swa-full-tokens-ratio 0.15 \
    --disable-precompile --skip-server-warmup \
    --log-level info \
    > /tmp/server.log 2>&1 &
disown
echo "Server started with rank=$RANK, PID=$!"
```

### 启动流程

```bash
# 1. 杀掉所有 pod 的 server
for pod in "${PODS[@]}"; do
    $KUBECTL exec "$pod" -- bash -c 'pkill -9 -f "launch_server" 2>/dev/null' || true
done
sleep 5

# 2. 先启动 worker，最后启动 head node
$KUBECTL exec mimo-v6e16-3-w88cs -- bash /tmp/start_server.sh 3
$KUBECTL exec mimo-v6e16-2-p7zcz -- bash /tmp/start_server.sh 2
$KUBECTL exec mimo-v6e16-1-nn5v7 -- bash /tmp/start_server.sh 1
$KUBECTL exec mimo-v6e16-0-5c794 -- bash /tmp/start_server.sh 0

# 3. 等待 ready（约 50s）
$KUBECTL exec mimo-v6e16-0-5c794 -- \
    curl -s http://127.0.0.1:30271/get_server_info | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'status={d[\"status\"]}, context_length={d[\"context_length\"]}, max_total_tokens={d[\"max_total_num_tokens\"]}')
"
```

### 不同 context-length 的用途

| context-length | 用途 | 备注 |
|:-:|:-:|:-:|
| 16384 | 4K benchmark | 默认配置 |
| 32768 | 16K benchmark (input=16384+output=1024) | max_running 会减少 |
| 65536 | MMLU-Pro eval (max_tokens=32000) | max_running 进一步减少 |

修改方法：改 `start_server.sh` 中的 `--context-length` 值，重启所有 pod。

---

## 2. Benchmark: bench_serving (E2E)

所有 bench_serving 命令在 **pod-0** 上执行。

### 2.1 4K Decode (rate=3, 64 prompts) — 标准 E2E benchmark

```bash
# server: context-length=16384
$PYTHON -m sgl_jax.bench_serving --backend sgl-jax \
    --host 127.0.0.1 --port 30271 \
    --dataset-name random --random-input-len 4096 --random-output-len 1024 \
    --random-range-ratio 1.0 --num-prompts 64 --request-rate 3 --flush-cache
```

### 2.2 4K Prefill (rate=3, 64 prompts)

```bash
# server: context-length=16384
$PYTHON -m sgl_jax.bench_serving --backend sgl-jax \
    --host 127.0.0.1 --port 30271 \
    --dataset-name random --random-input-len 4096 --random-output-len 1 \
    --random-range-ratio 1.0 --num-prompts 64 --request-rate 3 --flush-cache
```

### 2.3 单 batch size 最大吞吐（rate=1000 瞬发）

```bash
# bs=32, 4K decode
$PYTHON -m sgl_jax.bench_serving --backend sgl-jax \
    --host 127.0.0.1 --port 30271 \
    --dataset-name random --random-input-len 4096 --random-output-len 1024 \
    --random-range-ratio 1.0 --num-prompts 32 --request-rate 1000 \
    --flush-cache --seed 42
```

---

## 3. Benchmark: mimo_bench_suite.sh (全面扫描)

自动化脚本，对多个 batch size 做 JIT warmup 后逐个跑 benchmark。

### 3.1 4K 场景 (Scenario 1 & 2)

```bash
# server: context-length=16384
cd /workspace/sgl-jax
nohup bash benchmark/mimo_bench_suite.sh 4k > /tmp/bench_4k.log 2>&1 &

# 场景:
#   1: input=4096, output=1   (prefill), bs=1,4,8,16,24,32,48,64,96,128
#   2: input=4096, output=1024 (decode),  bs=1,4,8,16,24,32,48,64,96,128
```

### 3.2 16K 场景 (Scenario 3 & 4)

```bash
# server: context-length=32768 (必须 >= 16384+1024=17408)
# 使用 4K 的同一个 RESULT_DIR 合并结果
RESULT_DIR=/tmp/mimo_bench_XXXXXXXX nohup bash benchmark/mimo_bench_suite.sh 16k > /tmp/bench_16k.log 2>&1 &

# 场景:
#   3: input=16384, output=1    (prefill), bs=1,4,8,16,24,32
#   4: input=16384, output=1024 (decode),  bs=1,4,8,16,24,32
```

### 3.3 生成报告

```bash
RESULT_DIR=/tmp/mimo_bench_XXXXXXXX bash benchmark/mimo_bench_suite.sh report
```

### 3.4 查看进度

```bash
# 检查是否还在运行
pgrep -af mimo_bench_suite

# 查看最新日志
tail -20 /tmp/bench_4k.log
tail -20 /tmp/bench_16k.log
```

---

## 4. Benchmark: decode_bench.py (Decode-Only 吞吐)

短 input，隔离测 decode 性能，评估 PD 分离的理论收益。

```bash
# server: context-length=16384
$PYTHON benchmark/decode_bench.py \
    --host 127.0.0.1 --port 30271 \
    --batch-sizes 1,4,8,16,24,32 \
    --input-len 128 --output-len 2048
```

---

## 5. MMLU-Pro Eval

### 5.1 前提

- eval 脚本: `/tmp/eval_mmlu_pro_v4.py`
- 数据文件: `/tmp/mmlu_pro_500.json` (493 题)
- 如果 pod 重建过需要重新上传:

```bash
$KUBECTL cp /tmp/eval_mmlu_pro_v4.py mimo-v6e16-0-5c794:/tmp/eval_mmlu_pro_v4.py
$KUBECTL cp /tmp/mmlu_pro_500.json  mimo-v6e16-0-5c794:/tmp/mmlu_pro_500.json
```

### 5.2 启动

```bash
# server: context-length=65536 (给 thinking 留空间; OOM 则降到 32768)
cd /workspace/sgl-jax
PYTHONUNBUFFERED=1 nohup .venv/bin/python -u \
    /tmp/eval_mmlu_pro_v4.py /tmp/mmlu_pro_500.json 0 32000 2 0.6 \
    > /tmp/mmlu_pro_eval_$(date +%Y%m%d).log 2>&1 &
```

参数说明: `eval_mmlu_pro_v4.py <data_file> <start_idx> <max_tokens> <concurrency> <temperature>`

| 参数 | 值 | 说明 |
|:----:|:--:|:-----|
| data_file | /tmp/mmlu_pro_500.json | 493 题 MMLU-Pro 子集 |
| start_idx | 0 | 从第几题开始（用于断点续跑） |
| max_tokens | 32000 | 给 thinking 留足空间 |
| concurrency | 2 | 低并发避免 KV cache 不够 |
| temperature | 0.6 | 比 chat 的 0.8 低，减少推理循环 |

### 5.3 查看进度

```bash
# 最新日志
tail -20 /tmp/mmlu_pro_eval_20260402.log

# 统计准确率
python3 -c "
import re
lines = [l for l in open('/tmp/mmlu_pro_eval_20260402.log') if re.match(r'\s+\[\d+\]', l)]
ok = sum(1 for l in lines if ' OK ' in l)
trunc = sum(1 for l in lines if 'TRUNCATED' in l)
total = len(lines)
nt = total - trunc
print(f'Progress: {total}/493 ({total*100//493}%)')
print(f'Accuracy: {ok/total*100:.1f}% ({ok}/{total})')
print(f'Non-trunc accuracy: {ok/nt*100:.1f}% ({ok}/{nt})')
print(f'Truncation rate: {trunc/total*100:.1f}% ({trunc}/{total})')
"
```

---

## 6. 通用运维命令

### 代码部署

```bash
# 需要同步的文件列表
FILES=(
    "python/sgl_jax/srt/managers/schedule_policy.py"
    "python/sgl_jax/srt/layers/linear.py"
    "python/sgl_jax/srt/models/mimo_v2_flash.py"
    "benchmark/mimo_bench_suite.sh"
    "benchmark/decode_bench.py"
)

# 同步到所有 pod
for pod in "${PODS[@]}"; do
    for f in "${FILES[@]}"; do
        $KUBECTL cp "/Users/jiongxuan/workspace/sgl-jax/$f" "$pod:/workspace/sgl-jax/$f"
    done
done

# 验证
for pod in "${PODS[@]}"; do
    echo -n "$pod: "
    $KUBECTL exec "$pod" -- grep -c "SWA pool should not limit" \
        /workspace/sgl-jax/python/sgl_jax/srt/managers/schedule_policy.py
done
```

### Server 健康检查

```bash
$KUBECTL exec mimo-v6e16-0-5c794 -- \
    curl -s http://127.0.0.1:30271/get_server_info | python3 -m json.tool
```

### 查看 server 日志

```bash
$KUBECTL exec mimo-v6e16-0-5c794 -- tail -50 /tmp/server.log
```

### 强制杀掉所有 server

```bash
for pod in "${PODS[@]}"; do
    $KUBECTL exec "$pod" -- bash -c 'pkill -9 -f "launch_server" 2>/dev/null' || true
done
```
