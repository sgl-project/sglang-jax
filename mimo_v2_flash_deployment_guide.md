# MiMo-V2-Flash 部署与调试指南

## 模型信息

- 模型: `MiMo-V2-Flash` (小米 MoE 推理模型)
- 架构: 256 experts, FP8 量化, block_size=128
- 特殊点: head_dim=192 (Q/K), v_head_dim=128 (V), 每层 KV head 数量不同
- HuggingFace: `XiaomiMiMo/MiMo-V2-Flash`

## 硬件要求

- TPU v6e-16 (4 nodes × 4 chips = 16 chips)
- TP=16, EP=16, MoE backend: `epmoe`
- Spot 价格约 $9.14/h (us-central1-b)

## 1. 创建集群

使用 SkyPilot 启动 TPU v6e-16 集群:

```bash
# 使用 spot 实例，idle 120 分钟自动关闭
sky launch scripts/tpu_v6e16_mimo.sky.yaml -n <cluster-name> -y --use-spot -i 120
```

集群启动后自动执行:
1. Clone sgl-jax 仓库到 `~/sky_workdir/sgl-jax`
2. 创建 Python 3.12 venv 并安装依赖 (`uv pip install -e "python[all,multimodal]"`)

## 2. 同步代码到集群

集群创建后默认拉取 main 分支。如需部署开发分支的代码:

```bash
# 方法1: rsync 本地代码到所有节点
CLUSTER=<cluster-name>
NODES=$(sky status --ip $CLUSTER)
for ip in $NODES; do
  rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
    python/sgl_jax/ \
    gcpuser@$ip:~/sky_workdir/sgl-jax/python/sgl_jax/
done

# 方法2: 使用 sky rsync (仅同步到 head node)
sky rsync $CLUSTER python/sgl_jax/ ~/sky_workdir/sgl-jax/python/sgl_jax/

# 方法3: SSH 到集群后 git checkout
ssh $CLUSTER "cd ~/sky_workdir/sgl-jax && git fetch origin feat/mimo-v2-flash && git checkout feat/mimo-v2-flash"
```

## 3. 下载模型

SSH 到 head node 后下载模型到本地磁盘:

```bash
ssh <cluster-name>

# 安装 huggingface-cli (如果没有)
pip install huggingface_hub

# 下载到 /models 目录 (所有节点共享 NFS 或各节点分别下载)
sudo mkdir -p /models
sudo chmod 777 /models
huggingface-cli download XiaomiMiMo/MiMo-V2-Flash --local-dir /models/MiMo-V2-Flash

# 如果需要加速，可以用 RAM disk (v6e-16 有足够内存)
sudo mkdir -p /mnt/model_ram
sudo mount -t tmpfs -o size=80G tmpfs /mnt/model_ram
cp -r /models/MiMo-V2-Flash /mnt/model_ram/MiMo-V2-Flash
```

注意: v6e-16 是 4 节点集群，需要在每个 worker 节点上都有模型。如果没有共享存储，需要在每个节点上分别下载或复制。

## 4. 启动服务

### 4.1 基本启动 (所有节点)

在 head node 上创建启动脚本，然后通过 SSH 分发到各 worker:

```bash
# /tmp/start_server.sh — 单节点启动脚本
cat > /tmp/start_server.sh << 'EOF'
#!/bin/bash
set -euo pipefail
cd ~/sky_workdir/sgl-jax
source .venv/bin/activate
export PYTHONPATH=~/sky_workdir/sgl-jax/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false

HEAD_IP="$1"
NODE_RANK="$2"
NUM_NODES="${3:-4}"
MODEL_PATH="${4:-/models/MiMo-V2-Flash}"

pkill -f "sgl_jax.launch_server" || true
sleep 2

python -u -m sgl_jax.launch_server \
  --model-path $MODEL_PATH \
  --trust-remote-code \
  --tp-size 16 --ep-size 16 \
  --moe-backend epmoe \
  --nnodes $NUM_NODES --node-rank $NODE_RANK \
  --dist-init-addr ${HEAD_IP}:10011 \
  --host 127.0.0.1 --port 30271 \
  --context-length 1024 \
  --max-total-tokens 2048 \
  --max-prefill-tokens 256 \
  --mem-fraction-static 0.7 \
  --disable-precompile --skip-server-warmup \
  --log-level info \
  > ~/server.log 2>&1 &

echo "Server started on node $NODE_RANK, log: ~/server.log"
EOF
chmod +x /tmp/start_server.sh
```

```bash
# /tmp/start_all.sh — 启动所有节点
cat > /tmp/start_all.sh << 'EOF'
#!/bin/bash
HEAD_IP=$(hostname -I | awk '{print $1}')
WORKER_IPS=("10.x.x.x" "10.x.x.x" "10.x.x.x")  # 替换为实际 worker IP
SSH_OPTS="-F /dev/null -o StrictHostKeyChecking=no -i ~/.ssh/sky-cluster-key"

# 启动 head node (rank 0)
bash /tmp/start_server.sh $HEAD_IP 0

# 启动 worker nodes (rank 1, 2, 3)
for i in "${!WORKER_IPS[@]}"; do
  RANK=$((i + 1))
  scp $SSH_OPTS /tmp/start_server.sh ${WORKER_IPS[$i]}:/tmp/
  ssh $SSH_OPTS ${WORKER_IPS[$i]} "bash /tmp/start_server.sh $HEAD_IP $RANK" &
done
wait
echo "All nodes started. Head: $HEAD_IP"
EOF
chmod +x /tmp/start_all.sh
```

```bash
# /tmp/kill_all.sh — 停止所有节点
cat > /tmp/kill_all.sh << 'EOF'
#!/bin/bash
WORKER_IPS=("10.x.x.x" "10.x.x.x" "10.x.x.x")
SSH_OPTS="-F /dev/null -o StrictHostKeyChecking=no -i ~/.ssh/sky-cluster-key"

pkill -f "sgl_jax.launch_server" || true
for ip in "${WORKER_IPS[@]}"; do
  ssh $SSH_OPTS $ip "pkill -f sgl_jax.launch_server" || true
done
echo "All nodes stopped."
EOF
chmod +x /tmp/kill_all.sh
```

### 4.2 开启 Precompile 的启动方式

Precompile 会预编译常用 batch size/seq len 组合的 XLA kernel，减少运行时编译延迟:

```bash
# 与基本启动相同，但去掉 --disable-precompile 和 --skip-server-warmup
python -u -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash \
  --trust-remote-code \
  --tp-size 16 --ep-size 16 \
  --moe-backend epmoe \
  --nnodes 4 --node-rank $NODE_RANK \
  --dist-init-addr ${HEAD_IP}:10011 \
  --host 127.0.0.1 --port 30271 \
  --context-length 1024 \
  --max-total-tokens 2048 \
  --max-prefill-tokens 256 \
  --mem-fraction-static 0.7 \
  --log-level info \
  > ~/server.log 2>&1 &
```

注意: precompile 启动较慢 (可能需要 10-20 分钟)，但运行时性能更好。

### 4.3 验证服务健康

```bash
# 检查服务是否就绪
curl http://127.0.0.1:30271/get_model_info
curl http://127.0.0.1:30271/v1/models

# 简单推理测试
curl http://127.0.0.1:30271/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiMo-V2-Flash",
    "messages": [{"role": "user", "content": "25*37=?"}],
    "temperature": 0.3,
    "max_tokens": 128
  }'
```

## 5. 运行评测

### 5.1 Benchmark (bench_one_batch_server)

```bash
cd ~/sky_workdir/sgl-jax
source .venv/bin/activate
export PYTHONPATH=~/sky_workdir/sgl-jax/python

# 基本 benchmark
python -m sgl_jax.bench_one_batch_server \
  --model None \
  --base-url http://127.0.0.1:30271 \
  --batch-size 1 16 64 \
  --input-len 128 512 \
  --output-len 64

# 带报告输出
python -m sgl_jax.bench_one_batch_server \
  --model None \
  --base-url http://127.0.0.1:30271 \
  --batch-size 1 16 64 \
  --input-len 128 \
  --output-len 64 \
  --show-report
```

### 5.2 Quality Eval (GSM8K / MGSM / MMLU)

需要先安装依赖:

```bash
pip install openai httpx numpy pandas
export OPENAI_API_KEY=EMPTY
```

```bash
# GSM8K + MGSM 中文 (推荐 max_tokens=768 以给推理模型足够 CoT 空间)
python .tmp_mimo_remote_eval.py \
  --base-url http://127.0.0.1:30271/v1 \
  --evals gsm8k mgsm_zh \
  --gsm8k-examples 30 --gsm8k-max-tokens 768 \
  --mgsm-examples 30 --mgsm-max-tokens 768 \
  --threads 16

# MMLU (max_tokens 较小即可)
python .tmp_mimo_remote_eval.py \
  --base-url http://127.0.0.1:30271/v1 \
  --evals mmlu \
  --mmlu-examples 30 --mmlu-max-tokens 256 \
  --threads 16

# 中文 QA Suite (6 cases)
python .tmp_mimo_zh_qa_suite.py
```

### 5.3 Sampling 参数

| Eval 类型 | temperature | top_p | max_tokens |
|----------|-------------|-------|------------|
| 数学 (GSM8K/MGSM) | 0.8 | 0.95 | 768 |
| QA/多选 (MMLU) | 0.3 | 0.95 | 256 |
| 中文 QA | 0.3 | 0.95 | 128 |

### 5.4 System Prompt

英文:
```
You are MiMo, an AI assistant developed by Xiaomi.
Today's date: 2026-03-25 Wednesday. Your knowledge cutoff date is December 2024.
```

中文:
```
你是MiMo（中文名称也是MiMo），是小米公司研发的AI智能助手。
今天的日期：2026-03-25 星期三，你的知识截止日期是2024年12月。
```

## 6. 已知问题与注意事项

### 6.1 input_len=512 TTFT 异常

`input_len=512` 下 TTFT 异常高 (46-145s)，疑为 precompile 缺少对应 shape 的编译缓存导致运行时重编译。`input_len=128` 正常 (0.04-10.56s)。

### 6.2 k/v_proj 无法使用 blockwise kernel

TP=16 后 k_proj per-device n_out=192, v_proj=128，小于 TPU MXU 最小 tile 宽度 256，走 dequant fallback。未来可考虑:
- 在 init 阶段预 dequant 为 bf16 (省掉每次 forward 的 scale expand)
- DP+TP Hybrid: Attention DP=4 × TP=4 使 per-device n_out 足够大

### 6.3 MMLU 分数偏低

主要因 `context-length=1024` 限制了推理模型 chain-of-thought 输出，不代表模型能力问题。增大 context-length 需要更多 KV cache 内存。

### 6.4 SSH 配置

集群 head node 的 SSH config 可能有格式问题，使用以下方式连接 worker:
```bash
ssh -F /dev/null -o StrictHostKeyChecking=no -i ~/.ssh/sky-cluster-key <worker-ip>
```

## 7. 关闭集群

```bash
# 停止集群 (保留磁盘，可 sky start 恢复)
sky stop <cluster-name>

# 彻底销毁集群
sky down <cluster-name>
```

## 8. 基线数据参考

详见:
- `mimo_v2_flash_eval_golden_2026-03-25.md` — 初始 golden baseline
- `mimo_v2_flash_eval_baseline_2026-03-27.md` — block-quant 修复后 baseline
- `mimo_v2_flash_blockwise_workaround.md` — block-quant 问题详细记录
