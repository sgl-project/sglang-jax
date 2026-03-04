# MiMo-V2-Flash 复现命令

目标：快速在远端 TPU 上重现当前问题（XLA shard_map 导出断言/SIGABRT）。默认路径与参数基于 `~/sky_workdir/sgl-jax` 代码仓与 `/models/MiMo-V2-Flash` 模型目录。

## 1. 基础环境

```bash
ssh sgl-jax-tpu
cd ~/sky_workdir
source sgl-jax/.venv/bin/activate
```

## 2. 常规启动（JIT 正常）

> 适合复现当前的 shard_map 导出崩溃；内存占用约 28–30GB HBM。

```bash
MODEL_ID=/models/MiMo-V2-Flash
PORT=30000
LOG=server.log
rm -f $LOG

python -m sgl_jax.launch_server \
  --model-path $MODEL_ID \
  --tp-size 8 \
  --port $PORT \
  --host 0.0.0.0 \
  --mem-fraction-static 0.2 \
  --trust-remote-code \
  --context-length 2048 \
  --disable-precompile \
  > $LOG 2>&1 &
PID=$!

# 等待服务就绪
for i in {1..120}; do
  grep -q "Uvicorn running on" $LOG && break
  sleep 2
done

# 发送最小请求（可能在编译阶段崩溃）
REQ='{"model":"/models/MiMo-V2-Flash","messages":[{"role":"user","content":"Hello, give one short sentence joke about AI."}],"max_tokens":5,"temperature":0.7,"top_p":1.0,"logprobs":true,"top_logprobs":20}'
curl -s -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" -d "$REQ" | tee /tmp/logprob_response.json

# 收尾：收集日志并停止
tail -n 4000 $LOG > /tmp/server_tail_debug.log
kill $PID || true
```

预期现象：在首次 prefill 编译阶段触发 `ShardMapExportPass` 断言，进程 SIGABRT，`/tmp/logprob_response.json` 为空。

## 3. 禁用 JIT 的 OOM 复现实验（可选）

> 仅用于说明：`JAX_DISABLE_JIT=1` 会在 SWA KV buffer 创建时 OOM（需要 ~655MB，但 HBM 剩余不足）。

```bash
MODEL_ID=/models/MiMo-V2-Flash
LOG=server.log
rm -f $LOG

JAX_DISABLE_JIT=1 python -m sgl_jax.launch_server \
  --model-path $MODEL_ID \
  --tp-size 8 \
  --port 30000 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.2 \
  --trust-remote-code \
  --context-length 2048 \
  --disable-precompile \
  > $LOG 2>&1 &
PID=$!
sleep 30
tail -n 400 $LOG
kill $PID || true
```

预期现象：KV 缓存分配阶段报 HBM OOM，未进入前向。

## 4. 快速日志位置

- 运行日志：`~/sky_workdir/server.log`
- 上一次裁剪日志：`/tmp/server_tail_debug.log`
- curl 响应（若成功）：`/tmp/logprob_response.json`

## 5. 备注

- 模型与权重路径：`/models/MiMo-V2-Flash`
- 当前 MoE 量化 scale 已按 block 形状加载；gmm 打印（正常路径）
  - gmm1 rhs_scale 形状 `(8,32,1,256)`，块大小 128
  - gmm2 rhs_scale 形状 `(8,1,1,4096)`，块大小 256
- 远端代码已同步最新清理后的 `moe.py`、`model_runner.py`。

## 6. v6e-16 启动脚本（当前可用）

### 申请机器

```bash
sky launch scripts/tpu_v6e16_mimo.sky.yaml -n jiongxuan-v6e16-mimo -y --use-spot -i 120
```

> 注：sky 会自动分配集群名（如 `sky-b537-jiongxuan`），以 `sky status` 输出为准。
> v6e-8 OOM（ep=8 每 chip 需 34.5 GiB，超出 31.25 GiB），需用 v6e-16。

### Run 命令（在机器上执行）

```bash
cd ~/sky_workdir/sgl-jax
source .venv/bin/activate

export PYTHONPATH=~/sky_workdir/sgl-jax/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash \
  --trust-remote-code \
  --tp-size 16 --ep-size 16 \
  --moe-backend epmoe \
  --nnodes 1 --node-rank 0 \
  --dist-init-addr $(hostname -I | awk '{print $1}'):10011 \
  --host 127.0.0.1 --port 30271 \
  --context-length 1024 \
  --max-total-tokens 2048 \
  --max-prefill-tokens 256 \
  --mem-fraction-static 0.3 \
  --disable-precompile --skip-server-warmup \
  --log-level info
```

### 机器信息

- TPU: v6e-16（4 nodes × 4 chips，每 chip 31.25 GiB HBM）
- Zone: asia-northeast1-b（其他 zone 无容量，spot 资源紧张）
- 费用: ~$15.10/h (spot)
- yaml: `scripts`

---

## 7. 近期量化相关改动（本地已改，待确认同步）

- `model_runner.py`：静态 FP8 不再跳过量化，静态/动态都会调用 `apply_linear_quantization` 与 `apply_moe_quantization`。
- `linear.py`：`QuantizedLinear.from_linear` 在静态 FP8 下直接使用 checkpoint 的 float8 权重 + scale，自动压缩 2D block scale；运行时用量化内核而非 BF16 matmul。
- `weight_utils.py`：加载 `*_weight_scale_inv` 时倒数并折叠 2D block scale -> 每输出通道（按 block size 重复），避免广播错误。
- `quantization_config.py`：支持 `weight_block_size`，HF `quantization_config` 的 `fmt/activation_scheme` 生成相应 dtype；`model_config` 自动解析 fp8 字段。
- 预期效果：激活量化可按配置生效，ignored_layers 仍可跳过（尤其 o_proj），block scale 与权重对应齐全。
