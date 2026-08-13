# GLM-5.2 FP8 服务交付与复现手册

## 1. 交付范围

本交付同时覆盖 GLM-5.2 的两类 FP8 checkpoint：

| 类型 | 模型目录示例 | 量化语义 | 量化配置来源 |
| --- | --- | --- | --- |
| block-wise | `/models/GLM-5.2-FP8` | 兼容历史 block-wise FP8 checkpoint；MoE 使用 checkpoint 对应的动态 activation quantization | checkpoint 内置配置 |
| channel-wise（当前默认） | `/models/GLM5.2-fp8-channel-wise` | FP8 per-channel weight；routed/shared MoE 为动态 per-token FP8 activation（W8A8），其余 Linear 保持 BF16 activation（W8A16） | `fp8_glm52_static_per_channel_moe_w8a8_linear_w8a16.yaml` |

channel-wise 启动脚本以当前已经验证的“MoE W8A8、其余 Linear W8A16”配置为准，不等价于旧的全 W8A16 配置。embedding、norm、router gate、`indexer.weights_proj` 等模块保持模型既有精度策略。

代码合入目标为 `epic/glm_5_2`。公开入口位于：

```text
benchmark/glm52/delivery/
├── README.md
├── serve/
│   ├── common.sh
│   ├── blockwise_8chip.sh
│   ├── blockwise_16chip.sh
│   ├── channelwise_8chip.sh
│   └── channelwise_16chip.sh
├── benchmark/
│   ├── common.sh
│   ├── run_8chip.sh
│   └── run_16chip.sh
├── eval/
│   └── run.sh
└── validation/
    └── validate_delivery_config.py
```

这些公开脚本只依赖普通 shell、Python 环境和多机 TPU 网络，不依赖特定调度系统。

## 2. 硬件和拓扑

| 物理 chips | JAX devices | v7x-8 hosts | WORLD | TP/DP/EP | benchmark 并发 | prefix cache |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 16 | 2 | 2 | 16/16/16 | C32 | shared：32 个请求共享一个 128K prefix |
| 16 | 32 | 4 | 4 | 32/32/32 | C64 | independent：64 个请求各有一个独立 128K prefix |

两种 workload 都要求每个 DP rank 恰好处理 2 个请求。16-chip 使用独立 prefix 的目的，是模拟 64 个相互独立的 128K 会话，而不是用共享 cache 低估 KV 容量和 warmup 成本。

服务固定使用 exact DSA、radix top-k、Pallas DSA、fused-MoE v2、BF16 KV cache、page size 64、round-robin DP scheduling。默认 context length 为 135,168。

## 3. 环境准备

每个 host 使用相同代码和 Python 环境：

```bash
git checkout epic/glm_5_2
python3 -m pip install -e './python[tpu]'
```

这里并没有绕过 pyproject：`./python` 指向仓库中的 Python project，pip 会读取 `python/pyproject.toml`；`[tpu]` 选择其中的 TPU optional-dependencies，`-e` 表示 editable install。它等价于：

```bash
cd python
python3 -m pip install -e '.[tpu]'
```

每个 host 必须能访问相同的完整 checkpoint 目录，且至少包含 `config.json`、`model.safetensors.index.json` 和 index 中列出的全部 shard。rank 0 的 25000 端口用于分布式初始化，30000 端口提供服务。

启动脚本默认设置 TPU DVFS p-state 7；如部署环境统一管理频率，可设置 `GLM52_DVFS_P_STATE=off`。不要设置跳过 GCSFuse warmup 或禁止 MoE bulk read 的环境变量，真实 checkpoint 加载依赖顺序预热和 host bulk load。

## 4. Serve 启动

### 4.1 8 chips / 16 devices

在两个 host 分别运行；`RANK` 为 0 和 1，`MASTER_ADDR` 都指向 rank 0：

```bash
WORLD=2 RANK=0 MASTER_ADDR=<rank-0-host> \
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/serve/channelwise_8chip.sh
```

block-wise checkpoint 改用：

```bash
WORLD=2 RANK=0 MASTER_ADDR=<rank-0-host> \
MODEL_PATH=/models/GLM-5.2-FP8 \
  benchmark/glm52/delivery/serve/blockwise_8chip.sh
```

### 4.2 16 chips / 32 devices

在四个 host 分别运行；`RANK` 为 0、1、2、3：

```bash
WORLD=4 RANK=0 MASTER_ADDR=<rank-0-host> \
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/serve/channelwise_16chip.sh
```

block-wise checkpoint 改用：

```bash
WORLD=4 RANK=0 MASTER_ADDR=<rank-0-host> \
MODEL_PATH=/models/GLM-5.2-FP8 \
  benchmark/glm52/delivery/serve/blockwise_16chip.sh
```

脚本会自动选择并发、precompile batch/token、TP/DP/EP、max prefill tokens 和内存比例。启动前会检查 host 数量、checkpoint index、channel-wise 量化语义，以及 EP16/EP32 的 fused-MoE v2 decode/extend 精确 tune 条目；检查失败时不会进入耗时的模型编译。

默认日志路径为 `/tmp/glm52-<quantization>-<8|16>chip-rank<RANK>.log`，可通过 `GLM52_SERVER_LOG` 修改。benchmark 必须读取 rank 0 的这份日志来验证 extend 没有被拆批。

## 5. Benchmark

服务健康后，只在 rank 0 执行：

```bash
QUANTIZATION=channelwise \
  benchmark/glm52/delivery/benchmark/run_8chip.sh

QUANTIZATION=channelwise \
  benchmark/glm52/delivery/benchmark/run_16chip.sh
```

block-wise 测试将 `QUANTIZATION` 改成 `blockwise`。输出默认写到当前目录，也可通过 `OUTPUT=/path/metrics.jsonl` 修改。

benchmark wrapper 故意不提供 prefix-mode 参数：8 chips 固定 shared，16 chips 固定 unique。底层脚本在输出性能数据前会同时检查：

- C32/DP16 或 C64/DP32，即每个 DP rank 恰好 2 个请求；
- warmup 与 measured request 使用一致的 round-robin placement；
- 128K prefix 完整命中，且每个请求完成 1K decode；
- measured extend 只有一个 scheduler batch；
- decode 达到完整 C32/C64，不是部分并发；
- `max_prefill_tokens`、`chunked_prefill_size`、context length、max running requests 和 KV token capacity 足以容纳 workload。

容量下限分别为：

| 拓扑 | 每 DP cached prefixes | 每 DP 最低 token capacity | 说明 |
| --- | ---: | ---: | --- |
| 8 chips / shared | 1 | 135,232 | 128K cache + 2×(1K extend + 1K decode) + 1 page |
| 16 chips / unique | 2 | 266,304 | 2×128K cache + 2×(1K extend + 1K decode) + 1 page |

如果 16-chip 服务的实际 KV capacity 小于 266,304 token/DP，脚本会在 warmup 前明确失败，不会把拆批或 cache eviction 后的结果当成有效性能数据。

## 6. Profile

profile 复用同一个正确性 benchmark。启动服务前设置轻量 capture 范围：

```bash
export SGLANG_PROFILE_MAX_HOSTS=1
export SGLANG_PROFILE_NUM_CHIPS_PER_TASK=1
export SGLANG_PROFILE_NUM_SPARSE_CORES_TO_TRACE=1
export SGLANG_PROFILE_NUM_SPARSE_CORE_TILES_TO_TRACE=1
```

服务 ready 后，在 rank 0 增加 `PROFILE_OUTPUT_DIR`：

```bash
PROFILE_OUTPUT_DIR=$PWD/artifacts/profiles/glm52-channelwise-8chip \
QUANTIZATION=channelwise \
  benchmark/glm52/delivery/benchmark/run_8chip.sh
```

16 chips 对应使用 `benchmark/run_16chip.sh`。默认分别 capture 一次 cache-hit extend 和 3 个 decode step，host/python tracer 为 0。profile 仍会执行与 benchmark 相同的 cache、并发和单 extend batch 校验。

## 7. Eval

固定 evaluator 版本，避免评分器变化：

```bash
SGL_EVAL_COMMIT=32fa49229575e433629c37379821b5a589a2e422
python3 -m pip install \
  "sgl-eval @ git+https://github.com/sgl-project/sgl-eval.git@$SGL_EVAL_COMMIT"
```

推荐用 8-chip 服务运行精度评测；8/16-chip 只改变并行切分和容量，不改变 checkpoint 或量化策略，因此不需要为了精度结论重复占用 16 chips。

GSM8K 默认只跑前 200 条，作为确定性的快速验收：

```bash
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/eval/run.sh gsm8k
```

需要正式精度结论时，必须显式指定全量 1,319 条：

```bash
EVAL_SCOPE=full \
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/eval/run.sh gsm8k
```

AIME26 仍默认运行完整 30 题：

```bash
MODEL_PATH=/models/GLM5.2-fp8-channel-wise \
  benchmark/glm52/delivery/eval/run.sh aime26
```

默认参数：

| 数据集/范围 | examples | threads | temperature | top-p | seed | max tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K quick（默认） | 200 | 128 | 0.0 | 1.0 | 3 | 4,096 |
| GSM8K full（`EVAL_SCOPE=full`） | 1,319 | 128 | 0.0 | 1.0 | 3 | 4,096 |
| AIME26 | 30 | 16 | 1.0 | 0.95 | 3 | 163,840 |

也可以用 `NUM_EXAMPLES=<N>` 显式覆盖 quick/full 的数量。两个数据集都开启 thinking，并传递 `enable_thinking=true`。AIME26 使用随机采样且只有 30 题，单题变化不能单独证明精度回归；需要结合重复实验或更大确定性数据集判断。

## 8. 当前已验证结果

### 8.1 精度

当前 channel-wise 混合量化 AIME26 全集为 28/30（93.3333%），零请求错误、零截断；同一 checkpoint 的全 W8A16 Pallas 基线同为 28/30。GSM8K 的全 W8A16 Pallas 参考为 1,251/1,319（94.8446%）；混合量化 GSM8K 全集仍需以最终完整的 1,319 条、零错误产物作为正式结论，不能用部分进度代替。

### 8.2 8-chip C32 性能

在相同 8-chip、C32、shared 128K cache-hit、1K extend、1K decode workload 下：

| 指标 | channel-wise：MoE W8A8 / 其余 W8A16 | channel-wise：全 W8A16 Pallas | 变化 |
| --- | ---: | ---: | ---: |
| Output throughput | 904.006 tok/s | 852.295 tok/s | +6.07% |
| Decode throughput | 1,045.846 tok/s | 1,003.402 tok/s | +4.23% |
| TPOT p50 | 30.595 ms | 31.889 ms | -4.06% |
| TTFT p50 | 4.947 s | 5.822 s | -15.03% |
| Measured wall | 36.248 s | 38.447 s | -5.72% |

有效结果满足：32/32 请求完成；每个请求命中 131,072 cached tokens；measured extend 为单批 `new_seq=32`、`new_token=32,768`、`cached_token=4,194,304`，每个 DP rank 为 2 个请求；decode 同样保持每 DP rank 2 个请求。

`decode_throughput_tok_s` 表示排除每个请求首 token 后，从最早首 token 到最后请求完成之间的 decode 输出吞吐，适合在同一脚本和 workload 下做横向对比。

### 8.3 8-chip decode profile

当前混合量化 decode module 为约 28.497 ms/step，trace 覆盖完整 75 个 MoE 层。主要 kernel family 的单 step device time 为：

| kernel family | device time / step |
| --- | ---: |
| fused MoE v2 | 12.091 ms |
| quantized matmul | 4.764 ms |
| MLA | 1.075 ms |

prefill trace 曾触及 trace event 数量上限，只覆盖 75 个 MoE 层中的 3 个，因此不能把该 trace 中的 MoE 小计直接乘层数作为完整 prefill 结论。decode trace覆盖完整，可用于逐 kernel 分析。

## 9. 验收清单

- [ ] 四个 serve wrapper 均能在对应 host 数量下启动，错误 WORLD 会立即失败。
- [ ] channel-wise wrapper 输出当前混合量化配置校验成功。
- [ ] EP16 的 token 32/32,768 和 EP32 的 token 64/65,536 精确 tune 条目校验成功。
- [ ] 8-chip benchmark 报告 `prefix_mode=shared`、C32、每 DP 2 请求、单 extend batch。
- [ ] 16-chip benchmark 报告 `prefix_mode=unique`、64 个 unique prefixes、C64、每 DP 2 请求、单 extend batch。
- [ ] 每请求 cache hit 不低于 131,008，且完成 1,024 个输出 token。
- [ ] 快速验收时 GSM8K 产物为 200 条；正式全量验收时显式使用 `EVAL_SCOPE=full`，产物为 1,319 条。
- [ ] AIME26 完整产物为 30 条，所有 eval 的请求错误为 0。
- [ ] 对外发布的分支、模型路径和 evaluator revision 已固定记录。
