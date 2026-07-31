# GLM-5.2 fused-MoE v2 A2A topology, load, and LLO

## 结论

- 已从 `fused_moe/v2/kernel.py` 单独抽取 A2A scatter 和 gather；不包含 metadata all-reduce、expert GEMM、SwiGLU、gather 后累加。
- 已在 4 hosts / 16 chips / 32 JAX devices 的 TPU v7x `2x2x4` torus 上成功运行 512-token 和 16K-token 两组负载。
- 已生成 6 份 rank-0 finalized LLO：两种 token 数分别包含 scatter FP8、scatter BF16、gather BF16。
- 最终实验：`exp-l0eek2ue0t`
- 最终 artifact：`art-lymb8z4zl6`
- artifact URI：`gs://tpu-for-training-falcon-logs/experiments/exp-l0eek2ue0t/artifacts/art-lymb8z4zl6/`
- 源码提交：`e87014ab275278577bce7d3143cee73546652da4`

标准分析均为 `OK`：

- operator analysis：`an-qbkjig8v1a`
- Pallas LLO analysis：`an-cang76p33k`
- topology/load/metrics/final-LLO readback：`an-pquwlajg36`

## 抽取边界

生产代码位置：

- scatter：`python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py:672`
- gather：`python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py:784`
- BT 调度：`python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py:2047`

抽取后的独立实现：

- `python/sgl_jax/srt/kernels/fused_moe/v2/bench_a2a_llo.py`
- scatter runner：`build_scatter_runner`
- gather runner：`build_gather_runner`
- topology/load 统计：`topology_dict`、`physical_load_dict`

保留的生产语义：

1. scatter 对每个 token 的每个 top-k expert 发起一行 activation DMA。
2. 本地 expert 使用 local async copy；远端 expert 使用 async remote copy。
3. 目标 rank 为 `expert_id // local_experts`，目标 semaphore slot 为
   `expert_id % local_experts`。
4. gather 按 source rank 的 `d2e_count` prefix segment，把 expert 输出送回
   token-owner rank。
5. 使用同样的 send/recv DMA semaphore 和跨 32 logical ranks barrier。
6. routing/count/start/size metadata 经 VMEM staging 到 SMEM 后再做动态索引。

有意排除：

- metadata all-reduce；
- expert 权重和 expert 计算；
- top-k weight 乘法和 gather 后累加；
- shared expert；
- 完整 fused-MoE 的流水重叠。

因此本结果描述 A2A scatter/gather 通信本身，不是完整 fused-MoE latency。

## 模型和输入

配置来自
[`zai-org/GLM-5.2/config.json`](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)：

| 参数 | 值 |
|---|---:|
| hidden size | 6144 |
| routed experts | 256 |
| top-k | 8 |
| MoE intermediate size | 2048 |
| model dtype | BF16 |

EP=32，所以每个 logical rank 持有 8 个 routed experts。

scatter 测试两种 activation payload：

- FP8：每个 routed row 为 `6144 × 1 byte = 6 KiB`；
- BF16：每个 routed row 为 `6144 × 2 bytes = 12 KiB`。

gather 传输 BF16 expert output，每个 routed row 为 12 KiB。

负载使用确定性的均衡 routing，保证 expert load 完全均衡。它是用于编译器复现的
受控负载，不是线上请求的 routing trace。真实 routing 会改变 rank-pair 流量、
gather segment 长度和热点程度，但不会改变 LLO 中的动态通信结构。

## 物理和逻辑拓扑

| 项目 | 值 |
|---|---|
| TPU | v7x |
| chips | 16 |
| cores/devices | 32，2 cores per chip |
| hosts | 4，4 chips / 8 devices per host |
| physical torus | `2x2x4` |
| JAX mesh | `data=1, tensor=32` |
| EP size | 32 |

logical rank 到物理位置的映射为：

| process/host | logical ranks | chip coordinates | cores |
|---:|---|---|---|
| 0 | 0–7 | `(x,y,z)`，`x∈{0,1}, y∈{0,1}, z=0` | 每个 coordinate 的 core 0/1 |
| 1 | 8–15 | `(x,y,z)`，`x∈{0,1}, y∈{0,1}, z=1` | 每个 coordinate 的 core 0/1 |
| 2 | 16–23 | `(x,y,z)`，`x∈{0,1}, y∈{0,1}, z=2` | 每个 coordinate 的 core 0/1 |
| 3 | 24–31 | `(x,y,z)`，`x∈{0,1}, y∈{0,1}, z=3` | 每个 coordinate 的 core 0/1 |

完整 32-device mapping 在 readback analysis 的 `handoff_summary.json` 中。

## Kernel shapes

`packing=4` 表示 FP8，`packing=2` 表示 BF16。

| tokens | phase | per-device source | per-device destination |
|---:|---|---|---|
| 512 | scatter FP8 | `[16,4,1536]` | `[8,512,4,1536]` |
| 512 | scatter BF16 | `[16,2,3072]` | `[8,512,2,3072]` |
| 512 | gather BF16 | `[8,512,2,3072]` | `[256,16,2,3072]` |
| 16384 | scatter FP8 | `[512,4,1536]` | `[16,8,1024,4,1536]` |
| 16384 | scatter BF16 | `[512,2,3072]` | `[16,8,1024,2,3072]` |
| 16384 | gather BF16 | `[16,8,1024,2,3072]` | `[16,256,32,2,3072]` |

512-token case 使用 `bt=16, num_bt=1`；16K-token case 使用
`bt=32, num_bt=16`。

## Logical rank-pair load

### 512 tokens

- 每个 rank 有 16 个 local tokens。
- 全局 routed rows：`512 × 8 = 4096`。
- 每个 expert 收到 16 rows。
- 每个 source rank 只连接 16 个 destination ranks。
- 非零 directed rank pair 为 8 rows，即 FP8 48 KiB 或 BF16 96 KiB。
- scatter row DMA 数：4096，其中 3968 为 remote、128 为 self。
- gather matrix 是 scatter matrix 的转置。

受控 routing 的两种 source-row pattern 为：

```text
[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
```

### 16K tokens

- 每个 rank 有 512 个 local tokens。
- 全局 routed rows：`16384 × 8 = 131072`。
- 每个 expert 在每个 BT 收到 32 rows。
- 每个 source rank 连接全部 32 个 destination ranks。
- 每个 directed rank pair 跨 16 个 BT 共 128 rows，即 FP8 768 KiB 或
  BF16 1.5 MiB。
- 每个 directed rank pair 在每个 BT 为 8 rows。
- scatter row DMA 数：131072，其中 126976 为 remote、4096 为 self。
- gather matrix 与 scatter matrix相同，因为这里的 32×32 矩阵完全均衡。

完整 32×32 scatter/gather row matrices 在 `handoff_summary.json` 中。

## Physical payload

下表中的 total 包含 self copy；remote 排除 self，但包含 same-chip peer core、
same-host other chip 和 cross-host。

| tokens | phase/dtype | total | remote | cross-host | same-host other chip | same-chip peer core | self |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | scatter FP8 | 24 MiB | 23.25 MiB | 18 MiB | 4.5 MiB | 0.75 MiB | 0.75 MiB |
| 512 | scatter BF16 | 48 MiB | 46.5 MiB | 36 MiB | 9 MiB | 1.5 MiB | 1.5 MiB |
| 512 | gather BF16 | 48 MiB | 46.5 MiB | 36 MiB | 9 MiB | 1.5 MiB | 1.5 MiB |
| 16384 | scatter FP8 | 768 MiB | 744 MiB | 576 MiB | 144 MiB | 24 MiB | 24 MiB |
| 16384 | scatter BF16 | 1536 MiB | 1488 MiB | 1152 MiB | 288 MiB | 48 MiB | 48 MiB |
| 16384 | gather BF16 | 1536 MiB | 1488 MiB | 1152 MiB | 288 MiB | 48 MiB | 48 MiB |

4×4 directed host-pair matrix 完全均衡：

- 512：每个 host pair 为 FP8 1.5 MiB 或 BF16 3 MiB；
- 16K：每个 host pair 为 FP8 48 MiB 或 BF16 96 MiB。

按最短 torus hops 划分：

| tokens | dtype | 0 hop | 1 hop | 2 hops | 3 hops | 4 hops |
|---:|---|---:|---:|---:|---:|---:|
| 512 | FP8 | 1.5 MiB | 6 MiB | 9 MiB | 6 MiB | 1.5 MiB |
| 512 | BF16 | 3 MiB | 12 MiB | 18 MiB | 12 MiB | 3 MiB |
| 16384 | FP8 | 48 MiB | 192 MiB | 288 MiB | 192 MiB | 48 MiB |
| 16384 | BF16 | 96 MiB | 384 MiB | 576 MiB | 384 MiB | 96 MiB |

## Measured latency

每个 case 运行 2 次 warmup 和 5 次 sample。下表的主值取 JAX process 0 的
sample median；range 为 4 个 process 各自 median 的范围。

| tokens | variant | process 0 median | 4-process range | global useful remote payload rate |
|---:|---|---:|---:|---:|
| 512 | scatter FP8 | 0.540 ms | 0.516–0.545 ms | 45.1 GB/s |
| 512 | scatter BF16 | 0.623 ms | 0.559–0.623 ms | 78.3 GB/s |
| 512 | gather BF16 | 0.542 ms | 0.540–0.583 ms | 90.0 GB/s |
| 16384 | scatter FP8 | 2.339 ms | 2.326–2.364 ms | 333.5 GB/s |
| 16384 | scatter BF16 | 3.839 ms | 3.828–3.852 ms | 406.4 GB/s |
| 16384 | gather BF16 | 3.645 ms | 3.645–3.700 ms | 428.1 GB/s |

`global useful remote payload rate = global remote payload bytes / latency`。它不是单 link
峰值，也不是硬件 ICI 理论带宽；latency 包含动态控制、metadata staging、
semaphore wait 和 32-rank barrier。该 workload 的 scatter 是 6 KiB/12 KiB
row-granular DMA，gather 在这个均衡 routing 下也形成 1-row segments，因此主要用于
复现当前通信代码生成和小包控制开销。

## LLO

最终 artifact 中每个 rank 有 108 个编译阶段 dump，4 ranks 共：

- 432 LLO files；
- 54,153,064 bytes；
- 6 个 kernel symbols；
- 无 MXU matmul/prepare/result 指令。

rank-0 的 6 份 finalized LLO：

| kernel | bytes | SHA256 |
|---|---:|---|
| scatter t512 FP8 | 58,763 | `af36156c863fa9555daa00a0c68d2dd4045bd49704c24c4128e8bd1a8b34d5b8` |
| scatter t512 BF16 | 58,753 | `febbf5a14c3f5e8a96950caa81716be1abb43a14bbc9631a6f54e7e94e3a6f9c` |
| gather t512 BF16 | 103,872 | `d2b12aa9f9e7c5c7b049434e77af72a6c7ac7c25bcad669abff1ab4928fee090` |
| scatter t16384 FP8 | 62,441 | `21e3b7742de01dd8f396bd366ba298ae6c390ab334e44f30e08e8f52eaab6081` |
| scatter t16384 BF16 | 62,432 | `7c6abd3033302089db17fe937b3fd2b281d597557fc65cb7ea3a187fb4db7a57` |
| gather t16384 BF16 | 115,419 | `79d249140783cfba7abd87e2eb5482a5074858c8bca476231d9f679e70ffe29c` |

rank-0 finalized LLO 的静态 op-site 计数：

| kernel class | `llo.enqueue_dma` | `llo.dma_done` | `llo.vwait.ge` | `llo.vsync.add.remote` |
|---|---:|---:|---:|---:|
| scatter，任一 tokens/dtype | 19 | 5 | 2 | 64 |
| gather，任一 tokens | 66 | 4 | 2 | 64 |

这些是静态 LLO op-site 数，不是一次执行实际发出的 DMA 数。动态 `fori_loop` 会重复执行
同一 op-site；`llo.vsync.add.remote` 主要来自两次 32-rank barrier signaling。

标准 Pallas LLO analysis 对全部 432 个阶段文件的归一化 inventory 为：

- DMA enqueue：2496；
- DMA done：336；
- wait：144；
- barrier：4752；
- scalar/control：61128；
- MXU：0。

这个汇总跨 rank、kernel 和编译阶段重复计数，只用于检查 artifact 完整性，不应解释为
一次 kernel invocation 的动态指令数。

## 获取交付物

```bash
# 完整 topology、32x32 row matrix、4x4 host matrix 和 payload 分类
falcon workflow analysis cat an-pquwlajg36 handoff_summary.json --output text

# 4 个 process 的全部 timing samples
falcon workflow analysis cat an-pquwlajg36 metrics.json --output text

# 6 份 rank-0 finalized LLO 的拼接文本
falcon workflow analysis cat an-pquwlajg36 rank-0-final-llo.txt --output text

# readback inventory 和每份 finalized LLO 的 hash
falcon workflow analysis cat an-pquwlajg36 report.md --output text

# 标准 Pallas LLO 指令族汇总
falcon workflow analysis cat an-cang76p33k report.md --output text
```

原始 artifact 保留 4 个 rank 的所有编译阶段 LLO、每-rank benchmark JSON 和
topology/load 文件。

## 复现

```bash
falcon workflow profile submit \
  -f benchmark/kernels/fused_moe_v2_a2a_llo_v7x32.yaml \
  --output json
```

提交新实验后，把三个 analysis YAML 中的 `exp_id` 更新为新实验，再分别运行：

```bash
falcon workflow analysis create \
  -f benchmark/kernels/fused_moe_v2_a2a_llo_readback_analysis.yaml \
  --output json

falcon workflow analysis create \
  -f benchmark/kernels/fused_moe_v2_a2a_llo_operator_analysis.yaml \
  --output json

falcon workflow analysis create \
  -f benchmark/kernels/fused_moe_v2_a2a_llo_pallas_analysis.yaml \
  --output json
```
