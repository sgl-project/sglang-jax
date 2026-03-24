# Fused MoE 内核执行流程（基于 sglang-jax v1 kernel）

> **内核源码**：`python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`（~3089 行）
> **公共 API**：`fused_ep_moe()` (L2573) → JIT + shard_map 封装，支持 2D mesh (DP+TP)
> **核心函数**：`_fused_ep_moe_kernel()` (L457) → Pallas kernel 主体
> **基线版本**：本内核基于 [tpu-inference v1 kernel](https://github.com/vllm-project/tpu-inference) 演进，增加了 HBM staging、三缓冲、shared expert 融合等优化

---

## 1. 概述

Fused EP MoE 内核实现了基于 Expert Parallelism (EP) 的 MoE 推理，将 all-to-all 通信和 FFN 计算融合到一个 Pallas 内核中。

与 tpu-inference 版本的关键区别：

- **预计算路由**：topk_weights/topk_ids 从外部传入，内核不做 gating/top-k 计算
- **分离 w1/w3**：gate 和 up 投影使用独立权重张量（非 `[2,H,F]` 融合布局）
- **2D mesh**：支持 DP+TP 两个轴（`dp_axis_name` + `tp_axis_name`），EP size = DP × TP
- **HBM staging**：token 通过 `bts` 参数分块从 HBM 加载到 VMEM，VMEM 限制从 100MB 降到 64MB
- **Shared expert 融合**：shared expert 计算穿插在 routed expert 流水线间隙中

**源码**: `python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`

---

## 2. 输入输出

| 张量 | Shape | 说明 |
|------|-------|------|
| `tokens_hbm` | `(local_num_tokens, t_packing, hidden_size // t_packing)` | 输入 token（packed bf16） |
| `w1_hbm` | `(local_num_experts, hidden_size, intermediate_size)` | Gate 投影权重 |
| `w3_hbm` | `(local_num_experts, hidden_size, intermediate_size)` | Up 投影权重 |
| `w2_hbm` | `(local_num_experts, intermediate_size, hidden_size)` | Down 投影权重 |
| `topk_weights_hbm` | `(local_num_tokens, top_k)` | 预计算的路由权重 |
| `topk_ids_hbm` | `(local_num_tokens, top_k)` | 预计算的 expert 索引 |
| `a2a_s_x2_hbm` | `(2, align_to(bt*num_devices, bts), t_packing, hidden_size//t_packing)` | Scatter 缓冲（HBM） |
| `a2a_s_acc_x2_hbm` | `(2, ...)` | FFN 结果缓冲（HBM） |
| `a2a_g_hbm` | `(num_experts, bt, t_packing, hidden_size//t_packing)` | Gather 缓冲 |
| `w1_shared_hbm` / `w3_shared_hbm` / `w2_shared_hbm` | — | Shared expert 权重 |
| `w1_shared_scale_hbm` / `w3_shared_scale_hbm` / `w2_shared_scale_hbm` | — | Shared expert FP8 scale |
| `output_hbm` | `(local_num_tokens, hidden_size)` | 最终输出 |

---

## 3. 流水线阶段

### 阶段 1：路由（Routing）

**代码**：`run_bt()` L2210-2238

- 预计算 topk：`topk_weights` / `topk_ids` 从 HBM 异步获取（`start_fetch_topk` L625，`wait_fetch_topk` L645）
- Expert sizes 通过 mask + sum 计算 (L2226-2229)
- Metadata allgather (`all_reduce_metadata` L659)：
  - Power-of-2 设备数：recursive-doubling allgather O(log2(N)) 轮 (L692-725)
  - 非 power-of-2：ring-based 回退 O(N) 轮 (L726-745)
- 与 tpu-inference 区别：无内核 softmax/sigmoid/top-k，路由在外部完成

### 阶段 2：All-to-All Scatter

**代码**：`start_a2a_scatter()` L791

- 使用 `lax.fori_loop` (L842-848) 代替 Python 展开循环，避免 MLIR 代码膨胀
- `@pl.when` 条件 DMA (L813, L823)：只在 local/remote size 非零时发起拷贝
- Scatter 目标是 `a2a_s_x2_hbm`（HBM 缓冲），而非 VMEM
  - tpu-inference 中 token 直接 scatter 到 VMEM
  - sglang-jax 中先 scatter 到 HBM，再通过 bts staging 加载到 VMEM

### 阶段 3：Expert FFN 计算

**代码**：`expert_ffn()` L1515

**Skip-expert 优化**：`lax.cond(has_tokens)` (L1917) 跳过空 expert 的全部计算，仅执行 gather-send drain 和下一 expert 权重预取

**Token HBM→VMEM Staging (bts)**：

- `b_stage_x2_vmem` 双缓冲 (L508)
- `start_stage_a2a_s_tile_from_hbm()` (L1532) / `wait_stage_a2a_s_tile()` (L1547)
- 每次从 HBM 加载 bts 个 token 到 VMEM，计算完再加载下一批

#### FFN1（Gate + Up 投影）

**代码**：`run_gate_up_slices()` L1611

- `with_static_bw` pattern (L1603)：`lax.cond` 确保静态信号量 ID
- nested `fori_loop` over bts token tiles (`run_ffn1_tile` L1650-1707)
- 在每个 bd1 slice 开始时预取下一个 token tile
- **W2 early prefetch**：FFN1 最后一个 bd1 slice 完成时，提前预取 W2 (L1646-1648)

#### FFN2（Down 投影）

**代码**：`run_down_slices()` L1734

- **三缓冲输出**：`a2a_s_acc_stage_x3_vmem` (L509)，3 个 buffer 轮转
  - `buf_load`：从 HBM 加载上次累积结果
  - `buf_compute`：当前 matmul 计算
  - `buf_store`：将上一轮结果写回 HBM
- **Next-expert weight prefetch** (L1886-1897)：FFN2 末尾预取下一 expert 的 W1/W3

### 阶段 4：All-to-All Gather

**代码**：`start_a2a_gather()` L879

- 从 `a2a_s_acc_x2_hbm` 发送计算结果到 `a2a_g_hbm`
- `wait_a2a_gather_recv_all()` (L940-962)：`lax.fori_loop` 逐 expert 检查非零 token 数再等待，避免空 expert 挂起

### 阶段 5：累加（Accumulation）

**代码**：`acc_and_store_output()` L1919

- 双缓冲累加流水线 `run_acc_pipeline()` (L2019)：ping-pong `buf_id` 重叠加载下一 tile 与处理当前 tile
- 动态 gather-wait：`wait_load_acc_bt()` (L1953) 通过 `fori_loop` 计数有效 token 再等待信号量
- **Shared expert 结果相加** (L2006-2010)：SE 累加结果 `b_se_acc_vmem` 加到输出中

### 阶段 6：输出写回

**代码**：`start_send_bo()` L2041，`wait_store_output()` L2051

- `@pl.when(is_valid)` guard (L2057)
- 与 tpu-inference 相同的双缓冲写回模式

---

## 4. Shared Expert 融合

**代码**：`run_shared_expert_slice()` L2071-2204

Shared expert（所有 token 共享的 dense FFN）的计算被切分成多个 block，穿插在 routed expert 的流水线间隙中：

```
Expert 循环 run_per_expert_pipelined (L2247-2297):

  Expert 0:
    SE block 0        ← scatter 前执行 (L2260-2262)
    SE block 1        ← scatter(E1) 启动后、scatter(E0) 等待前执行 (L2278)
    wait_scatter(E0)
    expert_ffn(E0)
    start_gather(E0)
    SE block 2        ← gather 启动后、scatter send 等待前执行 (L2288)
    wait_scatter_send(E0)

  Expert 1:
    SE block 3        ← 同上模式
    ...

  尾部清理:
    SE block N..total ← cleanup_body (L2304-2308)
```

- 独立的双缓冲权重：`b_se_w1_x2_vmem`，`b_se_w3_x2_vmem`，`b_se_w2_x2_vmem`
- 独立的 token 缓冲：`b_se_tokens_vmem` (L510)
- 跨 block 预取：FFN2 期间预取下一 block 的 W1/W3 (L2166-2175)
- SE 输出累加到 `b_se_acc_vmem`（F32），在阶段 5 与 MoE 输出合并

---

## 5. 核心设计模式

### 5.1 四级缓冲

| 层级 | 缓冲区 | 索引变量 | 目的 |
|------|--------|----------|------|
| Token 块 (bt) | `_x2_` 按 `bt_sem_id` | `bt_id % 2` | 重叠：取 topk(下一 bt) 与处理(当前 bt) |
| Expert (e) | `_x2_` 按 `e_sem_id` | 交替 0/1 | 重叠：scatter(下一 expert) 与 FFN(当前 expert) |
| 权重 (bw) | `_x2_` 按 `bw_sem_id` | via `with_static_bw` | 重叠：取权重(下一 tile) 与 matmul(当前 tile) |
| **Token staging (bts)** | `b_stage_x2_vmem` | `token_buf_id` | **新增**：重叠 HBM→VMEM 取 token(下一 bts tile) 与 FFN(当前 bts tile) |

### 5.2 FFN2 输出三缓冲

- 3 个 buffer 在 `a2a_s_acc_stage_x3_vmem` 中轮转
- 流水线：load（上次结果）/ compute（当前 tile）/ store（上一轮结果）
- 消除了 FFN2 tile 间的 store 等待 stall

### 5.3 HBM Staging 模式

- **tpu-inference**：所有 expert token 驻留在 VMEM，VMEM 限制 100 MB
- **sglang-jax**：token 在 HBM 中（`a2a_s_x2_hbm`），按 bts 大小分块加载到 VMEM
- **Trade-off**：消耗额外 HBM 带宽做 staging，但大幅降低 VMEM 压力，支持更大 token 数

### 5.4 with_static_bw Pattern

- `lax.cond(bw_sem_id == 0, body(0), body(1))` (L1603-1609)
- 强制在编译期生成静态信号量 ID，避免 Pallas DMA API 的动态索引问题

### 5.5 Skip-Expert 优化

- `lax.cond(has_tokens)` (L1917) 门控整个 FFN 计算
- 非活跃路径仍执行 gather-send drain 和 next-expert 权重预取

### 5.6 Recursive-Doubling Allgather

- Power-of-2 设备数：O(log2(N)) 轮 (L692-725)，使用 XOR 选择 peer
- 非 power-of-2：O(N) ring 回退 (L726-745)
- 对比 tpu-inference 的纯 ring O(N)，EP=32 时从 31 轮降到 5 轮

### 5.7 存储层级

| 存储 | 用途 | 示例 |
|------|------|------|
| HBM | 大容量存储 + token staging | token、权重、topk、a2a_s/a2a_s_acc 缓冲、a2a_g 缓冲、输出 |
| VMEM | 计算缓冲 | bts token staging、权重分块、累加器、SE 权重/token/acc |
| SMEM | 元数据/索引 | 路由表、expert 偏移/大小、DMA 计数 |

---

## 6. 调优参数

| 参数 | 说明 | 典型范围（Ring-1T） |
|------|------|---------------------|
| `bt` | 外层 token 块大小（路由/通信/输出 tiling） | 16–256 |
| **`bts`** | **Token staging tile（HBM→VMEM，expert FFN 内部）** | **bt 到 bt×ep_size** |
| `bf` | intermediate 维度的块大小 | 256–2560 |
| `bd1` | FFN1 中 hidden 维度的块大小 | 1024–6144 |
| `bd2` | FFN2 中 hidden 维度的块大小 | 1024–6144 |
| `btc` | Token 的计算 tile 大小（内层循环） | 16–64 |
| `bfc` | intermediate 维度的计算 tile 大小 | 256–2560 |
| `bd1c` | FFN1 中 hidden 维度的计算 tile 大小 | 1024–6144 |
| `bd2c` | FFN2 中 hidden 维度的计算 tile 大小 | 1024–6144 |
| **`bse`** | **Shared expert 中间维度的块大小** | **128–2048** |

**Block 配置管理**：`FusedMoEBlockConfig` pytree dataclass (L22-138)，支持 `effective_for()` 方法根据运行时参数（`num_tokens`、`ep_size`、`dtype`、`subc_quant_wsz`）调整配置。

**约束条件**：

- `bts` 必须整除 `bt × ep_size`
- `btc` 必须整除 `bts`
- `bf` / `bd1` / `bd2` 必须整除 `intermediate_size` / `hidden_size`
- `bfc` / `bd1c` / `bd2c` 必须整除 `bf` / `bd1` / `bd2`
- `bse` 对齐到 128
- VMEM 限制 64 MB

---

## 7. 流水线图

```
时间 →

bt_id=0:
  ┌──────────────────┐
  │ 获取 topk(bt=0)  │
  │ 计算 expert_sizes│
  │ AllReduce 元数据  │ ← O(log2(N)) recursive-doubling
  ├──────────────────┤
  │ SE block 0       │ ← shared expert 穿插
  │                  │
  │ Expert 0         │ ┌──scatter(E1)────────────────────────┐
  │  SE block 1      │ │                                      │
  │  wait_scatter    │◄┘                                      │
  │  FFN1 (bts tiles)│──▶ [W2 early prefetch]                 │
  │  FFN2 (3-buf)   │──▶ [next-expert W1/W3 prefetch]        │
  │  start_gather    │                                        │
  │  SE block 2      │    ┌──scatter(E2)──────────────┐       │
  │  wait_send       │    │                            │       │
  ├──────────────────┤    │                            │       │
  │ Expert 1         │    │                            │       │
  │  SE block 3      │    │                            │       │
  │  wait_scatter    │◄───┘                            │       │
  │  FFN1 (bts tiles)│                                 │       │
  │  FFN2 (3-buf)   │                                 │       │
  │  start_gather    │                                 │       │
  │  SE block 4      │                                 │       │
  │  wait_send       │                                 │       │
  ├──────────────────┤
  │ SE cleanup blocks│ ← 剩余 SE blocks
  │ wait_all_gather  │
  │ 累加 (双缓冲)   │ ← MoE + SE 结果合并
  │ 写回             │
  └──────────────────┘

bt_id=1:         ← topk 获取与 bt_id=0 累加重叠
  ┌──────────────────┐
  │ ...              │
  └──────────────────┘
```

---

## 8. Ring-1T 配置

Ring-1T 模型实际参数（来自 `config.json`）：

| 参数 | 值 |
|------|-----|
| `hidden_size` | 8192 |
| `moe_intermediate_size` | 2048 |
| `num_experts` | 256 |
| `top_k` | 8 |
| `num_shared_experts` | 1 |
| `score_function` | sigmoid |
| 量化 | FP8 W8A8 |

TPU v7 每个设备（chip）有 2 个 TensorCore（core），每个 core 是一个独立的 EP 计算单元：

| 设备数 (chips) | Core 数 | EP size | 每 core expert 数 |
|----------------|---------|---------|-------------------|
| 4 | 8 | 8 | 32 |
| 8 | 16 | 16 | 16 |
| **16** | **32** | **32** | **8** |
| 32 | 64 | 64 | 4 |

FP8 部署使用 16 设备，EP=32：

- `local_num_experts` = 256 / 32 = 8 每 core
- VMEM 限制 64 MB
- 14 种信号量 per ping-pong（含 token staging、SE 权重/scale、topk 获取等）

**并行策略**：DP Attention + EP MoE

- Attention：DP=4 × TP=8
- MoE：EP=32（全部 16 设备/32 cores）
- DP 使每设备 local tokens 减少，降低 A2A 通信量

---

## 9. 与 tpu-inference 版本对比

| 特性 | tpu-inference | sglang-jax |
|------|---------------|------------|
| 路由 | 内核内 gating + top-k | 预计算 topk_weights/topk_ids |
| 权重布局 | 融合 w1 `[2,H,F]` | 分离 w1, w3 |
| Mesh | 1D EP | 2D DP+TP |
| Metadata allgather | O(N) ring | O(log2(N)) recursive-doubling |
| Token 存储 | 全在 VMEM | HBM staging + bts 分块加载 |
| FFN2 输出缓冲 | 双缓冲 VMEM | 三缓冲 HBM staging |
| Shared expert | 无 | 穿插在 expert 流水线间隙 |
| Skip-expert | 动态循环边界 | lax.cond 门控整个 FFN |
| Scatter 循环 | Python 展开 | fori_loop |
| Next-expert 预取 | 无 | FFN2 期间预取 W1/W3 |
| W2 预取 | 隐式链式 | FFN1 末尾显式预取 |
| VMEM 限制 | 100 MB | 64 MB |
| 信号量数 | 5 types/ping-pong | 14 types/ping-pong |
| 调优参数 | 8 个 | 10 个（新增 bts, bse） |
| 量化 scale | 分离 w1/w2 粒度 | 统一 subc_quant_wsz |
