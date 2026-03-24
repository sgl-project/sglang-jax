# Fused MoE 内核执行流程（基于 tpu-inference v1 kernel）

> **内核源码**：`blogs/tpu-inference/fused_moe/v1/kernel.py`（~1699 行）
> **公共 API**：`fused_ep_moe()` → JIT + shard_map 封装
> **核心函数**：`_fused_ep_moe_kernel()` → Pallas kernel 主体

## 1. 概述

Fused EP MoE 内核（`_fused_ep_moe_kernel`）实现了基于 Expert Parallelism (EP) 的 MoE 推理，**将 all-to-all 通信和 FFN 计算融合到一个 Pallas 内核中**。这避免了传统方案中多次 kernel launch 的开销，同时实现了通信与计算的细粒度 overlap。

**核心思想**：传统方案将 "路由 → scatter → 计算 → gather → 累加" 拆成多个独立 kernel 调用；本内核将所有阶段融合到一个 Pallas kernel 中，通过显式的异步 DMA 通信和双缓冲流水线实现最大化重叠。

### 输入输出

| 张量 | Shape | 说明 |
|--------|-------|-------------|
| `tokens_hbm` | `(local_num_tokens, t_packing, hidden_size // t_packing)` | 输入 token（packed bf16） |
| `w1_hbm` | `(local_num_experts, 2, hidden_size, intermediate_size)` | Gate + Up 投影权重 |
| `w2_hbm` | `(local_num_experts, intermediate_size, hidden_size)` | Down 投影权重 |
| `gating_hbm` | `(local_num_tokens, padded_num_experts)` | 路由 logits |
| `output_hbm` | `(local_num_tokens, hidden_size)` | 最终输出 |
| `a2a_g_hbm` | `(num_experts, bt, t_packing, hidden_size // t_packing)` | All-to-all gather 临时缓冲（HBM） |

## 2. 流水线阶段

内核以 `bt` 个 token 为一个块进行处理。每个块（`run_per_bt`）执行以下阶段：

### 阶段 1：路由（Routing）

```
代码位置: run_per_bt() → get_top_k() + all_reduce_metadata()
```

1. **获取 gating logits**：从 HBM 异步 DMA 到 VMEM（跨 bt 块双缓冲）
2. **计算评分函数**：对 gating logits 应用 softmax 或 sigmoid
3. **Top-k 选择**：迭代找出每个 token 的 top-k 个 expert
   - 对每个 k：找最大 logit → 记录 expert 索引 → 掩码已选 expert → 重复
   - 产出：`top_k_logits_lst`（权重）、`t2e_routing`（token-to-expert 映射）、`expert_sizes`、`expert_starts`
4. **Ring all-reduce 元数据**：通过环形 all-reduce 在所有 EP 设备间传播 `expert_sizes` 和 `expert_starts`
   - 每个设备将本地 expert 分配计数发送给右邻居
   - 经过 `num_devices - 1` 轮后，每个设备都知道全局分配情况
   - 产出：每 expert 的 token 计数（`expert_sizes`）、起始偏移（`expert_starts`）、每设备每 expert 计数（`d2e_count`）
   - 结果存储在 SMEM 中，供 scatter/gather 阶段使用

### 阶段 2：All-to-All Scatter（逐 expert）

```
代码位置: start_a2a_scatter() / wait_a2a_scatter_recv() / wait_a2a_scatter_send()
```

对每个本地 expert `local_e_id`：
1. 遍历当前块中所有 `bt` 个 token 和所有 `top_k` 个选择
2. 对每个 token-expert 分配：
   - 如果 expert 在**本地**设备上：DMA 将 token 从 HBM 拷贝到 VMEM scatter 缓冲区
   - 如果 expert 在**远程**设备上：异步远程 DMA 拷贝到目标设备的 VMEM scatter 缓冲区
3. 记录发送计数，用于后续的发送等待同步

scatter 缓冲区 `a2a_s_x2_vmem` 通过 `e_sem_id` 进行双缓冲（在 expert 间 ping-pong），使得**下一个** expert 的 scatter 可以与**当前** expert 的 FFN 计算重叠。

### 阶段 3：Expert FFN 计算

```
代码位置: expert_ffn() → dynamic_ffn1() + dynamic_ffn2()
```

每个 expert 的 FFN 处理被分散到该设备上的 token。计算在多个维度上进行分块：

#### FFN1（Gate + Up 投影）

对每个权重矩阵的 `(bf_id, bd1_id)` 分块：
1. 从 HBM 获取权重块 `w1[local_e_id, 0, bd1_slice, bf_slice]` 和 `w1[local_e_id, 1, bd1_slice, bf_slice]` 到 VMEM（通过 `bw_sem_id` 双缓冲）
2. 对每个 token 微批次（每次 `btc` 个 token，动态循环至 `dyn_sz` 实际 token 数）：
   - 从 int32 解包 packed bf16 token
   - 计算 `acc1 += token_block @ w1_block`（gate 投影）
   - 计算 `acc3 += token_block @ w3_block`（up 投影）
   - 如有量化 scale 则应用

#### 激活函数

FFN1 和 FFN2 之间应用激活函数（SiLU、GELU 或 SwiGLU-OAI）：
```
act = activation(acc1) * acc3
```
这在 FFN2 计算中内联执行，避免物化中间结果。

#### FFN2（Down 投影）

对每个 `(bf_id, bd2_id)` 分块：
1. 从 HBM 获取权重块 `w2[local_e_id, bf_slice, bd2_slice]` 到 VMEM
2. 对每个 token 微批次：
   - 应用激活：`act = silu(acc1) * acc3`
   - 计算 `res += act @ w2_block`
   - 如有量化 scale 则应用
   - 将结果重新打包为 int32（2x bf16 → 1x int32）

结果写入 `a2a_s_acc_x2_vmem`（scatter 累加缓冲区，复用于 gather）。

### 阶段 4：All-to-All Gather（逐 expert）

```
代码位置: start_a2a_gather() / wait_a2a_gather_send() / wait_a2a_gather_recv_all()
```

每个 expert 的 FFN 计算完成后：
1. 对每个源设备：将计算结果发回源设备
   - 本地结果：DMA 从 VMEM 拷贝到 `a2a_g_hbm` 临时缓冲
   - 远程结果：异步远程 DMA 拷贝到源设备的 `a2a_g_hbm`
2. 所有 expert 处理完后，等待所有 gather 接收完成

### 阶段 5：累加（Accumulation）

```
代码位置: bt_acc()
```

1. 对 bt 块中每个 token，对每个 top-k 选择：
   - 查找被选中的 expert（`t2e_routing`）及其在 gather 缓冲区中的偏移
   - DMA 将 expert 结果从 `a2a_g_hbm` 拷贝到 `a2a_g_acc_vmem`
2. 加权求和：`output = Σ_k (expert_result_k * top_k_weight_k)`

### 阶段 6：输出写回

```
代码位置: start_send_bo() / wait_send_bo()
```

1. 将累加输出写入 `b_output_x2_vmem`（跨 bt 块双缓冲）
2. 异步 DMA 从 VMEM 拷贝到 `output_hbm`
3. 等待上一个 bt 块的写回完成后再复用缓冲区

## 3. 核心设计模式

### 3.1 三级双缓冲

内核在三个层级使用 ping-pong（x2）缓冲以最大化重叠：

| 层级 | 缓冲区后缀 | 索引变量 | 目的 |
|-------|------------|----------|------|
| **Token 块** | `_x2_` 按 `bt_sem_id` 索引 | `bt_id % 2` | 重叠：取 gating(下一 bt) 与处理(当前 bt) |
| **Expert** | `_x2_` 按 `e_sem_id` 索引 | 交替 0/1 | 重叠：scatter(下一 expert) 与 FFN(当前 expert) |
| **权重** | `_x2_` 按 `bw_sem_id` 索引 | `(bw_sem_id + 1) % 2` | 重叠：取权重(下一 tile) 与 matmul(当前 tile) |

### 3.2 重叠策略

expert 循环（`run_per_expert`）中的主要重叠模式：

```
Expert E (e_sem_id=0)          Expert E+1 (e_sem_id=1)
========================       ========================
                               start_a2a_scatter(E+1)  ← 提前启动
wait_a2a_scatter_recv(E)
start_fetch_bw1/bw3(E)
  ┌─ FFN1 循环 ─────────┐
  │ fetch_next_bw        │      [E+1 的 scatter 正在传输]
  │ wait_bw, matmul      │
  └──────────────────────┘
  ┌─ FFN2 循环 ─────────┐
  │ fetch_next_bw        │      [E+1 的 scatter 正在传输]
  │ wait_bw, act+matmul  │
  └──────────────────────┘
start_a2a_gather(E)            ← E+1 scatter 继续进行的同时 gather E 的结果
wait_a2a_scatter_send(E)
sync_barrier()
```

### 3.3 Token 打包（2x bf16 → 1x int32）

为减少 DMA 传输量并提高内存效率：
- 两个 bf16 值打包到一个 int32 中（`t_packing = 2`）
- Token 形状变为 `(num_tokens, t_packing, hidden_size // t_packing)` → 以 int32 存储为 `(num_tokens, hidden_size // t_packing)`
- 计算时通过 `bitcast` 和位移操作解包 token
- FFN2 后通过位移和 OR 操作重新打包结果

### 3.4 动态 Token 数量

Expert FFN 使用动态循环边界（`dyn_sz = expert_sizes[e_id]`）来跳过空 slot 的计算。`fori_loop` 的 `num_loops = cdiv(dyn_sz, btc)` 确保只处理实际分配给该 expert 的 token。

### 3.5 存储层级

| 存储 | 用途 | 示例 |
|------|------|------|
| **HBM** | 大容量存储 | token、权重、gating logits、输出、a2a_g 临时缓冲 |
| **VMEM** | 计算缓冲 | scatter/gather token 缓冲、权重分块、累加器 |
| **SMEM** | 元数据/索引 | 路由表、expert 偏移/大小、DMA 计数 |

## 4. 调优参数

| 参数 | 说明 | 典型范围（Ring-1T） |
|------|------|---------------------|
| `bt` | token 批次的块大小 | 16–256 |
| `bf` | intermediate 维度的块大小 | 256–2560 |
| `bd1` | FFN1 中 hidden 维度的块大小 | 1024–6144 |
| `bd2` | FFN2 中 hidden 维度的块大小 | 1024–6144 |
| `btc` | token 的计算 tile 大小（内层循环） | 16–64 |
| `bfc` | intermediate 维度的计算 tile 大小 | 256–2560 |
| `bd1c` | FFN1 中 hidden 维度的计算 tile 大小 | 1024–6144 |
| `bd2c` | FFN2 中 hidden 维度的计算 tile 大小 | 1024–6144 |

**约束条件**:
- `bt` 必须整除 `local_num_tokens`
- `btc` 必须整除 `bt`
- `bf/bd1/bd2` 必须整除 `intermediate_size/hidden_size`
- `bfc/bd1c/bd2c` 必须整除 `bf/bd1/bd2`
- 所有计算 tile 大小必须对齐到 `t_packing * 128`

**权衡**:
- 更大的 `bd1/bd2/bf` → 更少的权重获取迭代 → 更好的计算/内存重叠，但 VMEM 压力更大
- 更大的 `bt` → 每个 expert 批次更多 token → 更高的 MXU 利用率，但 scatter/gather 缓冲区更大
- 更大的 `btc` → 更少的 matmul 调用次数，但稀疏 expert 场景下可能 MXU 利用率不足

## 5. 流水线图

```
时间 →

bt_id=0:
  ┌────────────┐
  │ 获取 Gate  │
  │ Top-K      │
  │ AllReduce  │ ← 路由
  ├────────────┤
  │ Expert 0   │ ┌──scatter(E1)──────────────────────┐
  │  scatter   │ │                                    │
  │  FFN1      │ │                                    │
  │  FFN2      │ │                                    │
  │  gather    │ │                                    │
  ├────────────┤ │                                    │
  │ Expert 1   │ │  ┌──scatter(E2)──────────────┐     │
  │  wait_scat │◄┘  │                            │     │
  │  FFN1      │    │                            │     │
  │  FFN2      │    │                            │     │
  │  gather    │    │                            │     │
  ├────────────┤    │                            │     │
  │    ...     │    │                            │     │
  ├────────────┤
  │ wait_all   │ ← 等待所有 gather 接收完成
  │ 累加       │ ← top-k 结果加权求和
  │ 写回       │ ← 输出到 HBM
  └────────────┘

bt_id=1:         ← Gate 获取与 bt_id=0 累加重叠
  ┌────────────┐
  │ ...        │
  └────────────┘
```

## 6. Ring-1T 配置

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

TPU v7 每个设备（chip）有 **2 个 TensorCore（core）**，每个 core 是一个独立的 EP 计算单元。因此：

| 设备数 (chips) | Core 数 | EP size | 每 core expert 数 |
|---------------|---------|---------|-------------------|
| 4 | 8 | 8 | 32 |
| 8 | 16 | 16 | 16 |
| **16** | **32** | **32** | **8** |
| 32 | 64 | 64 | 4 |

FP8 部署使用 **16 设备（chips）**，EP size = 16 × 2 = **32**：
- `local_num_experts = 256 / 32 = 8` 每 core expert 数
- `t_packing = 2`（bf16 激活）
- Token 块 `bt` 范围 16–256，取决于总 token 数
- 针对不同 `num_tokens` 设置的调优块大小来自 `tuned_block_sizes.py`
- VMEM 限制 100 MB（`vmem_limit_bytes=100 * 1024 * 1024`）

**并行策略**：DP Attention + EP MoE
- Attention 阶段：DP=4 × TP=8（4 个 DP 组，每组 4 设备/8 cores 做 TP Attention）
- MoE 阶段：EP=32（全部 16 设备/32 cores 组成一个 EP 组做 all-to-all）
- DP 使每个设备的 local tokens 减少，降低 A2A 通信量
