# MiMo-V2-Flash 权重加载 → 分片 → 计算数据流

> 重点记录 shape 变化和 dim 对应关系，面向 TP=16 / EP=16 配置。

---

## 0. 关键配置

| 参数 | 值 |
|------|-----|
| `hidden_size` | 4096 |
| `num_attention_heads` (Q, 非SWA) | **64** |
| `num_key_value_heads` (KV, 非SWA) | 4 |
| `swa_num_attention_heads` (Q, SWA) | **64** |
| `swa_num_key_value_heads` (KV, SWA) | 8 |
| `head_dim` (Q/K) | 192 |
| `v_head_dim` | 128 |
| `n_routed_experts` | 256 |
| `moe_intermediate_size` | 2048 |
| TP (tensor parallel) | 16 |
| EP (expert parallel) | 16 |
| `tp_size_moe` = world / EP | **1**（MoE 内无 TP）|
| FP8 block_size_in (`subc_quant_wsz`) | 128 |
| FP8 block_size_out (attention) | 96 |

每个 TP shard 分配（非SWA层）：

| 类型 | 全局 | 每设备 |
|------|------|--------|
| Q heads | **64** | **4** |
| KV heads（padding 后）| 16 | 1 |
| Experts | 256 | 16 |

---

## 1. Attention 数据流

### 1.1 权重加载

#### q_proj

```
HF checkpoint
  weight_q     : [12288, 4096]  fp8    (out = 64heads×192, in = 4096)
  weight_scale : [128, 32]      fp32   (out_blocks = 12288/96, in_blocks = 4096/128)

WeightMapping: sharding=("tensor", None), kv_head_padding=False

加载后 sharding
  weight_q     P("tensor", None)  →  逻辑 [12288, 4096] | 每设备 [768, 4096]   (4 heads)
  weight_scale P(None, None)      →  逻辑 [128, 32]     | 每设备 [128, 32]       (全量复制)
```

> **scale 为什么全量复制？**
> `xla_quantized_matmul_local` 用 `lax.axis_index("tensor")` 计算当前设备的
> `out_global_offset`，从完整 scale 里取对应行。scale 被分片则索引错误（Bug 3 的根因）。

---

#### k_proj（含 kv_head_padding）

```
HF checkpoint
  weight_q     : [768, 4096]   fp8    (out = 4heads×192, in = 4096)
  weight_scale : [8, 32]       fp32   (out_blocks = 4heads × step_size=2, in_blocks = 32)
                                       step_size = ceil(head_dim/block_size_out) = ceil(192/96) = 2

Step 1: kv_head_padding（需要 tp_size=16 > num_kv_heads=4）

  weight  [768, 4096]  -repeat-per-head(reps=4)→  [3072, 4096]
    原始排列: [H0_192rows, H1_192rows, H2_192rows, H3_192rows]
    填充结果: [H0×4, H1×4, H2×4, H3×4]
              即 H0 连续复制 4 次，再 H1 连续复制 4 次……
    实现: jnp.repeat(weight, reps=16//4, axis=0)

  scale   [8, 32]   -tile-per-head(step=2, reps=4)→  [32, 32]
    原始: [S0a,S0b, S1a,S1b, S2a,S2b, S3a,S3b]   (每 head 2 行)
    填充: [S0a,S0b,S0a,S0b,S0a,S0b,S0a,S0b,       ← H0 的 2 行 scale 组重复 4 次
           S1a,S1b,S1a,S1b,S1a,S1b,S1a,S1b,
           S2a,S2b,...,  S3a,S3b,...]

Step 2: 加载后 sharding
  weight_q     P("tensor", None)  →  逻辑 [3072, 4096] | 每设备 [192, 4096]   (1 head)
  weight_scale P(None, None)      →  逻辑 [32, 32]     | 每设备 [32, 32]       (全量复制)
```

---

#### v_proj（含 kv_head_padding）

```
HF checkpoint
  weight_q     : [512, 4096]   fp8    (out = 4heads×128, in = 4096)
  weight_scale : [8, 32]       fp32   (step_size = ceil(128/96) = 2, 与 k_proj 相同)

kv_head_padding（同 k_proj 逻辑）:
  weight  [512, 4096]  → [2048, 4096]
  scale   [8, 32]      → [32, 32]

加载后 sharding
  weight_q     P("tensor", None)  →  逻辑 [2048, 4096] | 每设备 [128, 4096]   (1 head)
  weight_scale P(None, None)      →  逻辑 [32, 32]     | 每设备 [32, 32]
```

---

#### o_proj

```
HF checkpoint
  weight_q     : [4096, 8192]  fp8    (out = hidden_size=4096, in = 64heads×128=8192)
  weight_scale : [32, 64]      fp32   (out_blocks = 4096/128=32, in_blocks = 8192/128=64)

WeightMapping: sharding=("tensor", None)
  kernel_axes = ("tensor", None)  →  input_axis="tensor"（行并行），reduce_axis="tensor"（psum）

加载后 sharding（weight_q 以 [in, out] 格式存储）:
  weight_q     P("tensor", None)  →  逻辑 [8192, 4096] | 每设备 [512, 4096]
                                      dim0 对应 in 维度（64heads×128 的 1/16 = 4heads×128）
  weight_scale P(None, None)      →  逻辑 [64, 32]     | 每设备 [64, 32]
```

---

### 1.2 前向计算（Forward Pass）

输入 `hidden_states : [T, 4096]`，T = total tokens，`P(None, None)` 全量复制。

```
┌─────────────────────────────────────────────────────────────────────┐
│                          q_proj（列并行）                            │
│  shard_map in_specs:                                                │
│    x       P(None, None)    本地 [T, 4096]                          │
│    w_q     P("tensor",None) 本地 [768, 4096]   (4 heads)            │
│    w_scale P(None, None)    本地 [128, 32]                          │
│                                                                     │
│  kernel: [T,4096] @ [4096,768] → [T, 768]  × x_scale × w_scale     │
│                                                                     │
│  out_specs P(None, "tensor"):                                       │
│    q  逻辑 [T, 12288] | 每设备 [T, 768]  (4 Q heads × 192 dim)     │
└─────────────────────────────────────────────────────────────────────┘

q.reshape(-1, 64, 192):
  逻辑 [T, 12288] → [T, 64, 192]
  每设备 [T, 768] 对应全局 [T, 4, 192]  ← 4 Q heads per device

┌─────────────────────────────────────────────────────────────────────┐
│              k_proj（列并行，已 padding 至 16 KV heads）              │
│  shard_map in_specs:                                                │
│    x       P(None, None)    本地 [T, 4096]                          │
│    w_q     P("tensor",None) 本地 [192, 4096]  (1 KV head)           │
│    w_scale P(None, None)    本地 [32, 32]                           │
│                                                                     │
│  kernel: [T,4096] @ [4096,192] → [T, 192]  × x_scale × w_scale     │
│                                                                     │
│  out_specs P(None, "tensor"):                                       │
│    k  逻辑 [T, 3072] | 每设备 [T, 192]  (1 KV head × 192 dim)      │
└─────────────────────────────────────────────────────────────────────┘

k.reshape(-1, k.shape[-1] // head_dim, head_dim)
= k.reshape(-1, 3072 // 192, 192)
= k.reshape(-1, 16, 192):
  逻辑 [T, 3072] → [T, 16, 192]
  每设备 [T, 192] 对应全局 [T, 1, 192]  ← 1 KV head per device
                                          ← Bug 4 修复前是 [T*4, 4, 192]（错）

┌─────────────────────────────────────────────────────────────────────┐
│              v_proj（同 k_proj，v_head_dim=128）                      │
│  out_specs P(None, "tensor"):                                       │
│    v  逻辑 [T, 2048] | 每设备 [T, 128]  (1 KV head × 128 dim)      │
└─────────────────────────────────────────────────────────────────────┘

v.reshape(-1, 16, 128):
  逻辑 [T, 2048] → [T, 16, 128]
  每设备 [T, 128] 对应全局 [T, 1, 128]  ← 1 KV head per device

RoPE:
  q: [T, 64, 192] → [T, 64, 192]  (in-place rotate, shape 不变)
  k: [T, 16, 192] → [T, 16, 192]

┌─────────────────────────────────────────────────────────────────────┐
│                   FlashAttention split KV path                      │
│                                                                     │
│  head_dim padding: 192 → 256  (align to 128 boundary)              │
│  v_head_dim padding: 128 → 256                                      │
│  kv_dim_aligned = max(256, 256) = 256                               │
│                                                                     │
│  输入 pad:                                                           │
│    q: [T, 64, 192] → [T, 64, 256]                                   │
│    k: [T, 16, 192] → [T, 16, 256]                                   │
│    v: [T, 16, 128] → [T, 16, 256]                                   │
│                                                                     │
│  k_cache from pool: [cache, 16, 192] → pad → [cache, 16, 256]      │
│    reshape → [num_pages, page_size, 16, 256]                        │
│                                                                     │
│  shard_map in_specs P(None, "tensor"):                              │
│    q_local:       [T, 4, 256]                (4 Q heads)            │
│    k_local:       [T, 1, 256]                (1 KV head)            │
│    v_local:       [T, 1, 256]                                        │
│    k_cache_local: [num_pages, page_size, 1, 256]                    │
│    v_cache_local: [num_pages, page_size, 1, 256]                    │
│                                                                     │
│  wrapper 内本地 GQA: 4 Q heads : 1 KV head                          │
│  BF16 packing 对齐: local_kv_heads=1 → aligned=2                   │
│    → tile-replicate kv head（Bug 5 修复，不能零填充！）               │
│  ragged_paged_attention → attn_out_local [T, 4, 256]                │
│                                                                     │
│  out_specs P(None, "tensor"):                                       │
│    attn_output  逻辑 [T, 64, 256] | 每设备 [T, 4, 256]              │
│                                                                     │
│  slice: [T, 64, 256] → [T, 64, 192]  (去 head_dim padding)         │
│  reshape: [T, 64, 192] → [T, 12288]                                 │
└─────────────────────────────────────────────────────────────────────┘

slice v_head_dim:
  [T, 12288]
  → reshape [T, 64, 192]
  → [..., :128]  →  [T, 64, 128]
  → reshape [T, 8192]   ← o_proj 输入

┌─────────────────────────────────────────────────────────────────────┐
│                        o_proj（行并行）                               │
│  attn_output: [T, 8192]  P(None, "tensor")                         │
│    每设备 [T, 512]  (4 heads × 128 = 64 heads×128 / 16)             │
│                                                                     │
│  shard_map in_specs:                                                │
│    x       P(None, "tensor")  本地 [T, 512]                         │
│    w_q     P("tensor", None)  本地 [512, 4096]                      │
│    w_scale P(None, None)      本地 [64, 32]                         │
│                                                                     │
│  kernel: [T,512] @ [512, 4096] → [T, 4096]                         │
│  psum("tensor"): allreduce 16 设备 → [T, 4096]                      │
│                                                                     │
│  output: [T, 4096]  P(None, None)                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. MoE 数据流

### 2.1 权重加载

```
HF checkpoint（以 gate_proj/up_proj/down_proj 为例）
  experts.N.gate_proj.weight       : [2048, 4096]   fp8  per expert, N=0..255
  experts.N.gate_proj.weight_scale : [32, 1, 2048]  fp32 (k_blocks=32, subc_wsz=128)

加载后合并 3D（EPMoE 内部）:
  wi_0.value   : [256, 2048, 4096]  fp8
                  dim0=n_experts, dim1=intermediate(out), dim2=hidden(in)
  wi_0_scale   : [256, 32, 1, 2048] fp32
                  dim0=n_experts, dim1=k_blocks=4096/128, dim2=1(unused), dim3=intermediate

  wo.value     : [256, 4096, 2048]  fp8
                  dim0=n_experts, dim1=hidden(out), dim2=intermediate(in)

moe_mesh axes: ("expert"=16, "tensor"=1)   ← tp_size_moe=1

sharding:
  wi_0   P("expert", "tensor", None)       每设备 [16, 2048, 4096]   (16 experts, full intermediate)
  wi_1   P("expert", "tensor", None)       每设备 [16, 2048, 4096]
  wo     P("expert", None, "tensor")       每设备 [16, 4096, 2048]

  wi_0_scale P("expert", None, None, "tensor")  每设备 [16, 32, 1, 2048]
  wo_scale   P("expert", None, None, None)      每设备 [16, k_blocks_out, 1, hidden]
```

> `"tensor"` 轴 size=1，故 `P("tensor", None)` 不实际分片 intermediate 维度。
> 每设备完整持有 16 experts 的全部 intermediate（2048）。

---

### 2.2 前向计算

```
输入:
  hidden_states : [T, 4096]  P(None,None)  全量复制
  topk_ids      : [T, 8]     P(None)       top-8 expert ids
  topk_weights  : [T, 8]     P(None)

reshard 到 moe_mesh P(None): 各设备持有完整 [T, 4096]

┌────────────────────────────────────────────────────────────────────┐
│              shard_map（mesh=moe_mesh，expert×tensor=16×1）         │
│                                                                    │
│  in_specs:                                                         │
│    hidden_states P(None)           本地 [T, 4096]  (所有 token)    │
│    wi_0  P("expert","tensor",None) 本地 [16, 2048, 4096]           │
│    topk_ids      P(None)           本地 [T, 8]                     │
│                                                                    │
│  expert_shard_id = lax.axis_index("expert")  ∈ {0,1,...,15}       │
│  group_offset = expert_shard_id × 16          ∈ {0,16,...,240}    │
│                                                                    │
│  _permute:                                                         │
│    找出 topk_ids 中落在本设备 16 experts（ids range）的 token       │
│    x_permuted  : [local_T, 4096]  (本设备负责的 token，按expert排序)│
│    group_sizes : [16]             (每个 local expert 的 token 数)  │
│                                                                    │
│  GEMM1 (gmm):                                                      │
│    x_perm [local_T, 4096]                                          │
│    × wi_0_local [16, 2048, 4096]  → gate [local_T, 2048]          │
│    × wi_1_local [16, 2048, 4096]  → up   [local_T, 2048]          │
│    w_scale: [16, 32, 1, 2048] 按 group_offset 定位 expert 的 scale │
│                                                                    │
│  activation: silu(gate) × up  →  [local_T, 2048]                  │
│                                                                    │
│  GEMM2 (gmm):                                                      │
│    [local_T, 2048] × wo_local [16, 4096, 2048] → [local_T, 4096]  │
│                                                                    │
│  _unpermute: scatter → [T, 4096] × topk_weights → weighted sum    │
│                                                                    │
│  tp_size_moe=1 → 无 psum                                           │
│                                                                    │
│  out_specs P(None): result [T, 4096]  各设备一致                   │
└────────────────────────────────────────────────────────────────────┘

reshard back: [T, 4096]  P(None,None)
```

---

## 3. KV Cache

```
model_runner 分配:
  effective_kv_heads = get_total_num_kv_heads_with_replication(tp=16)
                     = max(4, 16) = 16   (tp > num_kv_heads，需要 replication)

MHATokenToKVPool (split KV):
  k_buffer : [cache_size, 16, 192]   (16 KV heads, head_dim=192)
  v_buffer : [cache_size, 16, 128]   (16 KV heads, v_head_dim=128)

FlashAttention:
  self.num_kv_heads = 16  (从 model_runner 传入)
  kv_partition_axis = "tensor"

  k_cache_paged: [num_pages, page_size, 16, 256]
    sharding: P(None, None, "tensor", None)
    每设备:   [num_pages, page_size,  1, 256]   (1 KV head per device)
```

---

## 4. 维度对应关系总表

| 维度 | 全局 size | 分片轴 | 每设备 | 说明 |
|------|-----------|--------|--------|------|
| Q heads | **64** | "tensor" | **4** | q_proj out |
| KV heads（padded）| 16 | "tensor" | 1 | k/v_proj out，原始 4→16 |
| head_dim (K) | 192 | — | 192 | reshape 后不分片 |
| v_head_dim | 128 | — | 128 | reshape 后不分片 |
| o_proj in (Q×v_head) | **8192** | "tensor"（in 轴）| **512** | 行并行 → psum |
| hidden_size (output) | 4096 | — | 4096 | o_proj out，replicated |
| n_experts | 256 | "expert" | 16 | MoE 专用 |
| intermediate | 2048 | "tensor"(size=1) | 2048 | tp_moe=1，不分片 |

---

## 5. 已修复 Bug 一览

| # | 位置 | 问题描述 | 修复方式 | commit |
|---|------|----------|----------|--------|
| 1 | `_apply_kv_head_padding` weight | 用 `jnp.tile` 重复整个头列表，应 repeat-per-head | `jnp.repeat(w, reps, axis=0)` | `4bf1099` |
| 2 | `_apply_kv_head_padding` scale | aligned 路径对 scale 用 `jnp.repeat`（逐元素），step_size=2 时错误 | 逐 head 循环 tile，每组 step_size 行一起复制 | `7dfa8c8` |
| 3 | `_col/row_linear_scale_sharding` | blockwise scale 返回 `("tensor",None)` 导致 fast-path 加载时被分片，kernel 用全局 offset 索引错误 | 改为 `(None, None)`（全量复制）| `ed59ec7` |
| 4 | attention forward reshape | `k.reshape(-1, self.k_head_num, head_dim)` 用原始 4，但 k_proj 输出已是 padding 后 3072 | `k.reshape(-1, k.shape[-1]//head_dim, head_dim)` | `f140635` |
| 5 | `_ragged_paged_attention_split_wrapper` | BF16 packing 零填充 local_kv_heads(1→2)，GQA 4q:2kv 中有一半 q heads 读到全零 kv | `jnp.tile` 复制真实 kv head 代替零填充 | `8f3b9da` |
