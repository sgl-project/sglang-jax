# fused_moe v2 — Roofline & Performance Analysis Report

**日期：** 2026-05-17
**Kernel：** `python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py`
**模型：** MiMo V2 Pro
**集群：** ka7x v7x 16-chip pod（ep=32）
**Commit：** `9b2225a0`（feat/fused-moe-v2-rewrite 分支）

---

## 0. V2 优化出发点（vs V1）

### 0.1 V1 的核心瓶颈

V1 kernel 对 expert FFN 做四维 tiling：`bt`(token batch) × `bd1/bd2`(hidden dim) × `bf/bfc`(intermediate dim) × `bts/btc`(FFN 内 token tile)。在 MiMo V2 Pro workload 下（H=6144, I=2048, fp8 weight），V1 的 hidden dim tiling (`bd1=2048`)将 H=6144 切成 3 块，导致：

1. **权重 DMA 碎片化**：每个 bf tile 的 W1 权重 DMA 形状为 `(bd1c, bfc) = (2048, 1024)`，需要循环 3 次 bd1 tiles × num_bf 次 bf tiles → DMA op 数量 = `3 × 2 × 3 (W1+W3+W2) × t_packing = 36` 次 DMA/expert
2. **GEMM tile 偏小**：V1 的单次 dot 形状 `(btc=64, bd1c=2048) × (bd1c=2048, bfc=1024) → (64, 1024)`，K=2048 不是完整 hidden dim，MXU pipeline 不能充分 amortize startup
3. **Token 重复加载**：每个 bd1 tile 需要重新加载相同的 token 数据，token read 总量 = `num_bd1 × bts × bd1c × sizeof(bf16) = 3 × 64 × 2048 × 2 ≈ 768 KB/expert`

### 0.2 V2 的设计思路（Strix-style）

V2 从 Strix double-buffer expert（`primatrix/strix/kernels/double_buffer_expert.py`）演化，核心改动：

| 维度 | V1 | V2 | 收益 |
|---|---|---|---|
| **Hidden dim tiling** | `bd1/bd2` 切分 H | **不切分**，full hidden_size per bf tile | Token 只需加载 1 次/bts tile |
| **Weight tile 形状** | `(bd1c, bfc)` e.g. `(2048, 1024)` | `(h_per_t, bf)` e.g. `(3072, 1024)` | GEMM K 维更大 → MXU 利用率 ↑ |
| **Double buffer 策略** | W1/W3/W2 各自 double-buffer per bd | W1/W3/W2 各自 double-buffer per bf | 同一 bts tile 内，W2 DMA 与 gate/up MXU overlap |
| **fp8 处理** | HBM 读 fp8 → VMEM dequant → bf16 dot | **相同** | — |
| **A2A scatter** | batch 或 pipelined | **相同** | — |
| **Output accumulation** | 128 次 tiny DMA gather | **相同**（瓶颈，见 §9） | — |

**核心公式：** V2 以「去掉 hidden dim tiling」为支点，将 token load 从 O(num_bd1 × bts) 降到 O(1 × bts)，GEMM K 从 `bd1c=2048` 提升到 `h_per_t=3072`。代价是 weight tile 更大（3072×bf per W），VMEM 占用更高。

### 0.3 预期 vs 实测

| 场景 | 预期 | 实测（ep=32 fp8 wall time）| 结论 |
|---|---|---|---|
| **Prefill (8192)** | MXU 利用率 ↑ → 快 | V1=2.924ms → V2=2.168ms (**-25.9%**) | ✅ 大幅胜出 |
| **Prefill (16384)** | 同上 | V1=5.577ms → V2=3.848ms (**-31.0%**) | ✅ 大幅胜出 |
| **Decode (512)** | 固定 overhead 主导 → 无优势 | V1=0.819ms → V2=0.932ms (**+13.8%**) | ❌ 慢于 V1 |
| **Decode (64)** | padding overhead → 更差 | V1=0.776ms → V2=0.912ms (**+17.5%**) | ❌ 慢于 V1 |

---

## 1. 输入参数

### 1.1 模型 (MiMo V2 Pro)
| 参数 | 值 | 来源 |
|---|---|---|
| `H` (hidden_size) | 6144 | `config.json: hidden_size` |
| `I` (moe_intermediate_size) | 2048 | `config.json: moe_intermediate_size` |
| `E` (n_routed_experts) | 384 | `config.json: n_routed_experts` |
| `top_k` | 8 | `config.json: num_experts_per_tok` |
| MoE layer 数 | **69** | `moe_layer_freq` 数组中 1 的个数 |
| Activation | SwiGLU | `hidden_act=silu` |
| Weight dtype | fp8 e4m3 | `quantization_config.fmt=e4m3` |
| Weight quant block | 128 | `weight_block_size=[128,128]` |
| Activation dtype | bf16 | `--dtype bfloat16` |
| Shared expert | 无 | `n_shared_experts=null` |

### 1.2 并行
| 参数 | 值 |
|---|---|
| EP | 32 |
| DP | 8 |
| TP | 32 |
| 配置 | 16 chip × 2 chiplet = 32 JAX device |

### 1.3 Hardware (TPU v7x Ironwood, per JAX device = per chiplet)
| 参数 | 值 |
|---|---|
| Bf16 peak | **1153.5 TFLOPs** |
| FP8 peak | 2307 TFLOPs |
| HBM BW | **3.69 TB/s** |
| VMEM | 64 MB |
| ICI BW (单向 per device) | ~300 GB/s |
| Roofline 平衡点 (bf16 path) | **312 FLOP/byte** |

> ⚠️ bf16 act × fp8 weight → 走 bf16 MXU path (1153.5 TFLOPs)，fp8 只减 HBM byte 不加速 compute。

### 1.4 V2 Tuned block configs（ep=32 fp8）

| num_tokens | bt | bf | btc | bts | bse | local_nt | padded_nt | num_bt |
|---|---|---|---|---|---|---|---|---|
| 64 | 8 | 512 | 8 | 8 | 256 | 2→8 (padded) | 256 | 1 |
| 128 | 8 | 256 | 8 | 8 | 256 | 4→8 | 256 | 1 |
| 256 | 8 | 512 | 8 | 8 | 256 | 8 | 256 | 1 |
| 512 | 16 | 256 | 16 | 16 | 256 | 16 | 512 | 1 |
| 8192 | 128 | 1024 | 128 | 128 | 256 | 256 | 8192 | 2 |
| 16384 | 128 | 1024 | 128 | 128 | 256 | 512 | 16384 | 4 |

> 对比 V1 tuned configs（同 workload ep=32 fp8）：

| num_tokens | bt | bf | bd1 | bd2 | bts | btc | bfc | bd1c | bd2c | bse |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 2 | 2048 | 2048 | 2048 | 4 | 4 | 2048 | 2048 | 2048 | 2048 |
| 128 | 4 | 2048 | 2048 | 2048 | 8 | 8 | 2048 | 2048 | 2048 | 2048 |
| 256 | 8 | 2048 | 2048 | 2048 | 8 | 8 | 2048 | 2048 | 2048 | 2048 |
| 512 | 16 | 2048 | 2048 | 2048 | 16 | 16 | 2048 | 2048 | 2048 | 2048 |
| 8192 | 128 | 1024 | 2048 | 2048 | 64 | 64 | 1024 | 2048 | 2048 | 1024 |
| 16384 | 128 | 1024 | 2048 | 2048 | 64 | 64 | 1024 | 2048 | 2048 | 1024 |

---

## 2. 理论 HBM Byte（per-layer per-device）

### 2.1 权重 byte

每 device 持有 E_loc = 384/32 = 12 个 expert 的权重。

| 张量 | shape (per device) | dtype | 字节数 |
|---|---|---|---|
| W1 (gate proj) | (12, 6144, 2048) | fp8 | 144.0 MB |
| W2 (down proj) | (12, 2048, 6144) | fp8 | 144.0 MB |
| W3 (up proj) | (12, 6144, 2048) | fp8 | 144.0 MB |
| **小计** | | | **432.0 MB** |
| Scale (W1+W2+W3) | | fp32 | ~108 KB（忽略）|

**V2 关键：权重在每个 bt iteration 的每个 expert 中完整读取一遍。** 如果一个 expert 的 `num_bts_tiles > 1`，权重会被重复加载。

**每 bt 的权重 read = 12 experts × 36 MB = 432 MB**

| num_tokens | bt | num_bt | avg_tokens/expert/bt | num_bts_tiles | 权重总 read |
|---|---|---|---|---|---|
| 512 | 16 | 1 | 10.7 | 1 | 432 MB |
| 8192 | 128 | 2 | 85.3 | 1 | 864 MB |
| 16384 | 128 | 4 | 85.3 | 1 | 1728 MB |

> V1 同样每 bt 读全部 expert 权重：V1 16384 (bt=128, num_bt=4) = 1728 MB，**与 V2 相同**。

### 2.2 Token byte

每 token = H × bf16 = 12288 byte。均匀路由下 tokens_recv/device = `num_tokens × top_k / ep_size`。

| num_tokens | tokens_recv | Input read | Output write | Token total |
|---|---|---|---|---|
| 512 | 128 | 1.5 MB | 1.5 MB | 3.0 MB |
| 8192 | 2048 | 25.2 MB | 25.2 MB | 50.3 MB |
| 16384 | 4096 | 50.3 MB | 50.3 MB | 100.7 MB |

### 2.3 Per-layer per-device HBM 总和

| 阶段 | 权重 | Token | **HBM total** | **时间下限** (÷3.69 TB/s) |
|---|---|---|---|---|
| 512 (decode) | 432 MB | 3.0 MB | **435.0 MB** | **118 μs** |
| 8192 (prefill) | 864 MB | 50.3 MB | **914.3 MB** | **248 μs** |
| 16384 (prefill) | 1728 MB | 100.7 MB | **1828.7 MB** | **496 μs** |

---

## 3. 理论 FLOPs + Arithmetic Intensity

### 3.1 FLOPs

Per-token-per-expert SwiGLU FLOPs = `6 × H × I = 6 × 6144 × 2048 = 7.55e7 FLOP`

| 阶段 | Total FLOP | Per-device FLOP | Compute 时间 (÷1153.5 TFLOPs) |
|---|---|---|---|
| 512 | 3.09e11 | **9.66 GFLOP** | **8.4 μs** |
| 8192 | 4.95e12 | **155 GFLOP** | **134 μs** |
| 16384 | 9.89e12 | **309 GFLOP** | **268 μs** |

### 3.2 Arithmetic Intensity

| 阶段 | FLOP/device | HBM byte/device | **AI** | vs 平衡点 (312) | 结论 |
|---|---|---|---|---|---|
| **512** | 9.66 G | 435 M | **22.2** | 0.071× | **绝对 memory-bound** |
| **8192** | 155 G | 914 M | **169** | 0.54× | **memory-bound**（含 weight re-read） |
| **16384** | 309 G | 1829 M | **169** | 0.54× | **memory-bound**（含 weight re-read） |

> ⚠️ **V2 的 AI 低于 V1 理论值**（V1 16384 AI=581，一次 pass 算法）。这是因为 V2 和 V1 都对权重做 num_bt 次完整读取。但 V2 的优势在于 MXU 利用率更高（更大 GEMM tile），实际执行时间更短。

### 3.3 Roofline 综合下限

| 阶段 | HBM bound | Compute bound | 结论 |
|---|---|---|---|
| 512 | **118 μs** | 8.4 μs | **memory-bound** |
| 8192 | **248 μs** | 134 μs | **memory-bound** |
| 16384 | **496 μs** | 268 μs | **memory-bound** |

> V1 和 V2 在这个 workload 下都是 memory-bound（每个 bt 重读全部 expert weight）。**V2 的真正优势不是改变 bound type，而是在 memory-bound 约束下提高 MXU 利用率、减少 DMA 碎片化、改善 compute/DMA overlap。**

---

## 4. 理论 ICI Byte

| 阶段 | Scatter tokens/dev | Scatter bytes | Total ICI (×2) | ICI 时间 (÷300 GB/s) |
|---|---|---|---|---|
| 512 | 124 | 1.5 MB | 3.0 MB | 10 μs |
| 8192 | 1984 | 24.4 MB | 48.8 MB | 163 μs |
| 16384 | 3968 | 48.7 MB | 97.5 MB | 325 μs |

ICI 在 512 decode 可完全被 HBM overlap，不构成瓶颈。8192/16384 prefill 的 ICI 时间较高，但在 V2 中与 expert FFN（weight DMA + compute）并行执行。

---

## 5. V2 代码逻辑 walkthrough

### 5.1 整体调用链

```
fused_ep_moe_v2()                           # 入口 (L1411)
├── block_config.effective_for()            # 参数解析 + padding (L1447-1464)
├── jax_allreduce_metadata_by_bt()          # JAX 层 all-gather metadata (L1365-1394)
│   ├── compute_local_expert_sizes()        #   vmap bincount per bt
│   └── lax.all_gather()                    #   ICI all-gather
├── kernel()                                # shard_map 入口 (L1671)
│   ├── pad tokens/topk if needed           #   (L1678-1688)
│   └── fused_moe = pallas_call(...)        #   Pallas kernel (L1598-1646)
│       └── _fused_ep_moe_kernel()          #   核心 kernel (L217)
│           ├── sync_barrier()              #   Phase 0: 全 device 同步
│           └── lax.fori_loop(run_bt)       #   Phase 1: bt iteration 主循环
```

### 5.2 run_bt 内部流程（per bt iteration）

```python
def run_bt(bt_id, e_sem_id):                    # L1174
    # 1. Topk prefetch (next bt 的 DMA 提前启动)
    start_fetch_topk(bt_id=next_bt_id)          # L1182

    # 2. Wait topk + compute t2e_routing
    wait_fetch_topk(bt_id)                      # L1185
    t2e_routing = b_topk_ids_x2_vmem[bt_sem_id] # L1187

    # 3. Metadata: HBM→VMEM→SMEM copy of starts/sizes/d2e_counts
    all_reduce_metadata(...)                     # L1189

    # 4. Wait previous output DMA (double-buffer)
    wait_store_output(bt_id - 2)                 # L1193

    # 5A. BATCH SCATTER path (expert_buffer_count >= local_num_experts)
    start_a2a_scatter_batch(...)                 # L1203
    for each expert:
        wait_a2a_scatter_recv(...)               # L1214
        expert_ffn(...)                          # L1218
        start_a2a_gather(...)                    # L1219
    wait_a2a_scatter_send_batch()                # L1240
    wait_a2a_gather_recv_all(...)                # L1241
    sync_barrier()                               # L1242
    acc_and_store_output(...)                    # L1244
    start_send_bo(bt_id)                         # L1245

    # 5B. PIPELINED SCATTER path (expert_buffer_count < local_num_experts)
    # 类似 5A，但 scatter 与 expert_ffn 交错执行
```

### 5.3 expert_ffn 内部流程（Strix-style double-buffer pipeline）

```python
def expert_ffn(bt_sem_id, e_sem_id, local_e_id):    # L854
    dyn_sz = expert_sizes[e_id]                       # 动态 token 数
    num_bts_tiles = ceil(dyn_sz / bts)

    for bts_id in range(num_bts_tiles):               # 外层：bts tile 循环
        # Token load (一次，在 VMEM 中持久化)
        DMA: a2a_s_x2_hbm → b_x_vmem                 # L883-891
        # 192 KB for bts=16 (decode), 1.5 MB for bts=128 (prefill)

        # Weight prologue: prefetch slot[0] of W1/W3/W2
        start_fetch_w1(e_id, slot=0, bf_id=0)         # L894
        start_fetch_w3(e_id, slot=0, bf_id=0)
        start_fetch_w2(e_id, slot=0, bf_id=0)
        # Double-buffer: prefetch slot[1] if num_bf >= 2
        start_fetch_w1(e_id, slot=1, bf_id=1)         # L898-900

        for bf_id in range(num_bf):                   # 内层：bf tile 循环 (static unroll)
            slot = bf_id % 2

            # === Wait weight DMA ===
            wait_fetch_w1(slot)                       # L905
            wait_fetch_w3(slot)
            wait_fetch_w2(slot)

            # === fp8 → bf16 dequant in VMEM ===
            dequant_w1(slot)                          # L909
            dequant_w3(slot)
            dequant_w2(slot)

            # === Gate/Up GEMM (per btc tile) ===
            for btc_id in range(bts // btc):          # L914-929
                for p_id in range(t_packing):   # =2
                    gate += x[btc] @ W1[p_id]   # (btc, h_per_t) × (h_per_t, bf)
                    up   += x[btc] @ W3[p_id]   # 同

            # === Activation + Down GEMM (per btc tile) ===
            for btc_id in range(bts // btc):          # L932-950
                act = silu(gate) * up
                for p_id in range(t_packing):
                    y_acc += act @ W2[p_id]     # (btc, bf) × (bf, h_per_t)
                    # bf_id=0 时 write，之后 accumulate

            # === Prefetch next-next bf tile ===
            start_fetch_w1(e_id, slot, bf_id+2)       # L953-957

        # === Writeback: f32→bf16, DMA to HBM ===
        b_y_stage = y_acc.astype(bf16)                # L960-970
        DMA: b_y_stage → a2a_s_acc_x2_hbm            # L972-980
```

### 5.4 acc_and_store_output 内部流程

```python
def acc_and_store_output(*, bt_sem_id, out_buf_id):   # L992
    acc_bt = gcd(bt, 16)                               # mini-tile size
    # Pipeline: double-buffer acc_bt tiles
    for i in range(bt // acc_bt):                      # L1064-1078
        # Start load next tile (prefetch)
        for t_i in range(acc_bt):                      # L998-1022
            for k_id in range(top_k):                  # =8
                # 1 DMA per (t_id, k_id): a2a_g_hbm[e_id, offset] → VMEM
                # 每次 12 KB (1 token × 6144 bytes bf16)
                DMA: a2a_g_hbm[e_id, offset] → a2a_g_acc_vmem[k_id, t_i]

        # Wait all DMAs for current tile
        # (acc_bt × top_k waits = 16 × 8 = 128 per tile)

        # Weighted accumulate
        for k_id in range(top_k):
            output_tile += a2a_g_acc_vmem[k_id] * topk_weights[:, k_id]
        # Store to output VMEM buffer
```

> ⚠️ **acc_and_store_output 是 decode 性能的主要瓶颈**。对于 512 tokens / ep=32（bt=16），执行 `bt × top_k = 16 × 8 = 128` 次 tiny DMA（每次 12KB），从 `a2a_g_hbm` 的 128 个不连续位置加载。DMA startup overhead 主导。

---

## 6. 流水线排布图

### 6.1 Decode: 512 tokens, ep=32, fp8, bt=16, bf=256

```
bt=16, num_bt=1, 12 experts, ~11 tokens/expert
num_bf=8 (2048/256), bts=16, btc=16
acc_bt=16, num_acc_tiles=1

Time (μs, estimated) →

Phase:   SYNC   TOPK+MD   SCATTER_BATCH   ┌─── EXPERT LOOP (12 experts) ───┐   WAIT    SYNC   ACC+STORE   OUT_DMA
         ┌───┐  ┌──────┐  ┌────────────┐   │ recv→FFN→gather × 12          │  ┌──────┐ ┌───┐ ┌─────────┐  ┌─────┐
TensorCore  │   │  │      │  │            │   │                               │  │      │ │   │ │         │  │     │
(compute)│   │  │      │  │            │   │ dequant+GEMM (per bf tile)     │  │      │ │   │ │ wt.acc  │  │     │
         └───┘  └──────┘  └────────────┘   │ ┌──┬──┬──┬──┐×12 experts      │  └──────┘ └───┘ └─────────┘  └─────┘
                                            │ │b0│b1│b2│b3│                 │
DMA      ─────  ────────  ──────────────   │ └──┴──┴──┴──┘                 │  ──────  ─────  ───────────  ───────
(HBM)                      128 scatter     │ W1/W3/W2 per bf (dbl-buf)     │  gather  barrier 128 tiny   output
                           ops             │ + token load (once)            │  waits          gather DMA  16×6144
                                            └───────────────────────────────┘

         ~50μs  ~30μs      ~40μs            ~120μs (10μs/expert)             ~40μs   ~50μs  ~200μs       ~10μs

Total estimated kernel time: ~540 μs
Measured wall time: 932 μs (含 JAX metadata allreduce ~200μs + dispatch overhead ~190μs)
```

**Expert FFN zoom (per expert, decode bt=16):**
```
Expert FFN internal pipeline (bts=16, btc=16, num_bf=8, ~11 tokens)
1 bts tile × 8 bf tiles (static unroll)

      bf=0          bf=1          bf=2          bf=3    ...    bf=7
┌──────────────┬──────────────┬──────────────┬───── ... ──────────────┐
│ wait W1/W3/W2│ wait W1/W3/W2│ wait W1/W3/W2│            wait      │
│ dequant ×3   │ dequant ×3   │ dequant ×3   │            dequant    │
│ gate+up GEMM │ gate+up GEMM │ gate+up GEMM │            gate+up   │
│ act+down GEMM│ act+down GEMM│ act+down GEMM│            act+down  │
│ prefetch[2]──│→prefetch[3]──│→prefetch[4]──│→ ...                  │
└──────────────┴──────────────┴──────────────┴───── ... ──────────────┘
│ writeback: f32→bf16 + DMA to HBM                                   │
└─────────────────────────────────────────────────────────────────────┘

Per bf tile:
  GEMM shape: (16, 3072) × (3072, 256) × t_packing=2 × 2 (gate+up) = 50.3 MFLOP
  + down:     (16, 256) × (256, 3072) × t_packing=2 = 25.2 MFLOP
  Total compute per bf: 75.5 MFLOP → 75.5e6 / 1153.5e12 = 0.065 μs

  Weight DMA per bf: (2, 3072, 256) × 1 byte × 3 weights = 4.72 MB → 4.72/3.69e3 = 1.28 μs
  ──────────────────────────────────────────────────────────────────────
  AI per bf tile: 75.5e6 / 4.72e6 = 16 FLOP/byte → deeply memory-bound
  Effective time ≈ 1.3 μs per bf tile (DMA 主导)

  8 bf tiles × 1.3 μs = 10.4 μs per expert (不含 token load, writeback)
```

### 6.2 Prefill: 8192 tokens, ep=32, fp8, bt=128, bf=1024

```
bt=128, num_bt=2, 12 experts, ~85 tokens/expert/bt
num_bf=2 (2048/1024), bts=128, btc=128
acc_bt=16, num_acc_tiles=8

Time (μs, estimated) →

     ┌────── bt_id=0 ──────────────────────────────────────────────────────────────┐
     │                                                                              │
     │ SYNC  TOPK+MD  SCATTER_B  ┌── EXPERT LOOP ──────────┐ WAIT  SYNC  ACC+STORE │ OUT_DMA
     │ ┌──┐  ┌─────┐  ┌───────┐  │ FFN×12: token_load+bf×2 │ ┌──┐  ┌──┐  ┌───────┐ │ ┌────┐
     │ │  │  │     │  │       │  │ ┌─────────────────────┐  │ │  │  │  │  │       │ │ │    │
     │ │  │  │     │  │       │  │ │ per expert:         │  │ │  │  │  │  │ 1024  │ │ │    │
     │ └──┘  └─────┘  └───────┘  │ │  tok_ld(1.5MB)     │  │ └──┘  └──┘  │ DMA   │ │ └────┘
     │                            │ │  bf=0: dq+GEMM     │  │             │ gather │ │
     │                            │ │  bf=1: dq+GEMM     │  │             │ + acc  │ │
     │                            │ │  writeback          │  │             └───────┘ │
     │                            │ └─────────────────────┘  │                      │
     │                            └──────────────────────────┘                      │
     │ ~50     ~30      ~80         ~360 (30μs/expert)         ~60  ~50   ~300      │ ~50
     └─────────────────────────────────────────────────────────── ~980 μs ───────────┘

     ┌────── bt_id=1 ──────────────────────── (同上) ──────────────────────────────────┐
     ...
     └──────────────────────────────────────────────────────────── ~980 μs ────────────┘

Total estimated kernel time: ~1960 μs
Measured wall time: 2168 μs (含 JAX metadata ~200μs)
```

**Expert FFN zoom (per expert, prefill bt=128):**
```
Expert FFN internal pipeline (bts=128, btc=128, num_bf=2, ~85 tokens)
1 bts tile × 2 bf tiles

                  bf=0                                    bf=1
┌─────────────────────────────────────────┬─────────────────────────────────────────┐
│ wait W1/W3/W2 (slot 0)                  │ wait W1/W3/W2 (slot 1)                  │
│ dequant fp8→bf16 ×3                     │ dequant fp8→bf16 ×3                     │
│ gate+up GEMM:                           │ gate+up GEMM:                           │
│   (128, 3072) × (3072, 1024) × tp=2    │   same                                  │
│   = 1.61 GFLOP                          │                                         │
│ act+down GEMM:                          │ act+down GEMM:                          │
│   (128, 1024) × (1024, 3072) × tp=2    │   same                                  │
│   = 805 MFLOP                           │                                         │
│ ──── prefetch slot[0] for bf_id+2 ─X    │ (no next bf)                            │
└─────────────────────────────────────────┴─────────────────────────────────────────┘
│ writeback: f32→bf16 + DMA to a2a_s_acc_x2_hbm                                     │
└────────────────────────────────────────────────────────────────────────────────────┘

Per bf tile:
  Compute: 1.61 + 0.805 = 2.42 GFLOP → 2.42e9 / 1153.5e12 = 2.1 μs
  Weight DMA: (2, 3072, 1024) × 1 byte × 3 = 18.9 MB → 18.9/3.69e3 = 5.1 μs
  ──────────────────────────────────────────────────────────────────────
  AI per bf: 2.42e9 / 18.9e6 = 128 FLOP/byte → memory-bound, compute/DMA overlap 有效

  With double-buffer overlap: effective time ≈ max(5.1, 2.1) = 5.1 μs
  2 bf tiles: ~10 μs + token_load(1.5MB → 0.4μs) + writeback(~1μs) = ~12 μs/expert

  12 experts: ~144 μs per bt iteration (不含 overhead)
  理想情况 vs 实测: 144×2 = 288 μs kernel compute vs ~1960 μs estimated
  差额主要来自 scatter/gather ICI + sync_barrier + acc_and_store_output
```

---

## 7. 实测结果（V1 vs V2 wall time）

**条件**：ep=32 fp8 均匀路由 MiMo V2 Pro，wall_timeit（time.monotonic, 10 iterations）。
V1 使用 `use_jax_allreduce_metadata=True`（wall time 包含 JAX metadata 开销）。

| num_tokens | V1 wall (ms) | V2 wall (ms) | Delta (ms) | **Delta (%)** | bound type |
|---|---|---|---|---|---|
| 64 | 0.776 | 0.912 | +0.136 | **+17.5%** | memory |
| 128 | 0.783 | 0.967 | +0.184 | **+23.5%** | memory |
| 256 | 0.799 | 0.887 | +0.088 | **+11.0%** | memory |
| 512 | 0.819 | 0.932 | +0.113 | **+13.8%** | memory |
| 8192 | 2.924 | 2.168 | -0.756 | **-25.9%** | borderline |
| 16384 | 5.577 | 3.848 | -1.729 | **-31.0%** | borderline |

### 7.1 Per-layer per-69-MoE-layer 换算

| 场景 | V2 wall/layer | V2 × 69 | V1 × 69 |
|---|---|---|---|
| 512 (decode) | 0.932 ms | 64.3 ms | 56.5 ms |
| 8192 (prefill) | 2.168 ms | 149.6 ms | 201.8 ms |
| 16384 (prefill) | 3.848 ms | 265.5 ms | 384.8 ms |

---

## 8. 为什么小 token V2 不如 V1

### 8.1 根因一览

| 因素 | 影响 | 机制 |
|---|---|---|
| **Token padding overhead** | 64/128 tokens 严重 | local_nt=2→8 (4×), local_nt=4→8 (2×); MXU 算 padding tokens 的 GEMM 全浪费 |
| **acc_and_store_output 碎片 DMA** | bt×top_k 次 tiny DMA | 128 次 12KB DMA 从不连续 HBM 位置 → DMA startup 主导（估 ~200μs） |
| **sync_barrier ×2** | ~100μs 固定开销 | 32 device 全对全 signal/wait，与 token 数无关 |
| **Metadata DMA** | ~30μs 固定开销 | starts/sizes/d2e_counts HBM→VMEM→SMEM |
| **JAX metadata allreduce** | ~200μs 固定开销 | lax.all_gather + bincount，在 pallas_call 之前 |
| **MXU 利用率低** | btc=8-16 时 GEMM 太小 | (16, 3072) × (3072, 256): M=16 远小于 MXU tile size，pipeline 利用率 <5% |

### 8.2 固定 overhead 占比分析（decode 512 估算）

```
JAX metadata allreduce:  ~200 μs   (21%)
sync_barrier ×2:         ~100 μs   (11%)
scatter_batch (128 DMA): ~40  μs   (4%)
metadata copy to SMEM:   ~30  μs   (3%)
expert_ffn × 12:         ~120 μs   (13%)     ← 有效计算部分
gather waits:            ~40  μs   (4%)
acc_and_store_output:    ~200 μs   (21%)     ← 128 tiny DMA 碎片
output DMA + misc:       ~20  μs   (2%)
host dispatch overhead:  ~180 μs   (19%)     ← XLA dispatch + JAX 开销
─────────────────────────────────────────
Total:                   ~930 μs
MXU 有效利用: ~120 / 930 = 13%
```

### 8.3 为什么 V1 decode 更快

V1 的 decode (512) wall time = 0.819ms。V1 和 V2 共享相同的固定开销（JAX metadata, host dispatch），差异来自 kernel 内部：

1. **V1 的 acc_and_store_output 可能更高效**：V1 使用不同的 accumulation 策略（待 LLO 验证）
2. **V1 scatter 模式**：V1 对 decode 也使用 batch scatter，但 DMA pattern 可能不同
3. **V1 无 token padding**：V1 的 bt=16 直接对 local_nt=16 工作，不需要 pad
4. **V1 权重 DMA 更碎片但更并行**：bd1=2048 将 hidden dim 切 3 片，更多 DMA ops 但单个更小、pipeline 更紧凑

---

## 9. 为什么大 token V2 优于 V1

### 9.1 MXU 利用率差异

| 指标 | V1 (16384) | V2 (16384) |
|---|---|---|
| 单次 GEMM (gate/up) | `(64, 2048) × (2048, 1024)` | `(128, 3072) × (3072, 1024)` |
| K 维大小 | 2048 | 3072 (1.5×) |
| M 维大小 | 64 | 128 (2×) |
| GEMM FLOP per dot | 268 MFLOP | 805 MFLOP (3×) |
| Token 重复加载次数 | 3× (num_bd1) | **1×** |
| DMA ops per expert per bf | 36 (W1+W2+W3 × bd × tp) | 6 (W1+W3+W2 × tp) |

V2 的单次 dot 是 V1 的 3×，这直接提升 MXU pipeline 利用率（amortize startup cost over more compute）。

### 9.2 Token load 节省

| Kernel | Token load per expert (bts=128, bd1=2048) | 计算 |
|---|---|---|
| V1 | 3 × 128 × 2048 × 2 bytes = 1.5 MB | num_bd1 × bts × bd1c × sizeof |
| V2 | 1 × 128 × 6144 × 2 bytes = 1.5 MB | 1 × bts × hidden_size × sizeof |

Token load 总 byte 相同（3×2048 = 1×6144），但 V2 只发 1 次 DMA（1.5 MB burst），V1 发 3 次（每次 0.5 MB）。DMA startup 开销 V2 更小。

### 9.3 Weight DMA pattern

V2 的 weight tile 更大（`3072×1024` vs `2048×1024`），但 DMA 次数更少。虽然单次 DMA 更大增加了 latency，double-buffer 策略让 DMA 与 compute overlap，净效果是正面的。

---

## 10. 性能甜蜜点分析

### 10.1 V2 优势区间

**V2 明显优于 V1 的条件：**
1. `tokens ≥ 4096`（per device local_nt ≥ 128）
2. `bt ≥ 64`（GEMM M 维足够大）
3. `bf ≥ 512`（intermediate dim tile 足够大，DMA burst 效率高）
4. `num_bts_tiles = 1`（避免权重重复加载）

**最佳工作点：** 8192-16384 tokens, ep=32, bt=128, bf=1024, btc=128

### 10.2 V2 劣势区间

**V2 不如 V1 的条件：**
1. `tokens ≤ 512`（固定 overhead 占比 > 80%）
2. `local_nt < 8`（需要 padding，计算浪费）
3. `btc ≤ 16`（GEMM M 维太小，MXU 利用率 < 5%）

**最差工作点：** 64-128 tokens, ep=32（padding 4-8× overhead + 固定开销主导）

### 10.3 Crossover 分析

根据实测数据，V2 从 ~2000 tokens 开始赶上 V1（线性外推 8192 和 512 的数据）。精确 crossover 需要补充 1024/2048/4096 的实测。

```
Tokens:  64   128   256   512   1024?  2048?  4096?  8192    16384
V2/V1:   1.18  1.24  1.11  1.14  ~1.05? ~0.95? ~0.85? 0.74   0.69
         ───── V2 slower ──────  ─cross─  ───── V2 faster ──────
```

### 10.4 生产场景映射

| 场景 | 典型 num_tokens | V2 表现 | 建议 |
|---|---|---|---|
| Prefill (chunked, dp=8) | 8192-16384 | ✅ V2 **25-31% faster** | 用 V2 |
| Decode (max-running=128) | 128 | ❌ V2 **24% slower** | 用 V1 |
| Decode (max-running=512) | 512 | ❌ V2 **14% slower** | 用 V1 |
| Mixed batch (spec decode) | 1024-4096 | 🔀 接近 crossover | 需实测 |

---

## 11. 优化方向（decode 性能提升路线）

### 11.1 短期（kernel 级修改）

| 优化 | 预期收益 | 复杂度 | 机制 |
|---|---|---|---|
| **Flat gather buffer** | ~150μs | 中 | 替换 128 次 tiny DMA 为 1 次 1.5MB burst DMA |
| **去掉多余 sync_barrier** | ~50μs | 低 | 分析 batch scatter 路径是否真需要 2 个 barrier |
| **Metadata DMA 优化** | ~10-15μs | 低 | 精简 staging buffer 大小，减少 VMEM→SMEM copy |

### 11.2 中期（架构级改动）

| 优化 | 预期收益 | 复杂度 | 机制 |
|---|---|---|---|
| **Decode-specific 代码路径** | ~100μs | 高 | 当 bt 小时跳过 pipelined 路径，用 simplified 单 bts direct 路径 |
| **V1/V2 dynamic dispatch** | 最优 | 低 | `if num_tokens <= threshold: use_v1() else: use_v2()` |
| **Reduce JAX metadata overhead** | ~100μs | 中 | In-kernel metadata（避免 JAX lax.all_gather 的 XLA dispatch 开销） |

### 11.3 验证方法

1. **消融实验**：用 `FUSED_MOE_BENCHMARK_DISABLE_*` flags 逐 stage 测 wall time
2. **LLO dump**：用 `LIBTPU_INIT_ARGS` 导出 VMEM allocation + VREG spill 数据
3. **Trace timing**：`jax.profiler.trace()` 抓 device_duration_ps 与 wall time 对比
4. **Flat gather PoC**：实现 plan 文件 `steady-launching-simon.md` 中的方案

---

## 12. 待补充数据

| 数据 | 工具 | 状态 |
|---|---|---|
| V2 decode 消融实验 (per-stage wall time) | ablation flags + 16-chip pod | 待执行 |
| LLO dump: VMEM 占用 + VREG spill | LIBTPU_INIT_ARGS | 待执行 |
| 1024/2048/4096 tokens 精确 crossover | bench_compare.py | 待执行 |
| xprof: HBM/MXU/ICI 利用率 | jax.profiler.trace() | 待执行 |
| acc_and_store_output 实际 DMA 时间 | LLO source annotation | 待执行 |
