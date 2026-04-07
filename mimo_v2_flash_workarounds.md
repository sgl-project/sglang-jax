# MiMo-V2-Flash TP=16 Workarounds

**Model**: MiMo-V2-Flash (256 experts, FP8, block_size=128)
**Cluster**: TPU v6e-16 (4 nodes × 4 chips, TP=16, EP=16)

---

## Workaround 1: k/v_proj pre-dequant to bf16 (blockwise kernel n_out < 256)

### 问题

TPU MXU 最小 output tile 宽度 = 256（`compute_tile_n = 256 * n_lane_multiplier`）。
TP=16 下 k_proj/v_proj 的 per-device n_out 小于该阈值：

| Projection | 全局 n_out | TP=16 local n_out | 是否可走 blockwise kernel |
|:---:|:---:|:---:|:---:|
| q_proj | 12288 | 768 | ✓ (768 ≥ 256) |
| k_proj | 3072 | 192 | ✗ (192 < 256) |
| v_proj | 2048 | 128 | ✗ (128 < 256) |

`_floor_multiple(192, 256) = 0`，kernel 无法构建有效 tile → 输出错误。

### 解法

模型加载完权重后，`finalize_quantized_layers()` 遍历所有 `QuantizedLinear` 层，
对 `should_use_blockwise_kernel() == False` 的层一次性 dequant 为 bf16：

```
fp8 weight × block_scale → bf16 weight (一次性)
后续 forward: lax.dot_general(x, w_bf16) — 纯 bf16 matmul
```

- **代价**: bf16 = 2× fp8 内存，但 k/v_proj 参数量很小（48 层 × 2 = 96 层），可接受
- **收益**: 消除每步 forward 的 scale expand + multiply；代码路径更简洁
- **涉及文件**:
  - `layers/linear.py`: `QuantizedLinear.maybe_pre_dequant()` + `_pre_dequanted` fast path
  - `quantization_utils.py`: `finalize_quantized_layers()` model tree walker
  - `model_loader/loader.py`, `model_runner.py`: 加载后调用 `finalize_quantized_layers()`
  - `blockwise_utils.py`: `should_use_blockwise_kernel()` 使用 `_MIN_MXU_TILE_N = 256` 阈值

### 验证

```
Pre-dequanted: 48 × k_proj + 48 × v_proj = 96 layers
q_proj: 0 pre-dequanted (stays quantized → blockwise kernel)
推理结果正确 (数学/中文/代码生成)
```

---

## Workaround 2: Split KV cache head replication (1 → 2 for bf16 packing)

### 问题

Pallas `ragged_paged_attention` kernel 通过 `bitcast(uint32)` + strided load
一次取 2 个 bf16 值，要求 KV head 数是 packing factor 的倍数：

```python
# util.py
def get_dtype_packing(dtype):
    return 32 // itemsize_bits(dtype)  # bf16 → 2, int8 → 4, float32 → 1
```

kernel 内部 `strided_load_bkv` 有硬性 assert：
```python
assert start % kv_packing == 0
assert step % kv_packing == 0
```

TP=16 下 local_kv_heads = 16/16 = 1，不满足 packing=2 的约束。

### 解法

KV cache pool 写入时用 `jnp.repeat` 将 1 head 复制为 2 heads 存储（零拷贝优化），
forward 时 cache 已满足 packing 对齐，只需 tile 当前步的新 Q/K/V token（1 个 token，代价极小）。

- **涉及文件**: `flashattention_backend.py` (`_call_split`), `memory_pool.py` (`SplitMHATokenToKVPool`)

### 代价：KV cache 内存 2×

```
原始:   1 local KV head × (K:192 + V:128) = 320 bf16/token/layer
零拷贝: 2 local KV heads × (K:192 + V:128) = 640 bf16/token/layer
```

48 层 × 2× = **KV cache 总量翻倍**。直接影响 `max_total_tokens`，
高并发下 radix cache 容量减半，eviction 更频繁，cache hit ratio 降低。

### TP 上限约束（通用）

无额外 KV cache 开销的条件：`local_kv_heads ≥ packing`，即 `TP ≤ num_kv_heads / 2`（bf16）。

| 全局 kv_heads | 最大 TP（bf16 无额外开销） | 16 chips 配置 |
|:---:|:---:|:---:|
| 16 | 8 | DP=2, TP=8 |
| 8 | 4 | DP=4, TP=4 |
| 4 | 2 | DP=8, TP=2 |
| 2 | 1 | 无法 TP |

MiMo-V2-Flash 每层 kv_heads 可能不同，需按最小 kv_heads 定 TP 上限。

---

## 未来优化方向

### DP+TP Hybrid
- Attention: DP×TP 使 per-device local_kv_heads ≥ 2 且 local n_out ≥ 256
- MoE: EP=16 充分利用 expert parallelism
- 同时解决 Workaround 1（n_out 太小）和 Workaround 2（head 太少）
- 需要调研 sgl-jax 框架对 hybrid parallelism 的支持

### 修改 Pallas kernel（中等改动量，高调试成本）
- 目标：让 bf16 kernel 原生支持 `num_kv_heads < packing` 的情况（如 1 head + bf16 packing=2）
- 彻底消除 head replication 的 2× 内存开销
- 已有 `kv_packing == 1` 代码路径（为 float32 设计），可作为基础
- 约 6-8 处修改点：`get_dtype_packing`、`prepare_kv`/`merge_kv` reshape、
  `prepare_inputs` head 对齐计算、VMEM 预算、Grid/scratch buffer partition
- `strided_load_bkv` 核心函数已有 `kv_packing==1` 分支，无需改动
- 性能预期：1 head 场景下 packing=2 无收益（无第二个 head 可打包），packing=1 等价或略优
- 风险：Pallas/Mosaic 编译器对 bf16 + `kv_packing=1` 组合可能未覆盖测试，需 TPU 实测
- 建议：先在 TPU 上用最小 repro 测试 bf16 + `kv_packing=1` 路径的编译和计算正确性
