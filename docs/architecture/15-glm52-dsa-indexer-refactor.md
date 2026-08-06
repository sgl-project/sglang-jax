# GLM-5.2 DSA Indexer / Attention 边界重构

## 目标

将 TPU exact DSA 的模块边界对齐 GPU 实现：一个 Indexer 拥有 projection、index key 表示准备、index cache 写入和 top-k，Attention 只消费最终的 `topk_indices`。

```text
model layer
  ├─ full Indexer
  │    projection → write index cache → score/top-k
  │                                      │
  ├─ shared Indexer ── reuse previous ────┘
  │                                      │
  └─ Attention ← finalized topk_indices ─┘
         ├─ dense MLA（模型定义的 dense layer）
         └─ exact sparse MLA
```

本次只调整所有权和数据流，不改变 GLM-5.2 的 Indexer score 定义、IndexShare 分组、前三层 dense 约束或 JAX cache 的函数式更新方式。page sparse 是过渡路径，不作为新接口的设计中心。

## 设计原则

### 1. Indexer 是完整的模型组件

[`GlmDsaIndexer`](../../python/sgl_jax/srt/models/glm5_moe.py) 持有 Indexer 权重和静态配置，其公开调用一次完成：

1. 从 `hidden_states` 和 `q_compressed` 生成 `q_idx`、`k_idx`、`idx_weights`；
2. 获取当前 full Indexer 对应的 index cache slot；
3. 先把本轮 `k_idx` 写入 index cache；
4. 使用更新后的 cache 计算当前 query 的 token top-k；
5. 返回更新后的 `index_cache` 和可选的 `topk_indices`。

`q_compressed` 既参与主 Attention 的 Q 投影，也参与 Indexer query 投影。当前 token 的 `k_idx` 在 top-k 之前写入 cache，因此它可以进入当前 token 的候选集合；causal mask 仍限制每个 query 只能选择不晚于自身的位置。

底层 JAX Indexer 操作保留在 [`dsa_indexer_ops.py`](../../python/sgl_jax/srt/layers/attention/dsa_indexer_ops.py) 中作为函数式 helper。通用 paged-cache scatter 位于 [`dsa_cache_ops.py`](../../python/sgl_jax/srt/layers/attention/dsa_cache_ops.py)，供两套 cache 更新复用。它们都不是第二个 runtime owner：不持有 projection 权重，不判断 full/shared，也不调度 Attention。

### 2. Attention 只消费最终选择

[`DSASparseAttentionBackend`](../../python/sgl_jax/srt/layers/attention/dsa_sparse_backend.py) 的 DSA 输入只有：

```python
topk_indices: jax.Array | None
```

它负责标准 MLA KV cache 写入以及 dense/exact sparse kernel 调用，不再访问 index cache，也不计算 Indexer score 或 top-k。

- `topk_indices is None`：走 dense MLA；
- exact 模式且有 `topk_indices`：将 sequence-local token 位置映射成物理 cache slot，执行 exact sparse MLA；
- page 模式仅保留最低限度兼容，page table 在旧 kernel 边界内部由 token top-k 转换，不进入模型层公共数据流。

### 3. dense/full/shared 是两组正交属性

`layer_id < index_skip_topk_offset` 是 GLM-5.2 模型的 dense attention 约束，当前为前三层。它不属于 TPU backend 特例。

`indexer_types` 则描述 IndexShare：

- `full`：拥有 Indexer 权重和独立 index cache slot；
- `shared`：不创建、不运行 Indexer，复用最近一个 full layer 产生的 `topk_indices`。

因此 dense full layer 仍可能提前产生 top-k：如果它的 IndexShare 分组延伸到后续 sparse layer，该 top-k 会作为该组的选择结果传下去，但 dense layer 自己仍执行 dense Attention。若整个分组都落在 dense prefix 内，则只更新 index cache，不产生无用 top-k。

模型循环是 IndexShare 的唯一编排位置：遇到 full layer 时用它的 fresh top-k 开启新分组，shared layer 保持已有值。Attention backend 不再推断或保存 IndexShare 状态。

## exact 数据流

以一个 full layer 为例：

```text
hidden_states
  ├─ q_a_proj + norm ───────────────→ q_compressed
  │                                      ├─ Indexer q projection ─→ q_idx
  │                                      └─ main q_b_proj ────────→ MLA Q
  ├─ Indexer k projection ───────────→ k_idx
  └─ Indexer weights projection ─────→ idx_weights

k_idx + old index cache
  └─ write current token first ──────→ updated index cache
                                          │
q_idx + idx_weights + updated cache ──────┘
  └─ score + causal mask + top-k ─────→ topk_indices

MLA Q + current latent KV + topk_indices
  └─ DSASparseAttentionBackend
       ├─ write current latent KV
       ├─ logical token → physical slot
       └─ exact sparse MLA
```

这里有两套独立的持久 cache：

- Indexer cache 保存 `k_idx`，供 score/top-k 使用；
- MLA KV cache 保存 latent KV 和 RoPE key，供 Attention 使用。

由于 JAX 使用函数式更新，两套更新后的 cache 都必须由模型返回给 memory pool；临时 `topk_indices` 只在线程内传递，不属于 cache，也不再通过组合类型包装。

## 按 layer 的 hybrid KV pool

GLM-5.2 的 dense prefix 使用 MLA kernel，后续层使用 DSA kernel。两者当前恰好消费相同的 latent-KV layout，但这个一致性不再作为 pool 接口的前提。

`LayerwiseHybridKVPool` 将一套全局 token/page 地址空间映射到多个独立子池：

```text
global layer_id
  ├─ dense prefix ─→ pool "mla" ─→ MLATokenToKVPool（当前）
  └─ sparse layers → pool "dsa" ─→ MLATokenToKVPool（当前）
                                      ↑
                              future DSA-specific pool
```

它负责：

- 将全局 `layer_id` 转换为 `(pool_name, local_layer_id)`；
- 将模型按全局 layer 顺序返回的函数式 KV writeback 重新按子池分组；
- 为每层暴露对应的 buffer 和 sharding；
- 让 Indexer cache 明确归属于 DSA 子池，而不要求 dense MLA 子池理解 Indexer 状态。

两个子池当前仍都是 `MLATokenToKVPool`，所以 tensor shape、数值和总 KV 容量不变。后续接入不同 layout 的 DSA kernel 时，只需要替换 `dsa` 子池的实现；模型循环、全局 layer id、allocator 和 IndexShare 数据流保持不变。

## 代码所有权

| 位置 | 职责 |
| --- | --- |
| [`GlmDsaLayerPlan`](../../python/sgl_jax/srt/models/glm5_moe.py) | 一次性固化每层 dense/full/shared、cache slot 与是否需要产出 top-k |
| [`GlmDsaIndexer`](../../python/sgl_jax/srt/models/glm5_moe.py) | projection、cache slot 所有权、调用 index cache write/top-k |
| [`update_index_cache_and_select`](../../python/sgl_jax/srt/layers/attention/dsa_indexer_ops.py) | 单一 `shard_map` 内的函数式 cache write 和 token top-k |
| [`select_indexer_topk`](../../python/sgl_jax/srt/kernels/dsa/topk.py) | 将 `approx`、`exact_lax`、`radix` 统一成有序的 `(values, indices)` ABI，并处理 radix 输入对齐 |
| [`radix_topk_pallas`](../../python/sgl_jax/srt/kernels/radix_topk/__init__.py) | 稳定的公共 kernel 入口；当前导出 versioned `v1` SparseCore 实现 |
| [`tuned_configs`](../../python/sgl_jax/srt/kernels/radix_topk/tuned_configs.py) | 按 TPU 设备和 `(score_size, topk)` 查询 radix 静态参数；未命中时回退安全默认值 |
| [`Glm5Attention`](../../python/sgl_jax/srt/models/glm5_moe.py) | 按 layer plan 调用 full Indexer或消费 shared top-k，并调用 Attention |
| [`Glm5Model`](../../python/sgl_jax/srt/models/glm5_moe.py) | 维护最近 full layer 的 `topk_indices`，汇总两类 cache 更新 |
| [`DSASparseAttentionBackend`](../../python/sgl_jax/srt/layers/attention/dsa_sparse_backend.py) | 消费最终 top-k，更新 MLA KV cache，执行 dense/exact sparse MLA |
| [`LayerwiseHybridKVPool`](../../python/sgl_jax/srt/mem_cache/memory_pool.py) | 按全局 layer 路由不同 layout 的 KV 子池并拆分函数式 writeback |

## 保持不变的语义

- score 仍为 `sum_h(relu(q_h @ k) * weight_h)`；
- `index_topk` 仍由模型 config 控制；
- `dsa_topk_impl` 只选择 top-k backend，模型层不感知 radix 的窗口和 digit 参数；
- 无效和 padding 位置仍输出 `-1`；
- 当前 token 的 index key 和 latent KV 都在本轮 attention 前完成函数式写入；
- full layer 才运行新的 Indexer，shared layer 复用最近 full layer 的结果；
- `index_skip_topk_offset` 明确控制模型的 dense prefix；
- q/k/index cache 的 sharding 和 `jax.shard_map` 边界不引入 host/device copy；
- JAX memory pool 继续接收并提交每层返回的新 cache value。

## 本轮不处理

- decode kernel 受限导致的 page 粒度优化；
- GPU 的短上下文 K-only Indexer 快路径；
- 以 `topk_pages` 为公共模型接口的 page-specific 编排；
- cache 原地 mutation（JAX 语义不支持）。

## 验收清单

- [x] 一个公开 Indexer 调用拥有 projection、index cache 写入和 top-k。
- [x] sparse Attention 不依赖 `q_idx`、`k_idx`、`idx_weights` 或 index cache。
- [x] exact 公共数据流只传递 `topk_indices`。
- [x] full/shared 由模型层编排，shared layer 不实例化 Indexer。
- [x] dense prefix 是显式模型 layer plan，不由 backend 推断。
- [x] MLA KV cache、index cache 和临时 top-k 分离返回与传递。
- [x] dense MLA 与 sparse DSA layer 通过独立子池路由，当前保持相同 KV layout。
- [ ] 在具备 TPU/JAX 运行环境后验证 full/shared/dense boundary 的逐元素结果。
