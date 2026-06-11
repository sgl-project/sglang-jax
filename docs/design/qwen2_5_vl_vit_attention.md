# Qwen2.5-VL ViT 块对角注意力设计说明

> 状态:bugfix 已落在分支 `sglang-jax-fix-dp-mm`(commit `9c711233`)。
> 范围:仅 `python/sgl_jax/srt/multimodal/models/qwen2_5VL/qwen2_5_vit.py`。
> 本文记录该 bug、修复方案与遗留权衡,避免日后被悄悄回退。

## 1. 动机

Qwen2.5-VL 的视觉 Transformer(ViT)交错使用两类注意力层:

- **全注意力层**(`fullatt_block_indexes`):注意力被限制在单个图像帧内。与
  HuggingFace 一致,逐帧边界为
  `cu_seqlens = repeat_interleave(h*w, t).cumsum()` —— 即每个 `(h*w)` patch 块是
  一个独立的注意力分组。
- **窗口层**(32 层里的另外 28 层):注意力被限制在单个空间窗口内。token 先按
  `window_index` 重排,使每个窗口的 token 连续,再用 `cu_window_seqlens` 标记逐
  窗口边界。

`cu_seqlens` 和 `cu_window_seqlens` 早就算好并传进了注意力调用,**但只被当作"非空"
真值标志使用,从未真正当成 mask**。因此对**每一张**图,注意力都是错的 —— 哪怕只是
一张正常尺寸的单图,因为一张图本身就横跨多个窗口。

这是一个**静默正确性 bug**:不崩、不报错,只是悄悄降低图像理解质量,属于最难发现的
那一类。

## 2. Bug 表现(修复前)

| 层类型 | 本应的注意力 | 代码实际做的 |
|--------|--------------|--------------|
| 全注意力 | 按**帧**块对角 | **完全没有 mask**,对整段拼接 patch 序列做 full attention → 不同图/帧的 patch 互相 attend |
| 窗口 | 按**窗口**块对角 | 在重排后的序列上用一维滑动距离 mask `abs(i-j) > window` |

这个一维距离 mask 同时在两个方向上出错:

1. 让**跨**窗口边界的 token 互相 attend(重排后它们的位置可能相距小于 `window`)。
2. **错误挡掉**同一窗口内、但位置相距超过 `window` 的 token。

## 3. 修复方案:段号块对角 mask

一个 token 的段号 = 它已"越过"的累积边界数量;两个 token 仅当段号相同才能互相 attend。
对 `cu_seqlens = [0, b1, ..., T]`:

```python
pos = jnp.arange(T)
seg = jnp.sum(pos[:, None] >= cu_seqlens[None, 1:], axis=1)  # 每个 token 的段号
attn_mask = seg[:, None] == seg[None, :]                      # True = 允许
```

- 全注意力层传 `cu_seqlens`(按帧分块),窗口层传 `cu_window_seqlens`(按窗口分块)。
  两类层现在唯一的区别就是**传哪个累积数组** —— `use_fullattn` / `window_size`
  这套 dead plumbing 已删除。
- `vision_attention` 现在接收 `attn_mask: [T,T] bool`,不再接收标量 `window_size`。
- **JIT 友好**(静态形状)且**对零宽段鲁棒**:当某个 grid 维恰好是窗口大小的整数倍
  时,window padding 会产生重复边界(零宽段),它只会跳过一个段号,分组不受影响。已
  用 numpy 数值复验:块对角分组正确、对重复边界鲁棒、单块退化情形正常。

### 为什么重排能保持一致

`hidden_states` 与 `rotary_pos_emb` 只按 `window_index` 重排一次,且窗口不跨帧。因此
全注意力层的逐帧块在重排后仍然连续,块对角构造对两类层都成立。语义与 HuggingFace /
上游参考实现一致。

## 4. 遗留权衡(尚未覆盖)

**GPU 上丢失了 flash attention。** `flash_mha` 无法表达逐图 / 逐窗口的块对角 mask,所以
只要提供了 mask(本模型现在每次都提供),注意力就走**native** 路径:显式构造
`[B,N,T,T]` 分数矩阵 + `[T,T]` mask,即 O(T²) 的时间与显存。

- flash 快路径只为无 mask 的情形保留,本模型已不再命中。
- 理由:视觉编码只在 prefill 阶段跑一次,不在 decode 热路径上。
- **风险:** 对高分辨率或多图 prefill,这不仅更慢 —— 物化的 O(T²) 缓冲可能在 flash
  attention 不会爆的地方**OOM**。这是 GPU 上服务大图时第一个要盯的点。后续改进方向是
  直接消费 `cu_seqlens` 的 varlen / 块对角 flash kernel。

## 5. 影响范围:仅 Qwen2.5-VL

其它视觉 / 多模态编码器已排查,**均不受影响**:

- `qwen3_omni_moe/vision_encoder.py` —— `_create_attention_mask` 已正确构造逐帧块对角
  mask,且**没有窗口层**。
- `encoders/clip.py` —— 单图 full/causal 注意力;无多图拼接、无窗口。
- `t5`、`wan`、`dits`、`vaes` —— 文本 / 扩散,注意力语义不同。

## 6. 后续事项

- 加一条 reviewer 检查:新的 VL ViT 落地时,确认 `cu_seqlens` 这类边界**确实被当成
  mask 使用**(全注意力层逐帧块对角;若有窗口层则逐窗口块对角),而不只是当真值标志。
- 若 GPU 服务大图成为目标,用块对角 flash kernel 重新审视这条 O(T²) 路径。
- 当等价修复在上游 `main` 落地后,本文可以下线。
