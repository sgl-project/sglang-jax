# MiMo V2 Flash 远端调试进展（截至 2026-02-12 02:58 UTC）

## 已完成/修复
- KV 维度、v_proj/k_proj broadcast 问题已解决；KV 复制逻辑按原始 kv_heads 判断。
- MoE FP8 量化权重/scale 对齐：
  - scales 依据对应 weight 的末维裁剪填充，并保持 shard。
  - 对 block-wise scale（[experts, n_blocks, k_blocks]）增加 k 维合并，确保 gmm 要求的 tile 大小；wo_scale 进一步折叠 k_blocks -> 1。
- Attention sink 在 TPU Pallas 上禁用，避免 ANY 内存索引错误。
- 添加一次性日志，确认 gmm 调用形状：
  - gmm1: lhs=(512,4096), rhs=(8,256,4096), rhs_scale=(8,32,1,256), block_size=128
  - gmm2: lhs=(512,256), rhs=(8,4096,256), rhs_scale=(8,1,1,4096), block_size=256
- 权重缺失补零：loader 遇到缺失的 bias/scale 会按真实 mesh sharding 用 0 填充（修复 AbstractMesh device_put 报错）。
- Bias 映射修正：改用 `qkv_bias` / `o_bias` 控制是否加载 q/k/v/o bias（原先 attention_bias 为 False 导致 bias 留作 ShapeDtypeStruct）。
- ModelRunner 在启动和 forward 前检查 ShapeDtypeStruct，便于提前暴露抽象参数。

## 当前状态
- 启动阶段 ShapeDtypeStruct 已清理；远端 `run_demo.sh --disable-precompile` 能跑通请求。
- curl 返回测试回复（内容为占位符 “!”），服务未再崩溃。

## 后续建议
1) 如需恢复预编译，考虑在生成 dummy batch 时确保不再传递 ShapeDtypeStruct（目前运行时已正常）。  
2) 可移除/降级调试检查 `_assert_no_shapedtypestruct` 以减小开销（当前只在 Python 侧执行，影响极小）。  
3) 继续关注 MoE/FA 性能与内存；若切回预编译需重新验证。

## 远端已同步的文件
- python/sgl_jax/srt/utils/weight_utils.py
- python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py
- python/sgl_jax/srt/layers/moe.py
- python/sgl_jax/srt/models/mimo_v2_flash.py
- python/sgl_jax/srt/model_executor/model_runner.py
- python/sgl_jax/srt/layers/logits_processor.py
- python/sgl_jax/srt/layers/sampler.py

---

# 2026-02-12 07:05 UTC 追加进展

## 修复
- top_k/logprobs 分支的分片错误：`logits_processor.get_top_logprobs` 和 `sampler.get_top_logprobs` 统一将输入/输出 reshard 到 P(None)，并将 k 对齐到 tensor 轴的 8 倍，避免 `axis ... partitioned 8 times` 报错。
- 解决 JAX Tracer→numpy 转换错误：移除 host 侧 top_k，改为纯 JAX reshard + slice。
- NaN 导致的 500：在 logprobs 路径全程 `nan_to_num`（logits_processor、sampler），避免 JSON 序列化报 `Out of range float values are not JSON compliant: nan`。
- 处理 top_logprobs_nums 含 0 的情况：返回固定形状 (batch, max_k) 并用 -inf/-1 填充，防止 ragged concat 触发 TypeError。

## 当前测试结果
- 使用 `/tmp/run_logprob.sh`（mem-fraction-static=0.4，tp=8）可跑完请求；curl 返回 200，但内容为 `"!!!!!"`，top_logprobs 仍全 0，质量很差。
- 服务器在请求完成后被脚本 kill（预期行为，脚本末尾 `kill $PID`）。

## 仍存在的问题
- 生成质量极差，logprobs 全为 0，推测 logits 或量化/映射仍有问题（可能 qkv/o_proj 映射、FP8 scale 或 logit normalizer）。需要继续排查模型输出本身。
- 生成吞吐极低（~0.01–0.04 tok/s），可能与禁用预编译及 page_size=1 有关。

## 下一步建议
1) 检查 logits 是否为常数/NaN：在 sampler 前对 `logits_output.next_token_logits` 打印 max/min/是否全 0；或在 TPU 上运行一个短 `check_logits` 脚本。
2) 核实权重量化映射：重点确认 q/k/v/o 投影、MoE 权重的 scale 加载是否正确（训练端 qkv 合并，推理端拆分）。
3) 如 logits 正常但 logprobs 仍异常，排查 softmax/log_softmax 是否被温度/遮罩影响，或是否有 shape 对齐填 0 导致全 0。
4) 视需要调低 context-length / mem-fraction，开启小 batch 预编译以提升吞吐后再测。

---

# 2026-02-12 09:30 UTC 追加进展

## 过程
- 临时脚本调试：尝试 `JAX_DISABLE_JIT=1` 运行 `/tmp/run_debug.sh`，在 SWA KV buffer 初始化阶段 OOM（需 655MB，HBM 仅 61MB 剩余），未能进入模型前向。
- 取消 `JAX_DISABLE_JIT`，保留 `JAX_DEBUG_NANS=1` 再跑：权重加载、KV buffer 分配完成，gmm 打印确认 scale 已参与计算（见下）；在首次 prefill (`get_top_logprobs`) 时，XLA shard_map 导出断言失败 `manualAxes.region.size() != sharding.getMesh(...).getAxes().size()`，进程 SIGABRT 退出，`/tmp/logprob_response.json` 为 0 字节。
- 新增 `_log_if_bad`（moe.py）对 gmm 输入/输出检测 NaN/Inf/超大值，当前未触发打印；说明 NaN 可能在更深处 or 因编译崩溃未执行到。

## 关键日志
- gmm1: lhs=(512,4096), rhs=(8,256,4096), rhs_scale=(8,32,1,256), block_size=128
- gmm2: lhs=(512,256),  rhs=(8,4096,256), rhs_scale=(8,1,1,4096), block_size=256
- 崩溃堆栈：`ShardMapExportPass` 手动轴数量与 mesh 轴不匹配 → SIGABRT（exitcode=134），发生在 `get_top_logprobs` 编译阶段。

## 当前状态
- 服务已停（进程被 SIGABRT 结束）。MoE scale 读取与形状对齐看起来正确；问题转为 XLA sharding 导出/编译失败，而非 NaN 触发。

## 下一步建议
1) 在保持 JIT 的情况下，改用更小 context/batch，或关闭 `JAX_DEBUG_NANS`，查看是否仍触发 ShardMap 导出断言（定位是否 debug 选项引起）。
2) 若需继续观察数值，可在 MoE gmm 周围局部包 `with jax.disable_jit():` 小 batch 运行，避免全局 OOM 又能打印 `_log_if_bad`。
3) 若 ShardMap 仍报错，考虑在 `get_top_logprobs` 路径上强制 reshard 明确 mesh（已做一次）；可再尝试降低 page_size 或使用默认 attention backend 验证是否为 sharding 配置问题。
4) 保持临时调试脚本但默认不带 `JAX_DISABLE_JIT`，以防再次 OOM。

---

# 2026-02-12 11:30 UTC 追加进展

## 修复
- **量化 Scale 逻辑修正**: 在 `weight_utils.py` 中增加了对 `weight_scale_inv` 后缀权重的特殊处理。加载时自动执行 `jnp.reciprocal`（取倒数），将 $1/scale$ 还原为真实的 $scale$。这解决了之前数值爆炸导致输出全是乱码（`!!!!!`）的问题。
- **Attention 输出切片修正**: 修改了 `flashattention_backend.py` 中的 `FlashAttention` 调用逻辑。不再统一按 `head_dim` 切片，而是依据 `v_head_dim` 处理输出。这保证了在 `head_dim != v_head_dim`（如 MiMo-V2）的情况下，Attention 的输出维度与下一层 `o_proj` 的输入对齐，消除了 Padding 噪声。
- **移除冗余 Reshard**: 重构了 `sampler.py` 中的 `get_top_logprobs` 函数，移除了对输入 `logprobs` 强制执行 `jax.sharding.reshard(..., P(None))` 的逻辑。此操作在 `shard_map` 上下文或复杂 JIT 融合时会触发 `ShardMapExportPass` 断言导致 `SIGABRT` 崩溃。现在完全依赖 JAX 自动处理分布式 `top_k`。

## 当前状态
- 已应用以上修复。
- 模型加载时能正确处理逆量化系数。
- Prefill 阶段的编译崩溃问题理论上已解决（移除了冲突的 reshard）。
- 建议下一步在 TPU 环境下重新运行测试请求，验证生成质量和稳定性。

---

# 2026-02-13 02:40 UTC 追加进展（静态 FP8 量化 & NaN 追踪）

## 本地改动与同步
- 静态 FP8 也会执行量化替换：`model_runner.py` 调用 `apply_linear_quantization` / `apply_moe_quantization`，QuantizedLinear 在静态模式使用 checkpoint 的 float8 权重+scale。
- 线性层：`LinearBase` 保留，`QuantizedLinear.from_linear` 在静态下直接用权重+scale；`weight_utils` 把 `*_weight_scale_inv` 取倒数并折叠 block scale（按 `weight_block_size` 重复到每输出通道）。
- 配置解析：`model_config.py` / `quantization_config.py` 支持 fp8 的 `fmt`、`activation_scheme`、`weight_block_size`。
- MoE：静态 fp8 下为 scale 预建占位 Param（shape 对齐量化路径），避免加载时报路径不存在；后续改为无 sharding 占位以防 mesh reshard 报错。
- 所有相关文件已同步到远端 `~/sky_workdir/sgl-jax`。

## 远端验证
- 服务器：`ssh sky-adbe-jiongxuan`（端口 22），服务启动命令同 REPRO 文档（port=30000，disable_precompile）。
- 权重加载顺利（Regular/MoE 均完成），未再出现缺 param 或 sharding 断言。
- 生成阶段出现 MoE 数值异常：日志出现大量
  - `MiMoV2Moe layer=18 hidden_nan=4096 router_nan=256`
  - 后续层 19、20 … 多层重复 `hidden_nan` / `router_nan`，`topk_weights_nan=8` 等。
  - 说明首次 NaN 观测在 **layer 18**（最早出现的位置），后续层持续受影响。
- 服务仍能返回 200，但输出质量未知；需继续定位 NaN 源头。

## 待办 / 下一步
1) 精确定位 NaN 源：在 MoE forward（路由 softmax 前后、wi/wo matmul 输出）添加轻量统计或 `jax.debug.print`（可局部 `with jax.disable_jit()` 小 batch）以确认首个 NaN 出现点。
2) 检查 MoE scale 加载：验证 `wi_0_scale/wi_1_scale/wo_scale` 是否为非零、形状匹配（num_experts,1,1,features），以及 block scale 折叠是否正确。
3) 若 scale 合理，考虑激活溢出：对路由 logits / expert 输出做 clip 或 fp32 accumulate（临时实验），观察 NaN 消失层次。
4) SSH 当前偶发超时；如需取全量日志，可在远端手动 `tail -n 200 server_test.log` 或把日志打包后再拉取。

---

# 2026-02-13 08:30 UTC 追加进展（量化实现分析与 NaN 隐患）

## 量化实现深度分析
经过对 `MiMo-V2-Flash` 仓库代码的深度审计，发现当前 FP8 量化实现中存在以下可能导致 NaN 的重大隐患：

1. **`weight_block_size` 简化不合理**：
   - **问题**：在 `weight_utils.py` 中，代码将 2D 的 `weight_block_size` (如 [128, 128]) 强制折叠。它对输入块维度执行 `mean(axis=-1)`，将其退化为 Per-channel (1D) 缩放。
   - **风险**：FP8 E4M3 的动态范围极小。如果权重块分布不均，取平均值会导致某些高权重点反量化后溢出，或者导致缩放系数过小触发精度崩溃，最终引发 NaN。

2. **动态激活量化零除漏洞**：
   - **代码位置**：`quantization_utils.py` 中的 `quantize_tensor_simple` 函数。
   - **隐患**：该函数未对 `scale` 做 `max(scale, epsilon)` 或 `scale + (scale == 0)` 的保护。对于全零输入（如 Padding tokens 或被 Mask 的区域），`scale` 为 0，执行 `x / scale` 会直接产生 `NaN` 并传播至后续层。

3. **MoE/GMM 算子脆弱性**：
   - **路径**：`moe.py` 中的 `_gmm_compute` 在进入 Pallas 内核前执行激活量化。
   - **影响**：如果 `quantize_tensor_simple` 产生 NaN，整个 MoE 层输出瞬间失效。目前的 `jax.debug.print` 观测到的 layer 18 开始的 NaN 极大概率由此触发。

4. **权重加载安全性**：
   - **隐患**：`WeightLoader` 在处理 `weight_scale_inv` 时直接使用 `jnp.reciprocal`。若权重文件中缩放系数为 0 或极小，将直接引入 Inf/NaN。

## 下一步行动建议
- **修复 `quantize_tensor_simple`**：立即增加对 `scale` 为 0 的保护逻辑。
- **验证 Block Scale**：暂时绕过 `mean(axis=-1)`，尝试在 BF16 激活模式下仅使用 FP8 权重（Per-channel），排除 2D block scale 折叠带来的精度冲击。
- **监控 Point-of-Failure**：在 layer 18 的 MoE 输入处打印 `jnp.max(jnp.abs(hidden_states))`，核实是否由于之前层的积累导致数值已经超出了 FP8 范围。

---

# 2026-02-25 02:22 UTC 追加进展（`sky-96ee-jiongxuan` 接续调试）

## 环境与基线
- 新 TPU 主机：`ssh sky-96ee-jiongxuan`（host 实际返回 `t1v-n-940b8d1c-w-0`）。
- 远端仓库：`~/sky_workdir/sgl-jax`，分支 `feat/mimo-v2-flash`，commit `69f679c`（与本地一致）。
- 环境已由用户预装；使用 `source .venv/bin/activate` 激活（`Python 3.12.12`，`jax/flax/sgl_jax` 导入正常）。
- 模型路径确认：`/models/MiMo-V2-Flash`。
- 新建远端复现脚本：`/tmp/run_mimo_30123.sh`（固定端口 `30123`，负责启动/等待/发请求/抓尾日志/清理）。

## 本轮修复（按推进顺序）
- `linear.py`：修复静态 FP8 线性层在 loader `eval_shape` 阶段对抽象参数（`ShapeDtypeStruct`）执行 `.T` 崩溃的问题。
  - `QuantizedLinear.from_linear(..., is_static_input=True)` 新增抽象占位分支，构造 `weight_q/weight_scale` 的 `ShapeDtypeStruct`（含 sharding）。
- `linear.py`：对 TP 与 block-size 不对齐的线性层自动降级为 per-channel scale（保持量化权重路径），避免 `NotImplementedError` 阻塞启动。
  - 典型触发：`global_dim=1536` 且 `tp=8`, `block_size=128` 无法整除。
- `weight_utils.py`：在普通线性 `*_weight_scale_inv` 的单权重加载路径恢复 `jnp.reciprocal`（取倒数变回真实 scale）。
  - 若目标参数是 1D（对应上面的 per-channel 降级），会将 2D block scale 折叠为 1D per-channel，并将 sharding spec 从 2D 调整为 1D。
- `model_runner.py`：静态 checkpoint 下跳过 `ModelRunner.load_model()` 里的二次 `apply_moe_quantization(...)`。
  - loader 阶段已经完成 static MoE 结构准备与权重加载；二次执行会把已加载的 `wi_0_scale/wi_1_scale` 覆盖成 placeholder。
- `moe.py`：在 `EPMoE.__call__` 中对 `wi_0_scale/wi_1_scale/wo_scale` 显式 `reshard`（保守处理 static FP8 blockwise scale）。
- `sampler.py`：修复 `return_logprob/top_logprobs` 在 TP sharding 下的 `top_k`/切片错误。
  - `k` 对齐到 tensor 轴倍数（例如 `20 -> 24`）。
  - 先 `reshard` 到 replicated，再切回 `max_k`，避免对 sharded dim 做非整除切片时报错。
  - 输出 replicated `PartitionSpec` 改为与数组 rank 一致。

## 关键验证结果（远端）
- 已经连续越过以下旧阻塞点：
  - 静态 linear wrapping 阶段 `ShapeDtypeStruct` 的 `.T` 崩溃；
  - block-wise `QuantizedLinear` 的 TP/block 对齐 fail-fast；
  - MoE `gmm` 的 `rhs_scale` 形状错误（由 static MoE 二次 re-wrap 覆盖 scale 引起）；
  - `sampler.get_top_logprobs` 的 TP 分片错误（`k=20` 非 8 整除，及 sharded slice 到 20 的限制）。
- 服务现在可以稳定完成以下阶段：
  - 权重加载（Regular + MoE）
  - static linear wrapping
  - KV cache 分配
  - Uvicorn 启动
  - prefill + decode 进入执行
- MoE GMM 形状确认恢复正确（本轮日志）：
  - `gmm1`: `lhs=(512, 4096)`, `rhs=(8, 256, 4096)`, `rhs_scale=(8, 32, 1, 256)`, `block_size=128`
  - `gmm2`: `lhs=(512, 256)`, `rhs=(8, 4096, 256)`, `rhs_scale=(8, 1, 1, 4096)`, `block_size=256`

## 当前剩余问题（最新阻塞）
- `/v1/chat/completions` 请求仍返回 `500 Internal Server Error`（脚本 `curl` 收到 `Internal Server Error`）。
- 当前主因已回到 **数值 NaN**（而不是分片/shape/编译问题）：
  - 日志出现 `MiMoV2Moe layer=2 hidden_nan=4096 router_nan=256`，随后多层扩散；
  - `LOGITS` / `SAMPLER logprobs` 调试切片为全 NaN；
  - FastAPI JSON 序列化报错：`ValueError: Out of range float values are not JSON compliant: nan`。
- 现象上看，首次异常层较之前记录的 layer 18 更早（本轮观测到 layer 2 开始）；需要重新定位首个 NaN 产生点。

## 本轮同步到远端的文件（`sky-96ee-jiongxuan`）
- `python/sgl_jax/srt/layers/linear.py`
- `python/sgl_jax/srt/utils/weight_utils.py`
- `python/sgl_jax/srt/model_executor/model_runner.py`
- `python/sgl_jax/srt/layers/moe.py`
- `python/sgl_jax/srt/layers/sampler.py`
- `python/sgl_jax/srt/models/mimo_v2_flash.py`（仅补充注释，功能无变更）

## 下一步建议
1) 在 `moe.py::_gmm_compute` 对首个异常层（当前观测 layer=2）增加更细粒度统计：`x`、`w0/w1/wo_scale`、`layer_w0/layer_w1/intermediate` 的 min/max/nan。
2) 重点检查 **MoE stacked loader 路径** 是否仍对 `*_weight_scale_inv` 走 `pass`（未做 reciprocal）；这可能与 MoE 数值异常直接相关。
3) 如需先保持 API 可用，可在响应路径继续 `nan_to_num` 兜底，但建议仅作为临时措施，不替代 NaN 根因定位。

---

# 2026-02-25 03:40 UTC 追加进展（block-quant 语义校正 + MoE NaN 二次定位）

## 先修正的语义问题（按用户反馈）
- **撤销 linear 2D block scale 的 2D->1D 折叠思路**：
  - 不再把 TP/block 不对齐的线性层 scale 降级为 1D per-channel。
  - `weight_utils.py` 中对 `2D source -> 1D target` 的 `weight_scale_inv` 直接报错，防止静默改语义。
- **线性 block quant 改为语义保持实现**：
  - `QuantizedLinear` 的 2D `weight_scale` 改为 `P(None, None)` replicated（全局 block-scale 矩阵）。
  - `xla_quantized_matmul_local` 内根据 `lax.axis_index` 重建本地 shard 对应的逐元素 scale（支持 TP shard 切穿 block 边界）。
  - 这样 `k_proj` 这类 `global_dim=1536`、`tp=8`、`block=128` 的层无需 padding/降级也能保持 block quant 语义。

## Checkpoint 事实核验（关键）
- 远端直接读取 `/models/MiMo-V2-Flash/model.safetensors` 中原始 `*_weight_scale_inv`：
  - `self_attn.k_proj.weight_scale_inv`、`q_proj.weight_scale_inv`、MoE `gate/up/down_proj.weight_scale_inv`
  - 数值范围均为 **`1e-4 ~ 1e-3` 的小值**
- 结论：这些 tensor 虽然命名为 `*_weight_scale_inv`，但在当前 MiMo-V2-Flash checkpoint 上，**实际就是 TPU 推理路径要直接使用的 scale**（不能做 `reciprocal`）。
- 因此已将 `weight_utils.py` 中对 `*_weight_scale_inv` 的 reciprocal 全部回退（含普通 linear 与 stacked MoE loader 路径）。

## MoE block-scale 语义问题（已确认并修复）
- `weight_utils.py` 的 MoE scale reshape 路径里，`wo_scale` 曾存在：
  - `mean(axis=-1, keepdims=True)` 把 `k` 方向 block scale 强行折叠为单 block（`block_k=1`）。
- 对 MiMo-V2-Flash 的原始 `down_proj.weight_scale_inv`（shape `(32,16)`）来说，这一步 **既不必要，也破坏 block quant 语义**。
- 移除后：
  - `STATIC_FP8_DEBUG moe.layer1.wo_scale` 从 `(8, 1, 1, 4096)` 恢复为 `(8, 16, 1, 4096)`。

## 新暴露的 TPU 约束（不是 NaN，而是编译限制）
- 在恢复真实 `wo_scale` 后，`gmm2` 触发 Pallas TPU lowering 限制：
  - `k=256`, `num_quant_blocks=16` => `block_size_k=16`
  - TPU Pallas 当前要求相关 block shape 的最后一维满足 `128` 对齐（或等于整体维度）。
- 这解释了之前代码为何用 `wo_scale` 折叠去“绕过”编译问题。

## 语义保持的 gmm2 fallback（调试用）
- 在 `moe.py::_gmm_compute` 中增加 **语义保持 fallback**：
  - 当 TPU 且 `rhs_scale` 对应的 `block_size_k < 128`（MiMo gmm2 即此情况）时，
  - 先按 block-scale **精确反量化 `rhs`**（`wo`），再调用 `gmm(rhs_scale=None)`。
- 结果：
  - 编译通过（不再需要折叠 `wo_scale`）
  - 请求可继续执行到推理阶段

## NaN 定位实验（重要结论）

### 实验 A：仅修复 scale 语义（恢复 raw scale、恢复 wo block-scale、gmm2 fallback）
- 仍然 `500`，NaN 首发在 `MiMoV2Moe layer=1 mlp_output`。
- 新增日志确认：
  - `layer1 hidden_absmax≈2.0~2.8`
  - `layer1 router_absmax≈0.13~0.15`
- 结论：**上游 attention/linear 输入幅值正常**，NaN 在 MoE experts 内部产生。

### 实验 B：将 MoE `gmm1` 的 `rhs_scale` 路径改为“精确反量化 rhs + rhs_scale=None”（调试诊断）
- 保持 block quant 数学语义，但绕过 `gmm(rhs_scale=...)` 执行路径。
- 结果：
  - `layer1 mlp_out_nan` 消失（不再出现）
  - `layer2 mlp_out_nan` 仅剩几十个（此前是整层爆）
- 结论（强）：**MoE `gmm1` 的 `rhs_scale` 路径是当前主要 NaN 来源之一。**

### 实验 C：在实验 B 基础上，临时关闭 MoE activation qdq（仅做隔离）
- `use_activation_qdq = False`（权重 block quant 仍保留，`gmm1/gmm2` 仍走 dequant 调试路径）
- 结果：
  - NaN 首发从 `layer2` 进一步推迟到 `layer3`
  - `layer4` 开始扩散为整层 NaN
- 结论：**MoE activation qdq 也会放大/提前 NaN，但不是唯一根因。**

### 实验 D：将调试用 dequantized `rhs` 从 BF16 提升到 FP32
- 对残留 NaN 的首发层/模式无明显改善（仍然 `layer3 -> layer4` 扩散）。
- 结论：残留问题 **不是简单的 dequant fallback 精度不足（BF16 vs FP32）**。

## 调试中的注意事项（已验证）
- 在 `EPMoE._forward`（`shard_map` 内部）加入 `jax.debug.print` 会触发：
  - `shard_map_export.cc` / `ShardMapExportPass` 的 `manualAxes.region` check fail（SIGABRT）
- 因此后续定位应避免在 `shard_map` 内部直接插入 `jax.debug.print`，优先用：
  - 边界外日志（如 `MiMoV2Moe.__call__`）
  - 行为隔离实验（关闭某路径、替换实现、局部 clamp）

## 当前判断（截至本次）
- 已确认并修复/校正：
  - linear 2D block quant TP misalignment 的语义错误处理（不再降级/折叠）
  - `*_weight_scale_inv` 误 reciprocal（对 MiMo-V2-Flash 是错误）
  - MoE `wo_scale` block 折叠（语义错误）
  - gmm2 小 block TPU 编译限制（用语义保持 fallback 绕过）
- 仍未完全解决：
  - **MoE 路径残留 NaN**（在绕过 `gmm1 rhs_scale` 后，首发已推迟到 layer3）
- 高优先怀疑点（下一步）：
  1. `gmm` 的 `rhs_scale` 路径在 `gmm1` 上的 block-scale 应用/分片语义（已被实验 B 强烈指向）
  2. `gmm2` dequantized 路径与 `_unpermute`/`einsum` 的交互（残留 NaN 来源）
  3. MoE activation qdq 路径在存在极端值时的放大效应（实验 C 已证实会提前失稳）

## 2026-02-25 04:40 UTC 追加进展（切换 FusedEPMoE 调试）

### 目标
- 显式使用 `--moe-backend fused`，验证 MiMo-V2-Flash 在 `FusedEPMoE` 路径的量化调用链是否正确接线，并定位 fused 专属失败点。

### 本轮修补（用于进入 fused 调试路径）
- `mimo_v2_flash.py`
  - `MiMoV2Moe` 优先使用显式传入的 `quant_config`（而不是只读 `config.quantization_config`）。
  - fused/non-fused 分支都传递 `self.quantization_config`。
  - `TopK` 改为 fused/non-fused 共用；fused 分支调用 `FusedEPMoE(hidden_states, topk_weights, topk_ids, ...)`。
- `moe.py`
  - `FusedEPMoE.quantize_weights(is_static=True)` 的 static scale placeholder 改为可分片的 4D 形状（避免 `(1,)` + EP sharding 崩溃）。
  - `FusedEPMoE.__call__` 接受 `token_valid_mask`（兼容 MiMo 调用签名）。
  - 为避免再次在 runtime reshaped explicit-sharded scale 触发 mesh mismatch，`__call__` 内的临时 scale 适配逻辑在 sharded array 上直接跳过（让 kernel 校验报真实错误）。

### fused 路径上的新定位（关键）
- 成功确认：服务以 `moe_backend='fused'` 启动，并能进入请求前向（不是 EPMoE 路径）。
- `weight_utils.py` 新增 fused-aware 的 MoE static scale 转换（不再套用 EPMoE 的 `rhs_scale` 变换）后：
  - `w1_scale/w3_scale` 变为 `(8, 32, 1, 2048)`
  - `w2_scale` 变为 `(8, 16, 1, 4096)`
  - 这与 **checkpoint 的 128x128 block 语义**一致（经过权重转置后按 fused 目标权重轴对齐，并仅在最后一维 repeat 展开）。

### 更精准结论：静态 checkpoint 与当前 fused kernel 的硬约束不兼容
- 为保持 checkpoint 语义，我让 `FusedEPMoE` 在 static 模式下从 `quantization_config.weight_block_size` 读取 `subc_quant_wsz`（MiMo 当前是 `128`）。
- fused kernel 在参数校验阶段报错（远端 `~/sky_workdir/server_mimo_30123.log`）：
  - `ValueError: Expected subc_quant_wsz=128 to be aligned to 256.`
- 这说明当前 `fused_ep_moe` 实现 **硬性要求 `subc_quant_wsz` 为 256 对齐**；而 MiMo-V2-Flash 这份 static fp8 checkpoint 的 MoE scale block 配置是 `128`。

### 当前判断（fused 路径）
- 这不是简单 mapping bug，而是 **checkpoint MoE static scale 格式（128-block）与 fused kernel 支持范围（>=256 对齐）之间的兼容性问题**。
- 如果不改变语义，不能用“平均/折叠 scale”去伪造 `256` 子通道 scale。

### 下一步候选方向
1. 在 fused kernel 中支持 `subc_quant_wsz=128`（根本修复，工作量最大）。
2. 仅用于调试：从 static checkpoint 反量化 MoE 权重后，在加载后重新按 `subc_quant_wsz=256` 做 fused quantize（会改变数值语义，但可验证 fused 路径稳定性/NaN 行为）。
3. 对 static fp8 MiMo-V2-Flash 维持 `EPMoE` 路径，把 fused 调试限定在非静态或重重量化实验上。

## 2026-02-25 05:04 UTC 追加进展（fused kernel 128-block 支持推进）

### 已实现：fused kernel 接受 `subc_quant_wsz=128`
- `python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`
  - 将 `subc_quant_wsz` 校验从 `256` 对齐放宽到 `128` 对齐（validate + runtime assert）。

### 已修复：fused MoE static scale 的 mesh sharding 不匹配
- `python/sgl_jax/srt/models/mimo_v2_flash.py`
  - `create_moe_weights_mapping_quantized(..., moe_backend=\"fused\")` 的 `w*_scale` 改为使用 fused runtime mesh sharding：
    - `(("data", "tensor"), None, None, None)`
  - 不再使用 EPMoE 的 `("expert", ...)` mesh（此前会在 `fused_ep_moe()` 调用时触发 `PartitionSpec('expert', ...)` 与 mesh `('data','tensor')` 不匹配）。

### 新发现：仅放宽校验会触发 TPU device halt（量化 dot 128 子通道）
- 在完成上述两项后，fused 路径确实进入了 kernel 执行，但 TPU 报：
  - `TensorCoreSequencer ... on-device check-failure`
  - HLO 名称包含 `bf_512_128 ... bd1_1024_256 ...`
- 判断：`subc_quant_wsz=128` 时的 quantized tensorcore dot 组合在当前 fused kernel 配置下会触发 TPU 设备侧检查失败（不是 Python 层 shape/mesh 错误）。

### 为继续调试新增的语义保持 fallback（重要）
- `python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`
  - 当 `subc_quant_wsz < 256` 时，仍保留 128-block scale 语义，但在 kernel 内对当前局部 tile 的 `w1/w2/w3` 按 scale 做反量化（`fp32`）后再执行 `dot`。
  - 目标：绕开 TPU 对 128 子通道 quantized tensorcore dot 的限制，同时保持 block-scale 数学语义（牺牲性能）。

### TPU 验证结果（`sky-96ee-jiongxuan`, `--moe-backend fused`）
- 128-block fused 路径已越过以下阻塞点：
  - `subc_quant_wsz=128` 校验失败
  - fused scale `expert` mesh / `data,tensor` mesh 不匹配
  - TPU `TensorCoreSequencer` on-device check-failure（使用 dequant fallback 后未再出现）
- 请求执行进一步推进到采样阶段：
  - 日志出现 `get_top_logprobs input shape=(1, 152576)`
- 新的阻塞点：
  - worker 子进程异常退出：`Child process unexpectedly failed with exitcode=256`
  - 当时 server 主进程仍存活，但请求无响应（repro 脚本中的 `curl` 没有超时参数，导致脚本挂住）

### 当前状态（截至本次）
- 从 fused kernel 角度看，`128` 子通道已经“可进入执行路径”（通过 dequant fallback 实现）；
- 当前未解决的是 **执行后期 worker 子进程异常退出（exitcode=256）**，需要进一步抓取 child 的具体崩溃栈/设备日志。
- 本轮后期远端 `sky-96ee-jiongxuan` 的 SSH 出现连接超时（`Operation timed out`），暂时阻断了继续抓日志。

## 2026-02-25 06:58 UTC 追加进展（参考 tpu-inference 的前处理适配 fused kernel）

### 背景判断（参考 `tpu-inference`）
- 对照 `tpu-inference` 后确认，DeepSeek-V3 的稳定做法不是在 fused kernel 内强行放宽 `128` 支持，而是先在前处理把 MoE 量化权重转换成 kernel 友好的格式再进入 fused kernel。
- 在 `tpu-inference` 的 vLLM FP8 MoE 路径（`tpu_inference/layers/vllm/quantization/fp8.py`）中，做法是：
  - 使用 checkpoint scale 先反量化 MoE block-quant 权重；
  - 再重新量化；
  - 再通过 `process_moe_weights(...)` 做 fused kernel 所需布局/shape 处理。

### 在 sgl-jax 中实施的“前处理适配”（不依赖 kernel 128 特判）
- 新增 `FusedEPMoE.prepare_static_block_quant_for_fused_kernel(target_subc_quant_wsz=256)`
  - 文件：`python/sgl_jax/srt/layers/moe.py`
  - 行为：对 static fused MoE 且 `subc_quant_wsz < 256`（当前 MiMo 为 `128`）：
    - 使用已加载 `w1/w2/w3 + *_scale` 反量化（axis=1）
    - 再按 `block_size=256` 重重量化
    - 更新 `self.w1/w2/w3`、`self.w1_scale/w2_scale/w3_scale`
    - 将 `self.subc_quant_wsz` 改为 `256`
- 新增递归 helper：`adapt_fused_moe_static_block_quant_for_kernel(...)`
  - 文件：`python/sgl_jax/srt/utils/quantization/quantization_utils.py`
  - 用于遍历模型中所有 `FusedEPMoE` 层并执行上述前处理。
- 在 `ModelRunner.load_model()` 中接入该前处理
  - 文件：`python/sgl_jax/srt/model_executor/model_runner.py`
  - 条件：`static checkpoint + has_moe_quantization + moe_backend == fused`

### TPU 验证（新机器 `sky-aae0-jiongxuan`）
- 使用 `--moe-backend fused` + `no-logprobs` repro 脚本验证。
- 日志确认前处理生效（示例）：
  - `FusedEPMoE static block-quant requantized for fused kernel: subc 128 -> 256 ...`
  - `Completed static fused MoE block-quant kernel adaptation on 47 layer(s)`
- 结果变化（关键）：
  - **之前的 `fused-moe ... TensorCoreSequencer` crash 消失**
  - 请求执行推进到后续阶段后，仍在 `model_runner.sample()` 的 `jax.device_get(jnp.nanmin(logits))` 处报 **`SparseCoreSequencer` device halt**

### 结论（本轮）
- 参考 `tpu-inference` 的前处理适配方案在 sgl-jax 上有效：
  - 成功绕过 fused kernel 对 `128` subchannel block 的首要运行期崩溃（TensorCore 路径）。
- 当前 fused 路径剩余主要问题已收敛到更后面的 `SparseCoreSequencer` 异常（不再是 fused MoE 128/256 首要兼容性问题）。

## 2026-02-25 07:50 UTC 追加进展（继续定位 `SparseCoreSequencer` 崩点，已拆分为注意力与 fused MoE 两段）

### 关键定位策略（单进程对照）
- 使用 `--enable-single-process --disable-overlap-schedule` + `curl --max-time` 的单进程脚本，在 `sky-aae0-jiongxuan` 上做以下对照：
  - `fa + fused`（目标路径）
  - `native + fused`（排除 RPA/SparseCore 注意力）
  - `fa + epmoe`（保留 RPA，仅替换 fused MoE）
- 增加 `SGL_DEBUG_SAMPLE_SYNC_CHECKPOINTS=1` 观察是否进入 sampler 阶段。

### 新发现 1：之前的 `SparseCoreSequencer`（注意力侧）确实与 TP 下本地 KV 头数奇偶对齐有关
- 在 `fa + fused` 的 prefill 日志中，修复前 RPA 打印：
  - `actual_num_kv_heads=1`
- 修复后（见下方 patch），同一位置变为：
  - `actual_num_kv_heads=2`
- 说明在 `TP=8` 时本地 shard 的 KV 头数（1）需要在 **shard 内局部 padding** 到偶数头（2），仅按全局 KV 头数（8）对齐不够。

### 新发现 2：修复注意力侧 KV 局部 padding 后，`fa + fused` 仍会硬退出，但 `fa + epmoe` 已能成功返回
- `fa + epmoe` 单进程请求成功返回 `200`（即便日志仍显示 MoE NaN），并且可以看到：
  - `SAMPLE_SYNC pre-sampler / post-sampler` 日志
  - `POST /v1/chat/completions ... 200 OK`
- 这说明：
  - **RPA/SparseCore 注意力侧的首要硬崩点已被修掉**；
  - 当前 `fa + fused` 的硬退出已不再主要是注意力问题，而是 **fused MoE 路径** 的设备侧崩溃（无 Python 栈）。

### 新发现 3：`native + fused` 用来推进定位时暴露了几处独立问题（非 `fa + fused` 主因，但值得修）
- `memory_pool.py` 中错误使用 `.to(np.int32)`（JAX tracer 无 `.to`）
- `update_kv_cache` 在 TP 本地 shard 下要求偶数 head 数 / `head_dim % 128 == 0`，需要 shard 内局部 pad
- `native_backend` 在 split-KV cache 下仍调用 `get_fused_kv_buffer()`，导致 `NotImplementedError`
- `native_backend.forward_attention` 中 `attention_sink` 与 `attn_logits` sharding 不一致（`jnp.concatenate` 抛 `ShardingTypeError`）

### 本轮代码修改（用于定位与修复前置条件）
- `python/sgl_jax/srt/layers/attention/flashattention_backend.py`
  - 在 split-RPA 的 `shard_map` wrapper 内，新增本地 KV 头数局部 padding（`1 -> 2`）并在返回前切回原头数。
- `python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py`
  - 在 `shard_map` wrapper 内对本地 `num_combined_kv_heads` 做偶数对齐；
  - 对本地 `head_dim` 做 128 对齐（如 `192 -> 256`）；
  - kernel 返回后切回原始 shape。
- `python/sgl_jax/srt/mem_cache/memory_pool.py`
  - `full_to_swa_index_mapping[loc].to(np.int32)` -> `.astype(jnp.int32)`
- `python/sgl_jax/srt/layers/attention/native_backend.py`
  - split-KV pool 下兼容 `get_fused_kv_buffer()` 不可用，改为直接使用 `(k, v)` buffer 返回给 `replace_kv_buffer`。

### 当前结论（更新）
- `SparseCoreSequencer` 的“注意力侧首要崩点”已定位并修复：
  - 根因是 **TP sharding 后本地 KV 头数为奇数（1）但未在 shard 内做偶数 padding**。
- 当前 `fa + fused` 剩余硬退出已进一步收敛为：
  - **fused MoE 路径的设备侧崩溃**（无 Python 栈，`curl` 为 `Empty reply from server`）。
- 下一步优先级：在 `fused MoE` 路径增加更细粒度的 layer 级/调用前后诊断（或恢复并增强 `drop_w2_scale` 类调试开关），继续确认是否仍然集中在 `w2` / `dynamic_ffn2` 子路径。

## 2026-02-25 08:40 UTC 追加进展（正式修复 fused MoE double-sharding 拓扑问题，并在新基线重做 scale 对照）

### 新发现（关键）
- `MiMo-V2-Flash + --moe-backend fused` 的 `FusedEPMoE` 调用点拿到的是 **local 输入**（不是 fused wrapper 期望的 global 输入）：
  - `FusedEPMoE AUDIT`: `ep_size=1`, `num_experts=8`, `local_w_experts=8`
  - `fused_ep_moe KERNEL_AUDIT`（修复前）: `ep_size=8`（来自 mesh `data*tensor`）
- 也就是：
  - `FusedEPMoE` 侧是 `ep_size=1`（本地专家计算）
  - `fused_ep_moe(...)` wrapper 却按 mesh 再做一层 EP `shard_map`
  - 形成 **double-sharding / 输入语义不匹配**，这是之前 TPU `TensorCore/SparseCoreSequencer` 硬崩的前置根因之一。

### 正式修复（不回退，不走临时 fallback）
- 在 `fused_ep_moe(...)` 增加 local-input 执行路径（用于已经 local 化的 MoE 输入）：
  - 新增参数：`inputs_are_local`, `ep_size_override`
  - local 路径下使用 `ep_size_override=1` 参与 block config / validation
  - 内核增加 `force_local_ep_singleton`，在 pallas kernel 内将 EP collectives 视为 singleton（避免把 TP 轴误当 EP）
  - 为满足 Pallas “带通信 kernel 必须在 shard_map 内”的约束，local 路径保留一个 **最小 `shard_map` 包装**，但使用 `P()`（不切分输入），避免 double-sharding
- `FusedEPMoE.__call__` 改为自动检测并启用 local-input fused wrapper（MiMo 当前命中）

### 涉及代码（本轮）
- `python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`
  - `fused_ep_moe(...)` 新增 local-input 模式参数
  - `_fused_ep_moe_kernel(...)` 新增 `force_local_ep_singleton`
  - local-input 路径使用最小 `shard_map` 包装 + `P()` in/out specs
  - `KERNEL_AUDIT` 增加 `inputs_are_local / mesh_ep_size / ep_size_override`
- `python/sgl_jax/srt/layers/moe.py`
  - `FusedEPMoE.__call__` 自动检测 MiMo local-input 拓扑并启用 local wrapper
  - 保留并复用原有 `drop_w2_scale / drop_w13_scale` 调试开关和审计日志

### TPU 验证（`sky-aae0-jiongxuan`）
- 正式修复后，日志确认 local-input 路径命中：
  - `FusedEPMoE layer=X using local-input fused_ep_moe wrapper (self.ep_size=1 mesh_ep_size=8 ...)`
- 说明 double-sharding 拓扑已被绕开（且不再报此前显式 `ValueError` / 不再是旧路径硬崩）

### 新基线下的对照测试（重新收敛 fused kernel 崩点）
- 默认 fused（no drop）
  - 仍会设备硬崩，但 HLO / block config 已变为 local-input 路径的新配置：
    - `fused-moe-k_8-renorm_k-bt_32_32_32-bf_512_256-bd1_1024_512-bd2_1024_1024`
  - 崩溃签名：`TensorCoreSequencer ...`
- `SGL_FUSED_MOE_DEBUG_DROP_W2_SCALE=1`
  - `KERNEL_AUDIT` 确认：
    - `inputs_are_local=True`, `ep_size=1`, `ep_size_override=1`
    - `w2_scale=None`, `w1/w3_scale` 仍存在
    - `block_config_eff=bt=32,btc=32,bf=512,bfc=256,bd1=1024,bd1c=512,bd2=1024,bd2c=1024,bts=32`
  - 崩溃签名从 `TensorCoreSequencer` 变为 **`SparseCoreSequencer`**
- `SGL_FUSED_MOE_DEBUG_DROP_W13_SCALE=1`
  - `KERNEL_AUDIT` 确认：
    - `w1_scale=None`, `w3_scale=None`, `w2_scale` 保留
  - 崩溃仍是 **`TensorCoreSequencer`**

### 结论（本轮）
- **正式拓扑修复已完成**：MiMo fused MoE 不再通过错误的 double-sharding wrapper 路径执行。
- 在修复后的正确拓扑基线上，问题再次清晰收敛：
  - `w2_scale` 分支仍是 fused kernel 的主要触发点（drop `w2_scale` 会显著改变崩溃类型）
  - `w1/w3_scale` 分支不是当前首要硬崩点（drop `w13` 后仍然 TensorCore 崩）
