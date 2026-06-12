# SGLang-JAX 项目设计文档

## 1. 项目概述

SGLang-JAX（包名 `sgl_jax`）是 SGLang 推理框架面向 Google TPU 的适配版本。原版 SGLang 基于 PyTorch/CUDA 构建，本项目将其核心运行时（SGLang Runtime, SRT）完整迁移至 JAX/XLA 技术栈，在保持 SGLang 的 API 兼容性（OpenAI-compatible API、RadixAttention 前缀共享、连续批处理）的同时，充分利用 TPU 硬件特性实现高性能推理。

**核心依赖：**
- `jax[tpu]==0.8.1` — JAX 框架 + TPU 后端
- `flax==0.12.4` — 神经网络模块（NNX API）
- `pathwaysutils` — Google Pathways 分布式运行时工具
- `transformers` / `safetensors` — 模型加载

**支持的 TPU 代际：** v4, v5, v5e, v5p, v6e, v7

**支持的模型：** Qwen2/3、Llama、Gemma2、Grok、DeepSeek-V3、Bailing MoE、MiMo 等

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────┐
│              HTTP Server (FastAPI)               │
│         OpenAI-compatible API endpoints          │
└──────────────┬──────────────────────────────────┘
               │ ZMQ IPC
┌──────────────▼──────────────────────────────────┐
│           TokenizerManager (主进程)               │
│    文本编码/解码 · 请求路由 · 流式输出            │
└──────────────┬──────────────────────────────────┘
               │ ZMQ IPC
┌──────────────▼──────────────────────────────────┐
│             Scheduler (子进程/线程)               │
│  连续批处理 · RadixCache · 请求调度 · 内存管理    │
│  ┌─────────────────────────────────────────────┐ │
│  │          JAX Device Mesh 初始化              │ │
│  │     jax.distributed.initialize()            │ │
│  │     pathwaysutils.initialize() (proxy模式)  │ │
│  │     create_device_mesh([data, tensor])       │ │
│  └─────────────────────────────────────────────┘ │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│         ModelWorker / TP Worker                  │
│  ┌───────────────────────────────────────────┐  │
│  │           ModelRunner                      │  │
│  │  nnx.split/merge · jax.jit 编译           │  │
│  │  ForwardBatch · KV Cache · Sampling        │  │
│  │  ┌─────────┐ ┌──────────┐ ┌────────────┐ │  │
│  │  │ Model   │ │Attention │ │  Pallas     │ │  │
│  │  │ Layers  │ │Backends  │ │  Kernels    │ │  │
│  │  └─────────┘ └──────────┘ └────────────┘ │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 2.2 三进程通信模型

系统采用三进程（或三线程）架构，通过 ZMQ IPC 通信：

1. **HTTP Server + TokenizerManager**：接收请求，分词，路由，流式返回
2. **Scheduler**：核心调度循环，管理 RadixCache、请求生命周期、批处理
3. **DetokenizerManager**：增量反分词

通过 `ServerArgs.enable_single_process` 可切换为线程模式。

### 2.3 请求处理流程

```
HTTP Request → TokenizerManager → ZMQ → Scheduler
   → PrefillAdder (构建prefill batch)
   → ModelWorker.forward_batch_generation()
     → ForwardBatch.init_new() (CPU→Device数据传输)
     → jitted_run_model() (JAX JIT编译的前向传播)
     → jitted_sampler() (JAX JIT编译的采样)
   → process_batch_result()
   → RadixCache 更新
   → ZMQ → DetokenizerManager → HTTP Response
```

---

## 3. TPU 支持实现详解

### 3.1 JAX 运行时初始化

**文件：** `srt/managers/scheduler.py:250-264`

Scheduler 初始化时完成 JAX 分布式运行时和设备网格的配置：

```python
# 多节点分布式初始化
if self.nnodes > 1:
    jax.distributed.initialize(server_args.dist_init_addr, self.nnodes, self.node_rank)

# Pathways 代理模式（Google Cloud Pathways 运行时）
platform = os.getenv("JAX_PLATFORMS", None)
if platform == "proxy":
    pathwaysutils.initialize()

# 创建 2D 设备网格：[data_parallel, tensor_parallel]
self.mesh = create_device_mesh(
    ici_parallelism=[-1, self.tp_size],  # -1 表示自动推断
    dcn_parallelism=[1, 1],
    device_indexes=server_args.device_indexes,
)
```

**设备网格（Device Mesh）：**

采用二维 `jax.sharding.Mesh`，轴名为 `["data", "tensor"]`，默认使用 `AxisType.Explicit` 显式分片模式：

```python
# srt/utils/mesh_utils.py
mesh = jax.sharding.Mesh(devices_array, ["data", "tensor"],
                         axis_types=(AxisType.Explicit, AxisType.Explicit))
```

- `tensor` 轴：张量并行，将模型权重和 KV 缓存沿注意力头维度切分
- `data` 轴：数据并行（自动填充），在多切片场景下用于 ICI/DCN 并行

`ici_parallelism=[-1, self.tp_size]` 中 data 轴设为 `-1`，`fill_unspecified_parallelism()` 自动计算：`data = total_devices / tp_size`。

**多节点支持：**

对于多节点部署，使用 `jax.distributed.initialize()` 建立跨节点通信，并通过 ZMQ pub/sub 将请求从 rank-0 广播到所有订阅节点。随机种子通过 `jax.experimental.multihost_utils.broadcast_one_to_all` 同步。

### 3.2 TPU 设备检测与内存管理

**文件：** `srt/utils/jax_utils.py`

#### 设备识别

```python
def get_device_name(num_devices=None):
    """解析 TPU 设备型号，返回标准名称如 'TPU v5e', 'TPU v6e-8'"""
    kind = jax.devices()[0].device_kind
    # 处理后缀：lite→e, e→e, p→p, TPU7x→TPU v7
```

#### HBM 容量查询

各代 TPU 的 HBM 容量硬编码：

| TPU 代际 | 单 Device HBM |
|----------|--------------|
| v5, v5p  | 95 GB        |
| v5e      | 16 GB        |
| v4, v6e  | 32 GB        |
| v7       | 96 GB        |

#### 可用内存计算

`get_available_device_memory()` 支持四种平台：

- **TPU**：`jax.local_devices()[i].memory_stats()["bytes_limit"] - bytes_in_use`
- **Proxy（Pathways）**：通过 `jax.live_arrays()` 遍历所有 `addressable_shards` 计算 HBM 占用
- **GPU**：同 TPU 方式，但过滤 `platform == "gpu"`
- **CPU**：使用 `psutil.virtual_memory().available`

分布式模式下，使用 `jax.shard_map` + `jax.lax.pmin` 求取跨设备最小可用内存，确保所有设备都能容纳 KV 缓存。

#### 关键 TPU 对齐常量

```python
TPU_HEAD_SIZE_ALIGNMENT = 128   # Head 维度必须对齐到 128
TPU_SECOND_LAST_MINOR = 8      # TPU 次末维对齐
```

Head 维度对齐到 128 是 TPU 硬件约束。例如，模型 `head_dim=128` 无需填充，而 `head_dim=96` 会被 padding 到 128。

### 3.3 JIT 编译与预编译策略

**文件：** `srt/model_executor/model_runner.py:185-270`

JAX 的 JIT 编译在首次调用时产生显著延迟。SGLang-JAX 采用以下策略：

#### NNX Graph-State 分离

```python
model_def, model_state = nnx.split(self.model)
self.model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)
```

将 Flax NNX 模型拆分为静态计算图（`model_def`）和可变状态（`model_state_leaves`），使得模型架构作为 `static_argnames` 传入 JIT，避免因架构变化触发重编译。

#### 核心 JIT 函数

```python
@partial(jax.jit, donate_argnames=["token_to_kv_pool"],
         static_argnames=["model_state_def"],
         compiler_options=jit_compiler_options)
def jitted_run_model(model_def, model_state_def, model_state_leaves,
                     forward_batch, token_to_kv_pool, logits_metadata):
    model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
    model = nnx.merge(model_def, model_state)
    return model(forward_batch, token_to_kv_pool, logits_metadata)
```

- `donate_argnames=["token_to_kv_pool"]`：KV 缓存通过捐赠机制避免拷贝，JAX 直接复用输入缓冲区存放输出
- `static_argnames=["model_state_def"]`：模型结构作为编译时常量
- TPU 日志记录器：当 `SGLANG_JAX_ENABLE_KERNEL_LOG_RECORDER=1` 时，设置 `compiler_options={"xla_tpu_enable_log_recorder": "true"}`

#### 预编译（Precompilation）

`tp_worker.py` 中的 `precompile_extend()` 和 `precompile_decode()` 在服务启动时对常见 batch size / token 数组合执行前向传播，预热 JIT 缓存，避免运行时编译卡顿。

#### 编译缓存监控

```python
jax._src.test_util.count_pjit_cpp_cache_miss()  # 统计 XLA 缓存未命中
```

### 3.4 张量并行与权重分片

#### 权重分片约定

所有模型层使用 `PartitionSpec`（`P`）标注权重分片策略：

| 层类型 | 权重名 | 分片策略 | 说明 |
|--------|--------|----------|------|
| QKV 投影 | `q_proj`, `k_proj`, `v_proj` | `P(None, "tensor")` | 列切分，输出沿 head 维度分布 |
| 输出投影 | `o_proj`, `c_proj` | `P("tensor", None)` | 行切分，部分和后 all-reduce |
| FFN 上投影 | `up_proj`, `gate_proj`, `w1`, `w3`, `w2` | `P(None, "tensor")` | 列切分（`w2` 亦归入此组，由 `_infer_default_sharding` 推断） |
| FFN 下投影 | `down_proj` | `P("tensor", None)` | 行切分 |
| 词嵌入 | `embedding`, `lm_head` | `P(None, None)` | 复制 |
| MoE 权重 | `wi_0`, `wi_1` | `P("expert", None, "tensor")` | 专家+张量并行，列切分 |
| MoE 权重 | `wo` | `P("expert", "tensor", None)` | 专家+张量并行，行切分 |

#### 线性层实现

**文件：** `srt/layers/linear.py`

```python
class LinearBase(nnx.Module):
    def __call__(self, x):
        output_sharding = NamedSharding(self.mesh, P(*([None]*(x.ndim-1)), self.kernel_axes[-1]))
        out = lax.dot_general(x, self.weight.value,
                              (((x.ndim-1,), (0,)), ((), ())),
                              preferred_element_type=self.params_dtype,
                              out_sharding=output_sharding)
```

使用 `lax.dot_general` 替代标准矩阵乘法，支持 `out_sharding` 提示，让 XLA 编译器生成最优的跨设备通信方案。

#### 量化线性层

`QuantizedLinear` 使用 `shard_map` 实现本地量化矩阵乘法，在 `in_specs` 中指定输入分片，XLA 自动插入必要的集合通信。

#### MoE 专家并行

**文件：** `srt/layers/moe.py`

`EPMoE` 创建独立的 `moe_mesh`，轴名为 `("expert", "tensor")`：

```python
devices = self.mesh.devices.flatten()
self.moe_mesh = jax.sharding.Mesh(
    devices.reshape(self.ep_size, self.tp_size),
    axis_names=("expert", "tensor"),
    axis_types=(AxisType.Explicit, AxisType.Explicit))
```

MoE 权重在 `use_abstract_mesh` 上下文中初始化，分片为 `P("expert", None, "tensor")`。执行时通过 `shard_map` 将 token 路由到对应专家设备。

### 3.5 KV 缓存设计

**文件：** `srt/mem_cache/memory_pool.py`

#### MHA TokenToKVPool（标准多头注意力）

5D 融合 KV 缓存格式：

```
kv_buffer: [num_pages+1, page_size, head_num*2//packing, packing, head_dim_aligned]
           ──────────── ───────── ────────────────────── ─────── ────────────────
           页数+1(*)    页大小     K/V交错头数            打包数   对齐头维度(128)

(*) 多分配一页用于 rpav3 内核兼容（代码注释: "this shape is more friendly to rpav3"）
```

- **K/V 交错**：K 和 V 的头交错排列（K0,V0,K1,V1,...），便于 TPU 向量化读取
- **dtype 打包**：bf16 打包数=2（32bit/16bit），fp8 打包数=4（32bit/8bit）
- **Head 维度对齐**：`head_dim` padding 到 128 的倍数（`TPU_HEAD_SIZE_ALIGNMENT`）
- **分片策略**：`P(None, None, "tensor", None, None)` — 沿 KV 头维度切分到 tensor 并行轴

缓冲区创建使用 JIT + `out_shardings` 确保直接在目标设备上分配：

```python
with self.mesh:
    kv_buf = jax.jit(
        lambda: jnp.zeros(shape=fused_buffer_shape, dtype=self.dtype),
        out_shardings=self.kv_sharding,
    )()
```

#### MLATokenToKVPool（多头潜在注意力）

DeepSeek-V3 的 MLA 架构使用低秩 KV 压缩，缓存的是潜在向量而非完整 K/V：

```
kv_buffer: [num_pages+1, aligned_page_size//packing, packing, align128(kv_lora_rank)+align128(qk_rope_head_dim)]
```

与 MHA 池相同，第一维为 `num_pages+1`（`(size + page_size) // page_size`）。

- 缓存维度由 `kv_lora_rank + qk_rope_head_dim` 决定（而非 `head_num * head_dim`）
- 复制存储（`P(None, None, None, None)`），不做张量切分
- 两个段（latent + rope）各自对齐到 128

#### SWAKVPool（滑动窗口注意力）

混合架构模型（如 Qwen3）同时拥有全注意力层和 SWA 层，维护两个独立子池：

- `full_kv_pool`：全注意力层的 KV 缓存
- `swa_kv_pool`：SWA 层的 KV 缓存（通常 head 数更少）
- `layers_mapping`：层 ID → `(local_id, is_swa)` 映射

#### KV 缓存更新

**文件：** `srt/kernels/update_kv_cache/update_kv_cache.py`

KV 缓存更新使用 Pallas TPU 内核实现原位写入，绕过 JAX 的不可变数组约束：

```python
# kv_cache_update_kernel 是普通函数，pl.pallas_call 在包装函数中命令式调用
def kv_cache_update_kernel(slices_ref, new_kv_hbm_ref, kv_cache_hbm_ref, _, scratch):
    # Pallas 内核体：直接在 HBM 中原位更新 KV 缓存页

# 在 kv_cache_update_impl / _kv_cache_update_wrapper 中：
kernel = pl.pallas_call(kv_cache_update_kernel, grid_spec=..., ...)
result = kernel(slices, new_kv, kv_cache, output, scratch)
```

外层通过 `jax.shard_map(check_vma=False)` 包装，允许跨设备通信重叠。

#### KV 缓存的函数式更新模式

由于 JAX 要求纯函数语义，KV 缓存更新采用"捐赠-替换"模式：

1. `jitted_run_model` 通过 `donate_argnames=["token_to_kv_pool"]` 捐赠旧 KV 缓存
2. 前向传播内部通过 Pallas 内核更新 KV 缓存
3. 返回新的 KV 缓存引用
4. 外层通过 `replace_kv_buffer()` 替换 Python 侧的缓冲区引用

### 3.6 TPU Pallas 内核

所有自定义内核使用 JAX Pallas（`jax.experimental.pallas.tpu`）编写，直接针对 TPU 硬件优化。

#### 3.6.1 Ragged Paged Attention v3

**文件：** `srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py`

源自 `vllm-project/tpu-inference` v0.11.1。

**核心设计：**
- 分为三个独立 Pallas 调用：`DECODE`（q_len=1）、`PREFILL`（q_len>1 静态）、`MIXED`（混合）
- 每个查询块（bq）重新初始化 `l/m/acc` 寄存器
- 支持滑动窗口（通过 per-bq-block start/end 索引精确跳过）
- 支持 `custom_mask`（投机解码）、`attention_sink`（流式推理）、`xai_temperature`（Grok 模型）

**TPU 版本自适应：**

```python
def get_tpu_version() -> int:
    """返回 TPU 数字版本号，用于内核参数调优"""
```

内核根据 TPU 代际调整块大小（`d_block_sizes`, `p_block_sizes`, `m_block_sizes`）和 VMEM 预算。

**分片集成：**

FlashAttention 后端通过 `shard_map` 包装 Pallas 内核：

```python
@shard_map(mesh, in_specs=..., out_specs=..., check_vma=False)
def attention_sharded(q, k, v, ...):
    return ragged_paged_attention_v3(q, k, v, ...)
```

KV 缓存沿 `kv_partition_axis="tensor"` 分片，查询在所有设备上复制。

#### 3.6.2 MLA v2 内核

**文件：** `srt/kernels/mla/v2/kernel.py`

为 DeepSeek-V3 吸收式 MLA（Absorbed MLA）设计的专用 Pallas 内核：
- 操作压缩的潜在 KV（`kv_lora_rank + qk_rope_head_dim`），而非完整 K/V 头
- 使用 ragged page index 布局
- 解码批大小参数用于内核微批次处理
- `num_kv_pages_per_block` 和 `num_queries_per_block` 按 TPU 代际调优

#### 3.6.3 Fused MoE v1 内核

**文件：** `srt/kernels/fused_moe/v1/kernel.py`

融合 MoE 内核，集成 All-to-All (A2A) 通信：
- `_A2A_HBM_FRACTION = 0.03`：按设备 HBM 的 3% 预留 A2A 环形缓冲区
- 支持专家并行（EP）的 token 路由和结果聚合
- 块配置（`FusedMoEBlockConfig`）按 TPU 代际和 token 数量自适应调整

#### 3.6.4 量化矩阵乘法内核

**文件：** `srt/kernels/quantized_matmul/`

- `kernel.py`：`xla_quantized_matmul_local` — XLA 级别的量化矩阵乘法
- `blockwise_kernel.py`：块级量化（`weight_block_size`），**仅支持 TPU 后端**

```python
# model_runner.py:331
if wbs is not None and jax.default_backend() != "tpu":
    raise RuntimeError("Block-wise quantization requires TPU backend")
```

#### 3.6.5 其他 Pallas 内核

| 内核 | 文件 | 功能 |
|------|------|------|
| GMM | `kernels/gmm/` | Megablox 分组矩阵乘法 |
| Simple GLA | `kernels/simple_gla/` | 门控线性注意力 |
| Speculative | `kernels/speculative/` | 投机解码（树构建、验证、采样） |
| Paged Attention | `kernels/paged_attention/` | 分页注意力（非 ragged 版本） |
| Flash Attention | `multimodal/kernels/flash_attention.py` | 多模态 Flash Attention |

### 3.7 注意力后端

**文件：** `srt/layers/attention/`

| 后端 | 命令行参数 | 适用场景 | 实现 |
|------|-----------|---------|------|
| FlashAttention | `--attention-backend fa` | MHA 模型（默认） | Pallas ragged paged attention v3 + `shard_map` |
| MLA | `--attention-backend fa`（MLA 模型自动选择） | DeepSeek-V3 MLA | Pallas MLA v2 内核 |
| FA_MHA | `--attention-backend fa_mha` | 强制 MHA 路径（MLA 模型解压缩 KV） | 标准 FlashAttention |
| Native | `--attention-backend native` | 调试/CPU 回退 | 纯 JAX numpy 操作 |
| LinearAttention | 自动选择 | 线性注意力模型（Bailing MoE v2.5） | Pallas GLA 内核 |

**FlashAttention 元数据：**

`FlashAttentionMetadata` 注册为 JAX pytree 节点，包含：
- `cu_q_lens` / `cu_kv_lens`：累积查询/KV 长度（ragged tensor 索引）
- `page_indices`：页索引（从 `cache_loc` 派生，按 `page_size` 步长采样）
- `seq_lens`：序列长度
- `distribution`：prefill/decode/mixed 分布计数
- `custom_mask`：投机解码掩码
- `swa_page_indices`：滑动窗口页索引映射

### 3.8 内存容量规划

**文件：** `srt/model_executor/model_runner.py:398-464`

#### 单 Token KV 缓存开销计算

```python
def _compute_cell_size(self) -> int:
    """每 token 每设备的 KV 缓存字节数"""
    align128 = lambda x: (x + 127) // 128 * 128

    if use_mla_backend:  # MLA 路径
        return (align128(kv_lora_rank) + align128(qk_rope_head_dim)) * num_layers * dtype_size
    else:  # MHA 路径
        return num_kv_heads_per_device * align128(head_dim) * 2 * num_layers * dtype_size
```

关键点：
- Head 维度对齐到 128（`TPU_HEAD_SIZE_ALIGNMENT`）
- MLA 路径无 `*2`（只存潜在向量，不存 V），且为复制存储
- MHA 路径 `*2`（K 和 V），按 `tp_size` 切分头数

#### 最大 Token 数计算

```python
available_kv_cache_bytes = available_device_memory - total_device_memory * (1 - mem_fraction_static)
max_tokens = available_kv_cache_bytes // cell_size
```

`mem_fraction_static`（默认 0.88）预留 12% 给权重和激活，剩余分配给 KV 缓存。

### 3.9 采样器

**文件：** `srt/layers/sampler.py`

Sampler 作为 `nnx.Module` 被 JIT 编译一次，支持：
- 贪婪解码（`argmax`）
- 随机采样（temperature + softmax + top-k/top-p/min-p）
- 两种 top-k/top-p 策略：排序法和掩码法
- 确定性采样（Gumbel 噪声 + 种子哈希）
- 词法约束（`apply_token_bitmask`）
- 频率/存在惩罚

RNG 种子通过 `jax.random.fold_in(base_key, step_counter)` 生成，避免 eager 分裂导致的串行化。

### 3.10 权重加载

**文件：** `srt/utils/weight_utils.py`

1. 从 Safetensors 文件在 **host** 端加载权重为 numpy 数组
2. 根据 `WeightMapping` 推断分片策略（列并行/行并行/复制）
3. 使用 `jax.device_put(array, NamedSharding(mesh, P(...)))` 将权重直接分片放置到 TPU 设备
4. 多节点场景下，host 0 扫描文件列表，通过 `broadcast_one_to_all` 同步到其他节点

### 3.11 TPU 集群部署

**文件：** `scripts/launch_tpu.sh`, `scripts/tpu_resource.sky.yaml`

使用 SkyPilot 在 GCP 上配置 TPU VM：

```yaml
resources:
  accelerators: $ACCELERATOR
  accelerator_args:
    tpu_vm: True
    runtime_version: v2-alpha-tpuv6e
  use_spot: True
```

Docker 镜像推送到 `asia-northeast1-docker.pkg.dev/tpu-service-473302/sglang-project`。

### 3.12 环境变量

| 变量 | 作用 |
|------|------|
| `JAX_PLATFORMS` | 设备选择：`tpu`/`gpu`/`cpu`/`proxy`（Pathways） |
| `SGLANG_JAX_ENABLE_KERNEL_LOG_RECORDER` | 启用 TPU XLA 日志记录器 |
| `SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK` | 禁用 TP 内存平衡检查 |
| `SGLANG_CI_SMALL_KV_SIZE` | CI 环境覆盖 KV 缓存大小 |

---

## 4. 与原版 SGLang（CUDA）的关键差异

| 方面 | SGLang (CUDA) | SGLang-JAX (TPU) |
|------|---------------|-------------------|
| **框架** | PyTorch + CUDA | JAX + XLA + Pallas TPU |
| **内核语言** | CUDA C++ / Triton | JAX Pallas (`jax.experimental.pallas.tpu`) |
| **设备默认** | `cuda` | `tpu` |
| **分布式** | NCCL / torch.distributed | `jax.distributed` + `jax.sharding.Mesh` + `shard_map` |
| **内存查询** | `torch.cuda.mem_get_info` | `jax.devices()[i].memory_stats()` |
| **KV 缓存** | CUDA paged attention | Pallas ragged paged attention，5D 融合 KV，head_dim 对齐 128 |
| **注意力内核** | FlashInfer / FlashAttention CUDA | Pallas ragged paged attention v3, MLA v2 |
| **MoE 内核** | Cutlass/SGLang CUDA 融合内核 | Pallas fused_ep_moe + A2A 环形缓冲 |
| **量化** | CUDA 量化内核 | 块级量化仅支持 TPU；Pallas quantized_matmul |
| **JIT** | torch.compile（可选） | `jax.jit` 必需；启动时预编译多种 batch/token 组合 |
| **分片** | torch.distributed.tensor | `NamedSharding` + `PartitionSpec` + `Mesh` |
| **权重加载** | PyTorch state_dict → GPU | Safetensors → numpy → `jax.device_put` + `NamedSharding` |
| **集合通信** | NCCL all-reduce | `jax.lax.psum/pmin` + `shard_map`，XLA 自动插入 |
| **多节点** | torchrun / ray | `jax.distributed.initialize()` + `pathwaysutils.initialize()` |
| **Head 对齐** | 自然对齐 | 强制对齐到 128（`TPU_HEAD_SIZE_ALIGNMENT`） |
| **KV 打包** | K 和 V 分离存储 | K/V 交错打包（bf16: 2x, fp8: 4x） |
| **调度器** | 每 GPU 进程一个 | 每 VM 一个，协调所有 TPU 设备 |
| **集群部署** | Cloud CLIs / Docker | SkyPilot + GCP TPU VM |

---

## 5. 目录结构

```
python/sgl_jax/
├── __main__.py                    # 入口
├── launch_server.py               # 服务启动参数解析
├── srt/
│   ├── entrypoints/               # HTTP/Engine 入口
│   │   ├── http_server.py         # FastAPI 服务
│   │   ├── engine.py              # Python API
│   │   └── openai/               # OpenAI 兼容 API
│   ├── managers/                  # 调度管理
│   │   ├── scheduler.py           # 核心调度器
│   │   ├── tokenizer_manager.py   # 分词管理
│   │   ├── detokenizer_manager.py # 反分词管理
│   │   ├── tp_worker.py           # 张量并行工作器
│   │   └── schedule_batch.py      # 批处理数据结构
│   ├── model_executor/            # 模型执行
│   │   ├── model_runner.py        # JIT 编译、前向传播、内存池
│   │   └── forward_batch_info.py  # ForwardBatch 定义
│   ├── models/                    # 模型架构实现
│   │   ├── qwen3.py, llama.py, deepseek_v3.py, ...
│   │   └── registry.py            # 模型注册
│   ├── layers/                    # 神经网络层
│   │   ├── linear.py              # 线性层（分片 matmul）
│   │   ├── radix_attention.py     # RadixAttention 模块
│   │   ├── moe.py                 # 专家并行 MoE
│   │   ├── sampler.py             # 采样器
│   │   └── attention/             # 注意力后端
│   │       ├── flashattention_backend.py  # Pallas FlashAttention
│   │       ├── mla_backend.py             # Pallas MLA
│   │       └── native_backend.py          # 纯 JAX 回退
│   ├── kernels/                   # Pallas TPU 内核
│   │   ├── ragged_paged_attention/  # Ragged Paged Attention v3
│   │   ├── fused_moe/              # 融合 MoE + A2A
│   │   ├── mla/                    # MLA v2 内核
│   │   ├── gmm/                    # 分组矩阵乘法
│   │   ├── quantized_matmul/       # 量化矩阵乘法
│   │   ├── update_kv_cache/        # KV 缓存原位更新
│   │   ├── simple_gla/             # 门控线性注意力
│   │   └── speculative/            # 投机解码内核
│   ├── mem_cache/                 # KV 缓存管理
│   │   ├── memory_pool.py         # MHATokenToKVPool / MLATokenToKVPool / SWAKVPool
│   │   ├── radix_cache.py         # RadixTree 前缀共享
│   │   └── allocator.py           # 页分配器
│   ├── configs/                   # 配置
│   ├── model_loader/              # 权重加载
│   ├── sampling/                  # 采样逻辑
│   ├── constrained/               # 结构化输出
│   ├── lora/                      # LoRA 适配器
│   ├── eplb/                      # 专家负载均衡
│   ├── speculative/               # 投机解码
│   ├── multimodal/                # 多模态支持
│   ├── utils/
│   │   ├── jax_utils.py           # TPU 设备检测、内存管理
│   │   ├── mesh_utils.py          # 设备网格创建
│   │   └── weight_utils.py        # 权重加载与分片
│   └── server_args.py             # 命令行参数
├── benchmark/                     # 性能基准测试
├── test/                          # 测试套件
├── scripts/
│   ├── launch_tpu.sh              # TPU 集群启动脚本
│   └── tpu_resource.sky.yaml      # SkyPilot TPU 资源配置
└── docs/                          # 文档
```