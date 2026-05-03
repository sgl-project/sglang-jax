# CompilationManager 重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将预编译逻辑从 `tp_worker.py` 提取到独立的 `CompilationManager` 类，统一 bucket 管理、dummy batch 构造和预编译流程。

**Architecture:** `CompilationManager` 拥有 bucket 计算和预编译执行，`tp_worker` 委托调用。保持 `schedule_batch` 的 padding 接口不变。支持 lazy 编译策略，uncommon 变体（如 `return_logprob=True`）在首次遇到时编译而非启动时。

**Tech Stack:** Python, JAX, NumPy

---

## 文件结构

| 操作 | 文件路径 | 职责 |
|------|---------|------|
| 新建 | `python/sgl_jax/srt/model_executor/compilation_manager.py` | CompilationManager 类 |
| 修改 | `python/sgl_jax/srt/managers/tp_worker.py` | 委托到 CompilationManager |
| 新建 | `tests/srt/model_executor/test_compilation_manager.py` | 单元测试 |

---

### Task 1: 创建 CompilationManager — Bucket 管理

**Files:**
- Create: `python/sgl_jax/srt/model_executor/compilation_manager.py`
- Test: `tests/srt/model_executor/test_compilation_manager.py`

- [ ] **Step 1: 写 bucket 计算的 failing tests**

测试 `_compute_token_buckets()` 和 `_compute_bs_buckets()` 的核心行为：
- token bucket 过滤：dp_size 整除、范围约束 `[max_bs, max_tokens]`
- bs bucket 过滤：最大值约束、fused MoE 最小值 `tp_size * 2`
- 边界情况：空列表时追加 max 值
- `pad_to_bucket()`：返回第一个 >= 输入值的 bucket

```python
# tests/srt/model_executor/test_compilation_manager.py
class TestCompilationManager(CustomTestCase):
    def test_compute_token_buckets_filters_by_dp_size(self):
        ...
    def test_compute_bs_buckets_enforces_fused_moe_minimum(self):
        ...
    def test_pad_to_bucket(self):
        ...
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/srt/model_executor/test_compilation_manager.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: 实现 CompilationManager 的 bucket 管理部分**

从 `tp_worker.py:264-293` (`normalize_token_paddings`) 和 `tp_worker.py:216-241` (bs bucket 计算) 提取逻辑到 `CompilationManager.__init__()` 中。

关键方法：
- `__init__(server_args, max_padded_batch_size, max_padded_num_tokens, dp_size, tp_size, page_size, max_req_len, ...)` — 接收必要参数而非整个 worker
- `_compute_token_buckets(user_paddings)` — 来自 `normalize_token_paddings()`
- `_compute_bs_buckets(user_paddings)` — 来自 `__init__:216-241`
- `_compute_cache_loc_buckets()` — 来自 `__init__:238-241`
- `pad_to_bucket(value, buckets)` — 静态方法，复用 `find_padding_size` 的逻辑

属性暴露：`token_buckets`, `bs_buckets`, `cache_loc_buckets`

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/srt/model_executor/test_compilation_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/sgl_jax/srt/model_executor/compilation_manager.py tests/srt/model_executor/test_compilation_manager.py
git commit -m "feat(compilation_manager): add bucket management (#609)"
```

---

### Task 2: 添加 Dummy Batch 构造和预编译方法

**Files:**
- Modify: `python/sgl_jax/srt/model_executor/compilation_manager.py`
- Test: `tests/srt/model_executor/test_compilation_manager.py`

- [ ] **Step 1: 添加 `_make_dummy_batch()` 测试**

验证生成的 `ModelWorkerBatch` 字段正确：input_ids 长度 = num_tokens，out_cache_loc 有效区间等。

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 实现 `_make_dummy_batch()`**

从 `tp_worker.py:439-508` (`generate_model_worker_batch()`) 迁移逻辑。新方法需要访问 `model_config.vocab_size`、`server_args.multimodal`、`server_args.enable_static_lora` 等配置。

构造函数新增参数：`vocab_size`, `multimodal`, `enable_static_lora`, `moe_backend`。

- [ ] **Step 4: 实现预编译入口方法**

```python
def precompile_all(self, forward_fn, prepare_lora_fn=None, future_token_ids_map=None):
    """单一入口：依次编译 extend 和 decode 变体。"""
    self._precompile_extend(forward_fn, prepare_lora_fn, future_token_ids_map)
    self._precompile_decode(forward_fn, prepare_lora_fn, future_token_ids_map)
```

`_precompile_extend()` 来自 `tp_worker.py:299-348`，`_precompile_decode()` 来自 `tp_worker.py:350-398`。

**关键设计决策：** `forward_fn` 是一个回调（即 `tp_worker.forward_batch_generation`），CompilationManager 不直接持有 model_runner 引用，而是通过函数参数接收执行能力。这保持了单向依赖。

- [ ] **Step 5: 运行测试确认通过**

- [ ] **Step 6: Commit**

```bash
git add python/sgl_jax/srt/model_executor/compilation_manager.py tests/srt/model_executor/test_compilation_manager.py
git commit -m "feat(compilation_manager): add dummy batch construction and precompile methods (#609)"
```

---

### Task 3: 集成到 tp_worker — 委托调用

**Files:**
- Modify: `python/sgl_jax/srt/managers/tp_worker.py`
- Modify: `python/sgl_jax/srt/model_executor/compilation_manager.py`

- [ ] **Step 1: 在 `tp_worker.__init__` 中创建 CompilationManager**

在 `tp_worker.py:207` 处（`self.max_padded_batch_size, self.max_padded_num_tokens = self.get_max_padded_size()` 之后），替换行 209-241 的 bucket 计算代码为：

```python
self.compilation_manager = CompilationManager(
    server_args=self.server_args,
    max_padded_batch_size=self.max_padded_batch_size,
    max_padded_num_tokens=self.max_padded_num_tokens,
    dp_size=self.dp_size,
    tp_size=self.tp_size,
    page_size=self.page_size,
    max_req_len=self.max_req_len,
    vocab_size=self.model_config.vocab_size,
    multimodal=self.server_args.multimodal,
)
# 保持向后兼容的属性引用
self.precompile_token_paddings = self.compilation_manager.token_buckets
self.precompile_bs_paddings = self.compilation_manager.bs_buckets
self.precompile_cache_loc_paddings = self.compilation_manager.cache_loc_buckets
```

- [ ] **Step 2: 修改 `run_precompile()` 委托到 CompilationManager**

将 `tp_worker.py:295-297` 的 `run_precompile()` 改为：

```python
def run_precompile(self, future_token_ids_map=None):
    prepare_lora = self.prepare_lora_batch if self.server_args.enable_lora else None
    self.compilation_manager.precompile_all(
        forward_fn=self.forward_batch_generation,
        model_runner=self.model_runner,
        mesh=self.mesh,
        prepare_lora_fn=prepare_lora,
        future_token_ids_map=future_token_ids_map,
    )
```

- [ ] **Step 3: 删除已迁移的方法**

删除 `tp_worker.py` 中以下方法（保留方法签名的 docstring 作为指引到 compilation_manager 的注释是不必要的——直接删除）：
- `normalize_token_paddings()` (264-293)
- `precompile_extend()` (299-348)
- `precompile_decode()` (350-398)
- `generate_model_worker_batch()` (439-508)

保留：
- `get_max_padded_size()` (405-430) — 被其他地方使用
- `get_precompile_paddings()` (432-437) — 现在从 compilation_manager 读取（已在 Step 1 设置属性别名）

- [ ] **Step 4: 运行现有测试确认不破坏**

Run: `python -m pytest tests/ -k "precompile or compilation" -v`

- [ ] **Step 5: Commit**

```bash
git add python/sgl_jax/srt/managers/tp_worker.py python/sgl_jax/srt/model_executor/compilation_manager.py
git commit -m "refactor(tp_worker): delegate precompile to CompilationManager (#609)"
```

---

### Task 4: Lazy 编译支持

**Files:**
- Modify: `python/sgl_jax/srt/model_executor/compilation_manager.py`
- Test: `tests/srt/model_executor/test_compilation_manager.py`

- [ ] **Step 1: 添加 variant tracking 测试**

测试 `_ensure_compiled()` 的行为：
- 首次遇到 `return_logprob=True` 时记录并返回 True（需要编译）
- 第二次遇到同变体时返回 False（已编译）

- [ ] **Step 2: 实现 lazy 编译状态跟踪**

```python
class CompilationManager:
    def __init__(self, ...):
        ...
        # 已编译的变体集合，key = (forward_mode, token_bucket, bs_bucket, return_logprob)
        self._compiled_variants: set[tuple] = set()

    def _ensure_compiled(self, variant_key: tuple) -> bool:
        """返回 True 表示该变体首次出现，需要编译。"""
        if variant_key in self._compiled_variants:
            return False
        self._compiled_variants.add(variant_key)
        return True
```

注意：这不会自动触发 JIT 编译——JAX 的 `jit` cache 会自动编译首次遇到的新 shape/aux_data 组合。这个 tracking 只用于日志和监控：当 `_ensure_compiled` 返回 True 且不在预编译阶段时，说明发生了运行时 lazy 编译。

- [ ] **Step 3: 在预编译阶段注册已编译变体**

在 `_precompile_extend()` 和 `_precompile_decode()` 的循环体中，每次成功编译后调用 `_compiled_variants.add(variant_key)`。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/srt/model_executor/test_compilation_manager.py -v`

- [ ] **Step 5: Commit**

```bash
git add python/sgl_jax/srt/model_executor/compilation_manager.py tests/srt/model_executor/test_compilation_manager.py
git commit -m "feat(compilation_manager): add lazy compilation tracking (#609)"
```

---

### Task 5: 端到端验证 — TPU Pod 测试

**Files:** 无代码变更

- [ ] **Step 1: 在 pod 上启动服务**

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
  --model-path /inference-models/Qwen3-30B-A3B-Instruct-2507 \
  --trust-remote-code \
  --dist-init-addr=0.0.0.0:10011 \
  --nnodes=1 --tp-size=4 --ep-size=4 \
  --device=tpu --moe-backend=fused \
  --random-seed=3 \
  --mem-fraction-static=0.8 \
  --chunked-prefill-size=2048 \
  --dtype=bfloat16 \
  --max-running-requests 128 \
  --skip-server-warmup \
  --page-size=128
```

- [ ] **Step 2: 验证预编译日志**

检查日志中：
1. `[EXTEND] Precompile finished` 和 `[DECODE] Precompile finished` 正常完成
2. 无异常 cache miss 警告

- [ ] **Step 3: 发送测试请求验证 cache miss = 0**

```bash
curl http://localhost:30000/v1/chat/completions \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
```

检查服务端日志，确认 `cache_miss_count` 未增加。

- [ ] **Step 4: 运行 evalscope 精度评测**

验证重构后精度未退化。

- [ ] **Step 5: 发送 logprob 请求验证 lazy 编译**

```bash
curl http://localhost:30000/v1/chat/completions \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16, "logprobs": true, "top_logprobs": 5}'
```

预期：首次 logprob 请求触发 lazy 编译（日志显示 cache miss），后续 logprob 请求不再触发。
