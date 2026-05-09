# RFC: Eagle/MTP 重构RFC

## 背景

本 RFC 的目标是在 `sglang-jax` 中支持 `mimo-v2.5-pro` MTP 推测解码，并借此重构当前集中在 `EAGLEWorker` 中的 EAGLE/MTP 控制流。现有实现能支撑普通 EAGLE，但控制面、draft runner、target verify 和 scheduler 更新耦合较重，不适合作为 MTP、DP attention 和 scheduler overlap 的长期基础。

最终目标是一套行为稳定、性能达标、边界清晰，并能兼容 DP attention 与后续 overlap scheduler 的 EAGLE/MTP 实现。

## 总体路线

本次重构分三阶段推进，每个阶段只解决一个主问题：

1. **Phase 1: 功能基线 + 数据流重构。** 拆分 `BaseSpecWorker` / `BaseDraftWorker`，让普通 EAGLE 和 `mimo-v2.5-pro` MTP 共享 verify/data contract；同时让 Phase 1 数据契约完成 DP attention 兼容，并为后续 scheduler overlap 收紧显式状态边界。
2. **Phase 2: EAGLE/MTP 流程性能优化。** 在功能路径稳定后基于 profiling 优化真实热点，性能对标 sglang `mimo-v2.5-pro` MTP 无 scheduler overlap 路径；是否引入 kernel 由 profiling 决定。
3. **Phase 3: Spec decode + scheduler overlap。** 在性能达标后单独确定 overlap 路线。已有 demo 验证 async dispatch 可行，但是否替代多线程 overlap 需要独立 RFC 讨论。

## 阶段详细计划

### Phase 1: EAGLE 功能基线 + 数据流重构(DDL:5-15)

Phase 1 重构 EAGLE/MTP 的功能路径和数据边界。普通 EAGLE 使用重构后的 `EAGLEWorker` + `EagleDraftWorker`，`mimo-v2.5-pro` MTP 使用 `MultiLayerEAGLEWorker` + 多 runner `MultiLayerDraftWorker`，二者共享 target verify 和数据契约逻辑。

主要工作：

1. 固定 `SpecInput`、`EagleDraftInput`、`EagleVerifyInput` 和 `GenerationBatchResult` 的字段语义、ownership、device/host 边界、shape 约束和跨轮状态归属。
2. 拆分 `BaseSpecWorker` / `BaseDraftWorker`，把 spec 编排、draft runner、target verify 和 scheduler 更新的职责边界固定下来。
3. 将普通 EAGLE draft 逻辑迁移到 `EagleDraftWorker`，并将 `EAGLEWorker` 收敛为普通 EAGLE 的组合式 spec worker。
4. 新增 `MultiLayerEAGLEWorker` / `MultiLayerDraftWorker`，接入 `mimo-v2.5-pro` MTP 的多 runner、多 layer 权重加载、per-step hidden states、top-k 拼接和 draft extend。
5. 完成 DP attention 兼容所需的数据契约和 target verify 相关 layout 处理，包括 DP padded layout、per-DP token accounting、rank-local KV slot、verify metadata 和输出 selector。
6. 收紧 overlap-ready 边界：所有 host snapshot、device tensor、KV 生命周期、输出还原和下一轮 draft state 都必须通过显式字段传递；Phase 1 仍保持 spec decode 与 overlap scheduler 互斥。
7. 保持现有非 overlap EAGLE 行为和 precompile shape 稳定，解决当前 JIT/cache miss 与接受率问题。

Phase 1 交付物：

1. `mimo-v2.5-pro` MTP 能通过 `MultiLayerEAGLEWorker` 完成 prefill、draft decode、target verify 和 draft extend。
2. `EAGLEWorker` 和 `MultiLayerEAGLEWorker` 类结构抽象改造完成。
3. 数据契约和 target verify 路径兼容 DP attention；scheduler overlap 只完成数据边界收敛，不启用执行路径。
4. 无新增 JIT/cache miss，功能边界清晰，代码架构简洁，接受率和精度符合要求。
5. speculative unit tests、`mimo-v2.5-pro` MTP E2E（3-layer MTP，topk=1，greedy sample）通过。

### Phase 2: EAGLE 流程性能优化(DDL:5-20)

Phase 2 的目标是在 Phase 1 功能基线稳定后，基于 profiling 找到 EAGLE/MTP 流程中的真实性能热点，并针对性优化无 scheduler overlap 场景下的端到端性能。这个阶段不重新调整 worker 主抽象，不把 scheduler overlap 纳入实现范围，也不为假设瓶颈提前写 kernel。

主要工作：

1. 建立 `mimo-v2.5-pro` MTP profiling 口径，覆盖 draft decode、target verify、draft extend、tree build、sampling、KV 分配、host/device 同步、padding 和 metadata 构造。
2. 对比 sglang `mimo-v2.5-pro` 开启 MTP、但不开 scheduler overlap 的性能，确定 JAX 侧需要追平的 latency、throughput 和接受长度分布指标。
3. 根据 profiling 结果优化 EAGLE 内部流程，包括不必要的 host/device 同步、重复 metadata 构造、padding 过量、shape 抖动和跨 runner 数据搬运。
4. 对 MTP 多 runner 流程做针对性优化，优先保持代码路径清晰；只有 profiling 证明 Python/JAX 组合逻辑不是主要瓶颈时，才考虑 kernel 优化。
5. 如 tree build、sampling、mask 构造或 top-k 拼接成为明确热点，再评估是否通过 Pallas kernel 或 fused kernel 优化。
6. 持续验证 Phase 1 的功能、接受率、精度、KV cache 行为和 precompile shape 不因性能优化回退。

Phase 2 交付物：

1. `mimo-v2.5-pro` MTP 有可重复的 profiling 报告和性能基线。
2. 明确 EAGLE/MTP 流程的主要性能热点，并完成针对性优化。
3. 性能对标 sglang on GPU `mimo-v2.5-pro` 开启 MTP、无 scheduler overlap 的路径。
4. 如需要 kernel 优化，给出基于 profiling 的必要性、目标 kernel 范围和预期收益；不在没有数据支撑时提前引入 kernel 复杂度。
5. 优化后仍保持 Phase 1 的功能边界、数据契约、接受率、精度和 JIT/cache miss 要求。

### Phase 3: Spec decode + scheduler overlap(DDL:5-30)

Phase 3 的目标是在 EAGLE/MTP 功能稳定且性能达标后，再处理 spec decode 与 scheduler overlap 的兼容开发。这个阶段需要先确定 overlap 实现路线：是采用 async dispatch 重构，还是基于现有多线程 overlap 机制增加信号标记和状态管理；该路线选择需要单独 RFC 讨论，不在本期实现中直接展开。

主要工作：

1. 基于 Phase 1 的 overlap-ready 契约，梳理 target verify、draft extend、scheduler 更新、KV 裁剪和下一轮 draft state 之间的异步边界。
2. 评估 async dispatch 路线：参考已验证可行性的 demo，判断是否用 async dispatch 替代多线程和显式信号队列，降低 overlap 状态管理复杂度。
3. 评估现有多线程路线：明确如果继续基于 `ModelWorkerClient`、`tp_worker_overlap_thread.py` 和 overlap 队列实现，需要增加哪些信号、快照和生命周期保护。
4. 对比两条路线在代码复杂度、debug 难度、JAX dispatch 行为、DP attention layout、KV 生命周期和失败恢复上的取舍。
5. 在单独 RFC 中确定 scheduler overlap 方案后，再实现 spec decode + scheduler overlap 兼容。

Phase 3 交付物：

1. 一份独立 scheduler overlap RFC，明确 async dispatch 与多线程路线的选择和理由。
2. 方案确定后， `mimo-v2.5-pro` MTP 能在 overlap scheduler 下完成 decode。
3. overlap 与非 overlap 路径的输出、accept length、KV cache 释放行为、DP token accounting、接受率和精度一致。

## 时间线预期

当前时间线按 **5.15 功能冻结，5.20 性能基线，5.30 overlap 方案与最小兼容** 来推进。这个节奏比较激进，因此每个节点只保留 checkpoint 级验收，详细验收标准以各 Phase 计划为准。

| 时间节点 | 阶段 | Checkpoint 验收 | 不纳入默认验收 |
|---|---|---|---|
| 5.15 | Phase 1 | 功能冻结：普通 EAGLE 不回退，`mimo-v2.5-pro` MTP E2E 跑通，数据契约和 DP/overlap-ready 边界落地，无新增 JIT/cache miss | EAGLE 流程性能优化；scheduler overlap 启用；scheduler overlap 组合验证 |
| 5.20 | Phase 2 | 性能基线：完成 profiling、首轮优化和 sglang 无 overlap 路径对比，达到 `mimo-v2.5-pro` MTP 性能要求 | 所有潜在性能优化全部完成；默认引入 Pallas/fused kernel；scheduler overlap 兼容 |
| 5.30 | Phase 3 | overlap 最小兼容：完成 overlap RFC，并跑通选定路线下普通 EAGLE 与 MTP 的最小 overlap decode 路径 | DP attention + overlap 完整矩阵验证；所有失败模式完整覆盖；进一步性能优化 |

## Phase 1 类结构重构

Phase 1 的类结构只保留一套新路径，不引入 `V2` 命名，也不保留旧实现与新实现并行的双路径。普通 EAGLE 和 MTP 通过相同的 `BaseSpecWorker` / `BaseDraftWorker` 契约接入。

### Phase 1 类间依赖关系

```text
Scheduler
  -> EAGLEWorker
       -> target_worker: ModelWorker
       -> draft_worker: EagleDraftWorker
  -> MultiLayerEAGLEWorker
       -> target_worker: ModelWorker
       -> draft_worker: MultiLayerDraftWorker
            -> draft_runner_list: list[ModelRunner]

BaseSpecWorker
  - target_worker
  - draft_worker
  - forward_batch_speculative_generation()
  - verify()

EAGLEWorker
  - 普通 EAGLE spec decode 编排

MultiLayerEAGLEWorker
  - MTP spec decode 编排
  - 复用 BaseSpecWorker verify/data contract 逻辑

EagleDraftWorker
  - draft()
  - draft_forward()
  - draft_extend_for_prefill()
  - draft_extend_for_decode()
  - prepare/padding helpers owned by draft path

MultiLayerDraftWorker
  - draft()
  - draft_forward() across MTP steps
  - draft_extend_for_prefill()
  - draft_extend_for_decode()
  - mimo-v2.5-pro 多 runner / hidden_states contract
```

职责边界：

| 对象 | 拥有职责 | 不应拥有职责 |
|---|---|---|
| `Scheduler` | batch 选择、running batch 更新、结果处理、stream 输出 | draft/verify 内部步骤 |
| `EAGLEWorker` | 普通 EAGLE spec decode 编排、target verify、组装 `GenerationBatchResult` | draft runner 细节、MTP layer 细节 |
| `MultiLayerEAGLEWorker` | MTP spec decode 编排、target verify、组装 `GenerationBatchResult`、共享 data contract 逻辑 | MTP draft runner 细节、scheduler 状态更新 |
| `EagleDraftWorker` | draft model runner、draft attention metadata、draft prefill、draft decode、draft extend | scheduler 状态更新、target sampling |
| `MultiLayerDraftWorker` | MTP 多 runner、per-step hidden_states、topk 合并、MTP draft extend | scheduler 状态更新、target sampling、独立编排 verify |
| `EagleDraftInput` | 下一轮 draft 的状态 | 调度策略 |
| `EagleVerifyInput` | target verify 的输入和采样辅助信息 | draft runner 生命周期 |

### Phase 1 运行时数据流

Prefill 路径：

```text
BaseSpecWorker.forward_batch_speculative_generation(model_worker_batch)
  -> target_worker.forward_batch_generation(capture_hidden_mode=FULL)
  -> draft_worker.draft_extend_for_prefill(
       model_worker_batch,
       target_hidden_states,
       next_token_ids,
     )
  -> GenerationBatchResult(next_draft_input=EagleDraftInput)
```

Decode 路径：

```text
BaseSpecWorker.forward_batch_speculative_generation(model_worker_batch)
  -> draft_worker.draft(model_worker_batch)
       returns EagleVerifyInput
  -> BaseSpecWorker.verify(model_worker_batch)
       target verify + tree sampling
       returns GenerationBatchResult(next_draft_input=EagleDraftInput)
  -> draft_worker.draft_extend_for_decode(model_worker_batch, batch_output)
       updates next_draft_input.topk_p/topk_index/hidden_states
  -> return batch_output
```

`EAGLEWorker` 与 `MultiLayerEAGLEWorker` 都遵循 `BaseSpecWorker` 契约。二者的主流程只依赖各自的 `BaseDraftWorker` 实现，不直接持有普通 EAGLE 或 MTP 的 draft runner 细节。Phase 1 内需要把 `EagleDraftWorker` 和 `MultiLayerDraftWorker` 的共同接口设计好，并把 verify/data contract 逻辑沉到可共享的基类或 util 中。

## Phase1 数据结构重构

Phase 1 的第一项工作是收紧数据契约。字段尽量兼容现有代码，但需要明确每个字段属于哪一层、在 host 还是 device 上消费、shape 是否参与 JIT cache key。

Phase 1 的 contract 需要做到 DP attention compatible 和 overlap-ready：DP attention 兼容覆盖 target model / target verify 路径；scheduler overlap 只收敛数据边界，不引入 future，也不改变当前 overlap 互斥。`SpecInput`、`EagleDraftInput`、`EagleVerifyInput` 和 `GenerationBatchResult` 的字段语义必须足够明确，使 DP attention 依赖统一 token accounting，未来 overlap 也能把同步流程拆成异步边界，而不是在 worker 主流程里增加普通 EAGLE/MTP 特判。

### SpecInput

`SpecInput` 是所有 spec 输入的共同基类或协议。Phase 1 至少需要固定以下能力：

| 能力 | 用途 |
|---|---|
| `is_draft_input()` | 区分下一轮 draft 状态 |
| `is_verify_input()` | 区分 target verify 输入 |
| `get_spec_adjust_token_coefficient()` | 为后续 DP attention / padding 提供 token accounting 入口 |
| `get_logical_token_num()` | 返回本轮 scheduler 逻辑上应推进的 token 数，通常对应 accepted token 数 |
| `get_allocated_token_num()` | 返回本轮 KV cache 已经或需要预分配的 token 数 |
| `get_verify_token_num()` | 返回 target verify 实际消费的 flatten token 数 |
| `filter_batch()` | running batch 过滤时保持 spec state 同步 |
| `merge_batch()` | chunked prefill 与 running decode 合并时保持 spec state 同步 |

`logical_token_num`、`allocated_token_num` 和 `verify_token_num` 必须区分清楚：前者用于 scheduler 更新 request 逻辑长度和输出；第二个用于 KV cache 分配、释放和 over-allocation 裁剪；第三个用于 target verify 的 attention metadata、padding 和后续 DP attention token accounting。普通 EAGLE 和 MTP 都通过这些入口暴露 token 数，不允许在 `EAGLEWorker`、`MultiLayerEAGLEWorker` 或 scheduler 中按具体 worker 类型重新计算。

### EagleDraftInput

表示下一轮 draft 所需状态：

| 字段 | Phase 1 语义 |
|---|---|
| `topk_p` | 上一轮 draft/draft_extend 得到的 top-k 概率，device array，供下一轮 draft 消费 |
| `topk_index` | 上一轮 draft/draft_extend 得到的 top-k token ids，device array，供下一轮 draft 消费 |
| `hidden_states` | draft 模型下一步需要的 hidden states，device array，只保存下一轮 draft 起点所需的最小状态 |
| `verified_id` | 已确认 token，device array，用作下一轮 draft 输入起点 |
| `accept_length` | draft extend 阶段使用的接受长度，host array，用于选择 accepted hidden states 和更新 draft state |
| `allocate_lens` | 当前已经分配到 req_to_token_pool 的 KV 长度，host array，用于下一轮预分配和 over-allocated slot 释放 |
| `new_seq_lens` | verify 后得到的新 seq_lens，host array，表示 scheduler 逻辑长度，Phase 1 先保留为可选字段 |

`EagleDraftInput` 是下一轮 draft 的唯一持久 spec state。它不能持有 worker、runner、attention backend、future、pool handle 或 scheduler batch 引用；overlap 阶段如果要把 draft 和 verify 拆开，只能通过这些显式字段传递状态。MTP 的 per-step hidden states 可以在 `MultiLayerDraftWorker` 的单轮 draft 内部局部传递，但跨轮只允许落到 `hidden_states` 这个最小下一轮状态中。

### EagleVerifyInput

表示 target verify 所需输入：

| 字段 | Phase 1 语义 |
|---|---|
| `draft_token` | flatten 后的待验证 draft token，device array，target verify 直接消费 |
| `custom_mask` | target verify 的 tree attention mask，device array，shape 参与 JIT cache key |
| `positions` | target verify positions，device/host 边界按当前 `ForwardBatch` 约定保持不变 |
| `retrive_index` | tree verify 索引，host/device 边界按 sampling kernel 当前约定保持不变 |
| `retrive_next_token` | tree child 指针，供 tree sampling 使用 |
| `retrive_next_sibling` | tree sibling 指针，供 tree sampling 使用 |
| `draft_token_num` | 每个 request 的 verify token 数，host array，用于 verify attention metadata 和 DP token accounting |
| `spec_steps` | draft steps，static metadata，不应在同一 precompile shape 内动态变化 |
| `topk` | 每步 top-k，static metadata，不应在同一 precompile shape 内动态变化 |

`EagleVerifyInput` 必须完整描述 target verify 所需的 token、position、mask 和 tree 索引。`BaseSpecWorker.verify()` 不应回读 draft worker 内部临时状态；未来 scheduler overlap 如果把 target verify 延迟执行，也应只需要保存或传递 `EagleVerifyInput` 和 `ModelWorkerBatch` 中已有的显式字段。

Phase 1 不引入 `future_indices` 作为必需字段。overlap 相关字段可以预留，但不进入当前执行路径。

### GenerationBatchResult

`GenerationBatchResult` 是 `EAGLEWorker` 返回给 scheduler 的唯一结果对象。Phase 1 需要明确以下字段语义：

| 字段 | Phase 1 语义 |
|---|---|
| `next_token_ids` | scheduler 追加到 request output 的 token；spec decode 下是每个 request 的 accepted token 列表，表示逻辑输出 |
| `accept_lens` | 每个 request 本轮接受 token 数，包含 bonus token，是 scheduler 推进 request 的主长度 |
| `next_draft_input` | 下一轮 decode 使用的 `EagleDraftInput`，是跨轮唯一持久 draft state |
| `logits_output` | target verify 后按 accepted index 过滤过的 logits/hidden states，device result |
| `allocate_lens` | 本轮 spec 预分配后的 KV 长度，用于 finished request 释放冗余 slot，不能和 `accept_lens` 混用 |
| `new_seq_lens` | verify 后的 scheduler 逻辑长度；若不单独存储，必须可由旧 `seq_lens` 和 `accept_lens` 唯一推出 |

`GenerationBatchResult` 是 scheduler 与 spec worker 之间的异步边界候选对象。Phase 1 仍同步返回它，但字段必须能表达三类长度：`accept_lens` 表示逻辑接受长度，`new_seq_lens` 表示 scheduler 视角的新序列长度，`allocate_lens` 表示 KV cache 已分配长度。overlap 阶段不应通过读取 worker 内部状态补齐这些信息。

### Host / device 边界

Phase 1 需要固定以下边界，避免后续 overlap 或 DP attention 接入时改变 JIT pytree 和 cache key：

| 数据 | 边界 |
|---|---|
| request ids、batch size、`seq_lens`、`accept_lens`、`allocate_lens`、`new_seq_lens` | host 侧 scheduler 状态 |
| `draft_token_num`、padding 长度、page 对齐长度 | host 侧 metadata，可参与 attention metadata 构造 |
| `draft_token`、`positions`、`custom_mask`、`topk_p`、`topk_index`、`hidden_states`、logits | device array，进入 forward 或 sampling |
| `spec_steps`、`topk`、precompile padding shape | static metadata，变化会触发新的编译形态 |

worker、runner、pool、future 和 Python callback 都不能进入 `SpecInput` pytree children，也不能作为 JIT traced 参数。需要跨阶段传递的内容必须是显式 host metadata 或 device array。

### DP attention 兼容约束

Phase 1 的数据契约只对 target model / target verify 路径预留 DP attention 能力，不要求 draft runner 自身支持 DP attention，也不重定义 spec worker / draft worker 主抽象。draft 内部的 hidden states、top-k 和 KV cache 可以继续保持 draft-local order；进入 target verify、返回 scheduler 或继续喂给 logprob 路径前，必须显式转换布局。

DP padded layout 是 target verify、sampling、accepted output 和 scheduler 更新之间的契约：

```text
request-level: [dp0 real reqs + pad | dp1 real reqs + pad | ...]
token-level:   [dp0 real tokens + pad | dp1 real tokens + pad | ...]
```

关键约束：

1. **布局语义。** `EagleVerifyInput` 和 target-side `GenerationBatchResult` 必须携带足够 metadata 描述 DP padded layout。`seq_lens`、`accept_lens`、`allocate_lens`、`new_seq_lens`、`draft_token_num`、`next_token_ids` 不能默认表示原始请求顺序。面向 scheduler 或 logprob 的结果必须通过 `per_dp_bs_size`、`real_bs_per_dp`、`logits_indices_selector` 或等价 selector 还原。
2. **draft-local 状态。** `EagleDraftInput.hidden_states`、`topk_p`、`topk_index`、`verified_id` 和 draft KV cache 可以保持 draft-local request order。draft worker 只负责在生成 `EagleVerifyInput` 前转成 target verify 需要的 DP padded layout，并在 target verify / draft extend 后把下一轮 draft state 还原成 draft-local order。
3. **per-DP token accounting。** `get_logical_token_num()`、`get_allocated_token_num()`、`get_verify_token_num()` 必须按 DP rank 独立计算后再 merge。DP attention 下不能对全局 flat `seq_lens`、`draft_token_num` 或 `extend_seq_lens` 直接 cumsum；attention metadata 必须先 reshape 成 `(dp_size, per_dp_bs_size)` 或等价 per-DP 形态。
4. **rank-local KV。** DP attention 下 KV allocator 返回 rank-local slot index。`allocate_lens`、`out_cache_loc`、`cache_loc`、over-allocation 裁剪、finished request 释放、prefix cache key、KV release 和 paged metadata 都必须保留 `dp_rank`，不能跨 DP rank 复用 rank-local slot。
5. **verify metadata 静态 shape。** `EagleVerifyInput` 必须完整描述 target verify 的 DP-aware metadata。`draft_token_num`、`positions`、`cache_loc`、`custom_mask` 和 tree index 按 rank 独立 padding；`custom_mask` / tree attention mask 的 shape 由 static `spec_steps`、`topk` 和 per-DP padding 策略决定，不能随各 rank 真实 token 数变化。
6. **worker 层不写 DP 特判。** `EAGLEWorker` 和 `MultiLayerEAGLEWorker` 不直接按 worker 类型或 DP rank 计算 split、padding、global token count。DP 差异应沉到 `SpecInput` token accounting、`ScheduleBatch` merge、`ForwardBatch` 初始化、KV allocator 和 attention backend metadata 中。MTP 的多 runner / 多 step 只影响 `MultiLayerDraftWorker` 内部，不改变 scheduler 看到的 DP contract。

### Overlap-ready 契约

Phase 1 的同步流程需要能自然拆成以下边界：

```text
draft_state(EagleDraftInput)
  -> draft_worker.draft(...)
  -> verify_input(EagleVerifyInput)
  -> BaseSpecWorker.verify(...)
  -> verify_result(GenerationBatchResult)
  -> draft_worker.draft_extend_for_decode(...)
  -> next_draft_state(EagleDraftInput)
```

每个边界都只能依赖显式输入输出。`draft_worker.draft()` 产出的 `EagleVerifyInput` 必须足够 target verify 独立消费；`BaseSpecWorker.verify()` 产出的 `GenerationBatchResult` 必须足够 scheduler 更新 request、释放 KV 和构造下一轮 draft state。Phase 1 不保存 future，也不把 `next_token_ids`、`accept_lens`、`next_draft_input` 放入独立 overlap state；但这些字段的 ownership 要按未来可异步消费的方式固定。

如果 Phase 3 采用 async dispatch，dispatch 前必须冻结 `EagleVerifyInput + batch layout + KV allocation snapshot`，completion 后只能通过 `GenerationBatchResult.next_draft_input` 更新跨轮状态。需要特别处理以下字段：

1. **host snapshot 字段。** `req_pool_indices`、`seq_lens`、`extend_seq_lens`、`allocate_lens`、`out_cache_loc`、`cache_loc`、`real_bs`、`real_bs_per_dp`、`per_dp_bs_size` 和 `logits_indices_selector` 必须在 dispatch 前形成不可变快照。scheduler 后续的 running batch filter/merge、DP padding 变化或 request finish 不能影响已经 dispatch 的 verify。
2. **device tensor 字段。** `draft_token`、`positions`、`custom_mask`、`topk_p`、`topk_index`、`hidden_states`、verify logits 和 aux hidden states 不能依赖 worker 内部临时变量，也不能被下一轮 draft 覆盖。需要跨异步边界消费的 tensor 必须作为 `EagleVerifyInput`、`GenerationBatchResult` 或 `EagleDraftInput` 的显式字段存在。
3. **KV 生命周期字段。** `allocate_lens`、`accept_lens`、`new_seq_lens` 和 over-allocated KV slots 的释放必须等 verify result resolve 后再处理。async verify 未完成前，scheduler 不能根据旧 `seq_lens` 或下一轮 batch 状态提前裁剪、复用或释放这些 slot。
4. **输出还原字段。** `next_token_ids`、`accept_lens`、`logits_output`、`logits_indices_selector` 和 DP padded layout metadata 必须足够把 async result 还原到原请求顺序。spec decode 不能假设 completion 时 running batch 的顺序仍和 dispatch 时一致。
5. **下一轮 draft state 字段。** `verified_id`、`hidden_states`、`topk_p`、`topk_index`、`allocate_lens` 和 `new_seq_lens` 只能由 verify completion 与 `draft_extend_for_decode()` 共同生成，作为 `GenerationBatchResult.next_draft_input` 返回。scheduler 不应旁路推断下一轮 draft state，也不应读取 draft worker 内部缓存补齐状态。

因此，overlap-ready 的核心约束不是提前引入 future 类型，而是保证同步路径中的所有跨边界状态都已经显式化。Phase 3 可以把 `GenerationBatchResult` 包进 future 或 async handle，但不应再增加依赖 worker、runner、pool handle 或 scheduler batch 引用的隐式状态。

scheduler overlap 兼容归入 Phase 3；是否前置重构 overlap 逻辑、采用 async dispatch，或继续基于多线程加信号标记实现，需要在独立 RFC 中讨论。

## 文件级改动计划

按以下顺序拆 PR 或 commit。Phase 1 采用“先架构后功能”的推进方式，但不保留当前普通 EAGLE 旧实现的兼容路径；当前普通 EAGLE 性能不可用，允许在 PR1 中用组合式新路径整体替换。

### PR1 骨架重写：Base worker + 单层 EAGLE 新路径

实现状态：PR1 只落地 worker boundary rewrite 与 contract tests；可运行的 MultiLayer MTP 仍属于后续 PR/PR2；DP attention 当前只保持 contract/边界，不在 PR1 启用完整运行路径。

PR1 只建立全新的 worker 边界和普通 EAGLE 最小新路径，不接入 `mimo-v2.5-pro` 多层 MTP。验收目标是让 scheduler、spec worker、draft worker、verify/result contract 的职责边界先稳定下来，后续 PR 再在同一契约上接入 MultiLayer MTP。

PR1 范围：

1. 新增 `BaseSpecWorker` / `BaseDraftWorker` 接口。
2. 新增 `EagleDraftWorker`，承接 draft prefill、draft decode、draft extend 和 draft runner 相关 helper。
3. 重写 `EAGLEWorker` 为组合式 spec worker，只负责编排 target prefill、draft、target verify、draft extend 和 `GenerationBatchResult` 组装。
4. 显式化 `EagleDraftInput`、`EagleVerifyInput`、`GenerationBatchResult` 在 PR1 路径中的 ownership、host/device 边界和跨轮状态归属。
5. scheduler 只切到新的 `EAGLEWorker` 入口，不保留旧 `EAGLEWorker` 双路径或兼容 shim。
6. PR1 不启用 scheduler overlap，不接入 MultiLayer MTP，不要求普通 EAGLE 与旧实现数值/性能兼容。

PR1 非目标：

1. 不新增 `MultiLayerEAGLEWorker` / `MultiLayerDraftWorker` 的可运行 MTP 路径。
2. 不做性能优化或 kernel 重写。
3. 不放开 spec decode 与 overlap scheduler 的互斥。
4. 不为旧普通 EAGLE 行为增加兼容层。

### 1. 新增 base worker 抽象

新增文件：

```text
python/sgl_jax/srt/speculative/base_spec_worker.py
```

内容：

```text
BaseDraftWorker
  - draft(model_worker_batch)
  - draft_extend_for_prefill(...)
  - draft_extend_for_decode(...)

BaseSpecWorker
  - target_worker
  - draft_worker
  - forward_batch_speculative_generation(model_worker_batch)
```

这里定义 PR1 所需接口，并让 `EAGLEWorker` 在同一个 PR 内切到新接口；不保留旧 `EAGLEWorker` 运行路径。

### 2. 拆出 EagleDraftWorker

新增文件：

```text
python/sgl_jax/srt/speculative/eagle_draft_worker.py
```

从现有 `EAGLEWorker` 迁移以下职责。迁移时允许重排方法签名和数据流，不要求保持旧 worker 内部结构兼容：

| 当前方法 | Phase 1 归属 |
|---|---|
| `forward_draft_extend` | `EagleDraftWorker.draft_extend_for_prefill` |
| `capture_for_decode` | `EagleDraftWorker.capture_for_decode` |
| `padding_for_decode` | `EagleDraftWorker.prepare_for_draft_decode` 或内部 helper |
| `draft` | `EagleDraftWorker.draft` |
| `draft_forward` | `EagleDraftWorker.draft_forward` |
| `draft_extend_after_verify` | `EagleDraftWorker.draft_extend_for_decode` |
| `topk_probs_from_logits` | 可保留在 utility 层 |
| `select_top_k_tokens*` | 可保留在 utility 层或继续放同文件 |

### 3. 接入 MultiLayerEAGLEWorker / MultiLayerDraftWorker

该部分从 PR2 开始推进。PR1 只保证 `BaseSpecWorker` / `BaseDraftWorker` 的接口足够承载该路径，实际可执行的多层 MTP 路径不属于 PR1 范围。

新增文件：

```text
python/sgl_jax/srt/speculative/multi_layer_eagle_worker.py
python/sgl_jax/srt/speculative/multi_layer_draft_worker.py
```

1. `MultiLayerEAGLEWorker` 和 `EAGLEWorker` 都实现 `BaseSpecWorker`。
2. `MultiLayerDraftWorker` 和 `EagleDraftWorker` 都实现 `BaseDraftWorker`。
3. `mimo-v2.5-pro` MTP 使用多个独立 draft model runner，每个 runner 对应一个 MTP step/layer。
4. `mimo-v2.5-pro` MTP 的 hidden_states 输入/输出、topk 拼接和 draft_extend 输出必须落在 `MultiLayerDraftWorker` 内部。
5. `MultiLayerEAGLEWorker` 复用 `BaseSpecWorker` 的 verify/data contract 逻辑，不复制普通 EAGLE 的 verify 实现。
6. Phase 1 需要跑通 `mimo-v2.5-pro` MTP 端到端路径

### 4. 重写 EAGLEWorker

修改文件：

```text
python/sgl_jax/srt/speculative/eagle_worker.py
```

职责：

1. 持有 `target_worker`。
2. 初始化 `EagleDraftWorker`。
3. 保持 `forward_batch_speculative_generation()` 入口。
4. 在 prefill 中调用 target forward 和 draft prefill。
5. 在 decode 中调用 draft、verify、draft extend。
6. `verify()` 仍使用现有 `EagleVerifyInput.prepare_for_verify()` 和 `EagleVerifyInput.sample()`。

### 5. 保持 scheduler worker 选择稳定

修改：

```text
python/sgl_jax/srt/managers/scheduler.py
```

当前逻辑：

```text
if spec_algorithm.is_eagle():
    self.draft_worker = EAGLEWorker(...)
```

Phase 1 普通 EAGLE 使用重写后的 `EAGLEWorker(...)`；多层 EAGLE/MTP 从后续 PR 接入 `MultiLayerEAGLEWorker(...)`。scheduler 不保留旧 `EAGLEWorker` 和新 `EAGLEWorker` 的双路径选择。

### 6. DP attention 兼容

Phase 1 需要补齐 DP-aware 数据结构处理。该工作从 PR1 开始固定数据字段语义，但完整 DP attention 兼容可按后续 PR 逐步落地。重点不是改 worker 主抽象，而是让 target verify 相关的 spec state、scheduler batch、model worker batch 和 attention metadata 能表达 DP padded layout。draft model 仍保持 TP / draft-local batch 语义，不要求 draft runner 自身支持 DP attention。

修改文件：

```text
python/sgl_jax/srt/speculative/eagle_util.py
python/sgl_jax/srt/model_executor/forward_batch_info.py
python/sgl_jax/srt/managers/schedule_batch.py
python/sgl_jax/srt/layers/attention/flashattention_backend.py
python/sgl_jax/srt/layers/attention/mla_backend.py
```

需要补齐的内容：

1. 在 `SpecInput` / `EagleVerifyInput` / `GenerationBatchResult` 中明确 target verify 的 DP padded layout 语义；`EagleDraftInput` 只需明确哪些字段保持 draft-local order，哪些字段在进入 target verify 或返回 scheduler 前需要转换。
2. 在 `GenerationBatchResult` 的构造和消费路径中区分 DP padded order、scheduler request order 和 logprob selector order，避免 `next_token_ids`、`accept_lens`、`logits_output` 和 `next_draft_input` 被默认当作普通 flat batch。
3. 在 `ScheduleBatch` 的 spec decode 分支中按 DP rank 独立处理 `allocate_lens`、`out_cache_loc`、`cache_loc`、over-allocation 裁剪和 finished request 释放，保持 KV slot 的 rank-local 语义。
4. 在 `ForwardBatch.init_new()` 中保证 `spec_info` 内的 token/request metadata 与 `input_ids`、`seq_lens`、`positions`、`cache_loc` 使用一致的 DP sharding 和 padding 约定。
5. 在 attention backend metadata 构造中，target verify 不能对全局 flat `draft_token_num` 或 `seq_lens` 直接 cumsum，必须先 reshape 到 per-DP 视图，再按 rank 内部生成 `cu_q_lens`、`cu_kv_lens`、page index 和 mask metadata。
6. 保持 `EAGLEWorker`、`MultiLayerEAGLEWorker`、`EagleDraftWorker` 和 `MultiLayerDraftWorker` 不直接写 DP split 特判；它们只消费和产出已经符合 DP contract 的 `SpecInput` / `GenerationBatchResult`。

### 7. 保持 server args 互斥

Phase 1 不修改：

```text
python/sgl_jax/srt/server_args.py
```

spec decode 仍不兼容, 等 Phase 3 再移除这条限制：

```text
--disable-overlap-schedule
```
