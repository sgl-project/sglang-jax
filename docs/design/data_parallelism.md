# Data Parallelism Design Document for SGLang-JAX

## 1. Motivation and Scope

Many modern LLM architectures (GQA, MLA, etc.) significantly reduce the number of KV heads. When sharded using larger Tensor Parallelism (TP), each TP shard replicates the same heads, leading to redundant KV cache storage and attention computation.

Data Parallelism (DP) allows KV heads to be sharded across DP rather than replicated across TP, saving memory and improving utilization. It also enables multi-replica inference without launching multiple independent servers — a single controller coordinates the replicated model weights.

### Focus Areas

- **Control Plane Design**: How requests are scheduled to DP, and how DP interacts with the existing single-controller scheduler.
- **Model Forward**: Compatibility under DP for ragged attention kernels and the dense model stack.

## 2. Requirements and Non-Goals

### Functional Requirements

- Support Data Parallelism for dense models on TPUs.
- Maintain Single-Controller as the default mental model.
- Ensure prefix cache / KV cache compatibility under DP.
- Shortest-queue hybrid strategy scheduling.

### Non-Goals

- No multi-controller / multi-process DDP-style inference orchestration.
- No DP+MoE support (to be designed separately).
- No cache-aware scheduling in DP at this stage.

## 3. Architecture Comparison

### SGLang (PyTorch)

SGLang DP consists of multiple independent DP instances executing different user requests in parallel. A `DataParallelController` handles load balancing at the DP level, forwarding requests and receiving results. Each `Scheduler` only sees its own portion of requests (with `dp_rank`).

**Drawbacks**:
1. Weak load balancing support (round-robin is naive; shortest-queue and prefix-cache-aware are complex to implement).
2. Not single-controller anymore — different DPs have different Python control flows.

### SGLang-JAX (Current)

SGLang-JAX uses a replicated state machine architecture. Each controller runs the same Python script, calls `jax.distributed.initialize()` for handshake and topology construction, and initiates the same model run — following the SPMD + single controller paradigm.

## 4. Control Plane Design: DP-Aware Single-Controller

### Core Concept

Maintain a single central controller:
- Replicate prefix cache metadata (Radix Cache) and KVCache Allocator to `DP_SIZE` copies, distinguished by `dp_id`.
- When scheduling requests, be aware of all DPs' prefix matching and available space.
- Multi-host still uses replicated state machine architecture.

### Initialization Flow

1. Scheduler creates `ModelRunner`.
2. `ModelRunner` initializes `ReqToTokenPool` (no DP replication needed).
3. Initialize `TokenToKVPool` with shape `[page_nums * page_size, head_num*2, head_dim]`, `PartitionSpec('data', 'tensor', None)`.
4. Initialize `TokenToKVPoolAllocator` with `DP_SIZE` attribute — allocating/releasing token space requires specifying `DP_Index` when `dp > 1`.
5. Initialize Prefix Cache (Radix Cache) with `DP_SIZE` attribute — prefix key = `[token_prefix, dp_id]` when `dp > 1`.
6. `SchedulePolicy` adds DP-related info and supports strategies like `prefix_aware` / `shortest_queue`.

### Request Scheduling Flow

1. Tokenizer receives requests, forwards to Scheduler.
2. Scheduler broadcasts requests to all schedulers via `recv_requests`.
3. `req.dp_id = self.select_dp_for_request(req)` — selects a `DP_ID` per policy (e.g., shortest running queue, round-robin).
4. `get_next_batch_to_run` schedules new requests (prefix matching, `PrefillAdder`).
5. `update_running_batch` updates batch.
6. In `run_batch`, reorder metadata by DP. Example:
   - DP=2, 4 requests `{0, 1, 2, 3}`, `ReqsToDPMap={0:0, 1:1, 2:1, 3:0}`
   - `input_ids` are reordered so that DP0's tokens occupy the first half and DP1's the second half.
   - `seq_lens`, `out_cache_loc`, `cache_loc`, `positions`, etc. are all reordered with `PartitionSpec('data')`.
7. Model worker executes forward.
8. `process_batch_result` extracts results, reorders back to original request order.

### Advantages

- **Unified mental model**: One Controller = One Scheduler = one SPMD entry point, just DP-aware.
- **Stronger global optimization**: Global rate limit, batching, DP-aware admission control are straightforward.
- **Safer prefix reuse**: Scheduler is aware of DP topology, avoiding incorrect cross-DP prefix reuse.

### Disadvantages

- More invasive implementation across the controller stack.
- Heavier centralized state and CPU overhead.

## 5. Model Forward: DP Support for Dense Models

### 5.1 Sharding Strategy

Mesh axes: `('data', 'tensor')`

| Component | Sharding |
|-----------|----------|
| Inputs (`input_ids`, `positions`, `cache_loc`, etc.) | `P('data')` |
| KV cache (`num_pages, page_size, num_kv_heads, head_dim`) | `P('data', None, 'tensor', None)` |
| `hidden_states` | `P('data', None)` |
| `q_proj` / `k_proj` / `v_proj` | `P(None, 'tensor')` |
| `o_proj` | `P('tensor', None)` |
| `gate_proj`, `up_proj` | `P(None, 'tensor')` |
| `down_proj` | `P('tensor', None)` |
| `q`, `k`, `v` | `P('data', 'tensor')` |
| `logits` | `P('data', 'tensor')` |
| sampled tokens | `P('data')` |

### 5.2 Ragged Attention Kernel Compatibility

Requirements:
- The first dimension (token / request / page_indices) is treated as the DP dimension.
- Compatible with `shard_map` / `NamedSharding` to run along the data axis.
- Does not assume global continuity of token indices across DP ranks.

The FlashAttention kernel is wrapped in `jax.shard_map`: `shard_map` handles cross-device parallelism (SPMD), while the Pallas kernel handles single-device computation. The implementation adds `in_specs` and `out_specs` for DP partitioning:

```python
in_specs = (
    P('data', self.kv_partition_axis),  # queries
    P('data', self.kv_partition_axis),  # keys
    P('data', self.kv_partition_axis),  # values
    P('data', None, self.kv_partition_axis, None),  # kv_cache_fused
    P('data'),  # kv_lens
    P('data'),  # page_indices
    P('data'),  # cu_q_lens
    P('data'),  # cu_kv_lens
    P(),        # distribution
    P(),        # custom_mask
)
out_specs = (
    P('data', self.kv_partition_axis),  # attention output
    P('data', self.kv_partition_axis, None),  # updated kv_cache_fused
)
```

### 5.3 LogitsProcessor & Sampler

- **Sampler**: Change sharding in `_regular_sampling` from `P(None, None)` to `P('data', None)`.
- **LogitsProcessor**: Add sharding constraint `NamedSharding(mesh, P('data', None))` to `all_logprobs` in `get_top_logprobs` and `get_token_ids_logprobs`.
- **SamplingMetadata**: Add `dp_size` parameter to `from_model_worker_batch`.
- **Penalty**: Reorder penalty tensors and apply `P('data', 'tensor')` sharding.

## 6. Work Breakdown

### Core Components
- Enable DP in `TokenToKVPool` and `TokenToKVPoolAllocator`.
- Enable DP in Prefix Cache (Radix Cache).

### Scheduling Layer
- Add DP support in Scheduler initialization and scheduling logic.
- Rework metadata handling in `get_model_worker_batch` and `process_batch_result`.
- Support DP-aware scheduling strategies.

### Model Layer (parallel with scheduling)
- Enable DP in ragged attention kernel.
- Add DP support for logits and sampler.
- Enable full model-level DP support.

### Other
- Integration testing and performance benchmarking.
- Penalty, logprobs, and speculative decoding with DP support.
- DP-related load monitoring and metrics.
