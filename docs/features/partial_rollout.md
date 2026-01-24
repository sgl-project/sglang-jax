# SGLang-Jax Interruptible Sampling

## 1. Motivation
The motivation for supporting pause generation and continue generation is to support interruptible sampling in RL framework, especially for google/Tunix.
Design
Generally, the request flow for sglang-jax engine is: http request → tokenizer manager → scheduler. The http request contains requests of PauseGenerationReqInput or ContinueGenerationReqInput. Then, tokenizer manager will send the request to the scheduler through ZMQ. In the scheduler, the event loop will process the input request and invoke the handler based on the type of input request.
According to endpoints required in docs, SGLangJax needs to provide four APIs. They are /abort_request, /flush_cache, /pause_generation and /continue_generation. Besides APIs in http_server, we will add corresponding functions for class.Engine. Currently Tunix uses Engine.generate to get outputs.

## 2. Detailed Design

### 2.1 API Modifications

#### 2.1.1 HTTP API

```python
@app.post("/pause_generation")
async def pause_generation(obj: PauseGenerationReqInput, request: Request):
    """
    Description: Deal with requests according to mode. Now support abort, in_place and retract.
    Input: PauseGenerationReqInput, request obj for pause generation
    Return: status code indicating it work correctly
    """
    await _global_state.tokenizer_manager.pause_generation(obj)
    return ORJSONResponse(
        content={"message": "Generation paused successfully.", "status": "ok"},
        status_code=200,
    )
```

**Pause Generation Behavior (implemented in `Scheduler.pause_generation` at `scheduler.py`):**

1. **Overlap Mode Handling**: If the scheduler is running in overlap mode and there is a `last_batch` still in flight, the scheduler will first finish processing that batch before pausing. This ensures no in-flight computation is left hanging.

2. **Mode-specific behavior**:
   - **`in_place` mode**: The scheduler's event loop is paused, but the KV cache is **preserved**. All requests remain in their current state and will resume from where they left off when `continue_generation` is called.
   - **`retract` mode**: The scheduler's event loop is paused, and the KV cache is **cleared**. All unfinished requests in the `running_batch` are retracted and moved back to the `waiting_queue`. When `continue_generation` is called, these requests will need to redo prefill computation.

```python
@app.post("/continue_generation")
async def continue_generation(obj: ContinueGenerationReqInput, request: Request):
    """
    Description: continue previous paused generation
    Input: ContinueGenerationReqInput, request obj to indicate continue generating
    Return: status code indicating it work correctly
    """
    await _global_state.tokenizer_manager.continue_generation(obj)
    return ORJSONResponse(
        content={"message": "Generation continued successfully.", "status": "ok"},
        status_code=200,
    )

```

**Continue Generation Behavior (implemented in `Scheduler.continue_generation` at `scheduler.py`):**

- Sets `_engine_paused = False` to resume the scheduler's event loop.
- If previously paused with `in_place` mode: generation continues with the existing KV cache, no recomputation needed.
- If previously paused with `retract` mode: requests in the waiting queue will be scheduled for prefill again since their KV cache was cleared.

```python
@app.post("/abort_request")
async def abort_request(obj: AbortReq, request: Request):
    """Abort a request."""
    try:
        _global_state.tokenizer_manager.abort_request(rid=obj.rid, abort_all=obj.abort_all)
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)
```

```python
@app.api_route("/flush_cache", methods=["GET", "POST"])
async def flush_cache():
    """
    Descriptioin: requests will be sent to tokenizer manager. It will flush all cache: tree_cache, req_to_token_pool, token_to_kv_pool_allocator(free physical cache through allocator)

    Return: status code indicating it work correctly
    """
    ret = await _global_state.tokenizer_manager.flush_cache()
    content = (
        "Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n"
    )
    if ret.success and ret.flushed_items:
        content += f"Flushed items: {ret.flushed_items}\n"
    elif not ret.success and ret.error_msg:
        content += f"Reason: {ret.error_msg}\n"
    return Response(
        content=content,
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )
```

#### 2.1.2 Engine API

```python
def pause_generation(self, mode: str = "retract"):
    """
    Input: the pause generation mode: ["abort", "retract", "in-place"]

    Description: Deal with requests according to mode. Now support abort, in_place and retract.
    """
    obj = PauseGenerationReqInput(mode=mode)
    return self.loop.run_until_complete(self.tokenizer_manager.pause_generation(obj))
```

**Engine Pause Generation Behavior:**

The `Engine.pause_generation()` method wraps the `TokenizerManager.pause_generation()` call. The actual logic is implemented in `Scheduler.pause_generation()` (see `scheduler.py:1465-1491`):

1. **Overlap Mode**: When `enable_overlap=True` and a `last_batch` exists, the scheduler finishes processing the in-flight batch result before pausing to avoid leaving computations in an inconsistent state.

2. **`in_place` mode**:
   - Pauses the scheduler without touching the KV cache
   - Requests stay in `running_batch` with their computed KV cache intact
   - Ideal for short pauses where you want to resume quickly

3. **`retract` mode**:
   - Clears the KV cache via `running_batch.retract_all()`
   - Moves all unfinished requests from `running_batch` back to `waiting_queue`
   - Allows `flush_cache` to be called during pause
   - Requests will redo prefill when generation continues

```python
def continue_generation(self):
    """
    Description: continue previous paused generation
    """
    obj = ContinueGenerationReqInput()
    return self.loop.run_until_complete(self.tokenizer_manager.continue_generation(obj))
```

**Engine Continue Generation Behavior:**

The `Engine.continue_generation()` method resumes a previously paused engine. The actual logic is in `Scheduler.continue_generation()` (see `scheduler.py:1489-1491`):

- Sets `_engine_paused = False` to allow the scheduler's event loop to resume processing
- The behavior after resumption depends on how the engine was paused:
  - After `in_place` pause: Continues decoding from the preserved KV cache state
  - After `retract` pause: Schedules retracted requests for new prefill computation since KV cache was cleared

```python
def abort_request(self, rid: str | None = None, abort_all: bool = False):
    """
    Description: Abort a request.

    Input: rid is request id, abort_all determines whether abort all requests
    """
    self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)
```

```python
def flush_cache(self):
        """
        Descriptioin: requests will be sent to tokenizer manager. It will flush all cache: tree_cache, req_to_token_pool, token_to_kv_pool_allocator(free physical cache through allocator)
        """
        return self.loop.run_until_complete(self.tokenizer_manager.flush_cache())
```

---

## 3. Request/Response Data Structures (`io_struct.py`)

### 3.1 `AbortReq`

Request structure for aborting one or all requests.

```python
@dataclass
class AbortReq(BaseReq):
    # Whether to abort all requests
    abort_all: bool = False
    finished_reason: dict[str, Any] | None = None
    aborted_message: str | None = None
```

---

### 3.2 `PauseGenerationReqInput`

Request structure for pausing generation with different modes.

```python
@dataclass
class PauseGenerationReqInput(BaseReq):
    mode: Literal["abort", "retract", "in_place"] = "abort"
```

**Mode Descriptions:**


`abort` Abort and return all requests currently being processed.
`in_place` Pause the scheduler's event loop; requests stay in place with their KV cache preserved.
`retract` Pause the scheduler and retract all running requests back to the waiting queue.

---

### 3.3 `ContinueGenerationReqInput`

Request structure for resuming paused generation.

```python
@dataclass
class ContinueGenerationReqInput(BaseReq):
    pass
```

---

### 3.4 `FlushCacheReqInput`

Request structure for flushing all cached data.

```python
@dataclass
class FlushCacheReqInput(BaseReq):
    pass
```

---

### 3.5 `FlushCacheReqOutput`

Response structure returned after a flush cache operation.

```python
@dataclass
class FlushCacheReqOutput(BaseReq):
    success: bool
    flushed_items: int = 0
    error_msg: str = ""
```

---

## 4. Test Cases

This section describes the test cases for verifying the interruptible sampling functionality.

---

### 4.1 `test_engine_pause_continue.py`

This test file verifies that `pause_generation` and `continue_generation` APIs work correctly with different pause modes.

**Test Cases:**

1. **`test_1_pause_generation_retract_mode`**: Tests the `retract` mode behavior:
   - Verifies running batch becomes empty after pause
   - Verifies requests are moved from running batch to waiting queue
   - Verifies KV cache is cleared (available tokens increase)
   - Verifies requests complete successfully after `continue_generation`

2. **`test_2_pause_generation_in_place_mode`**: Tests the `in_place` mode behavior:
   - Verifies running batch size remains unchanged after pause
   - Verifies same requests stay in running batch (by checking request IDs)
   - Verifies waiting queue is unchanged
   - Verifies KV cache is NOT cleared (available tokens remain same)
   - Verifies requests complete successfully after `continue_generation`

3. **`test_3_pause_continue_multiple_cycles`**: Tests multiple pause/continue cycles with alternating modes (retract and in_place).

4. **`test_4_pause_generation_abort_mode`**: Tests the `abort` mode behavior:
   - Verifies all requests are aborted (running batch and waiting queue become empty)
   - Verifies requests return with `abort` finish reason

---

### 4.2 `test_engine_flush_cache.py`

This test file verifies that `flush_cache` properly clears all cache components.

**Verified Cache Components:**

- `tree_cache` (radix/prefix cache)
- `req_to_token_pool`
- `token_to_kv_pool_allocator` (KV cache)

**Test Cases:**

1. **`test_1_flush_cache_after_generation`**: Tests that flush_cache restores all states after generation completes:
   - Verifies KV cache is restored (`available_kv_tokens` returns to initial value)
   - Verifies tree cache is cleared (`tree_cache_size` becomes 0)
   - Verifies req_to_token_pool is cleared (`req_to_token_pool_used` becomes 0)
   - Verifies `forward_ct_decode` counter is reset to 0
   - Verifies `new_token_ratio` is reset to initial value

2. **`test_2_flush_cache_clears_scheduling_state`**: Tests that flush_cache clears all scheduling state:
   - Verifies `running_batch_size` becomes 0
   - Verifies `waiting_queue_size` becomes 0
   - Verifies `cur_batch` is None
   - Verifies `last_batch` is None
   - Verifies `chunked_req` is None

3. **`test_3_generation_works_after_flush`**: Tests that generation still works correctly after flush_cache and produces deterministic results with `temperature=0`.

4. **`test_4_multiple_flush_cycles`**: Tests multiple generate-flush cycles work correctly, verifying all states are properly restored after each flush.

---

### 4.3 `test_engine_determine_generation.py`

This test file verifies **deterministic generation** behavior, specifically testing whether pause/continue operations produce the same results as uninterrupted generation when using `temperature=0`.

**Key Question:** If a decode process is interrupted by `pause_generation` (retract mode) and then resumed via `continue_generation`, will the output be identical to an uninterrupted generation?

**Test Cases:**

1. **`test_1_single_request_retract_vs_no_pause`**: Tests single request determinism:
   - Compares output from retract mode (pause → continue) vs baseline (no pause)
   - Verifies outputs are identical with `temperature=0`

2. **`test_2_single_request_abort_vs_no_pause`**: Tests single request abort behavior:
   - Verifies partial text (before abort) is a prefix of baseline
   - Verifies re-generated result after abort matches baseline

3. **`test_3_multiple_requests_retract_vs_no_pause`**: Tests multiple concurrent requests determinism with retract mode.

4. **`test_4_multiple_requests_abort_vs_no_pause`**: Tests multiple concurrent requests abort behavior and re-generation determinism.

**Batch Invariant Issue:**

During testing, we discovered a **batch invariant issue** where JAX operators (specifically `jax.lax.dot_general`) can produce slightly different results depending on batch size. This is a known JAX issue documented in [jax-ml/jax#34080](https://github.com/jax-ml/jax/issues/34080).

**Workaround:** To ensure deterministic results, the test configures the engine with small padding values:

```python
cls.engine = Engine(
    ...
    chunked_prefill_size=4,
    precompile_bs_paddings=[4],
    precompile_token_paddings=[4],
    page_size=4,
    ...
)
```

By setting `chunked_prefill_size`, `precompile_bs_paddings`, and `precompile_token_paddings` to small values (e.g., 4), we minimize the batch size variation and ensure consistent results across pause/continue cycles.
