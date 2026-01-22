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
    rid: str = ""                              # Request ID to abort
    abort_all: bool = False                    # Whether to abort all requests
    finished_reason: dict[str, Any] | None = None  # Reason for abort (internal use)
    aborted_message: str | None = None         # Optional abort message
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
