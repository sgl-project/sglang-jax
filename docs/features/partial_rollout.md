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

```python
def continue_generation(self):
    """
    Description: continue previous paused generation
    """
    obj = ContinueGenerationReqInput()
    return self.loop.run_until_complete(self.tokenizer_manager.continue_generation(obj))
```

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

#### 3.1.1 Request Input Structures (`io_struct.py`)

##### A. `PauseGenerationReqInput`

```python
@dataclass
class PauseGenerationReqInput(BaseReq):
    mode: Literal["abort", "retract", "in_place"] = "abort"
    def __post_init__(self):
        allowed = ["abort", "retract", "in_place"]
        if self.mode not in allowed:
            raise ValueError(f"Invalid mode: {self.mode!r}. " f"Expected one of {allowed}.")
```

##### B. `ContinueGenerationReqInput`
```python
@dataclass
class ContinueGenerationReqInput(BaseReq):
    pass
```

##### C. `FlushCacheReqInput`
```python
@dataclass
class FlushCacheReqInput(BaseReq):
    pass
```

##### D. `FlushCacheReqOutput`
```python
@dataclass
class FlushCacheReqOutput(BaseReq):
    success: bool
    flushed_items: int = 0
    error_msg: str = ""
```
---