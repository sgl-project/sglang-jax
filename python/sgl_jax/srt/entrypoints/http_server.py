"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements HTTP APIs for the inference engine via fastapi.
"""

import asyncio
import dataclasses
import json
import logging
import multiprocessing as multiprocessing
import os
import random
import threading
import time
from http import HTTPStatus
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

from contextlib import asynccontextmanager

import orjson
import requests
import uvicorn
import uvloop
from fastapi import Depends, FastAPI, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sgl_jax.srt.entrypoints.engine import _launch_subprocesses_or_threads
from sgl_jax.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
    ScoringRequest,
    V1RerankReqInput,
)
from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sgl_jax.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sgl_jax.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sgl_jax.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
from sgl_jax.srt.entrypoints.openai.serving_score import OpenAIServingScore
from sgl_jax.srt.function_call.function_call_parser import FunctionCallParser
from sgl_jax.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    EmbeddingReqInput,
    GenerateReqInput,
    OpenSessionReqInput,
    ParseFunctionCallReq,
    ProfileReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    SeparateReasoningReqInput,
    SetInternalStateReq,
    StartTraceReqInput,
    StopTraceReqInput,
    TraceStatusReqInput,
)
from sgl_jax.srt.managers.template_manager import TemplateManager
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.reasoning_parser import ReasoningParser
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils import (
    add_api_key_middleware,
    get_bool_env_var,
    kill_process_tree,
    set_uvicorn_logging_configs,
)
from sgl_jax.utils import get_exception_traceback
from sgl_jax.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# Store global states
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: TokenizerManager
    template_manager: TemplateManager
    scheduler_info: Dict


_global_state: Optional[_GlobalState] = None


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    # Initialize OpenAI serving handlers
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_chat = OpenAIServingChat(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_embedding = OpenAIServingEmbedding(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_score = OpenAIServingScore(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_rerank = OpenAIServingRerank(
        _global_state.tokenizer_manager
    )

    server_args: ServerArgs = fast_api_app.server_args
    if server_args.warmups is not None:
        logger.info("Warmup skipped (not implemented)")
        logger.info("Warmup ended")

    warmup_thread = getattr(fast_api_app, "warmup_thread", None)
    if warmup_thread is not None:
        warmup_thread.start()
    yield


# Fast API
app = FastAPI(
    lifespan=lifespan,
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handlers to change validation error status codes
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Override FastAPI's default 422 validation error with 400"""
    exc_str = str(exc)
    errors_str = str(exc.errors())

    if errors_str and errors_str != exc_str:
        message = f"{exc_str} {errors_str}"
    else:
        message = exc_str

    err = ErrorResponse(
        message=message,
        type=HTTPStatus.BAD_REQUEST.phrase,
        code=HTTPStatus.BAD_REQUEST.value,
    )

    return ORJSONResponse(
        status_code=400,
        content=err.model_dump(),
    )


async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )


HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))


##### Native API endpoints #####


@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """Check the health of the inference server by generating one token."""

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    rid = f"HEALTH_CHECK_{time.time()}"

    if _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
        )
    else:
        gri = EmbeddingReqInput(rid=rid, input_ids=[0], sampling_params=sampling_params)

    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(gri, request):
            break

    tic = time.perf_counter()
    task = asyncio.create_task(gen())
    while time.perf_counter() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:
            task.cancel()
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
            _global_state.tokenizer_manager.health_check_failed = False
            return Response(status_code=200)

    task.cancel()
    tic_time = time.strftime("%H:%M:%S", time.localtime(tic))
    last_receive_time = time.strftime(
        "%H:%M:%S", time.localtime(_global_state.tokenizer_manager.last_receive_tstamp)
    )
    logger.error(
        f"Health check failed. Server couldn't get a response from detokenizer for last "
        f"{HEALTH_CHECK_TIMEOUT} seconds. tic start time: {tic_time}. "
        f"last_heartbeat time: {last_receive_time}"
    )
    _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
    _global_state.tokenizer_manager.health_check_failed = True
    return Response(status_code=503)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information."""
    result = {
        "model_path": _global_state.tokenizer_manager.model_path,
        "tokenizer_path": _global_state.tokenizer_manager.server_args.tokenizer_path,
        "is_generation": _global_state.tokenizer_manager.is_generation,
        "preferred_sampling_params": _global_state.tokenizer_manager.server_args.preferred_sampling_params,
    }
    return result


@app.get("/get_server_info")
async def get_server_info():
    # Returns interna states per DP.
    internal_states: List[Dict[Any, Any]] = (
        await _global_state.tokenizer_manager.get_internal_state()
    )
    return {
        **dataclasses.asdict(_global_state.tokenizer_manager.server_args),
        **_global_state.scheduler_info,
        "internal_states": internal_states,
        "version": __version__,
    }


@app.get("/get_load")
async def get_load():
    return await _global_state.tokenizer_manager.get_load()


# example usage:
# curl -s -X POST http://localhost:30000/set_internal_state -H "Content-Type: application/json" -d '{"server_args": {"max_micro_batch_size": 8}}'
@app.api_route("/set_internal_state", methods=["POST", "PUT"])
async def set_internal_state(obj: SetInternalStateReq, request: Request):
    res = await _global_state.tokenizer_manager.set_internal_state(obj)
    return res


# fastapi implicitly converts json in the request to obj (dataclass)
@app.api_route("/generate", methods=["POST", "PUT"])
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                logger.error(f"[http_server] Error: {e}")
                yield b"data: " + orjson.dumps(
                    out, option=orjson.OPT_NON_STR_KEYS
                ) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return ret
        except ValueError as e:
            logger.error(f"[http_server] Error: {e}")
            return _create_error_response(e)


@app.api_route("/generate_from_file", methods=["POST"])
async def generate_from_file_request(file: UploadFile, request: Request):
    """Handle a generate request, this is purely to work with input_embeds."""
    content = await file.read()
    input_embeds = json.loads(content.decode("utf-8"))

    obj = GenerateReqInput(
        input_embeds=input_embeds,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 512,
        },
    )

    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        logger.error(f"Error: {e}")
        return _create_error_response(e)


@app.api_route("/encode", methods=["POST", "PUT"])
async def encode_request(obj: EmbeddingReqInput, request: Request):
    """Handle an embedding request."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


@app.api_route("/classify", methods=["POST", "PUT"])
async def classify_request(obj: EmbeddingReqInput, request: Request):
    """Handle a reward model request. Now the arguments and return values are the same as embedding models."""
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        return _create_error_response(e)


@app.api_route("/flush_cache", methods=["GET", "POST"])
async def flush_cache():
    """Flush the radix cache."""
    ret = await _global_state.tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    """Start profiling."""
    if obj is None:
        obj = ProfileReqInput()

    await _global_state.tokenizer_manager.start_profile(
        output_dir=obj.output_dir,
        start_step=obj.start_step,
        num_steps=obj.num_steps,
        host_tracer_level=obj.host_tracer_level,
        python_tracer_level=obj.python_tracer_level,
    )
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    """Stop profiling."""
    await _global_state.tokenizer_manager.stop_profile()
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


@app.api_route("/start_trace", methods=["GET", "POST"])
async def start_trace_async(obj: Optional[StartTraceReqInput] = None):
    """Start precision tracing."""
    if obj is None:
        obj = StartTraceReqInput()
    try:
        if not precision_tracer._enable_precision_tracer:
            return ORJSONResponse(
                content={
                    "message": "Precision tracer is disabled. Server must be started with --enable-precision-tracer flag.",
                    "status": "error",
                },
                status_code=400,
            )

        # Generate unique output file name if not provided
        if obj.output_file:
            output_file = obj.output_file
        else:
            timestamp = int(time.time())
            unique_suffix = random.randint(1000, 9999)
            output_file = (
                f"debug_outputs/request_traces_{timestamp}_{unique_suffix}.jsonl"
            )

        precision_tracer.start_trace(req_num=obj.req_num, output_file=output_file)
        logger.info(f"[HTTP] Sending trace state to scheduler...")
        trace_state = {
            "precision_tracer": {
                "trace_active": True,
                "max_requests": obj.req_num,
                "output_file": output_file,
            }
        }

        try:
            result = await _global_state.tokenizer_manager.set_internal_state(
                SetInternalStateReq(request_id="trace_state", state_data=trace_state)
            )
            logger.info(f"[HTTP] Set internal state result: {result}")
        except Exception as e:
            logger.info(f"[HTTP] Error setting internal state: {e}")
            precision_tracer.stop_trace()
            return ORJSONResponse(
                content={
                    "message": f"Failed to sync trace state to scheduler: {e}",
                    "status": "error",
                },
                status_code=500,
            )

        return ORJSONResponse(
            content={
                "message": "Precision tracing started successfully.",
                "status": "ok",
                "req_num": obj.req_num,
                "output_file": output_file,
            },
            status_code=200,
        )
    except Exception as e:
        return ORJSONResponse(
            content={
                "message": f"Failed to start tracing: {str(e)}",
                "status": "error",
            },
            status_code=500,
        )


@app.api_route("/stop_trace", methods=["GET", "POST"])
async def stop_trace_async(obj: Optional[StopTraceReqInput] = None):
    """Stop precision tracing."""
    try:
        output_file = precision_tracer.stop_trace()
        print(f"[HTTP] Sending stop trace state to scheduler...")
        trace_state = {
            "precision_tracer": {
                "trace_active": False,
                "max_requests": None,
                "output_file": None,
            }
        }

        try:
            result = await _global_state.tokenizer_manager.set_internal_state(
                SetInternalStateReq(request_id="trace_state", state_data=trace_state)
            )
            print(f"[HTTP] Stop trace internal state result: {result}")
        except Exception as e:
            print(f"[HTTP] Error stopping trace state to scheduler: {e}")
            return ORJSONResponse(
                content={
                    "message": f"Trace stopped locally but failed to sync to scheduler: {e}",
                    "status": "warning",
                    "output_file": output_file,
                },
                status_code=200,
            )

        return ORJSONResponse(
            content={
                "message": "Precision tracing stopped successfully.",
                "status": "ok",
                "output_file": output_file,
            },
            status_code=200,
        )
    except Exception as e:
        return ORJSONResponse(
            content={"message": f"Failed to stop tracing: {str(e)}", "status": "error"},
            status_code=500,
        )


@app.api_route("/trace_status", methods=["GET", "POST"])
async def trace_status_async(obj: Optional[TraceStatusReqInput] = None):
    """Get precision tracing status."""
    try:
        return ORJSONResponse(
            content={
                "status": "ok",
                "trace_active": precision_tracer._trace_active,
                "request_counter": precision_tracer._request_counter,
                "max_requests": precision_tracer._max_requests,
                "output_file": precision_tracer._trace_output_file,
                "active_request_traces": len(precision_tracer._request_traces),
            },
            status_code=200,
        )
    except Exception as e:
        return ORJSONResponse(
            content={
                "message": f"Failed to get trace status: {str(e)}",
                "status": "error",
            },
            status_code=500,
        )


@app.api_route("/release_memory_occupation", methods=["GET", "POST"])
async def release_memory_occupation(
    obj: ReleaseMemoryOccupationReqInput, request: Request
):
    """Release GPU memory occupation temporarily."""
    try:
        await _global_state.tokenizer_manager.release_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/resume_memory_occupation", methods=["GET", "POST"])
async def resume_memory_occupation(
    obj: ResumeMemoryOccupationReqInput, request: Request
):
    """Resume GPU memory occupation."""
    try:
        await _global_state.tokenizer_manager.resume_memory_occupation(obj, request)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/open_session", methods=["GET", "POST"])
async def open_session(obj: OpenSessionReqInput, request: Request):
    """Open a session, and return its unique session id."""
    try:
        session_id = await _global_state.tokenizer_manager.open_session(obj, request)
        if session_id is None:
            raise Exception(
                "Failed to open the session. Check if a session with the same id is still open."
            )
        return session_id
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/close_session", methods=["GET", "POST"])
async def close_session(obj: CloseSessionReqInput, request: Request):
    """Close the session."""
    try:
        await _global_state.tokenizer_manager.close_session(obj, request)
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/configure_logging", methods=["GET", "POST"])
async def configure_logging(obj: ConfigureLoggingReq, request: Request):
    """Configure the request logging options."""
    _global_state.tokenizer_manager.configure_logging(obj)
    return Response(status_code=200)


@app.post("/abort_request")
async def abort_request(obj: AbortReq, request: Request):
    """Abort a request."""
    try:
        _global_state.tokenizer_manager.abort_request(
            rid=obj.rid, abort_all=obj.abort_all
        )
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.post("/parse_function_call")
async def parse_function_call_request(obj: ParseFunctionCallReq, request: Request):
    """
    A native API endpoint to parse function calls from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = FunctionCallParser(tools=obj.tools, tool_call_parser=obj.tool_call_parser)

    # 2) Call the non-stream parsing method (non-stream)
    normal_text, calls = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "normal_text": normal_text,
        "calls": [
            call.model_dump() for call in calls
        ],  # Convert pydantic objects to dictionaries
    }

    return ORJSONResponse(content=response_data, status_code=200)


@app.post("/separate_reasoning")
async def separate_reasoning_request(obj: SeparateReasoningReqInput, request: Request):
    """
    A native API endpoint to separate reasoning from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = ReasoningParser(model_type=obj.reasoning_parser)

    # 2) Call the non-stream parsing method (non-stream)
    reasoning_text, normal_text = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "reasoning_text": reasoning_text,
        "text": normal_text,
    }

    return ORJSONResponse(content=response_data, status_code=200)


@app.post("/pause_generation")
async def pause_generation(request: Request):
    """Pause generation."""
    await _global_state.tokenizer_manager.pause_generation()
    return ORJSONResponse(
        content={"message": "Generation paused successfully.", "status": "ok"},
        status_code=200,
    )


@app.post("/continue_generation")
async def continue_generation(request: Request):
    """Continue generation."""
    await _global_state.tokenizer_manager.continue_generation()
    return ORJSONResponse(
        content={"message": "Generation continued successfully.", "status": "ok"},
        status_code=200,
    )


##### OpenAI-compatible API endpoints #####


@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
    """OpenAI-compatible text completion endpoint."""
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request
    )


@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


@app.post(
    "/v1/embeddings",
    response_class=ORJSONResponse,
    dependencies=[Depends(validate_json_request)],
)
async def openai_v1_embeddings(request: EmbeddingRequest, raw_request: Request):
    """OpenAI-compatible embeddings endpoint."""
    return await raw_request.app.state.openai_serving_embedding.handle_request(
        request, raw_request
    )


@app.get("/v1/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models. OpenAI-compatible endpoint."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]
    model_cards = []
    for served_model_name in served_model_names:
        model_cards.append(
            ModelCard(
                id=served_model_name,
                root=served_model_name,
                max_model_len=_global_state.tokenizer_manager.model_config.context_len,
            )
        )
    return ModelList(data=model_cards)


@app.get("/v1/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieves a model instance, providing basic information about the model."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]

    if model not in served_model_names:
        return ORJSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    return ModelCard(
        id=model,
        root=model,
        max_model_len=_global_state.tokenizer_manager.model_config.context_len,
    )


## SageMaker API
@app.get("/ping")
async def sagemaker_health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.post("/invocations")
async def sagemaker_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


@app.post("/v1/score", dependencies=[Depends(validate_json_request)])
async def v1_score_request(request: ScoringRequest, raw_request: Request):
    """Endpoint for the decoder-only scoring API. See Engine.score() for detailed documentation."""
    return await raw_request.app.state.openai_serving_score.handle_request(
        request, raw_request
    )


@app.api_route(
    "/v1/rerank", methods=["POST", "PUT"], dependencies=[Depends(validate_json_request)]
)
async def v1_rerank_request(request: V1RerankReqInput, raw_request: Request):
    """Endpoint for reranking documents based on query relevance."""
    return await raw_request.app.state.openai_serving_rerank.handle_request(
        request, raw_request
    )


def _create_error_response(e):
    return ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    # Initialize precision tracer enable state in HTTP server process
    precision_tracer.set_enable_precision_tracer(server_args.enable_precision_tracer)

    tokenizer_manager, template_manager, scheduler_info = (
        _launch_subprocesses_or_threads(server_args=server_args, port_args=None)
    )
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Send a warmup request - we will create the thread launch it
    # in the lifespan after all other warmups have fired.
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            launch_callback,
        ),
    )
    app.warmup_thread = warmup_thread

    try:
        # Update logging configs
        set_uvicorn_logging_configs()
        app.server_args = server_args
        # Listen for HTTP requests
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        warmup_thread.join()


def _execute_server_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection],
):
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(url + "/get_model_info", timeout=5, headers=headers)
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return success

    model_info = res.json()

    # Send a warmup request
    request_name = "/generate" if model_info["is_generation"] else "/encode"
    max_new_tokens = 2 if model_info["is_generation"] else 1
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [[10, 11, 12] for _ in range(server_args.dp_size)]
        if server_args.dp_size == 1:
            json_data["input_ids"] = json_data["input_ids"][0]
    else:
        json_data["text"] = ["The capital city of France is"] * server_args.dp_size
        if server_args.dp_size == 1:
            json_data["text"] = json_data["text"][0]

    try:
        res = requests.post(
            url + request_name,
            json=json_data,
            headers=headers,
            timeout=600,
        )
        assert res.status_code == 200, f"{res}"

    except Exception:
        last_traceback = get_exception_traceback()
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return False

    # Debug print
    # logger.info(f"warmup request returns: {res.json()=}")
    return success


def _wait_and_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection],
    launch_callback: Optional[Callable[[], None]] = None,
):
    if not server_args.skip_server_warmup:
        if not _execute_server_warmup(
            server_args,
            pipe_finish_writer,
        ):
            return

    logger.info("The server is fired up and ready to roll!")

    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")

    if launch_callback is not None:
        launch_callback()
