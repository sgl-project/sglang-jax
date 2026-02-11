import httpx
import logging
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable
from http import HTTPStatus

import requests
import uvicorn
from fastapi import Depends, File, Form, Request, UploadFile
from fastapi.responses import ORJSONResponse, Response

from sgl_jax.srt.entrypoints.http_server import _GlobalState, app, set_global_state
from sgl_jax.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sgl_jax.srt.jinja_template_utils import process_content_for_template_format
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.managers.template_manager import TemplateManager
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.global_scheduler import run_global_scheduler_process
from sgl_jax.srt.multimodal.manager.io_struct import (
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    DataType,
    GenerateMMReqInput,
    GenerateVLMReqInput,
    ImageGenerationsRequest,
    VideoGenerationsRequest,
)
from sgl_jax.srt.multimodal.manager.multimodal_detokenizer import (
    run_multimodal_detokenizer_process,
)
from sgl_jax.srt.multimodal.manager.multimodal_tokenizer import MultimodalTokenizer
from sgl_jax.srt.server_args import PortArgs
from sgl_jax.srt.utils import (
    configure_logger,
    kill_process_tree,
    set_uvicorn_logging_configs,
)
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)

DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}User: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}

{% endif %}{% endfor %}{% if add_generation_prompt %}
Assistant: {% endif %}"""


def _create_error_response(e):
    return ORJSONResponse({"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST)


@app.api_route("/api/v1/images/generation", methods=["POST", "PUT"])
async def images_generation(obj: ImageGenerationsRequest, request: Request):
    try:
        from sgl_jax.srt.entrypoints.http_server import _global_state

        internal_obj = await _convert_to_internal_request(obj)
        ret = await _global_state.tokenizer_manager.generate_request(
            internal_obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        logger.error("[http_server] Error: %s", e)
        return _create_error_response(e)


async def _convert_to_internal_request(obj: ImageGenerationsRequest | VideoGenerationsRequest):
    if type(obj) is ImageGenerationsRequest:
        num_frames = 1
        data_type = DataType.IMAGE
        num_inference_steps = obj.num_inference_steps if obj.num_inference_steps is not None else 50
    elif type(obj) is VideoGenerationsRequest:
        num_frames = obj.num_frames
        data_type = DataType.VIDEO
        num_inference_steps = obj.num_inference_steps if obj.num_inference_steps is not None else 50
    else:
        raise Exception(f"not supported type {type(obj)}")
    return GenerateMMReqInput(
        prompt=obj.prompt,
        neg_prompt=obj.neg_prompt,
        size=obj.size,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        data_type=data_type,
        save_output=obj.save_output,
    )


def _extract_openai_prompt(
    request: ChatCompletionRequest,
    tokenizer,
    messages_override: list[dict] | None = None,
) -> tuple[str, list[str] | None, list[str] | None]:
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized for chat completions.")
    openai_messages = []
    image_data = []
    video_data = []
    audio_data = []
    modalities = []

    messages = messages_override if messages_override is not None else request.messages
    for message in messages:
        if message is None:
            continue
        if hasattr(message, "content") and message.content is None:
            message.content = ""
        if hasattr(message, "model_dump"):
            msg_dict = message.model_dump()
        elif isinstance(message, dict):
            msg_dict = {k: v for k, v in message.items() if v is not None}
            if msg_dict.get("content") is None:
                msg_dict["content"] = ""
        else:
            continue
        processed_msg = process_content_for_template_format(
            msg_dict,
            "openai",
            image_data,
            video_data,
            audio_data,
            modalities,
        )
        openai_messages.append(processed_msg)

    if audio_data:
        raise ValueError("Audio inputs are not supported for this model.")

    assistant_prefix = None
    if (
        openai_messages
        and openai_messages[-1].get("role") == "assistant"
        and request.continue_final_message
    ):
        assistant_prefix = openai_messages[-1].get("content")
        openai_messages = openai_messages[:-1]

    tools = None
    if request.tools and request.tool_choice != "none":
        if not isinstance(request.tool_choice, str):
            tools = [
                item.model_dump()
                for item in request.tools
                if item.function.name == request.tool_choice.function.name
            ]
        else:
            tools = [item.model_dump() for item in request.tools]

    chat_template_kwargs = request.chat_template_kwargs or {}
    if (
        not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None
    ) and "chat_template" not in chat_template_kwargs:
        chat_template_kwargs["chat_template"] = DEFAULT_CHAT_TEMPLATE

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support chat templates.")

    try:
        prompt = tokenizer.apply_chat_template(
            openai_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            **chat_template_kwargs,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            openai_messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

    if isinstance(assistant_prefix, str):
        prompt += assistant_prefix
    elif isinstance(assistant_prefix, list):
        text_parts = []
        for part in assistant_prefix:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        if text_parts:
            prompt += "".join(text_parts)

    return prompt, (image_data or None), (video_data or None)


@app.api_route("/api/v1/videos/generation", methods=["POST", "PUT"])
async def videos_generation(obj: VideoGenerationsRequest, request: Request):
    try:
        from sgl_jax.srt.entrypoints.http_server import _global_state

        internal_obj = await _convert_to_internal_request(obj)
        ret = await _global_state.tokenizer_manager.generate_request(
            internal_obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        logger.error("[http_server] Error: %s", e)
        return _create_error_response(e)


# consistent with the openai interface
# https://developers.openai.com/api/reference/python/resources/audio
@app.post("/v1/audio/speech")
async def create_speech(obj: AudioSpeechRequest, request: Request):
    """OpenAI-compatible Text-to-Speech endpoint.

    Returns binary audio data in the specified format.
    """
    try:
        from sgl_jax.srt.entrypoints.http_server import _global_state

        audio_data = await _global_state.tokenizer_manager.create_speech(obj, request)

        # 返回二进制音频流
        media_type = f"audio/{obj.response_format}"
        return Response(content=audio_data, media_type=media_type)
    except ValueError as e:
        logger.error("[http_server] create_speech error: %s", e)
        return _create_error_response(e)


async def parse_transcription_request(
    request: Request,
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
    timestamp_granularities: str | None = Form(None),
    chunking_strategy: str | None = Form(None),
    known_speaker_names: str | None = Form(None),
    known_speaker_references: str | None = Form(None),
    include: str | None = Form(None),
    stream: bool = Form(False),
) -> AudioTranscriptionRequest:
    """Parse multipart/form-data and wrap into AudioTranscriptionRequest."""
    import json

    # Validate input
    if file is None and url is None:
        raise ValueError("Either 'file' or 'url' parameter is required")
    if file is not None and url is not None:
        raise ValueError("Cannot provide both 'file' and 'url' parameters")

    # Load audio bytes
    audio_bytes = None
    if file is not None:
        audio_bytes = await file.read()
    elif url is not None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                audio_bytes = response.content
        except httpx.HTTPError as e:
            raise ValueError(f"Failed to download audio from URL: {e}") from e

    # Parse JSON string parameters
    granularities = json.loads(timestamp_granularities) if timestamp_granularities else None
    chunking = json.loads(chunking_strategy) if chunking_strategy else None
    speaker_names = json.loads(known_speaker_names) if known_speaker_names else None
    speaker_refs = json.loads(known_speaker_references) if known_speaker_references else None
    include_list = json.loads(include) if include else None

    return AudioTranscriptionRequest(
        file=audio_bytes,
        url=url,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=granularities,
        chunking_strategy=chunking,
        known_speaker_names=speaker_names,
        known_speaker_references=speaker_refs,
        include=include_list,
        stream=stream,
    )


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    obj: AudioTranscriptionRequest = Depends(parse_transcription_request),
):
    try:
        from sgl_jax.srt.entrypoints.http_server import _global_state

        result = await _global_state.tokenizer_manager.create_transcription(obj, request)

        # Return different formats based on response_format
        if obj.response_format == "text":
            return Response(content=result, media_type="text/plain")
        elif obj.response_format in ("srt", "vtt"):
            return Response(content=result, media_type="text/plain")
        else:  # json, verbose_json, diarized_json
            return result

    except ValueError as e:
        logger.error("[http_server] create_transcription error: %s", e)
        return _create_error_response(e)


@app.post("/v1/chat/completions")
async def chat_completions(obj: ChatCompletionRequest, request: Request):
    try:
        from sgl_jax.srt.entrypoints.http_server import _global_state

        prompt, image_data, video_data = _extract_openai_prompt(
            obj, _global_state.tokenizer_manager.tokenizer
        )
        if not image_data and not video_data:
            try:
                raw_body = await request.json()
            except Exception:
                raw_body = None
            raw_messages = raw_body.get("messages") if isinstance(raw_body, dict) else None
            if isinstance(raw_messages, list):
                prompt, image_data, video_data = _extract_openai_prompt(
                    obj, _global_state.tokenizer_manager.tokenizer, raw_messages
                )
        max_new_tokens = obj.max_completion_tokens or obj.max_tokens
        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": obj.temperature,
            "top_p": obj.top_p,
            "top_k": obj.top_k,
            "min_p": obj.min_p,
            "frequency_penalty": obj.frequency_penalty,
            "presence_penalty": obj.presence_penalty,
            "repetition_penalty": obj.repetition_penalty,
            "stop": obj.stop,
        }
        internal_obj = GenerateVLMReqInput(
            prompt=prompt,
            image_data=image_data,
            video_data=video_data,
            stream=obj.stream,
            n=obj.n,
            rid=obj.rid if isinstance(obj.rid, str) else None,
            sampling_params=sampling_params,
            stop=obj.stop,
        )
        ret = await _global_state.tokenizer_manager.generate_request(
            internal_obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        logger.error("[http_server] Error: %s", e)
        return _create_error_response(e)


@app.post("/abort_request")
async def abort_request(obj: AbortReq, request: Request):
    """Abort a multimodal generation request.

    This endpoint allows clients to cancel in-flight multimodal generation
    requests by their request id (rid). The abort is propagated through
    the tokenizer, scheduler, and stages to cancel any associated work.
    """
    try:
        from sgl_jax.srt.entrypoints.http_server import _global_state

        _global_state.tokenizer_manager.abort_request(rid=obj.rid, abort_all=obj.abort_all)
        return Response(status_code=200)
    except Exception as e:
        logger.error("[http_server] abort_request error: %s", e)
        return _create_error_response(e)


def launch(
    server_args: MultimodalServerArgs,
    pipe_finish_writer: mp.connection.Connection | None = None,
    launch_callback: Callable[[], None] | None = None,
):
    """
    Launch SJMRT (SGLang_JAX_Multimodal Runtime) Server.

    The SJMRT server consists of an HTTP server, and a engine which composed by several threads.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of several thread:
        1. MultimodalTokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. multimodal_main_engine (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
            2.1 global_scheduler (thread): Manage Request lifestyle
            2.2 Stage * N (thread) forward request by different stage, which have different devices and mesh
        3. MultimodalDetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and MultimodalTokenizerManager both run in the main process.
    2. Communation within HTTP server <-> MultimodalTokenizerManager <-> MultimodalDetokenizerManager <-> Engine via the ZMQ library.
    3. GlobalScheduler and Stage * N is in the same process.
    """
    # Allocate ports
    port_args = PortArgs.init_new(server_args)
    mp.set_start_method("spawn", force=True)
    # Launch processes
    processes = []

    # 1. Global Scheduler (Main Engine)
    scheduler_pipe_readers = []
    scheduler_procs = []
    reader, writer = mp.Pipe(duplex=False)
    scheduler_proc = mp.Process(
        target=run_global_scheduler_process,
        args=(server_args, port_args, writer),
    )
    scheduler_pipe_readers.append(reader)
    scheduler_proc.start()
    processes.append(scheduler_proc)
    scheduler_procs.append(scheduler_proc)

    # 2. Multimodal Detokenizer
    detokenizer_proc = mp.Process(
        target=run_multimodal_detokenizer_process,
        args=(server_args, port_args),
    )
    detokenizer_proc.start()
    processes.append(detokenizer_proc)

    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(
                "Node %s jax_scheduler is dead. Please check if there are relevant logs.",
                i,
            )
            scheduler_procs[i].join()
            logger.error("Exit code: %s", scheduler_procs[i].exitcode)
            raise

        if data["status"] != "ready":
            raise RuntimeError("Initialization failed. Please see the error messages above.")
    # 3. Multimodal Tokenizer (In-process)
    tokenizer_manager = MultimodalTokenizer(server_args, port_args)

    # Initialize Template Manager
    template_manager = TemplateManager()
    # template_manager.initialize_templates(model_path=server_args.model_path) # Optional: Init if needed

    # Set global state for the app
    # Scheduler info is not yet available from the separate process, using empty dict for now
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info={},
        )
    )

    # Send a warmup request - we will create the thread and launch it
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
        configure_logger(server_args)
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
        for p in processes:
            if p.is_alive():
                p.terminate()
        kill_process_tree(os.getpid())


def _is_wan_model(model_path: str) -> bool:
    """Check if the model is a Wan model based on model path."""
    return "wan" in model_path.lower()

def _execute_multimodal_server_warmup(
    server_args: MultimodalServerArgs,
    pipe_finish_writer: mp.connection.Connection | None,
) -> bool:
    """Execute warmup request for multimodal server.

    For Wan models, sends an image generation request as warmup.
    """
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    # Wait until the server is launched
    success = False
    last_traceback = None
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(url + "/health", timeout=5, headers=headers)
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()

    if not success:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error("Initialization failed. warmup error: %s", last_traceback)
        kill_process_tree(os.getpid())
        return False

    # Send a warmup request
    # For Wan models, send an image generation request
    if "Qwen2.5-VL" in server_args.model_path:
        request_endpoint = "/v1/chat/completions"
        json_data = {
            "model": "Qwen/Qwen2.5-VL",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What can you see in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://github.com/IS-Model-Framework/sglang-jax/blob/dev/vl/test/srt/example_image.png?raw=true"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 3,
        }
    elif _is_wan_model(server_args.model_path):
        request_endpoint = "/api/v1/images/generation"
        json_data = {
            "prompt": "warmup request",
            "size": "480*832",
            "num_inference_steps": 2,
            "save_output": False,
        }
    elif "MiMo-Audio" in server_args.model_path:
        request_endpoint = "/v1/audio/transcriptions"
        # audio_url = "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3"
        audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_zh.wav"
        logger.info("Downloading warmup audio from: %s", audio_url)
        audio_response = requests.get(audio_url, timeout=30)
        audio_bytes = audio_response.content

        files = {"file": ("warmup_audio.mp3", audio_bytes, "audio/mpeg")}
        data = {"model": server_args.model_path}

        try:
            res = requests.post(
                url + request_endpoint,
                files=files,
                data=data,
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200, f"{res.status_code}: {res.text}"
            logger.info("Audio model warmup completed successfully")
        except Exception:
            last_traceback = get_exception_traceback()
            if pipe_finish_writer is not None:
                pipe_finish_writer.send(last_traceback)
            logger.error("Initialization failed. warmup error: %s", last_traceback)
            kill_process_tree(os.getpid())
            return False

        return True
    else:
        # Default to image generation for other multimodal models
        request_endpoint = "/api/v1/images/generation"
        json_data = {
            "prompt": "warmup request",
            "size": "480*832",
            "num_inference_steps": 2,
            "save_output": False,
        }
    try:
        res = requests.post(
            url + request_endpoint,
            json=json_data,
            headers=headers,
            timeout=600,
        )
        assert res.status_code == 200, f"{res}"
    except Exception:
        last_traceback = get_exception_traceback()
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error("Initialization failed. warmup error: %s", last_traceback)
        kill_process_tree(os.getpid())
        return False

    return True


def _wait_and_warmup(
    server_args: MultimodalServerArgs,
    pipe_finish_writer: mp.connection.Connection | None,
    launch_callback: Callable[[], None] | None = None,
):
    """Wait for server to start and execute warmup request."""
    if not server_args.skip_server_warmup and not _execute_multimodal_server_warmup(
        server_args,
        pipe_finish_writer,
    ):
        return

    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")

    logger.info("The server is fired up and ready to roll!")

    if launch_callback is not None:
        launch_callback()
