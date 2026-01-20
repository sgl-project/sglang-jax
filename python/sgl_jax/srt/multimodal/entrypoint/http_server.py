import logging
import multiprocessing as mp
import os
from collections.abc import Callable
from http import HTTPStatus

import uvicorn
from fastapi import Request
from fastapi.responses import ORJSONResponse, Response

from sgl_jax.srt.entrypoints.http_server import _GlobalState, app, set_global_state
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.managers.template_manager import TemplateManager
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.global_scheduler import run_global_scheduler_process
from sgl_jax.srt.multimodal.manager.io_struct import (
    DataType,
    GenerateMMReqInput,
    ImageGenerationsRequest,
    VideoGenerationsRequest,
)
from sgl_jax.srt.multimodal.manager.multimodal_detokenizer import (
    run_multimodal_detokenizer_process,
)
from sgl_jax.srt.multimodal.manager.multimodal_tokenizer import MultimodalTokenizer
from sgl_jax.srt.server_args import PortArgs
from sgl_jax.srt.utils import kill_process_tree, set_uvicorn_logging_configs

logger = logging.getLogger(__name__)


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
        for p in processes:
            if p.is_alive():
                p.terminate()
        kill_process_tree(os.getpid())
