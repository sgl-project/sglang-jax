import logging
import multiprocessing as mp
import os
from collections.abc import Callable

import uvicorn
from fastapi import Request

from sgl_jax.srt.entrypoints.http_server import _GlobalState, app, set_global_state
from sgl_jax.srt.managers.template_manager import TemplateManager
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.global_scheduler import run_global_scheduler_process
from sgl_jax.srt.multimodal.manager.io_struct import (
    ImageGenerationsRequest,
    ImageResponse,
    VideoGenerationsRequest,
    VideoResponse,
)
from sgl_jax.srt.multimodal.manager.multimodal_detokenizer import (
    run_multimodal_detokenizer_process,
)
from sgl_jax.srt.multimodal.manager.multimodal_tokenizer import MultimodalTokenizer
from sgl_jax.srt.server_args import PortArgs
from sgl_jax.srt.utils import kill_process_tree, set_uvicorn_logging_configs

logger = logging.getLogger(__name__)


@app.api_route("/api/v1/images/generation", methods=["POST", "PUT"])
async def images_generation(obj: ImageGenerationsRequest, request: Request):
    print(f"receive req {obj}")
    return ImageResponse(id="test")


@app.api_route("/api/v1/videos/generation", methods=["POST", "PUT"])
async def videos_generation(obj: VideoGenerationsRequest, request: Request):
    print(f"receive req {obj}")
    return VideoResponse(id="test")


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

    # Launch processes
    processes = []

    # 1. Global Scheduler (Main Engine)
    scheduler_proc = mp.Process(
        target=run_global_scheduler_process,
        args=(server_args, port_args),
    )
    scheduler_proc.start()
    processes.append(scheduler_proc)

    # 2. Multimodal Detokenizer
    detokenizer_proc = mp.Process(
        target=run_multimodal_detokenizer_process,
        args=(server_args, port_args),
    )
    detokenizer_proc.start()
    processes.append(detokenizer_proc)

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
