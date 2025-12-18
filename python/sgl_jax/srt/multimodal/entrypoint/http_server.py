from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs


def launch(
    server_args: MultimodalServerArgs,
    # pipe_finish_writer: multiprocessing.connection.Connection | None = None,
    # launch_callback: Callable[[], None] | None = None,
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
