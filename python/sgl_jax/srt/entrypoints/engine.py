"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements python APIs for the inference engine.
"""

import asyncio
import atexit
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import uvloop
import zmq
import zmq.asyncio
from flax import nnx

from sgl_jax.srt.utils.common_utils import SUPPORTED_LORA_TARGET_MODULES
from sgl_jax.utils import traverse_and_update

# ruff: noqa: E402
# Fix a bug of Python threading
threading._register_atexit = lambda *args, **kwargs: None

from sgl_jax.srt.entrypoints.EngineBase import EngineBase
from sgl_jax.srt.hf_transformers_utils import get_generation_config
from sgl_jax.srt.managers.detokenizer_manager import (
    run_detokenizer_process,
    run_detokenizer_thread,
)
from sgl_jax.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sgl_jax.srt.managers.scheduler import (
    run_scheduler_loop_thread_after_create,
    run_scheduler_process,
)
from sgl_jax.srt.managers.template_manager import TemplateManager
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    configure_logger,
    get_zmq_socket,
    kill_process_tree,
    launch_dummy_health_check_server,
    prepare_model_and_tokenizer,
    set_ulimit,
)
from sgl_jax.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class Engine(EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    """

    def __init__(self, **kwargs):
        """
        The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"
            server_args = ServerArgs(**kwargs)

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        self.port_args = PortArgs.init_new(server_args)
        logger.info("server_args=%s", server_args)

        # Launch subprocesses or threads
        tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses_or_threads(
            server_args=server_args,
            port_args=self.port_args,
        )
        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.scheduler_info = scheduler_info
        self.default_sampling_params: dict[str, Any] | None = None
        context = zmq.Context(2)
        self.send_to_rpc = get_zmq_socket(context, zmq.DEALER, self.port_args.rpc_ipc_name, True)

    def generate(
        self,
        prompt: list[str] | str | None = None,
        sampling_params: list[dict] | dict | None = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: list[list[int]] | list[int] | None = None,
        return_logprob: list[bool] | bool | None = False,
        logprob_start_len: list[int] | int | None = None,
        top_logprobs_num: list[int] | int | None = None,
        token_ids_logprob: list[list[int]] | list[int] | None = None,
        stream: bool = False,
    ) -> dict | Iterator[dict]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            stream=stream,
        )
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = loop.run_until_complete(generator.__anext__())
            return ret

    async def async_generate(
        self,
        sampling_params: list[dict] | dict | None = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: list[list[int]] | list[int] | None = None,
        return_logprob: list[bool] | bool | None = False,
        logprob_start_len: list[int] | int | None = None,
        top_logprobs_num: list[int] | int | None = None,
        token_ids_logprob: list[list[int]] | list[int] | None = None,
        stream: bool = False,
    ) -> dict | AsyncIterator[dict]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        obj = GenerateReqInput(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            stream=stream,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream is True:
            return generator
        else:
            return await generator.__anext__()

    def apply_dummy_lora_ab_buffer(self, target_modules: list | None = None):
        if target_modules is None or len(target_modules) == 0:
            logger.warning("No %v is specified, so skip to apply", target_modules)
            return

        if "all" in target_modules:
            target_modules = SUPPORTED_LORA_TARGET_MODULES

        logger.info("Applying dummy LoRA buffers to modules: %v", target_modules)

        model_runner = self.scheduler_info["scheduler"].tp_worker.worker.model_runner
        model_state = nnx.split(model_runner.model)[1]
        new_state = traverse_and_update(model_state, target_modules)
        nnx.update(model_runner.model, new_state)

    def encode(
        self,
        prompt: str | list[str] | list[dict] | list[list[dict]],
    ) -> dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(
            text=prompt,
        )
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = loop.run_until_complete(generator.__anext__())
        return ret

    async def async_encode(
        self,
        prompt: str | list[str] | list[dict] | list[list[dict]],
    ) -> dict:
        """
        Asynchronous version of encode method.

        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(
            text=prompt,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)
        return await generator.__anext__()

    def rerank(
        self,
        prompt: list[list[str]],
    ) -> dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(text=prompt, is_cross_encoder_request=True)
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = loop.run_until_complete(generator.__anext__())
        return ret

    def shutdown(self):
        """Shutdown the engine"""
        kill_process_tree(os.getpid(), include_parent=False)
        if self.server_args.enable_single_process:
            self.send_to_rpc.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def flush_cache(self):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.flush_cache())

    def start_profile(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.tokenizer_manager.start_profile())

    def stop_profile(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.tokenizer_manager.stop_profile())

    def get_server_info(self):
        loop = asyncio.get_event_loop()
        internal_states = loop.run_until_complete(self.tokenizer_manager.get_internal_state())
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),
            **self.scheduler_info,
            "internal_states": internal_states,
            "version": __version__,
        }

    def release_memory_occupation(self, tags: list[str] | None = None):
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.release_memory_occupation(obj, None))

    def resume_memory_occupation(self, tags: list[str] | None = None):
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.resume_memory_occupation(obj, None))

    def score(
        self,
        query: str | list[int] | None = None,
        items: str | list[str] | list[list[int]] | None = None,
        label_token_ids: list[int] | None = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> list[list[float]]:
        """
        Score the probability of specified token IDs appearing after the given (query + item) pair. For example:
        query = "<|user|>Is the following city the capital of France? "
        items = ["Paris <|assistant|>", "London <|assistant|>", "Berlin <|assistant|>"]
        label_token_ids = [2332, 1223] # Token IDs for "Yes" and "No"
        item_first = False

        This would pass the following prompts to the model:
        "<|user|>Is the following city the capital of France? Paris <|assistant|>"
        "<|user|>Is the following city the capital of France? London <|assistant|>"
        "<|user|>Is the following city the capital of France? Berlin <|assistant|>"
        The api would then return the probabilities of the model producing "Yes" and "No" as the next token.
        The output would look like:
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]]


        Args:
            query: The query text or pre-tokenized query token IDs. Must be provided.
            items: The item text(s) or pre-tokenized item token IDs. Must be provided.
            label_token_ids: List of token IDs to compute probabilities for. If None, no token probabilities will be computed.
            apply_softmax: Whether to normalize probabilities using softmax.
            item_first: If True, prepend items to query. Otherwise append items to query.

        Returns:
            List of dictionaries mapping token IDs to their probabilities for each item.
            Each dictionary in the list corresponds to one item input.

        Raises:
            ValueError: If query is not provided, or if items is not provided,
                      or if token IDs are out of vocabulary, or if logprobs are not available for the specified tokens.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                request=None,
            )
        )

    async def async_score(
        self,
        query: str | list[int] | None = None,
        items: str | list[str] | list[list[int]] | None = None,
        label_token_ids: list[int] | None = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> list[list[float]]:
        """
        Asynchronous version of score method.

        See score() for detailed documentation.
        """
        return await self.tokenizer_manager.score_request(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
            item_first=item_first,
            request=None,
        )

    def get_default_sampling_params(self) -> SamplingParams:
        if self.default_sampling_params is None:
            config = get_generation_config(
                self.server_args.model_path,
                self.server_args.trust_remote_code,
                self.server_args.revision,
            )
            if config is not None:
                self.default_sampling_params = config.to_diff_dict()
            else:
                self.default_sampling_params = {}

            if self.server_args.preferred_sampling_params is not None:
                self.default_sampling_params.update(
                    json.loads(self.server_args.preferred_sampling_params)
                )

            available_params = [
                "repetition_penalty",
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "max_new_tokens",
            ]
            if any(p in self.default_sampling_params for p in available_params):
                diff_sampling_param = {
                    p: self.default_sampling_params.get(p)
                    for p in available_params
                    if self.default_sampling_params.get(p) is not None
                }
                self.default_sampling_params = diff_sampling_param

            else:
                self.default_sampling_params = {}

        if self.default_sampling_params:
            return SamplingParams(**self.default_sampling_params)
        return SamplingParams()


def _set_envs_and_config(server_args):
    # Set ulimit
    set_ulimit()

    def sigchld_handler(signum, frame):
        pid, exitcode = os.waitpid(0, os.WNOHANG)
        if exitcode != 0:
            logger.warning(
                "Child process unexpectedly failed with exitcode=%s. pid=%s",
                exitcode,
                pid,
            )
            logger.warning("Child process pid=%s frame=%s", pid, frame)

    signal.signal(signal.SIGCHLD, sigchld_handler)

    # Register the signal handler.
    # The child processes will send SIGQUIT to this process when any error happens
    # This process then clean up the whole process tree
    def sigquit_handler(signum, frame):
        logger.error("Received sigquit from a child process. It usually means the child failed.")
        kill_process_tree(os.getpid())

    signal.signal(signal.SIGQUIT, sigquit_handler)
    if not server_args.enable_single_process:
        # Set mp start method
        mp.set_start_method("spawn", force=True)
    else:
        ## close resource tracker process
        from multiprocessing import resource_tracker

        resource_tracker._resource_tracker._fd = -1


def _launch_subprocesses(
    server_args, port_args: PortArgs | None = None
) -> tuple[TokenizerManager, TemplateManager, dict]:
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info("server_args=%s", server_args)

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    if server_args.dp_size == 1:
        scheduler_pipe_readers = []
        reader, writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=run_scheduler_process,
            args=(
                server_args,
                port_args,
                None,
                writer,
            ),
        )
        # with memory_saver_adapter.configure_subprocess():
        proc.start()
        scheduler_procs.append(proc)
        scheduler_pipe_readers.append(reader)
    else:
        pass

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                "Scheduler or DataParallelController %s terminated with %s",
                proc.pid,
                proc.exitcode,
            )
        return None, None, None

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Launch tokenizer process
    tokenizer_manager = TokenizerManager(server_args, port_args)

    # Initialize templates
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        model_path=server_args.model_path,
    )

    # Wait for the model to finish loading
    scheduler_infos = []
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
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, template_manager, scheduler_info


def _launch_threads(
    server_args, port_args: PortArgs | None = None
) -> tuple[TokenizerManager, TemplateManager, dict]:
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info("server_args=%s", server_args)

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_threads = []
    scheduler_infos = []
    if server_args.dp_size == 1:
        scheduler_pipe_readers = []
        scheduler_info = run_scheduler_loop_thread_after_create(server_args, port_args)
        scheduler_infos.append(scheduler_info)
    else:
        pass

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for thread in scheduler_threads:
            thread.join()
            logger.error("Scheduler or DataParallelController %s terminated", thread.name)
        return None, None, None

    # Launch detokenizer thread
    detoken_thread = threading.Thread(
        target=run_detokenizer_thread,
        args=(
            server_args,
            port_args,
        ),
        daemon=True,
    )
    detoken_thread.start()

    # Launch tokenizer process
    tokenizer_manager = TokenizerManager(server_args, port_args)

    # Initialize templates
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        model_path=server_args.model_path,
    )

    # Wait for the model to finish loading
    for i in range(len(scheduler_infos)):
        if scheduler_infos[i]["status"] != "ready":
            raise RuntimeError("Initialization failed. Please see the error messages above.")

    # Assume all schedulers have the same scheduler_info
    assert len(scheduler_infos) > 0, "scheduler_infos is empty"
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, template_manager, scheduler_info


def _launch_subprocesses_or_threads(
    server_args, port_args: PortArgs | None = None
) -> tuple[TokenizerManager, TemplateManager, dict]:
    if server_args.enable_single_process:
        return _launch_threads(server_args, port_args)
    else:
        return _launch_subprocesses(server_args, port_args)
