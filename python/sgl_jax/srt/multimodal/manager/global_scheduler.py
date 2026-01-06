import logging
import os
import queue
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import psutil
import setproctitle
import zmq

from sgl_jax.srt.multimodal.manager.device_manager import DeviceManager
from sgl_jax.srt.multimodal.manager.io_struct import TokenizedGenerateMMReqInput
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.manager.stage import Stage
from sgl_jax.srt.multimodal.manager.utils import load_stage_configs_from_yaml
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import configure_logger, kill_itself_when_parent_died
from sgl_jax.srt.utils.common_utils import get_zmq_socket
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


class GlobalScheduler:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs) -> None:
        context = zmq.Context(2)
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, False
        )
        self.send_to_detokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.detokenizer_ipc_name, False
        )

        self.stage_configs = load_stage_configs_from_yaml(
            "./sgl_jax/srt/multimodal/models/static_configs/wan2_1_stage_config.yaml"
        )
        self.device_manager = DeviceManager()
        self._init_stage()
        self.req_store = dict()

    def _init_stage(self):
        def _build_stage(idx_cfg: tuple[int, Any]) -> tuple[int, Stage]:
            idx, cfg = idx_cfg
            return idx, Stage(cfg, device_manager=self.device_manager)

        with ThreadPoolExecutor(
            max_workers=min(len(self.stage_configs), max(1, os.cpu_count() or 1))
        ) as executor:
            futures = [
                executor.submit(_build_stage, (idx, cfg))
                for idx, cfg in enumerate(self.stage_configs)
            ]
            results: list[tuple[int, Stage]] = []
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        self.stage_list = [st for _, st in results]
        self.in_queues = [queue.Queue() for _ in range(len(self.stage_list))]
        self.out_queues = [queue.Queue() for _ in range(len(self.stage_list))]
        for i, stage in enumerate(self.stage_list):
            stage.set_in_queue(self.in_queues[i])
            stage.set_out_queue(self.out_queues[i])
            stage.set_stage_index(i)
        self.start_stage()
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateMMReqInput, self.convert_request),
            ]
        )

    def convert_request(self, input: TokenizedGenerateMMReqInput):
        req = Req(
            rid=input.rid,
            input_ids=input.input_ids,
            negative_input_ids=input.negative_input_ids,
            num_outputs_per_prompt=input.n,
            height_latents=int(input.size.split("*")[0]),
            width_latents=int(input.size.split("*")[1]),
            num_frames=input.num_frames,
            data_type=input.data_type,
            save_output=input.save_output,
        )
        if req.rid in self.req_store:
            raise RuntimeError(f"{req.rid} is already in req_store")
        self.req_store[req.rid] = req
        return req

    def start_stage(self):
        import threading

        for stage in self.stage_list:
            thread = threading.Thread(target=stage.run_stage)
            thread.start()
        for i, q in enumerate(self.out_queues):
            status = q.get()
            if status["status"] != "ready":
                raise Exception(f"stage {i} init failed")

    def recv_request(self):
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)
        return recv_reqs

    def event_loop(self):
        import time

        while True:
            reqs = self.recv_request()
            print("recv_reqs from tokenizer", reqs)
            time.sleep(3)
            if reqs:
                for req in reqs:
                    self.in_queues[0].put_nowait(self._request_dispatcher(req))
            else:
                for i, stage in enumerate(self.stage_list):
                    stage_result = Req.from_stage(stage.try_collect(), self.req_store)
                    print("stage result", stage_result)
                    if stage_result is None:
                        continue
                    else:
                        if self.stage_configs[i].final_output:
                            self.send_to_detokenizer.send_pyobj(stage_result)
                        else:
                            self.in_queues[i + 1].put_nowait(
                                stage_result.to_stage_req(self.stage_configs[i].scheduler)
                            )


def run_global_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang-jax::global_scheduler")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        scheduler = GlobalScheduler(server_args, port_args)
        # TODO: Implement event loop
        pipe_writer.send(
            {
                "status": "ready",
            }
        )
        scheduler.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("GlobalScheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)
    return scheduler
