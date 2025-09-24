"""DetokenizerManager is a process that detokenizes the token ids."""

import dataclasses
import logging
import os
import signal
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import psutil
import setproctitle
import zmq

from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.managers.io_struct import BatchStrOut, BatchTokenIDOut
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    configure_logger,
    get_zmq_socket,
    kill_itself_when_parent_died,
)
from sgl_jax.utils import (
    TypeBasedDispatcher,
    find_printable_text,
    get_exception_traceback,
)

logger = logging.getLogger(__name__)

# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0


class DetokenizerManager:
    """DetokenizerManager is a process that detokenizes the token ids."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Init inter-process communication
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        )

        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )

        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)
        self.is_dummy = server_args.load_format == "dummy"

        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchTokenIDOut, self.handle_batch_token_id_out),
            ]
        )

    def event_loop(self):
        """The event loop that handles requests"""
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()

            output = self._request_dispatcher(recv_obj)
            # if recv_obj is not None:
            self.send_to_tokenizer.send_pyobj(output)

    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if no_stop_trim or not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            return output[:pos] if pos != -1 else output

        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            assert len(output) > 0
            return output[:-1]
        return output

    # def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOut):
    #     # If it is embedding model, no detokenization is needed.
    #     return recv_obj

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOut):
        bs = len(recv_obj.rids)

        # Initialize decode status
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if rid not in self.decode_status:
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=recv_obj.decode_ids[i],
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                self.decode_status[rid] = s
            else:
                s = self.decode_status[rid]
                s.decode_ids.extend(recv_obj.decode_ids[i])

            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                    recv_obj.no_stop_trim[i],
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

        # Flatten any nested lists in read_ids and surr_ids to avoid TypeError
        def flatten_token_ids(token_ids_list):
            flattened = []
            for i, token_ids in enumerate(token_ids_list):

                def deep_flatten(lst):
                    """Recursively flatten nested lists to ensure all elements are integers"""
                    if not isinstance(lst, list):
                        if isinstance(lst, int):
                            return [lst]
                        elif hasattr(lst, "item"):  # Handle numpy scalars/arrays
                            if hasattr(lst, "__len__") and len(lst) == 1:
                                return [int(lst.item())]
                            elif hasattr(lst, "__iter__"):
                                return [
                                    int(x.item() if hasattr(x, "item") else x)
                                    for x in lst
                                ]
                            else:
                                return [int(lst.item())]
                        else:
                            return []

                    result = []
                    for item in lst:
                        if isinstance(item, list):
                            result.extend(deep_flatten(item))
                        elif isinstance(item, int):
                            result.append(item)
                        elif hasattr(item, "item"):  # Handle numpy scalars/arrays
                            if hasattr(item, "__len__") and len(item) == 1:
                                result.append(int(item.item()))
                            elif hasattr(item, "__iter__"):
                                for x in item:
                                    result.append(
                                        int(x.item() if hasattr(x, "item") else x)
                                    )
                            else:
                                result.append(int(item.item()))
                        else:
                            print(
                                f"[WARNING] Skipping non-int, non-list, non-numpy item: {type(item)}, {item}"
                            )
                        # Skip non-integer, non-list items
                    return result

                flat_ids = deep_flatten(token_ids)

                # Additional safety check to ensure no nested lists remain
                for j, token_id in enumerate(flat_ids):
                    if not isinstance(token_id, int):
                        print(
                            f"[ERROR] Non-integer found at flat_ids[{j}]: {type(token_id)}, {token_id}"
                        )
                        raise ValueError(
                            f"Expected integer token ID, got {type(token_id)}: {token_id}"
                        )

                flattened.append(flat_ids)
            return flattened

        flattened_surr_ids = flatten_token_ids(surr_ids)

        surr_texts = self.tokenizer.batch_decode(
            flattened_surr_ids,
            skip_special_tokens=recv_obj.skip_special_tokens[0],
            spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
        )

        flattened_read_ids = flatten_token_ids(read_ids)

        read_texts = self.tokenizer.batch_decode(
            flattened_read_ids,
            skip_special_tokens=recv_obj.skip_special_tokens[0],
            spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
        )

        # Incremental decoding
        output_strs = []
        output_ids_list = []
        for i in range(bs):
            try:
                s = self.decode_status[recv_obj.rids[i]]
            except KeyError:
                raise RuntimeError(
                    f"Decode status not found for request {recv_obj.rids[i]}. "
                    "It may be due to the request being evicted from the decode status due to memory pressure. "
                    "Please increase the maximum number of requests by setting "
                    "the SGLANG_DETOKENIZER_MAX_STATES environment variable to a bigger value than the default value. "
                    f"The current value is {DETOKENIZER_MAX_STATES}. "
                    "For more details, see: https://github.com/sgl-project/sglang/issues/2812"
                )
            new_text = read_texts[i][len(surr_texts[i]) :]
            new_token_ids = read_ids[i][len(surr_ids[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming chunk: update the decode status
                if len(new_text) > 0 and not new_text.endswith("�"):
                    s.decoded_text = s.decoded_text + new_text
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    new_text = ""
                else:
                    new_text = find_printable_text(new_text)

            output_str = self.trim_matched_stop(
                s.decoded_text + new_text,
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )

            processed_new_token_ids = process_special_tokens_spaces(
                new_token_ids,
                recv_obj.skip_special_tokens[i],
                self.tokenizer.all_special_ids,
            )

            # Incrementally send text.
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            output_strs.append(incremental_output)
            output_ids_list.append(processed_new_token_ids)

        return BatchStrOut(
            rids=recv_obj.rids,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=output_ids_list,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_hidden_states=recv_obj.output_hidden_states,
            cache_miss_count=recv_obj.cache_miss_count,
        )


def process_special_tokens_spaces(
    token_ids: Optional[List[int]] = None,
    skip_special_tokens: Optional[bool] = None,
    all_special_ids: Optional[List[int]] = None,
):
    if all_special_ids is None or not skip_special_tokens or token_ids is None:
        return token_ids
    return [token for token in token_ids if token not in all_special_ids]


class LimitedCapacityDict(OrderedDict):
    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        # Set the new item
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang-jax::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = DetokenizerManager(server_args, port_args)
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)


def run_detokenizer_thread(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    current_thread = threading.current_thread()
    current_thread.name = "sglang-jax::detokenizer"
    configure_logger(server_args)
    current_process = psutil.Process()

    try:
        manager = DetokenizerManager(server_args, port_args)
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        current_process.send_signal(signal.SIGQUIT)
