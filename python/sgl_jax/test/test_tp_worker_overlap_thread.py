import threading
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import jax
import pytest

from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient


def test_model_worker_client_exposes_page_size_from_wrapped_worker():
    client = object.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace(page_size=128)

    assert client.page_size == 128


def test_model_worker_client_raises_when_wrapped_worker_lacks_page_size():
    client = object.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace()

    with pytest.raises(AttributeError):
        _ = client.page_size


def test_synchronize_for_profile_waits_for_complete_forward_state():
    client = object.__new__(ModelWorkerClient)
    client.input_queue = Queue()
    client.output_queue = Queue()
    client._last_forward_result = (object(), object(), object())

    thread = threading.Thread(target=client.forward_thread_func_)
    thread.start()
    with mock.patch.object(jax, "block_until_ready") as block_until_ready:
        client.synchronize_for_profile()
    client.input_queue.put((None, None, None, None))
    thread.join(timeout=1)

    block_until_ready.assert_called_once_with(client._last_forward_result)
    assert not thread.is_alive()


def test_profile_sync_result_includes_model_outputs_and_memory_pools():
    client = object.__new__(ModelWorkerClient)
    memory_pools = object()
    client.worker = SimpleNamespace(model_runner=SimpleNamespace(memory_pools=memory_pools))
    logits_output = object()
    next_token_ids = object()

    client._record_profile_sync_result(logits_output, next_token_ids)

    assert client._last_forward_result == (logits_output, next_token_ids, memory_pools)
