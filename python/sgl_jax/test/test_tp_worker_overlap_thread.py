import threading
from dataclasses import dataclass
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.managers import tp_worker_overlap_thread as worker_module
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


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("multi_host", [False, True])
def test_next_metadata_preserves_multihost_submission_order(monkeypatch, fused, multi_host):
    events = []
    errors = Queue()
    forward_launched = threading.Event()
    allow_sample_submission = threading.Event()
    submission_wait_entered = threading.Event()
    second_batch_prepared = threading.Event()

    class SubmissionEvent(threading.Event):
        def wait(self, timeout=None):
            if not self.is_set():
                submission_wait_entered.set()
            if not super().wait(timeout=5):
                raise TimeoutError("Previous batch submission did not finish")
            return True

    @dataclass
    class SamplingInfo:
        bid: int
        sampling_info_done: object = None
        penalizer_orchestrator: object = None

        def update_penalties(self):
            events.append(f"cpu{self.bid}")

    class Tokens:
        def __init__(self, bid):
            self.bid = bid

        def copy_to_host_async(self):
            events.append(f"copy{self.bid}")

    def batch(bid):
        return SimpleNamespace(
            bid=bid,
            sampling_info=SamplingInfo(bid),
            launch_done=threading.Event(),
            seq_lens=np.array([1], dtype=np.int32),
            req_pool_indices=np.array([0], dtype=np.int32),
        )

    def metadata(batch, *_):
        events.append(f"metadata{batch.bid}")

    def forward(batch, *_args, **_kwargs):
        # The existing launch_done event precedes sampler submission.
        batch.launch_done.set()
        if batch.bid == 1:
            forward_launched.set()
            if not allow_sample_submission.wait(timeout=5):
                raise TimeoutError("Sampler submission was not released")
        events.append(f"sample{batch.bid}")
        result = (None, Tokens(batch.bid), 0)
        return (*result, object()) if fused else result

    def set_future(future_map, _seq_lens, _req_pool, tokens, _mesh):
        events.append(f"future{tokens.bid}")
        return future_map

    def gather(tokens):
        events.append(f"gather{tokens.bid}")
        return tokens

    def unexpected_device_wait(*_):
        raise AssertionError("Submission ordering must not wait for device completion")

    monkeypatch.setattr(worker_module.SamplingMetadata, "from_model_worker_batch", metadata)
    monkeypatch.setattr(
        worker_module.ForwardBatch,
        "init_new",
        lambda batch, _: SimpleNamespace(
            input_ids=[0], seq_lens=batch.seq_lens, req_pool_indices=batch.req_pool_indices
        ),
    )
    monkeypatch.setattr(worker_module, "resolve_future_token_ids", lambda ids, *_: ids)
    monkeypatch.setattr(worker_module, "set_future_token_ids", set_future)
    monkeypatch.setattr(worker_module.jax, "block_until_ready", unexpected_device_wait)

    client = object.__new__(ModelWorkerClient)
    client.input_queue = Queue()
    client.output_queue = Queue()
    client._submission_done = SubmissionEvent() if multi_host else None
    if client._submission_done is not None:
        client._submission_done.set()
    client.mesh = None
    client.future_map_size = 4
    client.future_token_ids_map = object()
    client.async_gather_fn = gather
    client.worker = SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=128),
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(get_forward_metadata=lambda _: None)
        ),
        server_args=SimpleNamespace(enable_lora=False, disaggregation_mode="null"),
        get_model_runner=lambda: None,
        _pd_fuse_for_batch=lambda _: fused,
        forward_batch_generation=forward,
    )

    def capture_errors(fn):
        try:
            fn()
        except BaseException as exc:
            errors.put(exc)

    def prepare_second_batch():
        try:
            client.forward_batch_generation(batch(2))
        finally:
            second_batch_prepared.set()

    first_batch = batch(1)
    client.forward_batch_generation(first_batch)
    worker = threading.Thread(
        target=lambda: capture_errors(client.forward_thread_func_), daemon=True
    )
    scheduler = threading.Thread(target=lambda: capture_errors(prepare_second_batch), daemon=True)
    worker.start()
    try:
        assert forward_launched.wait(timeout=5)
        assert first_batch.launch_done.is_set()
        scheduler.start()
        if multi_host:
            assert submission_wait_entered.wait(timeout=5)
            assert "cpu2" in events
            assert "metadata2" not in events
        else:
            assert second_batch_prepared.wait(timeout=5)
            assert "metadata2" in events
        assert "sample1" not in events
    finally:
        allow_sample_submission.set()
        if scheduler.ident is not None:
            scheduler.join(timeout=5)
        client.input_queue.put((None, None, None, None))
        worker.join(timeout=5)

    assert not scheduler.is_alive()
    assert not worker.is_alive()
    if not errors.empty():
        raise errors.get()
    assert client.output_queue.qsize() == 2
    if multi_host:
        last_submission = "sample1" if fused else "copy1"
        assert events.index("cpu2") < events.index(last_submission) < events.index("metadata2")
    else:
        assert events.index("metadata2") < events.index("sample1")
