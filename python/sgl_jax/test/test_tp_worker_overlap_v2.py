import threading
from queue import Queue
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient
from sgl_jax.srt.managers.tp_worker_overlap_v2 import ModelWorkerOverlap


def _make_logits_output():
    return LogitsProcessorOutput(
        next_token_logits=jnp.zeros((4, 8), dtype=jnp.float32),
        hidden_states=jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
        next_token_logprobs=jnp.array([-0.1, -1.0, -0.2, -2.0], dtype=jnp.float32),
        input_token_logprobs=jnp.array([-0.3, -0.4, -0.5], dtype=jnp.float32),
        next_token_top_logprobs_val=jnp.array(
            [
                [-0.1, -0.2, -0.3],
                [-1.0, -1.1, -1.2],
                [-0.4, -0.5, -0.6],
                [-2.0, -2.1, -2.2],
            ],
            dtype=jnp.float32,
        ),
        next_token_top_logprobs_idx=jnp.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=jnp.int32,
        ),
        next_token_token_ids_logprobs_val=jnp.arange(24, dtype=jnp.float32).reshape(4, 6),
    )


def _make_batch():
    return SimpleNamespace(
        return_logprob=True,
        return_output_logprob_only=False,
        logits_indices_selector=np.array([0, 2], dtype=np.int32),
        top_logprobs_nums=[2, 0, 1, 0],
        token_ids_logprobs=[[1, 3], None, [2], None],
    )


def _resolve_with_legacy_path(logits_output, next_token_ids, batch):
    worker = object.__new__(ModelWorkerOverlap)
    worker._materialize_logprobs_to_host(
        logits_output,
        batch,
        batch.logits_indices_selector,
    )

    client = object.__new__(ModelWorkerClient)
    client.output_queue = Queue()
    client.output_queue.put((None, logits_output, next_token_ids, 7))
    launch_done = threading.Event()
    launch_done.set()
    return client.resolve_last_batch_result(launch_done)


def _resolve_with_v2_path(logits_output, next_token_ids, batch):
    worker = object.__new__(ModelWorkerOverlap)
    launch_done = threading.Event()
    launch_done.set()
    return worker.resolve_last_batch_result(
        logits_output,
        next_token_ids,
        batch,
        7,
        launch_done,
    )


def test_v2_resolver_matches_legacy_host_output_contract():
    next_token_ids = jnp.array([11, 0, 22, 0], dtype=jnp.int32)
    legacy_logits, legacy_ids, legacy_misses = _resolve_with_legacy_path(
        _make_logits_output(), next_token_ids, _make_batch()
    )
    v2_logits, v2_ids, v2_misses = _resolve_with_v2_path(
        _make_logits_output(), next_token_ids, _make_batch()
    )

    assert v2_ids == legacy_ids
    assert v2_misses == legacy_misses
    assert isinstance(v2_logits.next_token_logprobs, list)
    assert isinstance(v2_logits.input_token_logprobs, list)
    np.testing.assert_allclose(
        v2_logits.next_token_logprobs,
        legacy_logits.next_token_logprobs,
    )
    np.testing.assert_allclose(
        v2_logits.input_token_logprobs,
        legacy_logits.input_token_logprobs,
    )
    assert (
        v2_logits.next_token_top_logprobs_val
        == legacy_logits.next_token_top_logprobs_val
    )
    assert (
        v2_logits.next_token_top_logprobs_idx
        == legacy_logits.next_token_top_logprobs_idx
    )
    assert (
        v2_logits.next_token_token_ids_logprobs_val
        == legacy_logits.next_token_token_ids_logprobs_val
    )
    assert (
        v2_logits.next_token_token_ids_logprobs_idx
        == legacy_logits.next_token_token_ids_logprobs_idx
    )
    np.testing.assert_array_equal(v2_logits.hidden_states, legacy_logits.hidden_states)
