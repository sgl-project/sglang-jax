"""DP-attention relay-layout regressions for Frozen-KV MTP."""

from __future__ import annotations

import os
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.speculative.eagle_util import build_chain_verify_inputs_device
from sgl_jax.srt.speculative.frozen_kv_mtp_worker import (
    FrozenKvMtpDraftWorker,
    _frozen_kv_top1_token,
    _select_frozen_kv_proposal_start,
)
from sgl_jax.srt.speculative.relay_buffer import (
    create_spec_seed_relay_buffers,
    gather_spec_seed_relay_buffers,
)


@pytest.mark.skipif(jax.device_count() < 2, reason="requires two CPU test devices")
def test_top1_preserves_data_sharding_after_vocab_replication():
    """Fused proposal selection requires token IDs and relay masks to agree."""
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(1, 2),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    logits = jax.device_put(
        jnp.asarray([[0.0, 3.0, 1.0], [4.0, 2.0, 1.0]], dtype=jnp.float32),
        NamedSharding(mesh, P("data", None)),
    )

    token = jax.jit(_frozen_kv_top1_token)(logits)

    assert token.sharding.spec == P("data")
    np.testing.assert_array_equal(token, [1, 0])


@pytest.mark.skipif(jax.device_count() < 2, reason="requires two CPU test devices")
def test_proposal_select_normalizes_replicated_model_output_to_relay_layout():
    """Explicit TPU sharding rejects P(None) and P("data") select cases."""
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(1, 2),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    relay_token = jax.device_put(jnp.asarray([10, 20]), NamedSharding(mesh, P("data")))
    relay_hidden = jax.device_put(
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), NamedSharding(mesh, P("data", None))
    )
    relay_seed_mask = jax.device_put(jnp.asarray([True, False]), NamedSharding(mesh, P("data")))
    seed_logits = jax.device_put(
        jnp.asarray([[0.0, 3.0, 1.0], [4.0, 2.0, 1.0]]), NamedSharding(mesh, P())
    )
    seed_hidden = jax.device_put(relay_hidden + 100, NamedSharding(mesh, P()))

    token, hidden = jax.jit(_select_frozen_kv_proposal_start)(
        relay_token,
        relay_hidden,
        relay_seed_mask,
        seed_logits,
        seed_hidden,
    )

    assert token.sharding.spec == P("data")
    assert hidden.sharding.spec == P("data", None)
    np.testing.assert_array_equal(token, [1, 20])
    np.testing.assert_allclose(hidden, [[101.0, 102.0], [3.0, 4.0]])


@pytest.mark.skipif(jax.device_count() < 2, reason="requires two CPU test devices")
def test_proposal_select_normalizes_dp1_mask_to_replicated_relay_layout():
    """A size-one data mesh still requires exact explicit-sharding equality."""
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(1, 2),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    relay_token = jax.device_put(jnp.asarray([10, 20]), NamedSharding(mesh, P()))
    relay_hidden = jax.device_put(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), NamedSharding(mesh, P()))
    relay_seed_mask = jax.device_put(jnp.asarray([True, False]), NamedSharding(mesh, P("data")))
    seed_logits = jax.device_put(
        jnp.asarray([[0.0, 3.0, 1.0], [4.0, 2.0, 1.0]]), NamedSharding(mesh, P())
    )
    seed_hidden = jax.device_put(relay_hidden + 100, NamedSharding(mesh, P()))

    token, hidden = jax.jit(_select_frozen_kv_proposal_start)(
        relay_token,
        relay_hidden,
        relay_seed_mask,
        seed_logits,
        seed_hidden,
    )

    assert token.sharding.spec == P(None)
    assert hidden.sharding.spec == P(None, None)
    np.testing.assert_array_equal(token, [1, 20])
    np.testing.assert_allclose(hidden, [[101.0, 102.0], [3.0, 4.0]])


@pytest.mark.skipif(jax.device_count() < 2, reason="requires two CPU test devices")
def test_chain_packing_normalizes_dynamic_and_static_row_layouts():
    """Packed verify metadata follows the draft-token batch layout."""
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(1, 2),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    replicated = NamedSharding(mesh, P())
    data_sharded = NamedSharding(mesh, P("data"))
    verified_id = jax.device_put(jnp.asarray([10, 20]), replicated)
    token_list = jax.device_put(jnp.asarray([[11, 12, 13], [21, 22, 23]]), replicated)
    seq_lens = jax.device_put(jnp.asarray([100, 200]), data_sharded)

    packed = build_chain_verify_inputs_device(verified_id, token_list, seq_lens, 4, 2)

    assert packed.sharding.spec == P(None, None)
    np.testing.assert_array_equal(packed[1], jnp.asarray([100, 101, 102, 103, 200, 201, 202, 203]))


@pytest.mark.skipif(jax.device_count() < 2, reason="requires two CPU test devices")
def test_single_live_prefill_publishes_into_a_dp2_seed_relay():
    """A compact c1 update must not be partitioned across two DP replicas."""
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(2, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    worker = FrozenKvMtpDraftWorker.__new__(FrozenKvMtpDraftWorker)
    worker._worker = types.SimpleNamespace(mesh=mesh)
    worker.server_args = types.SimpleNamespace(dp_size=2)
    worker._jit_publish_seed_relay = None
    worker._jit_gather_seed_relay = None
    req_pool = types.SimpleNamespace(req_to_token=np.zeros((8, 1), dtype=np.int32))
    worker.seed_relay_buffers = create_spec_seed_relay_buffers(
        mesh,
        req_pool,
        dp_size=2,
        hidden_size=3,
        hidden_dtype=jnp.float32,
    )
    batch = types.SimpleNamespace(
        logits_indices_selector=np.asarray([0], dtype=np.int32),
        # The scheduler bucket is DP-divisible even though only slot zero is live.
        req_pool_indices=jnp.asarray([3, 0], dtype=jnp.int32),
    )

    worker._publish_seed_relay(
        model_worker_batch=batch,
        verified_id=jnp.asarray([42], dtype=jnp.int32),
        draft_token_ids=jnp.asarray([43], dtype=jnp.int32),
        hidden_states=jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32),
        is_target_seed=jnp.asarray([True]),
    )

    indices = jax.device_put(jnp.asarray([3, 0], dtype=jnp.int32), NamedSharding(mesh, P("data")))
    token_ids, draft_ids, hidden, is_target_seed = gather_spec_seed_relay_buffers(
        worker.seed_relay_buffers, indices, dp_size=2
    )
    np.testing.assert_array_equal(token_ids, [42, 0])
    np.testing.assert_array_equal(draft_ids, [43, 0])
    np.testing.assert_allclose(hidden, [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(is_target_seed, [True, False])


@pytest.mark.skipif(jax.device_count() < 2, reason="requires two CPU test devices")
@pytest.mark.parametrize(
    ("selector", "expected_tokens"),
    [
        ([0, 1], [51, 52, 0, 0]),
        ([0, 2], [51, 0, 52, 0]),
    ],
    ids=("concentrated_on_rank_zero", "split_across_ranks"),
)
def test_compact_updates_use_explicit_dp_padded_slots(selector, expected_tokens):
    """A compact row count does not imply balanced DP-attention ranks."""
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(2, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    worker = FrozenKvMtpDraftWorker.__new__(FrozenKvMtpDraftWorker)
    worker._worker = types.SimpleNamespace(mesh=mesh)
    worker.server_args = types.SimpleNamespace(dp_size=2)
    worker._jit_publish_seed_relay = None
    worker._jit_gather_seed_relay = None
    req_pool = types.SimpleNamespace(req_to_token=np.zeros((8, 1), dtype=np.int32))
    worker.seed_relay_buffers = create_spec_seed_relay_buffers(
        mesh,
        req_pool,
        dp_size=2,
        hidden_size=2,
        hidden_dtype=jnp.float32,
    )
    batch = types.SimpleNamespace(
        logits_indices_selector=np.asarray(selector, dtype=np.int32),
        req_pool_indices=jax.device_put(
            jnp.asarray([3, 4, 5, 6], dtype=jnp.int32),
            NamedSharding(mesh, P("data")),
        ),
    )

    worker._publish_seed_relay(
        model_worker_batch=batch,
        verified_id=jnp.asarray([51, 52], dtype=jnp.int32),
        draft_token_ids=jnp.asarray([61, 62], dtype=jnp.int32),
        hidden_states=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        is_target_seed=jnp.asarray([True, True]),
    )

    indices = jax.device_put(
        jnp.asarray([3, 4, 5, 6], dtype=jnp.int32), NamedSharding(mesh, P("data"))
    )
    token_ids, _, _, is_target_seed = gather_spec_seed_relay_buffers(
        worker.seed_relay_buffers, indices, dp_size=2
    )
    np.testing.assert_array_equal(token_ids, expected_tokens)
    np.testing.assert_array_equal(is_target_seed, np.asarray(expected_tokens) != 0)


@pytest.mark.skipif(jax.device_count() < 2, reason="requires two CPU test devices")
def test_dp2_verify_accepts_compact_allocate_lens_with_padded_metadata():
    """Verify handoff must not index compact allocation rows by padded slots."""
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape(2, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    worker = FrozenKvMtpDraftWorker.__new__(FrozenKvMtpDraftWorker)
    worker._worker = types.SimpleNamespace(mesh=mesh)
    worker.server_args = types.SimpleNamespace(dp_size=2)
    worker.speculative_num_steps = 3
    worker._jit_publish_seed_relay = None
    worker._jit_gather_seed_relay = None
    req_pool = types.SimpleNamespace(req_to_token=np.zeros((8, 1), dtype=np.int32))
    worker.seed_relay_buffers = create_spec_seed_relay_buffers(
        mesh,
        req_pool,
        dp_size=2,
        hidden_size=3,
        hidden_dtype=jnp.float32,
    )
    batch = types.SimpleNamespace(
        logits_indices_selector=np.asarray([0], dtype=np.int32),
        seq_lens=jax.device_put(
            jnp.asarray([10, 0], dtype=jnp.int32), NamedSharding(mesh, P("data"))
        ),
        req_pool_indices=jax.device_put(
            jnp.asarray([3, 0], dtype=jnp.int32), NamedSharding(mesh, P("data"))
        ),
    )
    verified_tokens = jax.device_put(
        jnp.arange(100, 108, dtype=jnp.int32), NamedSharding(mesh, P("data"))
    )
    hidden = jax.device_put(
        jnp.arange(24, dtype=jnp.float32).reshape(8, 3),
        NamedSharding(mesh, P("data", None)),
    )
    accept_lengths = jax.device_put(
        jnp.asarray([3, 0], dtype=jnp.int32), NamedSharding(mesh, P("data"))
    )

    # This field is intentionally compact: BaseDraftWorker._get_cur_allocate_lens
    # has already selected live rows before entering FrozenKvMtpWorker.verify.
    worker.publish_seed_after_verify_device(
        model_worker_batch=batch,
        verified_tokens=verified_tokens,
        target_hidden=hidden,
        accept_lengths=accept_lengths,
        allocate_lens=np.asarray([20], dtype=np.int32),
    )

    indices = jax.device_put(jnp.asarray([3, 0], dtype=jnp.int32), NamedSharding(mesh, P("data")))
    token_ids, draft_ids, selected_hidden, is_target_seed = gather_spec_seed_relay_buffers(
        worker.seed_relay_buffers, indices, dp_size=2
    )
    np.testing.assert_array_equal(token_ids, [102, 0])
    np.testing.assert_array_equal(draft_ids, [102, 0])
    np.testing.assert_allclose(selected_hidden, [[6.0, 7.0, 8.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(is_target_seed, [True, False])
