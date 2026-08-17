from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.speculative.overlap_utils import prefetch_published_new_seq_lens
from sgl_jax.srt.speculative.relay_buffer import (
    DFlashRelayBuffers,
    SpecRelayBuffers,
    build_relay_batch_plan,
    create_dflash_relay_buffers,
    create_spec_relay_buffers,
    gather_relay_buffers,
    scatter_relay_buffers,
)


def test_build_relay_batch_plan_separates_compact_and_padded_indices():
    batch = SimpleNamespace(
        req_pool_indices=np.array([10, 11, 0, 20, 0, 0], dtype=np.int32),
        logits_indices_selector=np.array([0, 1, 3], dtype=np.int32),
        real_bs_per_dp=[2, 1],
        per_dp_bs_size=3,
    )

    plan = build_relay_batch_plan(batch)

    np.testing.assert_array_equal(
        plan.future_indices,
        np.array([10, 11, 20], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        plan.padded_indices,
        np.array([10, 11, 0, 20, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        plan.valid_mask,
        np.array([True, True, False, True, False, False]),
    )


def test_prefetch_published_new_seq_lens_uses_explicit_result_field():
    new_seq_lens = Mock()
    result = SimpleNamespace(published_new_seq_lens=new_seq_lens)

    assert prefetch_published_new_seq_lens(result) is new_seq_lens
    new_seq_lens.copy_to_host_async.assert_called_once_with()


def _relay_test_mesh():
    devices = np.asarray(jax.devices())
    data_size = 2 if devices.size >= 2 and devices.size % 2 == 0 else 1
    mesh = Mesh(
        devices[:data_size].reshape(data_size),
        ("data",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    return mesh, data_size


def _scatter_and_gather(mesh, dp_size, buffers, payload):
    per_dp_bs = 2
    indices = np.tile(np.array([1, 3], dtype=np.int32), dp_size)
    valid_mask = np.tile(np.array([True, False]), dp_size)
    data_sharding = NamedSharding(mesh, P("data"))
    indices = jax.device_put(indices, data_sharding)
    valid_mask = jax.device_put(valid_mask, data_sharding)

    scatter = jax.jit(
        lambda state, idx, mask, values: scatter_relay_buffers(
            state,
            idx,
            mask,
            values,
            dp_size=dp_size,
        )
    )
    gather = jax.jit(
        lambda state, idx: gather_relay_buffers(
            state,
            idx,
            dp_size=dp_size,
        )
    )
    with jax.set_mesh(mesh):
        updated = scatter(buffers, indices, valid_mask, payload)
        gathered = gather(updated, indices)
    return gathered, np.arange(0, dp_size * per_dp_bs, per_dp_bs)


def test_generic_relay_scatter_gather_supports_spec_payload():
    mesh, dp_size = _relay_test_mesh()
    capacity = 4
    total_bs = dp_size * 2
    req_pool = SimpleNamespace(
        req_to_token=np.zeros((capacity, 1), dtype=np.int32),
    )
    buffers = create_spec_relay_buffers(
        mesh,
        req_pool,
        dp_size=dp_size,
        num_steps=3,
        hidden_size=2,
        hidden_dtype=jnp.float32,
    )
    payload = SpecRelayBuffers(
        topk_index=jax.device_put(
            jnp.arange(total_bs * 3, dtype=jnp.int32).reshape(total_bs, 3) + 10,
            NamedSharding(mesh, P("data", None)),
        ),
        hidden_states=jax.device_put(
            jnp.arange(total_bs * 2, dtype=jnp.float32).reshape(total_bs, 2) + 20,
            NamedSharding(mesh, P("data", None)),
        ),
        verified_id=jax.device_put(
            jnp.arange(total_bs, dtype=jnp.int32) + 30,
            NamedSharding(mesh, P("data")),
        ),
        new_seq_lens=jax.device_put(
            jnp.arange(total_bs, dtype=jnp.int32) + 40,
            NamedSharding(mesh, P("data")),
        ),
    )

    gathered, valid_rows = _scatter_and_gather(mesh, dp_size, buffers, payload)

    assert gathered.topk_index.sharding.spec == P("data", None)
    assert gathered.hidden_states.sharding.spec == P("data", None)
    assert gathered.verified_id.sharding.spec == P("data")
    assert gathered.new_seq_lens.sharding.spec == P("data")
    expected_ids = np.zeros(total_bs, dtype=np.int32)
    expected_ids[valid_rows] = np.asarray(payload.verified_id)[valid_rows]
    np.testing.assert_array_equal(np.asarray(gathered.verified_id), expected_ids)


def test_generic_relay_scatter_gather_supports_dflash_payload():
    mesh, dp_size = _relay_test_mesh()
    capacity = 4
    total_bs = dp_size * 2
    req_pool = SimpleNamespace(
        req_to_token=np.zeros((capacity, 1), dtype=np.int32),
    )
    buffers = create_dflash_relay_buffers(
        mesh,
        req_pool,
        dp_size=dp_size,
    )
    payload = DFlashRelayBuffers(
        verified_id=jax.device_put(
            jnp.arange(total_bs, dtype=jnp.int32) + 50,
            NamedSharding(mesh, P("data")),
        ),
        new_seq_lens=jax.device_put(
            jnp.arange(total_bs, dtype=jnp.int32) + 60,
            NamedSharding(mesh, P("data")),
        ),
    )

    gathered, valid_rows = _scatter_and_gather(mesh, dp_size, buffers, payload)

    assert gathered.verified_id.sharding.spec == P("data")
    assert gathered.new_seq_lens.sharding.spec == P("data")
    expected_ids = np.zeros(total_bs, dtype=np.int32)
    expected_ids[valid_rows] = np.asarray(payload.verified_id)[valid_rows]
    np.testing.assert_array_equal(np.asarray(gathered.verified_id), expected_ids)
