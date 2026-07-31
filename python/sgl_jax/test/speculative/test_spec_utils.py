import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.speculative.spec_utils import greedy_chain_verify


def _logits_from_predict(target_predict, *, vocab_size=128):
    target_predict = np.asarray(target_predict, dtype=np.int32)
    logits = np.full((target_predict.size, vocab_size), -1.0, dtype=np.float32)
    logits[np.arange(target_predict.size), target_predict.reshape(-1)] = 10.0
    return logits


def test_greedy_chain_verify_handles_all_partial_reject_and_padding():
    draft_tokens = jnp.array(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
        ],
        dtype=jnp.int32,
    )
    target_predict = jnp.array(
        [
            [11, 12, 13, 99],
            [77, 78, 79, 80],
            [31, 88, 89, 90],
            [41, 42, 43, 91],
        ],
        dtype=jnp.int32,
    )
    valid_mask = jnp.array([True, True, True, False])

    result = greedy_chain_verify(
        draft_tokens.reshape(-1),
        jnp.asarray(_logits_from_predict(target_predict)),
        draft_width=4,
        valid_mask=valid_mask,
    )

    np.testing.assert_array_equal(
        np.asarray(result.accepted_children),
        np.array(
            [
                [True, True, True],
                [False, False, False],
                [True, False, False],
                [False, False, False],
            ]
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(result.accepted_draft_lens),
        np.array([3, 0, 1, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(result.accept_lens),
        np.array([4, 1, 2, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(result.next_verified_id),
        np.array([99, 77, 88, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(result.target_predict).reshape(4, 4),
        np.asarray(target_predict),
    )


def test_greedy_chain_verify_is_jittable():
    verify = jax.jit(
        lambda draft, logits: greedy_chain_verify(
            draft,
            logits,
            draft_width=4,
        )
    )
    result = verify(
        jnp.array([10, 11, 12, 13], dtype=jnp.int32),
        jnp.asarray(_logits_from_predict([11, 12, 13, 99])),
    )

    np.testing.assert_array_equal(np.asarray(result.accept_lens), np.array([4]))
    np.testing.assert_array_equal(np.asarray(result.next_verified_id), np.array([99]))


def test_greedy_chain_verify_keeps_masked_outputs_data_sharded():
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = np.asarray(jax.devices())
    data_size = 2 if devices.size >= 2 and devices.size % 2 == 0 else 1
    tensor_size = devices.size // data_size
    mesh = Mesh(
        devices.reshape(data_size, tensor_size),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    bs = data_size * 2
    draft_width = 4
    draft_tokens = np.tile(np.array([[10, 11, 12, 13]], dtype=np.int32), (bs, 1))
    target_predict = np.tile(np.array([[11, 12, 13, 99]], dtype=np.int32), (bs, 1))
    vocab_size = max(128, tensor_size * 128)
    valid_mask = np.ones((bs,), dtype=bool)
    valid_mask[-1] = False

    result = greedy_chain_verify(
        jax.device_put(draft_tokens.reshape(-1), NamedSharding(mesh, P("data"))),
        jax.device_put(
            _logits_from_predict(target_predict, vocab_size=vocab_size),
            NamedSharding(mesh, P("data", "tensor")),
        ),
        draft_width=draft_width,
        valid_mask=jax.device_put(valid_mask, NamedSharding(mesh, P("data"))),
    )

    assert result.target_predict.sharding.spec == P("data")
    assert result.accepted_children.sharding.spec == P("data", None)
    assert result.accepted_draft_lens.sharding.spec == P("data")
    assert result.accept_lens.sharding.spec == P("data")
    assert result.next_verified_id.sharding.spec == P("data")
    np.testing.assert_array_equal(
        np.asarray(result.accept_lens),
        np.array([4] * (bs - 1) + [0], dtype=np.int32),
    )
