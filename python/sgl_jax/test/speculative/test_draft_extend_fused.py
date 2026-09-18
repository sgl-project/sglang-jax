from functools import partial

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
from sgl_jax.srt.speculative.draft_extend_fused import (
    _make_draft_extend_metadata,
    _make_target_verify_metadata,
    _per_dp_cumsum_device,
)


def _explicit_mesh(dp_size):
    if jax.device_count() < dp_size:
        pytest.skip(f"requires {dp_size} devices")
    return Mesh(
        np.asarray(jax.devices()).reshape(dp_size, -1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


@pytest.mark.parametrize("use_jit", [False, True])
@pytest.mark.parametrize(
    "dp_size,lens,expected",
    [
        (1, [4, 0], [0, 4, 4]),
        (2, [4, 0, 2, 3], [0, 4, 4, 0, 2, 5]),
        (2, [0, 0, 2, 0], [0, 0, 0, 0, 2, 2]),
    ],
)
def test_per_dp_cumsum_preserves_rank_boundaries_and_sharding(dp_size, lens, expected, use_jit):
    mesh = _explicit_mesh(dp_size)
    sharding = NamedSharding(mesh, P("data"))
    cumsum = partial(_per_dp_cumsum_device, dp_size=dp_size)
    if use_jit:
        cumsum = jax.jit(cumsum)

    with jax.set_mesh(mesh):
        values = jax.device_put(np.asarray(lens, dtype=np.int32), sharding)
        result = cumsum(values)

    # Each rank starts at zero; padding must not consume query/KV offsets.
    np.testing.assert_array_equal(np.asarray(result), expected)
    assert result.dtype == np.int32
    assert result.sharding == sharding


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("target_verify", [False, True])
def test_fused_metadata_builds_sharded_query_and_kv_offsets(dp_size, target_verify):
    mesh = _explicit_mesh(dp_size)
    sharding = NamedSharding(mesh, P("data"))

    with jax.set_mesh(mesh):
        seq_lens = jax.device_put(np.asarray([5, 0, 3, 0][: dp_size * 2], np.int32), sharding)
        query_lens = jax.device_put(np.asarray([2, 0, 1, 0][: dp_size * 2], np.int32), sharding)
        allocated_lens = jax.device_put(np.asarray([8, 0, 4, 0][: dp_size * 2], np.int32), sharding)
        pages = jax.device_put(
            np.asarray([10, 11, 0, 0, 20, 0, 0, 0][: dp_size * 4], np.int32), sharding
        )
        old_metadata = FlashAttentionMetadata(page_indices=pages)
        if target_verify:
            make_metadata = jax.jit(
                partial(
                    _make_target_verify_metadata,
                    speculative_num_draft_tokens=2,
                    page_size=4,
                    dp_size=dp_size,
                )
            )
            verify_lens = jax.device_put(
                np.asarray([3, 0, 1, 0][: dp_size * 2], np.int32), sharding
            )
            result = make_metadata(old_metadata, verify_lens, allocated_lens)
        else:
            make_metadata = jax.jit(
                partial(_make_draft_extend_metadata, page_size=4, dp_size=dp_size)
            )
            result = make_metadata(old_metadata, seq_lens, allocated_lens, query_lens=query_lens)

    # Exercise fused EAGLE3 callers with padded slots and page-aligned KV lengths.
    expected_cu_q = [0, 2, 2, 0, 2, 2] if target_verify else [0, 2, 2, 0, 1, 1]
    np.testing.assert_array_equal(np.asarray(result.cu_q_lens), expected_cu_q[: dp_size * 3])
    np.testing.assert_array_equal(np.asarray(result.cu_kv_lens), [0, 8, 8, 0, 4, 4][: dp_size * 3])
    np.testing.assert_array_equal(np.asarray(result.seq_lens), np.asarray(seq_lens))
    np.testing.assert_array_equal(np.asarray(result.page_indices), np.asarray(pages))
    np.testing.assert_array_equal(np.asarray(result.distribution), [0, 1, 1] * dp_size)
    for value in jax.tree.leaves(result):
        assert value.sharding == sharding


@pytest.mark.parametrize("dp_size", [1, 2, 8])
@pytest.mark.parametrize("is_greedy", [True, False])
def test_verify_device_handoff_and_rng_step(monkeypatch, dp_size, is_greedy):
    """Exercise the real verify wrapper with CPU substitutes for model/TPU kernels."""
    from types import SimpleNamespace

    import jax.numpy as jnp

    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.speculative import draft_extend_fused as fused

    mesh = _explicit_mesh(dp_size)
    data = NamedSharding(mesh, P("data"))
    rep = NamedSharding(mesh, P())
    bs = 16

    def model(batch, pools, metadata):
        logits = jax.sharding.reshard(jnp.zeros((bs * 2, 4)), data)
        hidden = jax.sharding.reshard(jnp.zeros((bs * 2, 2)), data)
        return SimpleNamespace(next_token_logits=logits, hidden_states=hidden), (), None, None

    def verify_kernel(**kw):
        # Expose the actual RNG consumed by verify, including stochastic coins.
        marker = (
            jax.random.randint(kw["simulation_rng"], (bs,), 0, 10000)
            if is_greedy
            else (kw["coin_f"] * 10000).astype(jnp.int32)
        )
        index = jnp.arange(bs, dtype=jnp.int32)
        return fused.GreedySampleAndPrepareOutput(
            hidden_states=kw["target_hidden"],
            positions=kw["positions"],
            new_seq_lens=kw["seq_lens"] + 2,
            select_index=index,
            safe_index=index,
            verified_id=index,
            accept_lens=jnp.ones((bs,), dtype=jnp.int32),
            sel_pos=jnp.zeros((bs,), dtype=jnp.int32),
            predict=marker,
        )

    monkeypatch.setattr(fused.nnx, "merge", lambda *_: model)
    monkeypatch.setattr(fused, "_verify_greedy", verify_kernel)
    monkeypatch.setattr(fused, "_verify_rejection_sampling", verify_kernel)
    verify_body = fused._build_verify(1).__wrapped__

    @jax.jit
    def run(seq_lens, step, key):
        batch = SimpleNamespace(seq_lens=seq_lens, spec_info=SimpleNamespace())
        return verify_body(
            None,
            jax.tree.structure(None),
            [],
            batch,
            (),
            LogitsMetadata(None),
            jnp.zeros_like(seq_lens),
            jnp.zeros((bs, 1), dtype=jnp.int32),
            None,
            None,
            None,
            seq_lens,
            key,
            step,
            None,
            None,
            None,
            speculative_num_steps=1,
            speculative_num_draft_tokens=2,
            return_target_logits=False,
            use_relay_state=False,
            dp_size=dp_size,
            is_greedy=is_greedy,
        )

    with jax.set_mesh(mesh):
        lengths = np.arange(bs, dtype=np.int32) * 3
        seq_lens = jax.device_put(lengths, data)
        step = jax.device_put(np.int32(0), rep)
        key = jax.device_put(jax.random.key(3), rep)
        for expected_step in (1, 2, 3):
            result = run(seq_lens, step, key)
            host_lengths, draft_lengths, step = result[5], result[-2], result[-1]
            assert host_lengths.sharding.is_equivalent_to(rep, ndim=1)
            assert draft_lengths.sharding == data
            # No D2H/H2D is allowed when handing the verify output to draft.
            with jax.transfer_guard("disallow"):
                ready = fused._prepare_device_array(draft_lengths, data)
                assert ready is draft_lengths
            np.testing.assert_array_equal(np.asarray(host_lengths), lengths + 2)
            np.testing.assert_array_equal(np.asarray(draft_lengths), lengths + 2)
            assert int(step) == expected_step
            old_rng = jax.random.fold_in(key, expected_step)
            if is_greedy:
                expected = jax.random.randint(jax.random.fold_in(old_rng, 1), (bs,), 0, 10000)
            else:
                _, coin_f_key = jax.random.split(old_rng)
                expected = (jax.random.uniform(coin_f_key, (bs,)) * 10000).astype(jnp.int32)
            np.testing.assert_array_equal(np.asarray(result[12]), np.asarray(expected))
