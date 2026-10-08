"""pack_metadata / unpack_metadata round trip for the V4 per-step metadata (CPU-only)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4RuntimeMetadata,
    pack_metadata,
    unpack_metadata,
)


def _trees():
    attention = {
        "query_positions": np.arange(9, dtype=np.int32),
        "valid_token_mask": np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool),
        "cu_q_lens": np.array([0, 3, 3, 7], np.int32),
        "nested": {"pages": np.arange(12, dtype=np.int32).reshape(3, 4), "flag": np.array([True])},
    }
    tables = (
        {"rows": np.array([5, 6, 7], np.int32), "ids": np.array([-1, 0, 1], np.int32)},
        {"rows": np.zeros((0,), np.int32), "ids": np.zeros((0,), np.int32)},
        {"rows": np.array([9], np.int32), "ids": np.array([2], np.int32)},
    )
    return attention, tables


def _assert_tree_equal(a, b):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    for x, y in zip(la, lb):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape, (x.dtype, y.dtype, x.shape, y.shape)
        np.testing.assert_array_equal(x, y)


def test_pack_unpack_round_trip_host():
    attention, tables = _trees()
    packed, layout = pack_metadata(attention, tables)
    assert packed.dtype == np.int32 and packed.ndim == 1
    got_attention, got_tables = unpack_metadata(packed, layout)
    _assert_tree_equal((attention, tables), (got_attention, tuple(got_tables)))


def test_unpack_inside_jit_and_resolve():
    attention, tables = _trees()
    packed, layout = pack_metadata(attention, tables)
    md = DeepseekV4RuntimeMetadata(None, None, False, None, None, (), jnp.asarray(packed), layout)
    assert md.has_metadata()

    @jax.jit
    def f(md):
        att, tabs = md.resolve()
        return att["query_positions"].sum(), att["valid_token_mask"].sum(), tabs[0]["rows"]

    s, v, rows = f(md)
    assert int(s) == 36 and int(v) == 6
    np.testing.assert_array_equal(np.asarray(rows), [5, 6, 7])
    # pytree round trip keeps the static layout in aux and the vector as the only new leaf
    leaves, treedef = jax.tree_util.tree_flatten(md)
    md2 = jax.tree_util.tree_unflatten(treedef, leaves)
    _assert_tree_equal(md2.resolve(), md.resolve())


def test_pack_rejects_non_integer_leaves():
    with pytest.raises(TypeError):
        pack_metadata({"x": np.zeros(3, np.float32)}, ())


def test_resolve_is_memoized_per_instance():
    attention, tables = _trees()
    packed, layout = pack_metadata(attention, tables)
    md = DeepseekV4RuntimeMetadata(None, None, False, None, None, (), jnp.asarray(packed), layout)
    a1 = md.resolve()
    a2 = md.resolve()
    # the unpacked leaves are built once per instance and shared by every call
    assert a1[0]["cu_q_lens"] is a2[0]["cu_q_lens"] and a1[1][0]["rows"] is a2[1][0]["rows"]
    # a fresh instance (what jit's tree_unflatten produces per trace) does not share the memo
    leaves, treedef = jax.tree_util.tree_flatten(md)
    md2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert md2.resolve()[0]["cu_q_lens"] is not a1[0]["cu_q_lens"]
    _assert_tree_equal(md2.resolve(), a1)


def test_four_tree_pack_exposes_hca_view():
    attention, tables = _trees()
    kernel = {"cu": np.array([0, 2, 5], np.int32), "valid": np.array([1, 0, 1, 1], bool)}
    init_slots = np.array([3, 17], np.int32)
    packed, layout = pack_metadata(attention, tables, kernel, init_slots)
    md = DeepseekV4RuntimeMetadata(
        None, "sched", False, None, None, (), jnp.asarray(packed), layout
    )
    att, tabs = md.resolve()
    _assert_tree_equal((attention, tables), (att, tuple(tabs)))
    view = md.hca_metadata()
    assert view.schedule == "sched" and view.use_uniform_prefill_fast_path is False
    _assert_tree_equal(kernel, view.kernel)
    np.testing.assert_array_equal(np.asarray(view.state_init_slots), init_slots)
    # unpacked once per instance
    assert md._unpacked() is md._unpacked()
