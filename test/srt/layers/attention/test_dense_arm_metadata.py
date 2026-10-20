"""dense_arm_metadata: sequences with no query rows in a prefill window must not reach the dense MLA kernel with
kv_len > 0 (mixed-kernel hang pattern, knife 1 V2 2026-10-07)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsa_sparse_backend import (
    dense_arm_metadata,
    window_forward_metadata,
)
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata


def _md(q_lens, kv_lens):
    md = MLAAttentionMetadata()
    md.cu_q_lens = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), dtype=jnp.int32)
    md.seq_lens = jnp.asarray(kv_lens, dtype=jnp.int32)
    md.cu_kv_lens = jnp.asarray(np.concatenate([[0], np.cumsum(kv_lens)]), dtype=jnp.int32)
    md.page_indices = jnp.arange(64, dtype=jnp.int32)
    md.distribution = jnp.asarray([0, 0, len(q_lens)], dtype=jnp.int32)
    return md


def test_window_b_hides_prefix_only_sequences():
    # 8 x 1k batch, window B = rows 4096..8191: seqs 0-3 have q_len 0 but kv_len 1024 in the raw window metadata
    md = _md([1024] * 8, [1024] * 8)
    wb = window_forward_metadata(md, 4096, 4096)
    assert [int(v) for v in wb.seq_lens[:4]] == [1024] * 4
    d = dense_arm_metadata(wb)
    assert [int(v) for v in d.seq_lens] == [0, 0, 0, 0, 1024, 1024, 1024, 1024]
    assert d.cu_q_lens is wb.cu_q_lens and d.page_indices is wb.page_indices
    # the kernel's MIXED range skips the leading q_len == 0 sequences entirely
    assert [int(v) for v in d.distribution] == [0, 4, 8]
    wa = window_forward_metadata(md, 0, 4096)
    da = dense_arm_metadata(wa)
    assert [int(v) for v in da.distribution] == [0, 0, 4]
    assert [int(v) for v in da.seq_lens] == [1024] * 4 + [0] * 4


def test_single_window_padding_unchanged():
    # bs-64 bucket: 1 real + 63 padded (q 0, kv 0) -> identical
    md = _md([1024] + [0] * 63, [1024] + [0] * 63)
    d = dense_arm_metadata(md)
    assert [int(v) for v in d.seq_lens] == [int(v) for v in md.seq_lens]
    assert [int(v) for v in d.distribution] == [0, 0, 1]


def test_non_flat_metadata_passthrough():
    md = _md([512, 512], [512, 512])
    md.cu_q_lens = jnp.zeros((2, 3), dtype=jnp.int32)  # dp>1 style: not [S+1]
    assert dense_arm_metadata(md) is md


def test_window_without_rows_gets_empty_range():
    md = _md([1024] * 8, [1024] * 8)
    wb = window_forward_metadata(md, 8192, 4096)  # no rows in this window
    d = dense_arm_metadata(wb)
    assert [int(v) for v in d.distribution] == [0, 0, 0]


def test_explicit_mesh_sharding_preserved():
    """Under the production Explicit (data, tensor) mesh every metadata field must keep its P('data') sharding:
    _run_dense's shard_map declares in_specs=P(data) for distribution (gp786: a replicated distribution crashed).
    Production builds the window metadata and the dense-arm metadata inside the model jit, so do the same here.
    """
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = np.array(jax.devices()[:4]).reshape(1, 4)
    mesh = jax.sharding.Mesh(
        devices, axis_names=("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    data = NamedSharding(mesh, P("data"))
    md = _md([1024] * 8 + [0] * 56, [1024] * 8 + [0] * 56)
    with jax.set_mesh(mesh):
        args = [
            jax.device_put(getattr(md, n), data)
            for n in ("cu_q_lens", "cu_kv_lens", "page_indices", "seq_lens", "distribution")
        ]

        @jax.jit
        def f(cuq, cukv, pi, sl, dist):
            m = MLAAttentionMetadata()
            m.cu_q_lens, m.cu_kv_lens, m.page_indices, m.seq_lens, m.distribution = (
                cuq,
                cukv,
                pi,
                sl,
                dist,
            )
            wb = window_forward_metadata(m, 4096, 4096)
            d = dense_arm_metadata(wb)
            return wb.seq_lens, d.seq_lens, d.distribution

        wsl, sl, dist = f(*args)
    assert dist.sharding.spec == args[4].sharding.spec, (dist.sharding, args[4].sharding)
    assert sl.sharding.spec == wsl.sharding.spec
    assert np.asarray(dist).tolist() == [0, 4, 8]
    assert np.asarray(wsl)[:4].tolist() == [1024] * 4
    assert np.asarray(sl)[:8].tolist() == [0] * 4 + [1024] * 4
