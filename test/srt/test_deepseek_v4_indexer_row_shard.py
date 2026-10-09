"""DSV4_INDEXER_ROW_SHARD: per-device row-block indexer scoring + all-gathered membership equals
the replicated full-T membership (CPU, 8 host devices, shard_map over 'tensor').

Runs in its own process: the XLA_FLAGS / JAX_PLATFORMS / DSV4_* settings below must be
in place before jax initialises, and they stay set for anything imported afterwards
(run_suite starts one process per file)."""

import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["DSV4_CSA_TOPK_MASK"] = "1"
os.environ["DSV4_INDEXER_ROW_SHARD"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from test_deepseek_v4_indexer_kernel import CPS, RATIO, K, _synthetic_batch

if len(jax.devices()) < 8:
    pytest.skip(
        "needs 8 host devices (XLA_FLAGS set before jax initialised)", allow_module_level=True
    )


from sgl_jax.srt.layers.attention.dsv4 import dispatch
from sgl_jax.srt.layers.attention.dsv4.indexer import (
    csa_indexer_topk_kernel,
    membership_from_scores,
    visible_entries_for_query,
)
from sgl_jax.srt.layers.attention.dsv4.ref.indexer import csa_indexer_scores_ref


def cpu_scorer(
    q,
    w,
    buf,
    *,
    compressed_rows,
    seq_lens,
    q_lens,
    cu_q_lens,
    query_request_ids,
    valid_token_mask,
    k,
    ratio,
    compressed_page_size,
    return_scores,
):
    """CPU stand-in with csa_indexer_topk_kernel's contract for ONE request: the q_len
    queries are the last positions of seq_len; scores [T, E] over the gathered rows,
    entries not yet complete at a query's position (or padded rows) get -inf."""
    keys = jnp.asarray(buf)[jnp.asarray(compressed_rows)][:, None, :]
    scores = csa_indexer_scores_ref(jnp.asarray(q), jnp.asarray(w), keys)
    tokens = q.shape[0]
    q_len = jnp.asarray(q_lens)[0]
    pos = jnp.asarray(seq_lens)[0] - q_len + jnp.arange(tokens, dtype=jnp.int32)
    visible = visible_entries_for_query(pos, ratio)
    entries = jnp.arange(scores.shape[1], dtype=jnp.int32)
    legal = (entries[None, :] < visible[:, None]) & jnp.asarray(valid_token_mask, bool)[:, None]
    legal = legal & (jnp.arange(tokens) < q_len)[:, None]
    return jnp.where(legal, scores, -jnp.inf), jnp.zeros((1,), jnp.int32)


@pytest.mark.parametrize("prefix,q_len,padded", [(0, 512, 0), (256, 300, 212), (0, 40, 472)])
def test_row_shard_membership_matches_full(prefix, q_len, padded):
    b = _synthetic_batch([(prefix, q_len)], seed=prefix + q_len, padded_queries=padded)
    tables = types.SimpleNamespace(
        compressed_rows=b["tables"]["compressed_rows"],
        compressed_entry_ids=b["tables"]["compressed_entry_ids"],
    )
    num_entries = int(np.shape(b["tables"]["compressed_entry_ids"])[0])
    md = types.SimpleNamespace(
        cu_q_lens=b["cu_q_lens"],
        prefix_lens=jnp.asarray([prefix], jnp.int32),
        valid_token_mask=b["valid_token_mask"],
        page_size=CPS * RATIO,
        seq_lens=b["seq_lens"],
        q_lens=b["q_lens"],
        query_request_ids=b["query_request_ids"],
    )
    scorer = None if jax.default_backend() == "tpu" else cpu_scorer  # real kernel on TPU
    scores, offsets = (csa_indexer_topk_kernel if scorer is None else scorer)(
        b["q"],
        b["weights"],
        b["indexer_buffer"],
        compressed_rows=tables.compressed_rows,
        seq_lens=md.seq_lens,
        q_lens=md.q_lens,
        cu_q_lens=md.cu_q_lens,
        query_request_ids=md.query_request_ids,
        valid_token_mask=md.valid_token_mask,
        k=K,
        ratio=RATIO,
        compressed_page_size=CPS,
        return_scores=True,
    )
    want = membership_from_scores(
        scores,
        offsets,
        q_lens=md.q_lens,
        query_request_ids=md.query_request_ids,
        valid_token_mask=md.valid_token_mask,
        k=K,
        num_entries=num_entries,
    )

    mesh = Mesh(np.asarray(jax.devices()[:8]), axis_names=("tensor",))

    def local(q, w, buf):
        return dispatch._row_sharded_membership(
            q,
            w,
            buf,
            metadata=md,
            tables=tables,
            k=K,
            ratio=RATIO,
            num_entries=num_entries,
            scorer=scorer,
        )

    got = jax.jit(
        jax.shard_map(local, mesh=mesh, in_specs=(P(), P(), P()), out_specs=P(), check_vma=False)
    )(b["q"], b["weights"], b["indexer_buffer"])
    assert got.shape == want.shape and got.dtype == want.dtype
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # sanity: a real row selects min(visible complete groups, k) entries, padded rows none
    rows = np.asarray(want).sum(axis=1)
    visible = (prefix + np.arange(q_len) + 1) // RATIO
    assert (rows[:q_len] == np.minimum(visible, K)).all() and (rows[q_len:] == 0).all()
