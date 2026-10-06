"""EAGLE draft-decode attention layout: per-step metadata and tree mask.

The tree mask and the per-step attention metadata are built separately but
must describe the same KV window. These tests drive the real producer
(``select_top_k_tokens``) and check both sides against an ancestry oracle that
does not use the parent-pointer arithmetic: each draft row's hidden state
carries its own (request, branch) identity, so the hidden row a child inherits
names its parent.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    ragged_paged_attention,
)
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.eagle_draft_worker import select_top_k_tokens
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def _mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))


def _base(request):
    """First KV slot of a request; page-aligned for every page size tested."""
    return 1024 * (request + 1)


def _batch(seq_lens, padded_bs, topk, steps, page_size=1):
    """A dp=1 draft-decode batch with real slots first."""
    real = len(seq_lens)
    seq = np.zeros(padded_bs, dtype=np.int32)
    seq[:real] = seq_lens
    alloc = np.zeros(padded_bs, dtype=np.int32)
    alloc[:real] = np.asarray(seq_lens) + steps * topk
    # Each request owns its own pages so a window can be traced to its owner.
    pages = -(-alloc // page_size)
    cache_loc = np.concatenate(
        [_base(k) + np.arange(pages[k] * page_size, dtype=np.int32) for k in range(real)]
    )
    cache_loc = np.pad(cache_loc, (0, max(0, 4096 - len(cache_loc))))
    return SimpleNamespace(
        cache_loc=cache_loc,
        forward_mode=ForwardMode.DECODE,
        seq_lens=seq,
        logits_indices_selector=np.arange(real, dtype=np.int32),
        spec_info_padded=EagleDraftInput(allocate_lens=alloc),
        dp_size=1,
        per_dp_bs_size=padded_bs,
        speculative_num_steps=steps,
        speculative_eagle_topk=topk,
        spec_algorithm=SpeculativeAlgorithm.EAGLE3,
    )


def _draft_round(padded_bs, topk, steps, seed):
    """Run the real draft-step selection; return per-step parents and an oracle.

    ``oracle[s][slot, branch]`` is the step ``s - 1`` branch that the step ``s``
    branch descends from, read off the hidden row it inherited.
    """
    rng = np.random.default_rng(seed)
    hidden_dim = 2
    topk_p = jnp.asarray(rng.random((padded_bs, topk), dtype=np.float32))
    topk_index = jnp.asarray(rng.integers(0, 50, (padded_bs, topk), dtype=np.int32))
    hidden = jnp.zeros((padded_bs, hidden_dim), dtype=jnp.float32)
    scores = None
    parents_by_step, oracle = [], [None]
    for i in range(steps - 1):
        if i > 0:
            # Tag every row with its own branch before the selection reads it.
            branch = np.tile(np.arange(topk, dtype=np.float32), padded_bs)
            hidden = jnp.asarray(np.stack([branch, branch], axis=1))
        _, hidden, scores, tree_info = select_top_k_tokens(
            i, topk_p, topk_index, hidden, scores, topk
        )
        parents_by_step.append(tree_info[2])
        if i > 0:
            oracle.append(np.asarray(hidden)[:, 0].astype(np.int32).reshape(padded_bs, topk))
        # The next draft forward scores topk children for every branch.
        topk_p = jnp.asarray(rng.random((padded_bs * topk, topk), dtype=np.float32))
        topk_index = jnp.asarray(rng.integers(0, 50, (padded_bs * topk, topk), dtype=np.int32))
    return parents_by_step, oracle


def _expected_rows(seq_len, step, topk, oracle, slot, width):
    rows = np.zeros((topk, width), dtype=np.int32)
    base = seq_len - 1
    for branch in range(topk):
        rows[branch, :base] = 1
        node = branch
        for s in range(step, -1, -1):
            rows[branch, base + s * topk + node] = 1
            if s > 0:
                node = oracle[s][slot, node]
    return rows


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize(
    "seq_lens, padded_bs, topk, steps",
    [
        ([5, 9, 3], 4, 2, 4),
        ([17, 6], 2, 3, 3),
        ([125, 4, 70], 4, 4, 4),  # the widest row grows from 128 to 136 within the round
    ],
)
def test_draft_mask_matches_attention_window(seq_lens, padded_bs, topk, steps, page_size):
    backend = FlashAttention(8, 8, 128, page_size=page_size, mesh=_mesh())
    batch = _batch(seq_lens, padded_bs, topk, steps, page_size)
    metadata = backend.get_eagle_multi_step_metadata(batch)
    parents_by_step, oracle = _draft_round(padded_bs, topk, steps, seed=sum(seq_lens))

    widths = set()
    for i in range(steps - 1):
        mask = np.asarray(backend.get_eagle_draft_decode_mask(batch, i, parents_by_step[: i + 1]))
        kv_lens = np.asarray(metadata[i].seq_lens)
        assert mask.shape[:2] == (padded_bs * topk, 1)
        assert mask.shape[2] % 128 == 0
        widths.add(mask.shape[2])

        for slot, seq_len in enumerate(seq_lens):
            kv_len = seq_len - 1 + (i + 1) * topk
            assert kv_lens[slot] == kv_len
            # The page window is the request's own first pages covering kv_len.
            num_pages = -(-kv_len // page_size)
            first = sum(-(-(len_ - 1 + (i + 1) * topk) // page_size) for len_ in seq_lens[:slot])
            window = np.asarray(metadata[i].page_indices)[first : first + num_pages]
            np.testing.assert_array_equal(window, _base(slot) // page_size + np.arange(num_pages))

            rows = mask[slot * topk : (slot + 1) * topk, 0, :]
            np.testing.assert_array_equal(
                rows, _expected_rows(seq_len, i, topk, oracle, slot, mask.shape[2])
            )
        assert not mask[len(seq_lens) * topk :].any(), "padding rows must stay masked"
        assert not kv_lens[len(seq_lens) :].any()

    assert len(widths) == 1, "every step of a round must share one mask shape"


@pytest.mark.parametrize("seq_lens, steps", [([5, 9, 3], 4), ([1, 40], 3)])
def test_chain_window_is_unchanged(seq_lens, steps):
    """topk == 1 keeps the pre-tree layout: kv_len = seq_len + step."""
    backend = FlashAttention(8, 8, 128, page_size=1, mesh=_mesh())
    batch = _batch(seq_lens, len(seq_lens) + 1, 1, steps)
    metadata = backend.get_eagle_multi_step_metadata(batch)
    for i in range(steps):
        kv_lens = np.asarray(metadata[i].seq_lens)
        np.testing.assert_array_equal(kv_lens[: len(seq_lens)], np.asarray(seq_lens) + i)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Requires TPU DMA support")
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize(
    "seq_lens, padded_bs, topk, steps",
    [
        ([7, 5], 2, 2, 3),
        ([300, 4, 130], 4, 3, 4),  # multiple kv blocks per sequence
    ],
)
def test_draft_attention_matches_tree_reference(seq_lens, padded_bs, topk, steps, page_size):
    """Production metadata and mask through the real kernel.

    Every real branch must attend over exactly its oracle-visible positions, and
    the step's new K/V must land on the last ``topk`` slots of the request's own
    window and nowhere else.
    """
    head_dim, num_q_heads = 128, 4
    backend = FlashAttention(num_q_heads, 1, head_dim, page_size=page_size, mesh=_mesh())
    batch = _batch(seq_lens, padded_bs, topk, steps, page_size)
    metadata = backend.get_eagle_multi_step_metadata(batch)
    parents_by_step, oracle = _draft_round(padded_bs, topk, steps, seed=len(seq_lens))
    rng = np.random.default_rng(7)
    num_slots = _base(len(seq_lens))
    rows = padded_bs * topk

    for i in range(steps - 1):
        mask = backend.get_eagle_draft_decode_mask(batch, i, parents_by_step[: i + 1])
        step = metadata[i]
        kv_lens = np.asarray(step.seq_lens)
        # Integer values keep bf16 exact; -7 marks slots the kernel must not touch.
        cache = np.full((num_slots, 1, 1, 2, head_dim), -7, np.float32)
        new_kv = rng.integers(-4, 5, size=(rows, 2, head_dim)).astype(np.float32)
        expected_cache = cache.copy()
        windows = {}
        for b, seq_len in enumerate(seq_lens):
            kv = int(kv_lens[b])
            win = _base(b) + np.arange(kv)
            context = rng.integers(-4, 5, size=(kv - topk, 2, head_dim)).astype(np.float32)
            cache[win[: kv - topk], 0, 0] = context
            expected_cache[win[: kv - topk], 0, 0] = context
            expected_cache[win[kv - topk :], 0, 0] = new_kv[b * topk : (b + 1) * topk]
            windows[b] = np.concatenate([context, new_kv[b * topk : (b + 1) * topk]])
        queries = rng.uniform(-0.25, 0.25, (rows, num_q_heads, head_dim)).astype(np.float32)

        output, updated_cache = ragged_paged_attention(
            jnp.asarray(queries, jnp.bfloat16),
            jnp.asarray(new_kv[:, 0:1], jnp.bfloat16),
            jnp.asarray(new_kv[:, 1:2], jnp.bfloat16),
            jnp.asarray(cache.reshape(-1, page_size, *cache.shape[2:]), jnp.bfloat16),
            step.seq_lens,
            step.page_indices,
            step.cu_q_lens,
            step.cu_kv_lens,
            step.distribution,
            mask,
            causal=0,
            sm_scale=head_dim**-0.5,
        )
        output, updated_cache = jax.device_get((output, updated_cache))
        updated_cache = updated_cache.reshape(cache.shape)

        q_ref = jnp.asarray(queries, jnp.bfloat16).astype(np.float32)
        for b, seq_len in enumerate(seq_lens):
            kv = int(kv_lens[b])
            visible = _expected_rows(seq_len, i, topk, oracle, b, kv).astype(bool)
            keys, values = windows[b][:, 0], windows[b][:, 1]
            for r in range(topk):
                scores = np.asarray(q_ref[b * topk + r]) @ keys.T * head_dim**-0.5
                scores = np.where(visible[r], scores, -np.inf)
                probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
                probs /= probs.sum(axis=-1, keepdims=True)
                np.testing.assert_allclose(
                    np.asarray(output[b * topk + r], np.float32),
                    probs @ values,
                    atol=0.03,
                    rtol=0.03,
                    err_msg=f"step {i} request {b} branch {r}",
                )
        np.testing.assert_array_equal(updated_cache.astype(np.float32), expected_cache)


def test_draft_window_must_fit_the_allocation():
    backend = FlashAttention(8, 8, 128, page_size=1, mesh=_mesh())
    batch = _batch([5, 9], 2, 4, 4)
    batch.spec_info_padded.allocate_lens = np.asarray([8, 9 + 3 * 4], dtype=np.int32)
    with pytest.raises(AssertionError, match="pages but only"):
        backend.get_eagle_multi_step_metadata(batch)


def test_draft_page_table_grows_for_long_windows():
    """At page_size 1 a page is a token; a batch whose draft windows outgrow
    the default page table gets a larger one instead of an IndexError."""
    topk, steps, seq_len = 2, 3, 6000
    backend = FlashAttention(8, 8, 128, page_size=1, mesh=_mesh())
    batch = _batch([seq_len] * 3, 3, topk, steps)
    alloc = seq_len + steps * topk
    batch.cache_loc = np.concatenate([20000 * (k + 1) + np.arange(alloc) for k in range(3)])
    metadata = backend.get_eagle_multi_step_metadata(batch)

    for i in range(steps):
        kv_len = seq_len - 1 + (i + 1) * topk
        pages = np.asarray(metadata[i].page_indices)
        assert pages.size >= 3 * kv_len
        for k in range(3):
            np.testing.assert_array_equal(
                pages[k * kv_len : (k + 1) * kv_len], 20000 * (k + 1) + np.arange(kv_len)
            )
