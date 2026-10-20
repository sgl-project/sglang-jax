"""The paged_write_back run table never overflows r_cap = T//ps + 2*S + 2 for extend-form slot
layouts (new slots of every sequence are page-contiguous: chunked prefill, radix prefix on page
boundaries, MIXED 1-query extends) -- the premise of DSA_WRITEBACK_NOCOND."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import _build_write_runs

PS = 128


def _layout(rng, q_lens, prefix_lens, total_pages=4096):
    """Flat slot per new token: each sequence owns random pages; its new tokens fill slots
    prefix_len .. prefix_len+q_len-1 of its page list (page-contiguous within a page)."""
    locs = []
    for q, pre in zip(q_lens, prefix_lens):
        n_pages = (pre + q + PS - 1) // PS
        pages = rng.choice(total_pages, size=n_pages, replace=False)
        for t in range(pre, pre + q):
            locs.append(int(pages[t // PS]) * PS + t % PS)
    return np.asarray(locs, np.int32)


def test_runs_within_bound_extend_forms():
    rng = np.random.default_rng(0)
    cases = [
        ([4096], [0]),  # one 4k prompt
        ([2048, 2048], [0, 0]),  # two windows worth / two prompts
        ([4096] * 8, [0] * 8),  # cc8 batch of 4k prompts
        ([3504], [106496]),  # last 110k chunk with a long prefix
        ([1] * 64, list(rng.integers(0, 5000, 64))),  # MIXED: 64 one-token extends
        ([700, 1, 1300, 5, 2000], [0, 777, 128, 4096, 33]),  # mixed lengths, odd prefixes
        ([2] * 64, [PS - 1] * 64),  # 64 two-token extends, every one straddling a page boundary
        ([3] * 40, [PS - 2] * 40),  # 40 straddling three-token extends
    ]
    for q_lens, prefix_lens in cases:
        loc = _layout(rng, q_lens, prefix_lens)
        T, S = loc.shape[0], len(q_lens)
        pad = (-T) % 256
        loc = np.concatenate([loc, -np.ones(pad, np.int32)])  # token padding (loc == -1)
        Tp = loc.shape[0]
        bound = Tp // PS + 2 * S + 2
        _, n_raw = _build_write_runs(jnp.asarray(loc), kv_packing=2, r_cap=4 * Tp)
        assert int(n_raw) <= bound, (q_lens, prefix_lens, int(n_raw), bound)
