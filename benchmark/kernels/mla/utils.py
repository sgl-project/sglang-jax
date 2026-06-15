"""Uniform-shape data generators for the MLA v2 Pallas kernel.

The MLA kernel consumes the 4-tuple
``(ql_nope, q_pe, new_kv_c, new_k_pe)`` plus a ragged page-indices layout.
For tuning / benchmarking we just need realistic shapes — values can be
random, the cache pre-population doesn't need to be correct.

Helpers mirror the spirit of
``benchmark/kernels/flash_attention/utils.py``'s
``create_decode_uniform_data`` / ``create_prefill_uniform_data`` but produce
MLA-shaped tensors:

  ql_nope:     [T, n_h, kv_lora_rank]
  q_pe:        [T, n_h, qk_rope_head_dim]
  new_kv_c:    [T, kv_lora_rank]
  new_k_pe:    [T, qk_rope_head_dim]
  cache_kv:    [total_num_pages, page_size_per_kv_packing, kv_packing,
                align_to(kv_lora_rank,128) + align_to(qk_rope_head_dim,128)]
  kv_lens:     i32[num_seqs]
  page_indices: i32[sum(pages_per_seq)]   (ragged layout)
  cu_q_lens:   i32[num_seqs+1]
  cu_kv_lens:  i32[num_seqs+1]            (page-aligned cumsum)
  distribution: i32[3]

Distribution semantics mirror the kernel's dispatch:
  decode forward → [N, N, N] → BATCHED_DECODE + DECODE-tail run, MIXED empty
  extend forward → [0, 0, N] → MIXED runs, decode branches empty
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.mla.v2.kernel import align_to, get_dtype_packing


def _build_kernel_inputs(
    *,
    max_num_tokens: int,
    num_seqs: int,
    pages_per_seq: int,
    aligned_kv_len: int,
    actual_kv_len: int,
    num_q_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    distribution_array: np.ndarray,
    dtype,
    seed: int,
):
    kv_packing = get_dtype_packing(dtype)
    lkv_dim = align_to(kv_lora_rank, 128)
    r_dim = align_to(qk_rope_head_dim, 128)

    page_size_per_kv_packing = max(align_to(page_size, kv_packing) // kv_packing, 1)
    total_num_pages = max(num_seqs * pages_per_seq, 1)

    key = jax.random.PRNGKey(seed)
    k_q, k_qpe, k_kv, k_kpe, k_cache = jax.random.split(key, 5)

    ql_nope = jax.random.normal(k_q, (max_num_tokens, num_q_heads, kv_lora_rank), dtype=dtype)
    q_pe = jax.random.normal(k_qpe, (max_num_tokens, num_q_heads, qk_rope_head_dim), dtype=dtype)
    new_kv_c = jax.random.normal(k_kv, (max_num_tokens, kv_lora_rank), dtype=dtype)
    new_k_pe = jax.random.normal(k_kpe, (max_num_tokens, qk_rope_head_dim), dtype=dtype)
    cache_kv = jax.random.normal(
        k_cache,
        (total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim + r_dim),
        dtype=dtype,
    )

    kv_lens = jnp.full((num_seqs,), actual_kv_len, dtype=jnp.int32)
    # Each seq's pages are tightly concatenated:
    #   seq 0 -> [0, pages_per_seq)
    #   seq 1 -> [pages_per_seq, 2 * pages_per_seq)
    #   ...
    page_indices = jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32)
    distribution = jnp.asarray(distribution_array, dtype=jnp.int32)

    return dict(
        ql_nope=ql_nope,
        q_pe=q_pe,
        new_kv_c=new_kv_c,
        new_k_pe=new_k_pe,
        cache_kv=cache_kv,
        kv_lens=kv_lens,
        page_indices=page_indices,
        distribution=distribution,
    )


def create_mla_decode_uniform_data(
    max_num_tokens: int,
    num_q_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    kv_len: int = 16384,
    dtype=jnp.bfloat16,
    seed: int = 42,
) -> dict:
    """Decode-mode workload: each of `max_num_tokens` seqs decodes 1 token.

    distribution = [N, N, N] so the kernel runs BATCHED_DECODE + DECODE-tail
    (slot[0]) and MIXED grid is empty (slot[2]).
    """
    num_seqs = max_num_tokens
    aligned_kv_len = align_to(kv_len, page_size)
    pages_per_seq = aligned_kv_len // page_size

    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)
    cu_kv_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32) * aligned_kv_len
    distribution = np.full((3,), num_seqs, dtype=np.int32)

    base = _build_kernel_inputs(
        max_num_tokens=max_num_tokens,
        num_seqs=num_seqs,
        pages_per_seq=pages_per_seq,
        aligned_kv_len=aligned_kv_len,
        actual_kv_len=kv_len,
        num_q_heads=num_q_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        distribution_array=distribution,
        dtype=dtype,
        seed=seed,
    )
    base["cu_q_lens"] = cu_q_lens
    base["cu_kv_lens"] = cu_kv_lens
    return base


def create_mla_mixed_uniform_data(
    max_num_tokens: int,
    num_q_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    kv_len: int = 16384,
    dtype=jnp.bfloat16,
    seed: int = 42,
) -> dict:
    """Mixed/extend-mode workload: 1 seq doing a chunked prefill of `max_num_tokens`
    tokens against an existing prefix totalling `kv_len` tokens.

    distribution = [0, 0, 1] so only the MIXED pallas_call (slot[2]) does work.
    The kernel still compiles BATCHED_DECODE / DECODE branches but their grid
    is `(0,)` -> no-op.
    """
    if kv_len < max_num_tokens:
        raise ValueError(
            f"kv_len={kv_len} must be >= max_num_tokens={max_num_tokens}; "
            "for mixed, kv_len represents the total ctx including the new chunk."
        )
    num_seqs = 1
    aligned_kv_len = align_to(kv_len, page_size)
    pages_per_seq = aligned_kv_len // page_size

    cu_q_lens = jnp.asarray([0, max_num_tokens], dtype=jnp.int32)
    cu_kv_lens = jnp.asarray([0, aligned_kv_len], dtype=jnp.int32)
    distribution = np.asarray([0, 0, num_seqs], dtype=np.int32)

    base = _build_kernel_inputs(
        max_num_tokens=max_num_tokens,
        num_seqs=num_seqs,
        pages_per_seq=pages_per_seq,
        aligned_kv_len=aligned_kv_len,
        actual_kv_len=kv_len,
        num_q_heads=num_q_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        distribution_array=distribution,
        dtype=dtype,
        seed=seed,
    )
    base["cu_q_lens"] = cu_q_lens
    base["cu_kv_lens"] = cu_kv_lens
    return base
