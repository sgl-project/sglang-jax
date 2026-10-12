"""MiniMax-M3 MSA (block-sparse decode) attention backend.

Composes FlashAttention with the MSA-specific pieces: parsing
``sparse_attention_config``, startup validation, the per-token ``index_k``
cache class, and the decode context-page bucket policy consumed by the shared
precompile / AOT shape enumeration (``CompilationManager.iter_model_shapes``).
"""

from __future__ import annotations

import logging

import jax
import numpy as np

from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.mem_cache.memory_pool import MSATokenToKVPool

logger = logging.getLogger(__name__)


def msa_sparse_config(model_config) -> dict | None:
    """Return the MSA ``sparse_attention_config`` dict when the model enables it."""
    sa = getattr(model_config.hf_text_config, "sparse_attention_config", None)
    if isinstance(sa, dict) and sa.get("use_sparse_attention"):
        return sa
    return None


class MSAAttentionBackend(FlashAttention):
    """FlashAttention + MSA per-head block-sparse decode (MiniMax-M3).

    Prefill runs the dense FlashAttention path and additionally writes per-token
    ``index_k``; decode selects top-k blocks per index head (one KV/GQA group per
    tensor rank) and runs RPA over only those pages.
    """

    token_to_kv_pool_class = MSATokenToKVPool

    def __init__(
        self,
        num_attn_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        page_size: int,
        mesh: jax.sharding.Mesh,
        sparse_config: dict,
        context_len: int,
        total_num_kv_heads: int,
    ):
        super().__init__(num_attn_heads, num_kv_heads, head_dim, page_size=page_size, mesh=mesh)
        sa = sparse_config
        self.block_size = int(sa["sparse_block_size"])
        self.topk_blocks = int(sa["sparse_topk_blocks"])
        self.num_index_heads = int(sa["sparse_num_index_heads"])
        self.index_head_dim = int(sa["sparse_index_dim"])
        self.sparse_layer_ids = [i for i, f in enumerate(sa["sparse_attention_freq"]) if f]

        if page_size != self.block_size:
            raise ValueError(
                f"MSA models require --page-size {self.block_size} (== sparse_block_size); "
                f"got --page-size {page_size}. The decode top-k selection uses page "
                "granularity as the block granularity."
            )
        tp = int(mesh.shape[self.kv_partition_axis])
        n_idx, n_kv = self.num_index_heads, int(total_num_kv_heads)
        if n_idx != n_kv or tp < n_idx or tp % n_idx != 0:
            raise ValueError(
                "MSA per-head block selection requires one KV/GQA group per tensor rank: "
                f"sparse_num_index_heads={n_idx}, num_kv_heads={n_kv}, attention tp={tp}. "
                f"Use a tp that is a multiple of {n_kv}."
            )

        # Decode is traced at a few pages-per-seq buckets so a max_ctx=64K server
        # still uses small index_k gathers for short requests. The topk-sized
        # bucket compiles the `pages_per_seq <= topk` (skip-select ≡ dense) branch.
        max_pages = (int(context_len) + page_size - 1) // page_size
        raw = (self.topk_blocks, 4 * self.topk_blocks, max_pages)
        self.decode_page_buckets = sorted({min(b, max_pages) for b in raw if b > 0})
        logger.info(
            "[MSA] sparse layers=%d topk=%d index heads=%d dim=%d; decode page buckets=%s",
            len(self.sparse_layer_ids),
            self.topk_blocks,
            self.num_index_heads,
            self.index_head_dim,
            self.decode_page_buckets,
        )

    @property
    def token_to_kv_pool_kwargs(self) -> dict:
        return dict(
            sparse_layer_ids=list(self.sparse_layer_ids), index_head_dim=self.index_head_dim
        )

    def extra_kv_bytes_per_token(self, dtype_size: int) -> int:
        """Per-token ``index_k`` cache bytes (one index head per sparse layer, not tensor-sharded)."""
        padded = (self.index_head_dim + 127) // 128 * 128
        return len(self.sparse_layer_ids) * padded * dtype_size

    def _decode_page_limit(self, batch) -> int | None:
        """Smallest decode page bucket covering the longest request in ``batch``."""
        seq_pages = (np.asarray(batch.seq_lens) + self.page_size - 1) // self.page_size
        max_pages = int(seq_pages.max(initial=0))
        return next((b for b in self.decode_page_buckets if b >= max_pages), None)
