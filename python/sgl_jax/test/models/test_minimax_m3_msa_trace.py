"""MSA decode v2 sharding trace gate (CPU 4-device, mesh data=2 tensor=2).

Regression test for v1 ShardingTypeError on v6e-64: traces the full
FlashAttention.__call__ MSA shard_map path locally before TPU deploy.
RPA Pallas kernel is stubbed (TPU-only); test asserts shard_map in/out
specs are consistent and the path returns a 3-tuple of correct shape.
"""

import os

# MUST precede any jax import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# Tiny config — divisible by (data=2, tensor=2) on every sharded axis.
BS = 4
PAGE_SIZE = 128
KV_HEADS = 2
Q_HEADS = 4
HEAD_DIM = 128
IDX_HEADS = 2
IDX_DIM = 64
POOL_SIZE = 512
LAYER_NUM = 2
SPARSE_LAYER = 1
PAGES_PER_SEQ = 4
TOPK = 4
SEQ_LENS = np.array([200, 300, 257, 384], dtype=np.int32)


@pytest.fixture(scope="module")
def mesh():
    devs = jax.devices()
    assert len(devs) >= 4, f"need 4 CPU devices, got {len(devs)} (XLA_FLAGS not picked up?)"
    return jax.make_mesh((2, 2), ("data", "tensor"), devices=devs[:4])


@pytest.fixture
def rpa_stub(monkeypatch):
    """Replace Pallas RPA with a shape-preserving jnp stub (CPU can't run TPU kernel)."""
    from sgl_jax.srt.layers.attention import flashattention_backend as fab

    def _stub(q, k, v, kv_cache, *_args, **_kwargs):
        return jnp.zeros_like(q), kv_cache

    monkeypatch.setattr(fab, "ragged_paged_attention_v3", _stub)
    yield


def _shard(mesh, x, spec):
    return jax.device_put(x, NamedSharding(mesh, spec))


def _build_metadata(mesh):
    from sgl_jax.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )

    dp, per_dp_bs = 2, BS // 2
    md = FlashAttentionMetadata()
    md.seq_lens = _shard(mesh, SEQ_LENS, P("data"))
    # cu_q_lens: decode → arange(per_dp_bs+1) tiled per dp; shape dp*(per_dp_bs+1)=6
    cu_q = np.tile(np.arange(per_dp_bs + 1, dtype=np.int32), dp)
    md.cu_q_lens = _shard(mesh, cu_q, P("data"))
    aligned = ((SEQ_LENS + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    cu_kv = np.zeros((dp, per_dp_bs + 1), dtype=np.int32)
    cu_kv[:, 1:] = np.cumsum(aligned.reshape(dp, per_dp_bs), axis=1)
    md.cu_kv_lens = _shard(mesh, cu_kv.ravel(), P("data"))
    # page_indices: bs * pages_per_seq, values within local per-dp page range [0,3)
    pi = np.tile(np.arange(PAGES_PER_SEQ, dtype=np.int32) % 3, BS)
    md.page_indices = _shard(mesh, pi, P("data"))
    md.swa_page_indices = None
    md.distribution = _shard(mesh, np.full(dp * 3, per_dp_bs, dtype=np.int32), P("data"))
    md.custom_mask = None
    return md


def _build_forward_batch(mesh):
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    out_loc = _shard(mesh, np.array([199, 299, 100, 200], dtype=np.int32), P("data"))
    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.DECODE,
        batch_size=BS,
        input_ids=jnp.zeros((BS,), dtype=jnp.int32),
        req_pool_indices=jnp.arange(BS, dtype=jnp.int32),
        seq_lens=_shard(mesh, SEQ_LENS, P("data")),
        out_cache_loc=out_loc,
    )


@pytest.mark.unit
def test_msa_decode_shard_map_traces(mesh, rpa_stub):
    """FlashAttention.__call__ MSA path traces under (data=2, tensor=2) without
    ShardingTypeError; returns (attn_out, kv_upd, ik_upd) with expected shapes."""
    from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.mem_cache.memory_pool import MSATokenToKVPool

    with jax.set_mesh(mesh):
        pool = MSATokenToKVPool(
            sparse_layer_ids=[SPARSE_LAYER],
            index_head_dim=IDX_DIM,
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            dtype=jnp.bfloat16,
            head_num=KV_HEADS,
            head_dim=HEAD_DIM,
            layer_num=LAYER_NUM,
            mesh=mesh,
            dp_size=2,
        )
        backend = FlashAttention(
            num_attn_heads=Q_HEADS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            page_size=PAGE_SIZE,
            mesh=mesh,
        )
        backend.forward_metadata = _build_metadata(mesh)
        layer = RadixAttention(
            num_heads=Q_HEADS,
            head_dim=HEAD_DIM,
            scaling=HEAD_DIM**-0.5,
            num_kv_heads=KV_HEADS,
            layer_id=SPARSE_LAYER,
        )
        fb = _build_forward_batch(mesh)

        rng = np.random.default_rng(0)
        q = _shard(
            mesh, rng.standard_normal((BS, Q_HEADS, HEAD_DIM), np.float32), P("data", "tensor")
        )
        k = _shard(
            mesh, rng.standard_normal((BS, KV_HEADS, HEAD_DIM), np.float32), P("data", "tensor")
        )
        v = _shard(
            mesh, rng.standard_normal((BS, KV_HEADS, HEAD_DIM), np.float32), P("data", "tensor")
        )
        iq = _shard(
            mesh, rng.standard_normal((BS, IDX_HEADS, IDX_DIM), np.float32), P("data", None, None)
        )
        ik = _shard(mesh, rng.standard_normal((BS, 1, IDX_DIM), np.float32), P("data", None, None))

        out = backend(
            q,
            k,
            v,
            layer,
            fb,
            pool,
            index_q=iq,
            index_k=ik,
            msa_topk=TOPK,
            msa_local_blocks=1,
        )
        jax.block_until_ready(out)

    assert isinstance(out, tuple) and len(out) == 3, f"expected 3-tuple, got {type(out)}"
    attn_out, kv_upd, (ik_buf_upd, ikp_upd) = out
    assert attn_out.shape == (BS, Q_HEADS * HEAD_DIM), attn_out.shape
    assert kv_upd.ndim == 5 and kv_upd.shape == pool.kv_buffer[SPARSE_LAYER].shape, kv_upd.shape
    assert ik_buf_upd.shape == pool.index_k_buffer[0].shape, ik_buf_upd.shape
    assert ikp_upd.shape == pool.index_k_pooled[0].shape, ikp_upd.shape


@pytest.mark.unit
def test_msa_prefill_shard_map_traces(mesh, rpa_stub):
    """MSA layer in EXTEND mode (msa_topk=0): index_k write only, no topk."""
    from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.mem_cache.memory_pool import MSATokenToKVPool

    with jax.set_mesh(mesh):
        pool = MSATokenToKVPool(
            sparse_layer_ids=[SPARSE_LAYER],
            index_head_dim=IDX_DIM,
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            dtype=jnp.bfloat16,
            head_num=KV_HEADS,
            head_dim=HEAD_DIM,
            layer_num=LAYER_NUM,
            mesh=mesh,
            dp_size=2,
        )
        backend = FlashAttention(Q_HEADS, KV_HEADS, HEAD_DIM, page_size=PAGE_SIZE, mesh=mesh)
        backend.forward_metadata = _build_metadata(mesh)
        layer = RadixAttention(Q_HEADS, HEAD_DIM, HEAD_DIM**-0.5, KV_HEADS, layer_id=SPARSE_LAYER)
        fb = _build_forward_batch(mesh)

        q = _shard(mesh, np.zeros((BS, Q_HEADS, HEAD_DIM), np.float32), P("data", "tensor"))
        k = _shard(mesh, np.zeros((BS, KV_HEADS, HEAD_DIM), np.float32), P("data", "tensor"))
        v = _shard(mesh, np.zeros((BS, KV_HEADS, HEAD_DIM), np.float32), P("data", "tensor"))
        ik = _shard(mesh, np.zeros((BS, 1, IDX_DIM), np.float32), P("data", None, None))
        iq = _shard(mesh, np.zeros((BS, IDX_HEADS, IDX_DIM), np.float32), P("data", None, None))

        out = backend(q, k, v, layer, fb, pool, index_q=iq, index_k=ik, msa_topk=0)
        jax.block_until_ready(out)

    assert len(out) == 3
    ik_buf_upd, ikp_upd = out[2]
    assert ik_buf_upd.shape == pool.index_k_buffer[0].shape
    assert ikp_upd.shape == pool.index_k_pooled[0].shape


# ---------------------------------------------------------------------------
# Full integration path: model layer → RadixAttention → backend → MemoryPools.
# Catches glue bugs the direct backend(...) calls above bypass (R9: MSAIndexKProxy
# pytree, R9b: RadixAttention 3-tuple unpack).
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 256
INT_IDX_DIM = HEAD_DIM  # production geometry: sparse_index_dim == head_dim


def _tiny_sparse_config():
    from types import SimpleNamespace

    return SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=Q_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-6,
        rotary_dim=HEAD_DIM // 2,
        rope_theta=5_000_000,
        max_position_embeddings=2048,
        sparse_attention_config={
            "sparse_attention_freq": [0, 1],  # layer_id=1 sparse
            "sparse_num_index_heads": IDX_HEADS,
            "sparse_index_dim": INT_IDX_DIM,
            "sparse_topk_blocks": TOPK,
            "sparse_block_size": PAGE_SIZE,
            "sparse_local_block": 1,
        },
    )


@pytest.mark.unit
def test_msa_full_integration_trace(mesh, rpa_stub):
    """E2E: MiniMaxM3Attention → RadixAttention → FlashAttention MSA path
    → MemoryPools pytree + replace_all via MSAIndexKProxy. No direct backend()."""
    from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
    from sgl_jax.srt.mem_cache.memory_pool import MSATokenToKVPool
    from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
        _build_non_hybrid_memory_pools,
    )
    from sgl_jax.srt.models.minimax_m3 import MiniMaxM3Attention

    cfg = _tiny_sparse_config()
    with jax.set_mesh(mesh):
        pool = MSATokenToKVPool(
            sparse_layer_ids=[SPARSE_LAYER],
            index_head_dim=INT_IDX_DIM,
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            dtype=jnp.bfloat16,
            head_num=KV_HEADS,
            head_dim=HEAD_DIM,
            layer_num=LAYER_NUM,
            mesh=mesh,
            dp_size=2,
        )
        backend = FlashAttention(Q_HEADS, KV_HEADS, HEAD_DIM, page_size=PAGE_SIZE, mesh=mesh)
        backend.forward_metadata = _build_metadata(mesh)
        fb = _build_forward_batch(mesh)
        fb.attn_backend = backend  # RadixAttention dispatches via fb.attn_backend

        attn = MiniMaxM3Attention(cfg, mesh=mesh, layer_id=SPARSE_LAYER, dtype=jnp.bfloat16)
        assert attn.is_sparse, "config must select sparse layer"

        positions = _shard(mesh, SEQ_LENS - 1, P("data"))
        rng = np.random.default_rng(1)
        hidden = _shard(
            mesh,
            rng.standard_normal((BS, HIDDEN_SIZE), dtype=np.float32).astype(np.dtype("bfloat16")),
            P("data", None),
        )

        # 1. Model-layer call: _compute_index_qk → self.attn → fb.attn_backend (3-tuple)
        out = attn(positions, hidden, fb, pool)
        jax.block_until_ready(out)
        assert isinstance(out, tuple) and len(out) == 3, f"expected (out, kv, ik), got {type(out)}"
        o, kv_upd, ik_upd = out
        assert o.shape == (BS, HIDDEN_SIZE), o.shape
        assert kv_upd.shape == pool.kv_buffer[SPARSE_LAYER].shape, kv_upd.shape
        assert isinstance(ik_upd, tuple) and len(ik_upd) == 2
        assert ik_upd[0].shape == pool.index_k_buffer[0].shape
        assert ik_upd[1].shape == pool.index_k_pooled[0].shape

        # 2. MemoryPools wrapper: pytree-registered, contains msa_index_k proxy
        mp = _build_non_hybrid_memory_pools(pool)
        assert hasattr(mp, "msa_index_k")
        leaves, treedef = jax.tree.flatten(mp)  # R9 regression: proxy must be a pytree
        # Proxy contributes 0 leaves; all leaves come from MSATokenToKVPool.
        pool_leaves, _ = jax.tree.flatten(pool)
        assert len(leaves) == len(pool_leaves), (len(leaves), len(pool_leaves))
        jax.tree.unflatten(treedef, leaves)  # round-trip

        # 3. replace_all routes msa_index_k → proxy.replace_buffer → pool.index_k_buffer
        mp.replace_all(
            {
                "token_to_kv_pool": [pool.kv_buffer[0], kv_upd],
                "msa_index_k": [ik_upd],
            }
        )
        assert pool.index_k_buffer[0] is ik_upd[0]
        assert pool.index_k_pooled[0] is ik_upd[1]
