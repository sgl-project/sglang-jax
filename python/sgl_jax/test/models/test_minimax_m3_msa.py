"""MiniMax-M3 / MSA tests on a CPU (data=2, tensor=2) mesh.

Covers the swigluoai activation numerics, full-model construction, MSA backend
config ownership, actual-backend per-head page selection against an independent
NumPy oracle (ragged page counts), EXTEND index_k writes, and the
model -> RadixAttention -> backend -> MemoryPools integration path.

RPA (a TPU Pallas kernel) is stubbed; the stub smuggles each tensor rank's page
table out through the attention output so selection can be checked per rank.
The stub does not validate TPU attention output.
"""

import os

# MUST precede any jax import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

BS, DP, TP = 4, 2, 2
PER_DP_BS = BS // DP
PAGE_SIZE = 128
KV_HEADS = 2
Q_HEADS = 4
HEAD_DIM = 128
IDX_HEADS = 2
IDX_DIM = HEAD_DIM  # model asserts sparse_index_dim == head_dim (index RoPE reuses rotary_emb)
LAYER_NUM = 2
SPARSE_LAYER = 1
PAGES_PER_SEQ = 8  # per-request page bucket (> TOPK so decode takes the sparse branch)
TOPK = 4
CONTEXT_LEN = PAGES_PER_SEQ * PAGE_SIZE
POOL_SIZE = PAGE_SIZE * 40  # 42 pages -> 21 per data shard >= 1 + PER_DP_BS * PAGES_PER_SEQ
SPARSE_CONFIG = {
    "use_sparse_attention": True,
    "sparse_block_size": PAGE_SIZE,
    "sparse_topk_blocks": TOPK,
    "sparse_num_index_heads": IDX_HEADS,
    "sparse_index_dim": IDX_DIM,
    "sparse_local_block": 1,
    "sparse_attention_freq": [0, 1],  # layer 1 is sparse
}


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mesh():
    devs = jax.devices()
    assert len(devs) >= 4, f"need 4 CPU devices, got {len(devs)} (XLA_FLAGS not picked up?)"
    return jax.make_mesh(
        (DP, TP),
        ("data", "tensor"),
        devices=devs[:4],
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


@pytest.fixture
def rpa_stub(monkeypatch):
    """Replace the Pallas RPA kernel with a jnp stub. Records the traced
    page_indices length and writes this rank's page_indices into row 0 / local
    head 0 of the output so tests can read the per-rank page table."""
    from sgl_jax.srt.layers.attention import flashattention_backend as fab

    captured: dict = {}

    def _stub(q, k, v, kv_cache, kv_lens, page_indices, *_args, **_kwargs):
        n = page_indices.shape[0]
        captured["page_indices_len"] = n
        out = jnp.zeros_like(q).at[0, 0, :n].set(page_indices.astype(q.dtype))
        return out, kv_cache

    monkeypatch.setattr(fab, "ragged_paged_attention_v3", _stub)
    return captured


@pytest.fixture
def pool(mesh):
    from sgl_jax.srt.mem_cache.memory_pool import MSATokenToKVPool

    with jax.set_mesh(mesh):
        return MSATokenToKVPool(
            sparse_layer_ids=[SPARSE_LAYER],
            index_head_dim=IDX_DIM,
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            dtype=jnp.bfloat16,
            head_num=KV_HEADS,
            head_dim=HEAD_DIM,
            layer_num=LAYER_NUM,
            mesh=mesh,
            dp_size=DP,
        )


@pytest.fixture
def backend(mesh):
    from sgl_jax.srt.layers.attention.msa_backend import MSAAttentionBackend

    return MSAAttentionBackend(
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        page_size=PAGE_SIZE,
        mesh=mesh,
        sparse_config=SPARSE_CONFIG,
        context_len=CONTEXT_LEN,
        total_num_kv_heads=KV_HEADS,
    )


def _shard(mesh, x, spec):
    return jax.device_put(x, NamedSharding(mesh, spec))


def _phys(b_local: int, j: int) -> int:
    """Physical (per-data-shard) page of local request b, logical block j; avoids page 0."""
    return 1 + b_local * PAGES_PER_SEQ + j


def _n_pages(seq_len: int) -> int:
    return -(-int(seq_len) // PAGE_SIZE)


def _metadata(mesh, seq_lens, extend_lens=None):
    """FlashAttentionMetadata with the serving layout: per-DP sections, page_indices
    cumsum-packed per request (ragged), padded to PER_DP_BS * PAGES_PER_SEQ."""
    from sgl_jax.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )

    seq_lens = np.asarray(seq_lens, dtype=np.int32)
    md = FlashAttentionMetadata()
    md.seq_lens = _shard(mesh, seq_lens, P("data"))
    cu_q = np.zeros((DP, PER_DP_BS + 1), dtype=np.int32)
    cu_kv = np.zeros((DP, PER_DP_BS + 1), dtype=np.int32)
    pi = np.zeros((DP, PER_DP_BS * PAGES_PER_SEQ), dtype=np.int32)
    for d in range(DP):
        off = 0
        for b in range(PER_DP_BS):
            g = d * PER_DP_BS + b
            n = _n_pages(seq_lens[g])
            pi[d, off : off + n] = [_phys(b, j) for j in range(n)]
            off += n
            cu_kv[d, b + 1] = cu_kv[d, b] + n * PAGE_SIZE
            q_len = 1 if extend_lens is None else int(extend_lens[g])
            cu_q[d, b + 1] = cu_q[d, b] + q_len
    md.cu_q_lens = _shard(mesh, cu_q.ravel(), P("data"))
    md.cu_kv_lens = _shard(mesh, cu_kv.ravel(), P("data"))
    md.page_indices = _shard(mesh, pi.ravel(), P("data"))
    md.swa_page_indices = None
    if extend_lens is None:
        dist = np.full(DP * 3, PER_DP_BS, dtype=np.int32)
    else:
        dist = np.tile(np.array([0, PER_DP_BS, PER_DP_BS], dtype=np.int32), DP)
    md.distribution = _shard(mesh, dist, P("data"))
    md.custom_mask = None
    return md


def _forward_batch(mesh, mode, seq_lens, out_cache_loc):
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

    n_tokens = len(out_cache_loc)
    return ForwardBatch(
        bid=0,
        forward_mode=mode,
        batch_size=BS,
        input_ids=jnp.zeros((n_tokens,), dtype=jnp.int32),
        req_pool_indices=jnp.arange(BS, dtype=jnp.int32),
        seq_lens=_shard(mesh, np.asarray(seq_lens, dtype=np.int32), P("data")),
        out_cache_loc=_shard(mesh, np.asarray(out_cache_loc, dtype=np.int32), P("data")),
    )


def _qkv(mesh, n_tokens):
    q = _shard(mesh, np.zeros((n_tokens, Q_HEADS, HEAD_DIM), np.float32), P("data", "tensor"))
    k = _shard(mesh, np.zeros((n_tokens, KV_HEADS, HEAD_DIM), np.float32), P("data", "tensor"))
    v = _shard(mesh, np.zeros((n_tokens, KV_HEADS, HEAD_DIM), np.float32), P("data", "tensor"))
    return q, k, v


def _rank_pages(attn_out: np.ndarray, row: int, t: int, n: int) -> np.ndarray:
    """Page table smuggled out by rpa_stub for tensor rank t (its local head 0)."""
    col = t * (Q_HEADS // TP) * HEAD_DIM
    return attn_out[row, col : col + n].astype(np.int64)


def _radix_layer():
    from sgl_jax.srt.layers.radix_attention import RadixAttention

    return RadixAttention(Q_HEADS, HEAD_DIM, HEAD_DIM**-0.5, KV_HEADS, layer_id=SPARSE_LAYER)


# ---------------------------------------------------------------------------
# activation + construction
# ---------------------------------------------------------------------------
def _ref_swigluoai(gate, up, alpha, limit):
    gate = np.clip(gate, a_max=limit, a_min=None)
    up = np.clip(up, -limit, limit)
    return (up + 1.0) * gate * (1.0 / (1.0 + np.exp(-gate * alpha)))


@pytest.mark.unit
def test_swigluoai_matches_ref():
    from sgl_jax.srt.models.minimax_m3 import swigluoai

    rng = np.random.default_rng(0)
    gate = rng.standard_normal((7, 16)).astype(np.float32) * 10  # exercise clamp
    up = rng.standard_normal((7, 16)).astype(np.float32) * 10
    out = np.asarray(swigluoai(jnp.asarray(gate), jnp.asarray(up), alpha=1.702, limit=7.0))
    np.testing.assert_allclose(out, _ref_swigluoai(gate, up, 1.702, 7.0), rtol=1e-5, atol=1e-5)


def _tiny_text_config(**overrides):
    cfg = SimpleNamespace(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        rotary_dim=16,
        rope_theta=5_000_000,
        max_position_embeddings=2048,
        moe_layer_freq=[0, 0, 1, 1],
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        intermediate_size=64,
        dense_intermediate_size=96,
        shared_intermediate_size=64,
        routed_scaling_factor=2.0,
        scoring_func="sigmoid",
        use_routing_bias=True,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        ep_size=1,
        vocab_size=256,
        num_hidden_layers=4,
        sparse_attention_config={
            **SPARSE_CONFIG,
            "sparse_index_dim": 32,
            "sparse_attention_freq": [0, 1, 1, 1],
        },
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.unit
def test_causal_lm_construct_eval_shape(mesh):
    """Construct the full model (embed -> layers -> lm_head -> logits_processor)
    under eval_shape. Guards shared-layer constructor drift on main (e.g. #1670
    ParallelLMHead kwargs, TopK(mesh=...) for the Pallas top-k) and the
    dense / MoE / sparse dispatch of the decoder layers."""
    from sgl_jax.srt.models.minimax_m3 import MiniMaxM3SparseForCausalLM

    cfg = _tiny_text_config()
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(
            lambda: MiniMaxM3SparseForCausalLM(cfg, mesh=mesh, dtype=jnp.bfloat16)
        )
    assert model.lm_head.embedding.shape == (cfg.vocab_size, cfg.hidden_size)
    assert len(model.model.layers) == cfg.num_hidden_layers
    assert [layer.self_attn.is_sparse for layer in model.model.layers] == [False, True, True, True]
    dense, moe = model.model.layers[0], model.model.layers[2]
    assert dense.is_moe is False and hasattr(dense, "mlp") and not hasattr(dense, "moe_gate")
    assert dense.mlp.gate_proj.weight.shape[-1] == cfg.dense_intermediate_size
    assert moe.is_moe is True
    assert moe.topk.mesh is mesh  # bare pallas_call under an Explicit mesh otherwise
    assert moe.block_sparse_moe.activation == "swigluoai"
    # EPLB adapter: expert fields come from the model hook, not a runner backfill.
    fields = MiniMaxM3SparseForCausalLM.get_expert_location_config(SimpleNamespace(text_config=cfg))
    assert fields == {"num_experts": 4, "num_layers": 4, "num_groups": 1}


# ---------------------------------------------------------------------------
# MSA backend: config ownership
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_msa_backend_owns_config(mesh, backend):
    from sgl_jax.srt.layers.attention.msa_backend import MSAAttentionBackend
    from sgl_jax.srt.mem_cache.memory_pool import MSATokenToKVPool

    assert backend.sparse_layer_ids == [SPARSE_LAYER]
    assert backend.token_to_kv_pool_class is MSATokenToKVPool
    assert backend.token_to_kv_pool_kwargs == {
        "sparse_layer_ids": [SPARSE_LAYER],
        "index_head_dim": IDX_DIM,
    }
    assert backend.extra_kv_bytes_per_token(2) == 1 * 128 * 2  # dim padded to 128
    # [topk, 4*topk, max_pages] capped at max_pages and deduped
    assert backend.decode_page_buckets == [TOPK, PAGES_PER_SEQ]
    assert backend._decode_page_limit(SimpleNamespace(seq_lens=np.array([1, 300]))) == TOPK
    assert backend._decode_page_limit(SimpleNamespace(seq_lens=np.array([1000]))) == PAGES_PER_SEQ

    common = dict(page_size=PAGE_SIZE, mesh=mesh, context_len=CONTEXT_LEN)
    with pytest.raises(ValueError, match="page-size"):
        MSAAttentionBackend(
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            sparse_config=SPARSE_CONFIG,
            total_num_kv_heads=KV_HEADS,
            **{**common, "page_size": 64},
        )
    with pytest.raises(ValueError, match="one KV/GQA group per tensor rank"):
        MSAAttentionBackend(
            Q_HEADS,
            3,
            HEAD_DIM,
            total_num_kv_heads=3,
            sparse_config={**SPARSE_CONFIG, "sparse_num_index_heads": 3},
            **common,
        )


# ---------------------------------------------------------------------------
# decode: per-head block selection through the real backend vs a NumPy oracle
# ---------------------------------------------------------------------------
DECODE_SEQ_LENS = np.array([300, 1000, 640, 900], dtype=np.int32)  # 3 / 8 / 5 / 8 pages


def _oracle_selection(scores_by_block: dict, n_blocks: int) -> list:
    """Per-head MSA contract for one request: local (last) block forced, then the
    highest-scoring remaining blocks up to TOPK, reported in ascending block order."""
    q_block = n_blocks - 1
    others = sorted((j for j in range(n_blocks) if j != q_block), key=lambda j: -scores_by_block[j])
    return sorted([q_block] + others[: TOPK - 1])


@pytest.mark.unit
def test_decode_per_head_selection_matches_oracle(mesh, pool, backend, rpa_stub):
    """Ragged decode batch (3/8/5/8 pages) through FlashAttention's MSA path:
    every tensor rank must build its page table from its own index head, and the
    selected pages must equal an independent NumPy oracle (distinct scores, no ties).
    Fails under a head-collapsed (max over heads) selection."""
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

    rng = np.random.default_rng(3)
    n_local = pool.index_k_buffer[0].shape[0] // DP
    # index_k of physical page p (same on both data shards) = unit vector e_{p mod IDX_DIM}
    ik_np = np.zeros((DP * n_local, PAGE_SIZE, 1, IDX_DIM), np.float32)
    for d in range(DP):
        for p in range(n_local):
            ik_np[d * n_local + p, :, 0, p % IDX_DIM] = 1.0
    # index_q: head h of request g scores block j as iq[g, h, dim(phys(b, j))]; distinct values
    iq_np = np.zeros((BS, IDX_HEADS, IDX_DIM), np.float32)
    scores = {}
    for g in range(BS):
        b = g % PER_DP_BS
        n = _n_pages(DECODE_SEQ_LENS[g])
        for h in range(IDX_HEADS):
            vals = rng.permutation(64)[:n].astype(np.float32) / 8.0 - 4.0  # distinct
            scores[g, h] = {j: float(vals[j]) for j in range(n)}
            for j in range(n):
                iq_np[g, h, _phys(b, j) % IDX_DIM] = vals[j]
    out_loc = [
        _phys(g % PER_DP_BS, _n_pages(L) - 1) * PAGE_SIZE + (int(L) - 1) % PAGE_SIZE
        for g, L in enumerate(DECODE_SEQ_LENS)
    ]

    with jax.set_mesh(mesh):
        pool.index_k_buffer[0] = _shard(
            mesh, ik_np.astype(np.dtype("bfloat16")), P("data", None, None, None)
        )
        backend.forward_metadata = _metadata(mesh, DECODE_SEQ_LENS)
        fb = _forward_batch(mesh, ForwardMode.DECODE, DECODE_SEQ_LENS, out_loc)
        q, k, v = _qkv(mesh, BS)
        ik = _shard(mesh, np.zeros((BS, 1, IDX_DIM), np.float32), P("data", None, None))
        iq = _shard(mesh, iq_np, P("data", None, None))
        out = backend(
            q,
            k,
            v,
            _radix_layer(),
            fb,
            pool,
            index_q=iq,
            index_k=ik,
            msa_topk=TOPK,
            msa_local_blocks=1,
        )
        attn_out, kv_upd, ik_upd = jax.block_until_ready(out)

    assert attn_out.shape == (BS, Q_HEADS * HEAD_DIM)
    assert kv_upd.shape == pool.kv_buffer[SPARSE_LAYER].shape
    assert ik_upd.shape == pool.index_k_buffer[0].shape
    # sparse branch rewrites page_indices to PER_DP_BS * TOPK (the dense path keeps PAGES_PER_SEQ)
    assert rpa_stub["page_indices_len"] == PER_DP_BS * TOPK

    attn_np = np.asarray(attn_out)
    differs = 0
    for d in range(DP):
        for t in range(TP):  # tensor rank t owns KV head t == index head t
            pages = _rank_pages(attn_np, d * PER_DP_BS, t, PER_DP_BS * TOPK).reshape(
                PER_DP_BS, TOPK
            )
            for b in range(PER_DP_BS):
                g = d * PER_DP_BS + b
                n = _n_pages(DECODE_SEQ_LENS[g])
                n_valid = min(n, TOPK)
                want = [_phys(b, j) for j in _oracle_selection(scores[g, t], n)]
                assert (
                    pages[b, :n_valid].tolist() == want
                ), f"dp{d} rank{t} req{b}: {pages[b]} != {want}"
                assert (pages[b, n_valid:] == 0).all(), "invalid slots must map to the pad page"
            other = _rank_pages(attn_np, d * PER_DP_BS, 1 - t, PER_DP_BS * TOPK)
            differs += int(not np.array_equal(pages.ravel(), other))
    assert differs > 0, "both heads selected identical pages; per-head selection not exercised"


# ---------------------------------------------------------------------------
# extend: dense attention + per-token index_k writes
# ---------------------------------------------------------------------------
EXTEND_LENS = np.array([5, 3, 4, 4], dtype=np.int32)  # 8 tokens per DP rank


@pytest.mark.unit
def test_extend_writes_index_k(mesh, pool, backend, rpa_stub):
    """EXTEND through the MSA layer path: no top-k (dense page table), and every
    token's index_k lands in its own page/slot of the index_k cache."""
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

    rng = np.random.default_rng(5)
    n_tokens = int(EXTEND_LENS.sum())
    out_loc, owner = [], []
    for g, L in enumerate(EXTEND_LENS):
        for t in range(int(L)):
            out_loc.append(_phys(g % PER_DP_BS, 0) * PAGE_SIZE + t)
            owner.append((g // PER_DP_BS, g % PER_DP_BS, t))
    ik_np = rng.standard_normal((n_tokens, 1, IDX_DIM)).astype(np.float32)

    with jax.set_mesh(mesh):
        backend.forward_metadata = _metadata(mesh, EXTEND_LENS, extend_lens=EXTEND_LENS)
        fb = _forward_batch(mesh, ForwardMode.EXTEND, EXTEND_LENS, out_loc)
        q, k, v = _qkv(mesh, n_tokens)
        ik = _shard(mesh, ik_np, P("data", None, None))
        iq = _shard(
            mesh, np.zeros((n_tokens, IDX_HEADS, IDX_DIM), np.float32), P("data", None, None)
        )
        out = backend(
            q,
            k,
            v,
            _radix_layer(),
            fb,
            pool,
            index_q=iq,
            index_k=ik,
            msa_topk=TOPK,
            msa_local_blocks=1,
        )
        attn_out, kv_upd, ik_upd = jax.block_until_ready(out)

    assert attn_out.shape == (n_tokens, Q_HEADS * HEAD_DIM)
    assert kv_upd.shape == pool.kv_buffer[SPARSE_LAYER].shape
    assert rpa_stub["page_indices_len"] == PER_DP_BS * PAGES_PER_SEQ  # dense: no top-k in EXTEND
    n_local = ik_upd.shape[0] // DP
    got = np.asarray(ik_upd).astype(np.float32)
    want = np.asarray(jnp.asarray(ik_np).astype(jnp.bfloat16)).astype(np.float32)
    for tok, (d, b, t) in enumerate(owner):
        np.testing.assert_array_equal(got[d * n_local + _phys(b, 0), t, 0], want[tok, 0])


# ---------------------------------------------------------------------------
# integration: model layer -> RadixAttention -> backend -> MemoryPools
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 256


def _tiny_attention_config():
    return SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=Q_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-6,
        rotary_dim=HEAD_DIM // 2,
        rope_theta=5_000_000,
        max_position_embeddings=2048,
        sparse_attention_config=SPARSE_CONFIG,
    )


@pytest.mark.unit
def test_model_layer_to_cache_integration(mesh, pool, backend, rpa_stub):
    """MiniMaxM3Attention -> RadixAttention -> MSA backend -> MemoryPools pytree +
    replace_all via MSAIndexKProxy, on the ragged decode batch. Catches glue bugs the
    direct backend(...) calls bypass (3-tuple passthrough, proxy pytree)."""
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
    from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
        _build_non_hybrid_memory_pools,
    )
    from sgl_jax.srt.models.minimax_m3 import MiniMaxM3Attention

    out_loc = [
        _phys(g % PER_DP_BS, _n_pages(L) - 1) * PAGE_SIZE + (int(L) - 1) % PAGE_SIZE
        for g, L in enumerate(DECODE_SEQ_LENS)
    ]
    with jax.set_mesh(mesh):
        backend.forward_metadata = _metadata(mesh, DECODE_SEQ_LENS)
        fb = _forward_batch(mesh, ForwardMode.DECODE, DECODE_SEQ_LENS, out_loc)
        fb.attn_backend = backend  # RadixAttention dispatches via fb.attn_backend
        attn = MiniMaxM3Attention(
            _tiny_attention_config(), mesh=mesh, layer_id=SPARSE_LAYER, dtype=jnp.bfloat16
        )
        assert attn.is_sparse
        positions = _shard(mesh, DECODE_SEQ_LENS - 1, P("data"))
        rng = np.random.default_rng(1)
        hidden = _shard(
            mesh,
            rng.standard_normal((BS, HIDDEN_SIZE), dtype=np.float32).astype(np.dtype("bfloat16")),
            P("data", None),
        )
        out = jax.block_until_ready(attn(positions, hidden, fb, pool))

    assert isinstance(out, tuple) and len(out) == 3, f"expected (out, kv, ik), got {type(out)}"
    o, kv_upd, ik_upd = out
    assert o.shape == (BS, HIDDEN_SIZE)
    assert kv_upd.shape == pool.kv_buffer[SPARSE_LAYER].shape
    assert ik_upd.shape == pool.index_k_buffer[0].shape
    assert rpa_stub["page_indices_len"] == PER_DP_BS * TOPK

    mp = _build_non_hybrid_memory_pools(pool)
    assert hasattr(mp, "msa_index_k")
    leaves, treedef = jax.tree.flatten(mp)  # proxy must be a pytree with 0 own leaves
    assert len(leaves) == len(jax.tree.flatten(pool)[0])
    jax.tree.unflatten(treedef, leaves)
    mp.replace_all({"token_to_kv_pool": [pool.kv_buffer[0], kv_upd], "msa_index_k": [ik_upd]})
    assert pool.index_k_buffer[0] is ik_upd
