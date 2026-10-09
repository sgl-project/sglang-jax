"""CSA correctness, resource ownership and input contracts."""

import importlib
from test.srt.kernels.csa_attention.ref import reference, update_window
from test.srt.kernels.csa_compressor import ref as compressor_ref
from test.srt.kernels.csa_compressor.test_compressor import Case
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.kernels.csa import CSAMetadata, csa_step
from sgl_jax.srt.kernels.csa.tune import CSAIndexerSchedule, get_indexer_schedule
from sgl_jax.srt.kernels.csa_attention import CSAAttentionMetadata
from sgl_jax.srt.kernels.csa_attention.tune import get_csa_attention_schedule
from sgl_jax.srt.kernels.csa_compressor.compressor import CompressorMetadata
from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk

from .ref import select_records

compiled_step = jax.jit(
    csa_step,
    static_argnames=(
        "compressor_schedule",
        "attention_schedule",
        "indexer_schedule",
        "softmax_scale",
        "page_size",
    ),
)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Existing StreamIndex requires TPU")
@pytest.mark.parametrize("page_size,hidden", [(128, 128), (256, 4096)])
def test_csa_chunk_continuation(page_size, hidden):
    case = Case(batch=2, hidden=hidden, page_size=page_size, capacity=512)
    _run_steps(case, ((3, 1), (130, 31), (1, 1), (5, 0), (1, 1)), heads=64)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Existing StreamIndex requires TPU")
def test_csa_long_context_selection():
    case = Case(batch=1, capacity=8448)
    case.run((8191,))
    _run_steps(case, ((1,), (5,), (1,)))


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Existing StreamIndex requires TPU")
@pytest.mark.parametrize("page_size", [128, 256])
def test_csa_mixed_batch(page_size):
    case = Case(batch=3, page_size=page_size, capacity=2304)
    case.run((2047, 2051, 2055))
    _run_steps(
        case,
        ((1, 7, 3), (1, 1, 1)),
        distributions=((1, 1, 3), (3, 3, 3)),
    )


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Existing StreamIndex requires TPU")
def test_csa_pool_update_contract():
    case = Case(batch=2, capacity=256)
    _run_steps(case, ((3, 1), (1, 3)), check_resources=True)


def _run_steps(case, chunks, heads=8, *, distributions=None, check_resources=False):
    rng = np.random.default_rng(83)
    page_size, batch = case.page_size, case.batch
    per_request = case.capacity // page_size
    pages = np.arange(1, 1 + batch * per_request, dtype=np.int32).reshape(batch, -1)[:, ::-1]
    index_pages = np.concatenate((pages, np.zeros((1, per_request), np.int32)))
    # Independent SWA allocation; the compressor fixture allocates compressed/state pools.
    swa_shape = (
        case.pool.get_swa_buffer(1).shape
        if check_resources
        else ((batch * per_request + 1) * page_size, 512)
    )
    swa_host = rng.normal(0, 0.2, swa_shape).astype(ml_dtypes.bfloat16)
    swa = jnp.asarray(swa_host)
    for step, lengths in enumerate(chunks):
        x, compressor_md = case.inputs(lengths)
        cu = np.asarray(compressor_md.cu_q_lens)
        ends = np.r_[case.prefix + lengths, 1].astype(np.int32)
        req = np.r_[np.repeat(np.arange(batch), lengths), -1].astype(np.int32)
        first = np.maximum(0, case.prefix - 127) // page_size
        wp = np.concatenate([pages[r, start:] for r, start in enumerate(first)])
        wc = (
            np.r_[0, np.cumsum(per_request - first), np.sum(per_request - first)].astype(np.int32)
            * page_size
        )
        locations = np.full(len(x), -1, np.int32)
        for r, n in enumerate(lengths):
            positions = case.prefix[r] + np.arange(n)
            locations[cu[r] : cu[r + 1]] = (
                pages[r, positions // page_size] * page_size + positions % page_size
            )
        attention_md = CSAAttentionMetadata(
            req,
            cu,
            ends,
            wp,
            wc,
            index_pages.reshape(-1),
            np.arange(batch + 2, dtype=np.int32) * per_request * (page_size // 4),
            ends // 4,
            locations,
        )
        decode = all(n == 1 for n in lengths)
        schedule = get_csa_attention_schedule(jax.devices()[0].device_kind, decode=decode)
        distribution = [batch, batch, batch] if decode else [0, 0, batch]
        if distributions is not None:
            distribution = distributions[step]
        metadata = CSAMetadata(
            compressor_md, attention_md, index_pages, np.asarray(distribution, np.int32)
        )
        q = rng.normal(0, 0.2, (len(x), heads, 512)).astype(ml_dtypes.bfloat16)
        new = rng.normal(0, 0.2, (len(x), 512)).astype(ml_dtypes.bfloat16)
        iq = rng.normal(0, 0.2, (len(x), heads, 128)).astype(ml_dtypes.bfloat16)
        weights = rng.uniform(0.1, 1, (len(x), heads)).astype(np.float32)
        sink = np.zeros(heads, np.float32)
        expected = compressor_ref.compressor(
            x,
            case.weight,
            case.apes,
            case.norms,
            case.cos,
            case.sin,
            case.expected[:2],
            case.expected[2:],
            np.asarray(compressor_md.positions),
            cu,
            case.slots,
            np.asarray(compressor_md.cache_locations),
        )
        want_swa = update_window(new, swa_host, attention_md, window_size=128, page_size=page_size)
        args = (
            x,
            q,
            new,
            iq,
            weights,
            case.buffers[0],
            case.buffers[1],
            swa,
            case.buffers[2],
            case.buffers[3],
            case.weight,
            *case.apes,
            *case.norms,
            case.cos,
            case.sin,
            sink,
            metadata,
        )
        output, updates = jax.block_until_ready(
            compiled_step(
                *jax.tree.map(jnp.asarray, args),
                compressor_schedule=case.schedule,
                attention_schedule=schedule,
                indexer_schedule=get_indexer_schedule(page_size, schedule),
                softmax_scale=512**-0.5,
                page_size=page_size,
            )
        )
        # Validate the reused indexer's choices independently before conditioning attention on them.
        indexer = updates["indexer"]
        index_schedule = get_indexer_schedule(page_size, schedule)
        candidates = np.asarray(
            streamindex_topk(
                jnp.asarray(iq),
                jnp.asarray(weights),
                indexer.reshape(indexer.shape[0], indexer.shape[1] // 2, 2, 128),
                jnp.asarray(ends),
                jnp.asarray(index_pages.reshape(-1)),
                jnp.asarray(cu),
                jnp.asarray(distribution, jnp.int32),
                k=512,
                compression_ratio=4,
                num_kv_pages_per_block=index_schedule.kv_pages_per_block,
                num_queries_per_block=index_schedule.query_tile,
                decode_req_batch_size=index_schedule.decode_request_tile,
                topk_backend="xla",
            )
        )
        picks = select_records(
            iq,
            weights,
            expected[3],
            index_pages,
            cu,
            ends,
            512,
            candidates=candidates,
        )
        want = reference(
            q,
            new,
            swa_host,
            expected[2],
            picks,
            sink,
            attention_md,
            scale=512**-0.5,
            window_page_size=page_size,
        )
        actual = np.asarray(output).astype(np.float32)
        assert np.isfinite(actual).all() and np.isfinite(want).all()
        np.testing.assert_allclose(actual, want, rtol=2e-2, atol=1e-2)
        got = tuple(updates[k] for k in ("state", "indexer_state", "compressed", "indexer"))
        for value, target in zip(got, expected, strict=True):
            value, target = np.asarray(value, np.float32), np.asarray(target, np.float32)
            np.testing.assert_array_equal(np.isneginf(value), np.isneginf(target))
            assert not np.isnan(value).any() and not np.isposinf(value).any()
            finite = np.isfinite(target)
            np.testing.assert_allclose(value[finite], target[finite], rtol=2e-2, atol=1e-2)
        np.testing.assert_array_equal(np.asarray(updates["swa"]), want_swa)
        if check_resources:
            kv_updates = {key: updates[key] for key in ("swa", "compressed", "indexer")}
            state_updates = {"compressor": updates["state"], "indexer": updates["indexer_state"]}
            for pool, payload, expected_updates in (
                (
                    case.pool,
                    kv_updates,
                    {
                        "swa": updates["swa"],
                        "c4": updates["compressed"],
                        "indexer": updates["indexer"],
                    },
                ),
                (
                    case.state,
                    state_updates,
                    {"c4": updates["state"], "indexer": updates["indexer_state"]},
                ),
            ):
                replacement = pool.build_buffer_updates({1: payload})
                for family, old_arrays in pool.buffers.items():
                    for layer, old in enumerate(old_arrays):
                        expected_array = (
                            expected_updates[family]
                            if family in expected_updates
                            and layer == pool.layer_to_buffer[family][1]
                            else old
                        )
                        assert replacement[family][layer] is expected_array
                        assert pool.buffers[family][layer] is old
        case.buffers, case.expected = got, expected
        case.prefix += lengths
        swa, swa_host = updates["swa"], want_swa


@pytest.mark.parametrize("tokens,distribution", [(1, [1, 1, 1]), (5, [0, 0, 1])])
def test_existing_indexer_shape_contract(tokens, distribution):
    # Trace the real indexer; this is not a TPU execution or numerical test.
    result = jax.eval_shape(
        lambda *args: streamindex_topk(
            *args,
            k=512,
            compression_ratio=4,
            num_kv_pages_per_block=4,
            num_queries_per_block=1,
            decode_req_batch_size=1,
            topk_backend="xla",
        ),
        jnp.zeros((tokens, 8, 128), jnp.bfloat16),
        jnp.ones((tokens, 8), jnp.bfloat16),
        jnp.zeros((5, 16, 2, 128), jnp.bfloat16),
        jnp.array([128], jnp.int32),
        jnp.array([1, 2, 3, 4], jnp.int32),
        jnp.array([0, tokens], jnp.int32),
        jnp.array(distribution, jnp.int32),
    )
    assert result.shape == (tokens, 512) and result.dtype == jnp.int32


@pytest.mark.parametrize("tokens", [0, 3])
def test_step_update_ownership(monkeypatch, tokens):
    module = importlib.import_module("sgl_jax.srt.kernels.csa.csa")
    calls = []
    state, istate = object(), object()
    compressed = jnp.ones((2, 32, 512), jnp.bfloat16)
    indexer = jnp.ones((2, 32, 128), jnp.bfloat16)
    old_window, updated_window = object(), object()
    q = jnp.zeros((tokens, 8, 512), jnp.bfloat16)
    md = SimpleNamespace(
        seq_lens=jnp.array([4], jnp.int32),
        cu_q_lens=jnp.array([0, min(tokens, 2)], jnp.int32),
        query_seq_ids=jnp.array([0, 0, -1][:tokens], jnp.int32),
    )
    cm = CompressorMetadata(
        jnp.zeros(tokens, jnp.int32),
        md.cu_q_lens,
        jnp.zeros(1, jnp.int32),
        jnp.full(tokens, -1, jnp.int32),
        (),
    )
    metadata = CSAMetadata(cm, md, jnp.array([[1]], jnp.int32), jnp.array([0, 0, 1], jnp.int32))

    def compressor(*args, **kwargs):
        calls.append("compressor")
        assert args[-1] is metadata.compressor
        return state, istate, compressed, indexer

    def topk(*args, **kwargs):
        calls.append("indexer")
        assert args[2].shape == (2, 16, 2, 128)
        np.testing.assert_array_equal(args[2].reshape(indexer.shape), indexer)
        assert args[3] is md.seq_lens
        assert kwargs["compression_ratio"] == 4
        return jnp.zeros((tokens, 512), jnp.int32)

    def attention(*args, **kwargs):
        calls.append("attention")
        assert args[2] is old_window and args[3] is compressed
        np.testing.assert_array_equal(args[4][-1], -np.ones(512, np.int32))
        return q, updated_window

    monkeypatch.setattr(module, "csa_compressor", compressor)
    monkeypatch.setattr(module, "streamindex_topk", topk)
    monkeypatch.setattr(module, "csa_joint_attention", attention)
    output, updates = csa_step(
        jnp.zeros((tokens, 128)),
        q,
        None,
        jnp.zeros((tokens, 4, 128), jnp.bfloat16),
        jnp.zeros((tokens, 4)),
        None,
        None,
        old_window,
        jnp.zeros_like(compressed),
        jnp.zeros_like(indexer),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        metadata,
        compressor_schedule=None,
        attention_schedule=None,
        indexer_schedule=CSAIndexerSchedule(4, 1),
        softmax_scale=512**-0.5,
    )
    assert calls == (["compressor", "indexer", "attention"] if tokens else ["compressor"])
    assert output.shape == q.shape
    assert set(updates) == {"state", "indexer_state", "compressed", "indexer", "swa"}
    assert updates["state"] is state and updates["indexer_state"] is istate
    assert updates["compressed"] is compressed and updates["indexer"] is indexer
    assert updates["swa"] is (updated_window if tokens else old_window)


@pytest.mark.parametrize("field", ["cu_q_lens", "state_indices", "positions", "cache_locations"])
def test_mismatched_metadata(field):
    cm = CompressorMetadata(
        jnp.zeros(3, jnp.int32),
        jnp.array([0, 3], jnp.int32),
        jnp.zeros(1, jnp.int32),
        jnp.zeros(3, jnp.int32),
        (),
    )
    md = SimpleNamespace(
        seq_lens=jnp.array([3], jnp.int32),
        cu_q_lens=cm.cu_q_lens,
        query_seq_ids=jnp.zeros(3, jnp.int32),
    )
    cm = cm._replace(**{field: getattr(cm, field)[:-1]})
    metadata = CSAMetadata(cm, md, jnp.ones((1, 1), jnp.int32), jnp.array([0, 0, 1], jnp.int32))
    with pytest.raises(ValueError, match="metadata must share"):
        csa_step(
            jnp.zeros((3, 128), jnp.bfloat16),
            jnp.zeros((3, 8, 512), jnp.bfloat16),
            *([None] * 16),
            metadata,
            compressor_schedule=None,
            attention_schedule=None,
            indexer_schedule=None,
            softmax_scale=512**-0.5,
        )


@pytest.mark.parametrize("indices,valid", [([0, 2], True), ([0, 3], False), ([0, 0], False)])
def test_reference_accepts_only_valid_topk_ties(indices, valid):
    cache = np.zeros((2, 4, 128), np.float32)
    cache[1, :, 0] = [3, 2, 2, 1]
    args = (
        np.ones((1, 1, 128), np.float32),
        np.ones((1, 1), np.float32),
        cache,
        np.array([[1]]),
        np.array([0, 1]),
        np.array([16]),
        2,
    )
    candidates = np.array([indices], np.int32)
    if valid:
        np.testing.assert_array_equal(select_records(*args, candidates=candidates), candidates)
    else:
        with pytest.raises(AssertionError):
            select_records(*args, candidates=candidates)
