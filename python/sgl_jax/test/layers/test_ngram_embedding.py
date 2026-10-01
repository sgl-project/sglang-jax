import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.ngram_embedding import (
    NGramEmbedding,
    build_hash_params,
    compute_ngram_ids,
    ngram_context_row,
    ngram_context_row_split,
)
from sgl_jax.test.test_utils import CustomTestCase

HIDDEN_SIZE = 8
HC_COUNT = 4
HYPER_SIZE = HC_COUNT * HIDDEN_SIZE
PLE_EMBED_DIM = 16
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 2
CONV_KERNEL = 4
CONV_STATE_LEN = (CONV_KERNEL - 1) * NGRAM_SIZE
EPSILON = 1e-6
SEED = 42
VOCAB = 128
EOS = 7

ATOL = 2e-5
RTOL = 2e-5


class _Config:
    """The fields NGramEmbedding reads, without pulling in the real config."""

    hidden_size = HIDDEN_SIZE
    hc_count = HC_COUNT
    ple_embed_dim = PLE_EMBED_DIM
    ple_conv_kernel_size = CONV_KERNEL
    ngram_size = NGRAM_SIZE
    rms_norm_eps = EPSILON


def _make_mesh():
    devices = np.array(jax.devices())
    return Mesh(
        devices[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _params(**kw):
    base = dict(
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
        vocab_size=VOCAB,
        ngram_vocab_size_base=1000,
        eos_token_id=EOS,
    )
    base.update(kw)
    return build_hash_params(**base)


def _ref_ngram_ids(input_ids, cu_seqlens, context, params):
    """Independent reference: one token at a time, plain python ints.

    Walks each token's predecessors explicitly rather than reproducing the
    vectorized masking, so a broadcast or off-by-one there does not survive.
    """
    ctx_len = params.ngram_context_len
    out = np.zeros((len(input_ids), params.ngram_heads), dtype=np.int64)
    for b in range(len(cu_seqlens) - 1):
        start, end = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        # Prepend the chunk's context so a token's history is one flat list.
        history = list(context[b]) + list(input_ids[start:end])
        for pos in range(end - start):
            here = ctx_len + pos
            tokens = [int(history[here])]
            crossed = False
            for shift in range(1, ctx_len + 1):
                tok = int(history[here - shift])
                if crossed:
                    tok = params.eos_token_id
                crossed = crossed or tok == params.eos_token_id
                tokens.append(tok)
            for head in range(params.ngram_heads):
                order = head // params.heads_per_ngram + 2
                mixed = 0
                for i in range(order):
                    mixed ^= tokens[i] * int(params.multipliers[i])
                out[start + pos, head] = mixed % int(params.sizes[head]) + int(params.offsets[head])
    return out


def _ref_gate(hyper_input, ple_emb, key_w, value_w, nk, nq):
    """fp64 ground truth for the gate, looping over streams."""
    x = hyper_input.astype(np.float64)
    key = ple_emb.astype(np.float64) @ key_w.astype(np.float64)
    value = ple_emb.astype(np.float64) @ value_w.astype(np.float64)

    def norm(v, w, j):
        s = v[..., j * HIDDEN_SIZE : (j + 1) * HIDDEN_SIZE]
        wj = w.astype(np.float64)[j * HIDDEN_SIZE : (j + 1) * HIDDEN_SIZE]
        return s / np.sqrt(np.mean(s**2, axis=-1, keepdims=True) + EPSILON) * (1.0 + wj)

    out = np.empty_like(x)
    for j in range(HC_COUNT):
        d = np.sum(norm(key, nk, j) * norm(x, nq, j), axis=-1) / np.sqrt(HIDDEN_SIZE)
        g = 1.0 / (1.0 + np.exp(-(np.sign(d) * np.sqrt(np.maximum(np.abs(d), 1e-6)))))
        out[..., j * HIDDEN_SIZE : (j + 1) * HIDDEN_SIZE] = g[..., None] * value
    return out


def _assign(param, value, mesh, spec):
    value = jnp.array(value, param[...].dtype)
    param[...] = jax.device_put(value, NamedSharding(mesh, spec))


def _put(value, mesh, spec):
    """Place an input the way the runner does: conv state and the per-request
    metadata arrive from the pool already sharded, and the layer's shard_map
    pins those specs."""
    return jax.device_put(jnp.asarray(value), NamedSharding(mesh, spec))


def _make_layer(mesh, rng):
    with jax.set_mesh(mesh):
        layer = NGramEmbedding(_Config(), mesh, params_dtype=jnp.float32)
    w = {
        "key": rng.standard_normal((PLE_EMBED_DIM, HYPER_SIZE)).astype(np.float32),
        "value": rng.standard_normal((PLE_EMBED_DIM, HIDDEN_SIZE)).astype(np.float32),
        "nk": rng.standard_normal(HYPER_SIZE).astype(np.float32),
        "nq": rng.standard_normal(HYPER_SIZE).astype(np.float32),
        "nc": rng.standard_normal(HYPER_SIZE).astype(np.float32),
        "conv": rng.standard_normal((HYPER_SIZE, CONV_KERNEL)).astype(np.float32),
    }
    _assign(layer.key_proj.weight, w["key"], mesh, P(None, None))
    _assign(layer.value_proj.weight, w["value"], mesh, P(None, None))
    _assign(layer.norm_key.weight, w["nk"], mesh, P(None))
    _assign(layer.norm_query.weight, w["nq"], mesh, P(None))
    _assign(layer.norm_conv.weight, w["nc"], mesh, P(None))
    _assign(layer.conv1d_weight, w["conv"], mesh, P(None, None))
    return layer, w


class TestNGramHashLayout(CustomTestCase):
    def test_released_checkpoint_layout_and_multiplier_bound(self):
        params = _params(heads_per_ngram=8, vocab_size=248320, ngram_vocab_size_base=20_000_000)
        self.assertEqual((params.ngram_heads, params.total_vocab_size), (16, 320001446))
        self.assertEqual((params.sizes[0], params.sizes[-1]), (20000003, 20000171))
        self.assertEqual(len(set(params.sizes.tolist())), 16)
        np.testing.assert_array_equal(params.offsets, np.r_[0, np.cumsum(params.sizes)[:-1]])
        for multiplier in params.multipliers.tolist():
            self.assertEqual(multiplier % 2, 1)
            self.assertLess(multiplier * (248320 - 1), 2**63)


class TestComputeNGramIds(CustomTestCase):
    def test_hash_and_head_boundaries_against_python_oracle(self):
        # Includes empty requests, padding, T == B without being decode, and
        # all-EOS histories. Exercise both small and released-vocab token ids.
        for ngram_size, heads, vocab, layer_id in (
            (2, 2, 128, 0),
            (3, 8, 248320, 0),
            (4, 2, 128, 1),
        ):
            params = _params(
                ngram_size=ngram_size,
                heads_per_ngram=heads,
                vocab_size=vocab,
                ple_dense_layer_id=layer_id,
            )
            for lens in ([5, 1, 9], [1] * 6, [0, 1, 3, 0], [3, 2, 0, 0], [0, 0]):
                for eos in (False, True):
                    with self.subTest(ngram_size=ngram_size, lens=lens, eos=eos):
                        rng = np.random.default_rng(SEED)
                        cu = np.r_[0, np.cumsum(lens)]
                        ids = rng.integers(0, vocab, size=int(cu[-1]))
                        ctx = rng.integers(0, vocab, size=(len(lens), ngram_size - 1))
                        if len(ids):
                            ids[0] = vocab - 1  # largest valid product
                        if eos:
                            ids[1::3] = EOS
                            ctx[:, -1] = EOS
                        original_ctx = ctx.copy()
                        got = compute_ngram_ids(ids, cu, ctx, params)
                        np.testing.assert_array_equal(got, _ref_ngram_ids(ids, cu, ctx, params))
                        self.assertEqual(got.shape, (len(ids), params.ngram_heads))
                        self.assertEqual(got.dtype, np.int32)
                        self.assertTrue(
                            np.all((got >= params.offsets) & (got < params.offsets + params.sizes))
                        )
                        np.testing.assert_array_equal(ctx, original_ctx)

    def test_chunk_boundaries_preserve_hashes(self):
        params = _params()
        fill = np.array([10, EOS, 12, 13, 14, 15, EOS, 17, 18, 19, 20, 21])
        pad = np.full((1, NGRAM_SIZE - 1), EOS)
        whole = _ref_ngram_ids(fill, [0, len(fill)], pad, params)
        for split in (1, 2, 5, 11):
            with self.subTest(split=split):
                context = ngram_context_row_split(fill[:4], fill[4:], split, NGRAM_SIZE - 1, EOS)
                head = compute_ngram_ids(fill[:split], np.array([0, split]), pad, params)
                tail = compute_ngram_ids(
                    fill[split:], np.array([0, len(fill) - split]), context[None], params
                )
                np.testing.assert_array_equal(np.concatenate([head, tail]), whole)

    def test_request_context_boundaries(self):
        for prompt, output, start, length, expected in (
            ([10, 11, 12], [], 0, 2, [EOS, EOS]),
            ([10, 11, 12], [], 1, 2, [EOS, 10]),
            ([10, 11, 12], [13, 14], 4, 2, [12, 13]),
            ([10, 11], [12, 13, 14], 4, 2, [12, 13]),
            ([10, 11], [12], 2, 0, []),
        ):
            with self.subTest(start=start, prompt=prompt, length=length):
                np.testing.assert_array_equal(
                    ngram_context_row_split(prompt, output, start, length, EOS), expected
                )
                np.testing.assert_array_equal(
                    ngram_context_row(prompt + output, start, length, EOS), expected
                )
        with self.assertRaisesRegex(ValueError, "stale"):
            ngram_context_row_split([10, 11], [12], 4, 2, EOS)


def _ref_ple(hyper, emb, weights, pool, slots, cu, initial):
    """Independent token-by-token FP64 gate/norm/dilated-conv oracle."""
    gated = _ref_gate(hyper, emb, weights["key"], weights["value"], weights["nk"], weights["nq"])
    grouped = gated.reshape(-1, HC_COUNT, HIDDEN_SIZE)
    normalized = grouped / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + EPSILON)
    normalized = normalized.reshape(gated.shape) * (1 + weights["nc"].astype(np.float64))
    result, updated = gated.copy(), pool.copy()
    for row, slot in enumerate(slots):
        state = pool[slot].copy() if initial[row] else np.zeros_like(pool[slot])
        for token in range(cu[row], cu[row + 1]):
            window = np.concatenate([state, normalized[token, :, None]], axis=-1)
            conv = np.sum(window[:, ::NGRAM_SIZE] * weights["conv"], axis=-1)
            result[token] += conv / (1 + np.exp(-conv))
            state = window[:, 1:]
        if slot:
            updated[slot] = state
    return result, updated


class TestNGramEmbeddingLayer(CustomTestCase):
    def test_gate_matches_fp64_reference(self):
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        layer, w = _make_layer(mesh, rng)
        hyper = rng.standard_normal((6, HYPER_SIZE)).astype(np.float32)
        emb = rng.standard_normal((6, PLE_EMBED_DIM)).astype(np.float32)
        # TPU defaults fp32 matmuls to bf16 inputs (~1.7% here); this test is
        # about the gate's math, so pin the precision.
        with jax.set_mesh(mesh), jax.default_matmul_precision("float32"):
            got = layer.gate(jnp.asarray(hyper), jnp.asarray(emb))
        want = _ref_gate(hyper, emb, w["key"], w["value"], w["nk"], w["nq"])
        np.testing.assert_allclose(np.asarray(got), want, atol=ATOL, rtol=RTOL)

    def test_extend_then_decode_against_independent_oracle(self):
        for zero_conv in (False, True):
            with self.subTest(zero_conv=zero_conv):
                mesh, rng = _make_mesh(), np.random.default_rng(SEED)
                layer, weights = _make_layer(mesh, rng)
                if zero_conv:  # also checks that only the delta, not hyper_input, is returned
                    weights["conv"].fill(0)
                    _assign(layer.conv1d_weight, weights["conv"], mesh, P(None, None))
                slots, cu, initial = (
                    np.array([1, 3, 0]),
                    np.array([0, 2, 7, 8]),
                    np.array([False, True, False]),
                )
                pool = rng.standard_normal((5, HYPER_SIZE, CONV_STATE_LEN)).astype(np.float32)
                pool[1] = np.nan  # stale state must not leak into a fresh request
                expected_pool = pool.copy()
                state = _put(pool, mesh, P("data", "tensor", None))
                indices = _put(slots.astype(np.int32), mesh, P("data"))
                for decode in (False, True):
                    tokens = len(slots) if decode else int(cu[-1])
                    if decode:
                        cu, initial = np.arange(len(slots) + 1), np.ones(len(slots), bool)
                    hyper = rng.standard_normal((tokens, HYPER_SIZE)).astype(np.float32)
                    emb = rng.standard_normal((tokens, PLE_EMBED_DIM)).astype(np.float32)
                    want, expected_pool = _ref_ple(
                        hyper, emb, weights, expected_pool, slots, cu, initial
                    )
                    with jax.set_mesh(mesh), jax.default_matmul_precision("float32"):
                        args = (jnp.asarray(hyper), jnp.asarray(emb), state, indices)
                        init = _put(initial, mesh, P("data"))
                        if decode:
                            got, state = layer.forward_decode(*args, init)
                        else:
                            got, state = layer.forward_extend(
                                *args, _put(cu.astype(np.int32), mesh, P("data")), init
                            )
                    np.testing.assert_allclose(np.asarray(got), want, atol=ATOL, rtol=RTOL)
                    np.testing.assert_allclose(
                        np.asarray(state), expected_pool, atol=ATOL, rtol=RTOL
                    )
                np.testing.assert_array_equal(np.asarray(state)[[0, 2, 4]], pool[[0, 2, 4]])


@unittest.skipIf(len(jax.devices()) < 2, "tensor parallelism needs >= 2 devices")
class TestNGramEmbeddingSharding(CustomTestCase):
    """Channel-sharding the depthwise conv must not change a single value.

    Skipped on a one-device CPU runner; the TPU jobs exercise it. The layer
    reshards its activation from the projections' replicated layout to the
    pool's `P("data", "tensor", None)`, which is where a silent mismatch
    between the conv weight, the state and the activation would show up.
    """

    def test_tensor_parallel_matches_single_device(self):
        devices = np.array(jax.devices())

        def run(tp):
            mesh = Mesh(
                devices[:tp].reshape(1, tp),
                axis_names=("data", "tensor"),
                axis_types=(AxisType.Explicit, AxisType.Explicit),
            )
            rng = np.random.default_rng(SEED)
            layer, _ = _make_layer(mesh, rng)
            hyper = jnp.asarray(rng.standard_normal((8, HYPER_SIZE)).astype(np.float32))
            emb = jnp.asarray(rng.standard_normal((8, PLE_EMBED_DIM)).astype(np.float32))
            pool = _put(
                rng.standard_normal((6, HYPER_SIZE, CONV_STATE_LEN)).astype(np.float32),
                mesh,
                P("data", "tensor", None),
            )
            idx = _put(np.array([1, 3], np.int32), mesh, P("data"))
            cu = _put(np.array([0, 4, 8], np.int32), mesh, P("data"))
            init = _put(np.array([True, True]), mesh, P("data"))
            with jax.set_mesh(mesh):
                out, state = layer.forward_extend(hyper, emb, pool, idx, cu, init)
                dec, _ = layer.forward_decode(hyper[:2], emb[:2], pool, idx, init)
            return np.asarray(out), np.asarray(state), np.asarray(dec)

        ref = run(1)
        tp = 2
        while tp <= len(devices):
            for got, want in zip(run(tp), ref):
                np.testing.assert_array_equal(got, want)
            tp *= 2


if __name__ == "__main__":
    unittest.main()
