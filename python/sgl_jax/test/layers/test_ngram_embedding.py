import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.ngram_embedding import (
    NGramEmbedding,
    NGramHashParams,
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
    def test_released_checkpoint_vocab_layout(self):
        """The 16 heads take the 16 primes just above ngram_vocab_size_base."""
        params = _params(heads_per_ngram=8, vocab_size=248320, ngram_vocab_size_base=20_000_000)
        self.assertEqual(params.ngram_heads, 16)
        self.assertEqual(params.ngram_context_len, 2)
        self.assertEqual(int(params.sizes[0]), 20000003)
        self.assertEqual(int(params.sizes[-1]), 20000171)
        self.assertEqual(len(set(params.sizes.tolist())), 16)
        # 320,001,446 rows x 160 dim = the checkpoint's 51.2B parameters.
        self.assertEqual(params.total_vocab_size, 320001446)
        self.assertEqual(int(params.offsets[0]), 0)
        np.testing.assert_array_equal(params.offsets[1:], np.cumsum(params.sizes)[:-1])

    def test_multipliers_are_odd_and_cannot_overflow(self):
        params = _params(vocab_size=248320)
        self.assertEqual(len(params.multipliers), NGRAM_SIZE)
        for m in params.multipliers.tolist():
            self.assertEqual(m % 2, 1)
            # A token times its multiplier must stay inside int64, or the XOR
            # wraps and the hash silently stops matching the checkpoint.
            self.assertLess(m * (248320 - 1), (1 << 63) - 1)

    def test_layer_id_shifts_both_multipliers_and_primes(self):
        a = _params()
        b = _params(ple_dense_layer_id=1)
        self.assertFalse(np.array_equal(a.multipliers, b.multipliers))
        self.assertFalse(np.array_equal(a.sizes, b.sizes))


class TestComputeNGramIds(CustomTestCase):
    def _case(self, input_ids, cu_seqlens, context, params=None):
        params = params or _params()
        got = compute_ngram_ids(
            np.array(input_ids), np.array(cu_seqlens), np.array(context), params
        )
        want = _ref_ngram_ids(input_ids, cu_seqlens, context, params)
        np.testing.assert_array_equal(got, want)
        return got

    def test_matches_reference_on_a_ragged_batch(self):
        rng = np.random.default_rng(SEED)
        lens = [5, 1, 9]
        cu = np.concatenate([[0], np.cumsum(lens)])
        ids = rng.integers(0, VOCAB, size=int(cu[-1]))
        ctx = rng.integers(0, VOCAB, size=(len(lens), NGRAM_SIZE - 1))
        got = self._case(ids, cu, ctx)
        self.assertEqual(got.shape, (int(cu[-1]), 2 * HEADS_PER_NGRAM))
        self.assertEqual(got.dtype, np.int32)

    def test_ids_stay_inside_their_head_partition(self):
        rng = np.random.default_rng(SEED)
        params = _params()
        ids = rng.integers(0, VOCAB, size=32)
        out = compute_ngram_ids(
            ids, np.array([0, 32]), np.zeros((1, NGRAM_SIZE - 1), np.int64), params
        )
        lo = params.offsets[None, :]
        hi = lo + params.sizes[None, :]
        self.assertTrue(np.all(out >= lo) and np.all(out < hi))

    def test_chunk_boundary_reads_the_context(self):
        """Splitting a sequence in two must not change any token's ids."""
        rng = np.random.default_rng(SEED)
        params = _params()
        ids = rng.integers(0, VOCAB, size=10)
        ids[ids == EOS] = EOS + 1  # keep EOS out of it; barriers are tested below
        pad = np.full(NGRAM_SIZE - 1, EOS, np.int64)

        whole = compute_ngram_ids(ids, np.array([0, 10]), pad[None, :], params)
        first = compute_ngram_ids(ids[:4], np.array([0, 4]), pad[None, :], params)
        second = compute_ngram_ids(ids[4:], np.array([0, 6]), ids[2:4][None, :], params)
        np.testing.assert_array_equal(first, whole[:4])
        np.testing.assert_array_equal(second, whole[4:])

    def test_eos_is_a_barrier_that_swallows_older_positions(self):
        """Once the walk back hits EOS, every older position reads as EOS."""
        params = _params()
        # token 3 has predecessors (EOS, 5): the order-3 head must see EOS at
        # the older slot too, so a different token there changes nothing.
        a = compute_ngram_ids(
            np.array([9, EOS, 3]), np.array([0, 3]), np.zeros((1, 2), np.int64), params
        )
        b = compute_ngram_ids(
            np.array([4, EOS, 3]), np.array([0, 3]), np.zeros((1, 2), np.int64), params
        )
        np.testing.assert_array_equal(a[2], b[2])
        self.assertFalse(np.array_equal(a[0], b[0]))  # the changed token itself

    def test_decode_batch_matches_the_general_path(self):
        """One token per request takes a fast path that skips the position
        machinery; it has to agree with the ragged path token for token."""
        rng = np.random.default_rng(SEED)
        params = _params()
        ids = rng.integers(0, VOCAB, size=6)
        ctx = rng.integers(0, VOCAB, size=(6, NGRAM_SIZE - 1))
        decode = compute_ngram_ids(ids, np.arange(7), ctx, params)
        np.testing.assert_array_equal(decode, _ref_ngram_ids(ids, np.arange(7), ctx, params))
        # The same six requests plus one two-token request: T != B, so this
        # goes down the general path and its first six rows must be identical.
        ragged = compute_ngram_ids(
            np.concatenate([ids, [ids[0], ids[1]]]),
            np.array([0, 1, 2, 3, 4, 5, 6, 8]),
            np.concatenate([ctx, ctx[:1]]),
            params,
        )
        np.testing.assert_array_equal(ragged[:6], decode)

    def test_equal_token_and_request_counts_are_not_always_decode(self):
        """cu_seqlens [0, 0, 1, 4, 4] has 4 tokens and 4 requests but is
        ragged: the decode fast path must not fire."""
        rng = np.random.default_rng(SEED)
        params = _params()
        ids = rng.integers(0, VOCAB, size=4)
        cu = np.array([0, 0, 1, 4, 4])
        ctx = rng.integers(0, VOCAB, size=(4, NGRAM_SIZE - 1))
        got = compute_ngram_ids(ids, cu, ctx, params)
        np.testing.assert_array_equal(got, _ref_ngram_ids(ids, cu, ctx, params))

    def test_order2_and_order3_heads_differ(self):
        """The two head groups hash different-length n-grams, not the same one."""
        params = _params()
        ids = np.array([11, 12, 13, 14])
        out = compute_ngram_ids(ids, np.array([0, 4]), np.zeros((1, 2), np.int64), params)
        order2 = out[:, :HEADS_PER_NGRAM] - params.offsets[:HEADS_PER_NGRAM]
        order3 = out[:, HEADS_PER_NGRAM:] - params.offsets[HEADS_PER_NGRAM:]
        # Same primes would make this a tautology; check the residues differ.
        self.assertFalse(np.array_equal(order2[-1], order3[-1]))


# vLLM's fixed hash constants, copied verbatim from
# tests/models/qwen4_exp/test_ple.py so the ported cases hash identically.
# Deliberately NOT derived: the multipliers are ~1.8e16, so a token id above
# the vocab overflows int64 and the ids are only defined by the wraparound.
_VLLM_MULTIPLIERS = (
    18_014_398_509_481_983,
    17_114_398_509_481_981,
    16_214_398_509_481_979,
    15_314_398_509_481_977,
)
_VLLM_SIZES = (
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
)
_VLLM_EOS = 251
_VLLM_HEADS_PER_NGRAM = 8


def _vllm_params(context_len: int) -> NGramHashParams:
    """vLLM's ``_ngram_hash_params`` as an NGramHashParams."""
    num_heads = context_len * _VLLM_HEADS_PER_NGRAM
    sizes = np.array(_VLLM_SIZES[:num_heads], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    return NGramHashParams(
        multipliers=np.array(_VLLM_MULTIPLIERS[: context_len + 1], dtype=np.int64),
        sizes=sizes,
        offsets=offsets,
        total_vocab_size=int(sizes.sum()),
        heads_per_ngram=_VLLM_HEADS_PER_NGRAM,
        ngram_size=context_len + 1,
        eos_token_id=_VLLM_EOS,
    )


class TestComputeNGramIdsVsVLLM(CustomTestCase):
    """The case table from vLLM tests/models/qwen4_exp/test_ple.py
    ``test_fused_ngram_ids_correctness``, ported one for one.

    Same query_lens / eos_offsets / contexts / first_token_id, same fixed
    hash constants, same two assertions (matches an independent reference,
    and every id lands inside its head's partition).
    """

    CASES = {
        "single-token": ([1], [], [[11, 12]], 20),
        "bigram-only": ([3], [1], [[11]], 20),
        "power-of-two": ([4, 4], [0, 7], [[11, 12], [13, 14]], 20),
        "empty-request": ([4, 0, 3], [], [[11, 12], [13, 14], [15, 16]], 20),
        "trailing-padded-requests": (
            [3, 2, 0, 0],
            [],
            [[11, 12], [13, 14], [_VLLM_EOS, _VLLM_EOS], [_VLLM_EOS, _VLLM_EOS]],
            20,
        ),
        "three-requests": (
            [1, 33, 2],
            [5, 32],
            [[_VLLM_EOS, 11], [12, 13], [14, _VLLM_EOS]],
            20,
        ),
        "six-requests": (
            [5, 12, 16, 1, 16, 17],
            [10, 40],
            [[11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]],
            20,
        ),
        "four-gram": ([5, 3], [3], [[11, 12, 13], [14, 15, 16]], 20),
        "int64-overflow": (
            [4, 0, 3],
            [],
            [[200_000, 200_001], [250_000, 250_001], [300_000, 300_001]],
            350_000,
        ),
        "large-int32-ids": ([3], [], [[1_000_000_000, 1_000_000_001]], 1_000_000_002),
    }

    # vLLM's last two cases hash with multipliers 485x larger than
    # build_hash_params' cap, so the int64 product wraps negative. There our
    # uint64 reduce (compute_ngram_ids views `rolling` as unsigned) differs
    # from torch's signed remainder by exactly 2**64 % size. The cap makes
    # that regime unreachable in production -- see
    # test_the_ported_overflow_cases_are_out_of_reach below.
    OUT_OF_REGIME = ("int64-overflow", "large-int32-ids")

    def test_ported_cases(self):
        for name, (query_lens, eos_offsets, contexts, first_token_id) in self.CASES.items():
            if name in self.OUT_OF_REGIME:
                continue
            with self.subTest(name):
                params = _vllm_params(len(contexts[0]))
                cu = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int64)
                ids = np.arange(first_token_id, first_token_id + int(cu[-1]), dtype=np.int64)
                for off in eos_offsets:
                    ids[off] = _VLLM_EOS
                ctx = np.array(contexts, dtype=np.int64)

                got = compute_ngram_ids(ids, cu, ctx, params)
                want = _ref_ngram_ids(ids, cu, ctx, params)
                np.testing.assert_array_equal(got, want, err_msg=name)
                # vLLM's second assertion: ids stay inside their head partition.
                self.assertTrue(
                    np.all((got >= params.offsets) & (got < params.offsets + params.sizes)),
                    msg=name,
                )

    def test_the_ported_overflow_cases_are_out_of_reach(self):
        """Why the two skipped cases are skipped, as an assertion.

        build_hash_params caps every multiplier at (2**63-1)//vocab_size, so
        token * multiplier never sets the sign bit and XOR keeps it clear.
        `rolling` stays non-negative and the unsigned reduce is exact. Raise
        the cap and the two skipped cases become real.
        """
        params = _params(heads_per_ngram=8, vocab_size=248320, ngram_vocab_size_base=20_000_000)
        widest = int(params.multipliers.max()) * (248320 - 1)
        self.assertLess(widest, 2**63)
        self.assertGreater(_VLLM_MULTIPLIERS[0] * (248320 - 1), 2**63)


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

    def test_gate_is_one_scalar_per_stream_over_a_shared_value(self):
        """Within a stream the gated output is the value vector times a scalar."""
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        layer, w = _make_layer(mesh, rng)
        emb = rng.standard_normal((3, PLE_EMBED_DIM)).astype(np.float32)
        hyper = rng.standard_normal((3, HYPER_SIZE)).astype(np.float32)
        with jax.set_mesh(mesh), jax.default_matmul_precision("float32"):
            got = np.asarray(layer.gate(jnp.asarray(hyper), jnp.asarray(emb)))
        value = emb @ w["value"]
        streams = got.reshape(3, HC_COUNT, HIDDEN_SIZE)
        for t in range(3):
            for j in range(HC_COUNT):
                ratio = streams[t, j] / value[t]
                self.assertTrue(np.allclose(ratio, ratio[0], atol=ATOL))
                self.assertTrue(0.0 < ratio[0] < 1.0)  # sigmoid range

    def test_conv_state_is_dilated_and_nine_deep(self):
        """The N-gram conv spans (K-1)*ngram_size, not K-1 like GDN's."""
        mesh = _make_mesh()
        layer, _ = _make_layer(mesh, np.random.default_rng(SEED))
        self.assertEqual(layer.dilation, NGRAM_SIZE)
        self.assertEqual(layer.conv_state_len, CONV_STATE_LEN)
        self.assertNotEqual(layer.conv_state_len, CONV_KERNEL - 1)

    def test_decode_continues_extend(self):
        """One decode step equals a one-token extend against the same state."""
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        layer, _ = _make_layer(mesh, rng)
        n_slots, batch = 5, 2
        pool = _put(
            rng.standard_normal((n_slots, HYPER_SIZE, CONV_STATE_LEN)).astype(np.float32),
            mesh,
            P("data", "tensor", None),
        )
        idx = _put(np.array([1, 3], np.int32), mesh, P("data"))
        has_init = _put(np.array([True, True]), mesh, P("data"))
        cu = _put(np.array([0, 1, 2], np.int32), mesh, P("data"))
        hyper = jnp.asarray(rng.standard_normal((batch, HYPER_SIZE)).astype(np.float32))
        emb = jnp.asarray(rng.standard_normal((batch, PLE_EMBED_DIM)).astype(np.float32))

        with jax.set_mesh(mesh):
            y_d, s_d = layer.forward_decode(hyper, emb, pool, idx, has_init)
            y_e, s_e = layer.forward_extend(hyper, emb, pool, idx, cu, has_init)
        np.testing.assert_allclose(np.asarray(y_d), np.asarray(y_e), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(np.asarray(s_d), np.asarray(s_e), atol=ATOL, rtol=RTOL)

    def test_extend_adds_into_the_residual_stream(self):
        """A zero conv weight leaves output = hyper_input + gated, exactly."""
        mesh = _make_mesh()
        rng = np.random.default_rng(SEED)
        layer, _ = _make_layer(mesh, rng)
        _assign(
            layer.conv1d_weight,
            np.zeros((HYPER_SIZE, CONV_KERNEL), np.float32),
            mesh,
            P(None, None),
        )
        hyper = jnp.asarray(rng.standard_normal((4, HYPER_SIZE)).astype(np.float32))
        emb = jnp.asarray(rng.standard_normal((4, PLE_EMBED_DIM)).astype(np.float32))
        pool = _put(
            np.zeros((3, HYPER_SIZE, CONV_STATE_LEN), np.float32),
            mesh,
            P("data", "tensor", None),
        )
        with jax.set_mesh(mesh):
            gated = layer.gate(hyper, emb)
            out, _ = layer.forward_extend(
                hyper,
                emb,
                pool,
                _put(np.array([1], np.int32), mesh, P("data")),
                _put(np.array([0, 4], np.int32), mesh, P("data")),
                _put(np.array([False]), mesh, P("data")),
            )
        # SiLU(0) == 0, so the conv contributes nothing.
        np.testing.assert_allclose(np.asarray(out), np.asarray(hyper + gated), atol=ATOL, rtol=RTOL)


class TestNGramContextRow(CustomTestCase):
    """The per-request token tail the scheduler hands to compute_ngram_ids.

    Off-by-one here is silent: the hash still produces valid row ids, they are
    just the wrong ones, and only accuracy would show it.
    """

    CTX = NGRAM_SIZE - 1  # 2

    def test_mid_sequence_takes_the_two_preceding_tokens(self):
        fill = [10, 11, 12, 13, 14, 15]
        np.testing.assert_array_equal(ngram_context_row(fill, 4, self.CTX, EOS), np.array([12, 13]))

    def test_sequence_start_pads_with_eos(self):
        fill = [10, 11, 12]
        np.testing.assert_array_equal(
            ngram_context_row(fill, 0, self.CTX, EOS), np.array([EOS, EOS])
        )
        np.testing.assert_array_equal(
            ngram_context_row(fill, 1, self.CTX, EOS), np.array([EOS, 10])
        )

    def test_row_feeds_compute_ngram_ids_consistently(self):
        """A chunk split must give every token the same ids as the whole."""
        rng = np.random.default_rng(SEED)
        params = _params()
        fill = rng.integers(0, VOCAB, size=12).astype(np.int64)
        fill[fill == EOS] = EOS + 1
        ctx = self.CTX

        whole = compute_ngram_ids(
            fill, np.array([0, 12]), ngram_context_row(fill, 0, ctx, EOS)[None, :], params
        )
        for split in (1, 5, 11):
            head = compute_ngram_ids(
                fill[:split],
                np.array([0, split]),
                ngram_context_row(fill, 0, ctx, EOS)[None, :],
                params,
            )
            tail = compute_ngram_ids(
                fill[split:],
                np.array([0, 12 - split]),
                ngram_context_row(fill, split, ctx, EOS)[None, :],
                params,
            )
            np.testing.assert_array_equal(head, whole[:split], err_msg=f"split={split}")
            np.testing.assert_array_equal(tail, whole[split:], err_msg=f"split={split}")

    def test_decode_row_stops_short_of_the_token_being_decoded(self):
        """chunk_start is seq_len-1: the decoded token is fill_ids[-1] itself."""
        fill = [10, 11, 12, 13]  # 13 is this step's input
        seq_len = len(fill)
        np.testing.assert_array_equal(
            ngram_context_row(fill, seq_len - 1, self.CTX, EOS), np.array([11, 12])
        )

    def test_zero_context_len_is_empty(self):
        self.assertEqual(ngram_context_row([1, 2, 3], 2, 0, EOS).shape, (0,))

    def test_split_matches_the_concatenated_stream_everywhere(self):
        """The scheduler slices prompt/output separately to avoid concatenating
        a long stream per request per decode step; it must not change a row."""
        prompt = [10, 11, 12, 13]
        for n_out in range(4):
            output = [20 + i for i in range(n_out)]
            for chunk_start in range(len(prompt) + n_out + 1):
                np.testing.assert_array_equal(
                    ngram_context_row_split(prompt, output, chunk_start, self.CTX, EOS),
                    ngram_context_row(prompt + output, chunk_start, self.CTX, EOS),
                    err_msg=f"n_out={n_out} chunk_start={chunk_start}",
                )

    def test_split_rejects_a_stale_stream(self):
        """Overlap scheduling defers output_ids; reading past them would
        EOS-pad the front and silently hash the wrong n-gram."""
        with self.assertRaises(ValueError):
            ngram_context_row_split([10, 11], [12], 4, self.CTX, EOS)


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
