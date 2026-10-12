"""V4 resources, independent arithmetic, chunk continuation and write ownership."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.kernels.csa_compressor.compressor import (
    CompressorBlock,
    CompressorMetadata,
    csa_compressor,
)
from sgl_jax.srt.kernels.csa_compressor.tune import (
    CompressorSchedule,
    get_compressor_schedule,
)
from sgl_jax.srt.kernels.dsv4.state_init import init_state_slots
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools

from . import ref


class Case:
    def __init__(
        self, batch=2, hidden=128, page_size=128, seed=47, capacity=1024, state_capacity=None
    ):
        self.batch, self.hidden = batch, hidden
        self.capacity = capacity
        rng = np.random.default_rng(seed)
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
        spec = DeepseekV4CacheSpec((0, 4, 128))
        with jax.set_mesh(mesh):
            self.pool = DeepseekV4TokenToKVPool((batch + 2) * capacity, 1024, page_size, spec, mesh)
            self.state = DeepseekV4CompressStatePool(
                batch + 1 if state_capacity is None else state_capacity, spec, mesh
            )
        self.buffers = (
            self.state.get_buffer("c4", 1),
            self.state.get_buffer("indexer", 1),
            self.pool.get_compressed_buffer(1),
            self.pool.get_indexer_buffer(1),
        )
        self.expected = tuple(np.asarray(a).copy() for a in self.buffers)
        self.slots = np.r_[np.arange(batch)[::-1], self.state.padding_index].astype(np.int32)
        self.weight = (rng.standard_normal((hidden, 2560)) / np.sqrt(hidden)).astype(
            ml_dtypes.bfloat16
        )
        self.apes = tuple(rng.normal(0, 0.2, (4, 2 * d)).astype(np.float32) for d in (512, 128))
        self.norms = tuple(rng.uniform(0.8, 1.2, d).astype(np.float32) for d in (512, 128))
        angle = np.arange(capacity)[:, None] * rng.uniform(0.001, 0.1, (1, 32))
        self.cos, self.sin = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
        self.stream = rng.normal(size=(batch, capacity, hidden)).astype(ml_dtypes.bfloat16)
        self.prefix = np.zeros(batch, np.int32)
        self.page_size = page_size
        self.schedule = (
            get_compressor_schedule(hidden, device_kind=jax.devices()[0].device_kind)
            if jax.default_backend() == "tpu"
            else CompressorSchedule(min(hidden, 128))
        )

    def inputs(self, lengths, suppress=False):
        positions = np.concatenate(
            [np.arange(p, p + n) for p, n in zip(self.prefix, lengths, strict=True)]
            + [np.array([0])]
        ).astype(np.int32)
        x = np.concatenate(
            [
                self.stream[r, p : p + n]
                for r, (p, n) in enumerate(zip(self.prefix, lengths, strict=True))
            ]
            + [np.zeros((1, self.hidden), ml_dtypes.bfloat16)]
        )
        cu = np.cumsum([0, *lengths, 1]).astype(np.int32)
        ids = np.full((self.batch + 1, max(1, max(lengths))), -1, np.int32)
        loc = np.full(len(x), -1, np.int32)
        for r, (p, n) in enumerate(zip(self.prefix, lengths, strict=True)):
            ids[r, :n] = np.arange(cu[r], cu[r + 1])
            pos = np.arange(p, p + n)
            # Reverse physical page order; do not confuse token positions and record offsets.
            page = (
                1
                + r * (self.capacity // self.page_size)
                + (self.capacity // self.page_size - 1 - pos // self.page_size)
            )
            loc[cu[r] : cu[r + 1]] = page * (self.page_size // 4) + (pos % self.page_size) // 4
        ids[-1, 0] = cu[-2]
        if suppress:
            loc[:] = -1
        ints = lambda a: jnp.asarray(a, jnp.int32)
        md = CompressorMetadata(
            ints(positions),
            ints(cu),
            ints(self.slots),
            ints(loc),
            (CompressorBlock(ints(ids), ints(np.arange(self.batch + 1))),),
        )
        return x, md

    def run(self, lengths, suppress=False, *, norm_eps=1e-6):
        x, md = self.inputs(lengths, suppress)
        original = tuple(np.asarray(a).copy() for a in self.buffers)
        expected = ref.compressor(
            x,
            self.weight,
            self.apes,
            self.norms,
            self.cos,
            self.sin,
            self.expected[:2],
            self.expected[2:],
            np.asarray(md.positions),
            np.asarray(md.cu_q_lens),
            self.slots,
            np.asarray(md.cache_locations),
            norm_eps=norm_eps,
        )
        args = tuple(
            jnp.asarray(v) for v in (x, self.weight, *self.apes, *self.norms, self.cos, self.sin)
        )
        out = jax.block_until_ready(
            csa_compressor(
                *args,
                *self.buffers,
                md,
                schedule=self.schedule,
                norm_eps=norm_eps,
                interpret=jax.default_backend() != "tpu",
            )
        )
        for got, want, before, input_array in zip(
            out, expected, original, self.buffers, strict=True
        ):
            actual, want = np.asarray(got).astype(np.float32), np.asarray(want).astype(np.float32)
            np.testing.assert_array_equal(np.asarray(input_array), before)
            np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(want))
            assert not np.isnan(actual).any() and not np.isposinf(actual).any()
            finite = np.isfinite(want)
            np.testing.assert_allclose(
                actual[finite], want[finite], rtol=2e-2, atol=1e-2, equal_nan=False
            )
        # Unowned families, padding slots, page zero and untouched request slot stay unchanged.
        for after, before in zip(out[:2], original[:2], strict=True):
            np.testing.assert_array_equal(after[-2:], before[-2:])
        for after, before in zip(out[2:], original[2:], strict=True):
            np.testing.assert_array_equal(after[0], before[0])
        self.buffers, self.expected = out, expected
        self.prefix += np.asarray(lengths, np.int32)
        return out


@pytest.mark.parametrize("page", [128, 256])
def test_ring_continuation(page):
    case = Case(page_size=page)
    for lengths in ((3, 1), (1, 6), (9, 4), (0, 5), (1, 1), (2, 3)):
        case.run(lengths)


def test_nondefault_norm_eps():
    case = Case(capacity=128)
    for lengths in ((7, 3), (1, 1), (4, 4)):
        case.run(lengths, norm_eps=0.1)


@pytest.mark.parametrize("bad_tokens", [np.zeros(2, np.int32), np.zeros((2, 1), np.float32)])
def test_invalid_block_tokens(bad_tokens):
    case = Case(capacity=128)
    x, md = case.inputs((1, 1))
    md = md._replace(
        blocks=(CompressorBlock(jnp.asarray(bad_tokens), md.blocks[0].request_indices),)
    )
    with pytest.raises(ValueError, match="block token_indices must be nonempty int32"):
        csa_compressor(
            *[
                jnp.asarray(a)
                for a in (x, case.weight, *case.apes, *case.norms, case.cos, case.sin)
            ],
            *case.buffers,
            md,
            schedule=case.schedule,
            interpret=jax.default_backend() != "tpu",
        )


@pytest.mark.parametrize("hidden", [128, 4096, 7168])
def test_prefill_and_decode(hidden):
    case = Case(hidden=hidden)
    for lengths in ((133, 24), (1, 1), (2, 3), (1, 1)):
        case.run(lengths)


@pytest.mark.parametrize("batch,hidden,length", [(1, 4096, 8192), (4, 7168, 1024), (32, 4096, 256)])
def test_long_continuation(batch, hidden, length):
    case = Case(batch=batch, hidden=hidden, capacity=length + 256)
    case.run(tuple(length - r % 4 for r in range(batch)))
    case.run(tuple(4 for _ in range(batch)))


@pytest.mark.parametrize("hidden", [4096, 7168])
@pytest.mark.parametrize("batch", [1, 4, 32])
def test_batched_decode_projection(batch, hidden):
    case = Case(batch=batch, hidden=hidden, capacity=128)
    for length in (7, 1, 1, 3):
        case.run((length,) * batch)


def test_decode_leaves_unused_state_slots_unchanged():
    case = Case(batch=1, capacity=128, state_capacity=129)
    untouched = tuple(np.asarray(s[1:]).copy() for s in case.buffers[:2])
    for length in (7, 1, 1, 3):
        case.run((length,))
    for state, before in zip(case.buffers[:2], untouched, strict=True):
        np.testing.assert_array_equal(state[1:], before)


def test_single_token_decode_ring_phases():
    case = Case(capacity=128)
    case.run((0, 3))
    for _ in range(16):
        case.run((1, 1))


def test_suppressed_cache_writes_still_update_state():
    case = Case()
    before = tuple(np.asarray(a).copy() for a in case.buffers)
    out = case.run((12, 8), suppress=True)
    for actual, wanted in zip(out[2:], before[2:], strict=True):
        np.testing.assert_array_equal(actual, wanted)
    assert not np.array_equal(out[0], before[0])


def test_empty_request_batch():
    case = Case()
    before = tuple(np.asarray(a).copy() for a in case.buffers)
    out = case.run((0, 0))
    for actual, wanted in zip(out, before, strict=True):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
def test_batch_and_resource_commit(batch):
    case = Case(batch=batch)
    pools = MemoryPools(token_to_kv_pool=case.pool, compressor_state_pool=case.state)
    untouched = (
        case.pool.get_swa_buffer(1),
        case.pool.get_compressed_buffer(2),
        case.state.get_buffer("c128", 2),
    )
    out = case.run(tuple(5 + r % 8 for r in range(batch)))
    pools.replace_all(
        {
            "token_to_kv_pool": case.pool.build_buffer_updates(
                {1: {"compressed": out[2], "indexer": out[3]}}
            ),
            "compressor_state_pool": case.state.build_buffer_updates(
                {1: {"compressor": out[0], "indexer": out[1]}}
            ),
        }
    )
    assert case.state.get_buffer("c4", 1) is out[0]
    assert case.state.get_buffer("indexer", 1) is out[1]
    assert case.pool.get_compressed_buffer(1) is out[2]
    assert case.pool.get_indexer_buffer(1) is out[3]
    assert case.pool.get_swa_buffer(1) is untouched[0]
    assert case.pool.get_compressed_buffer(2) is untouched[1]
    assert case.state.get_buffer("c128", 2) is untouched[2]


def test_multiple_blocks_and_slot_reuse():
    case = Case()
    case.run((9, 5))
    # Runtime resets recycled slots; the kernel must not reset continuing slots.
    slots = jnp.asarray(case.slots[:2])
    reset = []
    for state in case.buffers[:2]:
        template = jnp.zeros(state.shape[1:], jnp.float32)
        template = template.at[:, state.shape[-1] // 2 :].set(-jnp.inf)
        reset.append(init_state_slots(state, slots, template, capacity=case.state.padding_index))
    case.buffers = (
        *reset,
        *case.buffers[2:],
    )
    case.expected = (*[np.asarray(s).copy() for s in case.buffers[:2]], *case.expected[2:])
    case.prefix[:] = 0
    x, md = case.inputs((8, 8))
    block = md.blocks[0]
    blocks = tuple(
        CompressorBlock(block.token_indices[:, a:b], block.request_indices)
        for a, b in ((0, 3), (3, 4), (4, 8))
    )
    args = tuple(
        jnp.asarray(v) for v in (x, case.weight, *case.apes, *case.norms, case.cos, case.sin)
    )
    actual = jax.block_until_ready(
        csa_compressor(
            *args,
            *case.buffers,
            md._replace(blocks=blocks),
            schedule=case.schedule,
            interpret=jax.default_backend() != "tpu",
        )
    )
    expected = case.run((8, 8))
    for got, want in zip(actual, expected, strict=True):
        g, w = np.asarray(got).astype(np.float32), np.asarray(want).astype(np.float32)
        np.testing.assert_array_equal(np.isneginf(g), np.isneginf(w))
        finite = np.isfinite(w)
        assert not np.isnan(g).any()
        np.testing.assert_allclose(g[finite], w[finite], rtol=2e-2, atol=1e-2, equal_nan=False)


def test_wrong_ring_address_is_detected():
    case = Case()
    case.run((3, 3))
    wrong = list(case.buffers)
    wrong[0] = jnp.roll(wrong[0], 4, axis=1)
    case.buffers = tuple(wrong)
    with pytest.raises(AssertionError):
        case.run((1, 1))


def test_invalid_record_addresses_preserve_cache():
    case = Case()
    x, md = case.inputs((8, 8))
    locations = np.asarray(md.cache_locations).copy()
    locations[3] = 0
    locations[7] = np.prod(case.buffers[2].shape[:2])
    md = md._replace(cache_locations=jnp.asarray(locations))
    expected = ref.compressor(
        x,
        case.weight,
        case.apes,
        case.norms,
        case.cos,
        case.sin,
        case.expected[:2],
        case.expected[2:],
        np.asarray(md.positions),
        np.asarray(md.cu_q_lens),
        case.slots,
        locations,
    )
    actual = csa_compressor(
        *[jnp.asarray(a) for a in (x, case.weight, *case.apes, *case.norms, case.cos, case.sin)],
        *case.buffers,
        md,
        schedule=case.schedule,
        interpret=jax.default_backend() != "tpu",
    )
    for got, want in zip(actual[2:], expected[2:], strict=True):
        got, want = np.asarray(got).astype(np.float32), np.asarray(want).astype(np.float32)
        assert np.isfinite(got).all() and np.isfinite(want).all()
        np.testing.assert_allclose(got, want, rtol=2e-2, atol=1e-2, equal_nan=False)
