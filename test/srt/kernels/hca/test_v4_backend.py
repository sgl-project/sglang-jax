"""HCA through the unified V4 backend, metadata producer and pool commit."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.dsv4.paged_row_write import paged_row_write
from sgl_jax.srt.kernels.hca.attention import _small_page_scratch, _streaming_attention
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention.hca_backend import HCABackend
from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

from . import oracle
from .test_hca import ATOL, RTOL, _stream, _weights

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU")


@pytest.mark.parametrize("lengths", [(1, 15, 17, 0), (0, 0, 0), (129, 2, 63)])
def test_packed_swa_workspace(lengths):
    from sgl_jax.srt.kernels.hca.attention import _pack_chunk_kv

    batch, window, dim = len(lengths), 128, 512
    cu = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    tokens = int(cu[-1]) + 7
    rng = np.random.default_rng(95)
    history = jnp.asarray(rng.standard_normal((batch, window, dim)), jnp.bfloat16)
    values = rng.standard_normal((tokens, dim)).astype(np.float32)
    values[cu[-1] :] = np.nan  # Padded inputs must not enter any request's segment.
    values = jnp.asarray(values, jnp.bfloat16)
    seq_ids = np.pad(np.repeat(np.arange(batch), lengths), (0, 7))
    actual = jax.jit(_pack_chunk_kv)(
        history, values, jnp.asarray(cu), jnp.asarray(seq_ids), jnp.arange(tokens) < cu[-1]
    )
    rows = (tokens + batch * (window + 16) + 15) // 16 * 16
    expected = np.zeros((rows, dim), np.float32)
    for request, length in enumerate(lengths):
        start = int(cu[request]) // 16 * 16 + request * (window + 16)
        expected[start : start + window] = np.asarray(history[request], np.float32)
        expected[start + window : start + window + length] = np.asarray(
            values[cu[request] : cu[request + 1]], np.float32
        )
    assert np.isfinite(np.asarray(actual, np.float32)).all()
    np.testing.assert_array_equal(np.asarray(actual, np.float32), expected)


class Driver:
    def __init__(self, batch, max_length, page_size, padded=False, hidden=4096):
        self.hidden = hidden
        self.batch, self.padded, self.page_size = batch, padded, page_size
        self.capacity = batch + 1
        self.max_length = max_length
        self.mesh = jax.sharding.Mesh(
            np.asarray(jax.devices(), object).reshape(1, 1),
            ("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 2,
        )
        self.slots = np.arange(batch, dtype=np.int32)[::-1].copy()
        size = batch * ((max_length + page_size - 1) // page_size) * page_size
        spec = DeepseekV4CacheSpec((0, 4, 128))
        with jax.set_mesh(self.mesh):
            kv = DeepseekV4TokenToKVPool(size, size, page_size, spec, self.mesh)
            state = DeepseekV4CompressStatePool(self.capacity, spec, self.mesh)
        self.pools = MemoryPools(token_to_kv_pool=kv, compressor_state_pool=state)
        self.allocator = DeepseekV4TokenToKVPoolAllocator(kv)
        table_length = max(512, 1 << (max_length - 1).bit_length())
        self.requests = ReqToTokenPool(self.capacity, table_length)
        allocated_length = size // batch
        locs = self.allocator.alloc_extend(
            np.zeros(batch, np.int32),
            np.full(batch, allocated_length, np.int32),
            np.full(batch, -1, np.int32),
            size,
        )
        if locs is None:
            raise RuntimeError("allocation failed")
        for r, slot in enumerate(self.slots):
            rows = locs[r * allocated_length : (r + 1) * allocated_length]
            anchors = rows[::page_size][::-1]
            logical = np.arange(max_length)
            self.requests.req_to_token[slot, :max_length] = (
                anchors[logical // page_size] + logical % page_size
            )
        self.backend = HCABackend(
            mesh=self.mesh,
            compressor_hidden_size=hidden,
            page_size=page_size,
            max_context_len=table_length,
        )
        self.backend.bind_resources(self.requests, self.allocator)
        self.layer = SimpleNamespace(layer_id=2, scaling=512**-0.5)

    def put(self, value, dtype, spec):
        return jax.device_put(np.asarray(value, dtype=dtype), NamedSharding(self.mesh, spec))

    def prepare(self, prefixes, queries, mode, weights, stream=None, seed=10):
        prefixes, queries = np.asarray(prefixes, np.int32), np.asarray(queries, np.int32)
        lengths = prefixes + queries
        pos = np.concatenate(
            [np.arange(a, b, dtype=np.int32) for a, b in zip(prefixes, lengths, strict=True)]
        )
        locations = np.concatenate(
            [
                self.requests.req_to_token[s, a:b]
                for s, a, b in zip(self.slots, prefixes, lengths, strict=True)
            ]
        )
        slots = self.slots
        if self.padded:
            lengths = np.append(lengths, 0).astype(np.int32)
            prefixes = np.append(prefixes, 0).astype(np.int32)
            queries = np.append(queries, 0).astype(np.int32)
            slots = np.append(slots, -1).astype(np.int32)
            pos = np.append(pos, 0).astype(np.int32)
            locations = np.append(locations, 0).astype(np.int32)
        worker = SimpleNamespace(
            seq_lens=lengths,
            req_pool_indices=slots,
            positions=pos,
            extend_seq_lens=queries,
            extend_prefix_lens=prefixes,
            forward_mode=mode,
            out_cache_loc=locations,
            cache_loc=np.zeros(
                self.batch
                * max(1, (int(lengths.max()) + self.page_size - 1) // self.page_size)
                * self.page_size,
                np.int32,
            ),
        )
        metadata = self.backend.get_forward_metadata(
            worker,
            request_pool=self.requests,
            allocator=self.allocator,
        )
        rng = np.random.default_rng(seed)
        tensors = {}
        for name, tail, scale in [
            ("hidden", (self.hidden,), 0.05),
            ("q", (64, 512), 1.0),
            ("kv", (512,), 1.0),
        ]:
            if stream is None:
                value = rng.standard_normal((len(pos), *tail), dtype=np.float32) * scale
            else:
                value = np.concatenate(
                    [
                        stream[name][r, a:b]
                        for r, (a, b) in enumerate(
                            zip(
                                prefixes[: self.batch],
                                lengths[: self.batch],
                                strict=True,
                            )
                        )
                    ]
                )
                if self.padded:
                    value = np.concatenate([value, np.zeros((1, *tail), np.float32)])
            tensors[name] = self.put(
                value,
                jnp.bfloat16,
                P("data", "tensor", None) if name == "q" else P("data", None),
            )
        for k, value in weights.items():
            spec = P("tensor") if k == "sink" else P(*([None] * value.ndim))
            tensors[k] = self.put(
                value,
                jnp.bfloat16 if k in ("wkv", "wgate", "norm") else jnp.float32,
                spec,
            )
        tensors["fused"] = self.put(
            np.concatenate([weights["wkv"], weights["wgate"]]).T,
            jnp.bfloat16,
            P(None, None),
        )
        tensors["positions"] = self.put(pos, jnp.int32, P("data"))
        return tensors, metadata

    def executable(self, mode, norm_eps=1e-6, *, donate=False):
        graphdef, state = nnx.split(self.backend)

        def execute(data, pools, metadata):
            backend = nnx.merge(graphdef, state)
            backend.forward_metadata = metadata
            output, layer_updates = backend(
                data["q"],
                data["kv"],
                data["kv"],
                self.layer,
                SimpleNamespace(positions=data["positions"], forward_mode=mode),
                pools.token_to_kv_pool,
                compressor_state_pool=pools.compressor_state_pool,
                compressor_input=data["hidden"],
                wkv=data["wkv"],
                wgate=data["wgate"],
                ape=data["ape"],
                norm_weight=data["norm"],
                cos=data["cos"],
                sin=data["sin"],
                fused_weight=data["fused"],
                attention_sink=data["sink"],
                norm_eps=norm_eps,
            )
            assert set(layer_updates) == {"state", "swa", "compressed"}
            updates = backend.pack_pool_updates(
                {2: layer_updates},
                pools.token_to_kv_pool,
                pools.compressor_state_pool,
            )
            return output, updates

        return jax.jit(execute, donate_argnums=(1,) if donate else ())


@pytest.mark.parametrize("missing", ["state", "swa", "compressed"])
def test_rejects_partial_family_updates(missing):
    updates = {name: None for name in ("state", "swa", "compressed") if name != missing}
    with pytest.raises(ValueError, match="complete"):
        HCABackend.pack_pool_updates({2: updates}, object(), object())


def test_backend_forwards_norm_epsilon():
    driver = Driver(1, 128, 128)
    weights, stream = _weights(91), _stream(1, 128, 92)
    data, metadata = driver.prepare([0], [128], ForwardMode.EXTEND, weights, stream)
    epsilon = 0.1
    with jax.set_mesh(driver.mesh):
        out, updates = driver.executable(ForwardMode.EXTEND, epsilon)(data, driver.pools, metadata)
        jax.block_until_ready((out, updates))
        driver.pools.replace_all(updates)
    expected = oracle.records(stream["hidden"][0], weights, 1, norm_eps=epsilon)
    default = oracle.records(stream["hidden"][0], weights, 1)
    assert not np.allclose(default, expected, rtol=2e-2, atol=1e-2)
    row = driver.requests.req_to_token[driver.slots[0], 127] // 128
    cache = np.asarray(driver.pools.token_to_kv_pool.get_compressed_buffer(2), np.float32)
    np.testing.assert_allclose(
        cache.reshape(-1, 512)[row : row + 1], expected, rtol=2e-2, atol=1e-2
    )
    expected_output = oracle.request_outputs(
        stream, 0, list(range(128)), weights, 512**-0.5, norm_eps=epsilon
    )
    assert np.isfinite(np.asarray(out)).all()
    np.testing.assert_allclose(np.asarray(out, np.float32), expected_output, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("page", [128, 256])
def test_all_padding_leaves_all_families_unchanged(page):
    driver = Driver(2, 128, page, padded=True)
    data, metadata = driver.prepare([0, 0], [0, 0], ForwardMode.EXTEND, _weights(93))
    before = jax.tree.map(lambda a: np.array(a), driver.pools)
    with jax.set_mesh(driver.mesh):
        out, updates = driver.executable(ForwardMode.EXTEND)(data, driver.pools, metadata)
        jax.block_until_ready((out, updates))
        driver.pools.replace_all(updates)
    np.testing.assert_array_equal(np.asarray(out), 0)
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(driver.pools), strict=True):
        np.testing.assert_array_equal(old, np.asarray(new))


@pytest.mark.parametrize("page", [128, 256])
def test_long_swa_reuse_across_requests(page):
    weights = _weights(20260928)
    stream = _stream(3, 640, 29)
    queries = [577, 0, 353]  # Slab rollover, empty request, new owner and partial tail.
    driver = Driver(3, 640, page, padded=True)
    data, md = driver.prepare([0, 0, 0], queries, ForwardMode.EXTEND, weights, stream)
    with jax.set_mesh(driver.mesh):
        output, updates = driver.executable(ForwardMode.EXTEND)(data, driver.pools, md)
        jax.block_until_ready((output, updates))
    actual = np.asarray(output, np.float32).reshape(-1, 64, 512)
    expected = np.concatenate(
        [
            oracle.request_outputs(stream, r, range(n), weights, 512**-0.5)
            for r, n in enumerate(queries)
            if n
        ]
    )
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[:-1], expected, rtol=2e-2, atol=1e-2)
    np.testing.assert_array_equal(actual[-1], 0)


@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("hidden", [4096, 7168])
def test_stateful_backend(page, hidden):
    weights = _weights(20260924, hidden)
    stream = _stream(2, 384, 18, hidden)
    driver = Driver(2, 384, page, padded=True, hidden=hidden)
    prefixes = np.zeros(2, np.int32)
    steps = ([127, 63], [1, 1], [129, 82], [1, 1], [1, 129], [2, 129], [256, 256], [128, 128])
    for i, queries in enumerate(steps):
        if i in (5, 6):
            # Recompute on the same slots: reset numerical state, not ownership.
            prefixes[:] = 0
        if i == 6:
            driver.padded = False  # Fresh equal-length requests exercise native uniform prefill.
        mode = ForwardMode.DECODE if i in (1, 3) else ForwardMode.EXTEND
        data, md = driver.prepare(prefixes, queries, mode, weights, stream)
        before = jax.tree.map(lambda a: np.array(a), driver.pools)
        with jax.set_mesh(driver.mesh):
            output, updates = driver.executable(mode)(data, driver.pools, md)
            jax.block_until_ready((output, updates))
            # Non-donated inputs must remain immutable; all seven families survive.
            for old, current in zip(
                jax.tree.leaves(before), jax.tree.leaves(driver.pools), strict=True
            ):
                np.testing.assert_array_equal(old, np.asarray(current))
            driver.pools.replace_all(updates)
        assert set(driver.pools.token_to_kv_pool.buffers) == {
            "swa",
            "c4",
            "indexer",
            "c128",
        }
        assert set(driver.pools.compressor_state_pool.buffers) == {
            "c4",
            "indexer",
            "c128",
        }
        for name in ("c4", "indexer"):
            for old, new in zip(
                before.token_to_kv_pool.buffers[name],
                driver.pools.token_to_kv_pool.buffers[name],
                strict=True,
            ):
                np.testing.assert_array_equal(old, np.asarray(new))
            for old, new in zip(
                before.compressor_state_pool.buffers[name],
                driver.pools.compressor_state_pool.buffers[name],
                strict=True,
            ):
                np.testing.assert_array_equal(old, np.asarray(new))
        for layer in (0, 1):
            np.testing.assert_array_equal(
                before.token_to_kv_pool.get_swa_buffer(layer),
                np.asarray(driver.pools.token_to_kv_pool.get_swa_buffer(layer)),
            )
        actual = np.asarray(output, np.float32).reshape(-1, 64, 512)
        expected = np.concatenate(
            [
                oracle.request_outputs(
                    stream, r, list(range(int(p), int(p + n))), weights, 512**-0.5
                )
                for r, (p, n) in enumerate(zip(prefixes, queries, strict=True))
            ]
        )
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual[: sum(queries)], expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_array_equal(actual[sum(queries) :], 0)
        # Cache contents and untouched state rows are part of the interface gate.
        for r, (slot, pre, count) in enumerate(zip(driver.slots, prefixes, queries, strict=True)):
            end = int(pre + count)
            recs = oracle.records(stream["hidden"][r], weights, end // 128)
            locs = driver.requests.req_to_token[slot, 127:end:128] // 128
            cache = np.asarray(driver.pools.token_to_kv_pool.get_compressed_buffer(2), np.float32)
            rows = cache.reshape(-1, 512)[locs]
            np.testing.assert_allclose(rows, recs, rtol=2e-2, atol=1e-2)
            history = driver.requests.req_to_token[slot, max(0, end - 128) : end]
            window_locs = driver.allocator.full_to_swa_index_mapping[history]
            swa = np.asarray(driver.pools.token_to_kv_pool.get_swa_buffer(2), np.float32)
            np.testing.assert_array_equal(
                swa[window_locs], stream["kv"][r, max(0, end - 128) : end]
            )
            np.testing.assert_array_equal(cache[0], 0)
            np.testing.assert_array_equal(swa[:page], 0)
            positions = np.arange(max(0, end - 128), end)
            h = stream["hidden"][r, positions]
            kv = h @ weights["wkv"].T
            score = h @ weights["wgate"].T + weights["ape"][positions % 128]
            actual_state = np.asarray(driver.pools.compressor_state_pool.get_buffer("c128", 2))
            if i == 6:
                # Complete uniform groups need no carry; the next step checks continuation.
                np.testing.assert_array_equal(actual_state[slot, :, 0], 0)
                assert np.isneginf(actual_state[slot, :, 1]).all()
                continue
            np.testing.assert_allclose(
                actual_state[slot, positions % 128, 0], kv, rtol=2e-2, atol=1e-2
            )
            np.testing.assert_allclose(
                actual_state[slot, positions % 128, 1], score, rtol=2e-2, atol=1e-2
            )
        state = np.asarray(driver.pools.compressor_state_pool.get_buffer("c128", 2))
        assert np.all(state[driver.batch :, :, 0] == 0)
        assert np.all(np.isneginf(state[driver.batch :, :, 1]))
        prefixes += np.asarray(queries)


@pytest.mark.parametrize("tokens,run", [(17, 16), (257, 128), (131072, 128)])
def test_row_writer_metadata_stays_bounded(tokens, run):
    rng = np.random.default_rng(94)
    rows, dim = tokens + 128, 512
    rows = (rows + 15) // 16 * 16
    values = jnp.asarray(rng.standard_normal((tokens, dim), dtype=np.float32), jnp.bfloat16)
    loc = np.arange(tokens, dtype=np.int32) + 64
    valid = np.ones(tokens, bool)
    if tokens < 131072:
        loc = rng.integers(0, rows, tokens, dtype=np.int32)
        loc[2] = loc[-1]  # Last valid row wins, including across write segments.
        loc[3], loc[4] = -1, rows
        valid[5::7] = False
        valid[-1] = True
    expected = np.zeros((rows, dim), dtype=np.float32)
    host_values = np.asarray(values, np.float32)
    for i in range(tokens):
        if valid[i] and 0 <= loc[i] < rows:
            expected[loc[i]] = host_values[i]
    write = jax.jit(
        lambda cache, values, loc, valid: paged_row_write(cache, values, loc, valid, run=run)
    )
    out = write(
        jnp.zeros((rows, dim), jnp.bfloat16),
        values,
        jnp.asarray(loc),
        jnp.asarray(valid),
    )
    np.testing.assert_array_equal(np.asarray(out, np.float32), expected)


@pytest.mark.parametrize("records_per_page", [1, 2])
def test_hca_small_page_pipeline(records_per_page):
    counts = np.array([0, 1, 127, 128, 129, 259], np.int32)
    rng = np.random.default_rng(20260926)
    q = oracle.bf16(rng.normal(size=(len(counts), 8, 512)))
    windows = oracle.bf16(rng.normal(size=(len(counts), 128, 512)))
    cache = oracle.bf16(rng.normal(size=(800, 1, records_per_page, 512)))
    cache[0] = np.nan
    page_counts = (counts + records_per_page - 1) // records_per_page
    boundaries = np.cumsum(np.r_[0, page_counts])
    pages = rng.permutation(np.arange(1, 800))
    tables = [pages[lo:hi] for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True)]
    if counts[-1] % records_per_page:
        cache[tables[-1][-1], 0, -1] = np.nan
    starts = np.cumsum([0] + [len(t) for t in tables[:-1]]).astype(np.int32)
    window_lens = np.array([0, 128, 128, 128, 128, 128], np.int32)
    sink = np.zeros(8, np.float32)
    schedule = get_hca_kernel_schedule(
        jax.devices()[0].device_kind,
        page_size=records_per_page,
        max_compressed_entries=128,
        local_heads=8,
        head_dim=512,
    )
    output = _streaming_attention(
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(windows, jnp.bfloat16),
        jnp.asarray(window_lens),
        jnp.asarray(cache, jnp.bfloat16),
        jnp.asarray(np.concatenate(tables), jnp.int32),
        jnp.asarray(starts),
        jnp.asarray(counts),
        jnp.asarray(sink),
        schedule=schedule,
        softmax_scale=512**-0.5,
    )
    expected = []
    for row, count in enumerate(counts):
        records = cache[tables[row]].reshape(-1, 512)[:count]
        keys = np.concatenate([windows[row, : window_lens[row]], records])
        scores = q[row] @ keys.T * np.float32(512**-0.5)
        shift = np.maximum(np.max(scores, axis=1, initial=-np.inf), sink)[:, None]
        probs = np.exp(scores - shift)
        expected.append(
            probs @ keys / (np.sum(probs, axis=1, keepdims=True) + np.exp(sink[:, None] - shift))
        )
    actual = np.asarray(output, np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.stack(expected), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("rows", [15, 26, 32])
def test_flat_cache_alignment_contract(rows):
    cache = jnp.zeros((rows, 512), jnp.bfloat16)
    if rows % 16:
        with pytest.raises(ValueError, match="row count divisible"):
            _small_page_scratch(cache, 1, 512)
    else:
        assert _small_page_scratch(cache, 1, 512).shape == (16, 512)
