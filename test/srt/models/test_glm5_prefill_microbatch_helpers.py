"""Helpers behind DSA_PREFILL_MICROBATCH (glm5_moe): micro-batch bounds, the
per-window ForwardBatch view, the KV-pool overlay that hands a later
micro-batch the buffers an earlier one already rewrote in the same forward,
and the PIN=7 two-window schedule."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models import glm5_moe


class BoundsTest(unittest.TestCase):
    def test_disabled_or_too_small(self):
        self.assertIsNone(glm5_moe.prefill_microbatch_bounds(4096, 1, 128, 2048))
        self.assertIsNone(glm5_moe.prefill_microbatch_bounds(1024, 2, 128, 2048))
        # halves must sit on the TSA / zigzag alignment
        self.assertIsNone(glm5_moe.prefill_microbatch_bounds(2304, 2, 1024, 2048))

    def test_two_halves(self):
        self.assertEqual(
            glm5_moe.prefill_microbatch_bounds(4096, 2, 128, 2048), [(0, 2048), (2048, 4096)]
        )
        self.assertEqual(
            glm5_moe.prefill_microbatch_bounds(8192, 2, 128, 2048), [(0, 4096), (4096, 8192)]
        )

    def test_three_parts_non_divisible(self):
        self.assertEqual(
            glm5_moe.prefill_microbatch_bounds(4096, 3, 128, 2048),
            [(0, 1408), (1408, 2816), (2816, 4096)],
        )
        self.assertEqual(
            glm5_moe.prefill_microbatch_bounds(8192, 3, 128, 2048),
            [(0, 2816), (2816, 5504), (5504, 8192)],
        )
        self.assertEqual(
            glm5_moe.prefill_microbatch_bounds(2048, 3, 128, 2048),
            [(0, 768), (768, 1408), (1408, 2048)],
        )
        self.assertIsNone(
            glm5_moe.prefill_microbatch_bounds(256, 3, 128, 0)
        )  # fewer units than parts

    def test_four_parts(self):
        self.assertEqual(
            glm5_moe.prefill_microbatch_bounds(4096, 4, 128, 2048),
            [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)],
        )


class _Backend:
    def __init__(self, md):
        self.forward_metadata = md
        self.full_slot = {0: 0}


class _Pool:
    def __init__(self):
        self.kv_buffer = [jnp.zeros((4, 2)), jnp.ones((4, 2))]
        self.indexer_key_buffer = [jnp.full((4, 2), 7.0)]
        self.page_size = 128

    def get_fused_kv_buffer(self, layer_id):
        return self.kv_buffer[layer_id]

    def get_indexer_key_buffer(self, slot):
        return self.indexer_key_buffer[slot]


def _fb(T=512, S=3):
    md = MLAAttentionMetadata()
    md.cu_q_lens = jnp.asarray([0, 200, 450, 512], jnp.int32)
    md.cu_kv_lens = jnp.asarray([0, 256, 512, 640], jnp.int32)
    md.page_indices = jnp.arange(5, dtype=jnp.int32)
    md.seq_lens = jnp.asarray([200, 250, 62], jnp.int32)
    md.distribution = jnp.asarray([0, 0, 3], jnp.int32)
    return ForwardBatch(
        bid=1,
        forward_mode=ForwardMode.EXTEND,
        batch_size=S,
        input_ids=jnp.arange(T, dtype=jnp.int32),
        req_pool_indices=jnp.arange(S, dtype=jnp.int32),
        seq_lens=md.seq_lens,
        out_cache_loc=jnp.arange(T, dtype=jnp.int32) + 1000,
        positions=jnp.arange(T, dtype=jnp.int32) * 2,
        attn_backend=_Backend(md),
        extend_seq_lens=jnp.asarray([200, 250, 62], jnp.int32),
        expert_location_metadata="eplb",
    )


class SliceForwardBatchTest(unittest.TestCase):
    def test_slices_token_arrays_and_windows_metadata(self):
        fb = _fb()
        part = glm5_moe._slice_forward_batch(fb, 256, 512)
        np.testing.assert_array_equal(np.asarray(part.input_ids), np.arange(256, 512))
        np.testing.assert_array_equal(np.asarray(part.out_cache_loc), np.arange(256, 512) + 1000)
        np.testing.assert_array_equal(np.asarray(part.positions), np.arange(256, 512) * 2)
        self.assertIs(part.forward_mode, ForwardMode.EXTEND)
        self.assertEqual(part.expert_location_metadata, "eplb")
        self.assertIsNot(part.attn_backend, fb.attn_backend)
        self.assertEqual(part.attn_backend.full_slot, {0: 0})
        md = part.attn_backend.forward_metadata
        np.testing.assert_array_equal(np.asarray(md.cu_q_lens), [0, 0, 194, 256])
        np.testing.assert_array_equal(np.asarray(md.seq_lens), [200, 250, 62])
        # original untouched
        np.testing.assert_array_equal(
            np.asarray(fb.attn_backend.forward_metadata.cu_q_lens), [0, 200, 450, 512]
        )
        self.assertEqual(fb.input_ids.shape[0], 512)


class PoolViewTest(unittest.TestCase):
    def test_overlay_serves_rewritten_buffers(self):
        pool = _Pool()
        view = glm5_moe._MicrobatchPoolView(pool)
        self.assertIs(view.get_fused_kv_buffer(1), pool.kv_buffer[1])
        self.assertEqual(view.page_size, 128)  # passthrough
        new_kv = jnp.full((4, 2), 3.0)
        new_idx = jnp.full((4, 2), 9.0)
        view.update(1, 0, new_kv, new_idx)
        self.assertIs(view.get_fused_kv_buffer(1), new_kv)
        self.assertIs(view.get_fused_kv_buffer(0), pool.kv_buffer[0])
        self.assertIs(view.get_indexer_key_buffer(0), new_idx)
        view.update(0, None, jnp.zeros((1,)), None)
        self.assertIs(view.get_indexer_key_buffer(0), new_idx)


class _StubLayer:
    """Records the per-window call order and threads a fake paged buffer:
    attention 'writes' the window's rows into the buffer it read from the pool."""

    def __init__(self, layer_id, log):
        self.layer_id = layer_id
        self.log = log

    def __call__(
        self,
        positions,
        hidden,
        fb,
        pool,
        residual,
        dispatch_info=None,
        dsa_topk_in=None,
        dsa_topk_pages_in=None,
    ):
        from sgl_jax.srt.layers.attention.dsa_sparse_backend import DSAFusedCache

        kv = pool.get_fused_kv_buffer(self.layer_id)
        idx = pool.get_indexer_key_buffer(self.layer_id)
        loc = fb.out_cache_loc
        row0_before = float(kv[0, 0])  # what this window sees of row 0 before its own write
        kv = kv.at[loc].set(hidden[:, :1] + 1.0)
        idx = idx.at[loc].set(positions[:, None].astype(idx.dtype))
        self.log.append((self.layer_id, int(positions[0]), row0_before, dsa_topk_pages_in))
        out = hidden * 2.0 + float(self.layer_id)
        residual = hidden if residual is None else residual + 1.0
        pages = jnp.full((hidden.shape[0], 2), self.layer_id, jnp.int32)
        return out, residual, DSAFusedCache(kv=kv, idx=idx, topk=None, topk_pages=pages), None


class _StubPool:
    def __init__(self, n_layers, T):
        self.kv_buffer = [jnp.zeros((T, 1)) for _ in range(n_layers)]
        self.indexer_key_buffer = [jnp.zeros((T, 1)) for _ in range(n_layers)]

    def get_fused_kv_buffer(self, layer_id):
        return self.kv_buffer[layer_id]

    def get_indexer_key_buffer(self, slot):
        return self.indexer_key_buffer[slot]


class PrefillMicrobatchBoundsMethodTest(unittest.TestCase):
    """The gating method reads self.mesh (gp86 10-02: Glm5Model had no mesh attr)."""

    def _model(self):
        import types

        class _Fake:
            pass

        m = _Fake()
        m._prefill_microbatch_bounds = types.MethodType(
            glm5_moe.Glm5Model._prefill_microbatch_bounds, m
        )
        m.mesh = types.SimpleNamespace(shape={"tensor": 8})
        return m

    def test_glm5model_stores_mesh(self):
        import inspect

        self.assertIn("self.mesh = mesh", inspect.getsource(glm5_moe.Glm5Model.__init__))

    def test_bounds_from_forward_batch(self):
        m = self._model()
        fb = _fb(T=4096)
        with mock.patch.object(glm5_moe, "_PREFILL_MICROBATCH", 2):
            self.assertEqual(m._prefill_microbatch_bounds(fb), [(0, 2048), (2048, 4096)])
            fb.forward_mode = ForwardMode.DECODE
            self.assertIsNone(m._prefill_microbatch_bounds(fb))
            fb.forward_mode = ForwardMode.EXTEND
            small = _fb(T=1024)
            self.assertIsNone(m._prefill_microbatch_bounds(small))
        with mock.patch.object(glm5_moe, "_PREFILL_MICROBATCH", 1):
            self.assertIsNone(m._prefill_microbatch_bounds(fb))


class _BlockStubLayer(_StubLayer):
    """Layer exposing attn_block / mlp_block (the split the pinned schedule uses)."""

    def attn_block(
        self,
        positions,
        hidden,
        fb,
        pool,
        residual,
        dsa_topk_in=None,
        dsa_topk_pages_in=None,
        dsa_topk_reuse=False,
    ):
        out, residual, kvf, _ = _StubLayer.__call__(
            self,
            positions,
            hidden,
            fb,
            pool,
            residual,
            dsa_topk_in=dsa_topk_in,
            dsa_topk_pages_in=dsa_topk_pages_in,
        )
        self.log[-1] = ("attn",) + self.log[-1]
        return out, residual, kvf

    def mlp_block(self, hidden, dispatch_info=None):
        self.log.append(("mlp", self.layer_id, float(hidden[0, 0])))
        return hidden + 1.0, None


class PinAfterTest(unittest.TestCase):
    def test_pin_after_is_a_real_zero_add(self):
        x = jnp.ones((4, 3), jnp.bfloat16)
        anchor = jnp.full((1, 1), 7.0, jnp.bfloat16)
        y = glm5_moe._pin_after(x, anchor)
        np.testing.assert_array_equal(np.asarray(y), np.asarray(x))
        self.assertEqual(y.dtype, x.dtype)
        # the dependency must survive tracing: anchor is an input of the jaxpr's add
        import jax

        jaxpr = jax.make_jaxpr(glm5_moe._pin_after)(x, anchor)
        self.assertIn("mul", str(jaxpr))
        self.assertIn("add", str(jaxpr))

    def test_pin_after_nan_inf_anchor_is_still_zero(self):
        x = jnp.ones((4, 3), jnp.bfloat16)
        for v in (float("nan"), float("inf"), -float("inf")):
            anchor = jnp.full((2, 2), v, jnp.float32)
            y = glm5_moe._pin_after(x, anchor)
            np.testing.assert_array_equal(np.asarray(y), np.asarray(x))

    def test_pin_after_nd_int8_anchor(self):
        x = jnp.ones((4, 3), jnp.bfloat16)
        anchor = jnp.full((2, 8, 16, 4), 5, jnp.int8)
        y = glm5_moe._pin_after(x, anchor)
        np.testing.assert_array_equal(np.asarray(y), np.asarray(x))
        self.assertEqual(y.dtype, jnp.bfloat16)

    def test_force_fused_reduce_context(self):
        from sgl_jax.srt.kernels.collectives import oneshot_allreduce as osa

        with mock.patch.object(osa, "_MIN_BYTES", 0):
            self.assertFalse(osa.fused_moe_enabled())
            with osa.force_fused_reduce():
                self.assertTrue(osa.fused_moe_enabled())
            self.assertFalse(osa.fused_moe_enabled())


class _DenseStubLayer(_BlockStubLayer):
    """A dense (non-MoE) layer: the pinned schedule runs mlp_partial / mlp_reduce per window."""

    def mlp_partial(self, hidden, dispatch_info=None):
        self.log.append(("mlp", self.layer_id, float(hidden[0, 0])))
        return hidden + 1.0, None, True  # deferred partial

    def mlp_reduce(self, value, deferred):
        self.log.append(("reduce", self.layer_id, float(value[0, 0])))
        return value


class DenseLayerScheduleTest(unittest.TestCase):
    def test_dense_layer_order_and_anchors(self):
        T = 512
        fb = _fb(T)
        fb.attn_backend.full_slot = {0: 0, 1: 1}
        fb.out_cache_loc = jnp.arange(T, dtype=jnp.int32)
        fb.positions = jnp.arange(T, dtype=jnp.int32)
        log = []
        pins = []

        def fake_pin(x, anchor):
            pins.append(
                ("kv" if anchor.shape[0] == T else "partial", tuple(anchor.shape))
            )  # kv buffer [T,1], partial [256,1]
            return x

        layers = [_DenseStubLayer(i, log) for i in range(2)]
        parts = [glm5_moe._slice_forward_batch(fb, lo, hi) for lo, hi in [(0, 256), (256, 512)]]
        hidden = [p.input_ids[:, None].astype(jnp.float32) for p in parts]
        pool = glm5_moe._MicrobatchPoolView(_StubPool(2, T))
        h, r, kvs, idxs, topk = glm5_moe._microbatch_pinned_layers(
            layers,
            parts,
            hidden,
            [None, None],
            [None, None],
            [None, None],
            pool,
            {0: 0, 1: 1},
            pin=fake_pin,
        )
        kinds = [(e[0], e[1]) for e in log]
        self.assertEqual(
            kinds,
            [
                ("attn", 0),
                ("attn", 0),
                ("mlp", 0),
                ("reduce", 0),
                ("mlp", 0),
                ("reduce", 0),
                ("attn", 1),
                ("attn", 1),
                ("mlp", 1),
                ("reduce", 1),
                ("mlp", 1),
                ("reduce", 1),
            ],
        )
        # layer 0: B.attn pinned on A's kv; A.mlp on B's kv; B.mlp on A's partial (3 pins: hidden only, no residual yet
        # for B.attn -> 1; A.mlp 1; B.mlp 1)
        # layer 1: A.attn (hidden+residual) on B's partial -> 2; B.attn (hidden+residual) on A's kv -> 2; A.mlp 1; B.mlp 1
        self.assertEqual(len(pins), 3 + 6)
        self.assertEqual(
            [k for k, _ in pins],
            ["kv", "kv", "partial", "partial", "partial", "kv", "kv", "kv", "partial"],
        )
        self.assertEqual(len(kvs), 2)
        np.testing.assert_array_equal(np.asarray(kvs[0][:, 0]), np.arange(T) + 1.0)


class _MoeStubLayer(_DenseStubLayer):
    is_moe_layer = True

    def fused_reduce_supported(self):
        # EPMoE with deferrable shared experts on a tensor-only mesh: the two-stage path applies
        return True

    def mlp_dispatch(self, hidden, dispatch_info=None):
        self.log.append(("dispatch", self.layer_id, float(hidden[0, 0])))
        anchor = (hidden[:1, :1] * 0 + 7.0).astype(jnp.float32)  # [1,1], identifiable
        return dict(state=hidden, anchor=anchor, topk_weights=None, topk_ids=None, shared=1.0)

    def mlp_experts(self, d, combine_pin=None):
        self.log.append(
            ("experts", self.layer_id, None if combine_pin is None else float(combine_pin[0, 0]))
        )
        return d["state"] + d["shared"], None, True


class TwoStageScheduleTest(unittest.TestCase):
    def test_moe_layer_order_and_stream_pins(self):
        T = 512
        fb = _fb(T)
        fb.attn_backend.full_slot = {0: 0, 1: 1}
        fb.out_cache_loc = jnp.arange(T, dtype=jnp.int32)
        fb.positions = jnp.arange(T, dtype=jnp.int32)
        log = []
        pins = []

        def fake_pin(x, anchor):
            if anchor.shape == (1, 1) and float(anchor[0, 0]) == 7.0:
                kind = "dispatch-anchor"
            elif anchor.shape[0] == T:
                kind = "kv"
            else:
                kind = "partial"
            pins.append(kind)
            return x

        layers = [_MoeStubLayer(i, log) for i in range(2)]
        parts = [glm5_moe._slice_forward_batch(fb, lo, hi) for lo, hi in [(0, 256), (256, 512)]]
        hidden = [p.input_ids[:, None].astype(jnp.float32) for p in parts]
        pool = glm5_moe._MicrobatchPoolView(_StubPool(2, T))
        h, r, kvs, idxs, topk = glm5_moe._microbatch_pinned_layers(
            layers,
            parts,
            hidden,
            [None, None],
            [None, None],
            [None, None],
            pool,
            {0: 0, 1: 1},
            pin=fake_pin,
        )
        kinds = [(e[0], e[1]) for e in log]
        per_layer = [
            ("attn", 0),
            ("attn", 0),
            ("dispatch", 0),  # A
            ("experts", 0),  # A (no combine pin)
            ("dispatch", 0),  # B: its SparseCore dispatch gather is traced before A's all-reduce
            ("reduce", 0),  # A's all-reduce (operand pinned behind B's dispatch anchor)
            ("experts", 0),  # B, combine pinned behind A's post-all-reduce value
            ("reduce", 0),  # B's all-reduce
        ]
        self.assertEqual(kinds[:8], per_layer)
        self.assertEqual(kinds[8:], [(k, 1) for k, _ in per_layer])
        # B's experts carry a combine pin, A's do not
        experts = [e for e in log if e[0] == "experts"]
        self.assertEqual([e[2] is None for e in experts], [True, False, True, False])
        # pins per layer: B.attn behind A's kv, A.mlp behind B's kv, B.mlp behind A's partial,
        # A's all-reduce operand behind B's dispatch anchor; layer 1 adds the attention pins
        # (hidden + residual) on the previous layer's partial / kv.
        self.assertEqual(pins.count("dispatch-anchor"), 2)
        self.assertEqual(pins[:4], ["kv", "kv", "partial", "dispatch-anchor"])
        self.assertEqual(len(kvs), 2)
        # values: each window's rows go through attn (+1) and mlp (+1 shared) per layer, unchanged by the pins
        np.testing.assert_array_equal(np.asarray(kvs[0][:, 0]), np.arange(T) + 1.0)

    def test_dense_layers_fall_back_to_per_window_mlp(self):
        T = 512
        fb = _fb(T)
        fb.attn_backend.full_slot = {0: 0}
        fb.out_cache_loc = jnp.arange(T, dtype=jnp.int32)
        fb.positions = jnp.arange(T, dtype=jnp.int32)
        log = []
        layers = [_DenseStubLayer(0, log)]  # no is_moe_layer attribute -> dense path
        parts = [glm5_moe._slice_forward_batch(fb, lo, hi) for lo, hi in [(0, 256), (256, 512)]]
        hidden = [p.input_ids[:, None].astype(jnp.float32) for p in parts]
        pool = glm5_moe._MicrobatchPoolView(_StubPool(1, T))
        glm5_moe._microbatch_pinned_layers(
            layers,
            parts,
            hidden,
            [None, None],
            [None, None],
            [None, None],
            pool,
            {0: 0},
            pin=lambda x, a: x,
        )
        self.assertEqual(
            [(e[0], e[1]) for e in log],
            [("attn", 0), ("attn", 0), ("mlp", 0), ("reduce", 0), ("mlp", 0), ("reduce", 0)],
        )


class CallMicrobatchedTest(unittest.TestCase):
    def _model(self, log, n_layers=2):
        import types

        class _Fake:  # nnx.Module needs its pytree state; borrow the unbound method instead
            pass

        m = _Fake()
        m._call_microbatched = types.MethodType(glm5_moe.Glm5Model._call_microbatched, m)
        m.layers = [_MoeStubLayer(i, log) for i in range(n_layers)]
        m.embed_tokens = lambda ids: ids[:, None].astype(jnp.float32)
        m.norm = lambda h: h
        return m

    def test_windows_run_in_layer_major_order_and_thread_kv(self):
        T = 512
        fb = _fb(T)
        fb.attn_backend.full_slot = {0: 0, 1: 1}
        fb.out_cache_loc = jnp.arange(T, dtype=jnp.int32)
        fb.positions = jnp.arange(T, dtype=jnp.int32)
        log = []
        model = self._model(log)
        pool = _StubPool(2, T)
        hidden, kv_layers, idx_layers, topk = model._call_microbatched(
            fb, pool, [(0, 256), (256, 512)]
        )
        attn = [e for e in log if e[0] == "attn"]
        # layer-major: L0 wA, L0 wB, L1 wA, L1 wB
        self.assertEqual([(e[1], e[2]) for e in attn], [(0, 0), (0, 256), (1, 0), (1, 256)])
        # window B of layer 0 read the buffer window A rewrote (row 0 == ids[0]+1 == 1.0),
        # window A of layer 1 read the pristine layer-1 buffer (0.0 before its own write),
        # window B of layer 1 read A's layer-1 write (row 0 == layer-0 output 1.0, +1)
        self.assertEqual([e[3] for e in attn], [0.0, 1.0, 0.0, 2.0])
        # topk_pages thread is per window: layer 1 window B sees layer 0's pages of ITS rows
        self.assertEqual(int(attn[3][4][0, 0]), 0)
        self.assertEqual(attn[3][4].shape[0], 256)
        # per-layer result = last window's buffer, containing both windows' writes
        self.assertEqual(len(kv_layers), 2)
        np.testing.assert_array_equal(np.asarray(kv_layers[0][:, 0]), np.arange(T) + 1.0)
        np.testing.assert_array_equal(np.asarray(idx_layers[1][:, 0]), np.arange(T))
        # hidden states are in token order and equal the single-pass stub math (pins add 0)
        ref = jnp.arange(T, dtype=jnp.float32)[:, None]
        h = ref
        r = None
        for layer_id in range(2):
            out = h * 2.0 + layer_id
            r = h if r is None else r + 1.0
            h = out + 1.0
        np.testing.assert_allclose(np.asarray(hidden), np.asarray(h + r))
        self.assertEqual(topk, [None, None])


class SchedGroupTest(unittest.TestCase):
    def test_sets_group_for_window_a_attention_and_tags_b_reduce(self):
        from sgl_jax.srt.layers.attention import dsa_sparse_backend as backend

        T = 512
        fb = _fb(T)
        fb.attn_backend.full_slot = {0: 0, 1: 1}
        fb.out_cache_loc = jnp.arange(T, dtype=jnp.int32)
        fb.positions = jnp.arange(T, dtype=jnp.int32)
        log = []
        seen_groups = []

        class _Stub(_MoeStubLayer):
            def attn_block(self, *a, **k):
                seen_groups.append(backend.PREFILL_SCHED_GROUP)
                return super().attn_block(*a, **k)

        layers = [_Stub(i, log) for i in range(3)]
        parts = [glm5_moe._slice_forward_batch(fb, lo, hi) for lo, hi in [(0, 256), (256, 512)]]
        hidden = [p.input_ids[:, None].astype(jnp.float32) for p in parts]
        pool = glm5_moe._MicrobatchPoolView(_StubPool(3, T))
        glm5_moe._microbatch_pinned_layers(
            layers,
            parts,
            hidden,
            [None, None],
            [None, None],
            [None, None],
            pool,
            {0: 0, 1: 1, 2: 2},
            pin=lambda x, a: x,
        )
        # per layer: window A then window B attention; A of layer 1/2 carries group 1001/1002, B never
        self.assertEqual(seen_groups, [None, None, 1001, None, 1002, None])
        self.assertIsNone(backend.PREFILL_SCHED_GROUP)
        kinds = [(e[0], e[1]) for e in log]
        self.assertEqual(
            kinds[:8],
            [
                ("attn", 0),
                ("attn", 0),
                ("dispatch", 0),
                ("experts", 0),
                ("dispatch", 0),
                ("reduce", 0),
                ("experts", 0),
                ("reduce", 0),
            ],
        )

    def test_sched_tagged_wrapper_adds_xla_metadata(self):
        import jax as _jax

        from sgl_jax.srt.layers.attention import dsa_sparse_backend as backend

        x = jnp.ones((4,))
        # separate function objects: jax caches traces per function, and the tag is read at trace time
        f0 = backend._sched_tagged(lambda x: x * 2.0)
        self.assertNotIn("_scheduling_group_id", _jax.jit(f0).lower(x).as_text())
        backend.PREFILL_SCHED_GROUP = 1234
        try:
            f1 = backend._sched_tagged(lambda x: x * 3.0)
            txt = (
                _jax.jit(f1).lower(x).as_text()
            )  # context form lands as mhlo.frontend_attributes on the ops
        finally:
            backend.PREFILL_SCHED_GROUP = None
        self.assertIn("_scheduling_group_id", txt)
        self.assertIn("1234", txt)


if __name__ == "__main__":
    unittest.main()


class TwoStageFallbackTest(unittest.TestCase):
    def test_moe_layer_without_two_stage_support_uses_mlp_partial(self):
        """A MoE layer whose fused_reduce_supported() is False (fused backend, no shared
        experts, data axis > 1, non-deferrable shared down_proj) takes the per-window
        mlp_partial / mlp_reduce path instead of mlp_dispatch."""
        T = 512
        fb = _fb(T)
        fb.attn_backend.full_slot = {0: 0}
        fb.out_cache_loc = jnp.arange(T, dtype=jnp.int32)
        fb.positions = jnp.arange(T, dtype=jnp.int32)
        log = []

        class _NoTwoStage(_MoeStubLayer):
            def fused_reduce_supported(self):
                return False

            def mlp_dispatch(self, hidden, dispatch_info=None):
                raise AssertionError("mlp_dispatch must not be called on this layer")

        layers = [_NoTwoStage(0, log)]
        parts = [glm5_moe._slice_forward_batch(fb, lo, hi) for lo, hi in [(0, 256), (256, 512)]]
        hidden = [p.input_ids[:, None].astype(jnp.float32) for p in parts]
        pool = glm5_moe._MicrobatchPoolView(_StubPool(1, T))
        glm5_moe._microbatch_pinned_layers(
            layers,
            parts,
            hidden,
            [None, None],
            [None, None],
            [None, None],
            pool,
            {0: 0},
            pin=lambda x, a: x,
        )
        kinds = [e[0] for e in log]
        self.assertEqual(kinds.count("mlp"), 2, log)
        self.assertEqual(kinds.count("reduce"), 2, log)
        self.assertNotIn("dispatch", kinds, log)
