"""Tests for HybridLinearAttnBackend.

Maps to design doc test plan §4:
- TP-1: TestGetForwardMetadata.test_aggregates
- TP-2: TestForwardMetadataSetter.*
- TP-3: TestDispatch.*
- TP-4: TestGetMaxRunningReqests.*
- TP-5: TestModelRunnerIntegration (Task 3.2)
- TP-6: TestTwoLayerForwardOrdering (Task 3.3)
"""

from dataclasses import dataclass, field

import jax
from flax import nnx

from sgl_jax.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    AttentionBackendMetadata,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttentionBackendMetadata,
    HybridLinearAttnBackend,
    attn_backend_wrapper,
)


# ---------------------------------------------------------------------------
# Test fixtures: fake sub-backends + fake metadata + fake layer / batch
# ---------------------------------------------------------------------------


@dataclass
class _FakeFullMetadata(AttentionBackendMetadata):
    tag: str = "full"

    def tree_flatten(self):
        return (), self.tag

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(tag=aux_data)


@dataclass
class _FakeLinearMetadata:
    tag: str = "linear"


class _FakeFullBackend(AttentionBackend):
    """Records calls so dispatch tests can assert against them."""

    def __init__(self, max_running: int = 100):
        self.max_running = max_running
        self.forward_metadata = _FakeFullMetadata(tag="initial-full")
        self.calls = []

    def get_forward_metadata(self, batch):
        return _FakeFullMetadata(tag=f"full-from-{getattr(batch, 'name', 'batch')}")

    def __call__(self, *args, layer=None, forward_batch=None, **kwargs):
        self.calls.append({
            "layer_id": getattr(layer, "layer_id", None),
            "args": args,
            "kwargs": kwargs,
        })
        return ("full-out", layer.layer_id if layer is not None else None)

    def get_max_running_reqests(self, max_context_len, page_size):
        return self.max_running


class _FakeLinearBackend(nnx.Module):
    """Mocks the (not-yet-merged) LinearRecurrentAttentionBackend protocol."""

    def __init__(self, max_running: int = 100):
        self.max_running = max_running
        self.forward_metadata = _FakeLinearMetadata(tag="initial-linear")
        self.calls = []

    def get_forward_metadata(self, batch):
        return _FakeLinearMetadata(tag=f"linear-from-{getattr(batch, 'name', 'batch')}")

    def __call__(self, *args, layer=None, forward_batch=None, **kwargs):
        self.calls.append({
            "layer_id": getattr(layer, "layer_id", None),
            "args": args,
            "kwargs": kwargs,
        })
        return ("linear-out", layer.layer_id if layer is not None else None)

    def get_max_running_reqests(self, max_context_len, page_size):
        return self.max_running


@dataclass
class _FakeLayer:
    layer_id: int


@dataclass
class _FakeBatch:
    name: str = "batch"


def _make_hybrid(full_layers=(0, 2), full_max=100, linear_max=100):
    full = _FakeFullBackend(max_running=full_max)
    linear = _FakeLinearBackend(max_running=linear_max)
    hybrid = HybridLinearAttnBackend(full, linear, list(full_layers))
    return hybrid, full, linear


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_module_imports():
    assert HybridLinearAttnBackend is not None
    assert HybridLinearAttentionBackendMetadata is not None
    assert callable(attn_backend_wrapper)


def test_constructor_records_full_attn_layers_as_frozenset():
    hybrid, _, _ = _make_hybrid(full_layers=[0, 2, 2])
    assert hybrid.full_attn_layers == frozenset({0, 2})


# ---------------------------------------------------------------------------
# TP-1: get_forward_metadata aggregates sub-backend metadata
# ---------------------------------------------------------------------------


class TestGetForwardMetadata:
    def test_aggregates_sub_backend_metadata(self):
        hybrid, full, linear = _make_hybrid()
        batch = _FakeBatch(name="b1")

        meta = hybrid.get_forward_metadata(batch)

        assert isinstance(meta, HybridLinearAttentionBackendMetadata)
        assert meta.full_attn_metadata == full.get_forward_metadata(batch)
        assert meta.linear_attn_metadata == linear.get_forward_metadata(batch)

    def test_each_sub_backend_called_with_same_batch(self):
        hybrid, full, linear = _make_hybrid()
        batch = _FakeBatch(name="step42")
        meta = hybrid.get_forward_metadata(batch)
        # Both fields should reflect the batch name.
        assert meta.full_attn_metadata.tag == "full-from-step42"
        assert meta.linear_attn_metadata.tag == "linear-from-step42"


# ---------------------------------------------------------------------------
# TP-2: forward_metadata setter unpacks dataclass fields to sub-backends
# ---------------------------------------------------------------------------


class TestForwardMetadataSetter:
    def test_unpacks_to_sub_backends(self):
        hybrid, full, linear = _make_hybrid()

        fm = _FakeFullMetadata(tag="injected-full")
        lm = _FakeLinearMetadata(tag="injected-linear")
        value = HybridLinearAttentionBackendMetadata(
            full_attn_metadata=fm, linear_attn_metadata=lm,
        )

        hybrid.forward_metadata = value

        assert full.forward_metadata is fm
        assert linear.forward_metadata is lm
        # Hybrid keeps the aggregate (used by pytree traversal).
        assert hybrid.forward_metadata is value

    def test_setter_overwrites_previous_value(self):
        hybrid, full, linear = _make_hybrid()
        v1 = HybridLinearAttentionBackendMetadata(
            full_attn_metadata=_FakeFullMetadata(tag="a"),
            linear_attn_metadata=_FakeLinearMetadata(tag="a"),
        )
        v2 = HybridLinearAttentionBackendMetadata(
            full_attn_metadata=_FakeFullMetadata(tag="b"),
            linear_attn_metadata=_FakeLinearMetadata(tag="b"),
        )
        hybrid.forward_metadata = v1
        hybrid.forward_metadata = v2
        assert full.forward_metadata.tag == "b"
        assert linear.forward_metadata.tag == "b"


# ---------------------------------------------------------------------------
# TP-3: __call__ dispatches by layer.layer_id
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_full_attn_layer_routes_to_full_sub(self):
        hybrid, full, linear = _make_hybrid(full_layers=(0, 2))

        out = hybrid(layer=_FakeLayer(layer_id=0), forward_batch=_FakeBatch())

        assert out[0] == "full-out"
        assert len(full.calls) == 1
        assert len(linear.calls) == 0
        assert full.calls[0]["layer_id"] == 0

    def test_kda_layer_routes_to_linear_sub(self):
        hybrid, full, linear = _make_hybrid(full_layers=(0, 2))

        out = hybrid(layer=_FakeLayer(layer_id=1), forward_batch=_FakeBatch())

        assert out[0] == "linear-out"
        assert len(linear.calls) == 1
        assert len(full.calls) == 0
        assert linear.calls[0]["layer_id"] == 1

    def test_kwargs_passthrough(self):
        hybrid, full, linear = _make_hybrid()
        sentinel = object()
        hybrid(
            layer=_FakeLayer(layer_id=0),
            forward_batch=_FakeBatch(),
            extra_kw=sentinel,
        )
        assert full.calls[0]["kwargs"]["extra_kw"] is sentinel

    def test_assert_layer_required(self):
        hybrid, _, _ = _make_hybrid()
        try:
            hybrid(forward_batch=_FakeBatch())
        except AssertionError:
            return
        raise AssertionError("expected AssertionError when layer= is omitted")


class TestPytreeRoundtrip:
    def test_flatten_unflatten_preserves_state(self):
        hybrid, _, _ = _make_hybrid(full_layers=(0, 2))
        leaves, treedef = jax.tree_util.tree_flatten(hybrid)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt.full_attn_layers == hybrid.full_attn_layers
        assert isinstance(
            rebuilt._forward_metadata, HybridLinearAttentionBackendMetadata
        )


# ---------------------------------------------------------------------------
# TP-4: get_max_running_reqests returns min of sub-backends
# ---------------------------------------------------------------------------


class TestGetMaxRunningReqests:
    def test_returns_min_of_subs(self):
        hybrid, _, _ = _make_hybrid(full_max=37, linear_max=99)
        assert hybrid.get_max_running_reqests(1024, 16) == 37

    def test_returns_min_when_linear_smaller(self):
        hybrid, _, _ = _make_hybrid(full_max=99, linear_max=10)
        assert hybrid.get_max_running_reqests(2048, 8) == 10


# ---------------------------------------------------------------------------
# attn_backend_wrapper helper
# ---------------------------------------------------------------------------


class TestAttnBackendWrapper:
    def test_passthrough_when_kimi_linear_config_is_none(self):
        from types import SimpleNamespace

        runner = SimpleNamespace(kimi_linear_config=None)
        full = _FakeFullBackend()
        result = attn_backend_wrapper(runner, full)
        assert result is full  # identity — unchanged

    def test_raises_when_kda_backend_module_missing(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        cfg = SimpleNamespace(full_attention_layer_ids=[0, 2])
        runner = SimpleNamespace(kimi_linear_config=cfg)

        # Force the lazy import to fail.
        monkeypatch.setitem(
            sys.modules,
            "sgl_jax.srt.layers.attention.linear.kda_backend",
            None,
        )

        try:
            attn_backend_wrapper(runner, _FakeFullBackend())
        except ImportError as e:
            assert "KDAAttnBackend" in str(e)
            return
        raise AssertionError("expected ImportError")

    def test_builds_hybrid_with_full_attn_layers_from_config(self, monkeypatch):
        import sys
        from types import ModuleType, SimpleNamespace

        # Inject fake kda_backend module so the lazy import succeeds.
        fake_module = ModuleType(
            "sgl_jax.srt.layers.attention.linear.kda_backend"
        )

        class _FakeKDA(nnx.Module):
            def __init__(self, runner):
                self.runner = runner

        fake_module.KDAAttnBackend = _FakeKDA
        monkeypatch.setitem(
            sys.modules,
            "sgl_jax.srt.layers.attention.linear.kda_backend",
            fake_module,
        )

        cfg = SimpleNamespace(full_attention_layer_ids=[0, 2])
        runner = SimpleNamespace(kimi_linear_config=cfg)
        full = _FakeFullBackend()

        result = attn_backend_wrapper(runner, full)

        assert isinstance(result, HybridLinearAttnBackend)
        assert result.full_attn_backend is full
        assert isinstance(result.linear_attn_backend, _FakeKDA)
        assert result.full_attn_layers == frozenset({0, 2})


# ---------------------------------------------------------------------------
# TP-5: ModelRunner._get_attention_backend wraps in HybridLinearAttnBackend
# when kimi_linear_config is set.
# ---------------------------------------------------------------------------


class TestModelRunnerIntegration:
    def test_get_attention_backend_wraps_in_hybrid(self, monkeypatch):
        import sys
        from types import ModuleType, SimpleNamespace

        import jax
        from jax.sharding import Mesh

        from sgl_jax.srt.layers.attention.native_backend import NativeAttention
        from sgl_jax.srt.model_executor.model_runner import ModelRunner

        # Inject fake kda_backend module so attn_backend_wrapper's lazy import succeeds.
        fake_module = ModuleType(
            "sgl_jax.srt.layers.attention.linear.kda_backend"
        )

        class _FakeKDA(nnx.Module):
            def __init__(self, runner):
                self.runner = runner

        fake_module.KDAAttnBackend = _FakeKDA
        monkeypatch.setitem(
            sys.modules,
            "sgl_jax.srt.layers.attention.linear.kda_backend",
            fake_module,
        )

        # Single-device mesh required by NativeAttention's NamedSharding(...).
        mesh = Mesh(jax.devices()[:1], axis_names=("tensor",))

        cfg = SimpleNamespace(full_attention_layer_ids=[0, 2])
        runner = SimpleNamespace(
            server_args=SimpleNamespace(attention_backend="native", device="cpu"),
            num_attn_heads=4,
            num_kv_heads=4,
            model_config=SimpleNamespace(head_dim=64),
            page_size=1,
            mesh=mesh,
            kimi_linear_config=cfg,
        )

        # Call _get_attention_backend on the SimpleNamespace stand-in.
        backend = ModelRunner._get_attention_backend(runner)

        assert isinstance(backend, HybridLinearAttnBackend)
        assert isinstance(backend.linear_attn_backend, _FakeKDA)
        # full_attn_backend should be NativeAttention since we asked for "native".
        assert isinstance(backend.full_attn_backend, NativeAttention)
        assert backend.full_attn_layers == frozenset({0, 2})
