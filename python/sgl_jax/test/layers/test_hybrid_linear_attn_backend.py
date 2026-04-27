"""Tests for HybridLinearAttnBackend.

Maps to design doc test plan §4:
- TP-1: TestGetForwardMetadata.test_aggregates
- TP-2: TestForwardMetadataSetter.*
- TP-3: TestDispatch.*
- TP-4: TestGetMaxRunningReqests.*
- TP-5: TestModelRunnerIntegration
- TP-6: TestTwoLayerForwardOrdering
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    AttentionBackendMetadata,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    HybridLinearAttnBackendMetadata,
    LinearRecurrentAttnBackend,
    LinearRecurrentAttnBackendMetadata,
    attn_backend_wrapper,
)

# ---------------------------------------------------------------------------
# Test fixtures: fake sub-backends + fake metadata + fake layer / batch
# ---------------------------------------------------------------------------


@dataclass
class _FakeFullMetadata(AttentionBackendMetadata):
    tag: str = "full"

    def tree_flatten(self):
        return (), {"tag": self.tag}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(tag=aux_data["tag"])


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

    def __call__(self, q, k, v, layer=None, forward_batch=None, token_to_kv_pool=None, **kwargs):
        self.calls.append(
            {
                "layer_id": getattr(layer, "layer_id", None),
                "q": q,
                "k": k,
                "v": v,
                "forward_batch": forward_batch,
                "token_to_kv_pool": token_to_kv_pool,
                "kwargs": kwargs,
            }
        )
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

    def __call__(
        self,
        q,
        k,
        v,
        layer=None,
        forward_batch=None,
        mixed_qkv=None,
        a=None,
        b=None,
        recurrent_state_pool=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "layer_id": getattr(layer, "layer_id", None),
                "q": q,
                "k": k,
                "v": v,
                "forward_batch": forward_batch,
                "mixed_qkv": mixed_qkv,
                "a": a,
                "b": b,
                "recurrent_state_pool": recurrent_state_pool,
                "kwargs": kwargs,
            }
        )
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


def _qkv():
    """Tiny stand-in arrays for the q/k/v positional params."""
    q = jnp.zeros((1, 1, 1))
    k = jnp.zeros((1, 1, 1))
    v = jnp.zeros((1, 1, 1))
    return q, k, v


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_module_imports():
    assert HybridLinearAttnBackend is not None
    assert HybridLinearAttnBackendMetadata is not None
    assert LinearRecurrentAttnBackend is not None
    assert LinearRecurrentAttnBackendMetadata is not None
    assert callable(attn_backend_wrapper)


def test_constructor_records_full_attn_layers_as_frozenset():
    hybrid, _, _ = _make_hybrid(full_layers=[0, 2, 2])
    assert hybrid.full_attn_layers == frozenset({0, 2})


# ---------------------------------------------------------------------------
# Stub linear-recurrent classes are pytree-registered
# ---------------------------------------------------------------------------


class TestLinearRecurrentStubsArePytrees:
    def test_metadata_pytree_roundtrip(self):
        m = LinearRecurrentAttnBackendMetadata()
        leaves, treedef = jax.tree_util.tree_flatten(m)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, LinearRecurrentAttnBackendMetadata)

    def test_metadata_inside_jit(self):
        @jax.jit
        def identity(x):
            return x

        m = LinearRecurrentAttnBackendMetadata()
        out = identity(m)
        assert isinstance(out, LinearRecurrentAttnBackendMetadata)

    def test_backend_is_pytree(self):
        # Inherits AttentionBackend (nnx.Module) → registered automatically by
        # flax-nnx's metaclass. Smoke-check by flattening / unflattening.
        b = LinearRecurrentAttnBackend()
        leaves, treedef = jax.tree_util.tree_flatten(b)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, LinearRecurrentAttnBackend)


# ---------------------------------------------------------------------------
# TP-1: get_forward_metadata aggregates sub-backend metadata
# ---------------------------------------------------------------------------


class TestGetForwardMetadata:
    def test_aggregates_sub_backend_metadata(self):
        hybrid, full, linear = _make_hybrid()
        batch = _FakeBatch(name="b1")

        meta = hybrid.get_forward_metadata(batch)

        assert isinstance(meta, HybridLinearAttnBackendMetadata)
        assert meta.full_attn_metadata == full.get_forward_metadata(batch)
        assert meta.linear_attn_metadata == linear.get_forward_metadata(batch)

    def test_each_sub_backend_called_with_same_batch(self):
        hybrid, full, linear = _make_hybrid()
        batch = _FakeBatch(name="step42")
        meta = hybrid.get_forward_metadata(batch)
        assert meta.full_attn_metadata.tag == "full-from-step42"
        assert meta.linear_attn_metadata.tag == "linear-from-step42"

    def test_metadata_pytree_aux_is_empty_dict(self):
        # tree_flatten on HybridLinearAttnBackendMetadata returns ((...), {})
        meta = HybridLinearAttnBackendMetadata(
            full_attn_metadata=_FakeFullMetadata(tag="x"),
            linear_attn_metadata=_FakeLinearMetadata(tag="y"),
        )
        children, aux = meta.tree_flatten()
        assert aux == {}
        assert len(children) == 2


# ---------------------------------------------------------------------------
# TP-2: forward_metadata setter unpacks dataclass fields to sub-backends
# ---------------------------------------------------------------------------


class TestForwardMetadataSetter:
    def test_unpacks_to_sub_backends(self):
        hybrid, full, linear = _make_hybrid()

        fm = _FakeFullMetadata(tag="injected-full")
        lm = _FakeLinearMetadata(tag="injected-linear")
        value = HybridLinearAttnBackendMetadata(
            full_attn_metadata=fm,
            linear_attn_metadata=lm,
        )

        hybrid.forward_metadata = value

        assert full.forward_metadata is fm
        assert linear.forward_metadata is lm
        assert hybrid.forward_metadata is value

    def test_setter_overwrites_previous_value(self):
        hybrid, full, linear = _make_hybrid()
        v1 = HybridLinearAttnBackendMetadata(
            full_attn_metadata=_FakeFullMetadata(tag="a"),
            linear_attn_metadata=_FakeLinearMetadata(tag="a"),
        )
        v2 = HybridLinearAttnBackendMetadata(
            full_attn_metadata=_FakeFullMetadata(tag="b"),
            linear_attn_metadata=_FakeLinearMetadata(tag="b"),
        )
        hybrid.forward_metadata = v1
        hybrid.forward_metadata = v2
        assert full.forward_metadata.tag == "b"
        assert linear.forward_metadata.tag == "b"


# ---------------------------------------------------------------------------
# TP-3: __call__ dispatches by layer.layer_id with the new q/k/v/pool signature
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_full_attn_layer_routes_to_full_sub_without_linear_only_args(self):
        hybrid, full, linear = _make_hybrid(full_layers=(0, 2))
        q, k, v = _qkv()
        pool = object()

        out = hybrid(
            q,
            k,
            v,
            layer=_FakeLayer(layer_id=0),
            forward_batch=_FakeBatch(),
            pool=pool,
            mixed_qkv=jnp.ones((1, 1)),  # provided but should be IGNORED for full
            a=jnp.ones((1, 1)),
            b=jnp.ones((1, 1)),
        )

        assert out[0] == "full-out"
        assert len(full.calls) == 1
        assert len(linear.calls) == 0
        c = full.calls[0]
        assert c["layer_id"] == 0
        assert c["token_to_kv_pool"] is pool
        # Full sub-backend signature does not name mixed_qkv / a / b — they
        # must NOT have been forwarded.
        assert "mixed_qkv" not in c["kwargs"]
        assert "a" not in c["kwargs"]
        assert "b" not in c["kwargs"]
        # And the linear-only kwarg name `recurrent_state_pool` must NOT leak.
        assert "recurrent_state_pool" not in c["kwargs"]

    def test_kda_layer_routes_to_linear_sub_with_linear_only_args(self):
        hybrid, full, linear = _make_hybrid(full_layers=(0, 2))
        q, k, v = _qkv()
        pool = object()
        mixed_qkv = jnp.full((1, 1), 7.0)
        a = jnp.full((1, 1), 8.0)
        b = jnp.full((1, 1), 9.0)

        out = hybrid(
            q,
            k,
            v,
            layer=_FakeLayer(layer_id=1),
            forward_batch=_FakeBatch(),
            pool=pool,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )

        assert out[0] == "linear-out"
        assert len(linear.calls) == 1
        assert len(full.calls) == 0
        c = linear.calls[0]
        assert c["layer_id"] == 1
        assert c["recurrent_state_pool"] is pool
        assert c["mixed_qkv"] is mixed_qkv
        assert c["a"] is a
        assert c["b"] is b
        # And the full-only kwarg name `token_to_kv_pool` must NOT leak.
        assert "token_to_kv_pool" not in c["kwargs"]

    def test_kwargs_passthrough_to_full_sub(self):
        hybrid, full, _ = _make_hybrid()
        q, k, v = _qkv()
        sentinel = object()

        hybrid(
            q,
            k,
            v,
            layer=_FakeLayer(layer_id=0),
            forward_batch=_FakeBatch(),
            pool=None,
            extra_kw=sentinel,
        )
        assert full.calls[0]["kwargs"]["extra_kw"] is sentinel

    def test_kwargs_passthrough_to_linear_sub(self):
        hybrid, _, linear = _make_hybrid()
        q, k, v = _qkv()
        sentinel = object()

        hybrid(
            q,
            k,
            v,
            layer=_FakeLayer(layer_id=1),
            forward_batch=_FakeBatch(),
            pool=None,
            extra_kw=sentinel,
        )
        assert linear.calls[0]["kwargs"]["extra_kw"] is sentinel


class TestPytreeRoundtrip:
    def test_flatten_unflatten_preserves_state(self):
        hybrid, _, _ = _make_hybrid(full_layers=(0, 2))
        leaves, treedef = jax.tree_util.tree_flatten(hybrid)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt.full_attn_layers == hybrid.full_attn_layers
        assert isinstance(rebuilt._forward_metadata, HybridLinearAttnBackendMetadata)


# ---------------------------------------------------------------------------
# TP-4: get_max_running_reqests delegates to full_attn_backend
# ---------------------------------------------------------------------------


class TestGetMaxRunningReqests:
    def test_returns_full_attn_backend_value(self):
        hybrid, _, _ = _make_hybrid(full_max=37, linear_max=99)
        assert hybrid.get_max_running_reqests(1024, 16) == 37

    def test_ignores_linear_attn_backend_value(self):
        hybrid, _, _ = _make_hybrid(full_max=99, linear_max=10)
        assert hybrid.get_max_running_reqests(2048, 8) == 99


# ---------------------------------------------------------------------------
# attn_backend_wrapper helper — branches on runner.linear_recurrent_config
# ---------------------------------------------------------------------------


class TestAttnBackendWrapper:
    def test_passthrough_when_linear_recurrent_config_is_none(self):
        from types import SimpleNamespace

        runner = SimpleNamespace(linear_recurrent_config=None, kimi_linear_config=None)
        full = _FakeFullBackend()
        result = attn_backend_wrapper(runner, full)
        assert result is full  # identity — unchanged

    def test_raises_when_kda_backend_module_missing(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        cfg = SimpleNamespace(full_attention_layer_ids=[0, 2])
        runner = SimpleNamespace(linear_recurrent_config=cfg, kimi_linear_config=cfg)

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

        fake_module = ModuleType("sgl_jax.srt.layers.attention.linear.kda_backend")

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
        runner = SimpleNamespace(linear_recurrent_config=cfg, kimi_linear_config=cfg)
        full = _FakeFullBackend()

        result = attn_backend_wrapper(runner, full)

        assert isinstance(result, HybridLinearAttnBackend)
        assert result.full_attn_backend is full
        assert isinstance(result.linear_attn_backend, _FakeKDA)
        assert result.full_attn_layers == frozenset({0, 2})

    def test_raises_not_implemented_for_unwired_hybrid_config(self):
        # linear_recurrent_config set (umbrella detector says "hybrid"), but
        # no concrete config wired (kimi_linear_config is None) → wrapper must
        # surface a NotImplementedError instead of silently doing the wrong thing.
        from types import SimpleNamespace

        cfg = SimpleNamespace(full_attention_layer_ids=[0])
        runner = SimpleNamespace(linear_recurrent_config=cfg, kimi_linear_config=None)
        try:
            attn_backend_wrapper(runner, _FakeFullBackend())
        except NotImplementedError as e:
            assert "SimpleNamespace" in str(e)  # uses type(cfg).__name__
            return
        raise AssertionError("expected NotImplementedError")


# ---------------------------------------------------------------------------
# TP-5: ModelRunner._get_attention_backend wraps in HybridLinearAttnBackend
# when runner.linear_recurrent_config is set.
# ---------------------------------------------------------------------------


class TestModelRunnerIntegration:
    def test_get_attention_backend_wraps_in_hybrid(self, monkeypatch):
        import sys
        from types import ModuleType, SimpleNamespace

        from jax.sharding import Mesh

        from sgl_jax.srt.layers.attention.native_backend import NativeAttention
        from sgl_jax.srt.model_executor.model_runner import ModelRunner

        fake_module = ModuleType("sgl_jax.srt.layers.attention.linear.kda_backend")

        class _FakeKDA(nnx.Module):
            def __init__(self, runner):
                self.runner = runner

        fake_module.KDAAttnBackend = _FakeKDA
        monkeypatch.setitem(
            sys.modules,
            "sgl_jax.srt.layers.attention.linear.kda_backend",
            fake_module,
        )

        mesh = Mesh(jax.devices()[:1], axis_names=("tensor",))

        cfg = SimpleNamespace(full_attention_layer_ids=[0, 2])
        runner = SimpleNamespace(
            server_args=SimpleNamespace(attention_backend="native", device="cpu"),
            num_attn_heads=4,
            num_kv_heads=4,
            model_config=SimpleNamespace(head_dim=64),
            page_size=1,
            mesh=mesh,
            # The wrapper now reads both `linear_recurrent_config` (umbrella
            # hybrid detector) and `kimi_linear_config` (concrete dispatch).
            linear_recurrent_config=cfg,
            kimi_linear_config=cfg,
        )

        backend = ModelRunner._get_attention_backend(runner)

        assert isinstance(backend, HybridLinearAttnBackend)
        assert isinstance(backend.linear_attn_backend, _FakeKDA)
        assert isinstance(backend.full_attn_backend, NativeAttention)
        assert backend.full_attn_layers == frozenset({0, 2})


# ---------------------------------------------------------------------------
# kimi_linear_config + linear_recurrent_config @property on ModelRunner.
# ---------------------------------------------------------------------------


class TestHybridConfigProperties:
    def _runner_with_hf(self, hf_config):
        """Tiny shim that exposes the two ModelRunner @property descriptors."""
        from types import SimpleNamespace

        from sgl_jax.srt.model_executor.model_runner import ModelRunner

        class _Runner:
            kimi_linear_config = ModelRunner.kimi_linear_config
            linear_recurrent_config = ModelRunner.linear_recurrent_config

            def __init__(self, hf):
                self.model_config = SimpleNamespace(hf_config=hf)

        return _Runner(hf_config)

    def test_kimi_linear_config_returns_config_when_hf_matches(self):
        from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig

        cfg = KimiLinearConfig(num_hidden_layers=2)
        runner = self._runner_with_hf(cfg)
        assert runner.kimi_linear_config is cfg

    def test_kimi_linear_config_returns_none_for_other_configs(self):
        runner = self._runner_with_hf(object())
        assert runner.kimi_linear_config is None

    def test_linear_recurrent_config_is_kimi_for_now(self):
        # linear_recurrent_config currently ORs only kimi_linear_config; future
        # additions (hybrid_gdn_config, ...) extend the chain. Tightened to
        # additionally require the config's own is_linear_attn flag, so a real
        # KimiLinear hf_config must carry a populated linear_attn_config dict.
        from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig

        cfg = KimiLinearConfig(
            num_hidden_layers=2,
            linear_attn_config={
                "kda_layers": [2],
                "full_attn_layers": [1],
            },
        )
        runner = self._runner_with_hf(cfg)
        assert runner.linear_recurrent_config is cfg

    def test_linear_recurrent_config_none_for_degenerate_kimi_linear(self):
        # KimiLinearConfig type matches but linear_attn_config is None →
        # is_linear_attn is False → linear_recurrent_config must be None.
        # Without this gating the umbrella detector would say "hybrid" while
        # init_memory_pool would skip the hybrid pool, leaving the runtime
        # in an inconsistent state at first forward.
        from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig

        cfg = KimiLinearConfig(num_hidden_layers=2)
        runner = self._runner_with_hf(cfg)
        assert runner.kimi_linear_config is cfg  # type still matches
        assert runner.linear_recurrent_config is None  # but flag is False

    def test_linear_recurrent_config_none_when_no_hybrid(self):
        runner = self._runner_with_hf(object())
        assert runner.linear_recurrent_config is None


# ---------------------------------------------------------------------------
# TP-6: two-layer mock model — each sub-backend invoked once, in layer order.
# ---------------------------------------------------------------------------


class TestTwoLayerForwardOrdering:
    def test_two_layer_forward_calls_each_sub_once_in_order(self):
        """1 MLA layer (layer_id=0) + 1 KDA layer (layer_id=1)."""
        hybrid, full, linear = _make_hybrid(full_layers=(0,))
        q, k, v = _qkv()

        hybrid.forward_metadata = HybridLinearAttnBackendMetadata(
            full_attn_metadata=_FakeFullMetadata(tag="step-full"),
            linear_attn_metadata=_FakeLinearMetadata(tag="step-linear"),
        )

        results = []
        for layer_id in (0, 1):
            results.append(
                hybrid(
                    q,
                    k,
                    v,
                    layer=_FakeLayer(layer_id=layer_id),
                    forward_batch=_FakeBatch(name="step1"),
                    pool=None,
                )
            )

        assert [c["layer_id"] for c in full.calls] == [0]
        assert [c["layer_id"] for c in linear.calls] == [1]
        assert full.forward_metadata.tag == "step-full"
        assert linear.forward_metadata.tag == "step-linear"
        assert results[0][0] == "full-out"
        assert results[1][0] == "linear-out"
