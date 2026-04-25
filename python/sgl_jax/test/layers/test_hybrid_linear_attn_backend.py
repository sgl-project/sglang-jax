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
