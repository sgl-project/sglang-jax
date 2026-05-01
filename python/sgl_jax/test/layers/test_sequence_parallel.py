"""Sequence-parallel scatter tests for QuantizedLinear, EPMoE, and Grok modules.

Both layers fall back to a full all-reduce when ``should_scatter`` returns
False (small batches or tp_size==1) and switch to a reduce-scatter on the
sequence/token dimension when it returns True. These tests exercise both
branches and verify the result is numerically equivalent.

The Grok wiring tests guard against regressions of the form "the flag is
plumbed through ``ServerArgs`` and the model config but doesn't actually
reach the projection that needs to scatter."
"""

import ast
import inspect
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.linear import QuantizedLinear
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.models.grok import Grok1Attention, Grok1DecoderLayer, Grok1MLP
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor
from sgl_jax.test.test_utils import CustomTestCase

_MESH = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
_TP_SIZE = _MESH.shape.get("tensor", 1)
_MIN_LOCAL = 128  # default tpu_scatter_min_local_size


def _spec_dim(sharding, dim):
    """Return the partition-axis name at `dim` (or None if unsharded).

    Tolerates ``PartitionSpec`` of any length: missing trailing entries are
    treated as None, matching JAX's own semantics.
    """
    spec = sharding.spec
    return spec[dim] if dim < len(spec) else None


def _as_fp32(x):
    """Cast to fp32 numpy for tolerance-based comparison.

    ``psum`` and ``psum_scatter`` are mathematically identical but XLA may
    pick different reduction trees, so per-element bf16 outputs can drift by
    a few ULPs. Compare in fp32 with rtol/atol sized to that drift.
    """
    return np.asarray(x).astype(np.float32)


class TestQuantizedLinearScatter(CustomTestCase):
    """``QuantizedLinear.output_scatter_dimension`` behavior."""

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_scatter_active_above_threshold(self):
        """At/above threshold, output is reduce-scattered on dim 0 over `tensor`."""
        batch = _TP_SIZE * _MIN_LOCAL  # exactly at threshold
        scatter_out, baseline_out = self._run_pair(batch)

        # Scatter path: output sharded on dim 0 over the tensor axis.
        self.assertEqual(_spec_dim(scatter_out.sharding, 0), "tensor")
        # Baseline: fully replicated (psum without scatter).
        self.assertIsNone(_spec_dim(baseline_out.sharding, 0))

        # Same math, just different communication pattern. Tolerances cover
        # bf16 reduction-order drift over a 256-wide row-parallel sum (max
        # observed abs diff ~0.5 against mean |y| ~12).
        np.testing.assert_allclose(
            _as_fp32(scatter_out), _as_fp32(baseline_out), rtol=0.05, atol=1.0
        )

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_scatter_inactive_below_threshold(self):
        """Below the per-device min size, scatter is suppressed → psum path."""
        # Pick a batch divisible by tp_size but well below threshold.
        batch = _TP_SIZE * (_MIN_LOCAL // 2)
        scatter_out, _ = self._run_pair(batch)

        # Falls back to fully-replicated (no scatter).
        self.assertIsNone(_spec_dim(scatter_out.sharding, 0))

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_scatter_disabled_when_dimension_is_none(self):
        """``output_scatter_dimension=None`` always replicates regardless of size."""
        batch = _TP_SIZE * _MIN_LOCAL  # would-be scatter size
        x_host, weight_q, weight_scale = _make_quant_linear_inputs(batch, in_dim=256, out_dim=512)

        with jax.set_mesh(_MESH):
            ql = _build_quant_linear(weight_q, weight_scale, _MESH, output_scatter_dimension=None)
            x = jax.device_put(x_host, NamedSharding(_MESH, P(None, "tensor")))
            out, _ = ql(x)

        self.assertIsNone(_spec_dim(out.sharding, 0))

    def _run_pair(self, batch: int):
        in_dim, out_dim = 256, 512
        x_host, weight_q, weight_scale = _make_quant_linear_inputs(batch, in_dim, out_dim)

        with jax.set_mesh(_MESH):
            ql_scatter = _build_quant_linear(
                weight_q, weight_scale, _MESH, output_scatter_dimension=0
            )
            ql_baseline = _build_quant_linear(
                weight_q, weight_scale, _MESH, output_scatter_dimension=None
            )

            x = jax.device_put(x_host, NamedSharding(_MESH, P(None, "tensor")))
            out_scatter, _ = ql_scatter(x)
            out_baseline, _ = ql_baseline(x)

        return out_scatter, out_baseline


def _make_quant_linear_inputs(batch: int, in_dim: int, out_dim: int):
    key = jax.random.PRNGKey(0)
    k_x, k_w = jax.random.split(key)
    x_host = jax.random.normal(k_x, (batch, in_dim), dtype=jnp.bfloat16)
    w_host = jax.random.normal(k_w, (out_dim, in_dim), dtype=jnp.bfloat16)
    weight_q, weight_scale = quantize_tensor(jnp.int8, w_host.astype(jnp.float32), axis=1)
    return x_host, weight_q, weight_scale


def _build_quant_linear(weight_q, weight_scale, mesh, *, output_scatter_dimension):
    ql = QuantizedLinear(
        weight_q=weight_q,
        weight_scale=weight_scale,
        bias=None,
        activation_dtype=None,
        mesh=mesh,
        kernel_axes=("tensor", None),
        params_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        output_scatter_dimension=output_scatter_dimension,
    )
    # Row-parallel: weight is [out, in]; shard on the input axis.
    ql.weight_q = nnx.Param(weight_q, out_sharding=P(None, "tensor"))
    ql.weight_scale = nnx.Param(weight_scale, out_sharding=P(None))
    return ql


def _make_moe_mesh(ep_size: int, tp_size: int) -> Mesh:
    devices = np.array(jax.devices()[: ep_size * tp_size]).reshape(ep_size, tp_size)
    return Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _make_moe_inputs(batch: int, hidden_size: int, num_experts: int):
    key = jax.random.PRNGKey(7)
    k_x, k_topk = jax.random.split(key)
    x = jax.random.normal(k_x, (batch, hidden_size), dtype=jnp.bfloat16)
    topk_weights = jnp.ones((batch, 1), dtype=jnp.bfloat16)
    topk_ids = jax.random.randint(k_topk, (batch, 1), 0, num_experts)
    return x, topk_weights, topk_ids


class TestEPMoESequenceParallel(CustomTestCase):
    """``EPMoE.enable_sequence_parallel`` scatter behavior."""

    HIDDEN_SIZE = 512
    INTERMEDIATE_DIM = 1024
    NUM_EXPERTS = 4

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_seq_parallel_scatters_above_threshold(self):
        """With seq-parallel ON and a large enough batch, output is scattered
        on dim 0 over `tensor`, and matches the all-reduce baseline."""
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE * _MIN_LOCAL
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe_sp = self._build_moe(mesh, enable_sequence_parallel=True)
            moe_base = self._build_moe(mesh, enable_sequence_parallel=False)

            with jax.set_mesh(moe_sp.moe_mesh):
                out_sp = moe_sp(x, topk_weights, topk_ids)
                out_base = moe_base(x, topk_weights, topk_ids)

        self.assertEqual(_spec_dim(out_sp.sharding, 0), "tensor")
        self.assertIsNone(_spec_dim(out_base.sharding, 0))

        np.testing.assert_allclose(_as_fp32(out_sp), _as_fp32(out_base), rtol=0.1, atol=2048.0)

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_seq_parallel_replicates_below_threshold(self):
        """With seq-parallel ON but a tiny batch, ``should_scatter`` returns
        False and we fall back to the fully-replicated psum path."""
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE  # 8 tokens with tp=8 → way below 8*128
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe_sp = self._build_moe(mesh, enable_sequence_parallel=True)
            with jax.set_mesh(moe_sp.moe_mesh):
                out_sp = moe_sp(x, topk_weights, topk_ids)

        self.assertIsNone(_spec_dim(out_sp.sharding, 0))

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_seq_parallel_disabled_always_replicates(self):
        """``enable_sequence_parallel=False`` is the pre-21a6cf8d behavior:
        the output is always fully replicated, regardless of batch size."""
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE * _MIN_LOCAL  # would otherwise scatter
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe = self._build_moe(mesh, enable_sequence_parallel=False)
            with jax.set_mesh(moe.moe_mesh):
                out = moe(x, topk_weights, topk_ids)

        self.assertIsNone(_spec_dim(out.sharding, 0))

    def _build_moe(self, mesh: Mesh, *, enable_sequence_parallel: bool) -> EPMoE:
        return EPMoE(
            hidden_size=self.HIDDEN_SIZE,
            num_experts=self.NUM_EXPERTS,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=self.INTERMEDIATE_DIM,
            quantization_config=None,
            enable_sequence_parallel=enable_sequence_parallel,
        )


def _single_node_mesh() -> Mesh:
    """Mesh covering all visible devices on the ``tensor`` axis.

    Constructor wiring tests don't run forward, so a 1-device mesh is fine,
    but we use the full mesh so the test exercises the same sharding pathway
    used at runtime.
    """
    devices = np.array(jax.devices()).reshape(1, -1)
    return Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


class TestGrokLayerSequenceParallelWiring(CustomTestCase):
    """Verify ``enable_sequence_parallel`` reaches the projection that needs it.

    These were the silent-failure modes called out in review: the flag exists
    on ``ServerArgs`` and propagates onto the model config, but if a layer
    forgets to thread it into its row-parallel projection, sequence parallel
    becomes a no-op for that layer.
    """

    def test_grok1_mlp_wires_scatter_when_enabled(self):
        mesh = _single_node_mesh()
        with jax.set_mesh(mesh):
            mlp = Grok1MLP(
                hidden_size=128,
                intermediate_size=256,
                layer_id=0,
                mesh=mesh,
                enable_sequence_parallel=True,
            )
        # Only down_proj (row-parallel) should scatter; gate/up are column-parallel.
        self.assertEqual(mlp.down_proj.output_scatter_dimension, 0)
        self.assertIsNone(mlp.gate_proj.output_scatter_dimension)
        self.assertIsNone(mlp.up_proj.output_scatter_dimension)

    def test_grok1_mlp_disables_scatter_by_default(self):
        mesh = _single_node_mesh()
        with jax.set_mesh(mesh):
            mlp = Grok1MLP(hidden_size=128, intermediate_size=256, layer_id=0, mesh=mesh)
        self.assertIsNone(mlp.down_proj.output_scatter_dimension)

    def test_grok1_attention_wires_scatter_when_enabled(self):
        mesh = _single_node_mesh()
        cfg = SimpleNamespace(head_dim=64)
        with jax.set_mesh(mesh):
            attn = Grok1Attention(
                config=cfg,
                hidden_size=128,
                num_heads=2,
                num_kv_heads=2,
                mesh=mesh,
                enable_sequence_parallel=True,
            )
        self.assertEqual(attn.o_proj.output_scatter_dimension, 0)
        # q/k/v projections are column-parallel; they don't scatter on output.
        self.assertIsNone(attn.q_proj.output_scatter_dimension)
        self.assertIsNone(attn.k_proj.output_scatter_dimension)
        self.assertIsNone(attn.v_proj.output_scatter_dimension)

    def test_grok1_attention_disables_scatter_by_default(self):
        mesh = _single_node_mesh()
        cfg = SimpleNamespace(head_dim=64)
        with jax.set_mesh(mesh):
            attn = Grok1Attention(
                config=cfg,
                hidden_size=128,
                num_heads=2,
                num_kv_heads=2,
                mesh=mesh,
            )
        self.assertIsNone(attn.o_proj.output_scatter_dimension)

    def test_grok1_decoder_layer_threads_flag_to_attention_and_mlp(self):
        """Static check: ``Grok1DecoderLayer.__init__`` forwards
        ``config.enable_sequence_parallel`` to both the ``Grok1Attention`` and
        the ``Grok1MLP`` constructors.

        Done via AST inspection instead of full instantiation: building a
        ``Grok1DecoderLayer`` requires a complete MoE setup (gate, experts,
        weight allocation) which is far heavier than what this assertion
        needs. The bug class we're guarding against — forgetting to pass the
        kwarg through — is structural and visible in the source.
        """
        src = inspect.getsource(Grok1DecoderLayer.__init__)
        tree = ast.parse(src.lstrip())

        kwargs_by_callee: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"Grok1Attention", "Grok1MLP"}:
                    kwargs_by_callee[node.func.id] = [kw.arg for kw in node.keywords]

        self.assertIn(
            "Grok1Attention",
            kwargs_by_callee,
            "Grok1DecoderLayer.__init__ no longer instantiates Grok1Attention",
        )
        self.assertIn(
            "enable_sequence_parallel",
            kwargs_by_callee["Grok1Attention"],
            "Grok1DecoderLayer must thread enable_sequence_parallel into Grok1Attention; "
            "without it, attention's o_proj never reduce-scatters even with the flag set.",
        )
        # Grok1MLP only appears in the residual-MoE branch; assert only if present.
        if "Grok1MLP" in kwargs_by_callee:
            self.assertIn(
                "enable_sequence_parallel",
                kwargs_by_callee["Grok1MLP"],
                "Grok1DecoderLayer must thread enable_sequence_parallel into Grok1MLP.",
            )


if __name__ == "__main__":
    unittest.main()
