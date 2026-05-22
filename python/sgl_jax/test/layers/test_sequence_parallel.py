"""Sequence-parallel sharding-contract tests for row-parallel projections and MoE modules.

Bucket planning decides whether a shape uses full-token or token-sharded
layouts. These tests verify that modules apply the explicit ``out_sharding``
contract they are given at ``__call__`` time.

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

from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.models.grok import Grok1Attention, Grok1DecoderLayer, Grok1MLP
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.utils.parallel_utils import make_reduce_sharding, should_scatter
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor
from sgl_jax.test.test_utils import CustomTestCase

_MESH = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
_TP_SIZE = _MESH.shape.get("tensor", 1)
_MIN_LOCAL = 128
_TOTAL_DEVICES = len(jax.devices())


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


class TestMakeReduceSharding(CustomTestCase):
    """``make_reduce_sharding`` axis-derivation, threshold gating, and scatter_dim."""

    def test_enable_sp_above_threshold_scatters_token_axis(self):
        x = jnp.zeros((_TP_SIZE * _MIN_LOCAL, 64))
        sharding = make_reduce_sharding(x, _MESH, enable_sp=True)
        expected_axes = ("data", "tensor") if _TP_SIZE > 1 else "data"
        self.assertEqual(_spec_dim(sharding, 0), expected_axes)

    def test_enable_sp_below_threshold_falls_back_to_dp(self):
        x = jnp.zeros((max(_TP_SIZE // 2, 1), 64))
        sharding = make_reduce_sharding(x, _MESH, enable_sp=True)
        # Below per-device threshold → must drop back to DP only.
        self.assertEqual(_spec_dim(sharding, 0), "data")

    def test_enable_sp_false_always_dp(self):
        """``enable_sp=False`` forces DP regardless of how large the batch is."""
        x = jnp.zeros((_TP_SIZE * _MIN_LOCAL * 4, 64))
        sharding = make_reduce_sharding(x, _MESH, enable_sp=False)
        self.assertEqual(_spec_dim(sharding, 0), "data")

    def test_scatter_dim_param_targets_arbitrary_axis(self):
        """``scatter_dim=1`` puts the (data, tensor) axes on dim 1 (vocab-like)."""
        if _TP_SIZE < 2:
            self.skipTest("Needs >=2 tensor-parallel devices.")
        x = jnp.zeros((4, _TP_SIZE * _MIN_LOCAL))
        sharding = make_reduce_sharding(x, _MESH, scatter_dim=1, enable_sp=True)
        self.assertIsNone(_spec_dim(sharding, 0))
        self.assertEqual(_spec_dim(sharding, 1), ("data", "tensor"))

    def test_works_on_higher_rank_arrays(self):
        """3D ``[tokens, heads, head_dim]`` shape — tail dims stay replicated."""
        if _TP_SIZE < 2:
            self.skipTest("Needs >=2 tensor-parallel devices.")
        x = jnp.zeros((_TP_SIZE * _MIN_LOCAL, 8, 64))
        sharding = make_reduce_sharding(x, _MESH, enable_sp=True)
        self.assertEqual(_spec_dim(sharding, 0), ("data", "tensor"))
        self.assertIsNone(_spec_dim(sharding, 1))
        self.assertIsNone(_spec_dim(sharding, 2))

    def test_should_scatter_threshold_logic(self):
        """``should_scatter`` is the single source of truth for the threshold."""
        self.assertFalse(should_scatter(dim_size=10, num_devices=1))
        # Per-device slice must be >= TPU_SCATTER_MIN_LOCAL_SIZE.
        self.assertFalse(should_scatter(dim_size=_MIN_LOCAL - 1, num_devices=2))
        self.assertTrue(should_scatter(dim_size=2 * _MIN_LOCAL, num_devices=2))
        # Must divide evenly.
        self.assertFalse(should_scatter(dim_size=2 * _MIN_LOCAL + 1, num_devices=2))


class TestQuantizedLinearScatter(CustomTestCase):
    """``QuantizedLinear`` per-call ``out_sharding`` controls scatter behavior."""

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_scatter_active_above_threshold(self):
        """Explicit SP target on the call yields scattered output that matches DP."""
        batch = _TP_SIZE * _MIN_LOCAL
        scatter_out, baseline_out = self._run_pair(batch)

        self.assertEqual(_spec_dim(scatter_out.sharding, 0), ("data", "tensor"))
        self.assertEqual(_spec_dim(baseline_out.sharding, 0), "data")

        np.testing.assert_allclose(
            _as_fp32(scatter_out), _as_fp32(baseline_out), rtol=0.05, atol=1.0
        )

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_scatter_contract_applies_for_small_bucket(self):
        """Small buckets honor the caller's explicit ``out_sharding`` regardless of size."""
        batch = _TP_SIZE * (_MIN_LOCAL // 2)
        scatter_out, _ = self._run_pair(batch)
        self.assertEqual(_spec_dim(scatter_out.sharding, 0), ("data", "tensor"))

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_default_out_sharding_falls_back_to_dp(self):
        """No ``out_sharding=`` → standard TP fallback (DP only on dim 0)."""
        batch = _TP_SIZE * _MIN_LOCAL
        x_host, weight_q, weight_scale = _make_quant_linear_inputs(batch, 256, 512)

        with jax.set_mesh(_MESH):
            ql = _build_quant_linear(weight_q, weight_scale, _MESH)
            x = jax.device_put(x_host, NamedSharding(_MESH, P("data", "tensor")))
            out, _ = ql(x)

        self.assertEqual(_spec_dim(out.sharding, 0), "data")

    def _run_pair(self, batch: int):
        in_dim, out_dim = 256, 512
        x_host, weight_q, weight_scale = _make_quant_linear_inputs(batch, in_dim, out_dim)

        with jax.set_mesh(_MESH):
            ql = _build_quant_linear(weight_q, weight_scale, _MESH)
            x = jax.device_put(x_host, NamedSharding(_MESH, P("data", "tensor")))
            out_scatter, _ = ql(x, out_sharding=NamedSharding(_MESH, P(("data", "tensor"), None)))
            out_baseline, _ = ql(x)  # default DP

        return out_scatter, out_baseline


class TestLinearBaseScatter(CustomTestCase):
    """``LinearBase`` per-call ``out_sharding`` controls scatter behavior."""

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_row_parallel_scatter_active_above_threshold(self):
        batch = _TP_SIZE * _MIN_LOCAL
        in_dim, out_dim = 256, 512
        key = jax.random.PRNGKey(11)
        k_x, k_w = jax.random.split(key)
        x_host = jax.random.normal(k_x, (batch, in_dim), dtype=jnp.bfloat16)
        w_host = jax.random.normal(k_w, (in_dim, out_dim), dtype=jnp.bfloat16)

        with jax.set_mesh(_MESH):
            lin = _build_linear_base(w_host, _MESH)
            x = jax.device_put(x_host, NamedSharding(_MESH, P("data", "tensor")))
            scatter_out, _ = lin(x, out_sharding=NamedSharding(_MESH, P(("data", "tensor"), None)))
            baseline_out, _ = lin(x)  # default DP

        self.assertEqual(_spec_dim(scatter_out.sharding, 0), ("data", "tensor"))
        self.assertEqual(_spec_dim(baseline_out.sharding, 0), "data")
        np.testing.assert_allclose(
            _as_fp32(scatter_out), _as_fp32(baseline_out), rtol=0.05, atol=1.0
        )


def _make_quant_linear_inputs(batch: int, in_dim: int, out_dim: int):
    key = jax.random.PRNGKey(0)
    k_x, k_w = jax.random.split(key)
    x_host = jax.random.normal(k_x, (batch, in_dim), dtype=jnp.bfloat16)
    w_host = jax.random.normal(k_w, (out_dim, in_dim), dtype=jnp.bfloat16)
    weight_q, weight_scale = quantize_tensor(jnp.int8, w_host.astype(jnp.float32), axis=1)
    return x_host, weight_q, weight_scale


def _build_quant_linear(weight_q, weight_scale, mesh):
    ql = QuantizedLinear(
        weight_q=weight_q,
        weight_scale=weight_scale,
        bias=None,
        activation_dtype=None,
        mesh=mesh,
        kernel_axes=("tensor", None),
        params_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
    )
    # Row-parallel: weight is [out, in]; shard on the input axis.
    ql.weight_q = nnx.Param(weight_q, out_sharding=P(None, "tensor"))
    ql.weight_scale = nnx.Param(weight_scale, out_sharding=P(None))
    return ql


def _build_linear_base(weight, mesh):
    lin = LinearBase(
        input_size=weight.shape[0],
        output_size=weight.shape[1],
        use_bias=False,
        mesh=mesh,
        kernel_axes=("tensor", None),
        params_dtype=jnp.bfloat16,
    )
    lin.weight = nnx.Param(weight, out_sharding=P("tensor", None))
    return lin


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
    """``EPMoE`` per-call ``out_sharding`` controls scatter strategy."""

    HIDDEN_SIZE = 512
    INTERMEDIATE_DIM = 1024
    NUM_EXPERTS = 4

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_seq_parallel_scatters_when_target_includes_tensor(self):
        """``out_sharding`` containing ``tensor`` on token-axis → internal psum_scatter."""
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE * _MIN_LOCAL
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe = self._build_moe(mesh)
            with jax.set_mesh(moe.moe_mesh):
                out_sp = moe(
                    x,
                    topk_weights,
                    topk_ids,
                    out_sharding=NamedSharding(mesh, P(("data", "tensor"), None)),
                )
                out_base = moe(x, topk_weights, topk_ids)  # default DP

        self.assertEqual(_spec_dim(out_sp.sharding, 0), ("data", "tensor"))
        self.assertEqual(_spec_dim(out_base.sharding, 0), "data")

        np.testing.assert_allclose(_as_fp32(out_sp), _as_fp32(out_base), rtol=0.1, atol=2048.0)

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_seq_parallel_contract_applies_for_small_bucket(self):
        """Small buckets follow caller's ``out_sharding`` (no threshold gating in module)."""
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe = self._build_moe(mesh)
            with jax.set_mesh(moe.moe_mesh):
                out_sp = moe(
                    x,
                    topk_weights,
                    topk_ids,
                    out_sharding=NamedSharding(mesh, P(("data", "tensor"), None)),
                )

        self.assertEqual(_spec_dim(out_sp.sharding, 0), ("data", "tensor"))

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_no_out_sharding_defaults_to_dp(self):
        """Caller without ``out_sharding=`` gets plain-DP output (LinearBase-style default).

        This is the path used by the 6 models that haven't adopted SP yet
        (qwen2_moe, glm4_moe, etc.); they call ``self.mlp(x, topk_w, topk_ids)``
        without any sharding argument.
        """
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE * _MIN_LOCAL
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe = self._build_moe(mesh)
            with jax.set_mesh(moe.moe_mesh):
                out = moe(x, topk_weights, topk_ids)

        self.assertEqual(_spec_dim(out.sharding, 0), "data")

    @unittest.skipIf(_TP_SIZE < 2, "Needs >=2 tensor-parallel devices.")
    def test_seq_parallel_uses_model_tensor_axis_when_ep_equals_world(self):
        """EP-only MoE must still match attention o_proj's SP output contract."""
        mesh = _make_moe_mesh(ep_size=1, tp_size=_TP_SIZE)
        batch = _TP_SIZE * _MIN_LOCAL
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, _TP_SIZE)

        with jax.set_mesh(mesh):
            moe = EPMoE(
                hidden_size=self.HIDDEN_SIZE,
                num_experts=_TP_SIZE,
                num_experts_per_tok=1,
                ep_size=_TP_SIZE,
                mesh=mesh,
                intermediate_dim=self.INTERMEDIATE_DIM,
                quantization_config=None,
            )
            with jax.set_mesh(moe.moe_mesh):
                out = moe(
                    x,
                    topk_weights,
                    topk_ids,
                    out_sharding=NamedSharding(mesh, P(("data", "tensor"), None)),
                )

        self.assertEqual(_spec_dim(out.sharding, 0), ("data", "tensor"))

    def _build_moe(self, mesh: Mesh) -> EPMoE:
        return EPMoE(
            hidden_size=self.HIDDEN_SIZE,
            num_experts=self.NUM_EXPERTS,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=self.INTERMEDIATE_DIM,
            quantization_config=None,
        )


def _single_node_mesh() -> Mesh:
    """Mesh covering all visible devices on the ``tensor`` axis."""
    devices = np.array(jax.devices()).reshape(1, -1)
    return Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


class TestGrokLayerSequenceParallelWiring(CustomTestCase):
    """Verify ``enable_sequence_parallel`` flag reaches the projections that need it.

    The flag exists on ``ServerArgs`` and propagates onto the model config; if a
    layer forgets to thread it into its row-parallel projection, sequence
    parallel becomes a no-op for that layer.
    """

    def test_grok1_mlp_stores_enable_sequence_parallel_flag(self):
        mesh = _single_node_mesh()
        with jax.set_mesh(mesh):
            mlp = Grok1MLP(
                hidden_size=128,
                intermediate_size=256,
                layer_id=0,
                mesh=mesh,
                enable_sequence_parallel=True,
            )
        self.assertTrue(mlp.enable_sequence_parallel)

    def test_grok1_mlp_defaults_to_disabled(self):
        mesh = _single_node_mesh()
        with jax.set_mesh(mesh):
            mlp = Grok1MLP(hidden_size=128, intermediate_size=256, layer_id=0, mesh=mesh)
        self.assertFalse(mlp.enable_sequence_parallel)

    def test_grok1_attention_stores_enable_sequence_parallel_flag(self):
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
        self.assertTrue(attn.enable_sequence_parallel)

    def test_grok1_attention_defaults_to_disabled(self):
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
        self.assertFalse(attn.enable_sequence_parallel)

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
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"Grok1Attention", "Grok1MLP"}
            ):
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
        if "Grok1MLP" in kwargs_by_callee:
            self.assertIn(
                "enable_sequence_parallel",
                kwargs_by_callee["Grok1MLP"],
                "Grok1DecoderLayer must thread enable_sequence_parallel into Grok1MLP.",
            )


def _make_dp_tp_mesh(dp_size: int, tp_size: int) -> Mesh:
    """Make a ``(dp_size, tp_size)`` mesh with axis names ``("data", "tensor")``."""
    devices = np.array(jax.devices()[: dp_size * tp_size]).reshape(dp_size, tp_size)
    return Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


class TestDpSpComposition(CustomTestCase):
    """Verify scatter dim stripes across BOTH ``data`` and ``tensor`` under DP+SP."""

    HIDDEN_SIZE = 512
    INTERMEDIATE_DIM = 1024
    NUM_EXPERTS = 4

    @unittest.skipIf(_TOTAL_DEVICES < 8, "Needs >=8 devices for dp=2, tp=4.")
    def test_quantized_linear_scatter_combines_data_and_tensor(self):
        mesh = _make_dp_tp_mesh(dp_size=2, tp_size=4)
        batch = 2 * 4 * _MIN_LOCAL
        in_dim, out_dim = 256, 512
        x_host, weight_q, weight_scale = _make_quant_linear_inputs(batch, in_dim, out_dim)

        with jax.set_mesh(mesh):
            ql = _build_quant_linear(weight_q, weight_scale, mesh)
            x = jax.device_put(x_host, NamedSharding(mesh, P("data", "tensor")))
            out_scatter, _ = ql(x, out_sharding=NamedSharding(mesh, P(("data", "tensor"), None)))
            out_baseline, _ = ql(x)  # default DP

        self.assertEqual(_spec_dim(out_scatter.sharding, 0), ("data", "tensor"))
        self.assertEqual(_spec_dim(out_baseline.sharding, 0), "data")

        np.testing.assert_allclose(
            _as_fp32(out_scatter), _as_fp32(out_baseline), rtol=0.05, atol=1.0
        )

    @unittest.skipIf(_TOTAL_DEVICES < 8, "Needs >=8 devices for dp=2, tp=4.")
    def test_epmoe_seq_parallel_combines_data_and_tensor(self):
        mesh = _make_dp_tp_mesh(dp_size=2, tp_size=4)
        batch = 8 * _MIN_LOCAL
        x, topk_weights, topk_ids = _make_moe_inputs(batch, self.HIDDEN_SIZE, self.NUM_EXPERTS)

        with jax.set_mesh(mesh):
            moe = self._build_moe(mesh)
            with jax.set_mesh(moe.moe_mesh):
                out_sp = moe(
                    x,
                    topk_weights,
                    topk_ids,
                    out_sharding=NamedSharding(mesh, P(("data", "tensor"), None)),
                )
                out_base = moe(x, topk_weights, topk_ids)  # default DP

        self.assertEqual(_spec_dim(out_sp.sharding, 0), ("data", "tensor"))
        self.assertEqual(_spec_dim(out_base.sharding, 0), "data")

        np.testing.assert_allclose(_as_fp32(out_sp), _as_fp32(out_base), rtol=0.1, atol=2048.0)

    def _build_moe(self, mesh: Mesh) -> EPMoE:
        return EPMoE(
            hidden_size=self.HIDDEN_SIZE,
            num_experts=self.NUM_EXPERTS,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=self.INTERMEDIATE_DIM,
            quantization_config=None,
        )


if __name__ == "__main__":
    unittest.main()
