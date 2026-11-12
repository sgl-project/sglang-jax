"""CPU interpret-mode numeric tests for the FusedEPMoE v1 bf-resident VMEM caches.

These tests run the real Pallas kernel (not a simulation) under Pallas TPU
interpret mode on CPU and compare it against ``ref_moe``. Uninitialized VMEM is
filled with NaN, so a slice that reads a buffer the kernel never wrote (for
example the x 2D token cache ``b_x_2d_vmem`` on ``bf_id > 0``) shows up as NaN
in the output instead of passing on stale data.

Every case uses ``num_bd1 == num_bd2 == 1`` and ``num_bf > 1``, which turns on
both the x cache and the FFN2 partial-sum cache when ``vmem_2d_tile_relayout``
is set (the default). The cases keep ``bt <= 16`` so the output accumulation
runs a single acc tile: with two or more acc tiles both ping-pong buffers share
``a2a_acc_sems[0]``, a pre-existing ordering hazard that ``on_wait`` DMA mode
exposes independently of the caches.
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe,
    ref_moe,
)
from sgl_jax.srt.kernels.fused_moe.v1.tuned_block_configs import (
    DEFAULT_FUSED_MOE_BLOCK_CONFIG,
)
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor

jax.config.parse_flags_with_absl()

_REAL_PALLAS_CALL = pl.pallas_call


@contextlib.contextmanager
def _tpu_interpret(dma_execution_mode: str):
    """Run every pallas_call in TPU interpret mode and record its scratch shapes."""
    scratch_shapes = []

    def _interpret_pallas_call(*args, **kwargs):
        kwargs.pop("compiler_params", None)
        kwargs["interpret"] = pltpu.InterpretParams(
            uninitialized_memory="nan",
            out_of_bounds_reads="raise",
            dma_execution_mode=dma_execution_mode,
        )
        grid_spec = kwargs.get("grid_spec")
        if grid_spec is not None:
            scratch_shapes.extend(s for s in grid_spec.scratch_shapes if s is not None)
        return _REAL_PALLAS_CALL(*args, **kwargs)

    pl.pallas_call = _interpret_pallas_call
    try:
        yield scratch_shapes
    finally:
        pl.pallas_call = _REAL_PALLAS_CALL


def _gen_inputs(*, num_experts, top_k, num_tokens, hidden_size, intermediate_size, has_bias, seed):
    keys = jax.random.split(jax.random.key(seed), 8)
    tokens = (jax.random.normal(keys[0], (num_tokens, hidden_size), dtype=jnp.float32) / 4).astype(
        jnp.bfloat16
    )
    w_shape_1 = (num_experts, hidden_size, intermediate_size)
    w_shape_2 = (num_experts, intermediate_size, hidden_size)
    w1 = jax.random.normal(keys[1], w_shape_1, dtype=jnp.float32) / 10
    w2 = jax.random.normal(keys[2], w_shape_2, dtype=jnp.float32) / 10
    w3 = jax.random.normal(keys[3], w_shape_1, dtype=jnp.float32) / 10
    b1 = b2 = b3 = None
    if has_bias:
        b1 = jax.random.normal(keys[4], (num_experts, 1, intermediate_size), jnp.float32) / 10
        b2 = jax.random.normal(keys[5], (num_experts, 1, hidden_size), jnp.float32) / 10
        b3 = jax.random.normal(keys[6], (num_experts, 1, intermediate_size), jnp.float32) / 10
    # Softmax probabilities as gating output, so ref_moe's top-k weights are in [0, 1].
    logits = jax.random.normal(keys[7], (num_tokens, num_experts), dtype=jnp.float32)
    gating = jax.nn.softmax(logits, axis=-1)
    topk_weights, topk_ids = lax.top_k(gating, top_k)
    return tokens, w1, w2, w3, b1, b2, b3, gating, topk_weights, topk_ids.astype(jnp.int32)


def _quantize(w_dtype, w, quant_block_k):
    w_q, scale = quantize_tensor(w_dtype, w, axis=1, block_size=quant_block_k)
    # 4D scale layout expected by the kernel: (E, K // quant_block_k, 1, N).
    return w_q, scale.reshape(scale.shape[0], scale.shape[1], 1, scale.shape[2])


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class FusedMoEV1InterpretTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        # The interpret params are baked in at trace time; drop cached traces so each
        # case is traced under its own DMA execution mode.
        jax.clear_caches()

    def _run(
        self,
        *,
        ep_size=1,
        num_experts=8,
        top_k=2,
        num_tokens=16,
        hidden_size,
        intermediate_size,
        block_config,
        w_dtype=None,
        quant_block_k=None,
        act_fn="silu",
        has_bias=False,
        vmem_packed_sublane_transpose=True,
        dma_execution_mode="on_wait",
        atol=3e-2,
        rtol=3e-2,
    ):
        if len(jax.devices()) < ep_size:
            self.skipTest(f"needs {ep_size} devices (set JAX_NUM_CPU_DEVICES)")
        tokens, w1, w2, w3, b1, b2, b3, gating, topk_weights, topk_ids = _gen_inputs(
            num_experts=num_experts,
            top_k=top_k,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            has_bias=has_bias,
            seed=1234,
        )
        w1_scale = w2_scale = w3_scale = None
        if w_dtype is None:
            w1, w2, w3 = (w.astype(jnp.bfloat16) for w in (w1, w2, w3))
        else:
            w1, w1_scale = _quantize(w_dtype, w1, quant_block_k)
            w2, w2_scale = _quantize(w_dtype, w2, quant_block_k)
            w3, w3_scale = _quantize(w_dtype, w3, quant_block_k)

        mesh = Mesh(np.array(jax.devices()[:ep_size]).reshape(1, ep_size), ("data", "tensor"))
        ep_sharding = NamedSharding(mesh, P(("data", "tensor")))

        def shard(x):
            return None if x is None else jax.device_put(x, ep_sharding)

        with _tpu_interpret(dma_execution_mode) as scratch_shapes:
            actual = fused_ep_moe(
                mesh=mesh,
                tokens=shard(tokens),
                w1=shard(w1),
                w2=shard(w2),
                w3=shard(w3),
                topk_weights=shard(topk_weights),
                topk_ids=shard(topk_ids),
                top_k=top_k,
                act_fn=act_fn,
                quant_block_k=quant_block_k,
                w1_scale=shard(w1_scale),
                w2_scale=shard(w2_scale),
                w3_scale=shard(w3_scale),
                b1=shard(b1),
                b2=shard(b2),
                b3=shard(b3),
                block_config=block_config,
                vmem_packed_sublane_transpose=vmem_packed_sublane_transpose,
                tp_axis_name="tensor",
            )
            actual = np.asarray(jax.device_get(actual)).astype(np.float32)

        # Both bf-resident caches, (t_packing, a2a_max_tokens, hidden // t_packing), must be
        # allocated, otherwise this case does not exercise the cached path at all.
        t_packing = 2
        caches = [
            s
            for s in scratch_shapes
            if len(getattr(s, "shape", ())) == 3
            and s.shape[0] == t_packing
            and s.shape[2] == hidden_size // t_packing
            and s.dtype == jnp.bfloat16
        ]
        self.assertLen(caches, 2, msg=f"x/res 2D VMEM caches not both active: {scratch_shapes}")

        expected = ref_moe(
            tokens,
            w1,
            w2,
            w3,
            gating,
            top_k,
            act_fn=act_fn,
            quant_block_k=quant_block_k,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            b1=b1,
            b2=b2,
            b3=b3,
        )
        expected = np.asarray(expected).astype(np.float32)

        nan_count = int(np.isnan(actual).sum())
        self.assertEqual(nan_count, 0, msg=f"{nan_count}/{actual.size} NaN outputs")
        self.assertAllClose(actual, expected, atol=atol, rtol=rtol)

    @parameterized.named_parameters(
        ("fp8", jnp.float8_e4m3fn, True, "on_wait"),
        ("int8", jnp.int8, True, "on_wait"),
        ("fp8_no_sublane_transpose", jnp.float8_e4m3fn, False, "on_wait"),
        ("fp8_eager_dma", jnp.float8_e4m3fn, True, "eager"),
    )
    def test_quantized_multi_scale_group_ffn1_with_x_cache(
        self, w_dtype, vmem_packed_sublane_transpose, dma_execution_mode
    ):
        """Quantized FFN1 with several scale groups per bd1c tile and the x cache on.

        n_sg = (bd1c / t_packing) / quant_block_k = 256 / 128 = 2 and num_bf = 512 / 256 = 2,
        so FFN1 takes the multi-scale-group path and bf_id = 1 reads tokens only from
        b_x_2d_vmem.
        """
        self._run(
            hidden_size=512,
            intermediate_size=512,
            w_dtype=w_dtype,
            quant_block_k=128,
            block_config=FusedMoEBlockConfig(
                bt=16,
                bf=256,
                bd1=512,
                bd2=512,
                btc=16,
                bfc=256,
                bd1c=512,
                bd2c=512,
                bse=256,
                bts=16,
            ),
            vmem_packed_sublane_transpose=vmem_packed_sublane_transpose,
            dma_execution_mode=dma_execution_mode,
        )

    def test_quantized_default_block_config(self):
        """DEFAULT_FUSED_MOE_BLOCK_CONFIG (bf=512, bd1=bd1c=1024) for a quantized model with
        hidden=1024: n_sg = 512 / 256 = 2, num_bf = 2."""
        self._run(
            hidden_size=1024,
            intermediate_size=1024,
            w_dtype=jnp.float8_e4m3fn,
            quant_block_k=256,
            block_config=DEFAULT_FUSED_MOE_BLOCK_CONFIG,
        )

    def test_quantized_multi_scale_group_with_bias_and_token_tiles(self):
        """n_sg = 4 with biases and two token tiles per expert (bts < tokens per expert)."""
        self._run(
            num_experts=4,
            hidden_size=1024,
            intermediate_size=512,
            w_dtype=jnp.float8_e4m3fn,
            quant_block_k=128,
            has_bias=True,
            block_config=FusedMoEBlockConfig(
                bt=16,
                bf=256,
                bd1=1024,
                bd2=1024,
                btc=8,
                bfc=256,
                bd1c=1024,
                bd2c=1024,
                bse=256,
                bts=8,
            ),
        )

    def test_quantized_multi_scale_group_ep2(self):
        self._run(
            ep_size=2,
            num_tokens=32,
            hidden_size=512,
            intermediate_size=512,
            w_dtype=jnp.float8_e4m3fn,
            quant_block_k=128,
            block_config=FusedMoEBlockConfig(
                bt=16,
                bf=256,
                bd1=512,
                bd2=512,
                btc=16,
                bfc=256,
                bd1c=512,
                bd2c=512,
                bse=256,
                bts=16,
            ),
        )

    @parameterized.named_parameters(
        ("silu", "silu", True),
        ("gelu", "gelu", True),
        ("swigluoai", "swigluoai", True),
        ("silu_no_sublane_transpose", "silu", False),
    )
    def test_bf16_caches_and_fused_activation(self, act_fn, vmem_packed_sublane_transpose):
        """bf16 weights, num_bf = 3, activation fused into FFN1 (num_bd1 == 1)."""
        self._run(
            hidden_size=256,
            intermediate_size=384,
            act_fn=act_fn,
            has_bias=act_fn == "swigluoai",
            block_config=FusedMoEBlockConfig(
                bt=16,
                bf=128,
                bd1=256,
                bd2=256,
                btc=8,
                bfc=128,
                bd1c=256,
                bd2c=256,
                bse=128,
                bts=8,
            ),
            vmem_packed_sublane_transpose=vmem_packed_sublane_transpose,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
