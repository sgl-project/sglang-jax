from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

_FORCE_CPU = os.environ.get("SGLANG_JAX_TEST_FORCE_CPU", "0") == "1"
if _FORCE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"
    for _tpu_env in ("TPU_ACCELERATOR_TYPE", "TPU_WORKER_HOSTNAMES"):
        os.environ.pop(_tpu_env, None)

try:
    import torch
    from diffusers.models.transformers.transformer_flux import (
        FluxAttention as HFFluxAttention,
    )
    from diffusers.models.transformers.transformer_flux import (
        FluxTransformer2DModel as HFFluxTransformer2DModel,
    )
except ImportError:  # pragma: no cover
    torch = None
    HFFluxAttention = None
    HFFluxTransformer2DModel = None

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
except ImportError:  # pragma: no cover
    jax = None
    jnp = None
    nnx = None

if jax is not None:
    from sgl_jax.srt.multimodal.models.dits.flux import (
        FluxAttention as JaxFluxAttention,
    )
    from sgl_jax.srt.multimodal.models.dits.flux_dit_weights_mapping import to_mappings

MODEL_ROOT = Path(os.environ.get("FLUX_MODEL_PATH", "/models/FLUX1.0"))
TRANSFORMER_PATH = MODEL_ROOT / "transformer"
CONFIG_PATH = TRANSFORMER_PATH / "config.json"


def _copy_hf_state_dict_to_jax(*args, **kwargs):
    from sgl_jax.test.multimodal.test_flux_transformer_2d_model_parity_demo import (
        copy_hf_state_dict_to_jax,
    )

    return copy_hf_state_dict_to_jax(*args, **kwargs)


@unittest.skipIf(
    torch is None or HFFluxAttention is None or jax is None or nnx is None,
    "torch/diffusers/jax/flax not installed",
)
class TestFluxAttentionParityDemo(unittest.TestCase):
    def _has_tpu(self) -> bool:
        return any(device.platform == "tpu" for device in jax.devices())

    def _is_cpu_runtime(self) -> bool:
        return all(device.platform == "cpu" for device in jax.devices())

    def _skip_unless_cpu_runtime(self) -> None:
        if not self._is_cpu_runtime():
            self.skipTest(
                "Strict sdpa parity is intended for CPU runtime. "
                "Set SGLANG_JAX_TEST_FORCE_CPU=1 to enable it."
            )

    def _build_attention_pair(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        attention_impl: str = "sdpa",
    ):
        torch_attn = HFFluxAttention(
            query_dim=hidden_size,
            heads=num_heads,
            dim_head=head_dim,
            bias=True,
            added_kv_proj_dim=hidden_size,
            out_dim=hidden_size,
            out_bias=True,
            pre_only=False,
            elementwise_affine=True,
        ).cpu()
        torch_attn.eval()

        jax_attn = JaxFluxAttention(
            query_dim=hidden_size,
            heads=num_heads,
            dim_head=head_dim,
            bias=True,
            added_kv_proj_dim=hidden_size,
            out_dim=hidden_size,
            out_bias=True,
            pre_only=False,
            elementwise_affine=True,
            attention_impl=attention_impl,
            params_dtype=jnp.float32,
        )
        _copy_hf_state_dict_to_jax(
            torch_attn.state_dict(),
            jax_attn,
            to_mappings(),
            hf_prefix="transformer_blocks.0.attn",
            target_prefix="transformer_blocks.0.attn",
        )
        return torch_attn, jax_attn

    def _build_loaded_attention_pair(self, *, attention_impl: str):
        if not CONFIG_PATH.exists():
            raise unittest.SkipTest(f"Config not found: {CONFIG_PATH}")
        if HFFluxTransformer2DModel is None:
            raise unittest.SkipTest("diffusers FluxTransformer2DModel unavailable")
        if not hasattr(HFFluxTransformer2DModel, "from_pretrained"):
            raise unittest.SkipTest("diffusers FluxTransformer2DModel.from_pretrained unavailable")

        torch_model = HFFluxTransformer2DModel.from_pretrained(
            str(TRANSFORMER_PATH),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).cpu()
        torch_model.eval()
        torch_attn = torch_model.transformer_blocks[0].attn

        jax_attn = JaxFluxAttention(
            query_dim=torch_attn.query_dim,
            heads=torch_attn.heads,
            dim_head=torch_attn.head_dim,
            bias=torch_attn.use_bias,
            added_kv_proj_dim=torch_attn.added_kv_proj_dim,
            added_proj_bias=torch_attn.added_proj_bias,
            out_bias=torch_attn.to_out[0].bias is not None,
            out_dim=torch_attn.out_dim,
            context_pre_only=torch_attn.context_pre_only,
            pre_only=torch_attn.pre_only,
            elementwise_affine=getattr(torch_attn.norm_q, "weight", None) is not None,
            attention_impl=attention_impl,
            params_dtype=jnp.float32,
        )
        _copy_hf_state_dict_to_jax(
            torch_attn.state_dict(),
            jax_attn,
            to_mappings(),
            hf_prefix="transformer_blocks.0.attn",
            target_prefix="transformer_blocks.0.attn",
        )

        return torch_model, torch_attn, jax_attn

    def _assert_outputs_close(self, torch_hidden, torch_context, jax_hidden, jax_context):
        torch_hidden_np = torch_hidden.detach().cpu().numpy()
        torch_context_np = torch_context.detach().cpu().numpy()
        jax_hidden_np = np.asarray(jax_hidden)
        jax_context_np = np.asarray(jax_context)

        np.testing.assert_allclose(torch_hidden_np, jax_hidden_np, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(torch_context_np, jax_context_np, rtol=1e-4, atol=1e-4)

    def _assert_pair_close(self, torch_pair, jax_pair, atol=1e-4, rtol=1e-4):
        self.assertEqual(len(torch_pair), len(jax_pair))
        for torch_item, jax_item in zip(torch_pair, jax_pair, strict=True):
            np.testing.assert_allclose(
                torch_item.detach().cpu().numpy(),
                np.asarray(jax_item),
                atol=atol,
                rtol=rtol,
            )

    def _run_case(
        self,
        *,
        attention_mask_torch: torch.Tensor | None = None,
        image_rotary_emb_torch: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_impl: str = "sdpa",
        deterministic_inputs: bool = False,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ):
        batch_size, seq_len, context_len = 2, 8, 4
        hidden_size, num_heads, head_dim = 128, 4, 8
        torch_attn, jax_attn = self._build_attention_pair(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_impl=attention_impl,
        )

        if deterministic_inputs:
            hidden_states_torch = torch.linspace(
                -1.0,
                1.0,
                steps=batch_size * seq_len * hidden_size,
                dtype=torch.float32,
            ).reshape(batch_size, seq_len, hidden_size)
            encoder_hidden_states_torch = torch.linspace(
                1.0,
                -1.0,
                steps=batch_size * context_len * hidden_size,
                dtype=torch.float32,
            ).reshape(batch_size, context_len, hidden_size)
        else:
            hidden_states_torch = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
            encoder_hidden_states_torch = torch.randn(
                batch_size, context_len, hidden_size, dtype=torch.float32
            )

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        encoder_hidden_states_jax = jnp.asarray(encoder_hidden_states_torch.detach().cpu().numpy())
        attention_mask_jax = None
        if attention_mask_torch is not None:
            attention_mask_jax = jnp.asarray(attention_mask_torch.detach().cpu().numpy())
        image_rotary_emb_jax = None
        if image_rotary_emb_torch is not None:
            image_rotary_emb_jax = tuple(
                jnp.asarray(x.detach().cpu().numpy()) for x in image_rotary_emb_torch
            )

        with torch.no_grad():
            torch_hidden, torch_context = torch_attn(
                hidden_states=hidden_states_torch,
                encoder_hidden_states=encoder_hidden_states_torch,
                attention_mask=attention_mask_torch,
                image_rotary_emb=image_rotary_emb_torch,
            )

        jax_hidden, jax_context = jax_attn(
            hidden_states=hidden_states_jax,
            encoder_hidden_states=encoder_hidden_states_jax,
            attention_mask=attention_mask_jax,
            image_rotary_emb=image_rotary_emb_jax,
        )

        self._assert_pair_close(
            (torch_hidden, torch_context),
            (jax_hidden, jax_context),
            atol=atol,
            rtol=rtol,
        )

    def test_flux_attention_cross_attention_cpu_parity(self):
        self._skip_unless_cpu_runtime()
        torch.manual_seed(0)
        self._run_case()

    def test_flux_attention_cross_attention_mask_cpu_parity(self):
        self._skip_unless_cpu_runtime()
        torch.manual_seed(0)
        batch_size = 2
        total_kv_len = 12
        attention_mask_torch = torch.zeros((batch_size, 1, 1, total_kv_len), dtype=torch.float32)
        attention_mask_torch[:, :, :, -2:] = -1.0e4
        self._run_case(attention_mask_torch=attention_mask_torch)

    def test_flux_attention_cross_attention_rope_cpu_parity(self):
        self._skip_unless_cpu_runtime()
        torch.manual_seed(0)
        head_dim = 8
        total_kv_len = 12
        cos_torch = torch.randn(total_kv_len, head_dim, dtype=torch.float32)
        sin_torch = torch.randn(total_kv_len, head_dim, dtype=torch.float32)
        image_rotary_emb_torch = (cos_torch, sin_torch)
        self._run_case(image_rotary_emb_torch=image_rotary_emb_torch)

    def test_flux_attention_cross_attention_usp_loaded_weights_tpu_alignment(self):
        if not self._has_tpu():
            self.skipTest("USP parity test requires TPU.")

        torch_model, torch_attn, jax_attn = self._build_loaded_attention_pair(attention_impl="usp")

        hidden_size = torch_attn.query_dim
        context_size = torch_attn.added_kv_proj_dim
        batch_size, seq_len, context_len = 1, 8, 4

        hidden_states_torch = torch.linspace(
            -1.0,
            1.0,
            steps=batch_size * seq_len * hidden_size,
            dtype=torch.float32,
        ).reshape(batch_size, seq_len, hidden_size)
        encoder_hidden_states_torch = torch.linspace(
            1.0,
            -1.0,
            steps=batch_size * context_len * context_size,
            dtype=torch.float32,
        ).reshape(batch_size, context_len, context_size)

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        encoder_hidden_states_jax = jnp.asarray(encoder_hidden_states_torch.detach().cpu().numpy())

        with torch.no_grad():
            torch_hidden, torch_context = torch_attn(
                hidden_states=hidden_states_torch,
                encoder_hidden_states=encoder_hidden_states_torch,
            )

        jax_hidden, jax_context = jax_attn(
            hidden_states=hidden_states_jax,
            encoder_hidden_states=encoder_hidden_states_jax,
        )

        del torch_model

        self._assert_pair_close(
            (torch_hidden, torch_context),
            (jax_hidden, jax_context),
            atol=3e-2,
            rtol=3e-2,
        )


if __name__ == "__main__":
    unittest.main()
