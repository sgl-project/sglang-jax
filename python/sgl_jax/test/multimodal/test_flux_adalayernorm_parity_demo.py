from __future__ import annotations

import os
import unittest
from contextlib import nullcontext

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

try:
    import torch
    from diffusers.models.normalization import (
        AdaLayerNormContinuous as HFAdaLayerNormContinuous,
    )
    from diffusers.models.normalization import AdaLayerNormZero as HFAdaLayerNormZero
    from diffusers.models.normalization import (
        AdaLayerNormZeroSingle as HFAdaLayerNormZeroSingle,
    )
except ImportError:  # pragma: no cover
    torch = None
    HFAdaLayerNormContinuous = None
    HFAdaLayerNormZero = None
    HFAdaLayerNormZeroSingle = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None

if jax is not None:
    from sgl_jax.srt.multimodal.layers.adalayernorm import (
        FluxAdaLayerNormContinuous,
        FluxAdaLayerNormZero,
        FluxAdaLayerNormZeroSingle,
    )
from sgl_jax.srt.utils.weight_utils import WeightMapping


def _copy_hf_state_dict_to_jax(*args, **kwargs):
    from sgl_jax.test.multimodal.test_flux_transformer_2d_model_parity_demo import (
        copy_hf_state_dict_to_jax,
    )

    return copy_hf_state_dict_to_jax(*args, **kwargs)


_ADALN_ZERO_MAPPINGS = {
    "linear.weight": WeightMapping(target_path="linear.weight", transpose=True),
    "linear.bias": WeightMapping(target_path="linear.bias"),
    "emb.timestep_embedder.linear_1.weight": WeightMapping(
        target_path="emb.timestep_embedder.linear_1.weight",
        transpose=True,
    ),
    "emb.timestep_embedder.linear_1.bias": WeightMapping(
        target_path="emb.timestep_embedder.linear_1.bias",
    ),
    "emb.timestep_embedder.linear_2.weight": WeightMapping(
        target_path="emb.timestep_embedder.linear_2.weight",
        transpose=True,
    ),
    "emb.timestep_embedder.linear_2.bias": WeightMapping(
        target_path="emb.timestep_embedder.linear_2.bias",
    ),
    "emb.class_embedder.embedding_table.weight": WeightMapping(
        target_path="emb.class_embedder.embedding_table.embedding",
    ),
}

_ADALN_ZERO_SINGLE_MAPPINGS = {
    "linear.weight": WeightMapping(target_path="linear.weight", transpose=True),
    "linear.bias": WeightMapping(target_path="linear.bias"),
}

_ADALN_CONTINUOUS_MAPPINGS = {
    "linear.weight": WeightMapping(target_path="linear.weight", transpose=True),
    "linear.bias": WeightMapping(target_path="linear.bias"),
    "norm.weight": WeightMapping(target_path="norm.scale"),
    "norm.bias": WeightMapping(target_path="norm.bias"),
}


def _make_mesh():
    devices = np.array(jax.devices("cpu")[:1]).reshape((1, 1))
    try:
        return jax.sharding.Mesh(
            devices,
            ("data", "tensor"),
            axis_types=(
                jax.sharding.AxisType.Explicit,
                jax.sharding.AxisType.Explicit,
            ),
        )
    except TypeError:
        return jax.sharding.Mesh(devices, ("data", "tensor"))


def _mesh_context(mesh):
    try:
        return jax.sharding.use_mesh(mesh)
    except AttributeError:
        try:
            return jax.set_mesh(mesh)
        except AttributeError:
            return nullcontext()


@unittest.skipIf(
    torch is None
    or HFAdaLayerNormContinuous is None
    or HFAdaLayerNormZero is None
    or HFAdaLayerNormZeroSingle is None
    or jax is None,
    "torch/diffusers/jax not installed",
)
class TestFluxAdaLayerNormParityDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = _make_mesh()

    def _assert_close(self, torch_output, jax_output, *, atol=1e-4, rtol=1e-4):
        if isinstance(torch_output, tuple):
            self.assertIsInstance(jax_output, tuple)
            self.assertEqual(len(torch_output), len(jax_output))
            for torch_item, jax_item in zip(torch_output, jax_output, strict=True):
                np.testing.assert_allclose(
                    torch_item.detach().cpu().numpy(),
                    np.asarray(jax_item),
                    atol=atol,
                    rtol=rtol,
                )
            return

        np.testing.assert_allclose(
            torch_output.detach().cpu().numpy(),
            np.asarray(jax_output),
            atol=atol,
            rtol=rtol,
        )

    def test_adalayernorm_zero_parity(self):
        torch.manual_seed(0)
        batch_size, seq_len, dim = 2, 512, 1024

        torch_mod = HFAdaLayerNormZero(embedding_dim=dim).cpu().eval()
        with _mesh_context(self.mesh):
            jax_mod = FluxAdaLayerNormZero(
                dim=dim,
                mesh=self.mesh,
                eps=1e-6,
                params_dtype=jnp.float32,
            )
        _copy_hf_state_dict_to_jax(torch_mod.state_dict(), jax_mod, _ADALN_ZERO_MAPPINGS)

        hidden_states_torch = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
        emb_torch = torch.randn(batch_size, dim, dtype=torch.float32)

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        emb_jax = jnp.asarray(emb_torch.detach().cpu().numpy())

        with torch.no_grad():
            torch_output = torch_mod(hidden_states_torch, emb=emb_torch)
        jax_output = jax_mod(hidden_states_jax, emb=emb_jax)

        self._assert_close(torch_output, jax_output)

    def test_adalayernorm_zero_fp32_layer_norm_parity(self):
        torch.manual_seed(0)
        batch_size, seq_len, dim = 2, 8, 128

        torch_mod = HFAdaLayerNormZero(embedding_dim=dim, norm_type="fp32_layer_norm").cpu().eval()
        with _mesh_context(self.mesh):
            jax_mod = FluxAdaLayerNormZero(
                dim=dim,
                mesh=self.mesh,
                norm_type="fp32_layer_norm",
                eps=1e-6,
                params_dtype=jnp.float32,
            )
        _copy_hf_state_dict_to_jax(torch_mod.state_dict(), jax_mod, _ADALN_ZERO_MAPPINGS)

        hidden_states_torch = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
        emb_torch = torch.randn(batch_size, dim, dtype=torch.float32)

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        emb_jax = jnp.asarray(emb_torch.detach().cpu().numpy())

        with torch.no_grad():
            torch_output = torch_mod(hidden_states_torch, emb=emb_torch)
        jax_output = jax_mod(hidden_states_jax, emb=emb_jax)

        self._assert_close(torch_output, jax_output)

    def test_adalayernorm_zero_combined_timestep_label_embeddings_parity(self):
        torch.manual_seed(0)
        batch_size, seq_len, dim = 2, 8, 128
        num_classes = 11

        torch_mod = HFAdaLayerNormZero(embedding_dim=dim, num_embeddings=num_classes).cpu().eval()
        with _mesh_context(self.mesh):
            jax_mod = FluxAdaLayerNormZero(
                dim=dim,
                mesh=self.mesh,
                num_embeddings=num_classes,
                eps=1e-6,
                params_dtype=jnp.float32,
            )
        _copy_hf_state_dict_to_jax(torch_mod.state_dict(), jax_mod, _ADALN_ZERO_MAPPINGS)

        hidden_states_torch = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
        timestep_torch = torch.randint(0, 1000, (batch_size,), dtype=torch.int64)
        class_labels_torch = torch.randint(0, num_classes, (batch_size,), dtype=torch.int64)

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        timestep_jax = jnp.asarray(timestep_torch.detach().cpu().numpy())
        class_labels_jax = jnp.asarray(class_labels_torch.detach().cpu().numpy())

        with torch.no_grad():
            torch_output = torch_mod(
                hidden_states_torch,
                timestep=timestep_torch,
                class_labels=class_labels_torch,
                hidden_dtype=torch.float32,
            )
        jax_output = jax_mod(
            hidden_states_jax,
            timestep=timestep_jax,
            class_labels=class_labels_jax,
            hidden_dtype=jnp.float32,
        )

        self._assert_close(torch_output, jax_output)

    def test_adalayernorm_zero_single_parity(self):
        torch.manual_seed(0)
        batch_size, seq_len, dim = 2, 8, 128

        torch_mod = HFAdaLayerNormZeroSingle(embedding_dim=dim).cpu().eval()
        with _mesh_context(self.mesh):
            jax_mod = FluxAdaLayerNormZeroSingle(
                dim=dim,
                mesh=self.mesh,
                eps=1e-6,
                params_dtype=jnp.float32,
            )
        _copy_hf_state_dict_to_jax(torch_mod.state_dict(), jax_mod, _ADALN_ZERO_SINGLE_MAPPINGS)

        hidden_states_torch = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
        emb_torch = torch.randn(batch_size, dim, dtype=torch.float32)

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        emb_jax = jnp.asarray(emb_torch.detach().cpu().numpy())

        with torch.no_grad():
            torch_output = torch_mod(hidden_states_torch, emb=emb_torch)
        jax_output = jax_mod(hidden_states_jax, emb=emb_jax)

        self._assert_close(torch_output, jax_output)

    def test_adalayernorm_continuous_parity(self):
        torch.manual_seed(0)
        batch_size, seq_len, dim = 2, 8, 128

        torch_mod = (
            HFAdaLayerNormContinuous(
                embedding_dim=dim,
                conditioning_embedding_dim=dim,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                norm_type="layer_norm",
            )
            .cpu()
            .eval()
        )
        with _mesh_context(self.mesh):
            jax_mod = FluxAdaLayerNormContinuous(
                embedding_dim=dim,
                conditioning_embedding_dim=dim,
                mesh=self.mesh,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                params_dtype=jnp.float32,
            )
        _copy_hf_state_dict_to_jax(torch_mod.state_dict(), jax_mod, _ADALN_CONTINUOUS_MAPPINGS)

        hidden_states_torch = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
        conditioning_torch = torch.randn(batch_size, dim, dtype=torch.float32)

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        conditioning_jax = jnp.asarray(conditioning_torch.detach().cpu().numpy())

        with torch.no_grad():
            torch_output = torch_mod(hidden_states_torch, conditioning_torch)
        jax_output = jax_mod(hidden_states_jax, conditioning_jax)

        self._assert_close(torch_output, jax_output)

    def test_adalayernorm_continuous_rms_norm_parity(self):
        torch.manual_seed(0)
        batch_size, seq_len, dim = 2, 8, 128

        torch_mod = (
            HFAdaLayerNormContinuous(
                embedding_dim=dim,
                conditioning_embedding_dim=dim,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                norm_type="rms_norm",
            )
            .cpu()
            .eval()
        )
        with _mesh_context(self.mesh):
            jax_mod = FluxAdaLayerNormContinuous(
                embedding_dim=dim,
                conditioning_embedding_dim=dim,
                mesh=self.mesh,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                norm_type="rms_norm",
                params_dtype=jnp.float32,
            )
        _copy_hf_state_dict_to_jax(torch_mod.state_dict(), jax_mod, _ADALN_CONTINUOUS_MAPPINGS)

        hidden_states_torch = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
        conditioning_torch = torch.randn(batch_size, dim, dtype=torch.float32)

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        conditioning_jax = jnp.asarray(conditioning_torch.detach().cpu().numpy())

        with torch.no_grad():
            torch_output = torch_mod(hidden_states_torch, conditioning_torch)
        jax_output = jax_mod(hidden_states_jax, conditioning_jax)

        self._assert_close(torch_output, jax_output)


if __name__ == "__main__":
    unittest.main()
