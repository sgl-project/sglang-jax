from __future__ import annotations

import gc
import json
import os
import re
import unittest
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import numpy as np

try:
    import torch
    from diffusers.models.transformers.transformer_flux import (
        FluxTransformer2DModel as HFFluxTransformer2DModel,
    )
except ImportError:  # pragma: no cover
    torch = None
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
    from sgl_jax.srt.multimodal.configs.dits.flux_model_config import FluxModelConfig
    from sgl_jax.srt.multimodal.models.dits.flux import (
        FluxTransformer2DModel as JaxFluxTransformer2DModel,
    )
    from sgl_jax.srt.multimodal.models.dits.flux_weights_mapping import to_mappings
    from sgl_jax.srt.utils.weight_utils import WeightMapping


MODEL_ROOT = Path(os.environ.get("FLUX_MODEL_PATH", "/models/FLUX1.0"))
TRANSFORMER_PATH = MODEL_ROOT / "transformer"
CONFIG_PATH = TRANSFORMER_PATH / "config.json"


def _make_text_ids(seq_len: int) -> np.ndarray:
    ids = np.zeros((seq_len, 3), dtype=np.int64)
    ids[:, 0] = np.arange(seq_len, dtype=np.int64)
    return ids


def _make_image_ids(seq_len: int) -> np.ndarray:
    side = int(np.ceil(np.sqrt(seq_len)))
    row = np.arange(seq_len, dtype=np.int64) // side
    col = np.arange(seq_len, dtype=np.int64) % side
    return np.stack([np.zeros(seq_len, dtype=np.int64), row, col], axis=-1)


def _load_flux_config_from_local_checkpoint() -> FluxModelConfig:
    with CONFIG_PATH.open() as f:
        config_json = json.load(f)

    init_kwargs = {k: v for k, v in config_json.items() if not k.startswith("_")}
    return FluxModelConfig(
        **init_kwargs,
        model_path=str(TRANSFORMER_PATH),
        dtype=jnp.float32,
        weights_dtype=jnp.float32,
        attention_impl="sdpa",
    )


def set_jax_param(model, path: str, value: np.ndarray) -> None:
    current = nnx.state(model)
    for key in path.split("."):
        current = current[int(key)] if key.isdigit() else current[key]
    target = current[...]
    array = jnp.asarray(value, dtype=target.dtype)
    sharding = getattr(target, "sharding", None)
    if sharding is not None:
        array = jax.device_put(array, sharding)
    current[...] = array


def _resolve_mapping(
    hf_key: str,
    mappings: Mapping[str, WeightMapping],
) -> tuple[WeightMapping, str] | tuple[None, None]:
    for pattern, candidate in mappings.items():
        if "*" not in pattern:
            if hf_key != pattern:
                continue
            target_path = candidate.target_path
            if isinstance(target_path, list):
                return None, None
            return candidate, target_path

        match = re.fullmatch(re.escape(pattern).replace(r"\*", r"(.*?)"), hf_key)
        if match is None:
            continue
        target_path = candidate.target_path
        if isinstance(target_path, list):
            return None, None
        return candidate, target_path.replace("*", "{}").format(*match.groups())

    return None, None


def copy_hf_state_dict_to_jax(
    pt_state_dict,
    jax_model,
    mappings: Mapping[str, WeightMapping],
    *,
    hf_prefix: str = "",
    target_prefix: str = "",
) -> None:
    hf_prefix = f"{hf_prefix}." if hf_prefix else ""
    target_prefix = f"{target_prefix}." if target_prefix else ""

    for hf_key, tensor in pt_state_dict.items():
        lookup_key = f"{hf_prefix}{hf_key}"
        mapping, target_path = _resolve_mapping(lookup_key, mappings)
        if mapping is None or target_path is None:
            continue

        if target_prefix:
            if not target_path.startswith(target_prefix):
                continue
            target_path = target_path[len(target_prefix) :]

        weight = tensor.detach().cpu().float().numpy()
        if mapping.transpose_axes is not None and not lookup_key.endswith(".bias"):
            weight = np.transpose(weight, mapping.transpose_axes)
        elif mapping.transpose and not lookup_key.endswith(".bias"):
            weight = np.transpose(weight, (1, 0))

        set_jax_param(jax_model, target_path, weight)


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
    torch is None or HFFluxTransformer2DModel is None or jax is None,
    "torch/diffusers/jax not installed",
)
class TestFluxTransformer2DModelParityDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = _make_mesh()

    def _assert_close(self, actual, expected, *, name: str, atol: float, rtol: float) -> None:
        np.testing.assert_allclose(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            err_msg=f"{name} mismatch",
        )

    def test_flux_transformer_2d_model_random_init_state_dict_parity(self):
        torch.manual_seed(0)
        batch_size = 8
        image_seq_len = 128
        text_seq_len = 256

        hf_model = (
            HFFluxTransformer2DModel(
                patch_size=1,
                in_channels=16,
                out_channels=16,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=8,
                num_attention_heads=4,
                joint_attention_dim=20,
                pooled_projection_dim=12,
                guidance_embeds=False,
                axes_dims_rope=(2, 2, 4),
            )
            .cpu()
            .eval()
        )

        config = FluxModelConfig(
            patch_size=1,
            in_channels=16,
            out_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=4,
            joint_attention_dim=20,
            pooled_projection_dim=12,
            guidance_embeds=False,
            axes_dims_rope=(2, 2, 4),
            dtype=jnp.float32,
            weights_dtype=jnp.float32,
            attention_impl="sdpa",
        )
        with _mesh_context(self.mesh):
            jax_model = JaxFluxTransformer2DModel(
                config,
                dtype=jnp.float32,
                mesh=self.mesh,
            )
        copy_hf_state_dict_to_jax(
            hf_model.state_dict(),
            jax_model,
            to_mappings(has_guidance_embeds=config.guidance_embeds),
        )

        hidden_states_torch = torch.randn(batch_size, image_seq_len, 16, dtype=torch.float32)
        encoder_hidden_states_torch = torch.randn(batch_size, text_seq_len, 20, dtype=torch.float32)
        pooled_projections_torch = torch.randn(batch_size, 12, dtype=torch.float32)
        timestep_torch = torch.tensor([0, 1, 2, 5, 42, 150, 951, 999], dtype=torch.int64)
        txt_ids_torch = torch.from_numpy(_make_text_ids(text_seq_len))
        img_ids_torch = torch.from_numpy(_make_image_ids(image_seq_len))

        hidden_states_jax = jnp.asarray(hidden_states_torch.detach().cpu().numpy())
        encoder_hidden_states_jax = jnp.asarray(encoder_hidden_states_torch.detach().cpu().numpy())
        pooled_projections_jax = jnp.asarray(pooled_projections_torch.detach().cpu().numpy())
        timestep_jax = jnp.asarray(timestep_torch.detach().cpu().numpy())
        txt_ids_jax = jnp.asarray(txt_ids_torch.detach().cpu().numpy())
        img_ids_jax = jnp.asarray(img_ids_torch.detach().cpu().numpy())

        with torch.no_grad():
            torch_output = hf_model(
                hidden_states=hidden_states_torch,
                encoder_hidden_states=encoder_hidden_states_torch,
                pooled_projections=pooled_projections_torch,
                timestep=timestep_torch,
                img_ids=img_ids_torch,
                txt_ids=txt_ids_torch,
                return_dict=False,
            )[0]

        jax_output = jax_model(
            hidden_states=hidden_states_jax,
            encoder_hidden_states=encoder_hidden_states_jax,
            pooled_projections=pooled_projections_jax,
            timestep=timestep_jax,
            img_ids=img_ids_jax,
            txt_ids=txt_ids_jax,
            return_dict=False,
        )[0]

        np.testing.assert_allclose(
            torch_output.detach().cpu().numpy(),
            np.asarray(jax_output),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_flux_transformer_2d_model_parity_after_loading_weights_cpu(self):
        if not CONFIG_PATH.exists():
            raise unittest.SkipTest(f"Config not found: {CONFIG_PATH}")

        if not hasattr(HFFluxTransformer2DModel, "from_pretrained"):
            raise unittest.SkipTest("diffusers FluxTransformer2DModel.from_pretrained unavailable")

        batch_size = 2
        image_seq_len = 256
        text_seq_len = 512

        np.random.seed(0)
        hidden_states_np = np.random.randn(batch_size, image_seq_len, 64).astype(np.float32)
        encoder_hidden_states_np = np.random.randn(batch_size, text_seq_len, 4096).astype(
            np.float32
        )
        pooled_projections_np = np.random.randn(batch_size, 768).astype(np.float32)
        timestep_np = np.array([0, 1], dtype=np.int64)
        guidance_np = np.full((batch_size,), 3.5, dtype=np.float32)
        txt_ids_np = _make_text_ids(text_seq_len)
        img_ids_np = _make_image_ids(image_seq_len)

        hidden_states_torch = torch.from_numpy(hidden_states_np)
        encoder_hidden_states_torch = torch.from_numpy(encoder_hidden_states_np)
        pooled_projections_torch = torch.from_numpy(pooled_projections_np)
        timestep_torch = torch.from_numpy(timestep_np)
        guidance_torch = torch.from_numpy(guidance_np)
        txt_ids_torch = torch.from_numpy(txt_ids_np)
        img_ids_torch = torch.from_numpy(img_ids_np)

        hf_model = HFFluxTransformer2DModel.from_pretrained(
            str(TRANSFORMER_PATH),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).cpu()
        hf_model.eval()

        with torch.no_grad():
            hf_ids = torch.cat((txt_ids_torch, img_ids_torch), dim=0)
            hf_pos_cos, hf_pos_sin = hf_model.pos_embed(hf_ids)

            scaled_timestep_torch = timestep_torch.to(hidden_states_torch.dtype) * 1000
            scaled_guidance_torch = guidance_torch.to(hidden_states_torch.dtype) * 1000
            hf_temb = hf_model.time_text_embed(
                scaled_timestep_torch,
                scaled_guidance_torch,
                pooled_projections_torch,
            )
            hf_hidden_embed = hf_model.x_embedder(hidden_states_torch)
            hf_context_embed = hf_model.context_embedder(encoder_hidden_states_torch)

            torch_output = hf_model(
                hidden_states=hidden_states_torch,
                encoder_hidden_states=encoder_hidden_states_torch,
                pooled_projections=pooled_projections_torch,
                timestep=timestep_torch,
                img_ids=img_ids_torch,
                txt_ids=txt_ids_torch,
                guidance=guidance_torch,
                return_dict=False,
            )[0]

        del hf_model
        gc.collect()

        config = _load_flux_config_from_local_checkpoint()
        with _mesh_context(self.mesh):
            jax_model = JaxFluxTransformer2DModel(
                config,
                dtype=config.dtype,
                mesh=self.mesh,
            )
            jax_model.load_weights(str(TRANSFORMER_PATH))

        hidden_states_jax = jnp.asarray(hidden_states_np, dtype=config.dtype)
        encoder_hidden_states_jax = jnp.asarray(encoder_hidden_states_np, dtype=config.dtype)
        pooled_projections_jax = jnp.asarray(pooled_projections_np, dtype=config.dtype)
        timestep_jax = jnp.asarray(timestep_np)
        guidance_jax = jnp.asarray(guidance_np, dtype=config.dtype)
        txt_ids_jax = jnp.asarray(txt_ids_np)
        img_ids_jax = jnp.asarray(img_ids_np)

        jax_ids = jnp.concatenate((txt_ids_jax, img_ids_jax), axis=0)
        jax_pos_cos, jax_pos_sin = jax_model.pos_embed(jax_ids)

        scaled_timestep_jax = timestep_jax.astype(config.dtype) * 1000
        scaled_guidance_jax = guidance_jax.astype(config.dtype) * 1000
        jax_temb = jax_model.time_text_embed(
            scaled_timestep_jax,
            scaled_guidance_jax,
            pooled_projections_jax,
        )
        jax_hidden_embed, _ = jax_model.x_embedder(hidden_states_jax)
        jax_context_embed, _ = jax_model.context_embedder(encoder_hidden_states_jax)

        self._assert_close(
            np.asarray(hf_pos_cos.detach().cpu().numpy(), dtype=np.float32),
            np.asarray(jax_pos_cos, dtype=np.float32),
            name="pos_embed.cos",
            atol=1e-4,
            rtol=1e-4,
        )
        self._assert_close(
            np.asarray(hf_pos_sin.detach().cpu().numpy(), dtype=np.float32),
            np.asarray(jax_pos_sin, dtype=np.float32),
            name="pos_embed.sin",
            atol=1e-4,
            rtol=1e-4,
        )
        self._assert_close(
            np.asarray(hf_temb.detach().cpu().numpy(), dtype=np.float32),
            np.asarray(jax_temb, dtype=np.float32),
            name="time_text_embed",
            atol=1e-4,
            rtol=1e-4,
        )
        self._assert_close(
            np.asarray(hf_hidden_embed.detach().cpu().numpy(), dtype=np.float32),
            np.asarray(jax_hidden_embed, dtype=np.float32),
            name="x_embedder",
            atol=1e-4,
            rtol=1e-4,
        )
        self._assert_close(
            np.asarray(hf_context_embed.detach().cpu().numpy(), dtype=np.float32),
            np.asarray(jax_context_embed, dtype=np.float32),
            name="context_embedder",
            atol=1e-4,
            rtol=1e-4,
        )

        jax_output = jax_model(
            hidden_states=hidden_states_jax,
            encoder_hidden_states=encoder_hidden_states_jax,
            pooled_projections=pooled_projections_jax,
            timestep=timestep_jax,
            img_ids=img_ids_jax,
            txt_ids=txt_ids_jax,
            guidance=guidance_jax,
            return_dict=False,
        )[0]

        np.testing.assert_allclose(
            torch_output.detach().cpu().numpy().astype(np.float32),
            np.asarray(jax_output, dtype=np.float32),
            atol=2e-4,
            rtol=2e-4,
        )


if __name__ == "__main__":
    unittest.main()
