import glob
import json
import os
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from safetensors import safe_open
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sgl_jax.srt.utils.mesh_utils import create_device_mesh

MIMO_MODEL_PATH = "/models/MiMo-V2.5"
HF_VISION_PREFIX = "visual."

E2E_GRID_CASES = {
    "mixed_temporal_and_multiple_segments": ((1, 4, 6), (2, 2, 2)),
    "regular_temporal_video": ((2, 4, 4),),
    "large_grid_crosses_window_boundary": ((1, 16, 16),),
    "wide_aspect_ratio_col_order": ((1, 2, 32),),
}


@unittest.skipUnless(
    os.path.exists(MIMO_MODEL_PATH), f"MiMo model path not found: {MIMO_MODEL_PATH}"
)
class TestMiMoVisionEncoderE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sgl_jax.srt.models.mimo_v2_5.vision_encoder import MiMoVisionTransformer

        vision_config = load_vision_config_dict()
        hf_config = make_hf_vision_config(vision_config)
        cls.hf_transformer = load_hf_vision_transformer(MIMO_MODEL_PATH, hf_config)
        cls.config = make_checkpoint_vision_config(vision_config, cls.hf_transformer)
        cls.mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
        with jax.set_mesh(cls.mesh):
            cls.jax_transformer = MiMoVisionTransformer(
                cls.config,
                norm_eps=1e-6,
                dtype=jnp.float32,
                rngs=nnx.Rngs(0),
            )
            cls.jax_transformer.load_weights_from_safetensors(MIMO_MODEL_PATH, cls.config)

    def _run(self, grid_thw):
        config = self.config
        torch_grid_thw = torch.tensor(grid_thw, dtype=torch.int32)

        patch_dim = (
            config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
        )
        total_tokens = sum(t * h * w for t, h, w in grid_thw)
        rng = np.random.default_rng(13)
        pixel_values = rng.normal(size=(total_tokens, patch_dim)).astype(np.float32)

        with torch.no_grad():
            torch_output = self.hf_transformer(torch.from_numpy(pixel_values), torch_grid_thw)
            if isinstance(torch_output, tuple):
                torch_output = torch_output[0]

        with jax.set_mesh(self.mesh), jax.default_matmul_precision("highest"):
            jax_output = self.jax_transformer(jnp.asarray(pixel_values), grid_thw)

        np.testing.assert_allclose(
            np.asarray(jax_output),
            torch_output.detach().cpu().numpy(),
            rtol=1e-3,
            atol=1e-5,
        )

    def test_checkpoint_weights_match_hf_remote_code_across_grid_shapes(self):
        for case_name, grid_thw in E2E_GRID_CASES.items():
            with self.subTest(case=case_name, grid_thw=grid_thw):
                self._run(grid_thw)


def load_vision_config_dict(model_path: str = MIMO_MODEL_PATH) -> dict:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise unittest.SkipTest(f"MiMo config not found: {config_path}")

    with open(config_path) as f:
        raw_config = json.load(f)
    vision_config = raw_config.get("vision_config")
    if vision_config is None:
        raise unittest.SkipTest(f"MiMo vision_config not found in {config_path}")
    return vision_config


def make_hf_vision_config(vision_config: dict):
    return SimpleNamespace(**vision_config)


def make_checkpoint_vision_config(
    vision_config: dict,
    hf_transformer,
    model_path: str = MIMO_MODEL_PATH,
):
    depth = int(vision_config["depth"])
    qk_channels = int(hf_transformer.blocks[0].attn.head_dim)
    hf_head_dims = {int(block.attn.head_dim) for block in hf_transformer.blocks}
    if hf_head_dims != {qk_channels}:
        raise AssertionError(f"MiMo vision blocks use inconsistent head_dim values: {hf_head_dims}")
    config = SimpleNamespace(
        depth=depth,
        hidden_size=int(vision_config["hidden_size"]),
        hidden_act=vision_config["hidden_act"],
        intermediate_size=int(vision_config["intermediate_size"]),
        num_heads=int(vision_config["num_heads"]),
        num_key_value_heads=int(vision_config["num_key_value_heads"]),
        in_channels=int(vision_config["in_chans"]),
        patch_size=int(vision_config["patch_size"]),
        spatial_merge_size=int(vision_config["spatial_merge_size"]),
        temporal_patch_size=int(vision_config["temporal_patch_size"]),
        tokens_per_second=vision_config["tokens_per_second"],
        window_size=int(vision_config["window_size"]),
        out_hidden_size=int(vision_config["out_hidden_size"]),
        fullatt_block_indexes=vision_config["fullatt_block_indexes"],
        kv_channels=qk_channels,
        qk_channels=qk_channels,
        num_query_groups=int(vision_config["num_query_groups"]),
        vit_window_attn_types=vision_config["vit_window_attn_types"],
        visual_token_window_size=int(vision_config["visual_token_window_size"]),
        use_sink=bool(vision_config["use_sink"]),
    )
    assert_checkpoint_shapes_match_config(config, model_path)
    return config


def load_hf_vision_transformer(model_path: str, config):
    transformer_cls = get_class_from_dynamic_module(
        "modeling_mimo_v2.MiMoVisionTransformer",
        model_path,
        local_files_only=True,
    )
    hf_transformer = transformer_cls(config).eval().to(torch.float32)

    hf_state_dict = load_hf_vision_state_dict(model_path)
    missing, unexpected = hf_transformer.load_state_dict(hf_state_dict, strict=False)
    if unexpected:
        raise AssertionError(f"Unexpected HF vision keys from checkpoint: {unexpected}")

    with torch.no_grad():
        for name in missing:
            if not name.endswith(".bias"):
                raise AssertionError(f"Missing non-bias HF vision weight: {name}")
            hf_transformer.get_parameter(name).zero_()

    return hf_transformer


def load_hf_vision_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for filename in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                if not key.startswith(HF_VISION_PREFIX):
                    continue
                state_dict[key[len(HF_VISION_PREFIX) :]] = handle.get_tensor(key).to(torch.float32)
    return state_dict


def assert_checkpoint_shapes_match_config(config, model_path: str):
    patch_weight = load_unique_weight(
        (
            "visual.patch_embed.proj.weight",
            "patch_embed.proj.weight",
        ),
        model_path,
    )
    expected_patch_shape = (
        config.hidden_size,
        config.in_channels,
        config.temporal_patch_size,
        config.patch_size,
        config.patch_size,
    )
    if tuple(patch_weight.shape) != expected_patch_shape:
        raise AssertionError(
            f"Patch embedding weight shape {tuple(patch_weight.shape)} does not match "
            f"config-derived shape {expected_patch_shape}"
        )

    merger_weights = load_merger_weights(model_path)
    if int(merger_weights["fc2_weight"].shape[0]) != config.out_hidden_size:
        raise AssertionError(
            f"Patch merger output size {merger_weights['fc2_weight'].shape[0]} does not match "
            f"config out_hidden_size {config.out_hidden_size}"
        )

    attn_weights = load_attention_weights(0, model_path)
    q_size = config.num_heads * config.qk_channels
    kv_size = config.num_key_value_heads * config.qk_channels
    expected_qkv_shape = (q_size + 2 * kv_size, config.hidden_size)
    expected_proj_shape = (config.hidden_size, q_size)
    if tuple(attn_weights["qkv_weight"].shape) != expected_qkv_shape:
        raise AssertionError(
            f"Attention qkv weight shape {tuple(attn_weights['qkv_weight'].shape)} does not "
            f"match config-derived shape {expected_qkv_shape}"
        )
    if tuple(attn_weights["proj_weight"].shape) != expected_proj_shape:
        raise AssertionError(
            f"Attention proj weight shape {tuple(attn_weights['proj_weight'].shape)} does not "
            f"match config-derived shape {expected_proj_shape}"
        )


def load_unique_weight(
    candidate_suffixes: tuple[str, ...],
    model_path: str = MIMO_MODEL_PATH,
) -> torch.Tensor:
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise unittest.SkipTest(f"No safetensors found under {model_path}")

    matches = []
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if any(
                    key == suffix or key.endswith(f".{suffix}") for suffix in candidate_suffixes
                ):
                    matches.append((sf_file, key))

    if not matches:
        raise unittest.SkipTest(f"Could not find any of these weights: {candidate_suffixes}")
    if len(matches) > 1:
        exact_suffix_matches = [
            match
            for match in matches
            if any(match[1].endswith(suffix) for suffix in candidate_suffixes)
        ]
        if len(exact_suffix_matches) == 1:
            matches = exact_suffix_matches
        else:
            raise AssertionError(f"Ambiguous weights: {[key for _, key in matches]}")

    sf_file, key = matches[0]
    with safe_open(sf_file, framework="pt", device="cpu") as f:
        return f.get_tensor(key).to(torch.float32)


def load_optional_weight(
    candidate_suffixes: tuple[str, ...],
    model_path: str = MIMO_MODEL_PATH,
) -> torch.Tensor | None:
    try:
        return load_unique_weight(candidate_suffixes, model_path)
    except unittest.SkipTest:
        return None


def load_merger_weights(model_path: str = MIMO_MODEL_PATH):
    return {
        "fc2_weight": load_unique_weight(("visual.merger.mlp.2.weight",), model_path),
    }


def load_attention_weights(block_idx: int, model_path: str = MIMO_MODEL_PATH):
    prefix = f"visual.blocks.{block_idx}.attn"
    return {
        "qkv_weight": load_unique_weight((f"{prefix}.qkv.weight",), model_path),
        "proj_weight": load_unique_weight((f"{prefix}.proj.weight",), model_path),
        "sinks": load_optional_weight((f"{prefix}.sinks",), model_path),
    }


if __name__ == "__main__":
    unittest.main()
