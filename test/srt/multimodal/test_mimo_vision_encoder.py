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
        from sgl_jax.srt.multimodal.models.mimo_vision.vision_encoder import (
            MiMoVisionTransformer,
        )

        vision_config = load_vision_config_dict()
        cls.config = make_checkpoint_vision_config(vision_config)
        cls.hf_transformer = load_hf_vision_transformer(MIMO_MODEL_PATH, cls.config)
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


def make_checkpoint_vision_config(vision_config: dict, model_path: str = MIMO_MODEL_PATH):
    patch_weight = load_unique_weight(
        (
            "visual.patch_embed.proj.weight",
            "patch_embed.proj.weight",
        ),
        model_path,
    )
    merger_weights = load_merger_weights(model_path)
    depth = int(vision_config["depth"])
    fullatt_block_indexes = vision_config.get("fullatt_block_indexes") or []
    first_sink_block = next(
        (
            block_idx
            for block_idx in range(depth)
            if load_attention_weights(block_idx, model_path)["sinks"] is not None
        ),
        0,
    )
    attn_weights = load_attention_weights(first_sink_block, model_path)
    hidden_size, num_heads, num_kv_heads, head_dim = infer_attention_dims(
        vision_config, attn_weights
    )
    mlp_weights = load_mlp_weights(0, model_path)
    return SimpleNamespace(
        depth=depth,
        hidden_size=hidden_size,
        hidden_act=vision_config.get("hidden_act", "silu"),
        intermediate_size=int(mlp_weights["gate_weight"].shape[0]),
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        in_channels=int(patch_weight.shape[1]),
        patch_size=int(patch_weight.shape[3]),
        spatial_merge_size=int(vision_config.get("spatial_merge_size", 2)),
        temporal_patch_size=int(patch_weight.shape[2]),
        tokens_per_second=vision_config.get("tokens_per_second", 2),
        window_size=int(vision_config.get("window_size", 128)),
        out_hidden_size=int(merger_weights["fc2_weight"].shape[0]),
        fullatt_block_indexes=fullatt_block_indexes,
        initializer_range=vision_config.get("initializer_range", 0.02),
        kv_channels=head_dim,
        qk_channels=head_dim,
        num_query_groups=num_heads // num_kv_heads,
        vit_window_attn_types=vision_config.get("vit_window_attn_types") or [0] * depth,
        visual_token_window_size=int(vision_config.get("visual_token_window_size", 64)),
        use_sink=True,
    )


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


def load_mlp_weights(block_idx: int = 0, model_path: str = MIMO_MODEL_PATH):
    prefix = f"visual.blocks.{block_idx}.mlp"
    return {
        "gate_weight": load_unique_weight((f"{prefix}.gate_proj.weight",), model_path),
        "up_weight": load_unique_weight((f"{prefix}.up_proj.weight",), model_path),
    }


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


def infer_attention_dims(vision_config: dict, weights: dict[str, torch.Tensor]):
    hidden_size = int(weights["qkv_weight"].shape[1])
    qkv_size = int(weights["qkv_weight"].shape[0])
    q_size = int(weights["proj_weight"].shape[1])
    kv_total_size = qkv_size - q_size
    if kv_total_size <= 0 or kv_total_size % 2 != 0:
        raise AssertionError(
            f"Invalid MiMo attention qkv/proj shapes: qkv={weights['qkv_weight'].shape}, "
            f"proj={weights['proj_weight'].shape}"
        )
    kv_size = kv_total_size // 2

    if weights["sinks"] is not None:
        num_heads = int(weights["sinks"].shape[0])
        if q_size % num_heads != 0:
            raise AssertionError(f"Cannot infer head_dim from q_size={q_size}, sinks={num_heads}")
        head_dim = q_size // num_heads
    else:
        head_dim = int(
            vision_config.get("qk_channels")
            or vision_config.get("head_dim")
            or vision_config.get("kv_channels")
            or np.gcd(q_size, kv_size)
        )
        num_heads = q_size // head_dim
    if q_size % head_dim != 0 or kv_size % head_dim != 0:
        raise AssertionError(
            f"Cannot infer attention heads from q_size={q_size}, kv_size={kv_size}, "
            f"head_dim={head_dim}"
        )

    return hidden_size, num_heads, kv_size // head_dim, head_dim


if __name__ == "__main__":
    unittest.main()
