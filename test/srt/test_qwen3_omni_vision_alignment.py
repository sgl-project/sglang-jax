# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Test alignment between JAX Qwen3OmniMoe Vision Encoder and PyTorch HuggingFace model."""

import argparse
import glob
import logging
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers import Qwen3OmniMoeThinkerConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)

from sgl_jax.srt.configs.model_config import _get_and_verify_dtype
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.qwen3_omni_thinker_embedding import (
    Qwen3OmniMoeThinkerEmbedding,
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DTYPE = {
    "float32": (torch.float32, jnp.float32),
    "bfloat16": (torch.bfloat16, jnp.bfloat16),
}


# =============================================================================
# Core Utilities
# =============================================================================


def to_numpy(x):
    """Convert PyTorch/JAX tensor to numpy float32."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.array(x, dtype=np.float32)


def compare(output1, output2, name, threshold=1e-3):
    """Compare two outputs and return (passed, mae)."""
    np1, np2 = to_numpy(output1), to_numpy(output2)

    if np1.shape != np2.shape:
        logger.error("❌ %s: Shape mismatch! %s vs %s", name, np1.shape, np2.shape)
        return False, float("inf")

    mae = np.abs(np1 - np2).mean()
    max_diff = np.abs(np1 - np2).max()
    passed = mae < threshold
    logger.info("%s %s: MAE=%.2e, Max=%.2e", "✅" if passed else "❌", name, mae, max_diff)
    return passed, mae


def log_stats(arr, prefix):
    """Log array statistics."""
    arr = to_numpy(arr)
    logger.info("%s shape: %s, mean=%.4f, std=%.4f", prefix, arr.shape, arr.mean(), arr.std())


# =============================================================================
# Input Preparation
# =============================================================================


def create_dummy_input(config, resolution="small"):
    """
    Create dummy input for testing.

    Returns:
        pixel_values: (1, T_input, H, W, C)
        grid_thw: (1, 3) - [T_output, H_patches, W_patches]
    """
    temporal_patch_size = getattr(config, "temporal_patch_size", 2)
    patch_size = config.patch_size

    resolutions = {
        "small": (temporal_patch_size, 8, 8),
        "medium": (temporal_patch_size, 14, 14),
        "large": (temporal_patch_size, 28, 28),
    }

    temporal_input, h_patches, w_patches = resolutions.get(
        resolution, (temporal_patch_size, 14, 14)
    )
    height, width = h_patches * patch_size, w_patches * patch_size

    np.random.seed(42)
    pixel_values = np.random.randn(1, temporal_input, height, width, config.in_channels).astype(
        np.float32
    )
    pixel_values = pixel_values * 0.5 + 0.5

    temporal_output = temporal_input // temporal_patch_size
    grid_thw = np.array([[temporal_output, h_patches, w_patches]], dtype=np.int32)

    total_patches = temporal_output * h_patches * w_patches
    logger.info(
        "Input: (%d, %d, %d), Patches: %dx%dx%d=%d",
        temporal_input,
        height,
        width,
        temporal_output,
        h_patches,
        w_patches,
        total_patches,
    )
    return pixel_values, grid_thw


def create_video_input(config, temporal_frames, h_patches, w_patches):
    """Create video input with specified dimensions."""
    temporal_patch_size = getattr(config, "temporal_patch_size", 2)
    patch_size = config.patch_size
    height, width = h_patches * patch_size, w_patches * patch_size

    np.random.seed(42)
    pixel_values = np.random.randn(1, temporal_frames, height, width, config.in_channels).astype(
        np.float32
    )
    pixel_values = pixel_values * 0.5 + 0.5

    temporal_output = temporal_frames // temporal_patch_size
    grid_thw = np.array([[temporal_output, h_patches, w_patches]], dtype=np.int32)
    return pixel_values, grid_thw


def prepare_pytorch_input(pixel_values, config, dtype):
    """
    Convert pixel_values to PyTorch patch format with spatial-merge permutation.

    Returns: (N_patches, C, T_patch, H_patch, W_patch) in spatial-merge order
    """
    pt_input = torch.from_numpy(pixel_values).to(dtype)
    B, T, H, W, C = pt_input.shape
    patch_size = config.patch_size
    temporal_patch_size = config.temporal_patch_size
    merge_size = config.spatial_merge_size

    # (B, T, H, W, C) -> (B, C, T, H, W)
    pt_input = pt_input.permute(0, 4, 1, 2, 3)

    # Extract patches
    pt_input = pt_input.reshape(
        B,
        C,
        T // temporal_patch_size,
        temporal_patch_size,
        H // patch_size,
        patch_size,
        W // patch_size,
        patch_size,
    )
    pt_input = pt_input.permute(0, 2, 4, 6, 1, 3, 5, 7)

    # Apply spatial-merge permutation
    T_out = T // temporal_patch_size
    H_patches, W_patches = H // patch_size, W // patch_size

    pt_input = pt_input.reshape(
        B,
        T_out,
        H_patches // merge_size,
        merge_size,
        W_patches // merge_size,
        merge_size,
        C,
        temporal_patch_size,
        patch_size,
        patch_size,
    )
    pt_input = pt_input.permute(0, 1, 2, 4, 3, 5, 6, 7, 8, 9)
    return pt_input.reshape(-1, C, temporal_patch_size, patch_size, patch_size)


def prepare_jax_input(pixel_values, config):
    """Flatten to (B, C*T*H*W) so reshape(-1, C, t, p, p) preserves patch order."""
    B, T, H, W, C = pixel_values.shape
    t_patch = config.temporal_patch_size
    p = config.patch_size
    t_out, h_patches, w_patches = T // t_patch, H // p, W // p

    # (B, T, H, W, C) -> (B, C, T, H, W)
    patches = np.transpose(pixel_values, (0, 4, 1, 2, 3))
    # (B, C, T, H, W) -> (B, C, T_out, t, H_patches, p, W_patches, p)
    patches = patches.reshape(
        B,
        C,
        t_out,
        t_patch,
        h_patches,
        p,
        w_patches,
        p,
    )
    # (B, T_out, H_patches, W_patches, C, t, p, p)
    patches = np.transpose(patches, (0, 2, 4, 6, 1, 3, 5, 7))
    return patches.reshape(B, -1)


# =============================================================================
# Model Loading
# =============================================================================


def set_param(model, path, val):
    """Set parameter value in JAX model by path."""
    parts = path.split(".")
    param = model
    for p in parts:
        param = param[int(p)] if p.isdigit() else getattr(param, p)

    if not isinstance(param, nnx.Variable):
        raise ValueError(f"Expected nnx.Variable at {path}, got {type(param)}")
    param[...] = jnp.array(val, dtype=param[...].dtype)


def load_pytorch_model(model_path, precision):
    """Load PyTorch vision encoder model."""
    pt_dtype = DTYPE[precision][0]

    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoderConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoder as PTVisionEncoder,
    )

    # Load config
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        pt_config = None

        # Try different config paths
        if hasattr(config, "vision_config") and config.vision_config:
            pt_config = config.vision_config
        elif hasattr(config, "thinker_config") and hasattr(config.thinker_config, "vision_config"):
            pt_config = config.thinker_config.vision_config

        if pt_config is None:
            config_dict = config.to_dict()
            if "thinker_config" in config_dict and config_dict["thinker_config"]:
                vision_dict = config_dict["thinker_config"].get("vision_config")
                if vision_dict:
                    pt_config = Qwen3OmniMoeVisionEncoderConfig(**vision_dict)
            if pt_config is None and "vision_config" in config_dict:
                pt_config = Qwen3OmniMoeVisionEncoderConfig(**config_dict["vision_config"])

        if pt_config is None:
            logger.warning("vision_config not found, using defaults")
            pt_config = Qwen3OmniMoeVisionEncoderConfig()
    except Exception as e:
        logger.warning("Could not load config: %s, using defaults", e)
        pt_config = Qwen3OmniMoeVisionEncoderConfig()

    pt_model = PTVisionEncoder(pt_config).eval().to(pt_dtype)
    logger.info(
        "PyTorch model: depth=%d, hidden_size=%d",
        pt_config.depth,
        pt_config.hidden_size,
    )

    # Load weights
    try:
        from safetensors import safe_open

        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if safetensor_files:
            all_weights = {}
            for sf_file in safetensor_files:
                with safe_open(sf_file, framework="pt") as f:
                    for key in f.keys():
                        if key.startswith("thinker.visual."):
                            all_weights[key.replace("thinker.visual.", "")] = (
                                sf_file,
                                key,
                            )

            pt_state = pt_model.state_dict()
            loaded = 0
            for model_key in pt_state:
                if model_key in all_weights:
                    sf_file, original_key = all_weights[model_key]
                    with safe_open(sf_file, framework="pt") as f:
                        pt_state[model_key] = f.get_tensor(original_key)
                        loaded += 1

            pt_model.load_state_dict(pt_state)
            logger.info("✅ Loaded %d/%d PyTorch weights", loaded, len(pt_state))
    except Exception as e:
        logger.warning("Failed to load PT weights: %s", e)

    return pt_model, pt_config


def load_jax_model(pt_config, model_path, mesh, precision):
    """Load JAX vision encoder model."""

    vision_config = Qwen3OmniMoeVisionEncoderConfig(
        depth=pt_config.depth,
        hidden_size=pt_config.hidden_size,
        hidden_act=getattr(pt_config, "hidden_act", "gelu"),
        intermediate_size=pt_config.intermediate_size,
        num_heads=pt_config.num_heads,
        in_channels=pt_config.in_channels,
        patch_size=pt_config.patch_size,
        spatial_merge_size=pt_config.spatial_merge_size,
        temporal_patch_size=pt_config.temporal_patch_size,
        out_hidden_size=pt_config.out_hidden_size,
        num_position_embeddings=pt_config.num_position_embeddings,
        deepstack_visual_indexes=list(pt_config.deepstack_visual_indexes),
    )
    jax_config = Qwen3OmniMoeThinkerConfig(vision_config=vision_config)

    # Convert string precision to JAX dtype
    jax_dtype = _get_and_verify_dtype(jax_config, precision)
    jax_config.model_path = model_path
    jax_config.revision = None
    jax_config.dtype = jax_dtype
    jax_config.model_class = Qwen3OmniMoeThinkerEmbedding

    with jax.set_mesh(mesh):
        jax_model = Qwen3OmniMoeThinkerEmbedding(
            config=jax_config, mesh=mesh, rngs=nnx.Rngs(0), dtype=jax_dtype
        )

    jax_model.load_weights(jax_config)
    logger.info("✅ Loaded JAX weights from %s", model_path)

    return jax_model.visual


def load_models(model_path, mesh, precision):
    """Load both PyTorch and JAX models."""
    try:
        pt_model, pt_config = load_pytorch_model(model_path, precision)
        jax_model = load_jax_model(pt_config, model_path, mesh, precision)
        return pt_model, jax_model, pt_config
    except ImportError as e:
        logger.error("Import error: %s", e)
        return None, None, None


# =============================================================================
# Forward Pass Helpers
# =============================================================================


class PTForwardContext:
    """Context for running PyTorch forward pass step by step."""

    def __init__(self, pt_model, pixel_values, grid_thw):
        self.model = pt_model
        self.dtype = next(pt_model.parameters()).dtype
        self.config = pt_model.config

        self.input_flat = prepare_pytorch_input(pixel_values, self.config, self.dtype)
        self.grid = torch.from_numpy(grid_thw)

        # Pre-computed values (lazy init)
        self._cu_seqlens = None
        self._position_embeddings = None

    @property
    def cu_seqlens(self):
        if self._cu_seqlens is None:
            cu = torch.repeat_interleave(self.grid[:, 1] * self.grid[:, 2], self.grid[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            )
            self._cu_seqlens = torch.nn.functional.pad(cu, (1, 0), value=0)
        return self._cu_seqlens

    @property
    def position_embeddings(self):
        if self._position_embeddings is None:
            rotary = self.model.rot_pos_emb(self.grid)
            seq_len = rotary.size(0)
            rotary = rotary.reshape(seq_len, -1)
            emb = torch.cat((rotary, rotary), dim=-1)
            self._position_embeddings = (emb.cos(), emb.sin())
        return self._position_embeddings

    def get_initial_hidden(self):
        """Get hidden states after patch_embed + pos_embed."""
        hidden = self.model.patch_embed(self.input_flat)
        pos_embeds = self.model.fast_pos_embed_interpolate(self.grid)
        hidden = hidden + pos_embeds
        seq_len = hidden.size(0)
        return hidden.reshape(seq_len, -1)

    def run_blocks(self, hidden, block_idxs=None):
        """Run through transformer blocks, optionally capturing intermediate outputs."""
        outputs = {}
        for idx, block in enumerate(self.model.blocks):
            hidden = block(
                hidden,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=self.position_embeddings,
            )
            if block_idxs is not None and idx in block_idxs:
                outputs[idx] = hidden.clone()
        return hidden, outputs

    def full_forward(self):
        """Run full forward pass and return pooler_output."""
        output = self.model(hidden_states=self.input_flat, grid_thw=self.grid, return_dict=True)
        return output.pooler_output if hasattr(output, "pooler_output") else output[0]


class JAXForwardContext:
    """Context for running JAX forward pass step by step."""

    def __init__(self, jax_model, pixel_values, grid_thw, mesh):
        self.model = jax_model
        self.mesh = mesh
        self.pixel_values = jnp.array(prepare_jax_input(pixel_values, jax_model.config))
        self.grid = jnp.array(grid_thw)

        # Pre-computed values (lazy init)
        self._position_ids = None

    @property
    def position_ids(self):
        if self._position_ids is None:
            with jax.set_mesh(self.mesh):
                self._position_ids = self.model.compute_2d_position_ids(self.grid)
        return self._position_ids

    def get_initial_hidden(self):
        """Get hidden states after patch_embed + pos_embed (with spatial-merge permutation)."""
        with jax.set_mesh(self.mesh):
            patch_out = self.model.patch_embed(self.pixel_values)
            patch_permuted = self.model._apply_spatial_merge_permutation(patch_out, self.grid)
            pos_embeds = self.model.interpolate_pos_embed(self.grid)
            return patch_permuted + pos_embeds

    def run_blocks(self, hidden, block_idxs=None):
        """Run through transformer blocks, optionally capturing intermediate outputs."""
        outputs = {}
        with jax.set_mesh(self.mesh):
            for idx, block in enumerate(self.model.blocks):
                hidden = block(
                    hidden_states=hidden,
                    position_ids=self.position_ids,
                    attention_mask=None,
                )
                if block_idxs is not None and idx in block_idxs:
                    outputs[idx] = hidden
        return hidden, outputs

    def full_forward(self):
        """Run full forward pass and return output dict."""
        with jax.set_mesh(self.mesh):
            return self.model(pixel_values=self.pixel_values, grid_thw=self.grid)


# =============================================================================
# Tests
# =============================================================================


def test_forward_basic(model_path, mesh, precision):
    """Test basic forward pass with small resolution."""
    logger.info("\n" + "=" * 60 + "\nTest: Forward Pass (Basic)\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    pixel_values, grid_thw = create_dummy_input(config, resolution="small")

    with torch.no_grad():
        pt_ctx = PTForwardContext(pt_model, pixel_values, grid_thw)
        pt_pooler = pt_ctx.full_forward()

    jax_ctx = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh)
    jax_output = jax_ctx.full_forward()

    log_stats(pt_pooler, "PyTorch")
    log_stats(jax_output["pooler_output"], "JAX")

    threshold = 2e-3 if precision == "bfloat16" else 1e-3
    return compare(pt_pooler, jax_output["pooler_output"], "Pooler Output", threshold)


def test_forward_medium(model_path, mesh, precision):
    """Test forward pass with medium resolution (224x224)."""
    logger.info("\n" + "=" * 60 + "\nTest: Forward Pass (Medium)\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    pixel_values, grid_thw = create_dummy_input(config, resolution="medium")

    with torch.no_grad():
        pt_pooler = PTForwardContext(pt_model, pixel_values, grid_thw).full_forward()

    jax_output = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh).full_forward()

    threshold = 2e-3 if precision == "bfloat16" else 1e-3
    return compare(pt_pooler, jax_output["pooler_output"], "Pooler Output (Medium)", threshold)


def test_large_resolution(model_path, mesh, precision):
    """Test with large resolution (336x336)."""
    logger.info("\n" + "=" * 60 + "\nTest: Large Resolution\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    pixel_values, grid_thw = create_dummy_input(config, resolution="large")

    with torch.no_grad():
        pt_pooler = PTForwardContext(pt_model, pixel_values, grid_thw).full_forward()

    jax_output = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh).full_forward()

    threshold = 2e-3 if precision == "bfloat16" else 1e-3
    return compare(pt_pooler, jax_output["pooler_output"], "Large Resolution Pooler", threshold)


def test_deepstack_features(model_path, mesh, precision):
    """Test deepstack feature extraction."""
    logger.info("\n" + "=" * 60 + "\nTest: Deepstack Features\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    pixel_values, grid_thw = create_dummy_input(config, resolution="small")

    with torch.no_grad():
        pt_ctx = PTForwardContext(pt_model, pixel_values, grid_thw)
        pt_output = pt_model(
            hidden_states=pt_ctx.input_flat, grid_thw=pt_ctx.grid, return_dict=True
        )

    jax_output = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh).full_forward()

    if not hasattr(pt_output, "deepstack_features") or not pt_output.deepstack_features:
        logger.warning("PyTorch model doesn't return deepstack_features")
        return True, 0.0

    all_passed, total_mae = True, 0.0
    threshold = 2e-3 if precision == "bfloat16" else 1e-3

    for idx, (pt_feat, jax_feat) in enumerate(
        zip(pt_output.deepstack_features, jax_output["deepstack_features"])
    ):
        passed, mae = compare(pt_feat, jax_feat, f"Deepstack Feature {idx}", threshold)
        all_passed &= passed
        total_mae += mae

    return all_passed, total_mae / len(pt_output.deepstack_features)


def test_hidden_states(model_path, mesh, precision):
    """Test pre-merger hidden states alignment."""
    logger.info("\n" + "=" * 60 + "\nTest: Hidden States (Pre-Merger)\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    pixel_values, grid_thw = create_dummy_input(config, resolution="small")

    with torch.no_grad():
        pt_ctx = PTForwardContext(pt_model, pixel_values, grid_thw)
        pt_hidden_before = pt_ctx.get_initial_hidden()
        pt_hidden_after, _ = pt_ctx.run_blocks(pt_hidden_before)

    jax_ctx = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh)
    jax_hidden_before = jax_ctx.get_initial_hidden()
    jax_hidden_after, _ = jax_ctx.run_blocks(jax_hidden_before)

    log_stats(pt_hidden_before, "PT Before blocks")
    log_stats(jax_hidden_before, "JAX Before blocks")
    log_stats(pt_hidden_after, "PT After blocks")
    log_stats(jax_hidden_after, "JAX After blocks")

    pre_passed, _ = compare(pt_hidden_before, jax_hidden_before, "Pre-Block Hidden", 1e-3)
    threshold = 5e-3 if precision == "bfloat16" else 2e-3
    post_passed, mae = compare(pt_hidden_after, jax_hidden_after, "Post-Block Hidden", threshold)

    return pre_passed and post_passed, mae


def test_layer_outputs(model_path, mesh, precision):
    """Test layer-by-layer output alignment."""
    logger.info("\n" + "=" * 60 + "\nTest: Layer-by-Layer Outputs\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    pixel_values, grid_thw = create_dummy_input(config, resolution="small")
    all_passed, results = True, []

    with torch.no_grad():
        pt_ctx = PTForwardContext(pt_model, pixel_values, grid_thw)
        pt_patch = pt_model.patch_embed(pt_ctx.input_flat)
        pt_pos = pt_model.fast_pos_embed_interpolate(pt_ctx.grid)
        pt_hidden = pt_patch + pt_pos

    jax_ctx = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh)
    with jax.set_mesh(mesh):
        jax_patch_raw = jax_model.patch_embed(jax_ctx.pixel_values)
        jax_patch = jax_model._apply_spatial_merge_permutation(jax_patch_raw, jax_ctx.grid)
        jax_pos = jax_model.interpolate_pos_embed(jax_ctx.grid)
        jax_hidden = jax_patch + jax_pos

    # Compare each layer
    layers = [
        ("Patch Embed", pt_patch, jax_patch),
        ("Pos Embed", pt_pos, jax_pos),
        ("Hidden+Pos", pt_hidden, jax_hidden),
    ]

    for name, pt_out, jax_out in layers:
        passed, mae = compare(pt_out, jax_out, name, 1e-2)
        results.append((name, passed, mae))
        all_passed &= passed

    # Test first block
    with torch.no_grad():
        pt_block0 = pt_model.blocks[0](
            pt_hidden.reshape(pt_hidden.size(0), -1),
            cu_seqlens=pt_ctx.cu_seqlens,
            position_embeddings=pt_ctx.position_embeddings,
        )

    with jax.set_mesh(mesh):
        jax_block0 = jax_model.blocks[0](
            hidden_states=jax_hidden,
            position_ids=jax_ctx.position_ids,
            attention_mask=None,
        )

    passed, mae = compare(pt_block0, jax_block0, "Block[0]", 1e-1)
    results.append(("Block[0]", passed, mae))
    all_passed &= passed

    avg_mae = sum(r[2] for r in results if r[2] != float("inf")) / len(results)
    return all_passed, avg_mae


def test_weights_alignment(model_path, mesh, precision):
    """Test weight loading correctness."""
    logger.info("\n" + "=" * 60 + "\nTest: Weights Alignment\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    pt_state = pt_model.state_dict()

    # Key weight mappings: (pt_key, jax_path, transpose_axes)
    weight_checks = [
        # Patch Embedding
        ("patch_embed.proj.weight", "patch_embed.proj.kernel", (2, 3, 4, 1, 0)),
        ("patch_embed.proj.bias", "patch_embed.proj.bias", None),
        # Position Embedding
        ("pos_embed.weight", "pos_embed.embedding", None),
        # First block attention
        ("blocks.0.attn.qkv.weight", "blocks.0.attn.qkv_proj.weight", (1, 0)),
        ("blocks.0.attn.qkv.bias", "blocks.0.attn.qkv_proj.bias", None),
        ("blocks.0.attn.proj.weight", "blocks.0.attn.o_proj.weight", (1, 0)),
        ("blocks.0.attn.proj.bias", "blocks.0.attn.o_proj.bias", None),
        # First block MLP
        ("blocks.0.mlp.linear_fc1.weight", "blocks.0.mlp.fc1.weight", (1, 0)),
        ("blocks.0.mlp.linear_fc1.bias", "blocks.0.mlp.fc1.bias", None),
        ("blocks.0.mlp.linear_fc2.weight", "blocks.0.mlp.fc2.weight", (1, 0)),
        ("blocks.0.mlp.linear_fc2.bias", "blocks.0.mlp.fc2.bias", None),
        # First block LayerNorm
        ("blocks.0.norm1.weight", "blocks.0.norm1.scale", None),
        ("blocks.0.norm1.bias", "blocks.0.norm1.bias", None),
        # Last block (block 26)
        ("blocks.26.attn.qkv.weight", "blocks.26.attn.qkv_proj.weight", (1, 0)),
        ("blocks.26.attn.qkv.bias", "blocks.26.attn.qkv_proj.bias", None),
        ("blocks.26.attn.proj.weight", "blocks.26.attn.o_proj.weight", (1, 0)),
        ("blocks.26.mlp.linear_fc1.weight", "blocks.26.mlp.fc1.weight", (1, 0)),
        ("blocks.26.mlp.linear_fc2.weight", "blocks.26.mlp.fc2.weight", (1, 0)),
        ("blocks.26.norm1.weight", "blocks.26.norm1.scale", None),
        # Final merger
        ("merger.ln_q.weight", "merger.ln_q.scale", None),
        ("merger.mlp.0.weight", "merger.mlp_fc1.weight", (1, 0)),
        ("merger.mlp.2.weight", "merger.mlp_fc2.weight", (1, 0)),
        # Deepstack mergers
        ("merger_list.0.ln_q.weight", "deepstack_mergers.0.ln_q.scale", None),
        ("merger_list.0.ln_q.bias", "deepstack_mergers.0.ln_q.bias", None),
        ("merger_list.0.mlp.0.weight", "deepstack_mergers.0.mlp_fc1.weight", (1, 0)),
        ("merger_list.0.mlp.0.bias", "deepstack_mergers.0.mlp_fc1.bias", None),
        ("merger_list.0.mlp.2.weight", "deepstack_mergers.0.mlp_fc2.weight", (1, 0)),
        ("merger_list.0.mlp.2.bias", "deepstack_mergers.0.mlp_fc2.bias", None),
        ("merger_list.1.ln_q.weight", "deepstack_mergers.1.ln_q.scale", None),
        ("merger_list.1.mlp.0.weight", "deepstack_mergers.1.mlp_fc1.weight", (1, 0)),
        ("merger_list.1.mlp.2.weight", "deepstack_mergers.1.mlp_fc2.weight", (1, 0)),
        ("merger_list.2.ln_q.weight", "deepstack_mergers.2.ln_q.scale", None),
        ("merger_list.2.mlp.0.weight", "deepstack_mergers.2.mlp_fc1.weight", (1, 0)),
        ("merger_list.2.mlp.2.weight", "deepstack_mergers.2.mlp_fc2.weight", (1, 0)),
    ]

    all_passed, total_mae, checked = True, 0.0, 0

    for pt_key, jax_path, transpose_axes in weight_checks:
        if pt_key not in pt_state:
            continue

        pt_w = to_numpy(pt_state[pt_key])

        try:
            parts = jax_path.split(".")
            param = jax_model
            for p in parts:
                param = param[int(p)] if p.isdigit() else getattr(param, p)
            jax_w = np.array(
                param[...] if isinstance(param, nnx.Variable) else param,
                dtype=np.float32,
            )
        except Exception as e:
            logger.warning("Skip %s: %s", pt_key, e)
            continue

        if transpose_axes:
            pt_w = np.transpose(pt_w, transpose_axes)

        if pt_w.shape != jax_w.shape:
            logger.error("❌ %s: Shape mismatch %s vs %s", pt_key, pt_w.shape, jax_w.shape)
            all_passed = False
            continue

        mae = np.abs(pt_w - jax_w).mean()
        passed = mae < 1e-5
        status = "✅" if passed else "❌"
        logger.info("%s %s: MAE=%.2e", status, pt_key, mae)

        all_passed &= passed
        total_mae += mae
        checked += 1

    return all_passed, total_mae / max(checked, 1)


def test_intermediate_blocks(model_path, mesh, precision):
    """Test intermediate transformer block outputs."""
    logger.info("\n" + "=" * 60 + "\nTest: Intermediate Blocks\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    pixel_values, grid_thw = create_dummy_input(config, resolution="small")
    check_blocks = [0, 5, 13, 20, 26]

    with torch.no_grad():
        pt_ctx = PTForwardContext(pt_model, pixel_values, grid_thw)
        pt_hidden = pt_ctx.get_initial_hidden()
        _, pt_outputs = pt_ctx.run_blocks(pt_hidden, check_blocks)

    jax_ctx = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh)
    jax_hidden = jax_ctx.get_initial_hidden()
    _, jax_outputs = jax_ctx.run_blocks(jax_hidden, check_blocks)

    all_passed, results = True, []
    threshold = 5e-3 if precision == "bfloat16" else 2e-3

    for idx in check_blocks:
        if idx in pt_outputs and idx in jax_outputs:
            passed, mae = compare(pt_outputs[idx], jax_outputs[idx], f"Block[{idx}]", threshold)
            results.append((idx, passed, mae))
            all_passed &= passed

    logger.info("\n--- Error Accumulation ---")
    for idx, passed, mae in results:
        logger.info("Block[%d]: MAE=%.4e", idx, mae)

    return all_passed, sum(r[2] for r in results) / len(results) if results else float("inf")


def test_merger(model_path, mesh, precision):
    """Test merger layer alignment."""
    logger.info("\n" + "=" * 60 + "\nTest: Merger Layer\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    pixel_values, grid_thw = create_dummy_input(config, resolution="small")

    with torch.no_grad():
        pt_ctx = PTForwardContext(pt_model, pixel_values, grid_thw)
        pt_hidden = pt_ctx.get_initial_hidden()
        pt_hidden, _ = pt_ctx.run_blocks(pt_hidden)
        pt_merger_out = pt_model.merger(pt_hidden)

    jax_output = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh).full_forward()

    log_stats(pt_hidden, "PT Merger Input")
    log_stats(jax_output["last_hidden_state"], "JAX Merger Input")

    threshold = 5e-3 if precision == "bfloat16" else 2e-3
    pre_passed, _ = compare(
        pt_hidden, jax_output["last_hidden_state"], "Pre-Merger Hidden", threshold
    )

    threshold = 2e-3 if precision == "bfloat16" else 1e-3
    merger_passed, mae = compare(
        pt_merger_out, jax_output["pooler_output"], "Merger Output", threshold
    )

    return pre_passed and merger_passed, mae


def test_deepstack_layers(model_path, mesh, precision):
    """Test deepstack layer hidden states."""
    logger.info("\n" + "=" * 60 + "\nTest: Deepstack Layers\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    deepstack_indexes = list(config.deepstack_visual_indexes)
    logger.info("Deepstack indexes: %s", deepstack_indexes)

    pixel_values, grid_thw = create_dummy_input(config, resolution="small")

    # Get deepstack merger list from PyTorch
    pt_merger_list = getattr(pt_model, "deepstack_merger_list", None) or getattr(
        pt_model, "merger_list", None
    )

    with torch.no_grad():
        pt_ctx = PTForwardContext(pt_model, pixel_values, grid_thw)
        pt_hidden = pt_ctx.get_initial_hidden()
        _, pt_outputs = pt_ctx.run_blocks(pt_hidden, deepstack_indexes)

        pt_merged = {}
        if pt_merger_list:
            for idx in deepstack_indexes:
                if idx in pt_outputs:
                    merger_idx = deepstack_indexes.index(idx)
                    if merger_idx < len(pt_merger_list):
                        pt_merged[idx] = pt_merger_list[merger_idx](pt_outputs[idx].clone())

    jax_ctx = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh)
    jax_hidden = jax_ctx.get_initial_hidden()
    _, jax_outputs = jax_ctx.run_blocks(jax_hidden, deepstack_indexes)

    with jax.set_mesh(mesh):
        jax_merged = {
            idx: jax_model.deepstack_mergers[deepstack_indexes.index(idx)](jax_outputs[idx])
            for idx in deepstack_indexes
            if idx in jax_outputs
        }

    all_passed, results = True, []
    threshold_hidden = 5e-3 if precision == "bfloat16" else 2e-3
    threshold_merged = 2e-3 if precision == "bfloat16" else 1e-3

    logger.info("\n--- Hidden States ---")
    for idx in deepstack_indexes:
        if idx in pt_outputs and idx in jax_outputs:
            passed, mae = compare(
                pt_outputs[idx],
                jax_outputs[idx],
                f"Layer[{idx}] Hidden",
                threshold_hidden,
            )
            results.append((f"layer_{idx}_hidden", passed, mae))
            all_passed &= passed

    logger.info("\n--- Merged Features ---")
    for idx in deepstack_indexes:
        if idx in pt_merged and idx in jax_merged:
            passed, mae = compare(
                pt_merged[idx],
                jax_merged[idx],
                f"Layer[{idx}] Merged",
                threshold_merged,
            )
            results.append((f"layer_{idx}_merged", passed, mae))
            all_passed &= passed

    avg_mae = sum(r[2] for r in results if r[2] != float("inf")) / max(len(results), 1)
    return all_passed, avg_mae


def test_video_input(model_path, mesh, precision):
    """Test with video input (multiple temporal frames)."""
    logger.info("\n" + "=" * 60 + "\nTest: Video Input\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    temporal_patch_size = getattr(config, "temporal_patch_size", 2)
    pixel_values, grid_thw = create_video_input(config, temporal_patch_size * 2, 8, 8)

    logger.info("Video: %d frames -> %d temporal patches", temporal_patch_size * 2, 2)

    with torch.no_grad():
        pt_ctx = PTForwardContext(pt_model, pixel_values, grid_thw)
        pt_patch = pt_model.patch_embed(pt_ctx.input_flat)
        pt_pos = pt_model.fast_pos_embed_interpolate(pt_ctx.grid)
        pt_pooler = pt_ctx.full_forward()

    jax_ctx = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh)
    with jax.set_mesh(mesh):
        jax_patch_raw = jax_model.patch_embed(jax_ctx.pixel_values)
        jax_patch = jax_model._apply_spatial_merge_permutation(jax_patch_raw, jax_ctx.grid)
        jax_pos = jax_model.interpolate_pos_embed(jax_ctx.grid)

    jax_output = jax_ctx.full_forward()

    # Compare intermediates
    patch_passed, _ = compare(pt_patch, jax_patch, "Patch Embed (Video)", 1e-2)
    pos_passed, _ = compare(pt_pos, jax_pos, "Pos Embed (Video)", 1e-2)

    threshold = 2e-3 if precision == "bfloat16" else 1e-3
    passed, mae = compare(pt_pooler, jax_output["pooler_output"], "Video Pooler", threshold)

    return passed and patch_passed, mae


def test_non_square_image(model_path, mesh, precision):
    """Test with non-square images."""
    logger.info("\n" + "=" * 60 + "\nTest: Non-Square Images\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    configs = [(2, 14, 8), (2, 8, 14), (2, 20, 10), (2, 10, 20)]

    all_passed, total_mae, count = True, 0.0, 0
    threshold = 2e-3 if precision == "bfloat16" else 1e-3

    for temporal, h, w in configs:
        pixel_values, grid_thw = create_video_input(config, temporal, h, w)
        logger.info("Config: %dx%d patches", h, w)

        with torch.no_grad():
            pt_pooler = PTForwardContext(pt_model, pixel_values, grid_thw).full_forward()

        jax_output = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh).full_forward()

        passed, mae = compare(
            pt_pooler, jax_output["pooler_output"], f"Non-Square {h}x{w}", threshold
        )
        all_passed &= passed
        total_mae += mae
        count += 1

    return all_passed, total_mae / count


def test_long_video(model_path, mesh, precision):
    """Test with longer video sequences."""
    logger.info("\n" + "=" * 60 + "\nTest: Long Video\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    configs = [(4, 8, 8), (8, 8, 8), (6, 10, 10)]

    all_passed, total_mae, count = True, 0.0, 0
    threshold = 2e-3 if precision == "bfloat16" else 1e-3

    for temporal_frames, h, w in configs:
        pixel_values, grid_thw = create_video_input(config, temporal_frames, h, w)
        temporal_output = temporal_frames // getattr(config, "temporal_patch_size", 2)
        logger.info(
            "Video: %d frames -> %d temporal patches, %dx%d",
            temporal_frames,
            temporal_output,
            h,
            w,
        )

        with torch.no_grad():
            pt_pooler = PTForwardContext(pt_model, pixel_values, grid_thw).full_forward()

        jax_output = JAXForwardContext(jax_model, pixel_values, grid_thw, mesh).full_forward()

        passed, mae = compare(
            pt_pooler,
            jax_output["pooler_output"],
            f"Video {temporal_frames}f",
            threshold,
        )
        all_passed &= passed
        total_mae += mae
        count += 1

    return all_passed, total_mae / count


def test_batch_images(model_path, mesh, precision):
    """Test with multiple images (batch)."""
    logger.info("\n" + "=" * 60 + "\nTest: Batch Images\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    temporal_patch_size = getattr(config, "temporal_patch_size", 2)
    patch_size = config.patch_size
    h_patches, w_patches = 8, 8
    height, width = h_patches * patch_size, w_patches * patch_size

    np.random.seed(42)
    img1 = (
        np.random.randn(1, temporal_patch_size, height, width, config.in_channels).astype(
            np.float32
        )
        * 0.5
        + 0.5
    )
    img2 = (
        np.random.randn(1, temporal_patch_size, height, width, config.in_channels).astype(
            np.float32
        )
        * 0.5
        + 0.3
    )
    pixel_values = np.concatenate([img1, img2], axis=0)

    grid_thw = np.array([[1, h_patches, w_patches], [1, h_patches, w_patches]], dtype=np.int32)

    # Process separately and concatenate
    with torch.no_grad():
        pt_dtype = next(pt_model.parameters()).dtype
        pt_inputs = [
            prepare_pytorch_input(pixel_values[i : i + 1], config, pt_dtype) for i in range(2)
        ]
        pt_input_flat = torch.cat(pt_inputs, dim=0)
        pt_grid = torch.from_numpy(grid_thw)
        pt_output = pt_model(hidden_states=pt_input_flat, grid_thw=pt_grid, return_dict=True)
        pt_pooler = pt_output.pooler_output if hasattr(pt_output, "pooler_output") else pt_output[0]

    # JAX: process separately
    with jax.set_mesh(mesh):
        jax_outputs = [
            jax_model(
                pixel_values=jnp.array(prepare_jax_input(pixel_values[i : i + 1], config)),
                grid_thw=jnp.array(grid_thw[i : i + 1]),
            )["pooler_output"]
            for i in range(2)
        ]
        jax_pooler = jnp.concatenate(jax_outputs, axis=0)

    threshold = 2e-3 if precision == "bfloat16" else 1e-3
    return compare(pt_pooler, jax_pooler, "Batch Images", threshold)


def test_mlp_activation(model_path, mesh, precision):
    """Test MLP activation function (GELU-Tanh)."""
    logger.info("\n" + "=" * 60 + "\nTest: MLP Activation\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    test_input = np.random.randn(64, config.hidden_size).astype(np.float32)

    with torch.no_grad():
        pt_input = torch.from_numpy(test_input).to(next(pt_model.parameters()).dtype)
        pt_output = pt_model.blocks[0].mlp(pt_input)

    with jax.set_mesh(mesh):
        jax_output = jax_model.blocks[0].mlp(jnp.array(test_input))

    log_stats(pt_output, "PyTorch MLP")
    log_stats(jax_output, "JAX MLP")

    return compare(pt_output, jax_output, "MLP Output", 1e-3)


def test_deepstack_merger_weights(model_path, mesh, precision):
    """Test deepstack merger weights are correctly loaded."""
    logger.info("\n" + "=" * 60 + "\nTest: Deepstack Merger Weights\n" + "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    pt_merger_list = getattr(pt_model, "deepstack_merger_list", None) or getattr(
        pt_model, "merger_list", None
    )

    if not pt_merger_list:
        logger.warning("PyTorch model has no deepstack_merger_list")
        return len(jax_model.deepstack_mergers) == len(config.deepstack_visual_indexes), 0.0

    all_passed, results = True, []

    for i in range(min(len(pt_merger_list), len(jax_model.deepstack_mergers))):
        logger.info("\n--- Deepstack Merger[%d] ---", i)
        pt_merger, jax_merger = pt_merger_list[i], jax_model.deepstack_mergers[i]

        if hasattr(pt_merger, "ln_q") and hasattr(jax_merger, "ln_q"):
            pt_w = to_numpy(pt_merger.ln_q.weight)
            jax_w = np.array(jax_merger.ln_q.scale[...], dtype=np.float32)
            mae = np.abs(pt_w - jax_w).mean()
            passed = mae < 1e-5
            logger.info("%s Merger[%d] ln_q: MAE=%.2e", "✅" if passed else "❌", i, mae)
            results.append((f"merger_{i}_ln_q", passed, mae))
            all_passed &= passed

    avg_mae = sum(r[2] for r in results if r[2] != float("inf")) / max(len(results), 1)
    return all_passed, avg_mae


# =============================================================================
# =============================================================================
# TP Validation Tests
# =============================================================================


def test_tp_sharding(model_path, mesh, precision):
    """
    Test that TP sharding is correctly applied to model weights.

    Verifies:
    1. Weights are correctly sharded across devices
    2. Expected layers have TP sharding (VisionAttention, VisionMLP, VisionPatchMerger)
    3. Replicated layers are indeed replicated (LayerNorm, Embedding)
    """
    logger.info("\n%s\nTest: TP Sharding Verification\n%s", "=" * 60, "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    from jax.sharding import PartitionSpec as P

    # Get mesh shape for TP dimension
    tp_size = mesh.shape.get("tensor", 1) if hasattr(mesh.shape, "get") else 1
    logger.info("Mesh: %s, TP size: %s", mesh.shape, tp_size)

    all_passed = True
    results = []

    # Check sharding for key layers
    sharding_checks = [
        # (path, expected_spec, description)
        # VisionAttention - should be sharded
        ("blocks.0.attn.qkv_proj.weight", (None, "tensor"), "Block0 QKV weight"),
        ("blocks.0.attn.o_proj.weight", ("tensor", None), "Block0 O_proj weight"),
        # VisionMLP - should be sharded
        ("blocks.0.mlp.fc1.weight", (None, "tensor"), "Block0 MLP fc1 weight"),
        ("blocks.0.mlp.fc2.weight", ("tensor", None), "Block0 MLP fc2 weight"),
        # VisionPatchMerger - should be sharded (newly added)
        ("merger.mlp_fc1.weight", (None, "tensor"), "Merger fc1 weight"),
        ("merger.mlp_fc2.weight", ("tensor", None), "Merger fc2 weight"),
        # Deepstack merger - should be sharded
        (
            "deepstack_mergers.0.mlp_fc1.weight",
            (None, "tensor"),
            "Deepstack0 fc1 weight",
        ),
        (
            "deepstack_mergers.0.mlp_fc2.weight",
            ("tensor", None),
            "Deepstack0 fc2 weight",
        ),
        # LayerNorm - should be replicated
        ("blocks.0.norm1.scale", (None,), "Block0 norm1 (replicated)"),
        ("merger.ln_q.scale", (None,), "Merger ln_q (replicated)"),
    ]

    for path, expected_axes, desc in sharding_checks:
        try:
            # Navigate to parameter
            parts = path.split(".")
            param = jax_model
            for p in parts:
                param = param[int(p)] if p.isdigit() else getattr(param, p)

            if isinstance(param, nnx.Variable):
                value = param.get_value()
            else:
                value = param

            # Check sharding spec
            if hasattr(value, "sharding"):
                actual_spec = value.sharding.spec if hasattr(value.sharding, "spec") else None
                logger.info("  %s: shape=%s, sharding=%s", desc, value.shape, actual_spec)

                # For TP > 1, verify sharding matches expected
                if tp_size > 1 and actual_spec is not None:
                    # Convert expected to PartitionSpec for comparison
                    expected_p = P(*expected_axes)
                    if str(actual_spec) == str(expected_p):
                        logger.info("    ✅ Sharding matches expected: %s", expected_p)
                        results.append((desc, True, 0.0))
                    else:
                        logger.warning(
                            "    ⚠️ Sharding mismatch: expected %s, got %s",
                            expected_p,
                            actual_spec,
                        )
                        results.append((desc, False, 1.0))
                        all_passed = False
                else:
                    results.append((desc, True, 0.0))
            else:
                logger.info("  %s: shape=%s (no sharding info)", desc, value.shape)
                results.append((desc, True, 0.0))

        except Exception as e:
            logger.warning("  Failed to check %s: %s", path, e)
            results.append((desc, False, float("inf")))

    # Summary
    passed_count = sum(1 for _, p, _ in results if p)
    logger.info("\nSharding check: %d/%d passed", passed_count, len(results))

    return all_passed, 0.0 if all_passed else 1.0


def test_tp_output_consistency(model_path, mesh, precision):
    """
    Test that TP output is numerically consistent.

    When TP > 1, the sharded computation should produce the same result
    as unsharded computation (within numerical precision).
    """
    logger.info("\n%s\nTest: TP Output Consistency\n%s", "=" * 60, "=" * 60)

    pt_model, jax_model, config = load_models(model_path, mesh, precision)
    if pt_model is None:
        return False, float("inf")

    # Get TP size
    tp_size = mesh.shape.get("tensor", 1) if hasattr(mesh.shape, "get") else 1
    logger.info("Testing with TP size: %d", tp_size)

    # Create input
    pixel_values, grid_thw = create_dummy_input(config, resolution="small")

    # Run JAX forward
    with jax.set_mesh(mesh):
        jax_output = jax_model(
            pixel_values=jnp.array(prepare_jax_input(pixel_values, config)),
            grid_thw=jnp.array(grid_thw),
        )

    jax_pooler = jax_output["pooler_output"]

    # Run PyTorch forward for comparison
    with torch.no_grad():
        pt_input_flat = prepare_pytorch_input(
            pixel_values, config, next(pt_model.parameters()).dtype
        )
        pt_grid = torch.from_numpy(grid_thw)
        pt_output = pt_model(hidden_states=pt_input_flat, grid_thw=pt_grid, return_dict=True)

    pt_pooler = pt_output.pooler_output if hasattr(pt_output, "pooler_output") else pt_output[0]

    # Compare
    logger.info("JAX output shape: %s, mean: %.4f", jax_pooler.shape, float(jax_pooler.mean()))
    logger.info(
        "PyTorch output shape: %s, mean: %.4f",
        pt_pooler.shape,
        pt_pooler.float().mean().item(),
    )

    passed, mae = compare(
        pt_pooler,
        jax_pooler,
        f"TP={tp_size} Output Consistency",
        threshold=2e-3 if precision == "bfloat16" else 1e-3,
    )

    # Also check deepstack features
    if hasattr(pt_output, "deepstack_features") and pt_output.deepstack_features:
        for i, (pt_feat, jax_feat) in enumerate(
            zip(pt_output.deepstack_features, jax_output["deepstack_features"])
        ):
            feat_passed, feat_mae = compare(
                pt_feat,
                jax_feat,
                f"TP={tp_size} Deepstack[{i}]",
                threshold=2e-3 if precision == "bfloat16" else 1e-3,
            )
            passed &= feat_passed

    return passed, mae


# Test Registry & Main
# =============================================================================

TESTS = {
    "weights": test_weights_alignment,
    "layers": test_layer_outputs,
    "forward_basic": test_forward_basic,
    "forward_medium": test_forward_medium,
    "forward_large": test_large_resolution,
    "forward_video": test_video_input,
    "forward_non_square": test_non_square_image,
    "forward_long_video": test_long_video,
    "forward_batch": test_batch_images,
    "deepstack_features": test_deepstack_features,
    "deepstack_layers": test_deepstack_layers,
    "deepstack_merger_weights": test_deepstack_merger_weights,
    "hidden_states": test_hidden_states,
    "intermediate_blocks": test_intermediate_blocks,
    "merger": test_merger,
    "mlp_activation": test_mlp_activation,
    # TP validation tests
    "tp_sharding": test_tp_sharding,
    "tp_output": test_tp_output_consistency,
}


def main():
    import traceback

    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    parser = argparse.ArgumentParser(description="Qwen3OmniMoe Vision Encoder alignment tests")
    parser.add_argument(
        "--model_path",
        default=os.environ.get("QWEN3_OMNI_MODEL_PATH", "/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"),
    )
    parser.add_argument("--test_type", default="all", choices=["all"] + list(TESTS.keys()))
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--precision", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    precision_map = {"float32": "highest", "bfloat16": "default"}
    jax.config.update("jax_default_matmul_precision", precision_map[args.precision])

    logger.info(
        "Model: %s, Test: %s, Precision: %s",
        args.model_path,
        args.test_type,
        args.precision,
    )

    if not os.path.exists(args.model_path):
        logger.error("Model path does not exist: %s", args.model_path)
        return False

    devices = jax.devices()
    tp = args.tp_size or min(len(devices), 1)
    mesh = create_device_mesh(
        ici_parallelism=[1, tp],
        dcn_parallelism=[1, 1],
        devices=devices[:tp],
        use_explicit_sharding=True,
    )

    tests = list(TESTS.keys()) if args.test_type == "all" else [args.test_type]
    results = []
    t0 = time.time()

    for name in tests:
        try:
            passed, mae = TESTS[name](args.model_path, mesh, args.precision)
            results.append((name, passed, mae))
        except Exception as e:
            logger.error("%s: %s", name, e)
            traceback.print_exc()
            results.append((name, False, float("inf")))

    logger.info("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    for name, passed, mae in results:
        logger.info("%s %s: MAE=%.2e", "✅" if passed else "❌", name, mae)

    all_pass = all(r[1] for r in results)
    logger.info(
        "\n%s (%.1fs)",
        "✅ All PASSED" if all_pass else "❌ Some FAILED",
        time.time() - t0,
    )
    return all_pass


if __name__ == "__main__":
    main()
