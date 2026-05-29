# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Test alignment between JAX Qwen3OmniMoe Vision Encoder and PyTorch HuggingFace model.

Uses per-block comparison with shared inputs to prevent error accumulation across
27 transformer layers. Each block/merger receives IDENTICAL input from PyTorch,
isolating per-component error to within atol=1e-5.
"""

import glob
import os
import unittest

import jax
import numpy as np
import torch
from flax import nnx
from jax import numpy as jnp
from transformers import Qwen3OmniMoeThinkerConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)

from sgl_jax.srt.configs.model_config import _get_and_verify_dtype
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.qwen3_omni_moe_encoder import (
    Qwen3OmniMoeThinkerEmbedding,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

# Ensure JAX uses float32 precision for matmul
jax.config.update("jax_default_matmul_precision", "highest")

# Default model path
MODEL_PATH = os.environ.get("QWEN3_OMNI_MODEL_PATH", "/models/Qwen/Qwen3-Omni-30B-A3B-Instruct")


def to_numpy(x) -> np.ndarray:
    """Convert PyTorch/JAX tensor to numpy float32."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.array(x, dtype=np.float32)


def prepare_input(pixel_values: np.ndarray, config, dtype=None):
    """
    Convert pixel_values to patch format with spatial-merge permutation.

    Args:
        pixel_values: (B, T, H, W, C) numpy array
        config: Vision encoder config
        dtype: torch.dtype or None. Returns torch.Tensor if provided, numpy otherwise.

    Returns:
        Patches in (N_patches, C*t*p*p) format with spatial-merge order.
    """
    pt_input = torch.from_numpy(pixel_values).to(dtype)
    B, T, H, W, C = pt_input.shape
    p = config.patch_size
    t = config.temporal_patch_size
    m = config.spatial_merge_size

    # (B, T, H, W, C) -> (B, C, T, H, W) -> extract patches -> spatial-merge permutation
    pt_input = pt_input.permute(0, 4, 1, 2, 3)
    pt_input = pt_input.reshape(B, C, T // t, t, H // p, p, W // p, p)
    pt_input = pt_input.permute(0, 2, 4, 6, 1, 3, 5, 7)

    T_out, H_p, W_p = T // t, H // p, W // p
    pt_input = pt_input.reshape(B, T_out, H_p // m, m, W_p // m, m, C, t, p, p)
    pt_input = pt_input.permute(0, 1, 2, 4, 3, 5, 6, 7, 8, 9)
    result = pt_input.reshape(-1, C * t * p * p)
    return result if dtype is not None else result.numpy()


def create_input(config, temporal_frames: int, h_patches: int, w_patches: int, seed: int = 42):
    """Create random input with specified dimensions."""
    np.random.seed(seed)
    height = h_patches * config.patch_size
    width = w_patches * config.patch_size
    pixel_values = (
        np.random.randn(1, temporal_frames, height, width, config.in_channels).astype(np.float32)
        * 0.5
        + 0.5
    )

    t_out = temporal_frames // config.temporal_patch_size
    grid_thw = np.array([[t_out, h_patches, w_patches]], dtype=np.int32)
    return pixel_values, grid_thw


class TestQwen3OmniMoeVisionEncoderPrecision(unittest.TestCase):
    """Test Vision Encoder alignment between JAX and PyTorch.

    Uses per-block comparison with shared inputs to prevent error accumulation.
    Each component (block, merger, deepstack merger) is tested in isolation by
    feeding IDENTICAL input from PyTorch, keeping per-component error within atol=1e-5.
    """

    @classmethod
    def setUpClass(cls):
        """Load models once for all tests."""
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"Model path not found: {MODEL_PATH}")

        # Create mesh
        cpu_devices = jax.devices("cpu")
        cls.mesh = create_device_mesh(
            ici_parallelism=[-1, len(cpu_devices)],
            dcn_parallelism=[1, 1],
            devices=cpu_devices,
        )

        # Load PyTorch model
        cls.pt_model, cls.pt_config = cls._load_pytorch_model()

        # Load JAX model
        cls.jax_model = cls._load_jax_model(cls.pt_config)

    @classmethod
    def _load_pytorch_model(cls):
        """Load PyTorch vision encoder."""
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeVisionEncoder as PTVisionEncoder,
        )

        # Load config
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        pt_config = None
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
        if pt_config is None:
            pt_config = Qwen3OmniMoeVisionEncoderConfig()

        pt_model = PTVisionEncoder(pt_config).eval().to(torch.float32)

        # Load weights from safetensors
        safetensor_files = glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))
        if safetensor_files:
            all_weights = {}
            for sf_file in safetensor_files:
                with safe_open(sf_file, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        if key.startswith("thinker.visual."):
                            all_weights[key.replace("thinker.visual.", "")] = (sf_file, key)

            pt_state = pt_model.state_dict()
            for model_key in pt_state:
                if model_key in all_weights:
                    sf_file, original_key = all_weights[model_key]
                    with safe_open(sf_file, framework="pt") as f:
                        pt_state[model_key] = f.get_tensor(original_key)
            pt_model.load_state_dict(pt_state)

        return pt_model, pt_config

    @classmethod
    def _load_jax_model(cls, pt_config):
        """Load JAX vision encoder."""
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
        jax_dtype = _get_and_verify_dtype(jax_config, "float32")
        jax_config.model_path = MODEL_PATH
        jax_config.revision = None
        jax_config.dtype = jax_dtype
        jax_config.model_class = Qwen3OmniMoeThinkerEmbedding

        with jax.set_mesh(cls.mesh):
            jax_model = Qwen3OmniMoeThinkerEmbedding(
                config=jax_config, mesh=cls.mesh, rngs=nnx.Rngs(0), dtype=jax_dtype
            )
        jax_model.load_weights(jax_config)
        return jax_model.visual

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _assert_close(self, pt_tensor, jax_tensor, name: str, rtol=1e-3, atol=1e-5):
        """Assert two tensors are close."""
        pt_np = to_numpy(pt_tensor)
        jax_np = to_numpy(jax_tensor)
        self.assertEqual(pt_np.shape, jax_np.shape, f"{name}: shape mismatch")
        np.testing.assert_allclose(
            pt_np, jax_np, rtol=rtol, atol=atol, err_msg=f"{name}: values don't match"
        )

    def _prepare_pt_forward_context(self, pixel_values, grid_thw):
        """Prepare PyTorch forward context from raw pixel values.

        Returns:
            (pt_hidden, pt_cu_seqlens, pt_position_embeddings) where:
            - pt_hidden: (seq_len, hidden_size) initial hidden state after patch_embed + pos_embed
            - pt_cu_seqlens: cumulative sequence lengths for attention
            - pt_position_embeddings: (cos, sin) tuple for rotary embeddings
        """
        with torch.no_grad():
            pt_input = prepare_input(pixel_values, self.pt_config, torch.float32)
            return self._prepare_pt_context_from_patches(pt_input, grid_thw)

    def _prepare_pt_context_from_patches(self, pt_patches, grid_thw):
        """Prepare PyTorch forward context from pre-prepared patches.

        Args:
            pt_patches: (N_patches, C*t*p*p) as torch.Tensor
            grid_thw: (num_images, 3) as numpy array

        Returns:
            (pt_hidden, pt_cu_seqlens, pt_position_embeddings)
        """
        with torch.no_grad():
            pt_grid = torch.from_numpy(grid_thw)
            pt_pos = self.pt_model.fast_pos_embed_interpolate(pt_grid)
            pt_hidden = self.pt_model.patch_embed(pt_patches) + pt_pos

            seq_len, _ = pt_hidden.size()
            pt_hidden = pt_hidden.reshape(seq_len, -1)
            pt_rotary = self.pt_model.rot_pos_emb(pt_grid)
            pt_rotary = pt_rotary.reshape(seq_len, -1)
            pt_emb = torch.cat((pt_rotary, pt_rotary), dim=-1)
            pt_position_embeddings = (pt_emb.cos(), pt_emb.sin())

            pt_cu_seqlens = torch.repeat_interleave(
                pt_grid[:, 1] * pt_grid[:, 2], pt_grid[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            pt_cu_seqlens = torch.nn.functional.pad(pt_cu_seqlens, (1, 0), value=0)

        return pt_hidden, pt_cu_seqlens, pt_position_embeddings

    def _prepare_jax_context(self, grid_thw):
        """Prepare JAX forward context: position IDs and attention mask.

        Returns:
            (position_ids, attention_mask) where:
            - position_ids: (seq_len, 2) spatial position IDs for RoPE
            - attention_mask: attention mask or None for single sequence
        """
        with jax.set_mesh(self.mesh):
            jax_grid = jnp.array(grid_thw)
            position_ids = self.jax_model.compute_2d_position_ids(jax_grid)
            attention_mask = self.jax_model._create_attention_mask(jax_grid)
        return position_ids, attention_mask

    def _run_perblock_comparison(
        self,
        pt_hidden,
        pt_cu_seqlens,
        pt_position_embeddings,
        jax_position_ids,
        jax_attention_mask,
        rtol=1e-3,
        atol=1e-5,
        check_blocks=True,
        check_merger=True,
        check_deepstack=True,
    ):
        """Run per-block comparison with shared inputs from PyTorch.

        Core principle: Each block receives IDENTICAL input from PyTorch,
        preventing error accumulation across layers. This keeps per-component
        error within atol=1e-5 even though end-to-end would accumulate to ~1e-3.

        Args:
            pt_hidden: Initial hidden state from PyTorch (seq_len, hidden_size)
            pt_cu_seqlens: Cumulative sequence lengths for PyTorch attention
            pt_position_embeddings: (cos, sin) tuple for PyTorch rotary embeddings
            jax_position_ids: (seq_len, 2) position IDs for JAX rotary embeddings
            jax_attention_mask: Attention mask for JAX (or None)
            rtol, atol: Tolerance for comparison
            check_blocks: If True, compare each block's output
            check_merger: If True, compare final merger output
            check_deepstack: If True, compare deepstack merger outputs
        """
        deepstack_indices = list(self.pt_config.deepstack_visual_indexes)
        pt_hidden_at_layers = {}

        num_blocks = len(self.pt_model.blocks)

        for block_idx in range(num_blocks):
            if check_blocks:
                # Convert PyTorch hidden to JAX (SAME input for both)
                jax_hidden = jnp.array(pt_hidden.detach().float().numpy())

                # Run ONE block in PyTorch
                with torch.no_grad():
                    pt_out = self.pt_model.blocks[block_idx](
                        pt_hidden, pt_cu_seqlens, position_embeddings=pt_position_embeddings
                    )

                # Run ONE block in JAX
                with jax.set_mesh(self.mesh):
                    jax_out = self.jax_model.blocks[block_idx](
                        jax_hidden, jax_position_ids, jax_attention_mask
                    )

                # Compare: only single-block error, no accumulation
                self._assert_close(pt_out, jax_out, f"block_{block_idx}", rtol=rtol, atol=atol)
            else:
                # Just run PT block without JAX comparison
                with torch.no_grad():
                    pt_out = self.pt_model.blocks[block_idx](
                        pt_hidden, pt_cu_seqlens, position_embeddings=pt_position_embeddings
                    )

            # Record hidden state at deepstack layers
            if block_idx in deepstack_indices:
                pt_hidden_at_layers[block_idx] = pt_out.clone()

            # Use PyTorch output as next input (prevents accumulation)
            pt_hidden = pt_out

        # --- Merger comparison with shared input ---
        if check_merger:
            jax_hidden_for_merger = jnp.array(pt_hidden.detach().float().numpy())

            with torch.no_grad():
                pt_merged = self.pt_model.merger(pt_hidden)

            with jax.set_mesh(self.mesh):
                jax_merged = self.jax_model.merger(jax_hidden_for_merger)

            self._assert_close(pt_merged, jax_merged, "merger_output", rtol=rtol, atol=atol)

        # --- Deepstack merger comparison with shared input ---
        if check_deepstack and deepstack_indices:
            for i, layer_idx in enumerate(deepstack_indices):
                if layer_idx in pt_hidden_at_layers:
                    pt_h = pt_hidden_at_layers[layer_idx]
                    jax_h = jnp.array(pt_h.detach().float().numpy())

                    with torch.no_grad():
                        pt_feat = self.pt_model.deepstack_merger_list[i](pt_h)

                    with jax.set_mesh(self.mesh):
                        jax_feat = self.jax_model.deepstack_mergers[i](jax_h)

                    self._assert_close(
                        pt_feat, jax_feat, f"deepstack_{layer_idx}", rtol=rtol, atol=atol
                    )

    # =========================================================================
    # Weight Tests
    # =========================================================================

    def test_weights_patch_embed(self):
        """Test patch embedding weights are loaded correctly."""
        pt_state = self.pt_model.state_dict()

        # Patch embed conv kernel: PyTorch (out, in, t, h, w) -> JAX (t, h, w, in, out)
        pt_w = to_numpy(pt_state["patch_embed.proj.weight"])
        pt_w = np.transpose(pt_w, (2, 3, 4, 1, 0))
        jax_w = to_numpy(self.jax_model.patch_embed.proj.kernel)
        np.testing.assert_allclose(pt_w, jax_w, rtol=1e-5, atol=1e-5)

        # Patch embed bias
        pt_b = to_numpy(pt_state["patch_embed.proj.bias"])
        jax_b = to_numpy(self.jax_model.patch_embed.proj.bias)
        np.testing.assert_allclose(pt_b, jax_b, rtol=1e-5, atol=1e-5)

    def test_weights_pos_embed(self):
        """Test position embedding weights are loaded correctly."""
        pt_state = self.pt_model.state_dict()
        pt_w = to_numpy(pt_state["pos_embed.weight"])
        jax_w = to_numpy(self.jax_model.pos_embed.embedding)
        np.testing.assert_allclose(pt_w, jax_w, rtol=1e-5, atol=1e-5)

    def test_weights_block0_attention(self):
        """Test first block attention weights."""
        pt_state = self.pt_model.state_dict()

        # QKV weight: (out, in) -> (in, out)
        pt_qkv = to_numpy(pt_state["blocks.0.attn.qkv.weight"]).T
        jax_qkv = to_numpy(self.jax_model.blocks[0].attn.qkv_proj.weight)
        np.testing.assert_allclose(pt_qkv, jax_qkv, rtol=1e-5, atol=1e-5)

        # QKV bias
        pt_qkv_b = to_numpy(pt_state["blocks.0.attn.qkv.bias"])
        jax_qkv_b = to_numpy(self.jax_model.blocks[0].attn.qkv_proj.bias)
        np.testing.assert_allclose(pt_qkv_b, jax_qkv_b, rtol=1e-5, atol=1e-5)

        # Output projection weight
        pt_o = to_numpy(pt_state["blocks.0.attn.proj.weight"]).T
        jax_o = to_numpy(self.jax_model.blocks[0].attn.o_proj.weight)
        np.testing.assert_allclose(pt_o, jax_o, rtol=1e-5, atol=1e-5)

    def test_weights_block0_mlp(self):
        """Test first block MLP weights."""
        pt_state = self.pt_model.state_dict()

        pt_fc1 = to_numpy(pt_state["blocks.0.mlp.linear_fc1.weight"]).T
        jax_fc1 = to_numpy(self.jax_model.blocks[0].mlp.fc1.weight)
        np.testing.assert_allclose(pt_fc1, jax_fc1, rtol=1e-5, atol=1e-5)

        pt_fc2 = to_numpy(pt_state["blocks.0.mlp.linear_fc2.weight"]).T
        jax_fc2 = to_numpy(self.jax_model.blocks[0].mlp.fc2.weight)
        np.testing.assert_allclose(pt_fc2, jax_fc2, rtol=1e-5, atol=1e-5)

    def test_weights_merger(self):
        """Test merger weights."""
        pt_state = self.pt_model.state_dict()

        # LayerNorm
        pt_ln = to_numpy(pt_state["merger.ln_q.weight"])
        jax_ln = to_numpy(self.jax_model.merger.ln_q.scale)
        np.testing.assert_allclose(pt_ln, jax_ln, rtol=1e-5, atol=1e-5)

        # MLP fc1
        pt_fc1 = to_numpy(pt_state["merger.mlp.0.weight"]).T
        jax_fc1 = to_numpy(self.jax_model.merger.mlp_fc1.weight)
        np.testing.assert_allclose(pt_fc1, jax_fc1, rtol=1e-5, atol=1e-5)

        # MLP fc2
        pt_fc2 = to_numpy(pt_state["merger.mlp.2.weight"]).T
        jax_fc2 = to_numpy(self.jax_model.merger.mlp_fc2.weight)
        np.testing.assert_allclose(pt_fc2, jax_fc2, rtol=1e-5, atol=1e-5)

    # =========================================================================
    # Layer Output Tests
    # =========================================================================

    def test_patch_embed_output(self):
        """Test patch embedding output alignment."""
        pixel_values, grid_thw = create_input(
            self.pt_config, self.pt_config.temporal_patch_size, 8, 8
        )

        with torch.no_grad():
            pt_input = prepare_input(pixel_values, self.pt_config, torch.float32)
            pt_out = self.pt_model.patch_embed(pt_input)

        with jax.set_mesh(self.mesh):
            jax_input = jnp.array(prepare_input(pixel_values, self.jax_model.config, dtype=None))
            jax_out = self.jax_model.patch_embed(jax_input)

        self._assert_close(pt_out, jax_out, "patch_embed", rtol=1e-4, atol=1e-4)

    def test_pos_embed_output(self):
        """Test position embedding interpolation alignment."""
        pixel_values, grid_thw = create_input(
            self.pt_config, self.pt_config.temporal_patch_size, 8, 8
        )

        with torch.no_grad():
            pt_grid = torch.from_numpy(grid_thw)
            pt_pos = self.pt_model.fast_pos_embed_interpolate(pt_grid)

        with jax.set_mesh(self.mesh):
            jax_grid = jnp.array(grid_thw)
            jax_pos = self.jax_model.interpolate_pos_embed(jax_grid)

        self._assert_close(pt_pos, jax_pos, "pos_embed", rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Per-Block Alignment Tests (shared input prevents error accumulation)
    # =========================================================================

    def test_per_block_alignment(self):
        """Test each of the 27 transformer blocks independently with shared input.

        Each block receives IDENTICAL input from PyTorch. Only single-block error
        is measured (~1e-6 to 5e-6), preventing accumulation across 27 layers.
        """
        pixel_values, grid_thw = create_input(
            self.pt_config, self.pt_config.temporal_patch_size, 8, 8
        )
        pt_hidden, pt_cu_seqlens, pt_pos_emb = self._prepare_pt_forward_context(
            pixel_values, grid_thw
        )
        jax_pos_ids, jax_attn_mask = self._prepare_jax_context(grid_thw)
        self._run_perblock_comparison(
            pt_hidden,
            pt_cu_seqlens,
            pt_pos_emb,
            jax_pos_ids,
            jax_attn_mask,
            check_blocks=True,
            check_merger=False,
            check_deepstack=False,
        )

    def test_merger_alignment(self):
        """Test merger with shared input from PyTorch's last block output.

        Feeds PyTorch's final hidden state (after all blocks) to both PT and JAX
        mergers, isolating merger error to a single component.
        """
        pixel_values, grid_thw = create_input(
            self.pt_config, self.pt_config.temporal_patch_size, 8, 8
        )
        pt_hidden, pt_cu_seqlens, pt_pos_emb = self._prepare_pt_forward_context(
            pixel_values, grid_thw
        )
        jax_pos_ids, jax_attn_mask = self._prepare_jax_context(grid_thw)
        self._run_perblock_comparison(
            pt_hidden,
            pt_cu_seqlens,
            pt_pos_emb,
            jax_pos_ids,
            jax_attn_mask,
            check_blocks=False,
            check_merger=True,
            check_deepstack=False,
        )

    def test_deepstack_merger_alignment(self):
        """Test each deepstack merger with shared input.

        At deepstack layer indices (e.g., 8, 16, 24), feeds PyTorch's hidden state
        to both PT and JAX deepstack mergers. Isolates each merger's error.
        """
        pixel_values, grid_thw = create_input(
            self.pt_config, self.pt_config.temporal_patch_size, 8, 8
        )
        pt_hidden, pt_cu_seqlens, pt_pos_emb = self._prepare_pt_forward_context(
            pixel_values, grid_thw
        )
        jax_pos_ids, jax_attn_mask = self._prepare_jax_context(grid_thw)
        self._run_perblock_comparison(
            pt_hidden,
            pt_cu_seqlens,
            pt_pos_emb,
            jax_pos_ids,
            jax_attn_mask,
            check_blocks=False,
            check_merger=False,
            check_deepstack=True,
        )

    # =========================================================================
    # Forward Pass Tests (per-block comparison at different resolutions)
    # =========================================================================

    def test_forward_small(self):
        """Test full pipeline with small resolution (8x8 patches) - per-block comparison."""
        pixel_values, grid_thw = create_input(
            self.pt_config, self.pt_config.temporal_patch_size, 8, 8
        )
        pt_hidden, pt_cu_seqlens, pt_pos_emb = self._prepare_pt_forward_context(
            pixel_values, grid_thw
        )
        jax_pos_ids, jax_attn_mask = self._prepare_jax_context(grid_thw)
        self._run_perblock_comparison(
            pt_hidden,
            pt_cu_seqlens,
            pt_pos_emb,
            jax_pos_ids,
            jax_attn_mask,
        )

    def test_forward_medium(self):
        """Test full pipeline with medium resolution (14x14 patches) - per-block comparison."""
        pixel_values, grid_thw = create_input(
            self.pt_config, self.pt_config.temporal_patch_size, 14, 14
        )
        pt_hidden, pt_cu_seqlens, pt_pos_emb = self._prepare_pt_forward_context(
            pixel_values, grid_thw
        )
        jax_pos_ids, jax_attn_mask = self._prepare_jax_context(grid_thw)
        self._run_perblock_comparison(
            pt_hidden,
            pt_cu_seqlens,
            pt_pos_emb,
            jax_pos_ids,
            jax_attn_mask,
        )

    def test_forward_large(self):
        """Test full pipeline with large resolution (28x28 patches) - per-block comparison."""
        pixel_values, grid_thw = create_input(
            self.pt_config, self.pt_config.temporal_patch_size, 28, 28
        )
        pt_hidden, pt_cu_seqlens, pt_pos_emb = self._prepare_pt_forward_context(
            pixel_values, grid_thw
        )
        jax_pos_ids, jax_attn_mask = self._prepare_jax_context(grid_thw)
        self._run_perblock_comparison(
            pt_hidden,
            pt_cu_seqlens,
            pt_pos_emb,
            jax_pos_ids,
            jax_attn_mask,
        )

    def test_forward_video(self):
        """Test full pipeline with video input (4 temporal frames) - per-block comparison."""
        pixel_values, grid_thw = create_input(
            self.pt_config, self.pt_config.temporal_patch_size * 2, 8, 8
        )
        pt_hidden, pt_cu_seqlens, pt_pos_emb = self._prepare_pt_forward_context(
            pixel_values, grid_thw
        )
        jax_pos_ids, jax_attn_mask = self._prepare_jax_context(grid_thw)
        self._run_perblock_comparison(
            pt_hidden,
            pt_cu_seqlens,
            pt_pos_emb,
            jax_pos_ids,
            jax_attn_mask,
        )

    def test_forward_non_square(self):
        """Test full pipeline with non-square input - per-block comparison."""
        for h, w in [(14, 8), (8, 14), (20, 10)]:
            with self.subTest(h=h, w=w):
                pixel_values, grid_thw = create_input(
                    self.pt_config, self.pt_config.temporal_patch_size, h, w
                )
                pt_hidden, pt_cu_seqlens, pt_pos_emb = self._prepare_pt_forward_context(
                    pixel_values, grid_thw
                )
                jax_pos_ids, jax_attn_mask = self._prepare_jax_context(grid_thw)
                self._run_perblock_comparison(
                    pt_hidden,
                    pt_cu_seqlens,
                    pt_pos_emb,
                    jax_pos_ids,
                    jax_attn_mask,
                )

    def test_forward_long_video(self):
        """Test full pipeline with longer video (8 temporal frames) - per-block comparison."""
        temporal_frames = self.pt_config.temporal_patch_size * 4  # 4x temporal patches
        pixel_values, grid_thw = create_input(self.pt_config, temporal_frames, 8, 8, seed=123)
        pt_hidden, pt_cu_seqlens, pt_pos_emb = self._prepare_pt_forward_context(
            pixel_values, grid_thw
        )
        jax_pos_ids, jax_attn_mask = self._prepare_jax_context(grid_thw)
        self._run_perblock_comparison(
            pt_hidden,
            pt_cu_seqlens,
            pt_pos_emb,
            jax_pos_ids,
            jax_attn_mask,
        )

    def test_forward_batch_images(self):
        """Test full pipeline with multiple images concatenated - per-block comparison."""
        # Create two separate images
        pixel_values1, grid_thw1 = create_input(
            self.pt_config, self.pt_config.temporal_patch_size, 8, 8, seed=42
        )
        pixel_values2, grid_thw2 = create_input(
            self.pt_config, self.pt_config.temporal_patch_size, 8, 8, seed=123
        )

        # Prepare and concatenate patches
        with torch.no_grad():
            pt_input1 = prepare_input(pixel_values1, self.pt_config, torch.float32)
            pt_input2 = prepare_input(pixel_values2, self.pt_config, torch.float32)
            pt_patches = torch.cat([pt_input1, pt_input2], dim=0)

        grid_thw = np.concatenate([grid_thw1, grid_thw2], axis=0)

        # Prepare contexts from pre-concatenated patches
        pt_hidden, pt_cu_seqlens, pt_pos_emb = self._prepare_pt_context_from_patches(
            pt_patches, grid_thw
        )
        jax_pos_ids, jax_attn_mask = self._prepare_jax_context(grid_thw)

        self._run_perblock_comparison(
            pt_hidden,
            pt_cu_seqlens,
            pt_pos_emb,
            jax_pos_ids,
            jax_attn_mask,
        )

    # =========================================================================
    # Single-Operation Precision Tests
    # =========================================================================

    def test_layernorm_precision(self):
        """Verify Flax LayerNorm matches PyTorch LayerNorm within atol=1e-5.

        Uses actual model weights from block 0's norm1 to ensure realistic test.
        """
        np.random.seed(42)
        x_np = np.random.randn(64, self.pt_config.hidden_size).astype(np.float32)

        # PyTorch LayerNorm with actual model weights
        with torch.no_grad():
            pt_ln = (
                torch.nn.LayerNorm(self.pt_config.hidden_size, eps=1e-6).eval().to(torch.float32)
            )
            pt_ln.weight.copy_(self.pt_model.blocks[0].norm1.weight)
            pt_ln.bias.copy_(self.pt_model.blocks[0].norm1.bias)
            pt_out = pt_ln(torch.from_numpy(x_np))

        # JAX LayerNorm (using actual model's block 0 norm1)
        with jax.set_mesh(self.mesh):
            jax_out = self.jax_model.blocks[0].norm1(jnp.array(x_np))

        self._assert_close(pt_out, jax_out, "layernorm", rtol=1e-5, atol=1e-5)

    def test_gelu_precision(self):
        """Verify JAX GELU matches PyTorch GELU within atol=1e-6.

        Element-wise activation should have very tight precision.
        Note: atol=1e-6 accounts for normal float32 erf implementation
        differences between JAX (XLA) and PyTorch (libm) on CPU.
        """
        np.random.seed(42)
        x_np = np.random.randn(100, self.pt_config.hidden_size).astype(np.float32)

        pt_out = torch.nn.functional.gelu(torch.from_numpy(x_np))
        jax_out = jax.nn.gelu(jnp.array(x_np), approximate=False)

        self._assert_close(pt_out, jax_out, "gelu", rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
