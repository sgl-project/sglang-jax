# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0

"""Qwen3OmniMoe Vision Encoder main implementation."""

import jax
import jax.numpy as jnp
from flax import nnx

from .vision_config import Qwen3OmniMoeVisionConfig
from .vision_layers import Vision3DPatchEmbed, VisionPatchMerger, VisionTransformerBlock


class Qwen3OmniMoeVisionEncoder(nnx.Module):
    """
    Qwen3OmniMoe Vision Encoder for processing images and videos.

    Architecture:
        1. 3D Conv Patch Embedding
        2. Learnable Position Embeddings (interpolated)
        3. 27 Vision Transformer Blocks with 2D RoPE
        4. Deepstack feature extraction at layers 8, 16, 24
        5. Final Patch Merger to language model dimension
    """

    def __init__(
        self,
        config: Qwen3OmniMoeVisionConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs = None,
    ):
        self.config = config
        self.mesh = mesh

        if dtype is None:
            dtype = getattr(jnp, config.dtype)
        self.dtype = dtype

        if rngs is None:
            rngs = nnx.Rngs(0)

        # Patch embedding
        self.patch_embed = Vision3DPatchEmbed(
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            rngs=rngs,
        )

        # Learnable position embeddings
        self.pos_embed = nnx.Embed(
            num_embeddings=config.num_position_embeddings,
            features=config.hidden_size,
            rngs=rngs,
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        # Transformer blocks - use nnx.List for Flax NNX 0.12.0+ compatibility
        self.blocks = nnx.List(
            [
                VisionTransformerBlock(
                    config=config,
                    mesh=mesh,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(config.depth)
            ]
        )

        # Deepstack mergers (for intermediate features) - use nnx.List for Flax NNX 0.12.0+ compatibility
        self.deepstack_mergers = nnx.List(
            [
                VisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,  # PyTorch uses True for deepstack mergers
                    mesh=mesh,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in config.deepstack_visual_indexes
            ]
        )

        # Final merger
        self.merger = VisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,  # PyTorch uses False for final merger
            mesh=mesh,
            dtype=dtype,
            rngs=rngs,
        )

        self.spatial_merge_size = config.spatial_merge_size
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

    def interpolate_pos_embed(
        self,
        grid_thw: jax.Array,
    ) -> jax.Array:
        """
        Interpolate position embeddings for variable resolution input.
        Matches PyTorch's fast_pos_embed_interpolate logic.

        Args:
            grid_thw: (num_images, 3) - [T, H_patches, W_patches]

        Returns:
            pos_embeds: (total_patches, hidden_size)
        """
        grid_ts = grid_thw[:, 0]
        grid_hs = grid_thw[:, 1]
        grid_ws = grid_thw[:, 2]

        all_pos_embeds = []

        # Process each image/video separately
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            # Convert traced values to Python ints for use in shape operations
            t_val = int(t)
            h_val = int(h)
            w_val = int(w)

            # Create interpolation indices
            h_idxs = jnp.linspace(0, self.num_grid_per_side - 1, h_val)
            w_idxs = jnp.linspace(0, self.num_grid_per_side - 1, w_val)

            h_idxs_floor = jnp.floor(h_idxs).astype(jnp.int32)
            w_idxs_floor = jnp.floor(w_idxs).astype(jnp.int32)
            h_idxs_ceil = jnp.minimum(h_idxs_floor + 1, self.num_grid_per_side - 1)
            w_idxs_ceil = jnp.minimum(w_idxs_floor + 1, self.num_grid_per_side - 1)

            # Compute interpolation weights
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            # Compute base indices for 2D grid
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            # Four corners for bilinear interpolation
            idx_tl = base_h[:, None] + w_idxs_floor[None, :]  # Top-left
            idx_tr = base_h[:, None] + w_idxs_ceil[None, :]  # Top-right
            idx_bl = base_h_ceil[:, None] + w_idxs_floor[None, :]  # Bottom-left
            idx_br = base_h_ceil[:, None] + w_idxs_ceil[None, :]  # Bottom-right

            # Flatten indices
            idx_tl = idx_tl.reshape(-1)
            idx_tr = idx_tr.reshape(-1)
            idx_bl = idx_bl.reshape(-1)
            idx_br = idx_br.reshape(-1)

            # Fetch embeddings
            emb_tl = self.pos_embed(idx_tl)  # (h*w, hidden_size)
            emb_tr = self.pos_embed(idx_tr)
            emb_bl = self.pos_embed(idx_bl)
            emb_br = self.pos_embed(idx_br)

            # Bilinear interpolation weights
            dh_2d = dh[:, None]
            dw_2d = dw[None, :]
            w_tl = ((1 - dh_2d) * (1 - dw_2d)).reshape(-1, 1)
            w_tr = ((1 - dh_2d) * dw_2d).reshape(-1, 1)
            w_bl = (dh_2d * (1 - dw_2d)).reshape(-1, 1)
            w_br = (dh_2d * dw_2d).reshape(-1, 1)

            pos_embed = emb_tl * w_tl + emb_tr * w_tr + emb_bl * w_bl + emb_br * w_br

            # Repeat for temporal and apply spatial-merge permutation
            pos_embed = jnp.broadcast_to(
                pos_embed[None, :, :], (t_val, h_val * w_val, pos_embed.shape[-1])
            )
            merge_size = self.config.spatial_merge_size
            pos_embed = pos_embed.reshape(
                t_val,
                h_val // merge_size,
                merge_size,
                w_val // merge_size,
                merge_size,
                self.config.hidden_size,
            )
            pos_embed = jnp.transpose(pos_embed, (0, 1, 3, 2, 4, 5))
            pos_embed = pos_embed.reshape(-1, self.config.hidden_size)

            all_pos_embeds.append(pos_embed)

        # Concatenate all position embeddings
        return jnp.concatenate(all_pos_embeds, axis=0)

    def compute_2d_position_ids(
        self,
        grid_thw: jax.Array,
    ) -> jax.Array:
        """
        Compute 2D spatial position IDs for RoPE in spatial-merge order.

        This matches PyTorch's rot_pos_emb which generates positions in spatial-merge order:
        For merge_size=2 and 8x8 grid: (0,0), (0,1), (1,0), (1,1), (0,2), (0,3), (1,2), (1,3), ...

        Args:
            grid_thw: (num_images, 3) - [T, H_patches, W_patches]

        Returns:
            position_ids: (total_patches, 2) - [h_pos, w_pos] in spatial-merge order
        """
        all_pos_ids = []
        merge_size = self.config.spatial_merge_size

        for num_frames, height, width in grid_thw:
            # Convert traced values to Python ints
            num_frames_val = int(num_frames)
            height_val = int(height)
            width_val = int(width)

            merged_h = height_val // merge_size
            merged_w = width_val // merge_size

            # Create spatial-merge order position IDs
            # block_rows: [0, 1, 2, ...] for blocks
            # intra_row: [0, 1] for positions within block
            block_rows = jnp.arange(merged_h)
            block_cols = jnp.arange(merged_w)
            intra_row = jnp.arange(merge_size)
            intra_col = jnp.arange(merge_size)

            # Compute full-resolution positions
            # row_idx: block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            # This creates the pattern: for each (block_h, block_w), iterate (intra_h, intra_w)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            # Expand and flatten
            row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size))
            col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size))
            row_idx = row_idx.reshape(-1)
            col_idx = col_idx.reshape(-1)

            coords = jnp.stack([row_idx, col_idx], axis=-1)  # (h*w, 2)

            # Repeat for temporal dimension
            if num_frames_val > 1:
                coords = jnp.tile(coords, (num_frames_val, 1))

            all_pos_ids.append(coords)

        return jnp.concatenate(all_pos_ids, axis=0)

    def _apply_spatial_merge_permutation(
        self,
        hidden_states: jax.Array,  # (total_patches, hidden_size)
        grid_thw: jax.Array,  # (num_images, 3) - [T, H_patches, W_patches]
    ) -> jax.Array:
        """
        Apply spatial merge permutation to hidden_states to match pos_embeds order.

        The patch_embed outputs patches in row-major order (T, H, W flattened),
        but position embeddings and position IDs use spatial-merge-permuted order.
        This function reorders hidden_states to match.

        Args:
            hidden_states: (total_patches, hidden_size) in row-major order
            grid_thw: (num_images, 3) containing [T, H_patches, W_patches]

        Returns:
            hidden_states: (total_patches, hidden_size) in spatial-merge-permuted order
        """
        merge_size = self.spatial_merge_size
        all_permuted = []
        offset = 0

        for num_frames, height, width in grid_thw:
            # Convert traced values to Python ints
            num_frames_val = int(num_frames)
            height_val = int(height)
            width_val = int(width)

            num_patches = num_frames_val * height_val * width_val
            patches = hidden_states[offset : offset + num_patches]  # (t*h*w, hidden_size)

            # Reshape to (t, h, w, hidden_size)
            patches = patches.reshape(num_frames_val, height_val, width_val, -1)

            # Apply spatial merge permutation
            # (t, h, w, hidden_size) -> (t, h//merge_size, merge_size, w//merge_size, merge_size, hidden_size)
            patches = patches.reshape(
                num_frames_val,
                height_val // merge_size,
                merge_size,
                width_val // merge_size,
                merge_size,
                -1,
            )

            # Transpose to (t, h//merge_size, w//merge_size, merge_size, merge_size, hidden_size)
            patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))

            # Flatten back to (t*h*w, hidden_size)
            patches = patches.reshape(-1, self.config.hidden_size)

            all_permuted.append(patches)
            offset += num_patches

        return jnp.concatenate(all_permuted, axis=0)

    def _create_attention_mask(self, grid_thw: jax.Array) -> jax.Array:
        """
        Create attention mask to prevent cross-sequence attention.

        IMPORTANT: This matches PyTorch's cu_seqlens behavior where attention is computed
        WITHIN each temporal frame separately. Each (h*w) block of patches forms an
        independent attention group.

        For video with grid_thw = [[2, 8, 8]] (2 temporal patches, 8x8 spatial):
        - Patches 0-63 (t=0) attend only to each other
        - Patches 64-127 (t=1) attend only to each other

        Args:
            grid_thw: (num_images, 3) - [T, H_patches, W_patches]

        Returns:
            attention_mask: (total_patches, total_patches) or None if single sequence
                           - 0.0 for valid positions (can attend)
                           - -inf for masked positions (cannot attend)
        """
        # Calculate sequence lengths for each temporal frame
        # This matches PyTorch: cu_seqlens = repeat_interleave(h*w, t).cumsum()
        all_seq_lens = []
        for t, h, w in grid_thw:
            t_val = int(t)
            h_val = int(h)
            w_val = int(w)
            # Each temporal frame has h*w patches as one sequence
            seq_len = h_val * w_val
            for _ in range(t_val):
                all_seq_lens.append(seq_len)

        total_patches = sum(all_seq_lens)

        # Check if we can skip masking (single sequence)
        if len(all_seq_lens) == 1:
            return None

        # Create block diagonal mask matching PyTorch cu_seqlens behavior
        # Initialize with -inf (mask everything by default)
        mask = jnp.full((total_patches, total_patches), -1e10, dtype=jnp.float32)

        offset = 0
        for seq_len in all_seq_lens:
            # Allow attention within the same sequence (set to 0.0)
            mask = mask.at[offset : offset + seq_len, offset : offset + seq_len].set(0.0)
            offset += seq_len

        return mask

    def _validate_input_shapes(self, grid_thw: jax.Array):
        """
        Validate that input dimensions are divisible by spatial_merge_size.

        Args:
            grid_thw: (num_images, 3)
        """
        merge_size = self.spatial_merge_size

        # Check height divisibility
        h_rem = grid_thw[:, 1] % merge_size
        if jnp.any(h_rem != 0):
            raise ValueError(f"Input height must be divisible by spatial_merge_size {merge_size}")

        # Check width divisibility
        w_rem = grid_thw[:, 2] % merge_size
        if jnp.any(w_rem != 0):
            raise ValueError(f"Input width must be divisible by spatial_merge_size {merge_size}")

    def __call__(
        self,
        pixel_values: jax.Array,  # (B, T, H, W, C)
        grid_thw: jax.Array,  # (B, 3) - [T, H_patches, W_patches]
    ) -> dict[str, jax.Array]:
        """
        Forward pass of vision encoder.

        Args:
            pixel_values: (B, T, H, W, C) video/image tensor
            grid_thw: (B, 3) containing [T, H_patches, W_patches] for each input

        Returns:
            Dictionary containing:
                - last_hidden_state: (total_patches, hidden_size)
                - pooler_output: (total_patches, out_hidden_size)
                - deepstack_features: List of (total_patches, out_hidden_size)

        Raises:
            ValueError: If input dimensions are not compatible with spatial_merge_size
        """
        # 0. Validate input shapes
        self._validate_input_shapes(grid_thw)

        # 1. Patch Embedding
        hidden_states = self.patch_embed(pixel_values)  # (total_patches, hidden_size)

        # 2. Apply spatial merge permutation to match PyTorch order
        hidden_states = self._apply_spatial_merge_permutation(hidden_states, grid_thw)

        # 3. Add Position Embeddings (both in spatial-merge order)
        pos_embeds = self.interpolate_pos_embed(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 4. Compute 2D position IDs for RoPE
        position_ids = self.compute_2d_position_ids(grid_thw)

        # 5. Create attention mask to match PyTorch's cu_seqlens behavior
        # Mask is needed when:
        # - batch > 1 (multiple images) OR
        # - any image has T > 1 (video with multiple temporal patches)
        # This ensures each temporal frame attends only within itself
        attention_mask = self._create_attention_mask(grid_thw)

        # 6. Through Transformer Blocks
        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            # Extract deepstack features
            if layer_idx in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_idx)
                feature = self.deepstack_mergers[idx](hidden_states)
                deepstack_features.append(feature)

        # 7. Final projection (merger)
        merged_hidden_states = self.merger(hidden_states)

        return {
            "last_hidden_state": hidden_states,  # Pre-merger: (total_patches, hidden_size)
            "pooler_output": merged_hidden_states,  # Post-merger: (merged_patches, out_hidden_size)
            "deepstack_features": deepstack_features,
            "encoder_hidden_state": hidden_states,  # Alias for pre-merger output
        }

    def load_weights(self, model_path: str):
        """
        Load weights from HuggingFace checkpoint.

        Args:
            model_path: Path to model checkpoint directory

        Note:
            Weight loading must be done within mesh context for proper TP sharding.
        """
        import glob
        import os

        import numpy as np
        from flax import nnx
        from safetensors import safe_open

        from .vision_weights_mapping import create_vision_encoder_weight_mappings

        # Scan safetensors files
        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not safetensor_files:
            return

        # Build key -> file mapping
        key_to_file = {}
        for sf_file in safetensor_files:
            with safe_open(sf_file, framework="np") as f:
                for key in f.keys():  # noqa: SIM118
                    key_to_file[key] = sf_file

        # Possible HF prefixes
        prefixes = ["", "thinker.visual.", "vision_encoder.", "model.vision_encoder."]

        def find_hf_key(key: str) -> str | None:
            """Find actual HF key with prefix detection."""
            for prefix in prefixes:
                if prefix + key in key_to_file:
                    return prefix + key
            return None

        def get_param(path: str):
            """Navigate to parameter by dot-separated path."""
            obj = self
            for p in path.split("."):
                obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
            return obj

        # Load weights within mesh context for TP sharding
        weight_mappings = create_vision_encoder_weight_mappings(self.config)

        with jax.set_mesh(self.mesh):
            for mapping_key, mapping in weight_mappings.items():
                hf_key = find_hf_key(mapping_key)
                if hf_key is None:
                    continue

                try:
                    with safe_open(key_to_file[hf_key], framework="np") as f:
                        weight = f.get_tensor(hf_key)

                    # Apply transformations
                    if mapping.transpose_axes is not None:
                        weight = np.transpose(weight, mapping.transpose_axes)
                    elif mapping.transpose:
                        weight = np.transpose(weight)

                    # Set parameter
                    param = get_param(mapping.target_path)
                    if isinstance(param, nnx.Variable):
                        param[...] = jnp.array(weight, dtype=param[...].dtype)
                except Exception:
                    pass
