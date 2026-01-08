import math
from typing import Any

import jax.numpy as jnp

# Global cache dictionary for storing created rotary embedding instances
_ROPE_DICT: dict[tuple, Any] = {}


def _rotate_neox(x: jnp.ndarray) -> jnp.ndarray:
    """Neox-style rotation operation"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def _rotate_gptj(x: jnp.ndarray) -> jnp.ndarray:
    """GPT-J-style rotation operation"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return x.reshape(*x.shape[:-2], -1)


def _apply_rotary_emb(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    is_neox_style: bool,
) -> jnp.ndarray:
    """
    Apply rotary position embedding

    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use Neox-style rotary position embedding
    """
    cos = jnp.expand_dims(cos, axis=-2)
    sin = jnp.expand_dims(sin, axis=-2)

    if is_neox_style:
        x1, x2 = jnp.split(x, 2, axis=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return jnp.concatenate((o1, o2), axis=-1)
    else:
        return jnp.stack((o1, o2), axis=-1).reshape(*x.shape[:-1], -1)


class RotaryEmbedding:
    """Base rotary position embedding class"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        # Compute cos and sin cache
        self.cos_sin_cache = self._compute_cos_sin_cache()

    def _compute_inv_freq(self, base: int | float) -> jnp.ndarray:
        """Compute inverse frequencies"""
        inv_freq = 1.0 / (
            base ** (jnp.arange(0, self.rotary_dim, 2, dtype=self.dtype) / self.rotary_dim)
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> jnp.ndarray:
        """Compute cos and sin cache"""
        inv_freq = self._compute_inv_freq(self.base)
        t = jnp.arange(self.max_position_embeddings, dtype=self.dtype)

        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        return jnp.concatenate((cos, sin), axis=-1)

    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
        offsets: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass"""
        if offsets is not None:
            positions = positions + offsets

        positions = positions.flatten()
        num_tokens = positions.shape[0]

        # Get corresponding cos and sin values
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = jnp.split(cos_sin, 2, axis=-1)

        # Process query
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]

        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

        # Process key
        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]

        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)

        return query, key


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """Rotary position embedding with linear scaling support"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: list[float] | float,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: list[float] = scaling_factors
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

        # Compute mapping from scaling factors to offsets
        self._scaling_factor_to_offset: dict[float, int] = {}
        offsets: list[int] = []
        for i, scaling_factor in enumerate(self.scaling_factors):
            # max_len is calculated but not used elsewhere
            if not offsets:
                offset = 0
            else:
                last_offset = offsets[-1]
                prev_max_len = self.max_position_embeddings * self.scaling_factors[i - 1]
                offset = last_offset + int(prev_max_len)
            offsets.append(offset)
            self._scaling_factor_to_offset[scaling_factor] = offset

    def _compute_cos_sin_cache(self) -> jnp.ndarray:
        """Compute cos and sin cache with support for multiple scaling factors"""
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: list[jnp.ndarray] = []

        for scaling_factor in self.scaling_factors:
            max_len = int(self.max_position_embeddings * scaling_factor)
            t = jnp.arange(max_len, dtype=self.dtype) / scaling_factor

            freqs = jnp.einsum("i,j -> ij", t, inv_freq)
            cos = jnp.cos(freqs)
            sin = jnp.sin(freqs)
            cache = jnp.concatenate((cos, sin), axis=-1)
            cache_list.append(cache)

        return jnp.concatenate(cache_list, axis=0)

    @property
    def scaling_factor_to_offset(self) -> dict[float, int]:
        return self._scaling_factor_to_offset


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """Rotary position embedding with dynamic NTK scaling support"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

    def _compute_cos_sin_cache(self) -> jnp.ndarray:
        """Compute cos and sin cache with dynamic NTK scaling applied"""
        max_len = int(self.max_position_embeddings * self.scaling_factor)
        # Adjust base for dynamic scaling
        base = self.base * (
            (self.scaling_factor * max_len / self.max_position_embeddings)
            - (self.scaling_factor - 1)
        ) ** (self.rotary_dim / (self.rotary_dim - 2))

        inv_freq = self._compute_inv_freq(base)
        t = jnp.arange(max_len, dtype=self.dtype)

        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        return jnp.concatenate((cos, sin), axis=-1)


# YaRN-related helper functions
def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: jnp.dtype) -> jnp.ndarray:
    if low == high:
        high += 0.001  # Prevent division by zero

    linear_func = (jnp.arange(dim, dtype=dtype) - low) / (high - low)
    return jnp.clip(linear_func, 0, 1)


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """Rotary position embedding with YaRN scaling method support"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: jnp.dtype = jnp.float32,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling
        self.mscale = float(_yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> jnp.ndarray:
        pos_freqs = self.base ** (
            jnp.arange(0, self.rotary_dim, 2, dtype=self.dtype) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )

        # Get n-d rotation scaling (corrected for extrapolation)
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, self.dtype)
        ) * self.extrapolation_factor

        return inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    def _compute_cos_sin_cache(self) -> jnp.ndarray:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        max_len = int(self.max_position_embeddings * self.scaling_factor)
        t = jnp.arange(max_len, dtype=self.dtype)

        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        cos = jnp.cos(freqs) * self.mscale
        sin = jnp.sin(freqs) * self.mscale
        return jnp.concatenate((cos, sin), axis=-1)


class MRotaryEmbedding(RotaryEmbedding):
    """Rotary embedding with multimodal section support"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype = jnp.float32,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = False,
    ) -> None:
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

        # Validate and adjust mrope_section
        if self.mrope_section:
            expected_sum = rotary_dim // 2
            actual_sum = sum(self.mrope_section)
            if actual_sum != expected_sum:
                print(
                    f"MRoPE section sum mismatch: expected {expected_sum}, got {actual_sum}."
                    f" Adjusting mrope_section to match rotary_dim // 2 = {expected_sum}"
                )
                # Scale mrope_section proportionally
                if actual_sum > 0:
                    scale_factor = expected_sum / actual_sum
                    self.mrope_section = [
                        max(1, int(section * scale_factor)) for section in self.mrope_section
                    ]
                    # Adjust last element to ensure exact sum match
                    current_sum = sum(self.mrope_section)
                    if current_sum != expected_sum:
                        self.mrope_section[-1] += expected_sum - current_sum

        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

    def __call__(
        self,
        positions: jnp.ndarray,
        query: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass"""
        assert positions.ndim == 1 or positions.ndim == 2

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = jnp.split(cos_sin, 2, axis=-1)

        # Handle multimodal positions
        if positions.ndim == 2 and self.mrope_section:
            if self.mrope_interleaved:
                cos = self._apply_interleaved_rope(cos)
                sin = self._apply_interleaved_rope(sin)
            else:
                # Concatenate by sections
                cos_sections = jnp.split(cos, self.mrope_section, axis=-1)
                sin_sections = jnp.split(sin, self.mrope_section, axis=-1)
                cos = jnp.concatenate([cos_sections[i] for i in range(len(cos_sections))], axis=-1)
                sin = jnp.concatenate([sin_sections[i] for i in range(len(sin_sections))], axis=-1)

        # Process query
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]

        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

        # Process key
        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]

        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)

        return query, key

    def _apply_interleaved_rope(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply interleaved MRoPE"""
        x_t = x[0].copy()
        x_t = x_t.at[..., 1 : self.mrope_section[1] * 3 : 3].set(
            x[1, ..., 1 : self.mrope_section[1] * 3 : 3]
        )
        x_t = x_t.at[..., 2 : self.mrope_section[2] * 3 : 3].set(
            x[2, ..., 2 : self.mrope_section[2] * 3 : 3]
        )
        return x_t

    # Complete get_rope_index static method (adapted for JAX)
    @staticmethod
    def get_rope_index(
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        model_type: str,
        tokens_per_second: int | None = None,
        input_ids: jnp.ndarray | None = None,
        image_grid_thw: jnp.ndarray | None = None,
        video_grid_thw: jnp.ndarray | None = None,
        second_per_grid_ts: jnp.ndarray | None = None,
        **kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if model_type == "qwen3_omni_moe":
            # Adapt for qwen3-omni model
            return MRotaryEmbedding.get_rope_index_qwen3_omni(
                spatial_merge_size,
                image_token_id,
                video_token_id,
                vision_start_token_id,
                tokens_per_second,
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                **kwargs,
            )

        # Handle video grid_thw repetition
        if (
            model_type.startswith("qwen3_vl") or model_type.startswith("qwen3_vl_moe")
        ) and video_grid_thw is not None:
            # JAX version of repeat_interleave
            video_grid_thw = jnp.repeat(video_grid_thw, video_grid_thw[:, 0], axis=0)
            video_grid_thw = video_grid_thw.at[:, 0].set(1)

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            batch_size, seq_len = total_input_ids.shape

            # Initialize position_ids (JAX version)
            position_ids = jnp.ones(
                (3, batch_size, seq_len), dtype=total_input_ids.dtype, device=total_input_ids.device
            )

            image_index, video_index = 0, 0

            for i in range(batch_size):
                input_ids_i = total_input_ids[i]
                # Find positions of vision_start_token (JAX version)
                vision_start_indices = jnp.where(input_ids_i == vision_start_token_id)[0]

                # Fix: assign multiple variables separately to avoid unpacking error
                image_nums = 0
                video_nums = 0

                if vision_start_indices.size > 0:
                    vision_tokens = input_ids_i[vision_start_indices + 1]
                    # New: check if vision_tokens is empty (avoid index out of bounds)
                    if vision_tokens.size > 0:
                        image_nums = jnp.sum(vision_tokens == image_token_id)
                        video_nums = jnp.sum(vision_tokens == video_token_id)
                    else:
                        image_nums = 0
                        video_nums = 0

                input_tokens = input_ids_i.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums.item(), video_nums.item()

                for _ in range(remain_images + remain_videos):
                    # Find position of next image or video token
                    ed_image = (
                        input_tokens.index(image_token_id, st)
                        if (remain_images > 0 and image_token_id in input_tokens[st:])
                        else len(input_tokens) + 1
                    )
                    ed_video = (
                        input_tokens.index(video_token_id, st)
                        if (remain_videos > 0 and video_token_id in input_tokens[st:])
                        else len(input_tokens) + 1
                    )

                    if ed_image < ed_video:
                        # Handle image
                        t, h, w = (
                            image_grid_thw[image_index][0].item(),
                            image_grid_thw[image_index][1].item(),
                            image_grid_thw[image_index][2].item(),
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        # Handle video
                        t, h, w = (
                            video_grid_thw[video_index][0].item(),
                            video_grid_thw[video_index][1].item(),
                            video_grid_thw[video_index][2].item(),
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index].item()
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    # Calculate grid size
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t,
                        h // spatial_merge_size,
                        w // spatial_merge_size,
                    )
                    text_len = ed - st

                    # Add text position IDs
                    if text_len > 0:
                        st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                        text_pos = jnp.arange(text_len, dtype=position_ids.dtype) + st_idx
                        llm_pos_ids_list.append(jnp.tile(text_pos, (3, 1)))  # (3, text_len)

                    # Add vision position IDs
                    st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                    if model_type == "qwen2_5_vl":
                        # Time dimension calculation
                        range_tensor = jnp.arange(llm_grid_t)[:, None]
                        expanded_range = jnp.tile(range_tensor, (1, llm_grid_h * llm_grid_w))
                        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
                        t_index = time_tensor.astype(position_ids.dtype).flatten()
                    elif model_type in ("qwen2_vl", "qwen3_vl", "qwen3_vl_moe"):
                        t_index = jnp.repeat(
                            jnp.arange(llm_grid_t, dtype=position_ids.dtype),
                            llm_grid_h * llm_grid_w,
                        )
                    else:
                        raise RuntimeError(f"Unimplemented model type: {model_type}")

                    # Height and width dimensions
                    h_index = jnp.tile(
                        jnp.repeat(jnp.arange(llm_grid_h, dtype=position_ids.dtype), llm_grid_w),
                        llm_grid_t,
                    )
                    w_index = jnp.tile(
                        jnp.arange(llm_grid_w, dtype=position_ids.dtype), llm_grid_t * llm_grid_h
                    )

                    # Stack T/H/W position IDs
                    vision_pos = jnp.stack([t_index, h_index, w_index]) + st_idx
                    llm_pos_ids_list.append(vision_pos)

                    # Update start position
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # Handle remaining text
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - st
                    text_pos = jnp.arange(text_len, dtype=position_ids.dtype) + st_idx
                    llm_pos_ids_list.append(jnp.tile(text_pos, (3, 1)))

                # Concatenate position IDs
                llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1)  # (3, seq_len)
                position_ids = position_ids.at[:, i, :].set(llm_positions)

                # Calculate delta
                mrope_position_deltas.append(llm_positions.max().item() + 1 - len(input_tokens))

            # Convert delta to JAX array
            mrope_position_deltas = jnp.array(
                mrope_position_deltas, dtype=position_ids.dtype, device=position_ids.device
            ).reshape(-1, 1)
            return position_ids, mrope_position_deltas
        else:
            # Pure text case
            s = input_ids.shape[1]
            position_ids = jnp.tile(
                jnp.arange(s, dtype=input_ids.dtype)[None, None, :], (3, input_ids.shape[0], 1)
            ).astype(input_ids.dtype)

            # Calculate delta
            max_position_ids = jnp.max(position_ids, axis=(0, 2), keepdims=True).squeeze(2)
            mrope_position_deltas = max_position_ids + 1 - s
            return position_ids, mrope_position_deltas

    # Complete qwen3_omni-specific get_rope_index (adapted for JAX)
    @staticmethod
    def get_rope_index_qwen3_omni(
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        tokens_per_second: int | None = None,
        input_ids: jnp.ndarray | None = None,
        image_grid_thw: jnp.ndarray | None = None,
        video_grid_thw: jnp.ndarray | None = None,
        second_per_grid_ts: jnp.ndarray | None = None,
        **kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        audio_start_token_id = kwargs["audio_start_token_id"]
        position_id_per_seconds = kwargs["position_id_per_seconds"]
        use_audio_in_video = kwargs.get("use_audio_in_video", False)
        audio_seqlens = kwargs.get("audio_seqlens")
        second_per_grids = second_per_grid_ts

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            batch_size, seq_len = total_input_ids.shape

            # Initialize position_ids (JAX version, using float type)
            position_ids = jnp.zeros(
                (3, batch_size, seq_len), dtype=jnp.float32, device=total_input_ids.device
            )

            image_idx, video_idx, audio_idx = 0, 0, 0

            for i in range(batch_size):
                current_input_ids = total_input_ids[i]
                # Calculate multimodal token count
                vision_start_indices = jnp.where(current_input_ids == vision_start_token_id)[0]
                image_nums, video_nums, audio_nums = 0, 0, 0

                if vision_start_indices.size > 0:
                    vision_tokens = current_input_ids[vision_start_indices + 1]
                    image_nums = jnp.sum(vision_tokens == image_token_id)
                    video_nums = (
                        jnp.sum(vision_tokens == audio_start_token_id)
                        if use_audio_in_video
                        else jnp.sum(vision_tokens == video_token_id)
                    )

                audio_nums = jnp.sum(current_input_ids == audio_start_token_id)
                input_tokens = current_input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = (
                    image_nums.item(),
                    video_nums.item(),
                    audio_nums.item(),
                )

                multimodal_nums = (
                    image_nums + audio_nums
                    if use_audio_in_video
                    else image_nums + video_nums + audio_nums
                )

                for _ in range(multimodal_nums.item()):
                    st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0.0

                    # Find next modality start position
                    ed_vision_start = (
                        input_tokens.index(vision_start_token_id, st)
                        if (
                            (remain_videos > 0 or remain_images > 0)
                            and vision_start_token_id in input_tokens[st:]
                        )
                        else len(input_tokens) + 1
                    )

                    ed_audio_start = (
                        input_tokens.index(audio_start_token_id, st)
                        if (remain_audios > 0 and audio_start_token_id in input_tokens[st:])
                        else len(input_tokens) + 1
                    )

                    min_ed = min(ed_vision_start, ed_audio_start)
                    text_len = min_ed - st

                    # Add text positions
                    if text_len != 0:
                        text_pos = jnp.arange(text_len, dtype=jnp.float32) + st_idx
                        llm_pos_ids_list.append(jnp.tile(text_pos, (3, 1)))
                        st_idx = llm_pos_ids_list[-1].max().item() + 1

                    # Handle BOS/EOS length
                    if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        bos_len, eos_len = 2, 2
                    else:
                        bos_len, eos_len = 1, 1

                    # Add BOS positions
                    bos_pos = jnp.arange(bos_len, dtype=jnp.float32) + st_idx
                    llm_pos_ids_list.append(jnp.tile(bos_pos, (3, 1)))
                    st_idx = llm_pos_ids_list[-1].max().item() + 1

                    # Handle audio-only input
                    if min_ed == ed_audio_start:
                        audio_len = MRotaryEmbedding._get_feat_extract_output_lengths(
                            audio_seqlens[audio_idx].item()
                        )
                        audio_pos = jnp.arange(audio_len, dtype=jnp.float32) + st_idx
                        llm_pos_ids_list.append(jnp.tile(audio_pos, (3, 1)))

                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1

                    # Handle image input
                    elif (
                        min_ed == ed_vision_start
                        and current_input_ids[ed_vision_start + 1] == image_token_id
                    ):
                        grid_t = image_grid_thw[image_idx][0].item()
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]

                        t_index = (
                            jnp.arange(grid_t, dtype=jnp.float32) * 1 * position_id_per_seconds
                        )

                        llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision(
                            st_idx,
                            image_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                            current_input_ids.device,
                        )

                        image_len = (
                            image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        ).item()
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1

                    # Handle video input
                    elif (
                        min_ed == ed_vision_start
                        and current_input_ids[ed_vision_start + 1] == video_token_id
                    ):
                        grid_t = video_grid_thw[video_idx][0].item()
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            jnp.arange(grid_t, dtype=jnp.float32)
                            * second_per_grids[video_idx].item()
                            * position_id_per_seconds
                        )

                        llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision(
                            st_idx,
                            video_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                            current_input_ids.device,
                        )

                        video_len = (
                            video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        ).item()
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    # Handle audio in video
                    elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        # Audio part
                        audio_len = MRotaryEmbedding._get_feat_extract_output_lengths(
                            audio_seqlens[audio_idx].item()
                        )
                        audio_pos = jnp.arange(audio_len, dtype=jnp.float32) + st_idx
                        audio_llm_pos_ids = jnp.tile(audio_pos, (3, 1))

                        # Video part
                        grid_t = video_grid_thw[video_idx][0].item()
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            jnp.arange(grid_t, dtype=jnp.float32)
                            * second_per_grids[video_idx].item()
                            * position_id_per_seconds
                        )

                        video_llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision(
                            st_idx,
                            video_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                            current_input_ids.device,
                        )

                        # Interleave audio and video positions
                        merged_pos = []
                        audio_idx_ptr, video_idx_ptr = 0, 0
                        while (
                            audio_idx_ptr < audio_llm_pos_ids.shape[1]
                            and video_idx_ptr < video_llm_pos_ids.shape[1]
                        ):
                            if (
                                audio_llm_pos_ids[0, audio_idx_ptr]
                                <= video_llm_pos_ids[0, video_idx_ptr]
                            ):
                                merged_pos.append(
                                    audio_llm_pos_ids[:, audio_idx_ptr : audio_idx_ptr + 1]
                                )
                                audio_idx_ptr += 1
                            else:
                                merged_pos.append(
                                    video_llm_pos_ids[:, video_idx_ptr : video_idx_ptr + 1]
                                )
                                video_idx_ptr += 1

                        # Add remaining parts
                        if audio_idx_ptr < audio_llm_pos_ids.shape[1]:
                            merged_pos.append(audio_llm_pos_ids[:, audio_idx_ptr:])
                        if video_idx_ptr < video_llm_pos_ids.shape[1]:
                            merged_pos.append(video_llm_pos_ids[:, video_idx_ptr:])

                        if merged_pos:
                            llm_pos_ids_list.append(jnp.concatenate(merged_pos, axis=1))

                        video_len = (
                            video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        ).item()
                        st += text_len + bos_len + audio_len + video_len + eos_len

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                    # Add EOS positions
                    eos_pos = jnp.arange(eos_len, dtype=jnp.float32) + st_idx
                    llm_pos_ids_list.append(jnp.tile(eos_pos, (3, 1)))

                # Handle remaining text
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0.0
                    text_len = len(input_tokens) - st
                    text_pos = jnp.arange(text_len, dtype=jnp.float32) + st_idx
                    llm_pos_ids_list.append(jnp.tile(text_pos, (3, 1)))

                # Concatenate position IDs
                llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1)
                position_ids = position_ids.at[:, i, :].set(llm_positions)

                # Calculate delta
                mrope_position_deltas.append(llm_positions.max().item() + 1 - len(input_tokens))

            # Convert delta to JAX array
            mrope_position_deltas = jnp.array(
                mrope_position_deltas, dtype=jnp.float32, device=position_ids.device
            ).reshape(-1, 1)
            return position_ids, mrope_position_deltas
        else:
            # Pure text case
            s = input_ids.shape[1]
            position_ids = jnp.tile(
                jnp.arange(s, dtype=jnp.float32)[None, None, :], (3, input_ids.shape[0], 1)
            )

            # Calculate delta
            max_position_ids = jnp.max(position_ids, axis=(0, 2), keepdims=True).squeeze(2)
            mrope_position_deltas = max_position_ids + 1 - s
            return position_ids, mrope_position_deltas

    # Helper method: calculate feature extraction output length
    @staticmethod
    def _get_feat_extract_output_lengths(input_lengths: int) -> int:
        """Calculate output length of convolution layer and audio encoder"""
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    # Helper method: get position IDs for visual modality
    @staticmethod
    def _get_llm_pos_ids_for_vision(
        st_idx: float,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: jnp.ndarray,
        grid_hs: jnp.ndarray,
        grid_ws: jnp.ndarray,
        device: Any,
    ) -> jnp.ndarray:
        grid_h = (grid_hs[vision_idx] // spatial_merge_size).item()
        grid_w = (grid_ws[vision_idx] // spatial_merge_size).item()

        # Generate position IDs for H and W dimensions
        h_index = jnp.tile(
            jnp.repeat(jnp.arange(grid_h, dtype=jnp.float32, device=device), grid_w), len(t_index)
        )

        w_index = jnp.tile(
            jnp.arange(grid_w, dtype=jnp.float32, device=device), len(t_index) * grid_h
        )

        # Generate position IDs for T dimension
        t_index = jnp.repeat(t_index, grid_h * grid_w)

        # Stack and add start offset
        llm_pos_ids = jnp.stack([t_index, h_index, w_index], axis=0) + st_idx
        return llm_pos_ids


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: dict[str, Any] | None = None,
    dtype: jnp.dtype | None = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    """Get rotary embedding instance (with cache)"""
    if dtype is None:
        dtype = jnp.float32

    # Process rope_scaling parameter for cache key
    rope_scaling_tuple = tuple(sorted(rope_scaling.items())) if rope_scaling is not None else None

    # Handle partial rotary factor
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)

    # Generate cache key
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_tuple,
        dtype,
    )

    # Check cache
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    # Create new rotary embedding instance
    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        scaling_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = LinearScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )
        elif scaling_type == "dynamic":
            if "alpha" in rope_scaling:
                raise NotImplementedError("DynamicNTKAlphaRotaryEmbedding not yet implemented")
            else:
                scaling_factor = rope_scaling["factor"]
                rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                )
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")
            }
            rotary_emb = YaRNScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
            )
        elif scaling_type == "default" and "mrope_section" in rope_scaling:
            rotary_emb = MRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=rope_scaling["mrope_section"],
                mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type: {scaling_type}")

    # Store in cache
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


# Rotation helper functions
def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotate half of the input's hidden dimensions"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    unsqueeze_dim: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary position embedding"""
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
