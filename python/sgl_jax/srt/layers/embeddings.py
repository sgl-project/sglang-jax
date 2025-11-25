#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Embedding Layers."""

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn import dtypes
from flax.typing import PromoteDtypeFn
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils.profiling_utils import named_scope


class Embed(nnx.Module):
    """A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
      num_embeddings: number of embeddings.
      features: number of feature dimensions for each embedding.
      dtype: the dtype of the embedding vectors (default: float32).
      param_dtype: the dtype of the embedding parameters.
      promote_dtype: the dtype promotion function.
      kernel_axes: the axes of kernel weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.bfloat16,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        kernel_axes: tuple[str | None, ...] = (None, "tensor"),
        mesh: jax.sharding.Mesh | None = None,
    ):
        """
        Sets up the embedding parameters for the model.

        This method initializes the embedding parameters with logical partitioning.
        The embedding is represented as a parameter with the specified shape and data type.

        Args:
            num_embeddings: Number of embeddings in the vocabulary.
            features: Number of feature dimensions for each embedding.
            dtype: Data type for computations (forward pass, attend operations).
                   If None, uses the same dtype as the embedding parameter.
            param_dtype: Data type for storing the embedding parameters in memory.
                        Controls memory usage and precision of stored weights.
            promote_dtype: Function to handle dtype promotion during mixed-precision
                          computations between query/embedding tensors.
        """
        out_sharding = NamedSharding(mesh, P(*kernel_axes)) if mesh is not None else None
        self.embedding = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (num_embeddings, features),
                dtype=param_dtype,
                out_sharding=out_sharding,
            ),
        )
        self.kernel_axes = kernel_axes
        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype or self.embedding.value.dtype
        self.promote_dtype = promote_dtype
        self.mesh = mesh

    @named_scope
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        (embedding,) = self.promote_dtype((self.embedding.value,), dtype=self.dtype, inexact=False)
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, inputs.shape + (self.features,))

        output_pspec = P(*([None] * inputs.ndim), self.kernel_axes[-1])
        output_sharding = NamedSharding(self.mesh, output_pspec)
        output = embedding.at[inputs].get(out_sharding=output_sharding)
        return output

    def attend(self, query: jax.Array) -> jax.Array:
        """Attend over the embedding using a query array.

        Args:
          query: array with last dimension equal the feature depth `features` of the
            embedding.

        Returns:
          An array with final dim `num_embeddings` corresponding to the batched
          inner-product of the array of query vectors against each embedding.
          Commonly used for weight-sharing between embeddings and logit transform
          in NLP models.
        """
        query, embedding = self.promote_dtype((query, self.embedding.value), dtype=self.dtype)
        return jnp.dot(query, embedding.T)


class ParallelLMHead(Embed):
    """Language model head layer for vocabulary prediction.

    Inherits from Embed to enable weight tying with input embeddings.
    Note: This layer's __call__ method is disabled - weights should be used
    directly in the sampling/prediction phase.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.bfloat16,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        kernel_axes: tuple[str | None, ...] = ("tensor", None),
        mesh: jax.sharding.Mesh | None = None,
        use_bias: bool = False,
    ):
        """
        Initialize the language model head.

        Args:
            num_embeddings: Size of vocabulary.
            features: Hidden dimension size.
            dtype: Data type for computations. If None, uses param_dtype.
                   Enables mixed precision when different from param_dtype.
            param_dtype: Data type for parameter storage (weights and bias).
            promote_dtype: Function to handle dtype promotion during logits computation.
                          Controls how hidden_states and embedding tensors are promoted.
            use_bias: Whether to include bias parameters. Note: bias is currently
                     not used in logits computation, reserved for future extension.
        """
        super().__init__(
            num_embeddings=num_embeddings,
            features=features,
            dtype=dtype,
            param_dtype=param_dtype,
            promote_dtype=promote_dtype,
            kernel_axes=kernel_axes,
            mesh=mesh,
        )
        if use_bias:
            bias_sharding = NamedSharding(mesh, P(None, "tensor")) if mesh is not None else None
            self.bias = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_embeddings, self.features),
                    dtype=param_dtype,
                    out_sharding=bias_sharding,
                ),
            )
        else:
            self.bias = None

    def tie_weights(self, embed_tokens: Embed):
        """Tie the weights with word embeddings."""
        self.embedding = embed_tokens.embedding
        return self

    def __call__(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")


class RotaryEmbedding:
    """Rotary Position Embedding (safe to initialize inside JIT if needed)."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
    ):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        inv_freq_np = 1.0 / (base ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        self._inv_freq_np = inv_freq_np  # shape: (rotary_dim // 2,)

    @named_scope
    def __call__(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        positions = positions.flatten()  # [num_tokens]

        inv_freq = jnp.asarray(self._inv_freq_np, dtype=self.dtype)

        # Compute freqs = positions * inv_freq
        freqs = jnp.einsum("n,d->nd", positions.astype(jnp.float32), inv_freq)

        cos = jnp.cos(freqs).astype(self.dtype)
        sin = jnp.sin(freqs).astype(self.dtype)

        query_shape = query.shape
        num_tokens = positions.shape[0]
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)

        return query, key

    def _compute_inv_freq(self, base: int | float) -> jax.Array:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (
            base ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> jax.Array:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        cache = jnp.concatenate((cos, sin), axis=-1)
        return cache


class Llama3RotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

    def _compute_inv_freq(self, base: int | float) -> jax.Array:
        inv_freqs = super()._compute_inv_freq(base)
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
        else:
            smooth = 0
        new_freqs = jnp.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            jnp.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor + smooth * inv_freqs,
            ),
        )
        return new_freqs


def apply_interleaved_rope(x: jax.Array, mrope_section: list[int]) -> jax.Array:
    """Apply interleaved MRoPE to 3D rotary embeddings in JAX.

    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT].

    Args:
        x: Input tensor of shape [3, num_tokens, dim].
           x[0] is Time freqs, x[1] is Height freqs, x[2] is Width freqs.
        mrope_section: [t, h, w] section lengths.
           e.g. [16, 24, 24] -> total 64.

    Returns:
        A single tensor of shape [num_tokens, dim] with interleaved frequencies.
    """
    # x shape: [3, num_tokens, dim]
    # mrope_section example: [16, 24, 24] (sum=64)

    # Initialize with Time frequencies (x[0])
    # We will overwrite specific indices with Height and Width frequencies
    x_t = x[0]  # [num_tokens, dim]

    # Height indices: start at 1, end at h*3, step 3
    # Corresponds to x[..., 1::3] in the target layout
    # We take values from x[1] (Height) at the same slice
    h_slice = slice(1, mrope_section[1] * 3, 3)
    x_t = x_t.at[..., h_slice].set(x[1, ..., h_slice])

    # Width indices: start at 2, end at w*3, step 3
    # Corresponds to x[..., 2::3] in the target layout
    # We take values from x[2] (Width) at the same slice
    w_slice = slice(2, mrope_section[2] * 3, 3)
    x_t = x_t.at[..., w_slice].set(x[2, ..., w_slice])

    return x_t


class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections for JAX."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

        # Validation and Auto-correction Logic adapted from PyTorch implementation
        if self.mrope_section:
            expected_sum = rotary_dim // 2
            actual_sum = sum(self.mrope_section)
            if actual_sum != expected_sum:
                print(
                    f"MRoPE section sum mismatch: expected {expected_sum}, got {actual_sum}. "
                    f"Adjusting mrope_section to match rotary_dim // 2 = {expected_sum}"
                )
                # Auto-correct by scaling the mrope_section proportionally
                if actual_sum > 0:
                    scale_factor = expected_sum / actual_sum
                    self.mrope_section = [
                        max(1, int(section * scale_factor)) for section in self.mrope_section
                    ]
                    # Ensure the sum exactly matches by adjusting the last element
                    current_sum = sum(self.mrope_section)
                    if current_sum != expected_sum:
                        self.mrope_section[-1] += expected_sum - current_sum
                else:
                    # Fallback for zero sum
                    self.mrope_section = [expected_sum // len(self.mrope_section)] * len(
                        self.mrope_section
                    )
                    # Handle remainder
                    remainder = expected_sum % len(self.mrope_section)
                    for i in range(remainder):
                        self.mrope_section[i] += 1

            # Pre-calculate split indices for jnp.split
            # mrope_section is like [16, 24, 24], split indices should be [16, 40]
            self.split_indices = np.cumsum(self.mrope_section)[:-1].tolist()

    def __call__(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Args:
            positions: [num_tokens] (Text only) or
                       [3, num_tokens] (Multimodal T/H/W positions)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        # Handle Multimodal 3D Positions
        if positions.ndim == 2 and positions.shape[0] == 3:
            return self._forward_mrope(positions, query, key)

        # Fallback to standard RoPE for 1D positions
        return super().__call__(positions, query, key)

    def _forward_mrope(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # positions: [3, num_tokens]
        num_tokens = positions.shape[-1]

        # 1. Compute Cos/Sin for all 3 dimensions
        inv_freq = jnp.asarray(self._inv_freq_np, dtype=self.dtype)

        # freqs: [3, num_tokens, rotary_dim // 2]
        freqs = jnp.einsum("cn,d->cnd", positions.astype(jnp.float32), inv_freq)

        cos_all = jnp.cos(freqs).astype(self.dtype)
        sin_all = jnp.sin(freqs).astype(self.dtype)

        if self.mrope_interleaved:
            # --- Interleaved Mode ---
            # Direct manipulation on the [3, N, D] tensor
            cos = apply_interleaved_rope(cos_all, self.mrope_section)
            sin = apply_interleaved_rope(sin_all, self.mrope_section)
        else:
            # --- Chunked Mode (Existing Logic) ---
            # 2. Split and Select based on mrope_section
            cos_splits = jnp.split(cos_all, self.split_indices, axis=-1)
            sin_splits = jnp.split(sin_all, self.split_indices, axis=-1)

            # Select specific rows for specific sections
            # section 0 uses row 0 (Time), section 1 uses row 1 (Height), section 2 uses row 2 (Width)
            final_cos_list = []
            final_sin_list = []

            for i, split_tensor in enumerate(cos_splits):
                # split_tensor shape: [3, num_tokens, section_dim]
                # We take the i-th row: [num_tokens, section_dim]
                final_cos_list.append(split_tensor[i])

            for i, split_tensor in enumerate(sin_splits):
                final_sin_list.append(split_tensor[i])

            # Concatenate back: [num_tokens, rotary_dim // 2]
            cos = jnp.concatenate(final_cos_list, axis=-1)
            sin = jnp.concatenate(final_sin_list, axis=-1)

        # 3. Apply RoPE
        # Reshape query/key to [num_tokens, num_heads, head_size]
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]

        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]

        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)

        return query, key

    @staticmethod
    def get_rope_index(
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        model_type: str,
        tokens_per_second: int | None = None,
        input_ids: np.ndarray | None = None,
        image_grid_thw: np.ndarray | None = None,
        video_grid_thw: np.ndarray | None = None,
        second_per_grid_ts: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
        """
        if model_type == "qwen3_omni_moe":
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

        # Handle video grid modification for Qwen3-VL
        if (
            model_type.startswith("qwen3_vl") or model_type.startswith("qwen3_vl_moe")
        ) and video_grid_thw is not None:
            video_grid_thw = np.repeat(video_grid_thw, video_grid_thw[:, 0], axis=0)
            video_grid_thw[:, 0] = 1

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            # [3, batch, seq_len]
            position_ids = np.ones(
                (3, input_ids.shape[0], input_ids.shape[1]),
                dtype=input_ids.dtype,
            )
            image_index, video_index = 0, 0

            for i, input_ids_row in enumerate(total_input_ids):
                image_nums, video_nums = 0, 0
                # Find vision start tokens
                vision_start_indices = np.argwhere(input_ids_row == vision_start_token_id).squeeze(
                    1
                )

                # Determine if following tokens are image or video
                if vision_start_indices.size > 0:
                    # Safety check for index bounds
                    valid_indices = vision_start_indices + 1 < len(input_ids_row)
                    vision_tokens = input_ids_row[vision_start_indices[valid_indices] + 1]
                    image_nums = np.sum(vision_tokens == image_token_id)
                    video_nums = np.sum(vision_tokens == video_token_id)

                input_tokens = input_ids_row.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        try:
                            ed_image = input_tokens.index(image_token_id, st)
                        except ValueError:
                            ed_image = len(input_tokens) + 1
                    else:
                        ed_image = len(input_tokens) + 1

                    if video_token_id in input_tokens and remain_videos > 0:
                        try:
                            ed_video = input_tokens.index(video_token_id, st)
                        except ValueError:
                            ed_video = len(input_tokens) + 1
                    else:
                        ed_video = len(input_tokens) + 1

                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index]
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t = t.item()
                    llm_grid_h = h.item() // spatial_merge_size
                    llm_grid_w = w.item() // spatial_merge_size

                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    # Text part
                    llm_pos_ids_list.append(
                        np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
                    )

                    if model_type == "qwen2_5_vl":
                        range_tensor = np.arange(llm_grid_t).reshape(-1, 1)
                        expanded_range = np.tile(range_tensor, (1, llm_grid_h * llm_grid_w))
                        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
                        t_index = time_tensor.astype(np.int64).flatten()

                    elif model_type in ("qwen2_vl", "qwen3_vl", "qwen3_vl_moe"):
                        t_index = np.tile(
                            np.arange(llm_grid_t).reshape(-1, 1), (1, llm_grid_h * llm_grid_w)
                        ).flatten()
                    else:
                        raise RuntimeError(f"Unimplemented model type: {model_type}")

                    h_index = np.tile(
                        np.arange(llm_grid_h).reshape(1, -1, 1), (llm_grid_t, 1, llm_grid_w)
                    ).flatten()

                    w_index = np.tile(
                        np.arange(llm_grid_w).reshape(1, 1, -1), (llm_grid_t, llm_grid_h, 1)
                    ).flatten()

                    llm_pos_ids_list.append(
                        np.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # Process remaining text at the end
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
                    )

                llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                position_ids[..., i, :] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids_row))

            mrope_position_deltas = np.array(mrope_position_deltas).reshape(-1, 1)
            return position_ids, mrope_position_deltas
        else:
            # Standard 1D RoPE case
            s = input_ids.shape[1]
            position_ids = np.arange(s)
            position_ids = np.tile(position_ids.reshape(1, 1, -1), (3, input_ids.shape[0], 1))

            max_position_ids = position_ids.max(axis=0).max(axis=-1, keepdims=True)
            mrope_position_deltas = max_position_ids + 1 - s
            return position_ids, mrope_position_deltas

    @staticmethod
    def get_rope_index_qwen3_omni(
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        tokens_per_second: int | None = None,
        input_ids: np.ndarray | None = None,
        image_grid_thw: np.ndarray | None = None,
        video_grid_thw: np.ndarray | None = None,
        second_per_grid_ts: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        # For qwen3-omni
        audio_token_id = kwargs["audio_token_id"]
        audio_start_token_id = kwargs["audio_start_token_id"]
        position_id_per_seconds = kwargs["position_id_per_seconds"]
        use_audio_in_video = kwargs.get("use_audio_in_video", False)
        audio_seqlens = kwargs.get("audio_seqlens")
        second_per_grids = second_per_grid_ts

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            position_ids = np.zeros(
                (3, input_ids.shape[0], input_ids.shape[1]),
                dtype=np.float32,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0

            for i, current_input_ids in enumerate(total_input_ids):
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = np.argwhere(
                    current_input_ids == vision_start_token_id
                ).squeeze(1)

                if vision_start_indices.size > 0:
                    valid_indices = vision_start_indices + 1 < len(current_input_ids)
                    vision_tokens = current_input_ids[vision_start_indices[valid_indices] + 1]
                    image_nums = np.sum(vision_tokens == image_token_id)
                    video_nums = (
                        np.sum(vision_tokens == audio_start_token_id)
                        if use_audio_in_video
                        else np.sum(vision_tokens == video_token_id)
                    )
                audio_nums = np.sum(current_input_ids == audio_start_token_id)
                input_tokens = current_input_ids.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images, remain_videos, remain_audios = (
                    image_nums,
                    video_nums,
                    audio_nums,
                )
                multimodal_nums = (
                    image_nums + audio_nums
                    if use_audio_in_video
                    else image_nums + video_nums + audio_nums
                )

                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

                    # Find vision start
                    try:
                        if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                            remain_videos > 0 or remain_images > 0
                        ):
                            ed_vision_start = input_tokens.index(vision_start_token_id, st)
                        else:
                            ed_vision_start = len(input_tokens) + 1
                    except ValueError:
                        ed_vision_start = len(input_tokens) + 1

                    # Find audio start
                    try:
                        if audio_token_id in input_tokens and remain_audios > 0:
                            ed_audio_start = input_tokens.index(audio_start_token_id, st)
                        else:
                            ed_audio_start = len(input_tokens) + 1
                    except ValueError:
                        ed_audio_start = len(input_tokens) + 1

                    min_ed = min(ed_vision_start, ed_audio_start)

                    text_len = min_ed - st
                    if text_len != 0:
                        llm_pos_ids_list.append(
                            np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
                        )
                        st_idx += text_len

                    # Audio in Video
                    if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        bos_len, eos_len = 2, 2
                    else:
                        bos_len, eos_len = 1, 1

                    llm_pos_ids_list.append(
                        np.arange(bos_len).reshape(1, -1).repeat(3, axis=0) + st_idx
                    )
                    st_idx += bos_len

                    # Audio Only
                    if min_ed == ed_audio_start:
                        audio_len = MRotaryEmbedding._get_feat_extract_output_lengths(
                            audio_seqlens[audio_idx]
                        )
                        llm_pos_ids = np.arange(audio_len).reshape(1, -1).repeat(3, axis=0) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + audio_len + eos_len)
                        audio_idx += 1
                        remain_audios -= 1

                    # Image Only
                    elif (
                        min_ed == ed_vision_start
                        and current_input_ids[ed_vision_start + 1] == image_token_id
                    ):
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (np.arange(grid_t) * 1 * position_id_per_seconds).astype(
                            np.float32
                        )

                        llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision_numpy(
                            st_idx,
                            image_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + image_len + eos_len)
                        image_idx += 1
                        remain_images -= 1

                    # Video Only
                    elif (
                        min_ed == ed_vision_start
                        and current_input_ids[ed_vision_start + 1] == video_token_id
                    ):
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            np.arange(grid_t)
                            * second_per_grids[video_idx].item()
                            * position_id_per_seconds
                        ).astype(np.float32)

                        llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision_numpy(
                            st_idx,
                            video_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                        )

                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + video_len + eos_len)
                        video_idx += 1
                        remain_videos -= 1

                    # Audio in Video (omni logic)
                    elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        audio_len = MRotaryEmbedding._get_feat_extract_output_lengths(
                            audio_seqlens[audio_idx]
                        )
                        audio_llm_pos_ids = (
                            np.arange(audio_len).reshape(1, -1).repeat(3, axis=0) + st_idx
                        )
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            np.arange(grid_t)
                            * second_per_grids[video_idx].item()
                            * position_id_per_seconds
                        ).astype(np.float32)

                        video_llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision_numpy(
                            st_idx,
                            video_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                        )

                        video_data_index, audio_data_index = 0, 0
                        # Interleave video and audio positions
                        while (
                            video_data_index < video_llm_pos_ids.shape[-1]
                            and audio_data_index < audio_llm_pos_ids.shape[-1]
                        ):
                            if (
                                video_llm_pos_ids[0][video_data_index]
                                <= audio_llm_pos_ids[0][audio_data_index]
                            ):
                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_data_index : video_data_index + 1]
                                )
                                video_data_index += 1
                            else:
                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1]
                                )
                                audio_data_index += 1

                        if video_data_index < video_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index:])
                        if audio_data_index < audio_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index:])

                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st += int(text_len + bos_len + audio_len + video_len + eos_len)

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        np.arange(eos_len).reshape(1, -1).repeat(3, axis=0) + st_idx
                    )

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
                    )

                llm_positions = np.concatenate(
                    [item.astype(np.float32) for item in llm_pos_ids_list], axis=1
                ).reshape(3, -1)

                position_ids[..., i, :] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
            mrope_position_deltas = np.array(mrope_position_deltas).reshape(-1, 1)

            return position_ids, mrope_position_deltas
        else:
            # Fallback / Simple case
            s = input_ids.shape[1]
            position_ids = np.arange(s)
            position_ids = np.tile(position_ids.reshape(1, 1, -1), (3, input_ids.shape[0], 1))
            max_position_ids = position_ids.max(axis=0).max(axis=-1, keepdims=True)
            mrope_position_deltas = max_position_ids + 1 - s

            return position_ids, mrope_position_deltas

    @staticmethod
    def _get_feat_extract_output_lengths(input_lengths):
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    @staticmethod
    def _get_llm_pos_ids_for_vision_numpy(
        st_idx, vision_idx, spatial_merge_size, t_index, grid_hs, grid_ws
    ):
        """NumPy adaptation of _get_llm_pos_ids_for_vision"""
        grid_h = grid_hs[vision_idx] // spatial_merge_size
        grid_w = grid_ws[vision_idx] // spatial_merge_size

        h_index = np.tile(np.arange(grid_h).reshape(1, -1, 1), (len(t_index), 1, grid_w)).flatten()

        w_index = np.tile(np.arange(grid_w).reshape(1, 1, -1), (len(t_index), grid_h, 1)).flatten()

        t_index = np.tile(t_index.reshape(-1, 1), (1, grid_h * grid_w)).flatten()

        llm_pos_ids = np.stack([t_index, h_index, w_index], axis=0) + st_idx
        return llm_pos_ids


# @partial(jax.jit, static_argnames=["rotary_dim", "head_size", "is_neox_style"])
def rotary_embedding_forward(
    positions: jax.Array,
    query: jax.Array,
    key: jax.Array,
    cos_sin_cache: jax.Array,
    rotary_dim: int,
    head_size: int,
    is_neox_style: bool,
) -> tuple[jax.Array, jax.Array]:
    """Rotary Position Embedding."""
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.take(positions, axis=0)
    cos, sin = jnp.split(cos_sin, 2, axis=-1)

    query_shape = query.shape
    query = query.reshape(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = apply_rotary_emb(query_rot, cos, sin, is_neox_style)
    query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.reshape(num_tokens, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = apply_rotary_emb(key_rot, cos, sin, is_neox_style)
    key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)
    return query, key


# @partial(jax.jit, static_argnames=["is_neox_style"])
def apply_rotary_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    is_neox_style: bool,
) -> jax.Array:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)
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
        stacked = jnp.stack((o1, o2), axis=-1)
        return stacked.reshape(*stacked.shape[:-2], -1)


_ROPE_DICT: dict[tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: dict[str, Any] | None = None,
    dtype: jnp.dtype | None = jnp.bfloat16,
    partial_rotary_factor: float = 1.0,
    dual_chunk_attention_config: dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None

    if dual_chunk_attention_config is not None:
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())
    else:
        dual_chunk_attention_args = None

    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dual_chunk_attention_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        if "rope_type" in rope_scaling:
            scaling_type = rope_scaling["rope_type"]
        elif "type" in rope_scaling:
            scaling_type = rope_scaling["type"]
        else:
            raise ValueError("Unknown RoPE scaling type")

        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            rotary_emb = Llama3RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            )
        elif scaling_type == "default":
            if "mrope_section" in rope_scaling:
                rotary_emb = MRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=list(rope_scaling["mrope_section"]),
                    mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
                )
            else:
                rotary_emb = RotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


def _yarn_get_mscale(scaling_factor: float) -> float:
    # Approximate magnitude scaling correction used by YaRN
    if scaling_factor <= 1:
        return 1.0
    return math.sqrt(scaling_factor)


# Inverse dim formula to find dim based on number of rotations
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
    base: int,
    max_position_embeddings: int,
) -> tuple[float, float]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case
