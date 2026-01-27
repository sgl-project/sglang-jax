import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime
from sgl_jax.srt.utils.profiling_utils import named_scope


class NativeAttention(AttentionBackend):
    """Native Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
        mesh,
    ):
        self.num_heads = num_attn_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_attn_heads
        self.mesh = mesh
        self.kv_sharding = NamedSharding(self.mesh, P(None, "tensor", None))

    def tree_flatten(self):
        children = ()
        aux_data = {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "mesh": self.mesh,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(num_attn_heads=aux_data["num_heads"], num_kv_heads=aux_data["num_kv_heads"])

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Init the metadata for a forward pass and return it."""
        return None

    @named_scope
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        """
        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            is_causal: Whether to apply causal masking
        Returns:
            Tuple of (output tensor of shape [total_tokens, hidden_size], k, v)
        """
        # DEBUG: Check input Q/K/V for NaN (Layer 0 only)
        if layer.layer_id == 0:
            jax.debug.print(
                "NativeAttn0 input: q_nan={q_nan}, k_nan={k_nan}, v_nan={v_nan}, "
                "cache_loc_len={cache_loc_len}, out_cache_loc_len={out_cache_loc_len}",
                q_nan=jnp.any(jnp.isnan(q)),
                k_nan=jnp.any(jnp.isnan(k)),
                v_nan=jnp.any(jnp.isnan(v)),
                cache_loc_len=forward_batch.cache_loc.shape[0],
                out_cache_loc_len=forward_batch.out_cache_loc.shape[0],
            )

        # TODO(pc) support tree based native attention backend
        k_buffer, v_buffer, kv_fused = self._get_and_update_kv_cache(
            k, v, forward_batch, token_to_kv_pool, self.kv_sharding, layer.layer_id
        )

        # DEBUG: Check KV buffer after cache update (Layer 0 only)
        if layer.layer_id == 0:
            jax.debug.print(
                "NativeAttn0 after_cache: k_buffer_nan={k_nan}, v_buffer_nan={v_nan}, "
                "k_buffer_shape={k_shape}",
                k_nan=jnp.any(jnp.isnan(k_buffer)),
                v_nan=jnp.any(jnp.isnan(v_buffer)),
                k_shape=k_buffer.shape,
            )

        scale = 1.0 / jnp.sqrt(layer.head_dim) if layer.scaling is None else layer.scaling

        is_causal = True
        if (
            forward_batch.forward_mode == ForwardMode.DECODE
            or layer.attn_type == AttentionType.ENCODER_ONLY
        ):
            is_causal = False

        # Get xai_temperature_len from the layer if it exists and pass it down.
        xai_temp_len = getattr(layer, "xai_temperature_len", None)

        attn_output = forward_attention(
            q,
            k_buffer,
            v_buffer,
            forward_batch.seq_lens,
            forward_batch.cache_loc,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            layer.q_head_num,
            layer.kv_head_num,
            scale,
            is_causal,
            forward_batch.forward_mode,
            self.kv_sharding,
            xai_temperature_len=xai_temp_len,
            layer_id=layer.layer_id,  # Pass layer_id for debugging
        )

        # DEBUG: Check attention output (Layer 0 only)
        if layer.layer_id == 0:
            jax.debug.print(
                "NativeAttn0 output: attn_nan={attn_nan}, attn_inf={attn_inf}",
                attn_nan=jnp.any(jnp.isnan(attn_output)),
                attn_inf=jnp.any(jnp.isinf(attn_output)),
            )

        # Return full fused KV buffer for this layer so that caller can persist it outside JIT
        return attn_output, kv_fused

    def _get_and_update_kv_cache(
        self,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        kv_sharding: jax.NamedSharding,
        layer_id: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Get the kv cache from the forward batch.
        """
        # DEBUG & PROTECTION: Check for NaN before writing to cache (Layer 0 only)
        if layer_id == 0:
            k_has_nan = jnp.any(jnp.isnan(k))
            v_has_nan = jnp.any(jnp.isnan(v))
            jax.debug.print(
                "KV_CACHE_WRITE layer0: k_nan={k_nan}, v_nan={v_nan}, "
                "out_cache_loc={out_cache_loc}",
                k_nan=k_has_nan,
                v_nan=v_has_nan,
                out_cache_loc=forward_batch.out_cache_loc,
            )

        # Replace NaN with 0 to prevent cache pollution
        k = jnp.where(jnp.isnan(k), 0.0, k)
        v = jnp.where(jnp.isnan(v), 0.0, v)

        if is_tpu_runtime():
            if forward_batch.forward_mode.is_extend():
                token_to_kv_pool.set_kv_buffer(
                    layer_id, forward_batch.out_cache_loc, k, v, is_decode=False
                )
            else:
                token_to_kv_pool.set_kv_buffer(
                    layer_id, forward_batch.out_cache_loc, k, v, is_decode=True
                )
            # Use fused layer directly from pool; derive K/V views without extra merge
            fused_layer = token_to_kv_pool.get_fused_kv_buffer(layer_id)
            k = fused_layer.at[:, ::2, :].get(out_sharding=kv_sharding)
            v = fused_layer.at[:, 1::2, :].get(out_sharding=kv_sharding)
            fused_return = fused_layer
        else:
            updated_layer = token_to_kv_pool.set_kv_buffer_legacy(
                layer_id, forward_batch.out_cache_loc, k, v
            )
            # Functional style: treat updated_layer as authoritative fused buffer for this layer in this step
            # Derive K/V views for attention computation from fused buffer directly
            k = updated_layer.at[:, ::2, :].get(out_sharding=kv_sharding)
            v = updated_layer.at[:, 1::2, :].get(out_sharding=kv_sharding)
            # Return fused buffer directly for persistence outside JIT
            fused_return = updated_layer

        # DEBUG: Check for NaN after reading from cache (Layer 0 only)
        if layer_id == 0:
            jax.debug.print(
                "KV_CACHE_READ layer0: k_nan={k_nan}, v_nan={v_nan}",
                k_nan=jnp.any(jnp.isnan(k)),
                v_nan=jnp.any(jnp.isnan(v)),
            )

        # PROTECTION: Replace NaN in cached K/V with 0 to prevent attention explosion
        k = jnp.where(jnp.isnan(k), 0.0, k)
        v = jnp.where(jnp.isnan(v), 0.0, v)

        return k, v, fused_return

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        # native attention backend do not care the max running requests
        return 4096


# @partial(jax.jit, static_argnames=["num_heads", "num_kv_heads", "is_causal", "mode"])
def forward_attention(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    seq_lengths: jax.Array,
    loc: jax.Array,
    extend_prefix_lens: jax.Array,
    extend_seq_lens: jax.Array,
    num_heads,
    num_kv_heads,
    scale=None,
    is_causal=True,
    mode=ForwardMode.DECODE,
    kv_sharding=None,
    xai_temperature_len: float | None = None,
    layer_id: int = -1,  # Added for debugging
):
    """
    Forward pass using native JAX implementation with block-diagonal attention.
    This avoids padding while maintaining efficient matrix operations.

    Args:
        q: input token in decode mode, shape(batch_size, hidden_size), each batch has one token
        k_cache: prefix cache of key, shape(seq_len, hidden_size)
        v_cache: prefix cache of value, shape(seq_len, hidden_size)
        seq_lengths: sequence lengths of each batch
        loc: location of the key/value cache
        extend_prefix_lens: prefix lengths of each batch in extend mode
        extend_seq_lens: sequence lengths of each batch in extend mode
        num_heads: number of query heads
        num_kv_heads: number of key/value heads
        scale: scale for the attention weights
        seq_mask: boolean mask of shape [batch_size, total_prefix_len]
        xai_temperature_len: length of the xai temperature

    Returns:
        Output tensor of shape[batch_size, hidden_size]
    """

    cache_size = k_cache.shape[0]
    safe_loc = jnp.where(loc > 0, loc, cache_size)

    # DEBUG: Check loc for invalid values (Layer 0 only)
    if layer_id == 0:
        jax.debug.print(
            "forward_attn0: loc_min={loc_min}, loc_max={loc_max}, loc_len={loc_len}, "
            "num_zeros={num_zeros}, cache_size={cache_size}",
            loc_min=jnp.min(loc),
            loc_max=jnp.max(loc),
            loc_len=loc.shape[0],
            num_zeros=jnp.sum(loc == 0),
            cache_size=cache_size,
        )

    k_cache = k_cache.at[safe_loc].get(out_sharding=kv_sharding, mode="fill", fill_value=0)
    v_cache = v_cache.at[safe_loc].get(out_sharding=kv_sharding, mode="fill", fill_value=0)

    # DEBUG: Check indexed KV cache for NaN (Layer 0 only)
    if layer_id == 0:
        jax.debug.print(
            "forward_attn0 after_index: k_nan={k_nan}, v_nan={v_nan}",
            k_nan=jnp.any(jnp.isnan(k_cache)),
            v_nan=jnp.any(jnp.isnan(v_cache)),
        )

    # Handle both 2D and 3D input formats for q
    if len(q.shape) == 2:
        # Traditional format: [num_tokens, hidden_size]
        num_tokens, hidden_size = q.shape
        head_dim = hidden_size // num_heads
        q_heads = q.reshape(num_tokens, num_heads, head_dim)
    else:
        # Already in multi-head format: [num_tokens, num_heads, head_dim]
        num_tokens, num_heads_input, head_dim = q.shape
        assert num_heads_input == num_heads, f"Expected {num_heads} heads, got {num_heads_input}"
        hidden_size = num_heads * head_dim  # Calculate hidden_size for proper reshaping
        q_heads = q

    # KV cache from get_kv_buffer is already in multi-head format: [cache_size, num_kv_heads, head_dim]
    k_heads = k_cache
    v_heads = v_cache

    # Transpose for efficient matrix operations
    # q: shape of (num_heads, num_tokens, head_dim)
    # k, v: shape of (total_prefix_len, num_heads, head_dim)
    if num_kv_heads != num_heads:
        # For GQA attention, we need to copy k and v heads to match the number of query heads
        num_copies = num_heads // num_kv_heads
        # Use repeat to copy k and v heads
        # [total_prefix_len, num_kv_heads, head_dim] -> [total_prefix_len, num_heads, head_dim]
        k_heads = jnp.repeat(k_heads, num_copies, axis=1, out_sharding=kv_sharding)
        v_heads = jnp.repeat(v_heads, num_copies, axis=1, out_sharding=kv_sharding)

    # Transpose for matmul: [num_heads, num_tokens, head_dim]
    q_t = jnp.transpose(q_heads, (1, 0, 2))
    k_t = jnp.transpose(k_heads, (1, 0, 2))
    v_t = jnp.transpose(v_heads, (1, 0, 2))

    # DEBUG: Check Q/K value ranges before computing attention logits (Layer 0 only)
    if layer_id == 0:
        jax.debug.print(
            "forward_attn0 QK_range: q_min={q_min}, q_max={q_max}, k_min={k_min}, k_max={k_max}",
            q_min=jnp.min(q_t),
            q_max=jnp.max(q_t),
            k_min=jnp.min(k_t),
            k_max=jnp.max(k_t),
        )

    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)
    attn_logits = jnp.einsum("hqd,hkd->hqk", q_t, k_t) * scale

    # DEBUG: Check raw logits before masking (Layer 0 only)
    if layer_id == 0:
        jax.debug.print(
            "forward_attn0 raw_logits: min={logits_min}, max={logits_max}",
            logits_min=jnp.min(attn_logits),
            logits_max=jnp.max(attn_logits),
        )

    neg_inf = jnp.asarray(jnp.finfo(attn_logits.dtype).min, attn_logits.dtype)
    is_valid = loc > 0
    attn_logits = jnp.where(is_valid[jnp.newaxis, jnp.newaxis, :], attn_logits, neg_inf)

    # ** NEW: Apply XAI temperature scaling if specified **
    if xai_temperature_len is not None and xai_temperature_len > 0:
        query_len = q_heads.shape[0]

        # Determine the sequence position of each query token
        if mode == ForwardMode.EXTEND:
            q_starts = jnp.cumsum(extend_seq_lens) - extend_seq_lens
            q_batch_indicators = jnp.zeros(query_len, dtype=jnp.int32).at[q_starts].set(1)
            q_batch_ids = jnp.cumsum(q_batch_indicators) - 1
            q_relative_pos = jnp.arange(query_len, dtype=jnp.int32) - q_starts[q_batch_ids]
            q_positions = extend_prefix_lens[q_batch_ids] + q_relative_pos
        else:  # mode == ForwardMode.DECODE
            q_positions = seq_lengths

        # Calculate and apply the scaling factor
        xai_scale = 1.0 / jnp.log2(float(xai_temperature_len))
        log_pos = jnp.log2(jnp.maximum(q_positions.astype(jnp.float32), 1.0))
        temp_factor = log_pos * xai_scale
        regulator = jnp.where(q_positions > xai_temperature_len, temp_factor, 1.0)

        # Broadcast regulator from [num_tokens] to [1, num_tokens, 1] to scale weights
        attn_logits = attn_logits * regulator[None, :, None]

    # Apply appropriate masking
    if mode == ForwardMode.EXTEND:
        attn_logits = _apply_extend_mask(
            attn_logits, seq_lengths, extend_prefix_lens, extend_seq_lens, is_causal
        )
    else:
        attn_logits = _apply_decode_mask(attn_logits, seq_lengths)

    # DEBUG: Check attention logits after masking (Layer 0 only)
    if layer_id == 0:
        # Check if any row has ALL -inf values (would cause NaN in softmax)
        row_all_neg_inf = jnp.all(attn_logits == jnp.finfo(attn_logits.dtype).min, axis=-1)
        # Check logits range (excluding -inf values)
        finite_mask = attn_logits > jnp.finfo(attn_logits.dtype).min
        finite_logits = jnp.where(finite_mask, attn_logits, 0.0)
        logits_min = jnp.min(jnp.where(finite_mask, attn_logits, jnp.inf))
        logits_max = jnp.max(finite_logits)
        jax.debug.print(
            "forward_attn0 after_mask: logits_nan={logits_nan}, logits_inf={logits_inf}, "
            "any_row_all_masked={any_row_all_masked}, seq_lens={seq_lens}, "
            "logits_min={logits_min}, logits_max={logits_max}",
            logits_nan=jnp.any(jnp.isnan(attn_logits)),
            logits_inf=jnp.any(jnp.isinf(attn_logits)),
            any_row_all_masked=jnp.any(row_all_neg_inf),
            seq_lens=seq_lengths,
            logits_min=logits_min,
            logits_max=logits_max,
        )

    # Softmax
    attn_logits = attn_logits - jnp.max(attn_logits, axis=-1, keepdims=True)
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)

    # DEBUG: Check attention weights after softmax (Layer 0 only)
    if layer_id == 0:
        jax.debug.print(
            "forward_attn0 after_softmax: weights_nan={weights_nan}, weights_inf={weights_inf}",
            weights_nan=jnp.any(jnp.isnan(attn_weights)),
            weights_inf=jnp.any(jnp.isinf(attn_weights)),
        )

    attn_output = jnp.matmul(attn_weights, v_t)
    attn_output = jnp.transpose(attn_output, (1, 0, 2))
    return attn_output.reshape(num_tokens, hidden_size)


def _apply_extend_mask(
    attn_weights: jax.Array,
    seq_lengths: jax.Array,
    extend_prefix_lens: jax.Array,
    extend_seq_lens: jax.Array,
    is_causal: bool = True,
):
    """
    Applies a block-diagonal and optionally a causal mask in a unified,
    efficient way, correctly handling padding.
    """
    _, query_len, key_len = attn_weights.shape

    # --- Create validity masks to handle padding ---
    q_valid_mask = jnp.arange(query_len) < jnp.sum(extend_seq_lens)
    k_valid_mask = jnp.arange(key_len) < jnp.sum(seq_lengths)

    # --- 1. Generate Batch IDs (Optimized) ---
    q_starts = jnp.cumsum(extend_seq_lens, dtype=jnp.int32) - extend_seq_lens
    q_batch_indicators = jnp.zeros(query_len, dtype=jnp.int32).at[q_starts].set(1)
    q_batch_ids = jnp.cumsum(q_batch_indicators, dtype=jnp.int32) - 1

    full_seq_lens = seq_lengths
    k_starts = jnp.cumsum(full_seq_lens, dtype=jnp.int32) - full_seq_lens
    k_batch_indicators = jnp.zeros(key_len, dtype=jnp.int32).at[k_starts].set(1)
    k_batch_ids = jnp.cumsum(k_batch_indicators, dtype=jnp.int32) - 1

    # --- 2. Create block-diagonal mask ---
    final_mask = q_batch_ids[:, None] == k_batch_ids[None, :]

    # --- 3. Optionally add causal mask ---
    if is_causal:
        q_starts_per_pos = q_starts[q_batch_ids]
        q_relative_positions = jnp.arange(query_len, dtype=jnp.int32) - q_starts_per_pos
        prefix_lens_per_pos = extend_prefix_lens[q_batch_ids]
        q_actual_positions = prefix_lens_per_pos + q_relative_positions

        k_starts_per_pos = k_starts[k_batch_ids]
        k_relative_positions = jnp.arange(key_len, dtype=jnp.int32) - k_starts_per_pos

        causal_mask = q_actual_positions[:, None] >= k_relative_positions[None, :]
        final_mask = final_mask & causal_mask

    # --- 4. Apply the final combined mask ---
    # Combine with validity masks to handle padding
    final_mask = final_mask & q_valid_mask[:, None] & k_valid_mask[None, :]

    mask_value = jnp.finfo(attn_weights.dtype).min
    final_mask = final_mask[None, :, :]
    return jnp.where(final_mask, attn_weights, mask_value)


def _apply_decode_mask(attn_weights: jax.Array, seq_lengths: jax.Array):
    """Create a sequence mask that ensures tokens only attend within their sequence."""
    _, query_len, key_len = attn_weights.shape
    num_seqs = len(seq_lengths)

    def create_decode_sequence_mask():
        total_prefix_len = key_len
        seq_starts = jnp.cumsum(jnp.concatenate([jnp.array([0]), seq_lengths[:-1]]))
        seq_ends = seq_starts + seq_lengths
        all_positions = jnp.arange(total_prefix_len)
        seq_mask = (all_positions[None, :] >= seq_starts[:, None]) & (
            all_positions[None, :] < seq_ends[:, None]
        )
        return seq_mask

    per_sequence_mask = create_decode_sequence_mask()
    final_mask = jnp.zeros((query_len, key_len), dtype=jnp.bool_)
    final_mask = final_mask.at[:num_seqs, :].set(per_sequence_mask)

    mask_value = jnp.finfo(attn_weights.dtype).min
    final_mask = final_mask[None, :, :]
    return jnp.where(final_mask, attn_weights, mask_value)
