import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
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
        return cls(
            num_attn_heads=aux_data["num_heads"],
            num_kv_heads=aux_data["num_kv_heads"],
            mesh=aux_data["mesh"],
        )

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
        **kwargs,
    ):
        """
        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            is_causal: Whether to apply causal masking
        Returns:
            Tuple of (output tensor of shape [total_tokens, hidden_size], kv_fused 5D)
        """
        # TODO(pc) support tree based native attention backend
        k_buffer, v_buffer, kv_fused = self._get_and_update_kv_cache(
            k, v, forward_batch, token_to_kv_pool, layer.layer_id
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

        # Extract attention sink bias (e.g. MiMo-V2-Flash SWA layers)
        attention_sink = kwargs.get("attention_sink")
        if attention_sink is not None and hasattr(attention_sink, "value"):
            attention_sink = attention_sink.value

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
            mesh=self.mesh,
            xai_temperature_len=xai_temp_len,
            attention_sink=attention_sink,
            sliding_window_size=layer.sliding_window_size,
        )

        # Return full fused KV buffer for this layer so that caller can persist it outside JIT
        return attn_output, kv_fused

    def _get_and_update_kv_cache(
        self,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        layer_id: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Update KV cache and return (k_3d, v_3d, fused_5d).

        The 5D fused buffer is persisted outside JIT. The 3D k/v views are
        used by forward_attention for the actual attention computation.
        """
        if is_tpu_runtime():
            if forward_batch.forward_mode.is_extend():
                token_to_kv_pool.set_kv_buffer(
                    layer_id, forward_batch.out_cache_loc, k, v, is_decode=False
                )
            else:
                token_to_kv_pool.set_kv_buffer(
                    layer_id, forward_batch.out_cache_loc, k, v, is_decode=True
                )
            fused_5d = token_to_kv_pool.get_fused_kv_buffer(layer_id)
        else:
            fused_5d = token_to_kv_pool.set_kv_buffer_legacy(
                layer_id, forward_batch.out_cache_loc, k, v
            )

        # Flatten 5D -> 3D: [pages, page_size, heads_x2_per_pack, pack, hdim] -> [tokens, heads_x2, hdim]
        num_pages, page_size, heads_x2_per_pack, packing, head_dim = fused_5d.shape
        total_tokens = num_pages * page_size
        fused_3d = jax.lax.reshape(
            fused_5d,
            (total_tokens, heads_x2_per_pack * packing, head_dim),
            out_sharding=P(None, "tensor", None),
        )

        # Split interleaved [K0, V0, K1, V1, ...] into separate K and V
        k_3d = fused_3d.at[:, ::2, :].get(out_sharding=self.kv_sharding)
        v_3d = fused_3d.at[:, 1::2, :].get(out_sharding=self.kv_sharding)

        return k_3d, v_3d, fused_5d

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
    mesh: Mesh | None = None,
    xai_temperature_len: float | None = None,
    attention_sink: jax.Array | None = None,
    sliding_window_size: int | None = None,
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
        xai_temperature_len: length of the xai temperature
        attention_sink: per-head bias for phantom attention sink token
        sliding_window_size: sliding window size for attention

    Returns:
        Output tensor of shape[batch_size, hidden_size]
    """

    cache_size = k_cache.shape[0]
    safe_loc = jnp.where(loc > 0, loc, cache_size)
    k_cache = k_cache.at[safe_loc].get(out_sharding=kv_sharding, mode="fill", fill_value=0)
    v_cache = v_cache.at[safe_loc].get(out_sharding=kv_sharding, mode="fill", fill_value=0)

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

    # Pad Q to match K cache head_dim if KV pool was 128-aligned (e.g. 192 -> 256)
    k_cache_head_dim = k_heads.shape[-1]
    if k_cache_head_dim != head_dim:
        pad_size = k_cache_head_dim - head_dim
        q_heads = jnp.pad(q_heads, ((0, 0), (0, 0), (0, pad_size)))
        head_dim = k_cache_head_dim

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

    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)
    # (Q, H, K) layout: avoids broadcasting K-axis sharding into the Q-axis position
    # under Explicit-axis mesh + DP-sharded q_heads.
    attn_logits = jnp.einsum("qhd,khd->qhk", q_heads, k_heads) * scale
    neg_inf = jnp.asarray(jnp.finfo(attn_logits.dtype).min, attn_logits.dtype)
    # Reshard `loc` to replicated so the K-axis broadcast doesn't re-introduce
    # the `data` mesh axis (Q-axis already carries `data` under DP).
    is_valid = jax.sharding.reshard(loc, NamedSharding(mesh, P())) > 0
    attn_logits = jnp.where(is_valid[jnp.newaxis, jnp.newaxis, :], attn_logits, neg_inf)

    # ** Apply XAI temperature scaling if specified **
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

        # Broadcast regulator from [num_tokens] to [num_tokens, 1, 1] to scale weights
        attn_logits = attn_logits * regulator[:, None, None]

    # Apply appropriate masking
    if mode == ForwardMode.EXTEND:
        attn_logits = _apply_extend_mask(
            attn_logits,
            seq_lengths,
            extend_prefix_lens,
            extend_seq_lens,
            is_causal,
            sliding_window_size,
            mesh=mesh,
        )
    else:
        attn_logits = _apply_decode_mask(attn_logits, seq_lengths, sliding_window_size, mesh=mesh)

    # Softmax (with optional attention sink)
    max_logit = jnp.max(attn_logits, axis=-1, keepdims=True)
    attn_logits = attn_logits - max_logit
    exp_logits = jnp.exp(attn_logits)
    sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)

    if attention_sink is not None:
        # attention_sink: [num_heads] — acts as a phantom token in the softmax denominator.
        # Broadcast sink from [num_heads] to [1, num_heads, 1] to match (Q, H, K) layout
        sink_term = jnp.exp(attention_sink[None, :, None] - max_logit)
        sum_exp = sum_exp + sink_term

    attn_weights = exp_logits / sum_exp

    # attn_output: [num_tokens, num_heads, v_head_dim]
    attn_output = jnp.einsum("qhk,khd->qhd", attn_weights, v_heads)

    # Use v_head_dim from V (may differ from head_dim for split K/V models)
    v_head_dim = v_heads.shape[-1]
    return attn_output.reshape(num_tokens, num_heads * v_head_dim)


def _apply_extend_mask(
    attn_weights: jax.Array,
    seq_lengths: jax.Array,
    extend_prefix_lens: jax.Array,
    extend_seq_lens: jax.Array,
    is_causal: bool = True,
    sliding_window_size: int | None = None,
    mesh: Mesh | None = None,
):
    """
    Applies a block-diagonal and optionally a causal/SWA mask in a unified,
    efficient way, correctly handling padding.
    """
    query_len, _, key_len = attn_weights.shape

    # Reshard inputs to replicated: under DP these arrive as P("data"),
    # but we use them as scatter indices into [query_len]/[key_len] arrays
    # whose mesh has no `data` axis, which trips a jax 0.8.1 .at[].set() check.
    extend_seq_lens = jax.sharding.reshard(extend_seq_lens, NamedSharding(mesh, P()))
    seq_lengths = jax.sharding.reshard(seq_lengths, NamedSharding(mesh, P()))
    extend_prefix_lens = jax.sharding.reshard(extend_prefix_lens, NamedSharding(mesh, P()))

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

    # --- 3. Optionally add causal/SWA mask ---
    if is_causal or sliding_window_size is not None:
        q_starts_per_pos = q_starts[q_batch_ids]
        q_relative_positions = jnp.arange(query_len, dtype=jnp.int32) - q_starts_per_pos
        prefix_lens_per_pos = extend_prefix_lens[q_batch_ids]
        q_actual_positions = prefix_lens_per_pos + q_relative_positions

        k_starts_per_pos = k_starts[k_batch_ids]
        k_relative_positions = jnp.arange(key_len, dtype=jnp.int32) - k_starts_per_pos

        if is_causal:
            causal_mask = q_actual_positions[:, None] >= k_relative_positions[None, :]
            final_mask = final_mask & causal_mask

        if sliding_window_size is not None:
            swa_mask = (
                q_actual_positions[:, None] - k_relative_positions[None, :] < sliding_window_size
            )
            final_mask = final_mask & swa_mask

    # --- 4. Apply the final combined mask ---
    # Combine with validity masks to handle padding
    final_mask = final_mask & q_valid_mask[:, None] & k_valid_mask[None, :]

    mask_value = jnp.finfo(attn_weights.dtype).min
    # Broadcast from [Q, K] to [Q, 1, K] to match attn_weights [Q, H, K]
    final_mask = final_mask[:, None, :]
    return jnp.where(final_mask, attn_weights, mask_value)


def _apply_decode_mask(
    attn_weights: jax.Array,
    seq_lengths: jax.Array,
    sliding_window_size: int | None = None,
    mesh: Mesh | None = None,
):
    """Create a sequence mask that ensures tokens only attend within their sequence and window."""
    query_len, _, key_len = attn_weights.shape
    num_seqs = len(seq_lengths)

    # Reshard to replicated: avoids jax 0.8.1 .at[].set() check failure when
    # seq_lengths is P("data") under DP and used as scatter indices.
    seq_lengths = jax.sharding.reshard(seq_lengths, NamedSharding(mesh, P()))

    def create_decode_sequence_mask():
        total_prefix_len = key_len
        seq_starts = jnp.cumsum(jnp.concatenate([jnp.array([0]), seq_lengths[:-1]]))
        seq_ends = seq_starts + seq_lengths
        all_positions = jnp.arange(total_prefix_len)
        seq_mask = (all_positions[None, :] >= seq_starts[:, None]) & (
            all_positions[None, :] < seq_ends[:, None]
        )

        if sliding_window_size is not None:
            swa_mask = all_positions[None, :] >= (seq_ends[:, None] - sliding_window_size)
            seq_mask = seq_mask & swa_mask

        return seq_mask

    per_sequence_mask = create_decode_sequence_mask()
    final_mask = jnp.zeros((query_len, key_len), dtype=jnp.bool_)
    final_mask = final_mask.at[:num_seqs, :].set(per_sequence_mask)

    mask_value = jnp.finfo(attn_weights.dtype).min
    # Broadcast from [Q, K] to [Q, 1, K] to match attn_weights [Q, H, K]
    final_mask = final_mask[:, None, :]
    return jnp.where(final_mask, attn_weights, mask_value)
