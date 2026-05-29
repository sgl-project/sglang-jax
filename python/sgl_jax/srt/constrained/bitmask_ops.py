"""JAX-based bitmask operations for vocabulary masking on TPU."""

import jax
import jax.numpy as jnp
import numpy as np
from llguidance import LLInterpreter


def allocate_token_bitmask(batch_size: int, vocab_size: int) -> np.ndarray:
    """Allocate a token bitmask array.

    Args:
        batch_size: Batch size
        vocab_size: Vocabulary size

    Returns:
        Numpy array of shape [batch_size, vocab_size // 32] with dtype int32
    """
    num_int32_per_vocab = (vocab_size + 31) // 32
    return np.zeros((batch_size, num_int32_per_vocab), dtype=np.int32)


def fill_token_bitmask(
    matcher: LLInterpreter,
    vocab_mask: np.ndarray,
    batch_idx: int,
):
    """Fill the bitmask for a specific batch index using llguidance matcher.

    Args:
        matcher: LLMatcher or LLInterpreter instance
        vocab_mask: Bitmask array of shape [batch_size, vocab_size // 32], dtype=int32
        batch_idx: Index in the batch to fill
    """
    assert vocab_mask.dtype == np.int32, "Mask must be int32"
    assert vocab_mask.ndim == 2, "Mask must be 2D"
    v = vocab_mask[batch_idx, :]
    matcher.unsafe_compute_mask_ptr(
        v.ctypes.data,
        v.nbytes,
    )


@jax.jit
def apply_token_bitmask(
    logits: jax.Array,
    vocab_mask: jax.Array,
) -> jax.Array:
    """Apply token bitmask to logits.

    Sets logits to -inf where the bitmask bit is 0.

    Args:
        logits: Logits array of shape [batch_size, vocab_size]
        vocab_mask: Packed bitmask array of shape [batch_size, vocab_size // 32]

    Returns:
        Masked logits array of shape [batch_size, vocab_size]
    """
    if vocab_mask is None:
        return logits

    # Unpack the bitmask from int32 to bool (full length = num_int32 * 32)
    unpacked_mask_full = unpack_bitmask(vocab_mask)  # [Bmask, num_int32*32]
    vocab_size = logits.shape[-1]
    mask_len = unpacked_mask_full.shape[-1]

    # Match vocab dimension statically: pad with False or crop as needed
    if mask_len < vocab_size:
        pad = vocab_size - mask_len
        unpacked_mask = jnp.pad(
            unpacked_mask_full,
            ((0, 0), (0, pad)),
            mode="constant",
            constant_values=False,
        )
    elif mask_len > vocab_size:
        unpacked_mask = unpacked_mask_full[:, :vocab_size]
    else:
        unpacked_mask = unpacked_mask_full

    # Apply mask: set logits to -inf where mask is False (broadcast batch if needed)
    masked_logits = jnp.where(unpacked_mask, logits, -jnp.inf)
    return masked_logits


def unpack_bitmask(vocab_mask: jax.Array) -> jax.Array:
    """Unpack int32 bitmask to boolean array (no dynamic slicing).

    Args:
        vocab_mask: Packed bitmask [batch_size, num_int32]

    Returns:
        Boolean mask [batch_size, num_int32 * 32]
    """
    # For each int32, extract 32 bits
    bit_indices = jnp.arange(32)[None, :]  # [1, 32]

    def unpack_batch_item(mask_row):
        # mask_row: [num_int32]
        bits = jnp.bitwise_and(mask_row[:, None], 1 << bit_indices) != 0  # [num_int32, 32]
        return bits.reshape(-1)  # [num_int32 * 32]

    # Apply to all batch items
    unpacked = jax.vmap(unpack_batch_item)(vocab_mask)  # [batch, num_int32*32]
    return unpacked
