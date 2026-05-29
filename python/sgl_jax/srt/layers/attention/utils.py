from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime


def get_attention_impl():
    """Returns the attention implementation based on the current device.

    Returns:
        The FlashAttention implementation if on TPU, otherwise NativeAttention.
    """
    if is_tpu_runtime():
        return FlashAttention
    return NativeAttention
