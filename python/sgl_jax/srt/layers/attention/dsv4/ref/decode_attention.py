"""Dense FP32 CSA decode attention reference over already gathered KV."""

import jax.numpy as jnp

_NEG_INF = jnp.finfo(jnp.float32).min


def gathered_decode_attention_ref(
    q, window, compressed, window_valid, selected_valid, attention_sink, *, softmax_scale
):
    """Return [T,H,D]; SWA and selected records share one denominator and one sink."""
    mask = jnp.concatenate((window_valid, selected_valid), axis=1)
    keys = jnp.concatenate((window, compressed), axis=1).astype(jnp.float32)
    keys = jnp.where(mask[:, :, None], keys, 0.0)
    scores = jnp.einsum(
        "thd,tkd->thk", q.astype(jnp.float32), keys, preferred_element_type=jnp.float32
    )
    scores = jnp.where(mask[:, None, :], scores * softmax_scale, _NEG_INF)
    sink = attention_sink.astype(jnp.float32)[None, :, None]
    shift = jnp.maximum(jnp.max(scores, axis=-1, keepdims=True), sink)
    probs = jnp.where(mask[:, None, :], jnp.exp(scores - shift), 0.0)
    denominator = jnp.sum(probs, axis=-1, keepdims=True) + jnp.exp(sink - shift)
    out = jnp.einsum("thk,tkd->thd", probs, keys, preferred_element_type=jnp.float32) / denominator
    return out
