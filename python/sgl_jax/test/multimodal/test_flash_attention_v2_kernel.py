"""TPU v7x full-output checks against independent FP32 attention."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from sgl_jax.srt.multimodal.kernels.flash_attention_v2 import flash_attention


def reference(
    q, k, v, lengths, *, causal, window_size=(-1, -1), logits_soft_cap=0.0, attention_sink=None
):
    outputs = []
    start = 0
    for length in lengths:
        stop = start + length
        qs, ks, vs = (x[start:stop].astype(jnp.float32) for x in (q, k, v))
        repeats = q.shape[1] // k.shape[1]
        ks, vs = (jnp.repeat(x, repeats, axis=1) for x in (ks, vs))
        scores = jnp.einsum("qhd,khd->hqk", qs, ks, precision=jax.lax.Precision.HIGHEST)
        scores *= q.shape[-1] ** -0.5
        if logits_soft_cap:
            scores = logits_soft_cap * jnp.tanh(scores / logits_soft_cap)
        qi, ki = jnp.arange(length)[:, None], jnp.arange(length)[None, :]
        valid = jnp.ones((length, length), dtype=jnp.bool_)
        if causal:
            valid &= ki <= qi
        if window_size[0] >= 0:
            valid &= ki >= qi - window_size[0]
        if window_size[1] >= 0:
            valid &= ki <= qi + window_size[1]
        scores = jnp.where(valid, scores, -jnp.inf)
        if attention_sink is not None:
            sink = jnp.full((*scores.shape[:-1], 1), attention_sink, jnp.float32)
            probabilities = jax.nn.softmax(jnp.concatenate((scores, sink), axis=-1), axis=-1)[
                ..., :-1
            ]
        else:
            probabilities = jax.nn.softmax(scores, axis=-1)
        outputs.append(
            jnp.einsum("hqk,khd->qhd", probabilities, vs, precision=jax.lax.Precision.HIGHEST)
        )
        start = stop
    return jnp.concatenate(outputs)


@pytest.mark.skipif(
    not any(d.platform == "tpu" and d.device_kind == "TPU7x" for d in jax.devices()),
    reason="Flash Attention v2 requires TPU v7x",
)
@pytest.mark.parametrize(
    "dim,heads,kv_heads,lengths,mode,options",
    [
        (72, 16, 16, (63, 257, 1728), "normal", {"causal": False}),
        (256, 32, 32, (2048,), "normal", {"causal": True}),
        (256, 32, 32, (2048,), "forced", {"causal": True}),
        (256, 32, 32, (2048,), "mixed", {"causal": True}),
        (
            128,
            16,
            4,
            (63, 1985),
            "normal",
            {
                "causal": True,
                "window_size": (127, 63),
                "logits_soft_cap": 30.0,
                "attention_sink": 0.3,
            },
        ),
    ],
)
def test_full_output(dim, heads, kv_heads, lengths, mode, options):
    keys = jax.random.split(jax.random.key(42), 3)
    tokens = sum(lengths)
    q = jax.random.normal(keys[0], (tokens, heads, dim), jnp.bfloat16)
    k = jax.random.normal(keys[1], (tokens, kv_heads, dim), jnp.bfloat16)
    v = jax.random.normal(keys[2], (tokens, kv_heads, dim), jnp.bfloat16)
    boundaries = np.array([0, *np.cumsum(lengths)], np.int32)
    if dim == 72:
        boundaries = np.pad(boundaries, (0, 513 - len(boundaries)), mode="edge")
    cu = jnp.asarray(boundaries)
    force = jnp.arange(heads) % 2 if mode == "mixed" else mode == "forced"
    actual = jax.jit(
        lambda q, k, v, cu: flash_attention(q, k, v, cu, cu, force_recovery=force, **options)
    )(q, k, v, cu)
    expected = reference(q, k, v, lengths, **options)
    np.testing.assert_allclose(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=0.01, rtol=0.01
    )


@pytest.mark.parametrize("kind", ["cpu", "TPU v6e", "TPU7x"])
def test_device_target(kind):
    device = AbstractDevice(kind, num_cores=1, platform="cpu" if kind == "cpu" else "tpu")
    qkv = jax.ShapeDtypeStruct((0, 16, 72), jnp.bfloat16)
    cu = jax.ShapeDtypeStruct((2,), jnp.int32)
    with use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        if kind == "TPU7x":
            assert jax.eval_shape(flash_attention, qkv, qkv, qkv, cu, cu).shape == qkv.shape
        else:
            with pytest.raises(ValueError, match="requires a TPU v7x target"):
                jax.eval_shape(flash_attention, qkv, qkv, qkv, cu, cu)
