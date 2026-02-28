import unittest

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds, flash_attention


@jax.jit
def jit_flash_attention(q, k, v):
    q_len = q.shape[2]
    kv_len = k.shape[2]
    align_q_len = align_to(q_len, 128)
    align_kv_len = align_to(kv_len, 128)
    seg_q = None
    seg_kv = None
    segment_ids = None
    if q_len != align_q_len:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, align_q_len - q_len), (0, 0)))
        seg_q = jnp.concatenate(
            [jnp.ones((q.shape[0], q_len)), jnp.zeros((q.shape[0], align_q_len - q_len))], axis=1
        )
    if kv_len != align_kv_len:
        k = jnp.pad(k, ((0, 0), (0, 0), (0, align_kv_len - kv_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, align_kv_len - kv_len), (0, 0)))
        seg_kv = jnp.concatenate(
            [jnp.ones((k.shape[0], kv_len)), jnp.zeros((k.shape[0], align_kv_len - kv_len))], axis=1
        )
    if seg_q is not None and seg_kv is not None:
        segment_ids = SegmentIds(q=seg_q, kv=seg_kv)
    output = flash_attention(q, k, v, segment_ids=segment_ids, causal=False)
    output = output[:, :, :q_len, :]
    return output


def simple_attention(q, k, v):
    attn_weights = jnp.einsum("bhsd,bhtd->bhst", q, k)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    output = jnp.einsum("bhst,bhtd->bhsd", attn_weights, v)
    return output


def align_to(a, b):
    return pl.cdiv(a, b) * b


class TestFlashAttentionKernel(unittest.TestCase):
    """Test flash attention kernel"""

    def test_accuracy(self):
        """Test flash attention accuracy"""
        mesh = jax.make_mesh(
            (1, 1, 1, 1), axis_names=("x", "y", "z", "p"), devices=[jax.devices()[0]]
        )
        sharding = jax.sharding.NamedSharding(mesh, P(None, None, None, None))
        q_shape = (2, 12, 120, 128)
        kv_shape = (2, 12, 60, 128)
        key = jax.random.PRNGKey(1)
        key1, key2 = jax.random.split(key, num=2)
        q = jax.random.normal(key, q_shape)
        k = jax.random.normal(key1, kv_shape)
        v = jax.random.normal(key2, kv_shape)

        q = jax.device_put(q, sharding)
        k = jax.device_put(k, sharding)
        v = jax.device_put(v, sharding)

        flash_output = jit_flash_attention(q, k, v)
        simple_output = simple_attention(q, k, v)
        print(flash_output.shape, simple_output.shape)
        np.testing.assert_allclose(np.array(flash_output), np.array(simple_output), 1e-5, 1e-5)


if __name__ == "__main__":
    unittest.main()
