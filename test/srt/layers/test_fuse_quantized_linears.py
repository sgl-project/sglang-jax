"""CPU checks for ``fuse_quantized_linears``.

The fp8 blockwise matmul is a Mosaic kernel (TPU only), so on CPU we verify the
fused module is the exact row-concatenation of the per-module weights/scales and
that a dequantised reference matmul over the fused slices equals the per-module
reference bit-for-bit; kernel-level parity of fused vs unfused runs in the TPU
e2e gate (same kernel, same rows)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.layers.linear import QuantizedLinear, fuse_quantized_linears
from sgl_jax.srt.utils.quantization.per_channel_requant import skip_matches

BLOCK = (128, 128)


def _make(mesh, n_out, n_in, rng, activation_dtype):
    w_q = jnp.asarray(rng.standard_normal((n_out, n_in)), jnp.float8_e4m3fn)
    w_scale = jnp.asarray(rng.random((n_in // BLOCK[1], 1, n_out)) * 0.02 + 0.99, jnp.float32)
    return QuantizedLinear(
        weight_q=w_q,
        weight_scale=w_scale,
        bias=None,
        activation_dtype=activation_dtype,
        mesh=mesh,
        kernel_axes=(None, None),
        weight_block_size=BLOCK,
        scope_name=f"lin_{n_out}",
    )


def _dequant_matmul(x, w_q, w_scale):
    # w_q [n_out, n_in] fp8, w_scale [n_in//bk, 1, n_out] f32 -> x @ (w_q * scale).T
    n_out, n_in = w_q.shape
    in_blocks = w_scale.shape[0]
    bk = n_in // in_blocks
    w = np.asarray(w_q, np.float32).reshape(n_out, in_blocks, bk)
    sc = np.asarray(w_scale, np.float32)[:, 0, :].T  # [n_out, in_blocks]
    w = (w * sc[:, :, None]).reshape(n_out, n_in)
    return np.asarray(x, np.float32) @ w.T


def _run(activation_dtype):
    devs = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devs, ("data", "tensor"))
    rng = np.random.default_rng(0)
    n_in = 512
    outs = (256, 272, 128)  # 272: non-multiple of 128, like kv_a_proj's 576
    with mesh:
        mods = [_make(mesh, o, n_in, rng, activation_dtype) for o in outs]
        fused, splits = fuse_quantized_linears(mods, scope_name="fused")
    assert splits == list(outs), splits
    wq_f = np.asarray(fused.weight_q.value, np.float32)
    ws_f = np.asarray(fused.weight_scale.value, np.float32)
    n_tot = sum(outs)
    n_padded = -(-n_tot // 512) * 512  # zero-padded to a 512 multiple at build time
    assert wq_f.shape == (n_padded, n_in) and ws_f.shape == (n_in // BLOCK[1], 1, n_padded)
    np.testing.assert_array_equal(
        wq_f[:n_tot], np.concatenate([np.asarray(m.weight_q.value, np.float32) for m in mods], 0)
    )
    np.testing.assert_array_equal(
        ws_f[..., :n_tot],
        np.concatenate([np.asarray(m.weight_scale.value, np.float32) for m in mods], -1),
    )
    assert not wq_f[n_tot:].any() and (ws_f[..., n_tot:] == 1.0).all()
    assert fused.activation_dtype == activation_dtype and fused.weight_block_size == BLOCK
    x = jnp.asarray(rng.standard_normal((8, n_in)) * 0.1, jnp.bfloat16)
    ref = [_dequant_matmul(x, m.weight_q.value, m.weight_scale.value) for m in mods]
    out = _dequant_matmul(x, fused.weight_q.value, fused.weight_scale.value)
    parts = np.split(
        out, [sum(outs[: i + 1]) for i in range(len(outs))], axis=-1
    )  # last = padding tail
    assert parts[-1].shape[-1] == n_padded - n_tot and not parts[-1].any()
    for r, p_, o in zip(ref, parts, outs):
        assert p_.shape == r.shape == (8, o), (p_.shape, r.shape)
        np.testing.assert_allclose(
            p_, r, rtol=1e-5, atol=1e-5
        )  # numpy blocking differs by N width; TPU kernel parity is in the e2e gate


def test_fused_matches_unfused_bf16_act():
    _run(activation_dtype=None)


def test_fused_matches_unfused_fp8_act():
    _run(activation_dtype=jnp.float8_e4m3fn)


def test_not_fusable_returns_none():
    devs = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devs, ("data", "tensor"))
    rng = np.random.default_rng(1)
    with mesh:
        a = _make(mesh, 128, 512, rng, None)
        b = _make(mesh, 128, 256, rng, None)  # different input width
    assert fuse_quantized_linears([a, b], scope_name="x") is None


if __name__ == "__main__":
    test_fused_matches_unfused_bf16_act()
    test_fused_matches_unfused_fp8_act()
    test_not_fusable_returns_none()
    print("OK")


def test_member_paths_feed_the_requant_skip():
    # per_channel_requant_skip is a regex on the module path; the fused module must still be
    # skipped when one of the layers it absorbed matches (its own path no longer does)
    devs = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devs, ("data", "tensor"))
    rng = np.random.default_rng(1)
    with mesh:
        mods = [_make(mesh, 256, 512, rng, None), _make(mesh, 128, 512, rng, None)]
        fused, _ = fuse_quantized_linears(
            mods, scope_name="fused_in_proj", member_paths=["q_a_proj", "indexer/wk"]
        )
        plain, _ = fuse_quantized_linears(mods, scope_name="fused_in_proj")
    assert fused.fused_member_paths == ("q_a_proj", "indexer/wk")
    assert plain.fused_member_paths == ()
    path = "model/layers/3/self_attn/fused_in_proj"
    assert skip_matches("q_a_proj", path, fused.fused_member_paths)
    assert skip_matches(r"indexer/wk$", path, fused.fused_member_paths)
    assert skip_matches(r"layers/3/self_attn/q_a_proj", path, fused.fused_member_paths)
    assert not skip_matches("q_a_proj", path, plain.fused_member_paths)
    assert not skip_matches("o_proj", path, fused.fused_member_paths)
    assert not skip_matches(None, path, fused.fused_member_paths)
    assert skip_matches("fused_in_proj", path, ())
