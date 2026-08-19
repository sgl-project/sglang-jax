"""MXFP4 dequantization tests, including against REAL Kimi-K3 checkpoint tensors.

Synthetic round-trips can miss layout errors (pair order inside a byte, e8m0 bias, which axis the
group spans). The real-weight test pulls one shard of the released K3 checkpoint and checks the
unpacked shapes and value range are consistent with the declared config.
"""
import jax, jax.numpy as jnp, numpy as np, pytest
from sgl_jax.srt.layers.quantization.mxfp4 import (
    u8_unpack_e2m1, e8m0_to_fp32, dequantize_tensor_from_mxfp4_packed,
    is_mxfp4_packed_config, MXFP4_GROUP_SIZE)

# e2m1 has 16 representable values; this is the full codebook.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def test_unpack_doubles_last_dim_and_yields_only_codebook_values():
    u8 = jnp.asarray(np.arange(256, dtype=np.uint8).reshape(8, 32))
    out = u8_unpack_e2m1(u8)
    assert out.shape == (8, 64), out.shape          # two fp4 per byte
    vals = set(np.asarray(out.astype(jnp.float32)).ravel().tolist())
    assert vals <= set(E2M1), sorted(vals - set(E2M1))


def test_unpack_low_nibble_first():
    """Byte 0x21 must unpack as (codebook[1], codebook[2]) -- low nibble first."""
    out = np.asarray(u8_unpack_e2m1(jnp.asarray(np.array([[0x21]], np.uint8))).astype(jnp.float32))
    assert out.tolist() == [[E2M1[1], E2M1[2]]], out.tolist()


def test_e8m0_is_exponent_arithmetic_not_a_cast():
    """e8m0 is a bare exponent: value = 2**(u8 - bias). A cast would be wildly wrong."""
    u8 = jnp.asarray(np.array([127, 128, 126, 0], np.uint8))
    got = np.asarray(e8m0_to_fp32(u8))
    assert got[0] == pytest.approx(1.0)             # 127 is the bias -> 2^0
    assert got[1] == pytest.approx(2.0)
    assert got[2] == pytest.approx(0.5)
    # u8=0 -> exponent -127; 2**-127 is subnormal in fp32 and flushes to 0. That is correct
    # e8m0 behaviour (0 is the minimum exponent), not a conversion bug.
    assert got[3] >= 0.0 and got[3] < 1e-30


def test_full_dequant_roundtrip_group32():
    """Quantize a known tensor to the fp4 grid, pack, then dequantize exactly."""
    rng = np.random.default_rng(0)
    rows, k = 4, 128
    idx = rng.integers(0, 16, size=(rows, k)).astype(np.uint8)
    vals = np.array(E2M1, np.float32)[idx]                       # exact codebook values
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
    n_groups = k // MXFP4_GROUP_SIZE
    scale_u8 = np.full((rows, n_groups), 127 + 2, np.uint8)      # 2**2 = 4
    got = np.asarray(dequantize_tensor_from_mxfp4_packed(
        jnp.asarray(packed), jnp.asarray(scale_u8), axis=-1, out_dtype=jnp.float32))
    want = vals * 4.0
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_recognizes_k3_config_group_nesting():
    """K3 puts the format under config_groups.<g>.format, not at the top level."""
    k3 = {"config_groups": {"group_0": {"format": "mxfp4-pack-quantized",
                                        "weights": {"group_size": 32, "num_bits": 4}}}}
    assert is_mxfp4_packed_config(k3)
    assert not is_mxfp4_packed_config({"config_groups": {"g": {"format": "int8"}}})
    assert not is_mxfp4_packed_config(None)


REAL = pytest.mark.skipif(
    not __import__("os").path.exists("/dev/shm/k3_probe.safetensors"),
    reason="real K3 shard not staged")


@REAL
def test_real_k3_shard_unpacks_consistently():
    """Unpack real K3 tensors; shapes and value range must match the declared config."""
    from safetensors import safe_open
    f = safe_open("/dev/shm/k3_probe.safetensors", framework="np")
    keys = list(f.keys())
    packed = [k for k in keys if k.endswith("weight_packed")]
    # NOTE: shard 1 has NO packed tensors -- K3 leaves attention (A_log, dt_bias, q_conv1d),
    # shared experts, dense MLP and lm_head unquantized (config targets ["Linear"] only). The
    # MXFP4 MoE experts live in later shards; this test wants one of those.
    assert packed, f"no packed tensors; first keys: {keys[:5]}"
    t = {}
    for k in packed[:3]:
        t[k] = f.get_tensor(k)
        sk = k.replace("weight_packed", "weight_scale")
        if sk in keys:
            t[sk] = f.get_tensor(sk)
    for k in packed[:3]:
        w = t[k]
        sk = k.replace("weight_packed", "weight_scale")
        assert w.dtype == np.uint8, (k, w.dtype)
        out = np.asarray(u8_unpack_e2m1(jnp.asarray(w)).astype(jnp.float32))
        assert out.shape[-1] == w.shape[-1] * 2
        assert set(np.unique(out).tolist()) <= set(E2M1)
        if sk in t:
            s = t[sk]
            assert s.dtype == np.uint8, (sk, s.dtype)
            # one scale per group of 32 along the quantized axis
            assert out.shape[-1] == s.shape[-1] * MXFP4_GROUP_SIZE, (out.shape, s.shape)
            sf = np.asarray(e8m0_to_fp32(jnp.asarray(s)))
            assert np.all(sf > 0) and np.all(np.isfinite(sf))
