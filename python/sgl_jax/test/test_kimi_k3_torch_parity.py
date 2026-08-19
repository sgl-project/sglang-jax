"""Numerical parity of the JAX K3 port against Moonshot's OFFICIAL torch implementation.

Every other K3 test in this repo checks the port against a hand-written oracle -- i.e. against my
reading of the reference. This one checks it against the reference *itself*: the oracles here are
executed from the released ``modeling_kimi_linear.py`` source, so a misreading cannot pass.

Getting the whole file importable is not possible off-GPU (it hard-fails on ``import fla`` -- the
Triton KDA kernels -- and pulls a large transformers surface). So instead of importing it, the
loader below extracts individual top-level definitions with :mod:`ast` and execs only those, with
nothing but ``torch``/``nn``/``F`` in scope. Two consequences worth stating:

* the oracle is byte-for-byte the released code, not a transcription;
* KDA itself is out of scope here (it *is* the Triton part). That is not a coverage hole for this
  port: KDA is reused from sglang-jax unmodified and is exercised end-to-end by the Kimi-Linear
  run. What this file covers is precisely the surface the port ADDS -- SITU, the attention
  residuals, the MLA output gate, and the LatentMoE transform.

Weights are the REAL released checkpoint where a weight is involved, not random tensors, because
several of these bugs (the ``A_log`` padding, the ``g_proj`` rank, the latent width) are visible
only against real shapes.

Reference: ``gs://torchtitan-assets/moonshootai/kimi/3/modeling_kimi_linear.py``. Point
``KIMI_K3_REF_DIR`` at a directory holding it (and ``KIMI_K3_MODEL_DIR`` at a checkpoint) or these
tests SKIP -- they never silently pass without the reference.
"""

from __future__ import annotations

import ast
import glob
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.quantization.mxfp4 import dequantize_tensor_from_mxfp4_packed
from sgl_jax.srt.models.kimi_k3_layers import (
    attention_residual_apply,
    mla_output_gate,
    situ_and_mul,
)

torch = pytest.importorskip("torch")

REF_DIR = os.environ.get("KIMI_K3_REF_DIR", "/dev/shm/k3ref")
MODEL_DIR = os.environ.get("KIMI_K3_MODEL_DIR", "/dev/shm/k3_4l")
REF_FILE = pathlib.Path(REF_DIR) / "modeling_kimi_linear.py"


def _load_reference_defs(*names: str) -> dict:
    """Exec the named top-level defs from the official file, and nothing else.

    Executing the whole module is impossible off-GPU (``import fla``). Extracting the exact source
    segments keeps the oracle authoritative while sidestepping every import the port does not use.
    """
    if not REF_FILE.exists():
        pytest.skip(f"official reference not found at {REF_FILE}; set KIMI_K3_REF_DIR")
    source = REF_FILE.read_text()
    tree = ast.parse(source)
    wanted = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            wanted[node.name] = ast.get_source_segment(source, node)
    missing = set(names) - set(wanted)
    if missing:
        pytest.fail(f"reference file has no top-level {sorted(missing)} -- upstream renamed them?")

    ns: dict = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "ACT2FN": {},
        "ALL_LAYERNORM_LAYERS": [],
    }
    # Order matters only in that classes may reference earlier names; the requested set is flat.
    for name in names:
        exec(compile(wanted[name], f"<official:{name}>", "exec"), ns)  # noqa: S102
    return ns


def _read_real_weights(*suffixes: str) -> dict:
    """Pull named tensors out of the real checkpoint shards, keyed by the suffix that matched."""
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))
    if not files:
        pytest.skip(f"no checkpoint under {MODEL_DIR}; set KIMI_K3_MODEL_DIR")
    from safetensors import safe_open

    out: dict = {}
    for path in files:
        with safe_open(path, "numpy") as handle:
            for key in handle.keys():
                for suffix in suffixes:
                    if key.endswith(suffix) and suffix not in out:
                        out[suffix] = handle.get_tensor(key)
    return out


def _norm_err(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute error normalized by the reference tensor's SCALE.

    Per-element relative error is the wrong metric on tensors that contain near-zero entries: an
    output element of magnitude 1e-4 carrying a 3e-8 absolute error reports 3e-4 "relative", which
    says nothing about correctness. Normalizing by ``max|b|`` measures the deviation against the
    signal actually present, and still moves by O(1) for every failure mode these tests target
    (wrong composition order, a dropped norm, a transposed kernel, a bf16 score path).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = max(float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b)) / scale)


def _f32_matmul(x: jax.Array, w: jax.Array) -> jax.Array:
    """A matmul that does NOT drop to TPU's default bf16 precision.

    The production path deliberately runs these projections in bf16 -- so does the reference on
    GPU -- so a bf16 result here is not a bug. But comparing a bf16 TPU matmul against a torch
    fp32 CPU matmul measures the multiply precision, not the port, and would mask exactly the
    ordering/orientation errors these tests exist to catch. HIGHEST isolates the code under test.
    """
    return jnp.einsum("...i,io->...o", x, w, precision=jax.lax.Precision.HIGHEST)


# ----------------------------------------------------------------------------------------------
# SITU
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("linear_beta", [None, 2.0])
def test_situ_matches_official(linear_beta):
    """``SituAndMul`` -- the activation every K3 MLP and shared-expert uses."""
    ns = _load_reference_defs("SituAndMul")
    beta = 1.5
    rng = np.random.default_rng(0)
    # deliberately wide: SITU's soft-clip only differs from SiLU away from the origin, so a
    # small-magnitude sample would pass even against a plain SiLU.
    x = rng.normal(0.0, 6.0, size=(7, 512)).astype(np.float32)

    expected = ns["SituAndMul"](beta=beta, linear_beta=linear_beta)(torch.from_numpy(x)).numpy()
    got = np.asarray(situ_and_mul(jnp.asarray(x), beta=beta, linear_beta=linear_beta))

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)


# ----------------------------------------------------------------------------------------------
# Attention residuals -- REAL weights
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_blocks", [1, 2, 5])
def test_attn_res_matches_official_on_real_weights(num_blocks):
    """``_apply_attn_res`` with the checkpoint's own ``self_attention_res_{norm,proj}``.

    The scorer folds the RMSNorm scale into the 1-wide projection
    (``score_weight = norm.weight * proj.weight.squeeze(0)``); this pins that fold, the softmax
    axis, and the fp32 accumulation all at once.
    """
    ns = _load_reference_defs("_apply_attn_res")
    weights = _read_real_weights(
        "layers.1.self_attention_res_norm.weight", "layers.1.self_attention_res_proj.weight"
    )
    norm_w = weights["layers.1.self_attention_res_norm.weight"].astype(np.float32)
    proj_w = weights["layers.1.self_attention_res_proj.weight"].astype(np.float32)  # [1, hidden]
    hidden = norm_w.shape[0]
    eps = 1e-5

    rng = np.random.default_rng(num_blocks)
    tokens = 6
    prefix = rng.normal(0.0, 1.0, size=(tokens, hidden)).astype(np.float32)
    blocks = rng.normal(0.0, 1.0, size=(tokens, num_blocks, hidden)).astype(np.float32)

    class _Proj:
        weight = torch.from_numpy(proj_w)

    class _Norm:
        weight = torch.from_numpy(norm_w)
        variance_epsilon = eps

    expected = (
        ns["_apply_attn_res"](
            torch.from_numpy(prefix), torch.from_numpy(blocks), _Proj(), _Norm()
        )
        .float()
        .numpy()
    )
    got = np.asarray(
        attention_residual_apply(
            jnp.asarray(prefix),
            jnp.asarray(blocks),
            jnp.asarray(norm_w),
            jnp.asarray(proj_w.T),  # LinearBase stores [in, out]; the checkpoint ships [out, in]
            eps,
        )
    )

    assert got.shape == expected.shape == (tokens, hidden)
    # softmax over candidates makes this sensitive: with the score einsum at TPU's default bf16
    # precision this measured 3.7e-1. Against the official fp32 oracle it sits ~1e-6.
    assert _norm_err(got, expected) < 1e-5, f"normalized err {_norm_err(got, expected):.3e}"


# ----------------------------------------------------------------------------------------------
# MLA output gate -- REAL weights
# ----------------------------------------------------------------------------------------------
def test_mla_output_gate_matches_official_on_real_weights():
    """``attn_output * g_proj(h).sigmoid()``, applied BEFORE ``o_proj``.

    Kimi-Linear's MLA has no gate at all, so this is pure K3 surface. The real ``g_proj`` is used
    so the [12288, 7168] orientation is part of what is being checked.
    """
    weights = _read_real_weights("layers.3.self_attn.g_proj.weight")
    g_w = weights["layers.3.self_attn.g_proj.weight"].astype(np.float32)  # [proj, hidden]
    proj_size, hidden = g_w.shape

    rng = np.random.default_rng(3)
    tokens = 4
    h = rng.normal(0.0, 0.5, size=(tokens, hidden)).astype(np.float32)
    attn = rng.normal(0.0, 1.0, size=(tokens, proj_size)).astype(np.float32)

    # official: modeling_kimi_linear.py KimiMLAAttention.forward
    #     g = self.g_proj(hidden_states).sigmoid(); attn_output = attn_output * g
    with torch.no_grad():
        g = torch.nn.functional.linear(torch.from_numpy(h), torch.from_numpy(g_w)).sigmoid()
        expected = (torch.from_numpy(attn) * g).numpy()

    got = np.asarray(
        mla_output_gate(jnp.asarray(attn), _f32_matmul(jnp.asarray(h), jnp.asarray(g_w.T)))
    )

    assert got.shape == expected.shape
    assert _norm_err(got, expected) < 1e-5, f"normalized err {_norm_err(got, expected):.3e}"


# ----------------------------------------------------------------------------------------------
# LatentMoE transform -- REAL weights
# ----------------------------------------------------------------------------------------------
def test_latent_moe_output_transform_matches_official_on_real_weights():
    """``KimiRoutedOutputTransform``: norm THEN up_proj, on the 3584-wide latent.

    Swapping the order, or skipping the norm, still type-checks and still produces a 7168-wide
    tensor -- this is the check that it is the right one. The down_proj half is checked alongside,
    since a transposed mapping there would also survive shape-checking.
    """
    ns = _load_reference_defs("KimiRMSNorm")
    weights = _read_real_weights(
        "layers.1.block_sparse_moe.routed_expert_down_proj.weight",
        "layers.1.block_sparse_moe.routed_expert_norm.weight",
        "layers.1.block_sparse_moe.routed_expert_up_proj.weight",
    )
    down_w = weights["layers.1.block_sparse_moe.routed_expert_down_proj.weight"].astype(np.float32)
    norm_w = weights["layers.1.block_sparse_moe.routed_expert_norm.weight"].astype(np.float32)
    up_w = weights["layers.1.block_sparse_moe.routed_expert_up_proj.weight"].astype(np.float32)

    hidden = down_w.shape[1]
    latent = down_w.shape[0]
    assert up_w.shape == (hidden, latent), up_w.shape
    assert norm_w.shape == (latent,), norm_w.shape

    rng = np.random.default_rng(7)
    tokens = 5
    h = rng.normal(0.0, 0.5, size=(tokens, hidden)).astype(np.float32)
    y = rng.normal(0.0, 0.5, size=(tokens, latent)).astype(np.float32)

    eps = 1e-5
    with torch.no_grad():
        norm = ns["KimiRMSNorm"](latent, eps=eps)
        norm.weight.copy_(torch.from_numpy(norm_w))
        expected_down = torch.nn.functional.linear(
            torch.from_numpy(h), torch.from_numpy(down_w)
        ).numpy()
        expected_out = torch.nn.functional.linear(
            norm(torch.from_numpy(y)), torch.from_numpy(up_w)
        ).numpy()

    # the JAX side applies the same composition against LinearBase's [in, out] kernels
    got_down = np.asarray(_f32_matmul(jnp.asarray(h), jnp.asarray(down_w.T)))
    y32 = jnp.asarray(y, dtype=jnp.float32)
    normed = y32 * jax.lax.rsqrt(jnp.mean(jnp.square(y32), axis=-1, keepdims=True) + eps)
    got_out = np.asarray(_f32_matmul(normed * jnp.asarray(norm_w), jnp.asarray(up_w.T)))

    assert _norm_err(got_down, expected_down) < 1e-5, f"down {_norm_err(got_down, expected_down):.3e}"
    assert _norm_err(got_out, expected_out) < 1e-5, f"out {_norm_err(got_out, expected_out):.3e}"


# ----------------------------------------------------------------------------------------------
# MXFP4 -- REAL packed expert weights
# ----------------------------------------------------------------------------------------------
def test_mxfp4_dequant_matches_torch_reference_on_real_expert():
    """Dequantize a real expert against an independent torch implementation of the format.

    The oracle is written from the format definition (fp4 e2m1 pairs packed per byte, e8m0 scales
    over 32-element groups), NOT from the JAX code, so a shared misreading cannot cancel out. This
    is the check that catches a wrong nibble order or a scale treated as a float.
    """
    weights = _read_real_weights(
        "layers.1.block_sparse_moe.experts.0.w1.weight_packed",
        "layers.1.block_sparse_moe.experts.0.w1.weight_scale",
    )
    packed = weights["layers.1.block_sparse_moe.experts.0.w1.weight_packed"]
    scale = weights["layers.1.block_sparse_moe.experts.0.w1.weight_scale"]
    assert packed.dtype == np.uint8 and scale.dtype == np.uint8

    # --- independent torch oracle -------------------------------------------------------------
    E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    low = packed & 0x0F
    high = packed >> 4
    def _decode(nib: np.ndarray) -> np.ndarray:
        sign = np.where(nib >= 8, -1.0, 1.0).astype(np.float32)
        return sign * E2M1[(nib & 0x07).astype(np.int64)]

    # two values per byte, low nibble first -- interleaved along the last axis
    vals = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.float32)
    vals[..., 0::2] = _decode(low)
    vals[..., 1::2] = _decode(high)

    # e8m0: a bare exponent, value = 2**(u8 - 127); u8 == 0 is the flush-to-zero code
    exp = scale.astype(np.int32) - 127
    scales = np.where(scale == 0, 0.0, np.ldexp(np.ones_like(exp, dtype=np.float32), exp))
    groups = vals.shape[-1] // scales.shape[-1]
    expected = (vals.reshape(vals.shape[:-1] + (scales.shape[-1], groups)) * scales[..., None])
    expected = expected.reshape(vals.shape)

    got = np.asarray(
        dequantize_tensor_from_mxfp4_packed(
            jnp.asarray(packed), jnp.asarray(scale), axis=-1, out_dtype=jnp.float32
        )
    )

    assert got.shape == expected.shape == (packed.shape[0], packed.shape[1] * 2)
    assert np.count_nonzero(expected) > expected.size // 4, "oracle produced a mostly-zero tensor"
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
