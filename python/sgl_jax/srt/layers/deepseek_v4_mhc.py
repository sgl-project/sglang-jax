"""M1.2 -- the multi-stream hyper-connection (mHC) layer.

V4 keeps ``hc_mult`` parallel residual streams and wraps every attention and FFN
sublayer ``F`` in a pre/post pair::

    residual        = X
    u, post, comb   = hc_pre(X)          # collapse hc streams -> 1, emit the gates
    y               = F(norm(u))
    X'              = hc_post(y, residual, post, comb)   # expand 1 -> hc

and after the last layer collapses the streams once more for the LM head. This
module exposes the operations; M owns their placement and parameters. The
arithmetic lives in the mHC kernels from #341.

The sequential ``post`` then ``pre`` path is the correctness baseline. A fused
post/pre seam is optional and is outside this layer's initial interface.

Two backends
------------
The kernels in ``kernels/mhc`` are Pallas and TPU-only, so ``backend="reference"``
provides a pure-JAX path. It exists so M1.4 can be built and tested on CPU, the same
split M2.4 has against the HCA Pallas kernels. ``backend="auto"`` picks Pallas on TPU.
The reference is checked against the independent float64 NumPy oracle in
``test/srt/kernels/mhc/ref.py``, which was written from the published semantics rather
than from the kernels.

Three asymmetries that oracle calls load-bearing, reproduced here deliberately:

* ``pre`` adds ``hc_eps`` **after** the sigmoid; ``post`` does not, and carries a
  factor of two instead. (The kernel bakes that 2.0 in at ``mhc.py:62``, so there is
  no ``hc_post_alpha`` parameter to pass -- unlike the torch reference, which threads
  ``hc_post_mult_value``.)
* The Sinkhorn schedule is a row softmax, then one column pass, then
  ``sinkhorn_iters - 1`` row/column pairs, and every division is by ``sum + eps``.
* ``head_collapse`` applies the RMS scale to the activation **before** the projection
  with a bf16 rounding in between, whereas ``pre`` multiplies the projection result.
  Same-looking code, different numbers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgl_jax.srt.configs.deepseek_v4 import mhc_param_shapes, mix_hc_width

__all__ = [
    "DeepseekV4MHC",
    "collapse_head_reference",
    "expand_streams",
    "mhc_gates_reference",
    "post_reference",
    "pre_reference",
    "resolve_backend",
]


def expand_streams(hidden, hc_mult: int):
    """``[..., d]`` token embeddings -> ``[..., hc, d]`` identical streams.

    V4 does this once, after the embedding and before the first layer, and the shape
    is held until `collapse_head` at the very end.
    """
    hidden = jnp.asarray(hidden)
    if hidden.ndim < 2:
        raise ValueError(f"hidden must be [..., d], got {hidden.shape}")
    if hc_mult < 1:
        raise ValueError(f"hc_mult must be >= 1, got {hc_mult}")
    return jnp.repeat(hidden[..., None, :], hc_mult, axis=-2)


def resolve_backend(backend: str = "auto") -> str:
    """``"pallas"`` only where the mHC kernels can actually lower."""
    if backend not in ("auto", "pallas", "reference"):
        raise ValueError(f"unknown mHC backend {backend!r}")
    if backend != "auto":
        return backend
    return "pallas" if jax.default_backend() == "tpu" else "reference"


def _rms_scale(flat, norm_eps: float):
    return jax.lax.rsqrt(jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + norm_eps)


def mhc_gates_reference(mixes, hc_scale, hc_base, *, hc_mult: int, sinkhorn_iters: int, eps: float):
    """Split one projection into the pre, post and comb gates."""
    hc = hc_mult
    mixes = jnp.asarray(mixes, jnp.float32)
    scale = jnp.asarray(hc_scale, jnp.float32)
    base = jnp.asarray(hc_base, jnp.float32)
    if scale.shape != (3,):
        raise ValueError(f"hc_scale must be [3], got {scale.shape}")
    if base.shape != (mix_hc_width(hc),):
        raise ValueError(f"hc_base must be [{mix_hc_width(hc)}], got {base.shape}")
    if sinkhorn_iters < 1:
        raise ValueError(f"sinkhorn_iters must be >= 1, got {sinkhorn_iters}")

    # eps *after* the sigmoid for pre; post gets the factor of two and no eps.
    pre_gate = jax.nn.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post_gate = 2.0 * jax.nn.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])

    comb = mixes[..., 2 * hc :].reshape(*mixes.shape[:-1], hc, hc)
    comb = comb * scale[2] + base[2 * hc :].reshape(hc, hc)
    comb = jax.nn.softmax(comb, axis=-1) + eps
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    # One column pass has already happened; the remaining iterations are pairs.
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    return pre_gate, post_gate, comb


def pre_reference(
    x_streams,
    hc_fn,
    hc_scale,
    hc_base,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
):
    """Collapse hc streams to one and emit the gates `post` will need."""
    x = jnp.asarray(x_streams, jnp.float32)
    if x.ndim < 3:
        raise ValueError(f"x_streams must be [..., hc, d], got {x.shape}")
    flat = x.reshape(*x.shape[:-2], -1)
    # For `pre` the RMS scale multiplies the *projection*, not the activation.
    mixes = (flat @ jnp.asarray(hc_fn, jnp.float32).T) * _rms_scale(flat, norm_eps)
    pre_gate, post_gate, comb = mhc_gates_reference(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=hc_eps
    )
    y = jnp.sum(pre_gate[..., None] * x, axis=-2)
    return y, post_gate, comb


def post_reference(x, residual_streams, post_gate, comb):
    """Expand one stream to hc: ``out[j] = post[j]*x + sum_i comb[i,j]*res[i]``."""
    x = jnp.asarray(x, jnp.float32)
    residual = jnp.asarray(residual_streams, jnp.float32)
    mixed = jnp.einsum("...ij,...ih->...jh", jnp.asarray(comb, jnp.float32), residual)
    return jnp.asarray(post_gate, jnp.float32)[..., None] * x[..., None, :] + mixed


def collapse_head_reference(x_streams, hc_fn, hc_scale, hc_base, *, norm_eps: float, hc_eps: float):
    """Final hc -> 1 collapse: sigmoid gates only, no Sinkhorn.

    The RMS scale hits the activation **before** the projection here, with a bf16
    rounding in between. That boundary is part of the op, not an implementation
    detail, and it is what makes this different from `pre_reference`.
    """
    x = jnp.asarray(x_streams, jnp.float32)
    if x.ndim < 3:
        raise ValueError(f"x_streams must be [..., hc, d], got {x.shape}")
    flat = x.reshape(*x.shape[:-2], -1)
    normalized = (flat * _rms_scale(flat, norm_eps)).astype(jnp.bfloat16).astype(jnp.float32)
    mixes = normalized @ jnp.asarray(hc_fn, jnp.float32).T
    gate = jax.nn.sigmoid(mixes * jnp.asarray(hc_scale, jnp.float32)[0] + hc_base) + hc_eps
    return jnp.sum(gate[..., None] * x, axis=-2)


class DeepseekV4MHC:
    """Stateless mHC helper for one decoder layer: its pre/post pair.

    Holds no parameters -- only config scalars, the resolved backend, and the
    expected parameter shapes. The decoder layer owns two independent gate
    parameter sets -- one for the attention sublayer and one for the FFN, because
    each sublayer gets its own pre/post -- and passes one set to each `pre` call.
    The model-level head parameters likewise live on the model; `collapse_head` is
    exposed as a static entry point for it.

    Those parameters are float32 and must stay float32: they are gate coefficients
    fed to a Sinkhorn normalisation, not projections, so they do not follow the
    model's activation dtype.

    The operations act independently on leading token rows. M may shard those
    rows across devices while keeping the residual-stream and hidden axes local;
    gate parameters are replicated at the operation boundary.
    """

    def __init__(self, config, *, backend: str = "auto"):
        self.hc_mult = int(config.hc_mult)
        self.hidden_size = int(config.hidden_size)
        self.sinkhorn_iters = int(config.hc_sinkhorn_iters)
        self.hc_eps = float(config.hc_eps)
        self.norm_eps = float(config.rms_norm_eps)
        if self.hc_mult < 1 or self.hidden_size < 1:
            raise ValueError("mHC residual-stream count and hidden size must be positive")
        if self.sinkhorn_iters < 1:
            raise ValueError("hc_sinkhorn_iters must be >= 1")
        if not 0 < self.hc_eps < float("inf") or not 0 < self.norm_eps < float("inf"):
            raise ValueError("mHC epsilon values must be positive and finite")
        self.backend = resolve_backend(backend)
        self.shapes = mhc_param_shapes(config)

    def _check_streams(self, streams):
        streams = jnp.asarray(streams)
        if streams.ndim < 3 or streams.shape[-2:] != (self.hc_mult, self.hidden_size):
            raise ValueError(
                f"mHC streams must end in {(self.hc_mult, self.hidden_size)}, got {streams.shape}"
            )
        if streams.dtype not in (jnp.bfloat16, jnp.float32):
            raise ValueError(f"mHC streams must be bfloat16 or float32, got {streams.dtype}")
        return streams

    # -- parameter checking ----------------------------------------------------

    def check_params(self, fn, base, scale, *, kind: str = "fn"):
        """Validate one gate parameter set against the config-derived shapes."""
        expected = (self.shapes[kind], self.shapes["base"], self.shapes["scale"])
        if kind == "head_fn":
            expected = (self.shapes["head_fn"], self.shapes["head_base"], self.shapes["head_scale"])
        for name, array, want in zip(("fn", "base", "scale"), (fn, base, scale), expected):
            got = tuple(jnp.asarray(array).shape)
            if got != want:
                raise ValueError(f"mHC {kind} {name} must be {want}, got {got}")
            if jnp.asarray(array).dtype != jnp.float32:
                raise ValueError(
                    f"mHC {kind} {name} must stay float32; gate coefficients do not "
                    f"follow the activation dtype (got {jnp.asarray(array).dtype})"
                )

    # -- the pre/post pair -----------------------------------------------------

    def pre(self, x_streams, fn, base, scale):
        """``[..., hc, d]`` -> ``(sublayer input [..., d], post, comb)``."""
        self._check_streams(x_streams)
        self.check_params(fn, base, scale)
        if self.backend == "pallas":
            from sgl_jax.srt.kernels.mhc import mhc_pre_fused

            return mhc_pre_fused(
                x_streams,
                fn,
                scale,
                base,
                hc_mult=self.hc_mult,
                sinkhorn_iters=self.sinkhorn_iters,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
            )
        return pre_reference(
            x_streams,
            fn,
            scale,
            base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.sinkhorn_iters,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def post(self, y, residual_streams, post_gate, comb):
        """``(sublayer output, residual streams, gates)`` -> new streams."""
        residual_streams = self._check_streams(residual_streams)
        y = jnp.asarray(y)
        post_gate = jnp.asarray(post_gate)
        comb = jnp.asarray(comb)
        lead = residual_streams.shape[:-2]
        if y.shape != (*lead, self.hidden_size):
            raise ValueError(f"mHC post output must be {(*lead, self.hidden_size)}, got {y.shape}")
        if post_gate.shape != (*lead, self.hc_mult):
            raise ValueError(
                f"mHC post gates must be {(*lead, self.hc_mult)}, got {post_gate.shape}"
            )
        if comb.shape != (*lead, self.hc_mult, self.hc_mult):
            raise ValueError(
                f"mHC mixing matrix must be {(*lead, self.hc_mult, self.hc_mult)}, got {comb.shape}"
            )
        if y.dtype not in (jnp.bfloat16, jnp.float32):
            raise ValueError(f"mHC post output must be bfloat16 or float32, got {y.dtype}")
        if post_gate.dtype != jnp.float32 or comb.dtype != jnp.float32:
            raise ValueError("mHC post gates and mixing matrix must stay float32")
        if self.backend == "pallas":
            from sgl_jax.srt.kernels.mhc import mhc_post_fused

            return mhc_post_fused(y, residual_streams, post_gate, comb)
        return post_reference(y, residual_streams, post_gate, comb)

    def collapse_head(self, x_streams, head_fn, head_base, head_scale):
        """The model-level ``hc -> 1`` collapse before the final norm and LM head.

        Order matters and is easy to invert: this runs **before** the final RMSNorm,
        which runs before the LM head. The output follows the streams' dtype on both
        backends.
        """
        self._check_streams(x_streams)
        self.check_params(head_fn, head_base, head_scale, kind="head_fn")
        if self.backend == "pallas":
            from sgl_jax.srt.kernels.mhc import mhc_head_collapse_fused

            return mhc_head_collapse_fused(
                x_streams,
                head_fn,
                head_scale,
                head_base,
                hc_mult=self.hc_mult,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
            )
        return collapse_head_reference(
            x_streams,
            head_fn,
            head_scale,
            head_base,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        ).astype(jnp.asarray(x_streams).dtype)
