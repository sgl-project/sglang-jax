"""DeepSeek-V4 multi-stream hyper-connection operators for TPU.

mHC keeps H residual streams around each transformer block: pre collapses H
streams to one, post expands the block output back to H, and head performs the
final H-to-1 collapse.

mhc_gates builds the post gate and Sinkhorn-normalized mixing matrix.
mhc_pre_fused combines RMS, gate projection, and pre collapse.
mhc_post_fused combines stream expansion and residual remixing.
mhc_head_collapse_fused combines RMS, gate projection, and final collapse.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

from sgl_jax.srt.kernels.mhc.tune import (
    select_collapse_block_tokens,
    select_gates_block_tokens,
    select_post_backend,
    select_post_block_tokens,
)


def _device_kind() -> str:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("mHC requires an available JAX device")
    return devices[0].device_kind


def mix_hc_width(hc_mult: int) -> int:
    # pre(H) + post(H) + comb(H*H); DeepSeek-V4 uses H=4, so mix_hc=24.
    return (2 + hc_mult) * hc_mult


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _sinkhorn_gates_kernel(
    mixes_ref,  # [mix_hc, BT] f32
    scale_ref,  # [3, 1] f32
    base_ref,  # [mix_hc, 1] f32
    post_ref,  # [hc, BT] f32
    comb_ref,  # [hc*hc, BT] f32
    *,
    hc: int,
    sinkhorn_iters: int,
    eps: float,
):
    mixes = mixes_ref[...]
    base = base_ref[...]
    scale = scale_ref[...]

    # The pre gate is consumed inside collapse; post has no eps and carries 2x.
    post_ref[...] = 2.0 * jax.nn.sigmoid(mixes[hc : 2 * hc] * scale[1:2] + base[hc : 2 * hc])

    # Features-major layout keeps tokens on lanes while row/column reductions
    # operate on the two leading axes.
    c = mixes[2 * hc :] * scale[2:3] + base[2 * hc :]
    c = c.reshape(hc, hc, -1)

    # First iteration is row softmax plus eps, then column normalization.
    c = c - jnp.max(c, axis=1, keepdims=True)
    c = jnp.exp(c)
    c = c / jnp.sum(c, axis=1, keepdims=True) + eps
    c = c / (jnp.sum(c, axis=0, keepdims=True) + eps)

    # Later row/column iterations remain in VMEM.
    def body(_, cc):
        cc = cc / (jnp.sum(cc, axis=1, keepdims=True) + eps)
        cc = cc / (jnp.sum(cc, axis=0, keepdims=True) + eps)
        return cc

    # Unrolling preserves the dependent iteration order.
    c = jax.lax.fori_loop(0, sinkhorn_iters - 1, body, c, unroll=True)

    comb_ref[...] = c.reshape(hc * hc, -1)


@functools.partial(
    jax.jit,
    static_argnames=("hc_mult", "sinkhorn_iters", "eps", "block_tokens", "interpret"),
)
def mhc_gates(
    mixes: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    block_tokens: int | None = None,
    interpret: bool | None = None,
):
    hc = hc_mult
    mix_hc = mix_hc_width(hc)
    if mixes.ndim < 1 or mixes.shape[-1] != mix_hc:
        raise ValueError(f"mixes must end in mix_hc={mix_hc} for hc_mult={hc}, got {mixes.shape}")
    if hc_scale.shape != (3,):
        raise ValueError(f"hc_scale must be [3], got {hc_scale.shape}")
    if hc_base.shape != (mix_hc,):
        raise ValueError(f"hc_base must be [{mix_hc}], got {hc_base.shape}")
    if sinkhorn_iters < 1:
        raise ValueError(f"sinkhorn_iters must be >= 1, got {sinkhorn_iters}")

    lead = mixes.shape[:-1]
    # Put independent tokens on the TPU lane axis for parallel VPU processing.
    mixes_t = mixes.reshape(-1, mix_hc).T
    n = mixes_t.shape[1]
    bt = select_gates_block_tokens(_device_kind(), tokens=n, hc_mult=hc, block_tokens=block_tokens)
    n_pad = -(-n // bt) * bt  # round up
    if n_pad != n:
        mixes_t = jnp.pad(mixes_t, ((0, 0), (0, n_pad - n)))

    if interpret is None:
        interpret = get_interpret()

    kernel = functools.partial(
        _sinkhorn_gates_kernel, hc=hc, sinkhorn_iters=int(sinkhorn_iters), eps=float(eps)
    )
    post, comb = pl.pallas_call(
        kernel,
        grid=(n_pad // bt,),
        in_specs=[
            pl.BlockSpec((mix_hc, bt), lambda i: (0, i)),
            pl.BlockSpec((3, 1), lambda i: (0, 0)),
            pl.BlockSpec((mix_hc, 1), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((hc, bt), lambda i: (0, i)),
            pl.BlockSpec((hc * hc, bt), lambda i: (0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((hc, n_pad), jnp.float32),
            jax.ShapeDtypeStruct((hc * hc, n_pad), jnp.float32),
        ],
        interpret=interpret,
        name="mhc-sinkhorn-gates",
    )(
        mixes_t.astype(jnp.float32),
        hc_scale.astype(jnp.float32).reshape(3, 1),
        hc_base.astype(jnp.float32).reshape(mix_hc, 1),
    )

    if n_pad != n:
        post, comb = post[:, :n], comb[:, :n]
    post = post.T.reshape(*lead, hc_mult)
    comb = jnp.transpose(comb.reshape(hc, hc, n), (2, 0, 1)).reshape(*lead, hc_mult, hc_mult)
    return post, comb


# Collapse hc streams for pre and head.


def _collapse_kernel(
    x_ref,  # [BT, hc, d]  input dtype
    fn_ref,  # [rows, hc*d] f32   (rows = hc for "head", mix_hc for "pre")
    scale_ref,  # [4] f32 (padded; only the first 1 or 3 entries are used)
    base_ref,  # [rows] f32
    *outs,  # y_ref [BT, d], followed by mixes_ref [BT, rows] in "pre"
    hc: int,
    d: int,
    mode: str,
    hc_eps: float,
    norm_eps: float,
    dot_precision,
):
    x = x_ref[...]  # [BT, hc, d]
    bt = x.shape[0]
    y_ref, *extra_outs = outs
    # Fuse RMS with projection to reuse the resident activation. Its reduction
    # tree differs from XLA's, so a few ULP of output drift are expected.
    rms = jax.lax.rsqrt(
        jnp.sum(jnp.square(x.astype(jnp.float32).reshape(bt, hc * d)), axis=-1, keepdims=True)
        / (hc * d)
        + norm_eps
    )

    if mode == "head":
        # Head requires the FP32 normalize -> BF16 round -> projection boundary.
        xf = x.astype(jnp.float32).reshape(bt, hc * d)
        normalized = (xf * rms).astype(jnp.bfloat16)
        if dot_precision == jax.lax.Precision.HIGHEST:
            # Preserve the BF16 boundary, then widen for the FP32 MXU contract.
            normalized = normalized.astype(jnp.float32)
        mixes = jax.lax.dot_general(
            normalized,
            fn_ref[...],
            (((1,), (1,)), ((), ())),
            precision=dot_precision,
            preferred_element_type=jnp.float32,
        )
    else:
        xf = x.astype(jnp.float32).reshape(bt, hc * d)
        # Pre moves the RMS scalar after the linear projection.
        mixes = (
            jax.lax.dot_general(
                xf,
                fn_ref[...],
                (((1,), (1,)), ((), ())),
                precision=dot_precision,
                preferred_element_type=jnp.float32,
            )
            * rms
        )  # [BT, rows]

    scale = scale_ref[...].reshape(-1)
    base = base_ref[...].reshape(-1)

    if mode == "head":
        pre = jax.nn.sigmoid(mixes * scale[0] + base[None, :]) + hc_eps
    else:
        pre = jax.nn.sigmoid(mixes[:, :hc] * scale[0] + base[None, :hc]) + hc_eps
        # Preserve the full projection layout; slicing can change its reduction tree.
        extra_outs[0][...] = mixes

    # VPU accumulation preserves FP32 gates; Mosaic's MXU dot rounds them to BF16.
    # Stream each residual into one accumulator instead of widening [BT, hc, d].
    collapsed = jnp.zeros((bt, d), dtype=jnp.float32)
    for stream in range(hc):
        collapsed = collapsed + pre[:, stream, None] * x[:, stream, :].astype(jnp.float32)
    y_ref[...] = collapsed.astype(y_ref.dtype)


def _run(
    x_streams,
    hc_fn,
    hc_scale,
    hc_base,
    *,
    mode: str,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    block_tokens: int | None,
    interpret: bool | None,
    dot_precision,
):
    hc = hc_mult
    if x_streams.ndim != 3:
        raise ValueError(f"x_streams must be [N, hc, d], got {x_streams.shape}")
    n, hc_in, d = x_streams.shape
    if hc_in != hc:
        raise ValueError(f"x_streams axis 1 must be hc_mult={hc}, got {hc_in}")
    rows = hc if mode == "head" else mix_hc_width(hc)
    if hc_fn.shape != (rows, hc * d):
        raise ValueError(f"hc_fn must be [{rows}, {hc * d}] for mode={mode!r}, got {hc_fn.shape}")
    if hc_base.shape != (rows,):
        raise ValueError(f"hc_base must be [{rows}], got {hc_base.shape}")
    n_scale = 1 if mode == "head" else 3
    if hc_scale.size != n_scale:
        raise ValueError(f"hc_scale must have {n_scale} entries, got {hc_scale.shape}")
    if mode not in ("head", "pre"):
        raise ValueError(f"mode must be 'head' or 'pre', got {mode!r}")
    if mode == "pre" and sinkhorn_iters < 1:
        raise ValueError(f"sinkhorn_iters must be >= 1, got {sinkhorn_iters}")

    # VMEM scales with hc*d, so wider models use smaller token blocks.
    if block_tokens is None:
        block_tokens = select_collapse_block_tokens(
            _device_kind(),
            tokens=n,
            hc_mult=hc,
            hidden=d,
            activation_bytes=jnp.dtype(x_streams.dtype).itemsize,
            highest_precision=dot_precision == jax.lax.Precision.HIGHEST,
        )
    bt = max(8, int(block_tokens))
    n_pad = -(-n // bt) * bt
    if n_pad != n:
        x_streams = jnp.pad(x_streams, ((0, n_pad - n), (0, 0), (0, 0)))

    # Four entries give the scale vector a valid VMEM layout.
    scale_p = (
        jnp.zeros((4,), jnp.float32)
        .at[:n_scale]
        .set(hc_scale.astype(jnp.float32).reshape(-1)[:n_scale])
        .reshape(4, 1)
    )

    if interpret is None:
        interpret = get_interpret()

    out_shape = [jax.ShapeDtypeStruct((n_pad, d), x_streams.dtype)]
    out_specs = [pl.BlockSpec((bt, d), lambda i: (i, 0))]
    if mode == "pre":
        out_shape += [
            jax.ShapeDtypeStruct((n_pad, rows), jnp.float32),
        ]
        out_specs += [
            pl.BlockSpec((bt, rows), lambda i: (i, 0)),
        ]

    kernel = functools.partial(
        _collapse_kernel,
        hc=hc,
        d=d,
        mode=mode,
        hc_eps=float(hc_eps),
        norm_eps=float(norm_eps),
        dot_precision=dot_precision,
    )
    in_specs = [
        pl.BlockSpec((bt, hc, d), lambda i: (i, 0, 0)),
        pl.BlockSpec((rows, hc * d), lambda i: (0, 0)),
        pl.BlockSpec((4, 1), lambda i: (0, 0)),
        pl.BlockSpec((rows, 1), lambda i: (0, 0)),
    ]
    operands = [
        x_streams,
        hc_fn.astype(jnp.float32),
        scale_p,
        hc_base.astype(jnp.float32).reshape(rows, 1),
    ]
    res = pl.pallas_call(
        kernel,
        grid=(n_pad // bt,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        name=f"mhc-collapse-{mode}",
    )(*operands)

    if mode == "head":
        y = res if not isinstance(res, (tuple, list)) else res[0]
        return y[:n]
    y, mixes = res
    mixes = mixes[:n]
    # Sinkhorn uses a 2048-token tile; collapse is limited near 128 by projection.
    post, comb = mhc_gates(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc,
        sinkhorn_iters=sinkhorn_iters,
        eps=hc_eps,
    )
    return y[:n], post, comb


@functools.partial(
    jax.jit,
    static_argnames=(
        "hc_mult",
        "norm_eps",
        "hc_eps",
        "block_tokens",
        "interpret",
        "dot_precision",
    ),
)
def mhc_head_collapse_fused(
    x_streams: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
    block_tokens: int | None = None,
    interpret: bool | None = None,
    dot_precision=jax.lax.Precision.DEFAULT,
):
    if x_streams.ndim < 3:
        raise ValueError(f"x_streams must be [..., hc, d], got {x_streams.shape}")
    outer_shape = x_streams.shape[:-2]
    hidden = x_streams.shape[-1]
    # Flattening leading dimensions is a TPU layout bitcast.
    x_flat = x_streams.reshape(-1, x_streams.shape[-2], hidden)
    output = _run(
        x_flat,
        hc_fn,
        hc_scale,
        hc_base,
        mode="head",
        hc_mult=hc_mult,
        sinkhorn_iters=1,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        block_tokens=block_tokens,
        interpret=interpret,
        dot_precision=dot_precision,
    )
    return output.reshape(*outer_shape, hidden)


@functools.partial(
    jax.jit,
    static_argnames=(
        "hc_mult",
        "sinkhorn_iters",
        "norm_eps",
        "hc_eps",
        "block_tokens",
        "interpret",
        "dot_precision",
    ),
)
def mhc_pre_fused(
    x_streams: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    block_tokens: int | None = None,
    interpret: bool | None = None,
    dot_precision=jax.lax.Precision.DEFAULT,
):
    if x_streams.ndim < 3:
        raise ValueError(f"x_streams must be [..., hc, d], got {x_streams.shape}")
    outer_shape = x_streams.shape[:-2]
    hidden = x_streams.shape[-1]
    x_flat = x_streams.reshape(-1, x_streams.shape[-2], hidden)
    y, post, comb = _run(
        x_flat,
        hc_fn,
        hc_scale,
        hc_base,
        mode="pre",
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        block_tokens=block_tokens,
        interpret=interpret,
        dot_precision=dot_precision,
    )
    return (
        y.reshape(*outer_shape, hidden),
        post.reshape(*outer_shape, hc_mult),
        comb.reshape(*outer_shape, hc_mult, hc_mult),
    )


def _expand(x, res, post, comb, *, precision):
    """y[t,j,d] = post[t,j] * x[t,d] + sum_i comb[t,i,j] * res[t,i,d]."""
    mixed = jax.lax.dot_general(
        comb,
        res.astype(jnp.float32),
        (((1,), (1,)), ((0,), (0,))),
        precision=precision,
        preferred_element_type=jnp.float32,
    )  # [BT, hc, d]
    return post[:, :, None] * x[:, None, :].astype(jnp.float32) + mixed


def _post_kernel(x_ref, res_ref, post_ref, comb_ref, y_ref, *, precision):
    y = _expand(x_ref[...], res_ref[...], post_ref[...], comb_ref[...], precision=precision)
    y_ref[...] = y.astype(y_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_tokens",
        "backend",
        "interpret",
        "precision",
    ),
)
def mhc_post_fused(
    x: jax.Array,
    residual_streams: jax.Array,
    post: jax.Array,
    comb: jax.Array,
    *,
    block_tokens: int | None = None,
    backend: str = "auto",
    interpret: bool | None = None,
    precision=jax.lax.Precision.DEFAULT,
):
    """Use XLA while its live set fits VMEM, then aligned Pallas tiles."""
    if residual_streams.ndim != 3:
        raise ValueError(f"residual_streams must be [N, hc, d], got {residual_streams.shape}")
    n, hc, d = residual_streams.shape
    if x.shape != (n, d):
        raise ValueError(f"x must be [{n}, {d}], got {x.shape}")
    if post.shape != (n, hc):
        raise ValueError(f"post must be [{n}, {hc}], got {post.shape}")
    if comb.shape != (n, hc, hc):
        raise ValueError(f"comb must be [{n}, {hc}, {hc}], got {comb.shape}")

    if backend not in ("auto", "xla", "pallas"):
        raise ValueError(f"backend must be 'auto', 'xla', or 'pallas', got {backend!r}")
    selected_block_tokens = None
    if backend == "auto":
        selected_block_tokens = select_post_block_tokens(
            _device_kind(),
            tokens=n,
            hc_mult=hc,
            hidden=d,
            x_bytes=jnp.dtype(x.dtype).itemsize,
            residual_bytes=jnp.dtype(residual_streams.dtype).itemsize,
        )
        backend = select_post_backend(
            _device_kind(),
            tokens=n,
            hc_mult=hc,
            hidden=d,
            activation_bytes=jnp.dtype(residual_streams.dtype).itemsize,
            pallas_block_tokens=selected_block_tokens,
        )
    if backend == "xla":
        return _expand(
            x,
            residual_streams,
            post.astype(jnp.float32),
            comb.astype(jnp.float32),
            precision=precision,
        ).astype(x.dtype)

    if block_tokens is None:
        block_tokens = selected_block_tokens or select_post_block_tokens(
            _device_kind(),
            tokens=n,
            hc_mult=hc,
            hidden=d,
            x_bytes=jnp.dtype(x.dtype).itemsize,
            residual_bytes=jnp.dtype(residual_streams.dtype).itemsize,
        )
    bt = max(8, int(block_tokens))
    n_pad = -(-n // bt) * bt
    if n_pad != n:
        pad = n_pad - n
        x = jnp.pad(x, ((0, pad), (0, 0)))
        residual_streams = jnp.pad(residual_streams, ((0, pad), (0, 0), (0, 0)))
        post = jnp.pad(post, ((0, pad), (0, 0)))
        comb = jnp.pad(comb, ((0, pad), (0, 0), (0, 0)))

    if interpret is None:
        interpret = get_interpret()

    y = pl.pallas_call(
        functools.partial(_post_kernel, precision=precision),
        grid=(n_pad // bt,),
        in_specs=[
            pl.BlockSpec((bt, d), lambda i: (i, 0)),
            pl.BlockSpec((bt, hc, d), lambda i: (i, 0, 0)),
            pl.BlockSpec((bt, hc), lambda i: (i, 0)),
            pl.BlockSpec((bt, hc, hc), lambda i: (i, 0, 0)),
        ],
        out_specs=pl.BlockSpec((bt, hc, d), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((n_pad, hc, d), x.dtype),
        interpret=interpret,
        name="mhc-post",
    )(
        x,
        residual_streams,
        post.astype(jnp.float32),
        comb.astype(jnp.float32),
    )
    return y[:n]


__all__ = [
    "mhc_gates",
    "mhc_head_collapse_fused",
    "mhc_post_fused",
    "mhc_pre_fused",
]
