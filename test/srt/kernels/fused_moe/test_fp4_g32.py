"""CPU Pallas interpretation: actual FFN loops, with G32 ACC emulated explicitly.

This does not validate TPU lowering, asynchronous execution, or remote DMA.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.fused_moe.v2 import kernel
from sgl_jax.srt.kernels.fused_moe.v2.kernel import _fp4_decode_tile_layout

LUT = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def unpack(q):
    out = np.empty((q.shape[0] * 2, q.shape[1]), np.float32)
    out[::2], out[1::2] = LUT[q & 15], LUT[q >> 4]
    return out


def test_decode_all_codes():
    q = np.tile(np.arange(256, dtype=np.uint8), (128, 1))
    # Unit scales exercise every packed code through the production decoder.
    unit_scales = np.full((8, 256), 127, np.uint8)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(_fp4_decode_tile_layout)(q, unit_scales)).view(np.uint16),
        np.asarray(jnp.asarray(unpack(q), jnp.bfloat16)).view(np.uint16),
    )
    s = np.full((8, 256), 124, np.uint8)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(_fp4_decode_tile_layout)(q, s)).astype(np.float32), unpack(q) / 8
    )


def grouped_reference(x, q, s, group):
    raw = jnp.asarray(unpack(q), jnp.bfloat16)
    w = (
        raw.astype(jnp.float32) * jnp.repeat(jnp.exp2(jnp.asarray(s, jnp.float32) - 127), 32, 0)
    ).astype(jnp.bfloat16)
    return jnp.dot(x, w, preferred_element_type=jnp.float32)


def panel_call(
    group,
    x,
    q1,
    q3,
    q2,
    s1,
    s3,
    s2,
    route,
    previous,
    *,
    valid_f,
    panel_id,
    btc,
    interpret=True,
):
    b, h = x.shape
    f = q1.shape[1]

    def body(xr, a, c, d, sa, sc, sd, rr, prev, out, gate, up, expanded):
        out[...] = prev[...]
        kw = {"interpret": interpret}
        kernel._fp4_gate_up_k256(
            xr,
            gate,
            up,
            a,
            sa,
            c,
            sc,
            expanded,
            hidden=h,
            bts=b,
            btc=btc,
            valid_f=valid_f,
            **kw,
        )
        kernel._fp4_activate_down_k256(
            gate,
            up,
            rr,
            out,
            d,
            sd,
            expanded,
            hidden=h,
            bts=b,
            btc=btc,
            bf_id=panel_id,
            limit=1.0,
            valid_f=valid_f,
            **kw,
        )

    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((b, 2, h // 2), jnp.float32),
        scratch_shapes=(
            pltpu.VMEM((b, f), jnp.float32),
            pltpu.VMEM((b, f), jnp.float32),
            pltpu.VMEM((2, 256, 256), jnp.bfloat16),
        ),
        interpret=interpret,
    )(x.reshape(b, 2, h // 2), q1, q3, q2, s1, s3, s2, route, previous)


@pytest.mark.parametrize(
    "group,f,intermediate",
    [
        (32, 256, 768),
        (32, 512, 768),
        (32, 1024, 1280),
    ],
)
def test_full_ffn_panels_tail_and_multiple_rows(group, f, intermediate):
    rng = np.random.default_rng(17)
    b, h, btc = 32, 1024, 16
    x = jnp.asarray(rng.normal(0, 0.05, (b, h)), jnp.bfloat16)
    qs = [rng.integers(0, 256, (h // 2, intermediate), dtype=np.uint8) for _ in range(2)]
    q2 = rng.integers(0, 256, (intermediate // 2, h), dtype=np.uint8)

    def scales(k, n):
        return rng.integers(120, 125, (k // 32, n), dtype=np.uint8)
        return rng.uniform(0.007, 0.04, ((k + 511) // 512, 1, n)).astype(np.float32)

    ss = [scales(h, intermediate) for _ in range(2)]
    s2 = scales(intermediate, h)
    weights = rng.uniform(0.01, 0.5, (b, 1)).astype(np.float32)
    weights[-1] = 0
    route = jnp.asarray(np.broadcast_to(weights, (b, 128)))
    y = jnp.zeros((b, 2, h // 2), jnp.float32)
    for panel_id, start in enumerate(range(0, intermediate, f)):
        valid = min(f, intermediate - start)
        qp = [np.pad(q[:, start : start + valid], ((0, 0), (0, f - valid))) for q in qs]
        sp = [
            np.pad(
                s[..., start : start + valid],
                [(0, 0)] * (s.ndim - 1) + [(0, f - valid)],
            )
            for s in ss
        ]
        d = np.pad(q2[start // 2 : (start + valid) // 2], ((0, (f - valid) // 2), (0, 0)))
        sc = s2[start // group : (start + valid + group - 1) // group]
        sc = np.pad(sc, [(0, f // group - sc.shape[0])] + [(0, 0)] * (sc.ndim - 1))
        y = panel_call(
            group,
            x,
            *qp,
            d,
            *sp,
            sc,
            route,
            y,
            valid_f=valid,
            panel_id=panel_id,
            btc=btc,
        )
    g = grouped_reference(x, qs[0], ss[0], group).astype(jnp.bfloat16).astype(jnp.float32)
    u = grouped_reference(x, qs[1], ss[1], group).astype(jnp.bfloat16).astype(jnp.float32)
    z = (jax.nn.silu(jnp.minimum(g, 1)) * jnp.clip(u, -1, 1) * weights).astype(jnp.bfloat16)
    expected = grouped_reference(z, q2, s2, group)
    actual = np.asarray(y).reshape(b, h)
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.02, atol=0.002)
    rel = np.linalg.norm(actual - np.asarray(expected)) / max(np.linalg.norm(expected), 1e-12)
    assert rel < 0.01
    np.testing.assert_array_equal(actual[-1], 0)


def _nested_eqns(jaxpr):
    if hasattr(jaxpr, "eqns"):
        for eq in jaxpr.eqns:
            yield eq
            for param in eq.params.values():
                yield from _nested_eqns(param)
    elif hasattr(jaxpr, "jaxpr"):
        yield from _nested_eqns(jaxpr.jaxpr)
    elif isinstance(jaxpr, (list, tuple)):
        for item in jaxpr:
            yield from _nested_eqns(item)
    elif isinstance(jaxpr, dict):
        for item in jaxpr.values():
            yield from _nested_eqns(item)


@pytest.mark.parametrize("group", [32])
@pytest.mark.skipif(not hasattr(pltpu, "ACC"), reason="Native ACC requires JAX 0.11.2")
@pytest.mark.parametrize("row_tile", [16, 32])
def test_device_jaxpr_acc_and_logical_dot_boundaries(group, row_tile, tmp_path):
    # Uses real (not emulated) G32 ACC primitives, without executing a TPU.
    b, m, h, f = 32, row_tile, 5120, 512
    mesh = jax.sharding.AbstractMesh(
        (8, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    a = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
    x = a((b, h), jnp.bfloat16)
    q = a((h // 2, f), jnp.uint8)
    d = a((f // 2, h), jnp.uint8)
    sc = a(
        (h // 32, f),
        jnp.uint8,
    )
    sd = a((f // 32, h), sc.dtype)
    route = a((b, 128), jnp.float32)
    prev = a((b, 2, h // 2), jnp.float32)

    def fun(x, q, d, sc, sd, route, prev):
        return panel_call(
            group,
            x,
            q,
            q,
            d,
            sc,
            sc,
            sd,
            route,
            prev,
            valid_f=f,
            panel_id=0,
            btc=m,
            interpret=False,
        )

    with jax.sharding.use_abstract_mesh(mesh):
        graph = jax.make_jaxpr(fun)(x, q, d, sc, sd, route, prev)
    eqs = list(_nested_eqns(graph))
    pops = [e for e in eqs if e.primitive.name == "matmul_pop"]
    products = [e for e in eqs if e.primitive.name == "matmul_acc_lhs"]
    # Gate/Up: 2 projections * 2 N tiles * 2 rows; Down: 20 N * 2 rows.
    assert len(pops) == (4 + 20) * (b // m)
    assert len(products) == (4 * 20 + 20 * 2) * (b // m)
    for eq in products + pops:
        assert jax.tree.unflatten(eq.params["acc_transforms_tree"], []) == ()
        assert eq.invars[0].aval.shape == (m, 256)
    # Replay actual JAXPR equations through the real Mosaic lowering rules.
    from types import SimpleNamespace

    from jax._src.interpreters import mlir
    from jax._src.lib.mlir import ir
    from jax._src.lib.mlir.dialects import func
    from jax._src.pallas.mosaic import lowering

    with (
        jax.sharding.use_abstract_mesh(mesh),
        mlir.make_ir_context() as ctx,
        ir.Location.unknown(),
    ):
        lowering.tpu.register_dialect(ctx)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            typ = ir.VectorType.get((m, 256), ir.BF16Type.get())
            function = func.FuncOp("actual_acc_eqns", ([typ], []))
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                acc = lowering.AccRef(0, products[0].invars[0].aval.shape, jnp.float32, 0)
                lowering._matmul_acc_lhs_lowering_rule(
                    None, acc, block.arguments[0], **products[0].params
                )
                lowering._matmul_pop_lowering_rule(
                    SimpleNamespace(
                        lowering_context=SimpleNamespace(
                            dynamic_shape_replacement_fn=lambda shape: shape
                        )
                    ),
                    acc,
                    **pops[0].params,
                )
                func.ReturnOp([])
        module.operation.verify()
        (tmp_path / f"acc_lowering_m{m}.mlir").write_text(str(module))


@pytest.mark.parametrize("group", [32])
def test_prepare_padding_preserves_tail_and_metadata(group):
    from sgl_jax.srt.kernels.fused_moe.v2.kernel import prepare_fp4_weights

    e, h, i, f = 2, 1024, 768, 512
    q = jnp.full((e, h // 2, i), 0x21, jnp.uint8)
    d = jnp.full((e, i // 2, h), 0x43, jnp.uint8)
    dtype = jnp.uint8
    sg = (e, h // 32, i)
    sd = (e, i // 32, h)
    s = jnp.ones(sg, dtype)
    ds = jnp.ones(sd, dtype)
    weights, kw = prepare_fp4_weights(q, d, q, s, ds, s, group_size=group, bf=f)
    assert weights[0].shape == (e, h // 2, 1024)
    assert weights[1].shape == (e, 512, h)
    assert kw["logical_intermediate_size"] == 768
    np.testing.assert_array_equal(np.asarray(weights[0])[..., :768], q)
    np.testing.assert_array_equal(np.asarray(weights[0])[..., 768:], 0)
    np.testing.assert_array_equal(np.asarray(weights[1])[:, 384:], 0)
    np.testing.assert_array_equal(np.asarray(kw["w2_scale"])[:, : ds.shape[1]], ds)


def test_input_dma_starts_both_before_any_wait(monkeypatch):

    events = []

    class DMA:
        def __init__(self, source, target, sem):
            assert sem == "shared_byte_counter"
            self.source = source

        def start(self, **kwargs):
            events.append(("start", self.source))

        def wait(self):
            events.append(("wait", self.source))

    monkeypatch.setattr(kernel.pltpu, "make_async_copy", DMA)
    kernel._fp4_load_x_and_route("X", "X_vmem", "route", "route_vmem", "shared_byte_counter")
    assert events == [
        ("start", "X"),
        ("start", "route"),
        ("wait", "X"),
        ("wait", "route"),
    ]


@pytest.mark.parametrize("valid_rows", [16, 21, 32])
def test_tail_mask_preserves_full_tile_and_clears_nan_padding(valid_rows):
    from sgl_jax.srt.kernels.fused_moe.v2.kernel import _fp4_mask_tail_inputs

    b, start = 16, 16
    x = np.arange(b * 2 * 256, dtype=np.float32).reshape(b, 2, 256) / 100
    route = np.ones((b, 128), np.float32)
    valid = valid_rows - start
    x[valid:] = np.nan
    route[valid:] = np.nan

    def kernel(xr, rr, size, outx, outr):
        outx[...] = xr[...]
        outr[...] = rr[...]
        _fp4_mask_tail_inputs(outx, outr, tile_start=jnp.int32(start), valid_rows=size[0])

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(route.shape, jnp.float32),
        ),
        interpret=True,
    )
    ox, orr = call(jnp.asarray(x, jnp.bfloat16), route, jnp.array([valid_rows], jnp.int32))
    ex = np.asarray(jnp.asarray(x, jnp.bfloat16)).copy()
    er = route.copy()
    ex[valid:] = 0
    er[valid:] = 0
    np.testing.assert_array_equal(ox, ex)
    np.testing.assert_array_equal(orr, er)
    # The stores belonging to _fp4_mask_tail_inputs must remain inside a conditional.
    eqs = list(
        _nested_eqns(
            jax.make_jaxpr(call)(
                jnp.asarray(x, jnp.bfloat16), route, jnp.array([valid_rows], jnp.int32)
            )
        )
    )
    conds = [e for e in eqs if e.primitive.name == "cond"]
    assert conds
    stores = lambda branch: sum(
        e.primitive.name in ("swap", "addupdate") for e in _nested_eqns(branch)
    )
    assert any(min(stores(branch) for branch in e.params["branches"]) == 0 for e in conds)


@pytest.mark.parametrize(
    "devices,tokens,bf", [(8, 8192, 512), (8, 16384, 1024), (8, 2304, 512), (16, 4608, 512)]
)
@pytest.mark.parametrize("mxfp4", [False, True])
def test_distributed_entry_abstract_trace(devices, tokens, bf, mxfp4):
    """Trace both the new route and the existing BF16 path; no TPU execution."""
    from sgl_jax.srt.kernels.fused_moe.v2.kernel import (
        FusedMoEBlockConfig,
        fused_ep_moe_v2,
    )

    mesh = jax.sharding.AbstractMesh(
        (devices, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    h, logical_i, experts = 5120, 2304, 384
    storage_i = (logical_i + bf - 1) // bf * bf
    arr = jax.ShapeDtypeStruct
    x = arr((tokens, h), jnp.bfloat16)
    w = arr((experts, h // 2 if mxfp4 else h, storage_i), jnp.uint8 if mxfp4 else jnp.bfloat16)
    d = arr((experts, storage_i // 2 if mxfp4 else storage_i, h), w.dtype)
    s = arr((experts, h // 32, storage_i), jnp.uint8) if mxfp4 else None
    sd = arr((experts, storage_i // 32, h), jnp.uint8) if mxfp4 else None
    route = arr((tokens, 6), jnp.float32)
    ids = arr((tokens, 6), jnp.int32)
    config = FusedMoEBlockConfig(bt=min(512, tokens // devices), bf=bf, btc=64, bts=128, bse=bf)

    def call(x, w, d, s, sd, route, ids):
        return fused_ep_moe_v2(
            mesh,
            x,
            w,
            d,
            w,
            route,
            ids,
            6,
            block_config=config,
            w1_scale=s,
            w2_scale=sd,
            w3_scale=s,
            mxfp4=mxfp4,
            logical_intermediate_size=logical_i if mxfp4 else None,
            swiglu_limit=1.0,
        )

    if mxfp4 and not hasattr(pltpu, "ACC"):
        with jax.sharding.use_abstract_mesh(mesh), pytest.raises(RuntimeError, match="0.11.2"):
            jax.make_jaxpr(call)(x, w, d, s, sd, route, ids)
        return
    with jax.sharding.use_abstract_mesh(mesh):
        graph = jax.make_jaxpr(call)(x, w, d, s, sd, route, ids)
    assert graph.out_avals[0].shape == (tokens, h)
    assert graph.out_avals[0].dtype == jnp.bfloat16


def test_prepare_rejects_unsupported_group_and_rank():
    from sgl_jax.srt.kernels.fused_moe.v2.kernel import prepare_fp4_weights

    q = jnp.zeros((1, 256, 256), jnp.uint8)
    d = jnp.zeros((1, 128, 512), jnp.uint8)
    s = jnp.zeros((1, 16, 256), jnp.uint8)
    sd = jnp.zeros((1, 8, 512), jnp.uint8)
    with pytest.raises(ValueError, match="group32"):
        prepare_fp4_weights(q, d, q, s, sd, s, bf=256, group_size=512)
    with pytest.raises(ValueError, match="shape"):
        prepare_fp4_weights(q[0], d, q, s, sd, s, bf=256)


def test_e8m0_special_values():
    from sgl_jax.srt.kernels.fused_moe.v2.kernel import _fp4_decode_e8m0

    codes = jnp.asarray([0, 1, 126, 127, 128, 254, 255], jnp.uint8)
    actual = np.asarray(jax.jit(_fp4_decode_e8m0)(codes)).view(np.uint32)
    np.testing.assert_array_equal(
        actual,
        np.asarray(
            [0x00400000, 0x00800000, 0x3F000000, 0x3F800000, 0x40000000, 0x7F000000, 0x7FC00000],
            np.uint32,
        ),
    )
