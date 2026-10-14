"""Shared FP4 numerical/layout and panel-DMA primitives; no host side effects."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BK = BN = 256


def decode_e2m1(code):
    code = code.astype(jnp.uint8)
    exponent = (code >> 1) & 3
    mantissa = (code & 1).astype(jnp.float32)
    magnitude = jnp.where(
        exponent == 0,
        mantissa * 0.5,
        jnp.exp2(exponent.astype(jnp.float32) - 1) * (1 + mantissa * 0.5),
    )
    return jnp.where((code & 8) != 0, -magnitude, magnitude)


def decode_e8m0(code):
    """Finite E8M0: 2**(code-127); 255 is NaN, 0 is 2**-127.

    Bit construction avoids an intermediate power overflow; code 0 needs a
    float32 subnormal, not the zero exponent-bit pattern. Device FTZ must be
    validated separately from CPU interpretation.
    """
    code = code.astype(jnp.uint32)
    bits = jnp.where(code == 0, jnp.uint32(0x00400000), code << 23)
    bits = jnp.where(code == 255, jnp.uint32(0x7FC00000), bits)
    return lax.bitcast_convert_type(bits, jnp.float32)


def decode_tile_layout(packed, scales):
    """Construct adjacent BF16 rows in TPU's native second-minor packing.

    Do not stack FP32 low/high planes or create [group,32,N] temporaries.
    Scale multiplication remains FP32 before the BF16 rounding boundary.
    """
    scale = jnp.repeat(decode_e8m0(scales), 16, axis=0)

    def half(code):
        code = code.astype(jnp.uint32)
        mag = code & 7
        bits = jnp.where(
            mag < 2, mag * jnp.uint32(0x3F000000), jnp.uint32(0x3F000000) + (mag << 22)
        )
        bits |= (code & 8) << 28
        value = lax.bitcast_convert_type(bits, jnp.float32) * scale
        return lax.bitcast_convert_type(value.astype(jnp.bfloat16), jnp.uint16).astype(jnp.uint32)

    lo = half(packed & 15)
    hi = half(packed >> 4)
    return pltpu.bitcast(lo | (hi << 16), jnp.bfloat16)


def decode_raw(packed):
    """Expand FP4 to exact unscaled BF16, using native second-minor packing."""

    def half(code):
        code = code.astype(jnp.uint32)
        mag = code & 7
        bits = jnp.where(mag < 2, mag * jnp.uint32(0x3F00), jnp.uint32(0x3F00) + (mag << 6))
        return bits | ((code & 8) << 12)

    return pltpu.bitcast(half(packed & 15) | (half(packed >> 4) << 16), jnp.bfloat16)


def activation(gate, up, route, limit):
    g = gate.astype(jnp.bfloat16).astype(jnp.float32)
    u = up.astype(jnp.bfloat16).astype(jnp.float32)
    if limit is not None and limit > 0:
        g = jnp.minimum(g, limit)
        u = jnp.clip(u, -limit, limit)
    return (jax.nn.silu(g) * u * route).astype(jnp.bfloat16)


def start_panel_dma(
    weight,
    scale,
    packed_panel,
    scale_panel,
    semaphore,
    *,
    expert,
    bf_id,
    down=False,
    priority=1,
    group_size,
):
    # The wrapper pads physical panel capacity; logical I is passed separately.
    width = packed_panel.shape[0] * 2 if down else packed_panel.shape[1]
    if down:
        payload = weight.at[expert, pl.ds(bf_id * width // 2, width // 2), :]
        count = width // group_size
        scales = scale.at[expert, pl.ds(bf_id * count, count), ...]
    else:
        payload = weight.at[expert, :, pl.ds(bf_id * width, width)]
        scales = scale.at[expert, ..., pl.ds(bf_id * width, width)]
    pltpu.make_async_copy(payload, packed_panel, semaphore).start(priority=priority)
    pltpu.make_async_copy(scales, scale_panel, semaphore).start(priority=priority)


def wait_panel_dma(packed_panel, scale_panel, semaphore):
    pltpu.make_async_copy(packed_panel, packed_panel, semaphore).wait()
    pltpu.make_async_copy(scale_panel, scale_panel, semaphore).wait()


def load_x_and_route(x_source, x_target, route_source, route_target, semaphore):
    """Start both independent loads, then drain their total byte count.

    With a shared counting semaphore neither first wait identifies a particular
    transfer. Callers may consume either input only after this function returns.
    """
    x_dma = pltpu.make_async_copy(x_source, x_target, semaphore)
    route_dma = pltpu.make_async_copy(route_source, route_target, semaphore)
    x_dma.start(priority=1)
    route_dma.start()
    x_dma.wait()
    route_dma.wait()


def mask_tail_inputs(x, route, *, tile_start, valid_rows):
    """Leave complete tiles untouched; clear both inputs for invalid tail rows."""
    bts, packing, width = x.shape

    @pl.when(tile_start + bts > valid_rows)
    def tail():
        token_mask = tile_start + lax.broadcasted_iota(jnp.int32, (bts, width), 0) < valid_rows
        for lane in range(packing):
            x.at[:, lane, :][...] = jnp.where(token_mask, x[:, lane, :], 0)
        route_mask = tile_start + lax.broadcasted_iota(jnp.int32, route.shape, 0) < valid_rows
        route[...] = jnp.where(route_mask, route[...], 0)
