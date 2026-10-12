"""v6e scheduling for native BF16 resource buffers."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

LANES = pltpu.Tiling.COMPACT.shape[1]


@dataclass(frozen=True)
class CSAAttentionSchedule:
    query_tile: int = 32
    selected_tile: int = 256

    write_run: int = LANES
    decode: bool = False


def get_csa_attention_schedule(device_kind: str, *, decode: bool = False):
    if not any(s in device_kind.lower() for s in ("v6e", "v6 lite")):
        raise ValueError(f"No validated CSA schedule for {device_kind!r}")
    # Decode writes isolated rows; one native BF16 tile avoids padded writer groups.
    native_rows = pltpu.Tiling.COMPACT.shape[0] * (
        jnp.dtype(jnp.float32).itemsize // jnp.dtype(jnp.bfloat16).itemsize
    )
    return CSAAttentionSchedule(
        query_tile=1 if decode else 32,
        write_run=native_rows if decode else LANES,
        decode=decode,
    )
