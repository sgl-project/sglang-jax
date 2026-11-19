"""Bit-exactness of our fake-quant ops vs the reference stand-ins, on identical inputs."""

import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, ".")
import jax.numpy as jnp
import kernel as K  # reference stand-ins (torch)
import model_ref as J
import numpy as np
import torch

rng = np.random.default_rng(0)


def run(name, shape, scale, tfn, jfn):
    x = (rng.standard_normal(shape) * scale).astype(np.float32)
    t = torch.from_numpy(x.copy())
    tfn(t)
    got = np.asarray(jfn(jnp.asarray(x)))
    want = t.numpy()
    mism = (got != want).sum()
    print(
        f"{name:<34} shape={shape} mismatches={mism}/{got.size} max|d|={np.abs(got-want).max():.2e}"
    )


for scale in (0.01, 1.0, 30.0):
    run(
        f"fp8 act_quant  scale={scale}",
        (64, 256),
        scale,
        lambda t: K.act_quant(t, 32, "ue8m0", torch.float8_e8m0fnu, True),
        lambda x: J.act_quant_fp8(x),
    )
    run(
        f"fp4 e8m0 (indexer) scale={scale}",
        (64, 64),
        scale,
        lambda t: K.fp4_act_quant(t, 32, True),
        lambda x: J.act_quant_fp4(x, 32),
    )
    run(
        f"fp4 e4m3 (kv) scale={scale}",
        (64, 128),
        scale,
        lambda t: K.fp4_act_quant(t, 16, True, torch.float8_e4m3fn),
        lambda x: J.act_quant_fp4(x, 16, e4m3_scale=True),
    )
