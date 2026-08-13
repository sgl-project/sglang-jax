"""Auto-tuned block configs for fused_moe v2.

Same lookup approach as v1/tuned_block_configs.py but with v2's simpler
5-field FusedMoEBlockConfig (bt, bf, btc, bse, bts).
"""

# ruff: noqa: E501

from __future__ import annotations

import logging

import jax.numpy as jnp

from sgl_jax.srt.utils.jax_utils import get_device_name

from .kernel import FusedMoEBlockConfig

logger = logging.getLogger(__name__)

# Gather-compute overlap is unconditional for num_bt > 1 via the kernel's fixed-K
# rotating gather banks (fixed_gather_banks, default 2): the K-deep sliding window
# keeps gather SMEM/VMEM at O(K) for any token count, so there is no token-count
# gate any more.

# Key (without device_name):
#   (tokens_dtype, weight_dtype, num_tokens, num_experts, top_k,
#    hidden_size, intermediate_size, ep_size, use_shared_expert, use_grouped_topk,
#    enable_act_quant, quant_mode)
# enable_act_quant distinguishes Mode 1 fp8-token from Mode 2 bf16-token, while
# quant_mode distinguishes
# blockwise and per-channel FP8, whose winners differ for GLM-5.2. Lookup falls
# back through the pre-existing keys without quant_mode / enable_act_quant.
#
# Value: (bt, bf, btc, bse, bts)
# fmt: off
TUNED_BLOCK_CONFIGS: dict[str, dict[tuple, tuple[int, ...]]] = {
    "TPU v7": {
        # MiMo V2 Pro: E=384, H=6144, I=2048, top_k=8, fp8 e4m3, ep=32
        # Decode configs (tuned 2026-05-21)
        ('bfloat16', 'float8_e4m3fn', 64, 384, 8, 6144, 2048, 32, False, False): (8, 512, 8, 256, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 384, 8, 6144, 2048, 32, False, False): (8, 512, 8, 256, 8),
        ('bfloat16', 'float8_e4m3fn', 256, 384, 8, 6144, 2048, 32, False, False): (8, 512, 16, 256, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 384, 8, 6144, 2048, 32, False, False): (16, 1024, 32, 256, 32),
        ('bfloat16', 'float8_e4m3fn', 768, 384, 8, 6144, 2048, 32, False, False): (24, 1024, 32, 256, 32),
        # Prefill configs
        ('bfloat16', 'float8_e4m3fn', 2048, 384, 8, 6144, 2048, 32, False, False): (128, 512, 128, 256, None),
        ('bfloat16', 'float8_e4m3fn', 4096, 384, 8, 6144, 2048, 32, False, False): (128, 512, 128, 256, None),
        ('bfloat16', 'float8_e4m3fn', 8192, 384, 8, 6144, 2048, 32, False, False): (128, 1024, 56, 256, 112),
        ('bfloat16', 'float8_e4m3fn', 16384, 384, 8, 6144, 2048, 32, False, False): (256, 1024, 72, 256, 216),
        # MiMo V2 Pro with activation quantization, ep=128
        ('bfloat16', 'float8_e4m3fn', 512, 384, 8, 6144, 2048, 128, False, False, True): (8, 1024, 32, 256, 32),
        ('bfloat16', 'float8_e4m3fn', 768, 384, 8, 6144, 2048, 128, False, False, True): (8, 1024, 48, 256, 48),
        ('bfloat16', 'float8_e4m3fn', 2048, 384, 8, 6144, 2048, 128, False, False, True): (16, 1024, 64, 256, 64),
        ('bfloat16', 'float8_e4m3fn', 3072, 384, 8, 6144, 2048, 128, False, False, True): (24, 1024, 96, 256, 96),
        ('bfloat16', 'float8_e4m3fn', 4096, 384, 8, 6144, 2048, 128, False, False, True): (32, 1024, 64, 256, 128),
        ('bfloat16', 'float8_e4m3fn', 65536, 384, 8, 6144, 2048, 128, False, False, True): (256, 256, 8, 256, 856),
        # MiMo V2 Pro: E=384, H=6144, I=2048, top_k=8, fp8 e4m3, ep=8
        # Decode configs (tuned on bench-4 single-host v7x-16, 2026-05-21)
        ('bfloat16', 'float8_e4m3fn', 512, 384, 8, 6144, 2048, 8, False, False): (64, 1024, 32, 256, 32),
        # GLM-5.2: E=256, H=6144, I=2048, top_k=8, routed FP8 block-wise
        # K=128, ep=16, in-kernel shared expert, act_quant ON, no grouped top-k.
        # Tuned on 8 TPU v7x chips / 16 JAX devices, 2026-08-04
        # (Falcon exp-fvmfgcw2y9).
        # Decode shapes 16..256 were re-swept with random and hot-expert
        # routing on 2026-08-09 (Falcon exp-11xkf5j57c). Existing 32..256
        # entries remained optimal or statistically equivalent; 16 is new.
        ('bfloat16', 'float8_e4m3fn', 16, 256, 8, 6144, 2048, 16, True, False, True): (8, 512, 8, 128, 8),
        # GLM-5.2 ep16 BLOCKWISE / legacy fallback (6-tuple key, no quant_mode).
        # Values from epic's blockwise tune. Production GLM-5.2 uses PER-CHANNEL
        # and hits the 'per_channel' key below; these serve blockwise requests only.
        ('bfloat16', 'float8_e4m3fn', 32, 256, 8, 6144, 2048, 16, True, False, True): (8, 512, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 6144, 2048, 16, True, False, True): (8, 512, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 6144, 2048, 16, True, False, True): (8, 1024, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 6144, 2048, 16, True, False, True): (16, 512, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 6144, 2048, 16, True, False, True): (32, 1024, 16, 1024, 32),
        ('bfloat16', 'float8_e4m3fn', 1024, 256, 8, 6144, 2048, 16, True, False, True): (64, 1024, 64, 1024, 64),
        ('bfloat16', 'float8_e4m3fn', 2048, 256, 8, 6144, 2048, 16, True, False, True): (128, 1024, 64, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 4096, 256, 8, 6144, 2048, 16, True, False, True): (128, 1024, 64, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 6144, 2048, 16, True, False, True): (128, 1024, 64, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 6144, 2048, 16, True, False, True): (256, 512, 64, 512, 256),
        ('bfloat16', 'float8_e4m3fn', 32768, 256, 8, 6144, 2048, 16, True, False, True): (128, 1024, 64, 1024, 128),
        # GLM-5.2 ep16 PER-CHANNEL W8A8 -- the production path. Re-tuned 2026-08-12
        # with the FIXED tuner (corner-seeding + bse-dedup, commit ea52014; Falcon
        # exp-mmjfl55aod/exp-zsm7vwqave), replacing the 2026-08-07 values. Decode
        # self-selects bse=512, prefill btc=128; 16384 = bt128/bf1024/bse1024
        # 3.033ms (-16.6% vs old bt256/bf512/bse512), 32768 clean bench 6.031ms.
        # Tiling-only numerically (bit-identical bt/bts/btc; bse/bf K-dim accum
        # rel_err-equivalent, exp-trkv8umq53/exp-h4zshl4vzw). Tuple=(bt,bf,btc,bse,bts).
        ('bfloat16', 'float8_e4m3fn', 32, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (8, 1024, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (8, 1024, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (8, 512, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (16, 512, 16, 512, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (32, 512, 32, 512, 32),
        ('bfloat16', 'float8_e4m3fn', 1024, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (64, 1024, 64, 1024, 64),
        ('bfloat16', 'float8_e4m3fn', 2048, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (128, 1024, 128, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 4096, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (128, 1024, 128, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (128, 1024, 128, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (128, 1024, 128, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 32768, 256, 8, 6144, 2048, 16, True, False, True, 'per_channel'): (128, 1024, 128, 1024, 128),
        # GLM-5.2 ep16 W8A16 per-channel (activations stay BF16), production C32
        # 128K-prefix / 1K-extend / 1K-decode workload. Re-tuned full 32..32768 on
        # 2026-08-12 with the FIXED tuner (Falcon exp-iqqef5fdtr / exp-sif4qtwfwt),
        # replacing the 2026-08-11 bounded 8-candidate sweep (exp-39vsflxses) that
        # only had tok32/32768 and picked btc=32. The fixed tuner selects btc=128
        # for prefill HERE TOO -- W8A16 does NOT prefer small btc; the old btc=32
        # was a corner-seeding blind-spot, not a W8A16 trait. Tuple=(bt,bf,btc,bse,bts).
        ('bfloat16', 'float8_e4m3fn', 32, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (8, 1024, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (8, 1024, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (8, 1024, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (16, 512, 16, 512, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (32, 1024, 32, 1024, 32),
        ('bfloat16', 'float8_e4m3fn', 1024, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (64, 1024, 64, 512, 64),
        ('bfloat16', 'float8_e4m3fn', 2048, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (128, 1024, 128, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 4096, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (128, 1024, 128, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (128, 1024, 128, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (128, 1024, 128, 1024, 128),
        ('bfloat16', 'float8_e4m3fn', 32768, 256, 8, 6144, 2048, 16, True, False, False, 'per_channel'): (128, 1024, 128, 1024, 128),
        # GLM-5.2 W8A16 per-channel hot buckets for the C64 two-prefix
        # 128K-hit / 1K-extend / 1K-decode workload on EP32. A bounded
        # three-candidate decode sweep selected btc=16; the established
        # 64K extend config measured 11.379 ms. Falcon exp-kyhm3emafb
        # (2026-08-12). The bt=256 extend alternative exceeded v7 VMEM.
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 6144, 2048, 32, True, False, False, 'per_channel'): (8, 512, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 65536, 256, 8, 6144, 2048, 32, True, False, False, 'per_channel'): (128, 1024, 32, 1024, 160),
        # GLM-5.2 ep32 PER-CHANNEL W8A8 -- the production per-channel path on EP32
        # (C64 workload: 64 concurrency, 65536 prefill = 64 x 1K extend). Re-tuned
        # 2026-08-12 with the FIXED tuner (corner-seeding + bse-dedup; Falcon
        # exp-jetp6ikcno 32..4096 / exp-9djm99v09t 8192..65536). Before this, ep32
        # W8A8 fell back to the 6-tuple blockwise key below. Decode self-selects
        # bse=512/1024 (bse-fix); ep32's smaller per-device local rows keep btc<=160
        # -- it does NOT prefer ep16's large-btc regime (fixed-K let 65536 compile
        # where the old num_bt-banks OOM'd). Tuple=(bt,bf,btc,bse,bts).
        ('bfloat16', 'float8_e4m3fn', 32, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (8, 512, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (8, 512, 8, 512, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (8, 512, 16, 512, 16),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (8, 512, 16, 512, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (16, 512, 32, 512, 32),
        ('bfloat16', 'float8_e4m3fn', 1024, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (32, 1024, 64, 1024, 64),
        ('bfloat16', 'float8_e4m3fn', 2048, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (64, 1024, 80, 1024, 80),
        ('bfloat16', 'float8_e4m3fn', 4096, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (64, 1024, 80, 1024, 80),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (128, 1024, 160, 1024, 160),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (128, 1024, 160, 512, 160),
        ('bfloat16', 'float8_e4m3fn', 65536, 256, 8, 6144, 2048, 32, True, False, True, 'per_channel'): (64, 1024, 128, 1024, 128),
        # GLM-5.2: E=256, H=6144, I=2048, top_k=8, routed FP8 block-wise
        # K=128, ep=32, in-kernel shared expert, act_quant ON, no grouped top-k.
        # Tuned on 16 TPU v7x chips, 2026-08-03 (Falcon exp-bkbi8g86uy).
        ('bfloat16', 'float8_e4m3fn', 32, 256, 8, 6144, 2048, 32, True, False, True): (8, 512, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 6144, 2048, 32, True, False, True): (8, 512, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 6144, 2048, 32, True, False, True): (8, 512, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 6144, 2048, 32, True, False, True): (8, 512, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 6144, 2048, 32, True, False, True): (16, 1024, 32, 1024, 32),
        ('bfloat16', 'float8_e4m3fn', 1024, 256, 8, 6144, 2048, 32, True, False, True): (32, 1024, 64, 1024, 64),
        ('bfloat16', 'float8_e4m3fn', 2048, 256, 8, 6144, 2048, 32, True, False, True): (64, 1024, 40, 1024, 80),
        ('bfloat16', 'float8_e4m3fn', 4096, 256, 8, 6144, 2048, 32, True, False, True): (128, 1024, 32, 1024, 160),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 6144, 2048, 32, True, False, True): (128, 1024, 32, 1024, 160),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 6144, 2048, 32, True, False, True): (128, 1024, 32, 1024, 160),
        # 65536@ep32 num_bt=16 -> fixed-K gather banks (K=2) after the interleave_bt
        # switch removal (2026-08-12). Value from the 2026-08-06 broad-tune (Falcon
        # exp-xcd5xzd8tk); fixed-K perf on this ep32 shape is re-verify-pending.
        ('bfloat16', 'float8_e4m3fn', 65536, 256, 8, 6144, 2048, 32, True, False, True): (128, 1024, 32, 1024, 160),
        # Ling 2.6-1T: E=256, H=8192, I=2048, top_k=8, fp8 e4m3 per-channel, ep=32
        # Tuned 2026-05-27
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 8192, 2048, 32, False, True): (8, 256, 8, 256, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 8192, 2048, 32, False, True): (8, 256, 8, 256, 8),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 8192, 2048, 32, False, True): (8, 256, 16, 256, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 8192, 2048, 32, False, True): (16, 1024, 32, 256, 32),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 8192, 2048, 32, False, True): (128, 512, 160, 256, 160),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 8192, 2048, 32, False, True): (128, 512, 80, 256, 160),
        # Ling 2.6-1T with in-kernel shared expert, act_quant ON (Mode 1: fp8 token
        # x fp8 weight). 11-field key (last = enable_act_quant); use_grouped_topk=True
        # because Ling routes with n_group=8. Tuned 2026-05-31. SE-on vs no-SE: same
        # routed shape, bse maxed (per-block SE weight-DMA is the dominant SE cost).
        # act-OFF (W8A16, bf16 token) is a different VMEM regime, not tuned here ->
        # falls back to the legacy key / DEFAULT.
        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 8192, 2048, 32, True, True, True): (8, 256, 8, 128, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 8192, 2048, 32, True, True, True): (8, 512, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 8192, 2048, 32, True, True, True): (8, 256, 16, 128, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 8192, 2048, 32, True, True, True): (16, 1024, 32, 1024, 32),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 8192, 2048, 32, True, True, True): (128, 512, 80, 512, 160),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 8192, 2048, 32, True, True, True): (128, 512, 80, 512, 160),
    },
    "*": {},
}
# fmt: on

DEFAULT_V2_BLOCK_CONFIG = FusedMoEBlockConfig(
    bt=32,
    bf=512,
    btc=32,
    bse=256,
)


def should_interleave_fused_moe_v2_bt(*, num_tokens: int, ep_size: int) -> bool:
    """Deprecated shim: gather overlap is now unconditional for num_bt > 1 via
    the kernel's fixed-K gather banks, so there is no token-count gate. Retained
    only to validate inputs for callers that still probe it; always returns True.
    """
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")
    return True


def get_simplified_key(
    *,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    ep_size: int,
    use_shared_expert: bool,
    use_grouped_topk: bool,
    enable_act_quant: bool = False,
    quant_mode: str | None = None,
) -> tuple:
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")
    if quant_mode not in (None, "none", "blockwise", "per_channel"):
        raise ValueError(f"Unsupported {quant_mode=}")

    device = get_device_name()
    dtype_name = jnp.dtype(dtype).name
    weight_dtype_name = jnp.dtype(weight_dtype).name
    return (
        device,
        dtype_name,
        weight_dtype_name,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        ep_size,
        bool(use_shared_expert),
        bool(use_grouped_topk),
        bool(enable_act_quant),
        quant_mode,
    )


def get_tuned_fused_moe_v2_block_config(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    ep_size: int,
    use_shared_expert: bool = False,
    use_grouped_topk: bool = False,
    enable_act_quant: bool = False,
    quant_mode: str | None = None,
) -> FusedMoEBlockConfig:
    keys = get_simplified_key(
        dtype=dtype,
        weight_dtype=weight_dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=ep_size,
        use_shared_expert=use_shared_expert,
        use_grouped_topk=use_grouped_topk,
        enable_act_quant=enable_act_quant,
        quant_mode=quant_mode,
    )
    device_name = keys[0]
    table_key_quant = keys[1:]
    table_key = table_key_quant[:-1]
    table_key_legacy = table_key[:-1]

    def _lookup(k):
        cfg = None
        if device_name in TUNED_BLOCK_CONFIGS:
            cfg = TUNED_BLOCK_CONFIGS[device_name].get(k)
        if cfg is None:
            cfg = TUNED_BLOCK_CONFIGS.get("*", {}).get(k)
        return cfg

    cfg_tuple = _lookup(table_key_quant)
    if cfg_tuple is None:
        cfg_tuple = _lookup(table_key)
    if cfg_tuple is None:
        cfg_tuple = _lookup(table_key_legacy)

    if cfg_tuple is None:
        return DEFAULT_V2_BLOCK_CONFIG

    if len(cfg_tuple) != 5:
        raise ValueError(f"Unexpected v2 tuned config tuple length: {len(cfg_tuple)}")

    bt, bf, btc, bse, bts = cfg_tuple
    logger.info(
        "Using v2 tuned block config: num_tokens=%d bt=%d bf=%d btc=%d bse=%d bts=%s",
        num_tokens,
        bt,
        bf,
        btc,
        bse,
        bts,
    )

    return FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=bse, bts=bts)
