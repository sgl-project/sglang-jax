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

# Key (without device_name):
#   (tokens_dtype, weight_dtype, num_tokens, num_experts, top_k,
#    hidden_size, intermediate_size, ep_size, use_shared_expert, use_grouped_topk)
#
# Value: (bt, bf, btc, bse, bts)
# fmt: off
TUNED_BLOCK_CONFIGS: dict[str, dict[tuple, tuple[int, ...]]] = {
    "TPU v7": {
        # MiMo V2 Pro: E=384, H=6144, I=2048, top_k=8, fp8 e4m3, ep=32
        # Decode configs (tuned via bench_v2.py trace timing)
        ('bfloat16', 'float8_e4m3fn', 64, 384, 8, 6144, 2048, 32, False, False): (8, 512, 8, 256, 8),
        ('bfloat16', 'float8_e4m3fn', 128, 384, 8, 6144, 2048, 32, False, False): (8, 512, 8, 256, 8),
        ('bfloat16', 'float8_e4m3fn', 256, 384, 8, 6144, 2048, 32, False, False): (8, 512, 16, 256, 16),
        ('bfloat16', 'float8_e4m3fn', 512, 384, 8, 6144, 2048, 32, False, False): (16, 1024, 32, 256, 32),
        # Prefill configs
        ('bfloat16', 'float8_e4m3fn', 2048, 384, 8, 6144, 2048, 32, False, False): (128, 512, 128, 256, None),
        ('bfloat16', 'float8_e4m3fn', 4096, 384, 8, 6144, 2048, 32, False, False): (128, 512, 128, 256, None),
        ('bfloat16', 'float8_e4m3fn', 8192, 384, 8, 6144, 2048, 32, False, False): (256, 1024, 72, 256, 216),
        ('bfloat16', 'float8_e4m3fn', 16384, 384, 8, 6144, 2048, 32, False, False): (256, 1024, 72, 256, 216),
    },
    "*": {},
}
# fmt: on

DEFAULT_V2_BLOCK_CONFIG = FusedMoEBlockConfig(
    bt=32, bf=512, btc=32, bse=256,
)


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
) -> tuple:
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")

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
    )
    device_name = keys[0]
    table_key = keys[1:]

    cfg_tuple = None
    if device_name in TUNED_BLOCK_CONFIGS:
        cfg_tuple = TUNED_BLOCK_CONFIGS[device_name].get(table_key)
    if cfg_tuple is None:
        cfg_tuple = TUNED_BLOCK_CONFIGS.get("*", {}).get(table_key)

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
