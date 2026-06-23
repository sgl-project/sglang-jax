"""PD multi-host router — single-entry proxy for disaggregated serving.

Provides a MiniLoadBalancer-based HTTP proxy that fans out requests to
one selected (Prefill, Decode) pair, injecting bootstrap fields so the
engines can coordinate KV transfer.

Usage:
    python -m sgl_jax.srt.disaggregation.launch_router \\
        --pd-disaggregation --mini-lb \\
        --prefill http://prefill:30100 8998 \\
        --decode http://decode:30200
"""

from sgl_jax.srt.disaggregation.launch_router import launch_router, parse_router_args
from sgl_jax.srt.disaggregation.mini_lb import MiniLoadBalancer
from sgl_jax.srt.disaggregation.router_args import RouterArgs

__all__ = [
    "MiniLoadBalancer",
    "RouterArgs",
    "launch_router",
    "parse_router_args",
]
