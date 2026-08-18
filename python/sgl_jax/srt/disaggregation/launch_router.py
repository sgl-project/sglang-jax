from __future__ import annotations

import argparse
import logging
import sys

from sgl_jax.srt.disaggregation.mini_lb import MiniLoadBalancer
from sgl_jax.srt.disaggregation.router_args import RouterArgs

logger = logging.getLogger("router")


class CustomHelpFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


def launch_router(args: argparse.Namespace | RouterArgs) -> None:
    router_args = args if isinstance(args, RouterArgs) else RouterArgs.from_cli_args(args)
    if not router_args.mini_lb:
        raise RuntimeError("Only --mini-lb is supported in the sgl-jax router")
    if not router_args.pd_disaggregation:
        raise RuntimeError("sgl-jax router requires --pd-disaggregation")
    MiniLoadBalancer(router_args).start()


def parse_router_args(argv: list[str]) -> RouterArgs:
    parser = argparse.ArgumentParser(
        description="""SGL-JAX Router - PD single-entry proxy

Examples:
  python -m sgl_jax.srt.disaggregation.launch_router --pd-disaggregation --mini-lb \\
    --prefill http://prefill1:30100 8998 --decode http://decode1:30200

  python -m sgl_jax.srt.disaggregation.launch_router --pd-disaggregation --mini-lb \\
    --prefill http://127.0.0.1:30100,8998 --decode http://127.0.0.1:30200 \\
    --prefill-bootstrap-host 10.31.173.56
""",
        formatter_class=CustomHelpFormatter,
    )
    RouterArgs.add_cli_args(parser)
    return RouterArgs.from_cli_args(parser.parse_args(argv))


def main() -> None:
    launch_router(parse_router_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
