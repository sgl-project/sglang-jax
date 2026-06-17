"""Standalone PD bootstrap server entrypoint.

Usage: python -m sgl_jax.srt.disaggregation.run_bootstrap --host 0.0.0.0 --port 8998
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from sgl_jax.srt.disaggregation.bootstrap import BootstrapServer


def main() -> int:
    parser = argparse.ArgumentParser(description="PD bootstrap server (standalone)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )
    parser.add_argument(
        "--shared-secret",
        default=None,
        help="Enable Bearer auth with this secret. "
        "Overridden by SGL_JAX_PD_SHARED_SECRET if set.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    from sgl_jax.srt.disaggregation.pd_auth import resolve_secret

    secret = resolve_secret(args.shared_secret)
    server = BootstrapServer(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        shared_secret=secret,
    )
    server.start()
    logger.info(
        "PD bootstrap server listening on %s:%d (auth=%s)",
        args.host,
        args.port,
        "on" if secret else "off",
    )

    stop = False

    def _on_signal(signum, _frame):
        nonlocal stop
        logger.info("received signal %d, shutting down", signum)
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Park the main thread; the uvicorn server runs in a daemon
    # thread inside ``BootstrapServer``.
    while not stop:
        signal.pause()

    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
