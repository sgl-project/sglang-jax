"""Opt-in diagnostic launcher for the normal HTTP server, including spawn workers."""

import argparse
import os
import sys
from pathlib import Path


def configure():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--trace-directory", type=Path, required=True)
        parser.add_argument("--sample-pages", type=int, default=2)
        parser.add_argument("server_args", nargs=argparse.REMAINDER)
        args = parser.parse_args()
        if args.sample_pages < 0:
            parser.error("sample-pages must be nonnegative (0 disables device reads)")
        server_args = args.server_args
        if server_args[:1] == ["--"]:
            server_args = server_args[1:]
        os.environ["SGLANG_HYBRID_ACCEPTANCE_TRACE"] = str(args.trace_directory.resolve())
        os.environ["SGLANG_HYBRID_ACCEPTANCE_SAMPLE_PAGES"] = str(args.sample_pages)
        sys.argv = [sys.argv[0], *server_args]

    # A multiprocessing spawn re-executes this script as __mp_main__. Its argv
    # already contains only standard server flags; the opt-in travels via env.
    from sgl_jax.raiden import preload_raiden_if_requested

    preload_raiden_if_requested()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "srt"))
    from hybrid_hicache_observer import install

    install(
        os.environ["SGLANG_HYBRID_ACCEPTANCE_TRACE"],
        sample_pages=int(os.environ["SGLANG_HYBRID_ACCEPTANCE_SAMPLE_PAGES"]),
    )


if __name__ in ("__main__", "__mp_main__"):
    configure()
    if __name__ == "__main__":
        from sgl_jax.launch_server import main

        main()
