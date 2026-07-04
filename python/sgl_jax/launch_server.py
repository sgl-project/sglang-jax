"""Launch the inference server."""

import os

# tpu-raiden's JAX extension must be loaded before JAX/XLA initializes in the
# server process tree; otherwise the native allocator registry can abort on a
# late duplicate registration.
try:
    import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
except Exception:  # noqa: BLE001
    pass

from sgl_jax.srt.entrypoints import http_server
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils import kill_process_tree


def main():
    """Main entry point for launching the server."""
    try:
        server_args = ServerArgs.from_cli()
        if server_args.multimodal:
            from sgl_jax.srt.multimodal.entrypoint import (
                http_server as multimodal_http_server,
            )

            multimodal_http_server.launch(server_args)
        else:
            http_server.launch(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
