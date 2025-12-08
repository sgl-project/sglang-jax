"""Launch the inference server."""

import os

from sgl_jax.srt.entrypoints import http_server
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils import kill_process_tree


def main():
    """Main entry point for launching the server."""
    try:
        http_server.launch(ServerArgs.from_cli())
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
