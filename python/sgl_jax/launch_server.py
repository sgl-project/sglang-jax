"""Launch the inference server."""

import os
import sys

from sgl_jax.srt.entrypoints.http_server import launch_server
from sgl_jax.srt.server_args import prepare_server_args
from sgl_jax.srt.utils import kill_process_tree


def main():
    """Main entry point for launching the server."""
    try:
        server_args = prepare_server_args(sys.argv[1:])
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
