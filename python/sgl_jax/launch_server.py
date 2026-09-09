"""Launch the inference server."""

import os

from sgl_jax.raiden import preload_raiden_if_requested

preload_raiden_if_requested()


def main():
    """Main entry point for launching the server."""
    from sgl_jax.srt.server_args import ServerArgs
    from sgl_jax.srt.utils import kill_process_tree, set_prometheus_multiproc_dir

    try:
        server_args = ServerArgs.from_cli()

        # prometheus_client reads PROMETHEUS_MULTIPROC_DIR once, when first
        # imported -- and importing an entrypoint pulls it in transitively. Set
        # the directory first, or samples land in a registry /metrics never reads.
        if server_args.enable_metrics:
            set_prometheus_multiproc_dir()

        from sgl_jax.srt.entrypoints import http_server

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
