"""Launch the inference server."""

import os
import sys

from sgl_jax.raiden import preload_raiden_if_requested

# Select the host before importing JAX (including indirect ServerArgs imports).
# The compilation target still comes from the requested TPU topology.
if any(arg.split("=", 1)[0] == "--save-aot" for arg in sys.argv[1:]):
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    preload_raiden_if_requested()


def main():
    """Main entry point for launching the server."""
    from sgl_jax.srt.server_args import ServerArgs
    from sgl_jax.srt.utils import kill_process_tree, set_prometheus_multiproc_dir

    try:
        server_args = ServerArgs.from_cli()

        if server_args.save_aot:
            from sgl_jax.srt.model_executor.aot_server import export_server

            export_server(server_args)
            return

        # prometheus_client reads PROMETHEUS_MULTIPROC_DIR once, when first
        # imported -- and importing an entrypoint pulls it in transitively. Set
        # the directory first, or samples land in a registry /metrics never reads.
        if server_args.enable_metrics:
            set_prometheus_multiproc_dir()

        from sgl_jax.srt.entrypoints import http_server

        if server_args.encoder_only:
            from sgl_jax.srt.disaggregation.encoder import server as encoder_server

            encoder_server.launch(server_args)
        elif server_args.multimodal:
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
