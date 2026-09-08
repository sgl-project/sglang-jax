"""Guard the ordering prometheus_client's multiprocess backend depends on.

prometheus_client selects its value backend the first time it is imported,
from PROMETHEUS_MULTIPROC_DIR, and never revisits the choice. launch_server
sets that directory before importing an entrypoint, so the client must not be
pulled in while the entrypoint module itself is imported -- otherwise samples
land in a per-process registry that /metrics never reads.
"""

import subprocess
import sys


def _imports_prometheus(module: str) -> bool:
    out = subprocess.run(
        [sys.executable, "-c", f"import {module}, sys; print('prometheus_client' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip() == "True"


def test_launch_server_can_set_multiproc_dir_before_client_is_imported():
    """The arg-parsing path must stay clear of prometheus_client."""
    assert not _imports_prometheus("sgl_jax.launch_server")
    assert not _imports_prometheus("sgl_jax.srt.server_args")
