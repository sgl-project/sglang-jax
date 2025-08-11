import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import unittest
from contextlib import nullcontext
from typing import Optional, Sequence

import jax
import numpy as np
import psutil
import requests
from jax._src import mesh_utils

from sgl_jax.srt.utils.common_utils import get_bool_env_var, retry

DEFAULT_MODEL_NAME_FOR_TEST = "Qwen/Qwen-7B-Chat"

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 600


def is_in_ci():
    """Return whether it is in CI runner."""
    return get_bool_env_var("SGLANG_IS_IN_CI")


if is_in_ci():
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
        5000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 100
    )
else:
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
        7000 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")[0]) * 100
    )
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"

mesh_axes = [
    "data",  # data parallelism
    "tensor",  # tensor parallelism
    "pipeline",  # pipeline parallelism
    "expert",  # expert parallelism
]


def create_device_mesh(
    ici_parallelism: Sequence[int],
    dcn_parallelism: Sequence[int],
    devices=None,
    num_slices: int = 1,
    allow_split_physical_axes: bool = True,
) -> jax.sharding.Mesh:
    """Create a device mesh"""
    if devices is None:
        devices = jax.devices()

    ici_parallelism = fill_unspecified_parallelism(ici_parallelism, len(devices))
    if num_slices > 1:
        dcn_parallelism = fill_unspecified_parallelism(dcn_parallelism, num_slices)
        devices_array = mesh_utils.create_hybrid_device_mesh(
            ici_parallelism,
            dcn_parallelism,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
        devices_array = mesh_utils.create_device_mesh(
            ici_parallelism,
            devices=devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    mesh = jax.sharding.Mesh(devices_array, mesh_axes)
    return mesh


def fill_unspecified_parallelism(parallelism: Sequence[int], num_devices: int) -> Sequence[int]:
    if -1 not in parallelism:
        return parallelism

    assert parallelism.count(-1) == 1, "At most one axis can be unspecified."
    unspecified_axis_idx = parallelism.index(-1)
    determined_val = num_devices / np.prod(parallelism) * -1
    assert (
        determined_val >= 1 and determined_val.is_integer
    ), "Unspecified value unable to be determined with the given parallelism values"
    parallelism[unspecified_axis_idx] = int(determined_val)
    return parallelism


def jax_trace_context(log_dir: str):
    """Return a JAX trace context manager with options configured via env vars.

    The following environment variables are honored (all optional):

    1. ``JAX_TRACE_CREATE_PERFETTO_LINK`` – Boolean-like string (``1``, ``0``). Controls ``create_perfetto_link``.

    Example::

        os.environ["JAX_TRACE_HOST_TRACER_LEVEL"] = "2"
        with jax_trace_context("/tmp/trace"):
            ...  # code to profile
    """

    jax_trace_enabled = os.getenv("ENABLE_JAX_TRACE", "1")
    if jax_trace_enabled == "0":
        return nullcontext()

    create_perfetto_link = os.getenv("JAX_TRACE_CREATE_PERFETTO_LINK", "1") == "1"

    return jax.profiler.trace(
        log_dir, create_perfetto_trace=True, create_perfetto_link=create_perfetto_link
    )


class CustomTestCase(unittest.TestCase):
    def _callTestMethod(self, method):
        max_retry = int(os.environ.get("SGLANG_TEST_MAX_RETRY", "1" if is_in_ci() else "0"))
        retry(
            lambda: super(CustomTestCase, self)._callTestMethod(method),
            max_retry=max_retry,
        )


def popen_launch_server(
    model: str,
    base_url: str,
    timeout: float,
    api_key: Optional[str] = None,
    other_args: list[str] = [],
    env: Optional[dict] = None,
    return_stdout_stderr: Optional[tuple] = None,
    device: str = "tpu",
    pd_separated: bool = False,
):
    """Launch a server process with automatic device detection.

    Args:
        device: Device type ("auto", "cuda", "rocm" or "cpu").
                If "auto", will detect available platforms automatically.
    """
    other_args = list(other_args)
    other_args += ["--device", str(device)]

    _, host, port = base_url.split(":")
    host = host[2:]

    module = "sgl_jax.launch_pd_server" if pd_separated else "sgl_jax.launch_server"

    module_argv = [
        "-m",
        module,
        "--model-path",
        model,
        *[str(x) for x in other_args],
    ]

    if pd_separated:
        module_argv.extend(
            [
                "--lb-host",
                host,
                "--lb-port",
                port,
            ]
        )
    else:
        module_argv.extend(
            [
                "--host",
                host,
                "--port",
                port,
            ]
        )

    if api_key:
        module_argv += ["--api-key", api_key]

    command = [sys.executable, *module_argv]

    print(f"command={' '.join(command)}")

    # Merge environment variables, avoid overwriting PATH / PYTHONPATH etc
    env_final = os.environ.copy()
    if env:
        env_final.update(env)

    if return_stdout_stderr:
        process = subprocess.Popen(
            command,
            stdout=return_stdout_stderr[0],
            stderr=return_stdout_stderr[1],
            env=env_final,
            text=True,
        )
    else:
        process = subprocess.Popen(command, stdout=None, stderr=None, env=env_final)

    start_time = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start_time < timeout:

            return_code = process.poll()
            if return_code is not None:
                # Server failed to start (non-zero exit code) or crashed
                raise Exception(
                    f"Server process exited with code {return_code}. "
                    "Check server logs for errors."
                )

            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {api_key}",
                }
                response = session.get(
                    f"{base_url}/health_generate",
                    headers=headers,
                )
                if response.status_code == 200:
                    return process
            except requests.RequestException:
                pass

            return_code = process.poll()
            if return_code is not None:
                raise Exception(
                    f"Server unexpectedly exits ({return_code=}). Usually there will be error logs describing the cause far above this line."
                )

            time.sleep(10)

    kill_process_tree(process.pid)
    raise TimeoutError("Server failed to start within the timeout period.")


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass
