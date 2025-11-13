"""Common utilities."""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import ipaddress
import itertools
import json
import logging
import os
import random
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import zmq
from fastapi.responses import ORJSONResponse

logger = logging.getLogger(__name__)

PRECOMPILE_DEFAULT_TOKEN_PADDINGS = [1 << i for i in range(6, 14)]
PRECOMPILE_DEFAULT_BS_PADDINGS = [1 << i for i in range(0, 9)]

_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(
                "get_bool_env_var(%s) see non-understandable value=%s and treat as false",
                name,
                value,
            )
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def set_random_seed(seed: int) -> None:
    """Set the random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)


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
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()

    if include_parent:
        with contextlib.suppress(psutil.NoSuchProcess):
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


def set_ulimit(target_soft_limit=65535):
    # number of open files
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning("Fail to set RLIMIT_NOFILE: %s", e)

    # stack size
    resource_type = resource.RLIMIT_STACK
    current_soft, current_hard = resource.getrlimit(resource_type)
    target_soft_limit_stack_size = 1024 * target_soft_limit
    if current_soft < target_soft_limit_stack_size:
        try:
            resource.setrlimit(resource_type, (target_soft_limit_stack_size, current_hard))
        except ValueError as e:
            logger.warning("Fail to set RLIMIT_STACK: %s", e)


def add_api_key_middleware(app, api_key: str):
    @app.middleware("http")
    async def authentication(request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path.startswith("/health"):
            return await call_next(request)
        if request.url.path.startswith("/metrics"):
            return await call_next(request)
        if request.headers.get("Authorization") != "Bearer " + api_key:
            return ORJSONResponse(content={"error": "Unauthorized"}, status_code=401)
        return await call_next(request)


def prepare_model_and_tokenizer(model_path: str, tokenizer_path: str):
    if get_bool_env_var("SGLANG_USE_MODELSCOPE") and not os.path.exists(model_path):
        from modelscope import snapshot_download

        model_path = snapshot_download(model_path)
        tokenizer_path = snapshot_download(
            tokenizer_path, ignore_patterns=["*.bin", "*.safetensors"]
        )
    return model_path, tokenizer_path


def configure_logger(server_args, prefix: str = ""):
    if SGLANG_LOGGING_CONFIG_PATH := os.getenv("SGLANG_LOGGING_CONFIG_PATH"):
        if not os.path.exists(SGLANG_LOGGING_CONFIG_PATH):
            raise Exception(
                f"Setting SGLANG_LOGGING_CONFIG_PATH from env with {SGLANG_LOGGING_CONFIG_PATH} does not exists"
            )
        with open(SGLANG_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())
        logging.config.dictConfig(custom_config)
        return
    format = f"[%(asctime)s{prefix}] %(message)s"
    # format = f"[%(asctime)s.%(msecs)03d{prefix}] %(message)s"
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def get_zmq_socket(context: zmq.Context, socket_type: zmq.SocketType, endpoint: str, bind: bool):
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    socket = context.socket(socket_type)
    if endpoint.find("[") != -1:
        socket.setsockopt(zmq.IPV6, 1)

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in [zmq.PUSH, zmq.PUB]:
        set_send_opt()
    elif socket_type in [zmq.PULL, zmq.SUB]:
        set_recv_opt()
    elif socket_type in [zmq.DEALER, zmq.REP, zmq.REQ]:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    return socket


def delete_directory(dirpath):
    try:
        # This will remove the directory and all its contents
        shutil.rmtree(dirpath)
    except OSError as e:
        print(f"Warning: {dirpath} : {e.strerror}")


def dataclass_to_string_truncated(data, max_length=2048, skip_names: set[str] | None = None):
    if skip_names is None:
        skip_names = set()
    if isinstance(data, str):
        if len(data) > max_length:
            half_length = max_length // 2
            return f"{repr(data[:half_length])} ... {repr(data[-half_length:])}"
        else:
            return f"{repr(data)}"
    elif isinstance(data, (list, tuple)):
        if len(data) > max_length:
            half_length = max_length // 2
            return str(data[:half_length]) + " ... " + str(data[-half_length:])
        else:
            return str(data)
    elif isinstance(data, dict):
        return (
            "{"
            + ", ".join(
                f"'{k}': {dataclass_to_string_truncated(v, max_length)}"
                for k, v in data.items()
                if k not in skip_names
            )
            + "}"
        )
    elif dataclasses.is_dataclass(data):
        fields = dataclasses.fields(data)
        return (
            f"{data.__class__.__name__}("
            + ", ".join(
                f"{f.name}={dataclass_to_string_truncated(getattr(data, f.name), max_length)}"
                for f in fields
                if f.name not in skip_names
            )
            + ")"
        )
    else:
        return str(data)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


def pyspy_dump_schedulers():
    """py-spy dump on all scheduler in a local node."""
    try:
        pid = psutil.Process().pid
        # Command to run py-spy with the PID
        cmd = f"py-spy dump --pid {pid}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        logger.error("Pyspy dump for PID %s:\n%s", pid, result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("Pyspy failed to dump PID %s. Error: %s", pid, e.stderr)


def kill_itself_when_parent_died():
    if sys.platform == "linux":
        # sigkill this process when parent worker manager dies
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    else:
        logger.warning("kill_itself_when_parent_died is only supported in linux.")


def set_uvicorn_logging_configs():
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "[%(asctime)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def launch_dummy_health_check_server(host, port):
    import asyncio

    import uvicorn
    from fastapi import FastAPI, Response

    app = FastAPI()

    @app.get("/health")
    async def health():
        """Check the health of the http server."""
        return Response(status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        """Check the health of the http server."""
        return Response(status_code=200)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_keep_alive=5,
        loop="auto",
        log_config=None,
        log_level="warning",
    )
    server = uvicorn.Server(config=config)

    try:
        loop = asyncio.get_running_loop()
        logger.info("Dummy health check server scheduled on existing loop at %s:%s", host, port)
        loop.create_task(server.serve())

    except RuntimeError:
        logger.info("Starting dummy health check server at %s:%s", host, port)
        server.run()


def is_remote_url(url: str | Path) -> bool:
    """
    Check if the URL is a remote URL of the format:
    <connector_type>://<host>:<port>/<model_name>
    """
    if isinstance(url, Path):
        return False

    pattern = r"(.+)://(.*)"
    m = re.match(pattern, url)
    return m is not None


def retry(
    fn,
    max_retry: int,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    should_retry: Callable[[Any], bool] = lambda e: True,
):
    for try_index in itertools.count():
        try:
            return fn()
        except Exception as e:
            if try_index >= max_retry:
                raise Exception("retry() exceed maximum number of retries.") from e

            if not should_retry(e):
                raise Exception("retry() observe errors that should not be retried.") from e

            delay = min(initial_delay * (2**try_index), max_delay) * (0.75 + 0.25 * random.random())

            logger.warning(
                "retry() failed once (%sth try, maximum %s retries). Will delay %.2fs and retry. Error: %s",
                try_index,
                max_retry,
                delay,
                e,
            )
            traceback.print_exc()

            time.sleep(delay)


def lru_cache_frozenset(maxsize=128):
    def _to_hashable(o):
        try:
            hash(o)
            return o
        except TypeError:
            # Not hashable; convert based on type
            if isinstance(o, (dict)):
                return frozenset((_to_hashable(k), _to_hashable(v)) for k, v in o.items())
            elif isinstance(o, set):
                return frozenset(_to_hashable(v) for v in o)
            elif isinstance(o, (list, tuple)) or (
                isinstance(o, Sequence) and not isinstance(o, (str, bytes))
            ):
                return tuple(_to_hashable(v) for v in o)
            else:
                raise TypeError(f"Cannot make hashable: {type(o)}") from None

    def decorator(func):
        cache = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            h_args = tuple(_to_hashable(a) for a in args)
            h_kwargs = frozenset((_to_hashable(k), _to_hashable(v)) for k, v in kwargs.items())
            key = (h_args, h_kwargs)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        wrapper.cache_clear = cache.clear  # For manual cache clearing
        return wrapper

    return decorator


def cdiv(a, b):
    assert b != 0, f"b is equal to 0, b={b}"
    return (a + b - 1) // b
