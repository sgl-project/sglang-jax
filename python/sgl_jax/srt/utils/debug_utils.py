import functools
import os
from enum import IntEnum


class FrameworkLogLevel(IntEnum):
    ERROR = 0
    WARN = 1
    INFO = 2
    DEBUG = 3
    TRACE = 4


FRAMEWORK_LOG_LEVEL = FrameworkLogLevel(
    int(os.environ.get("SGLANG_FRAMEWORK_LOG_LEVEL", "0"))
)


def print_parameter_shardings(model):
    if FRAMEWORK_LOG_LEVEL < FrameworkLogLevel.DEBUG:
        return
    for name, param in model.named_parameters():
        print(f"{name}: shape={param.value.shape} sharding={param.value.sharding}")


def log_shardings(name):
    def decorator(fn):
        if FRAMEWORK_LOG_LEVEL < FrameworkLogLevel.DEBUG:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for i, a in enumerate(args):
                if hasattr(a, "aval") and hasattr(a.aval, "sharding"):
                    print(f"{name} input[{i}]: {a.aval.shape} {a.aval.sharding}")
            result = fn(*args, **kwargs)
            if hasattr(result, "aval") and hasattr(result.aval, "sharding"):
                print(f"{name} output: {result.aval.shape} {result.aval.sharding}")
            elif isinstance(result, tuple):
                for i, r in enumerate(result):
                    if hasattr(r, "aval") and hasattr(r.aval, "sharding"):
                        print(f"{name} output[{i}]: {r.aval.shape} {r.aval.sharding}")
            return result

        return wrapper

    return decorator
