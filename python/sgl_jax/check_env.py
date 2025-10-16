"""Check environment configurations and dependency versions."""

import importlib.metadata
import resource
import sys
from collections import OrderedDict

import jax

# List of packages to check versions
PACKAGE_LIST = [
    "sglang-jax",
    "jax",
    "jaxlib",
    "triton",
    "transformers",
    "numpy",
    "aiohttp",
    "fastapi",
    "huggingface_hub",
    "modelscope",
    "orjson",
    "packaging",
    "psutil",
    "pydantic",
    "python-multipart",
    "pyzmq",
    "uvicorn",
    "uvloop",
    "openai",
    "tiktoken",
]


def get_package_versions(packages):
    """
    Get versions of specified packages.
    """
    versions = {}
    for package in packages:
        package_name = package.split("==")[0].split(">=")[0].split("<=")[0]
        try:
            version = importlib.metadata.version(package_name)
            versions[package_name] = version
        except ModuleNotFoundError:
            versions[package_name] = "Module Not Found"
    return versions


def get_device_info():
    """
    Get TPU-related information if available.
    """
    device_info = {}
    device_list = jax.devices()
    if len(device_list) == 0:
        return {"Device": "no device found"}
    for i, device in enumerate(device_list):
        if "TPU" in device.device_kind:
            device_info[f"[{device.device_kind}-{i}]"] = f"{device}"
        elif "cpu" in device.device_kind:
            device_info[f"[{device.device_kind}-{i}]"] = f"{device}"
        else:
            raise ValueError(f"invalid device kind: {device.device_kind}")

    return device_info


def check_env():
    """
    Check and print environment information.
    """
    env_info = OrderedDict()
    env_info["Python"] = sys.version.replace("\n", "")
    env_info.update(get_package_versions(PACKAGE_LIST))
    env_info.update(get_device_info())

    ulimit_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    env_info["ulimit soft"] = ulimit_soft

    for k, v in env_info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    check_env()
