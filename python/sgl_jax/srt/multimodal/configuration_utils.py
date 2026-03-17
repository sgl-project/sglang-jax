import functools
import inspect
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            super().__setattr__(key, value)
        super().__setattr__("_FrozenDict__frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_FrozenDict__frozen", False):
            raise TypeError(f"{self.__class__.__name__} is immutable.")
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        if getattr(self, "_FrozenDict__frozen", False):
            raise TypeError(f"{self.__class__.__name__} is immutable.")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        raise TypeError(f"{self.__class__.__name__} is immutable.")

    def pop(self, key, default=None):
        raise TypeError(f"{self.__class__.__name__} is immutable.")

    def update(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} is immutable.")

    def setdefault(self, key, default=None):
        raise TypeError(f"{self.__class__.__name__} is immutable.")


class ConfigMixin:
    """
    Minimal local config mixin for multimodal JAX components.

    Currently supported:
    - storing init arguments on `self.config` via `register_to_config`
    - immutable config access through `FrozenDict`
    - rebuilding an object from a config dict via `from_config`
    - loading a local JSON config file via `load_config`
    - serializing config to JSON via `to_json_string` / `to_json_file`
    - saving config to a local directory via `save_config`

    Not supported yet:
    - Hugging Face Hub download/upload
    - `return_unused_kwargs` handling compatible with diffusers
    - compatibility-class filtering from diffusers' full `extract_init_dict`
    - deprecation utilities and hidden/legacy config migration logic
    """

    config_name = "config.json"
    ignore_for_config: list[str] = []

    def register_to_config(self, **kwargs):
        kwargs.pop("kwargs", None)
        hidden = {
            key: value
            for key, value in kwargs.items()
            if key not in set(getattr(self, "ignore_for_config", []))
        }
        self._internal_dict = FrozenDict(hidden)

    @property
    def config(self) -> FrozenDict:
        if not hasattr(self, "_internal_dict"):
            raise AttributeError("Config has not been registered yet.")
        return self._internal_dict

    def __getattr__(self, name: str) -> Any:
        if "_internal_dict" in self.__dict__ and name in self.__dict__["_internal_dict"]:
            return self.__dict__["_internal_dict"][name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @classmethod
    def _get_init_keys(cls) -> set[str]:
        return set(dict(inspect.signature(cls.__init__).parameters).keys())

    @classmethod
    def from_config(cls, config: FrozenDict | dict[str, Any], **kwargs):
        if config is None:
            raise ValueError("Please provide a config dictionary.")
        if isinstance(config, FrozenDict):
            config = dict(config)
        if not isinstance(config, dict):
            raise TypeError("`config` must be a dictionary or FrozenDict.")

        expected_keys = cls._get_init_keys()
        expected_keys.discard("self")
        expected_keys.discard("kwargs")
        expected_keys -= set(getattr(cls, "ignore_for_config", []))

        init_dict = {}
        for key in expected_keys:
            if key in kwargs:
                init_dict[key] = kwargs.pop(key)
            elif key in config:
                init_dict[key] = config[key]

        return cls(**init_dict)

    @classmethod
    def load_config(cls, pretrained_model_name_or_path: str | os.PathLike) -> dict[str, Any]:
        config_path = Path(pretrained_model_name_or_path)
        if config_path.is_dir():
            config_path = config_path / cls.config_name
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as reader:
            return json.load(reader)

    def to_json_string(self) -> str:
        if not hasattr(self, "_internal_dict"):
            config_dict: dict[str, Any] = {}
        else:
            config_dict = dict(self._internal_dict)

        config_dict["_class_name"] = self.__class__.__name__

        def _to_jsonable(value: Any):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, type(jnp.dtype("float32"))):
                return str(value)
            if isinstance(value, type):
                return value.__name__
            if type(value).__name__ == "Rngs":
                return None
            return value

        serializable = {}
        for key, value in config_dict.items():
            if key in {"mesh", "precision", "weights_dtype", "quant"}:
                continue
            jsonable = _to_jsonable(value)
            if jsonable is not None:
                serializable[key] = jsonable

        return json.dumps(serializable, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike) -> None:
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def save_config(self, save_directory: str | os.PathLike) -> None:
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
        os.makedirs(save_directory, exist_ok=True)
        self.to_json_file(os.path.join(save_directory, self.config_name))


def register_to_config(init):
    signature = inspect.signature(init)

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()

        init_kwargs = {key: value for key, value in bound.arguments.items() if key != "self"}

        init(self, *args, **kwargs)
        if not isinstance(self, ConfigMixin):
            raise TypeError(
                f"`@register_to_config` requires `{type(self).__name__}` to inherit from ConfigMixin."
            )
        self.register_to_config(**init_kwargs)

    return inner_init