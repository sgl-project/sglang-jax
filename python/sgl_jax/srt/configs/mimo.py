"""Shared MiMo configuration readers for model and host processing."""

from collections.abc import Mapping


def config_value(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def audio_int_list(value, length):
    if isinstance(value, str):
        values = [int(item) for item in value.split("-")]
    elif isinstance(value, int):
        values = [value]
    else:
        values = [int(item) for item in value]
    if len(values) == 1:
        values *= length
    if len(values) != length:
        raise ValueError(f"Expected {length} audio values, got {len(values)}.")
    return values
