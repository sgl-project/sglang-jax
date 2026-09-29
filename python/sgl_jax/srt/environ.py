"""Central registry for sgl-jax environment variables.

First, intentionally small version of the pattern used by sglang's
python/sglang/srt/environ.py: each variable is declared once here with its
type and default instead of scattering ``os.environ.get`` calls across files.
New environment variables should be added to ``Envs`` below; existing
scattered variables can migrate over time.
"""

import os
from typing import Any


class EnvField:
    def __init__(self, default: Any):
        self.default = default
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError

    def get(self) -> Any:
        value = os.environ.get(self.name)
        if value is None:
            return self.default
        return self.parse(value)


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ("true", "1", "yes", "y"):
            return True
        if value in ("false", "0", "no", "n"):
            return False
        raise ValueError(f"Invalid boolean value for {self.name}: {value!r}")


class Envs:
    # Compute the GPT-J (interleaved) rotary embedding directly in the
    # interleaved domain instead of strided even/odd slices plus a
    # stack+reshape re-interleave. Bit-identical; escape hatch only.
    SGLANG_JAX_ROTARY_INTERLEAVED = EnvBool(True)

    # Merge the rotated rope dims back into the DSA indexer query/key with
    # concatenate instead of a read-modify-write ``at[...].set``. Values are
    # identical; escape hatch only.
    SGLANG_JAX_INDEXER_ROPE_CONCAT = EnvBool(True)


envs = Envs()
