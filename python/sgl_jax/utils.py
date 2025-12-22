import json
import logging
import traceback
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TypeBasedDispatcher:
    def __init__(self, mapping: list[tuple[type, Callable]]):
        self._mapping = mapping

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                return fn(obj)
        raise ValueError(f"Invalid object: {obj}")


def find_printable_text(text: str) -> str:
    """Find printable text by removing invalid UTF-8 sequences."""
    if not text:
        return text

    # Try to encode/decode to clean up any invalid UTF-8 sequences
    try:
        # This will replace invalid sequences with the replacement character
        return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        # Fallback: just return the original text
        return text


def get_exception_traceback() -> str:
    """Get the current exception traceback as a string."""
    return traceback.format_exc()


def convert_json_schema_to_str(json_schema: dict | str | type[BaseModel]) -> str:
    """Convert a JSON schema to a string.
    Parameters
    ----------
    json_schema
        The JSON schema.
    Returns
    -------
    str
        The JSON schema converted to a string.
    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str


def _create_dummy_buffer(buffer):
    """Create dummy buffer with sequential values, preserving type and sharding."""
    if hasattr(buffer, "value"):
        # It's a Param-wrapped value
        arr = buffer.value
        # Get sharding from the actual array, not the Param wrapper
        sharding = arr.sharding if hasattr(arr, "sharding") else None
        new_arr = jax.device_put(
            jnp.arange(arr.size, dtype=arr.dtype).reshape(arr.shape),
            device=sharding,
        )
        # Re-wrap in the same type (e.g., nnx.Param)
        return type(buffer)(value=new_arr)
    else:
        # It's a raw Array
        sharding = buffer.sharding if hasattr(buffer, "sharding") else None
        new_arr = jax.device_put(
            jnp.arange(buffer.size, dtype=buffer.dtype).reshape(buffer.shape),
            device=sharding,
        )
        return new_arr


def traverse_and_update(state_obj, target_modules):
    """
    Recursively traverse state structure and update A_buffer/B_buffer in target modules.

    Args:
        state_obj: Can be State/Params (dict-like), list, or leaf values (Param, Array, etc.)

    Returns:
        Updated state with same type as input
    """
    if target_modules is None or len(target_modules) == 0:
        return state_obj
    # Case 1: State or Params (dict-like with .items() method, but not a Param leaf node)
    if hasattr(state_obj, "items") and not hasattr(state_obj, "value"):
        updated = {}

        for key, value in state_obj.items():
            if key in ("A_buffer", "B_buffer"):
                # Found a LoRA buffer to replace
                updated[key] = _create_dummy_buffer(value)
            else:
                # Regular key, recurse normally
                updated[key] = traverse_and_update(value, target_modules)

        # Preserve type: return State if input was State, otherwise return same type
        return type(state_obj)(updated)

    # Case 2: List (e.g., layers list)
    elif isinstance(state_obj, list):
        return [traverse_and_update(item, target_modules) for item in state_obj]

    # Case 3: Leaf nodes (Param objects, raw arrays, or other values)
    else:
        return state_obj
