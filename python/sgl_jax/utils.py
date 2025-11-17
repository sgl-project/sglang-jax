import json
import logging
import traceback
from collections.abc import Callable
from typing import Any

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
