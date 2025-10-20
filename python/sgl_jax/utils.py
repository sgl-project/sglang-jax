import logging
import traceback
from collections.abc import Callable
from typing import Any

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
