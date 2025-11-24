"""Base classes for grammar-constrained decoding backends."""

import concurrent.futures as futures
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class BaseGrammarObject:
    """Base class for grammar objects that maintain state during generation."""

    def __init__(self):
        self.finished = False

    def accept_token(self, token: int):
        raise NotImplementedError()

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int):
        raise NotImplementedError()

    def fill_vocab_mask(self, vocab_mask: np.ndarray, idx: int):
        raise NotImplementedError()

    def is_terminated(self) -> bool:
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()


class BaseGrammarBackend:
    """Base class for grammar backends with async compilation support."""

    def __init__(self, num_threads: int = 4):
        """Initialize the grammar backend.

        Args:
            num_threads: Number of threads for async grammar compilation.
        """
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.cache: dict[tuple[str, str], Any] = {}

    def get_cached_or_future_value(self, key: tuple[str, str]) -> tuple[Any, bool]:
        """Get a cached grammar object or submit async compilation.

        Args:
            key: Tuple of (constraint_type, constraint_string)
                 e.g., ("json", schema_str) or ("regex", pattern)

        Returns:
            Tuple of (grammar_object or Future, cache_hit: bool)
        """
        if key in self.cache:
            value = self.cache[key]
            # Check if it's a completed grammar or still a Future
            if isinstance(value, futures.Future):
                return value, False  # Still compiling
            else:
                return value, True  # Cache hit

        # Not in cache, submit async compilation
        key_type, key_string = key
        future = self.executor.submit(self._dispatch, key_type, key_string)
        self.cache[key] = future
        return future, False

    def set_cache(self, key: tuple[str, str], value: BaseGrammarObject):
        """Store a compiled grammar in the cache.

        Args:
            key: Cache key
            value: Compiled grammar object
        """
        self.cache[key] = value

    def reset(self):
        self.cache.clear()

    def _dispatch(self, key_type: str, key_string: str) -> BaseGrammarObject:
        """Dispatch grammar creation based on type.

        Args:
            key_type: Type of constraint ("json", "regex", "ebnf", "structural_tag")
            key_string: Constraint string (JSON schema, regex pattern, etc.)

        Returns:
            Compiled grammar object
        """
        if key_type == "json":
            return self.dispatch_json(key_string)
        elif key_type == "regex":
            return self.dispatch_regex(key_string)
        elif key_type == "ebnf":
            return self.dispatch_ebnf(key_string)
        elif key_type == "structural_tag":
            return self.dispatch_structural_tag(key_string)
        else:
            raise ValueError(f"Unknown constraint type: {key_type}")

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        """Create a grammar from JSON schema.

        Args:
            key_string: JSON schema string

        Returns:
            Grammar object
        """
        raise NotImplementedError()

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        """Create a grammar from regex pattern.

        Args:
            key_string: Regex pattern string

        Returns:
            Grammar object
        """
        raise NotImplementedError()

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        """Create a grammar from EBNF definition.

        Args:
            key_string: EBNF grammar string

        Returns:
            Grammar object
        """
        raise NotImplementedError()

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        """Create a grammar from structural tag configuration.

        Args:
            key_string: JSON string of structural tag config

        Returns:
            Grammar object
        """
        raise NotImplementedError()

    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=False)


# Sentinel object for invalid/failed grammar compilation
INVALID_GRAMMAR_OBJ = BaseGrammarObject()


def create_grammar_backend(
    server_args: ServerArgs,
    tokenizer,
    vocab_size: int,
    eos_token_ids: set | None = None,
) -> BaseGrammarBackend | None:
    name = server_args.grammar_backend

    if name == "llguidance":
        from sgl_jax.srt.constrained.llguidance_backend import get_guidance_backend

        grammar_backend = get_guidance_backend(
            tokenizer=tokenizer,
            num_threads=4,
            n_vocab=vocab_size,
            any_whitespace=not server_args.constrained_json_disable_any_whitespace,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Invalid grammar backend: {name}")

    return grammar_backend
