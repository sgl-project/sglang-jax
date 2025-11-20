"""LLGuidance backend for grammar-constrained decoding (JAX-compatible)."""

import json
import logging
import os

import numpy as np
from llguidance import LLMatcher, LLTokenizer, StructTag, grammar_from
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sgl_jax.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    BaseGrammarBackend,
    BaseGrammarObject,
)
from sgl_jax.srt.managers.tiktoken_tokenizer import EOS, TiktokenTokenizer

from .bitmask_ops import allocate_token_bitmask, fill_token_bitmask

logger = logging.getLogger(__name__)


class GuidanceGrammar(BaseGrammarObject):
    """Grammar object using llguidance library."""

    def __init__(
        self,
        llguidance_tokenizer: LLTokenizer,
        serialized_grammar: str,
    ):
        """Initialize a guidance grammar.

        Args:
            llguidance_tokenizer: LLTokenizer instance
            serialized_grammar: Serialized grammar string from llguidance
        """
        super().__init__()
        self.llguidance_tokenizer = llguidance_tokenizer
        self.serialized_grammar = serialized_grammar

        # Create matcher from serialized grammar
        self.ll_matcher = LLMatcher(
            llguidance_tokenizer,
            serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        # Cached bitmask (reused across calls)
        self.bitmask: np.ndarray | None = None

    def accept_token(self, token: int):
        """Accept a token and update the grammar state."""
        if not self.ll_matcher.consume_token(token):
            self.finished = True

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int) -> np.ndarray:
        """Allocate a vocabulary bitmask."""
        if self.bitmask is None or self.bitmask.shape[0] < batch_size:
            self.bitmask = allocate_token_bitmask(batch_size, vocab_size)
            bitmask = self.bitmask
        else:
            bitmask = self.bitmask[:batch_size]
        return bitmask

    def fill_vocab_mask(self, vocab_mask: np.ndarray, idx: int):
        """Fill the vocabulary bitmask for this grammar at batch index."""
        if self.ll_matcher.is_stopped():
            self.finished = True
            return

        n_ll_cols = (int(self.llguidance_tokenizer.vocab_size) + 31) // 32
        sub_mask = vocab_mask[:, :n_ll_cols]

        fill_token_bitmask(
            self.ll_matcher,
            sub_mask,
            idx,
        )

    def is_terminated(self) -> bool:
        """Check if the grammar has terminated."""
        return self.ll_matcher.is_stopped()

    def copy(self):
        return GuidanceGrammar(
            llguidance_tokenizer=self.llguidance_tokenizer,
            serialized_grammar=self.serialized_grammar,
        )


class GuidanceBackend(BaseGrammarBackend):
    """LLGuidance backend implementation (JAX-compatible)."""

    def __init__(
        self,
        tokenizer,
        any_whitespace: bool = True,
        whitespace_pattern: str | None = None,
        n_vocab: int = 0,
        num_threads: int = 4,
    ):
        """Initialize the llguidance backend.

        Args:
            tokenizer: HuggingFace tokenizer
            whitespace_pattern: Pattern for JSON whitespace (default: flexible)
            num_threads: Number of threads for async compilation
        """
        super().__init__(num_threads=num_threads)

        self.any_whitespace = any_whitespace
        self.whitespace_pattern = whitespace_pattern
        if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            from llguidance.hf import from_tokenizer

            self.llguidance_tokenizer = from_tokenizer(tokenizer, n_vocab=n_vocab)
        elif isinstance(tokenizer, TiktokenTokenizer):
            from llguidance.tiktoken import lltokenizer_from_encoding

            # TiktokenTokenizer wraps the underlying tiktoken.Encoding on `.tokenizer`
            encoding = tokenizer.tokenizer
            eos_id = tokenizer.eos_token_id
            if eos_id is None and hasattr(encoding, "_special_tokens"):
                eos_id = encoding._special_tokens.get(EOS)

            self.llguidance_tokenizer = lltokenizer_from_encoding(
                encoding,
                n_vocab=n_vocab,
                eos_token=eos_id,
            )
        else:
            raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)}")

        logger.info("Initialized GuidanceBackend with whitespace_pattern=%s", whitespace_pattern)

    def _from_serialized(self, serialized_grammar) -> GuidanceGrammar:
        try:
            return GuidanceGrammar(
                llguidance_tokenizer=self.llguidance_tokenizer,
                serialized_grammar=serialized_grammar,
            )
        except Exception as e:
            logger.error("Hit invalid grammar: %s, %s", serialized_grammar, e)
            return INVALID_GRAMMAR_OBJ

    def dispatch_json(self, key_string: str) -> GuidanceGrammar:
        try:
            serialized_grammar = LLMatcher.grammar_from_json_schema(
                key_string,
                defaults={
                    "whitespace_flexible": self.any_whitespace,
                    "whitespace_pattern": self.whitespace_pattern,
                },
            )
        except Exception as e:
            logger.error("Hit invalid json_schema: %s, %s", key_string, e)
            return INVALID_GRAMMAR_OBJ
        return self._from_serialized(serialized_grammar)

    def dispatch_regex(self, key_string: str) -> GuidanceGrammar:
        serialized_grammar = grammar_from("regex", key_string)
        return self._from_serialized(serialized_grammar)

    def dispatch_ebnf(self, key_string: str) -> GuidanceGrammar:
        try:
            serialized_grammar = grammar_from("ebnf", key_string)
            return self._from_serialized(serialized_grammar)
        except ValueError as e:
            logger.error("Hit invalid ebnf: %s, %s", key_string, e)
            return INVALID_GRAMMAR_OBJ

    def dispatch_structural_tag(self, key_string: str) -> GuidanceGrammar:
        try:
            structural_tag = json.loads(key_string)
            tags = [
                StructTag(
                    begin=structure["begin"],
                    grammar=structure["schema"],
                    end=structure["end"],
                    trigger=structural_tag["triggers"][0],  # TODO?
                )
                for structure in structural_tag["structures"]
            ]
            g = StructTag.to_grammar(tags)
            return self._from_serialized(g)
        except Exception as e:
            logging.error("Hit invalid structural_tag: %s, %s", key_string, e)
            return INVALID_GRAMMAR_OBJ
