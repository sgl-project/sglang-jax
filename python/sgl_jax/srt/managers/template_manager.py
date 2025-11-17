"""
Centralized template management for chat templates and completion templates.

This module provides a unified interface for managing both chat conversation templates
and code completion templates, eliminating global state and improving modularity.
"""

import logging
from typing import Optional

from sgl_jax.srt.conversation import get_conv_template_by_model_path, chat_template_exists

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Centralized manager for chat and completion templates.

    This class encapsulates all template-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for template management.
    """

    def __init__(self):
        pass
        self._chat_template_name: str | None = None
        self._completion_template_name: str | None = None
        self._jinja_template_content_format: str | None = None

    @property
    def chat_template_name(self) -> str | None:
        """Get the current chat template name."""
        return self._chat_template_name

    @property
    def completion_template_name(self) -> str | None:
        """Get the current completion template name."""
        return self._completion_template_name

    @property
    def jinja_template_content_format(self) -> str | None:
        """Get the detected template content format ('string' or 'openai' or None)."""
        return self._jinja_template_content_format

    def guess_chat_template_from_model_path(self, model_path: str) -> None:
        """
        Infer chat template name from model path.

        Args:
            model_path: Path to the model
        """
        template_name = get_conv_template_by_model_path(model_path)
        if template_name is not None:
            logger.info("Inferred chat template from model path: %s", template_name)
            self._chat_template_name = template_name

    def initialize_templates(
        self,
        tokenizer_manager,
        model_path: str,
        chat_template: Optional[str] = None,
    ) -> None:
        """
        Initialize all templates based on provided configuration.

        Args:
            tokenizer_manager: Tokenizer manager
            model_path: Path to the model
            chat_template: Chat template name
        """
        self.load_chat_template(tokenizer_manager, chat_template, model_path)

    def load_chat_template(
        self, tokenizer_manager, chat_template_arg: Optional[str], model_path: str
    ) -> None:
        """
        Load a chat template from various sources.

        Args:
            tokenizer_manager: The tokenizer manager instance
            chat_template_arg: Template name, file path, or None to auto-detect
            model_path: Path to the model
        """
        if chat_template_arg:
            self._load_explicit_chat_template(tokenizer_manager, chat_template_arg)
        else:
            # Guess chat template from model path
            self.guess_chat_template_from_model_path(model_path)

            # If no pre-defined template was found, fallback to HuggingFace template
            if self._chat_template_name is None:
                # Try HuggingFace template first
                hf_template = self._resolve_hf_chat_template(tokenizer_manager)
                if hf_template:
                    # override the chat template
                    if tokenizer_manager.tokenizer:
                        tokenizer_manager.tokenizer.chat_template = hf_template
                    logger.info("Using default HuggingFace chat template")
                else:
                    logger.info("No chat template found")

    def _load_explicit_chat_template(
        self, tokenizer_manager, chat_template_arg: str
    ) -> None:
        """Load explicitly specified chat template."""
        logger.info(f"Loading chat template from argument: {chat_template_arg}")

        if chat_template_exists(chat_template_arg):
            self._chat_template_name = chat_template_arg
            return

    def _resolve_hf_chat_template(self, tokenizer_manager) -> Optional[str]:
        """
        Resolve HuggingFace chat template.

        Returns the chat template string if found, None otherwise.
        """
        try:
            # if processor := tokenizer_manager.processor:
            #     if hasattr(processor, "chat_template") and processor.chat_template:
            #         return processor.chat_template
            if tokenizer := tokenizer_manager.tokenizer:
                if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                    return tokenizer.chat_template
        except Exception as e:
            logger.debug(f"Error getting chat template: {e}")

        logger.debug("No HuggingFace chat template found")
        return None