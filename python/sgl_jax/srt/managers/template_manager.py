"""
Centralized template management for chat templates and completion templates.

This module provides a unified interface for managing both chat conversation templates
and code completion templates, eliminating global state and improving modularity.
"""

import logging

from sgl_jax.srt.conversation import get_conv_template_by_model_path

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
        model_path: str,
    ) -> None:
        pass
        """
        Initialize all templates based on provided configuration.

        Args:
            model_path: Path to the model
        """
        self.guess_chat_template_from_model_path(model_path)
