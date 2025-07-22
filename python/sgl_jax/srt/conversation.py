"""
Conversation generation utilities.
This is a stub implementation for the migration from sglang.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_chat_conv(
    messages: List[Dict[str, Any]], tokenizer: Any = None, chat_template: Optional[str] = None
) -> str:
    """
    Generate a conversation from chat messages.

    Args:
        messages: List of chat messages with 'role' and 'content' keys
        tokenizer: Tokenizer to use for formatting (optional)
        chat_template: Chat template to use (optional)

    Returns:
        Formatted conversation string
    """
    logger.info(f"Generating chat conversation from {len(messages)} messages")

    # Basic implementation - just concatenate messages
    conversation_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        conversation_parts.append(f"{role}: {content}")

    conversation = "\n".join(conversation_parts)
    logger.debug(f"Generated conversation: {conversation[:100]}...")

    return conversation


class Conversation:
    """Basic conversation class for handling chat interactions."""

    def __init__(self, messages: List[Dict[str, Any]]):
        self.messages = messages
        self.history = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})
        self.history.append({"role": role, "content": content})

    def get_prompt(self) -> str:
        """Get the conversation as a prompt string."""
        return generate_chat_conv(self.messages)

    def clear(self):
        """Clear the conversation history."""
        self.messages.clear()
        self.history.clear()


# A global registry for all conversation templates
chat_templates: Dict[str, Conversation] = {}
matching_function_registry: List[Callable] = []


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in chat_templates, f"{template.name} has been registered."

    chat_templates[template.name] = template


def register_conv_template_matching_function(func):
    matching_function_registry.append(func)


def get_conv_template_by_model_path(model_path):
    for matching_func in matching_function_registry:
        conv_name = matching_func(model_path)
        if conv_name is not None:
            return conv_name
    return None
