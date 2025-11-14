"""
Conversation generation utilities.
This is a stub implementation for the migration from sglang.
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# Models in which system adds modality tokens at prompt start automatically
# when media inputs exceed modality tokens in prompt (e.g. 3 images but 2 <image> tokens)
_MODELS_REQUIRING_MODALITY_SUPPLEMENT = {"deepseek-vl2"}

# adapted from https://github.com/vllm-project/vllm/blob/5124f5bf51b83e6f344c1bc6652e8c4d81313b34/vllm/entrypoints/chat_utils.py#L856
def _get_full_multimodal_text_prompt(
    modality_token: str, modality_count: int, text_prompt: str
) -> str:
    """Combine multimodal prompts for a multimodal language model."""

    # For any existing placeholder in the text prompt, we leave it as is
    left: int = modality_count - text_prompt.count(modality_token)
    if left < 0:
        raise ValueError(
            f"Found more '{modality_token}' placeholders in input prompt than "
            "actual multimodal data items."
        )

    # NOTE: For now we always add missing modality_token at the front of
    # the prompt. This may change to be customizable in the future.
    return "\n".join([modality_token] * left + [text_prompt])


def generate_chat_conv(
    messages: list[dict[str, Any]],
    template_name: str
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
    logger.info("Generating chat conversation from %s messages", len(messages))
    
    conv = chat_templates[template_name].copy()
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        image_data=[],
        video_data=[],
        audio_data=[],
        modalities=[],
        image_token=conv.image_token,
        audio_token=conv.audio_token,
        video_token=conv.video_token,
        image_token_at_prefix=conv.image_token_at_prefix,
    )

    if isinstance(request.messages, str):
        raise ValueError("The messages should be a list of dict.")
    for message in request.messages:
        msg_role = message.role
        if msg_role == "system":
            if isinstance(message.content, str):
                conv.system_message = message.content
            elif isinstance(message.content, list):
                if (
                    len(message.content) != 1
                    or getattr(message.content[0], "type", None) != "text"
                ):
                    raise ValueError("The system message should be a single text.")
                else:
                    conv.system_message = getattr(message.content[0], "text", "")
        elif msg_role == "user":
            # Handle the various types of Chat Request content types here.
            if isinstance(message.content, str):
                conv.append_message(conv.roles[0], message.content)
            else:
                real_content = ""
                # calculate number of image_url
                num_image_url = 0
                for content in message.content:
                    if content.type == "image_url":
                        num_image_url += 1
                        conv.modalities.append(content.modalities)
                image_token = (
                    conv.image_token + "\n"
                    if conv.name != "qwen2-vl"
                    else conv.image_token
                )
                add_token_as_needed: bool = (
                    conv.name in _MODELS_REQUIRING_MODALITY_SUPPLEMENT
                )
                if add_token_as_needed:
                    image_token = ""

                audio_token = conv.audio_token
                video_token = conv.video_token
                for content in message.content:
                    if content.type == "text":
                        if num_image_url > 16:
                            real_content += "\n"  # for video
                        real_content += content.text
                    elif content.type == "image_url":
                        # NOTE: works for llava and intervl2_5
                        if conv.image_token_at_prefix:
                            real_content = image_token + real_content
                        else:
                            real_content += image_token
                        conv.append_image(
                            content.image_url.url, content.image_url.detail
                        )
                    elif content.type == "video_url":
                        real_content += video_token
                        conv.append_video(content.video_url.url)
                    elif content.type == "audio_url":
                        real_content += audio_token
                        conv.append_audio(content.audio_url.url)
                if add_token_as_needed:
                    real_content = _get_full_multimodal_text_prompt(
                        conv.image_token, num_image_url, real_content
                    )
                conv.append_message(conv.roles[0], real_content)
        elif msg_role == "assistant":
            parsed_content = ""
            if isinstance(message.content, str):
                parsed_content = message.content
            elif isinstance(message.content, list):
                if (
                    len(message.content) != 1
                    or getattr(message.content[0], "type", None) != "text"
                ):
                    raise ValueError(
                        "The assistant's response should be a single text."
                    )
                else:
                    parsed_content = getattr(message.content[0], "text", "")
            conv.append_message(conv.roles[1], parsed_content)
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    # Add a blank message for the assistant.
    conv.append_message(conv.roles[1], None)
    logger.debug("Generated conversation: %s...", conversation[:100])

    return conv


class Conversation:
    """Basic conversation class for handling chat interactions."""

    def __init__(self, messages: list[dict[str, Any]]):
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
chat_templates: dict[str, Conversation] = {}
matching_function_registry: list[Callable] = []


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

def chat_template_exists(template_name: str) -> bool:
    return template_name in chat_templates
    
# Reference: https://huggingface.co/docs/transformers/main/model_doc/qwen2_vl#usage-example
register_conv_template(
    Conversation(
        name="qwen2-vl",
        system_message="You are a helpful assistant.",
        system_template="<|im_start|>system\n{system_message}",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep="<|im_end|>\n",
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        stop_str=["<|im_end|>"],
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
        video_token="<|vision_start|><|video_pad|><|vision_end|>",
    )
)