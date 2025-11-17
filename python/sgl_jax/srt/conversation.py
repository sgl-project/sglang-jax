"""
Conversation generation utilities.
This is a stub implementation for the migration from sglang.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

logger = logging.getLogger(__name__)


class SeparatorStyle(IntEnum):
    """Separator styles for different models."""
    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    LLAMA3 = auto()
    DEEPSEEK_CHAT = auto()
    METAMATH = auto()
    YUAN2 = auto()
    GEMMA = auto()
    CLLM = auto()
    DEFAULT = auto()


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

    if isinstance(messages, str):
        raise ValueError("The messages should be a list of dict.")
    for message in messages:
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
    prompt = conv.get_prompt()
    logger.debug("Generated conversation: %s...", prompt[:100] if len(prompt) > 100 else prompt)

    return conv


@dataclass
class Conversation:
    """A class that manages a conversation with a specific format."""
    
    name: str
    system_template: str = "{system_message}"
    system_message: str = ""
    roles: tuple[str, str] = ("USER", "ASSISTANT")
    messages: list[tuple[str, str]] = field(default_factory=list)
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str | None = None
    stop_str: str | list[str] | None = None
    
    # Multimodal related
    image_data: list = field(default_factory=list)
    video_data: list = field(default_factory=list)
    audio_data: list = field(default_factory=list)
    modalities: list = field(default_factory=list)
    image_token: str = "<image>"
    audio_token: str = "<audio>"
    video_token: str = "<video>"
    image_token_at_prefix: bool = False

    def copy(self):
        """Create a copy of the conversation."""
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=list(self.messages),
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            image_data=list(self.image_data),
            video_data=list(self.video_data),
            audio_data=list(self.audio_data),
            modalities=list(self.modalities),
            image_token=self.image_token,
            audio_token=self.audio_token,
            video_token=self.video_token,
            image_token_at_prefix=self.image_token_at_prefix,
        )

    def append_message(self, role: str, message: str | None):
        """Append a new message."""
        self.messages.append((role, message))

    def append_image(self, image_url: str, detail: str = "auto"):
        """Append image data."""
        self.image_data.append({"url": image_url, "detail": detail})

    def append_video(self, video_url: str):
        """Append video data."""
        self.video_data.append({"url": video_url})

    def append_audio(self, audio_url: str):
        """Append audio data."""
        self.audio_data.append({"url": audio_url})

    def get_prompt(self) -> str:
        """Get the conversation prompt."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        
        if self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid separator style: {self.sep_style}")

    def clear(self):
        """Clear the conversation history."""
        self.messages.clear()
        self.image_data.clear()
        self.video_data.clear()
        self.audio_data.clear()
        self.modalities.clear()


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