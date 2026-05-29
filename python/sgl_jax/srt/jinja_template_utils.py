import logging

from sgl_jax.srt.managers.io_struct import ImageData

"""
Jinja template utilities for processing chat templates.
This is a stub implementation for the migration from sglang.
"""


logger = logging.getLogger(__name__)


def process_content_for_template_format(
    msg_dict: dict,
    content_format: str,
    image_data: list,
    video_data: list,
    audio_data: list,
    modalities: list,
) -> dict:
    """
    Process message content based on detected template format.

    Args:
        msg_dict: Message dictionary with content
        content_format: 'string' or 'openai' (detected via AST analysis)
        image_data: List to append extracted image URLs
        video_data: List to append extracted video URLs
        audio_data: List to append extracted audio URLs
        modalities: List to append modalities

    Returns:
        Processed message dictionary
    """
    if not isinstance(msg_dict.get("content"), list):
        # Already a string or None, no processing needed
        return {k: v for k, v in msg_dict.items() if v is not None}

    if content_format is None:
        # Heuristic fallback when template format detection isn't available.
        # Structured OpenAI-style content parts always include "type".
        content_format = (
            "openai"
            if any(isinstance(chunk, dict) and "type" in chunk for chunk in msg_dict["content"])
            else "string"
        )

    if content_format == "openai":
        # OpenAI format: preserve structured content list, normalize types
        processed_content_parts = []
        for chunk in msg_dict["content"]:
            if not isinstance(chunk, dict):
                if hasattr(chunk, "model_dump"):
                    chunk = chunk.model_dump()
                elif hasattr(chunk, "dict"):
                    chunk = chunk.dict()
            if isinstance(chunk, dict):
                chunk_type = chunk.get("type")

                if chunk_type == "image_url":
                    image_data.append(
                        ImageData(
                            url=chunk["image_url"]["url"],
                            detail=chunk["image_url"].get("detail", "auto"),
                        )
                    )
                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))
                    # Normalize to simple 'image' type for template compatibility
                    processed_content_parts.append({"type": "image"})
                elif chunk_type == "video_url":
                    video_data.append(chunk["video_url"]["url"])
                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))
                    # Normalize to simple 'video' type for template compatibility
                    processed_content_parts.append({"type": "video"})
                elif chunk_type == "audio_url":
                    audio_data.append(chunk["audio_url"]["url"])
                    # Normalize to simple 'audio' type
                    processed_content_parts.append({"type": "audio"})
                else:
                    # Keep other content as-is (text, etc.)
                    processed_content_parts.append(chunk)

        new_msg = {k: v for k, v in msg_dict.items() if v is not None and k != "content"}
        new_msg["content"] = processed_content_parts
        return new_msg

    else:  # content_format == "string"
        # String format: flatten to text only (for templates like DeepSeek)
        text_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text_parts.append(chunk["text"])
            # Note: For string format, we ignore images/audio since the template
            # doesn't expect structured content - multimodal placeholders would
            # need to be inserted differently

        new_msg = msg_dict.copy()
        new_msg["content"] = " ".join(text_parts) if text_parts else ""
        new_msg = {k: v for k, v in new_msg.items() if v is not None}
        return new_msg


def render_template(template: str, **kwargs) -> str:
    """
    Render a Jinja template with the given variables.

    Args:
        template: Template string
        **kwargs: Variables to pass to the template

    Returns:
        Rendered template string
    """
    try:
        from jinja2 import Template

        jinja_template = Template(template)
        return jinja_template.render(**kwargs)
    except ImportError:
        logger.warning("Jinja2 not available, using basic string formatting")
        # Fallback to basic string formatting
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            logger.warning("Template formatting failed, returning original template")
            return template
