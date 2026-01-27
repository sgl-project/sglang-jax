"""
Multimodal Prompt Builder for audio tasks.

This module provides centralized prompt template management for different
multimodal audio tasks (TTS, ASR, audio understanding).
"""

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

# ASR templates aligned with the official MiMo-Audio training templates.
# The model was trained with these specific prompts, so using them ensures
# the model produces natural text output with proper punctuation.
ASR_ZH_TEMPLATES = [
    "请将这段语音转换为文字",
    "帮我识别这个音频文件中的内容",
    "把这段录音转成文本",
    "请帮我把这段语音转为文字记录",
    "将这段音频内容转换成书面文字",
    "请对这段录音进行文字识别",
    "帮我把这段话从语音转为文本",
    "请将这段口述内容转成文字",
    "把这段声音文件转换为文字版本",
    "请帮我将这段语音消息转为文本",
]

ASR_EN_TEMPLATES = [
    "Please transcribe this audio file",
    "Convert this speech recording to text",
    "Transcribe the following voice message",
    "Turn this audio into readable text",
    "Write out the spoken words from this recording",
    "Please provide a text version of this audio",
    "Transcribe the speech in this audio file",
    "Convert the spoken content to written text",
    "Please type out what is said in this recording",
    "Create a written transcript of this audio",
    "Translate this voice recording into text form",
    "Please convert this audio message to text",
]


class MultimodalPromptBuilder:
    """Builder for multimodal audio task prompts.

    This class encapsulates prompt construction logic for different audio tasks,
    ensuring consistent formatting and making templates easy to maintain and test.
    """

    # Task-specific prompt templates
    TEMPLATES = {
        "asr": {
            "prefix": "<|im_start|>user\n",
            "suffix": "{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n",
            "default_prompt": None,  # Uses random template from ASR_ZH/EN_TEMPLATES
        },
        "tts": {
            "prefix": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "default_prompt": "convert text to speech",
        },
        "audio_understanding": {
            "prefix": "<|im_start|>user\n",
            "suffix": "{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n",
            "default_prompt": "identify the content in the audio",
        },
    }

    def __init__(self, tokenizer: Any = None):
        """Initialize the prompt builder.

        Args:
            tokenizer: Optional tokenizer for encoding prompts into token IDs.
        """
        self.tokenizer = tokenizer

    def build_asr_prompt(self, user_prompt: str | None = None) -> tuple[str, str, str]:
        """Build prompt for ASR (Automatic Speech Recognition) task.

        ASR format: [prefix] + [audio] + [suffix]
        - prefix: "<|im_start|>user\n"
        - audio: (inserted by scheduler)
        - suffix: "{prompt}<|im_end|>\n<|im_start|>assistant\n"

        Args:
            user_prompt: Optional user-provided instruction prompt.

        Returns:
            Tuple of (prefix_text, prompt_text, suffix_text)
        """
        template = self.TEMPLATES["asr"]
        prompt_text = (
            user_prompt if user_prompt else random.choice(ASR_ZH_TEMPLATES + ASR_EN_TEMPLATES)
        )
        prefix_text = template["prefix"]
        suffix_text = template["suffix"].format(prompt=prompt_text)

        logger.debug(
            "ASR prompt: prefix=%r, prompt=%r, suffix=%r", prefix_text, prompt_text, suffix_text
        )
        return prefix_text, prompt_text, suffix_text

    def build_tts_prompt(
        self, text: str, instructions: str | None = None
    ) -> tuple[str | None, str]:
        """Build prompt for TTS (Text-to-Speech) task.

        TTS format: [optional instructions] + [text to synthesize]

        Args:
            text: Text to synthesize into speech.
            instructions: Optional voice/style instructions.

        Returns:
            Tuple of (instructions_text, text)
        """
        logger.debug("TTS prompt: text=%r, instructions=%r", text, instructions)
        return instructions, text

    def build_audio_understanding_prompt(
        self, question: str, user_prompt: str | None = None
    ) -> tuple[str, str, str]:
        """Build prompt for audio understanding task.

        Audio understanding format: [prefix] + [audio] + [question]
        - prefix: "<|im_start|>user\n"
        - audio: (inserted by scheduler with SOSP/EOSP markers)
        - question: "{prompt}<|im_end|>\n<|im_start|>assistant\n"

        Args:
            question: Question about the audio.
            user_prompt: Optional context/instruction.

        Returns:
            Tuple of (prefix_text, prompt_text, suffix_text)
        """
        template = self.TEMPLATES["audio_understanding"]
        prompt_text = user_prompt or template["default_prompt"]
        prefix_text = template["prefix"]
        suffix_text = template["suffix"].format(prompt=question or prompt_text)

        logger.debug(
            "Audio understanding prompt: prefix=%r, prompt=%r, suffix=%r",
            prefix_text,
            prompt_text,
            suffix_text,
        )
        return prefix_text, prompt_text, suffix_text

    def tokenize_prompt(self, text: str) -> list[int] | None:
        """Tokenize a prompt text into token IDs.

        Args:
            text: Text to tokenize.

        Returns:
            List of token IDs, or None if tokenizer not available.
        """
        if self.tokenizer is None:
            logger.warning("Tokenizer not available, cannot tokenize prompt")
            return None

        try:
            encoded = self.tokenizer(text)
            return encoded["input_ids"]
        except Exception as e:
            logger.error("Failed to tokenize prompt: %s", e)
            return None

    def build_and_tokenize_asr(
        self, user_prompt: str | None = None
    ) -> tuple[list[int] | None, list[int] | None]:
        """Build and tokenize ASR prompt.

        Args:
            user_prompt: Optional user-provided instruction prompt.

        Returns:
            Tuple of (prefix_ids, suffix_ids)
        """
        prefix_text, _, suffix_text = self.build_asr_prompt(user_prompt)

        prefix_ids = self.tokenize_prompt(prefix_text)
        suffix_ids = self.tokenize_prompt(suffix_text)

        if logger.isEnabledFor(logging.DEBUG) and self.tokenizer:
            logger.debug("ASR tokenized prompt:")
            logger.debug("  prefix_text: %r", prefix_text)
            logger.debug("  prefix_ids: %s", prefix_ids)
            if prefix_ids:
                logger.debug("  prefix decoded: %r", self.tokenizer.decode(prefix_ids))
            logger.debug("  suffix_text: %r", suffix_text)
            logger.debug("  suffix_ids: %s", suffix_ids)
            if suffix_ids:
                logger.debug("  suffix decoded: %r", self.tokenizer.decode(suffix_ids))

        return prefix_ids, suffix_ids

    def build_and_tokenize_tts(
        self, text: str, instructions: str | None = None
    ) -> tuple[list[int] | None, list[int] | None]:
        """Build and tokenize TTS prompt.

        Args:
            text: Text to synthesize.
            instructions: Optional voice/style instructions.

        Returns:
            Tuple of (text_ids, instructions_ids)
        """
        instructions, text = self.build_tts_prompt(text, instructions)

        text_ids = self.tokenize_prompt(text) if text else None
        instructions_ids = self.tokenize_prompt(instructions) if instructions else None

        return text_ids, instructions_ids

    def build_and_tokenize_audio_understanding(
        self, question: str, user_prompt: str | None = None
    ) -> tuple[list[int] | None, list[int] | None]:
        """Build and tokenize audio understanding prompt.

        Args:
            question: Question about the audio.
            user_prompt: Optional context/instruction.

        Returns:
            Tuple of (prefix_ids, suffix_ids)
        """
        prefix_text, _, suffix_text = self.build_audio_understanding_prompt(question, user_prompt)

        prefix_ids = self.tokenize_prompt(prefix_text)
        suffix_ids = self.tokenize_prompt(suffix_text)

        return prefix_ids, suffix_ids

    def update_template(self, task: str, template_key: str, template_value: str) -> None:
        """Update a template for a specific task.

        This allows runtime customization of prompt templates.

        Args:
            task: Task name (asr, tts, audio_understanding).
            template_key: Template key to update (prefix, suffix, default_prompt).
            template_value: New template value.

        Raises:
            ValueError: If task or template_key is invalid.
        """
        if task not in self.TEMPLATES:
            raise ValueError(f"Unknown task: {task}")

        if template_key not in self.TEMPLATES[task]:
            raise ValueError(f"Unknown template key '{template_key}' for task '{task}'")

        logger.info(
            "Updating template: task=%s, key=%s, value=%r", task, template_key, template_value
        )
        self.TEMPLATES[task][template_key] = template_value
