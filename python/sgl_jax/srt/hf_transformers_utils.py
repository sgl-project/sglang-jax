"""Utilities for Huggingface Transformers."""

import json
import logging
import os
import threading
import warnings
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sgl_jax.srt.configs.bailing_hybrid import BailingHybridConfig
from sgl_jax.srt.configs.gemma4 import Gemma4Config
from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig
from sgl_jax.srt.configs.qwen3_5 import Qwen3_5DenseConfig, Qwen3_5HybridConfig
from sgl_jax.srt.configs.qwen4_exp import Qwen4ExpConfig
from sgl_jax.srt.managers.tiktoken_tokenizer import TiktokenTokenizer
from sgl_jax.srt.utils.common_utils import is_remote_url, lru_cache_frozenset

logger = logging.getLogger(__name__)


_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    cls.model_type: cls
    for cls in [
        BailingHybridConfig,
        KimiLinearConfig,
        Qwen3_5HybridConfig,
        Qwen3_5DenseConfig,
        Qwen4ExpConfig,
        Gemma4Config,
    ]
}

# These configs expose runner-specific fields (hybrid state and head layouts).
# GLM uses the native v5 config; it no longer needs a local placeholder.
for name, cls in _CONFIG_REGISTRY.items():
    AutoConfig.register(name, cls, exist_ok=True)


_UNSET = object()


def download_from_hf(
    model_path: str, allow_patterns: list[str] | None = _UNSET, cache_dir: str | None = None
):
    if os.path.exists(model_path):
        return model_path

    if allow_patterns is _UNSET:
        allow_patterns = ["*.json", "*.bin", "*.model", "*.py", "*.tiktoken", "*.jinja"]
    return snapshot_download(model_path, allow_patterns=allow_patterns, cache_dir=cache_dir)


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    if hasattr(config, "thinker_config"):
        text_config = get_hf_text_config(config.thinker_config)
    else:
        text_config = next(
            (
                getattr(config, name)
                for name in ("text_config", "llm_config", "language_config")
                if getattr(config, name, None) is not None
            ),
            config,
        )
    if text_config is config:
        return config
    assert hasattr(text_config, "num_attention_heads")
    # v5 no longer inherits these fields between composite and text configs.
    for name in ("pad_token_id", "bos_token_id", "eos_token_id", "tie_word_embeddings", "dtype"):
        if hasattr(config, name) and not hasattr(text_config, name):
            setattr(text_config, name, getattr(config, name))
        elif hasattr(text_config, name) and not hasattr(config, name):
            setattr(config, name, getattr(text_config, name))
    return text_config


def apply_model_config_overrides(config: PretrainedConfig, overrides: dict) -> None:
    """Apply overrides without replacing nested HF configs with plain dicts."""
    for key, value in overrides.items():
        current = getattr(config, key, None)
        if isinstance(value, dict) and isinstance(current, PretrainedConfig):
            current.update(value)
        else:
            setattr(config, key, value)


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    model_override_args: dict | None = None,
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = model
        model = Path(model).parent

    config = AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )
    text_config = get_hf_text_config(config=config)

    if isinstance(model, str) and text_config is not None:
        for key, val in text_config.__dict__.items():
            if not hasattr(config, key) and getattr(text_config, key, None) is not None:
                setattr(config, key, val)

    if isinstance(model, str) and config.model_type == "internvl_chat":
        for key, val in config.llm_config.__dict__.items():
            if not hasattr(config, key):
                setattr(config, key, val)

    if config.model_type == "multi_modality":
        config.update({"architectures": ["MultiModalityCausalLM"]})

    if model_override_args:
        apply_model_config_overrides(config, model_override_args)

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    return config


@lru_cache_frozenset(maxsize=32)
def get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    **kwargs,
):
    try:
        return GenerationConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
    except OSError:
        return None


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model configs."""
    text_config = get_hf_text_config(config)
    rope_scaling = getattr(text_config, "rope_parameters", None) or {}
    rope_scaling = rope_scaling.get("full_attention", rope_scaling)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"
_FASTOKENS_PATCHED = False
_FASTOKENS_PATCH_LOCK = threading.Lock()


def _validate_tokenizer_backend(tokenizer_backend: str):
    if tokenizer_backend not in {"huggingface", "fastokens"}:
        raise ValueError(
            "Unsupported tokenizer_backend "
            f"{tokenizer_backend!r}. Expected 'huggingface' or 'fastokens'."
        )


def _ensure_fastokens_patched():
    """Monkey-patch transformers process-wide to use the fastokens backend once."""
    global _FASTOKENS_PATCHED
    if _FASTOKENS_PATCHED:
        return

    with _FASTOKENS_PATCH_LOCK:
        if _FASTOKENS_PATCHED:
            return

        try:
            import fastokens
        except ImportError:
            raise ImportError(
                "The fastokens package is required when tokenizer_backend='fastokens'. "
                "Install it with: pip install 'sglang-jax[fastokens]'"
            ) from None

        fastokens.patch_transformers()
        _FASTOKENS_PATCHED = True
        logger.info("fastokens backend enabled - transformers patched successfully")


def _raise_fastokens_load_error(tokenizer_name: str, error: Exception):
    raise RuntimeError(
        f"fastokens failed to load tokenizer for {tokenizer_name!r}. "
        "This model's tokenizer may not be supported by fastokens — "
        "see https://github.com/crusoecloud/fastokens. "
        "Re-run without --tokenizer-backend=fastokens to use the default backend."
    ) from error


def _restore_checkpoint_tokenizer(tokenizer, model_path, revision=None, **overrides):
    """Preserve checkpoint tokenization when v5 reconstructs a legacy tokenizer."""
    from tokenizers import Tokenizer
    from transformers.utils.hub import cached_file

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return
    files = [
        cached_file(
            model_path,
            name,
            revision=revision,
            local_files_only=True,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        for name in ("tokenizer.json", "tokenizer_config.json")
    ]
    if not all(files):
        return
    with open(files[1]) as f:
        config = json.load(f)
    # These v4 classes accepted BOS/EOS flags; Qwen tokenizers did not.
    legacy_class = config.get("tokenizer_class", "").removesuffix("Fast")
    if legacy_class not in {
        "LlamaTokenizer",
        "CodeLlamaTokenizer",
        "GemmaTokenizer",
        "CohereTokenizer",
    }:
        return
    raw = Tokenizer.from_file(files[0])
    backend = tokenizer.backend_tokenizer
    if type(backend.pre_tokenizer) is not type(raw.pre_tokenizer):
        backend.pre_tokenizer = raw.pre_tokenizer
        backend.decoder = raw.decoder
    for name, default in (("add_bos_token", True), ("add_eos_token", False)):
        value = overrides.get(name, config.get(name))
        # The public setters rebuild the post-processor, including when the
        # default False flag already matches but the saved processor differs.
        setattr(tokenizer, name, default if value is None else value)


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: str | None = None,
    tokenizer_backend: str = "huggingface",
    sub_dir: str = "",
    download_dir: str | None = None,
    **kwargs,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast | TiktokenTokenizer:
    """Gets a tokenizer for the given model name via Huggingface."""
    _validate_tokenizer_backend(tokenizer_backend)
    revision = kwargs.pop("revision", tokenizer_revision)

    if tokenizer_name.endswith(".json"):
        # Tiktoken JSON files use their own backend and do not go through transformers.
        return TiktokenTokenizer(tokenizer_name)

    if tokenizer_backend == "fastokens":
        _ensure_fastokens_patched()

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if tokenizer_name == "mistralai/Devstral-Small-2505":
        tokenizer_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = tokenizer_name
        tokenizer_name = Path(tokenizer_name).parent

    if is_remote_url(tokenizer_name):
        raise ValueError(
            f"Remote URLs are not supported in JAX implementation. "
            f"Please use a local path or HuggingFace model name instead: {tokenizer_name}"
        )
    tokenizer_name = download_from_hf(tokenizer_name, cache_dir=download_dir)
    if sub_dir:
        # Only append sub_dir if it actually exists
        sub_dir_path = tokenizer_name + "/" + sub_dir
        if os.path.isdir(sub_dir_path):
            tokenizer_name = sub_dir_path
        # else: use the root path, tokenizer might be in model root

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )

    except Exception as e:
        if tokenizer_backend == "fastokens":
            _raise_fastokens_load_error(tokenizer_name, e)

        if isinstance(e, TypeError):
            # The LLaMA tokenizer causes a protobuf error in some environments.
            err_msg = (
                "Failed to load the tokenizer. If you are using a LLaMA V1 model "
                f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
                "original tokenizer."
            )
            raise RuntimeError(err_msg) from e

        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if (
            isinstance(e, ValueError)
            and not trust_remote_code
            and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            )
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant slowdown. Consider using a fast tokenizer instead.",
            stacklevel=2,
        )

    _restore_checkpoint_tokenizer(tokenizer, tokenizer_name, revision, **kwargs)
    attach_additional_stop_token_ids(tokenizer)
    return tokenizer


# Some models doesn't have an available processor, e.g.: InternVL
def get_tokenizer_from_processor(processor):
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: str | None = None,
    use_fast: bool | None = True,
    **kwargs,
):
    # pop 'revision' from kwargs if present.
    revision = kwargs.pop("revision", tokenizer_revision)

    config = AutoConfig.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **kwargs,
    )

    # fix: for Qwen2-VL model, inject default 'size' if not provided.
    if config.model_type in {"qwen2_vl"} and "size" not in kwargs:
        kwargs["size"] = {"shortest_edge": 3136, "longest_edge": 1003520}

    if config.model_type not in {"llava", "clip"}:
        kwargs["use_fast"] = use_fast

    processor = AutoProcessor.from_pretrained(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **kwargs,
    )

    tokenizer = get_tokenizer_from_processor(processor)

    _restore_checkpoint_tokenizer(tokenizer, tokenizer_name, revision, **kwargs)
    attach_additional_stop_token_ids(tokenizer)
    return processor


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set([tokenizer.get_added_vocab()["<|eom_id|>"]])
    else:
        tokenizer.additional_stop_token_ids = None


def check_gguf_file(model: str | os.PathLike) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"
