"""Utilities for Huggingface Transformers."""

import contextlib
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
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sgl_jax.srt.configs.bailing_hybrid import BailingHybridConfig
from sgl_jax.srt.configs.gemma4 import Gemma4Config
from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig
from sgl_jax.srt.configs.qwen3_5 import Qwen3_5DenseConfig, Qwen3_5HybridConfig
from sgl_jax.srt.managers.tiktoken_tokenizer import TiktokenTokenizer
from sgl_jax.srt.utils.common_utils import is_remote_url, lru_cache_frozenset

logger = logging.getLogger(__name__)


class GlmMoeDsaConfig(PretrainedConfig):
    # Empty stub (PR #1037): just claims the model_type; real fields live with
    # the model / stock HF config.
    model_type = "glm_moe_dsa"


_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    cls.model_type: cls
    for cls in [
        BailingHybridConfig,
        KimiLinearConfig,
        GlmMoeDsaConfig,
        Qwen3_5HybridConfig,
        Qwen3_5DenseConfig,
        Gemma4Config,
    ]
}

if "glm_moe_dsa" not in CONFIG_MAPPING:
    with contextlib.suppress(Exception):
        from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

        TOKENIZER_MAPPING._reverse_config_mapping["GlmMoeDsaConfig"] = "gpt2"

# Register local configs; suppress() defers to stock on a name clash (fine for
# bailing/kimi which don't clash, and for the glm stub where stock is preferable).
for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)

# Qwen3.5 is the exception: stock transformers >=5.3 owns ``qwen3_5_moe`` /
# ``qwen3_5`` so the loop skips ours, but ours isn't interchangeable (flattens
# rope_parameters + exposes the hybrid/GDN interface the runner needs). Force
# ours to win for both the MoE and dense root model types.
AutoConfig.register("qwen3_5_moe", Qwen3_5HybridConfig, exist_ok=True)
AutoConfig.register("qwen3_5", Qwen3_5DenseConfig, exist_ok=True)


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
    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    if hasattr(config, "language_config"):
        return config.language_config
    if hasattr(config, "thinker_config"):
        # qwen2.5 omni
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            thinker_config.text_config.torch_dtype = getattr(
                thinker_config, "dtype", getattr(thinker_config, "torch_dtype", None)
            )
            return thinker_config.text_config
        return thinker_config
    else:
        return config


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

    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model, revision=revision)
        # NOTE(HandH1998): Qwen2VL requires `_name_or_path` attribute in `config`.
        config._name_or_path = model

    if isinstance(model, str) and config.model_type == "internvl_chat":
        for key, val in config.llm_config.__dict__.items():
            if not hasattr(config, key):
                setattr(config, key, val)

    if config.model_type == "multi_modality":
        config.update({"architectures": ["MultiModalityCausalLM"]})

    if model_override_args:
        config.update(model_override_args)

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
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
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
    # Workaround: older versions of the transformers library (like ~=4.57.1) will crash when
    # loading tokenizers containing list-format extra_special_tokens (e.g. google/gemma-4).
    # Overriding it to an empty dict avoids the validation crash while keeping all vocab tokens intact.
    try:
        config_path = os.path.join(tokenizer_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                import json

                model_config_data = json.load(f)
                if model_config_data.get("model_type") == "gemma4":
                    kwargs.setdefault("extra_special_tokens", {})
    except Exception as e:
        logger.debug("Failed to inspect config.json for extra_special_tokens workaround: %s", e)
    if sub_dir:
        # Only append sub_dir if it actually exists
        sub_dir_path = tokenizer_name + "/" + sub_dir
        if os.path.isdir(sub_dir_path):
            tokenizer_name = sub_dir_path
        # else: use the root path, tokenizer might be in model root

    # Workaround: Intercept TokenizersBackend and list-type extra_special_tokens
    # to prevent loading failure in transformers < 5.0.
    # TODO(notabee): Clean this workaround up when transformers 5.0 is the minimum version.
    import json
    import tempfile

    import transformers
    from packaging.version import Version

    need_patch = Version(transformers.__version__) < Version("5.0.0")
    tokenizer_config_path = os.path.join(tokenizer_name, "tokenizer_config.json")
    tokenizer_load_path = tokenizer_name
    temp_dir_obj = None

    if need_patch and os.path.exists(tokenizer_config_path):
        try:
            with open(tokenizer_config_path, encoding="utf-8") as f:
                config_data = json.load(f)
            is_patched = False
            if config_data.get("tokenizer_class") == "TokenizersBackend":
                config_data.pop("tokenizer_class", None)
                is_patched = True
            if "extra_special_tokens" in config_data:
                config_data.pop("extra_special_tokens", None)
                is_patched = True

            # Simplify dict-valued special tokens to strings to avoid transformers TypeError
            special_token_keys = [
                "bos_token",
                "eos_token",
                "unk_token",
                "pad_token",
                "sep_token",
                "cls_token",
                "mask_token",
            ]
            for key in special_token_keys:
                if (
                    key in config_data
                    and isinstance(config_data[key], dict)
                    and "content" in config_data[key]
                ):
                    config_data[key] = config_data[key]["content"]
                    is_patched = True

            if "additional_special_tokens" in config_data and isinstance(
                config_data["additional_special_tokens"], list
            ):
                new_additional = []
                for token in config_data["additional_special_tokens"]:
                    if isinstance(token, dict) and "content" in token:
                        new_additional.append(token["content"])
                        is_patched = True
                    else:
                        new_additional.append(token)
                config_data["additional_special_tokens"] = new_additional

            if is_patched:
                # Create a temporary directory and symlink all files from tokenizer_name,
                # writing the patched config inside the temporary directory.
                temp_dir_obj = tempfile.TemporaryDirectory()
                temp_dir = temp_dir_obj.name
                try:
                    for item in os.listdir(tokenizer_name):
                        src_path = os.path.join(tokenizer_name, item)
                        dst_path = os.path.join(temp_dir, item)
                        if item == "tokenizer_config.json":
                            continue
                        os.symlink(src_path, dst_path)

                    patched_config_path = os.path.join(temp_dir, "tokenizer_config.json")
                    with open(patched_config_path, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=2)

                    tokenizer_load_path = temp_dir
                    warnings.warn(
                        f"Created patched tokenizer config workaround in temporary directory: {temp_dir} "
                        "to maintain compatibility with your transformers library version.",
                        stacklevel=2,
                    )
                except Exception as symlink_err:
                    tokenizer_load_path = tokenizer_name
                    temp_dir_obj.cleanup()
                    temp_dir_obj = None
                    warnings.warn(
                        f"Failed to create temporary directory for patched tokenizer config: {symlink_err}. "
                        "Falling back to default tokenizer directory.",
                        stacklevel=2,
                    )
        except Exception as e:
            warnings.warn(
                f"Failed to dynamically patch tokenizer_config.json: {e}",
                stacklevel=2,
            )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
        # Workaround: older transformers versions only read chat_template from tokenizer_config.json
        # and do not search for chat_template.jinja files. Explicitly load it if present in model files.
        try:
            jinja_template_path = os.path.join(tokenizer_name, "chat_template.jinja")
            if os.path.exists(jinja_template_path):
                with open(jinja_template_path) as f:
                    tokenizer.chat_template = f.read()
        except Exception as e:
            logger.debug("Failed to load chat_template.jinja: %s", e)

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
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant slowdown. Consider using a fast tokenizer instead.",
            stacklevel=2,
        )

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
