import json
import logging
import os
from enum import Enum, IntEnum, auto
from typing import List, Optional, Set, Union

import jax.numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_generation_config,
    get_hf_text_config,
)
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import get_bool_env_var

logger = logging.getLogger(__name__)


class AttentionArch(IntEnum):
    MLA = auto()
    MHA = auto()


class ModelImpl(str, Enum):
    AUTO = "auto"
    SGLANG = "sglang"
    TRANSFORMERS = "transformers"


class ModelConfig:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        context_length: Optional[int] = None,
        model_override_args: str = "{}",
        is_embedding: Optional[bool] = None,
        dtype: str = "auto",
        override_config_file: Optional[str] = None,
        is_draft_model: bool = False,
        model_impl: Union[str, ModelImpl] = ModelImpl.AUTO,
        quantization: Optional[str] = None,
        model_layer_nums: Optional[int] = None,
    ) -> None:

        self.model_path = model_path
        self.revision = revision
        self.model_impl = model_impl
        self.quantization = quantization

        # Parse args
        self.maybe_pull_model_tokenizer_from_remote()
        self.model_override_args = json.loads(model_override_args)
        kwargs = {}
        if override_config_file and override_config_file.strip():
            kwargs["_configuration_file"] = override_config_file.strip()

        self.hf_config = get_config(
            self.model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_override_args=self.model_override_args,
            **kwargs,
        )

        self.hf_generation_config = get_generation_config(
            self.model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )

        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.attention_chunk_size = getattr(
            self.hf_text_config, "attention_chunk_size", None
        )

        if (
            is_draft_model
            and self.hf_config.architectures[0] == "DeepseekV3ForCausalLM"
        ):
            self.hf_config.architectures[0] = "DeepseekV3ForCausalLMNextN"

        if is_draft_model and self.hf_config.architectures[0] == "MiMoForCausalLM":
            self.hf_config.architectures[0] = "MiMoMTP"
        # Check model type
        self.is_generation = is_generation_model(
            self.hf_config.architectures, is_embedding
        )
        self.is_multimodal = False
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)

        # Derive context length
        derived_context_len = get_context_length(self.hf_text_config)
        if context_length is not None:
            if context_length > derived_context_len:
                if get_bool_env_var(
                    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", default="True"
                ):
                    logger.warning(
                        f"Warning: User-specified context_length ({context_length}) is greater than the derived context_length ({derived_context_len}). "
                        f"This may lead to incorrect model outputs or CUDA errors."
                    )
                    self.context_len = context_length
                else:
                    raise ValueError(
                        f"User-specified context_length ({context_length}) is greater than the derived context_length ({derived_context_len}). "
                        f"This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config. "
                        f"To allow overriding this maximum, set the env var SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"
                    )
            else:
                self.context_len = context_length
        else:
            self.context_len = derived_context_len

        # Unify the config keys for hf_text_config
        self.head_dim = getattr(
            self.hf_text_config,
            "head_dim",
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads,
        )

        self.attention_arch = AttentionArch.MHA
        self.num_attention_heads = self.hf_text_config.num_attention_heads
        self.num_key_value_heads = getattr(
            self.hf_text_config, "num_key_value_heads", None
        )

        # for Dbrx and MPT models
        if self.hf_config.model_type in ["dbrx", "mpt"]:
            self.num_key_value_heads = getattr(
                self.hf_config.attn_config, "kv_n_heads", None
            )

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = self.hf_text_config.hidden_size
        self.num_hidden_layers = self.hf_text_config.num_hidden_layers
        self.vocab_size = self.hf_text_config.vocab_size

        # Override num_hidden_layers if model_layer_nums is specified
        if model_layer_nums is not None:
            if model_layer_nums <= 0:
                raise ValueError(
                    f"model_layer_nums must be positive, got {model_layer_nums}"
                )
            if model_layer_nums > self.num_hidden_layers:
                logger.warning(
                    f"model_layer_nums ({model_layer_nums}) is greater than the original "
                    f"num_hidden_layers ({self.num_hidden_layers}). Using original value."
                )
            elif model_layer_nums != self.num_hidden_layers:
                self.num_hidden_layers = model_layer_nums
                # Also update hf_config to ensure consistency across all components
                self.hf_config.num_hidden_layers = model_layer_nums

        # Cache attributes
        self.hf_eos_token_id = self.get_hf_eos_token_id()

        config = self.hf_config

        # multimodal
        self.image_token_id = getattr(config, "image_token_id", None) or getattr(
            config, "image_token_index", None
        )

    @staticmethod
    def from_server_args(server_args: ServerArgs, model_path: str = None, **kwargs):
        return ModelConfig(
            model_path=model_path or server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
            model_impl=server_args.model_impl,
            model_layer_nums=server_args.model_layer_nums,
            **kwargs,
        )

    # adapted from https://github.com/vllm-project/vllm/blob/main/vllm/config.py#L289
    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads (original, not replicated)."""
        # Use original value if it was stored during replication configuration
        if hasattr(self, "_original_hf_num_key_value_heads"):
            return self._original_hf_num_key_value_heads
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_text_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        # For DBRX and MPT
        if self.hf_config.model_type in ["mpt"]:
            if "kv_n_heads" in self.hf_config.attn_config:
                return self.hf_config.attn_config["kv_n_heads"]
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type in ["dbrx"]:
            return getattr(
                self.hf_config.attn_config,
                "kv_n_heads",
                self.hf_config.num_attention_heads,
            )

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_text_config.num_attention_heads

    def get_num_kv_heads(self, tensor_parallel_size) -> int:
        """Returns the number of KV heads per GPU."""
        from sgl_jax.srt.utils.jax_utils import get_num_kv_heads_by_tp

        total_num_kv_heads = self.get_total_num_kv_heads()
        return get_num_kv_heads_by_tp(total_num_kv_heads, tensor_parallel_size)

    def needs_kv_head_replication(self, tensor_parallel_size: int) -> bool:
        """Returns True if KV heads need to be replicated across devices."""
        total_num_kv_heads = self.get_total_num_kv_heads()
        return tensor_parallel_size > total_num_kv_heads

    def get_num_kv_head_replicas(self, tensor_parallel_size: int) -> int:
        """Returns the number of replicas for each original KV head."""
        total_num_kv_heads = self.get_total_num_kv_heads()
        if tensor_parallel_size > total_num_kv_heads:
            return (tensor_parallel_size + total_num_kv_heads - 1) // total_num_kv_heads
        else:
            return 1

    def get_total_num_kv_heads_with_replication(self, tensor_parallel_size: int) -> int:
        """Returns the total number of KV heads after replication."""
        total_num_kv_heads = self.get_total_num_kv_heads()
        if tensor_parallel_size > total_num_kv_heads:
            # When replication is needed, total becomes tensor_parallel_size
            # because each device gets 1 head and there are tp_size devices
            return tensor_parallel_size
        else:
            # No replication needed, return original
            return total_num_kv_heads

    def configure_for_tensor_parallel(self, tensor_parallel_size: int):
        """Configure model config for tensor parallel execution with KV head replication."""
        # Get per-device KV head count
        kv_heads_per_device = self.get_num_kv_heads(tensor_parallel_size)

        # Store original values for reference (only once)
        if not hasattr(self, "_original_num_key_value_heads"):
            self._original_num_key_value_heads = self.num_key_value_heads

        # Handle cases where HF config doesn't have num_key_value_heads (MHA models)
        if hasattr(self.hf_text_config, "num_key_value_heads"):
            if not hasattr(self, "_original_hf_num_key_value_heads"):
                self._original_hf_num_key_value_heads = (
                    self.hf_text_config.num_key_value_heads
                )
        else:
            # For MHA models without this attribute, it equals num_attention_heads
            if not hasattr(self, "_original_hf_num_key_value_heads"):
                self._original_hf_num_key_value_heads = (
                    self.hf_text_config.num_attention_heads
                )

        # CRITICAL: Set to TOTAL count for global sharding
        # JAX tensor parallel will automatically shard this across devices
        total_kv_heads = kv_heads_per_device * tensor_parallel_size
        self.num_key_value_heads = total_kv_heads

        # Only set HF config if the attribute exists, otherwise create it
        if hasattr(self.hf_text_config, "num_key_value_heads"):
            self.hf_text_config.num_key_value_heads = total_kv_heads
        else:
            # For MHA models, dynamically add the attribute
            setattr(self.hf_text_config, "num_key_value_heads", total_kv_heads)

    def get_original_kv_head_id(self, tp_rank: int, tensor_parallel_size: int) -> int:
        """Determine which original KV head this device should use."""
        from sgl_jax.srt.utils.jax_utils import get_original_kv_head_id

        total_num_kv_heads = self.get_total_num_kv_heads()
        return get_original_kv_head_id(
            tp_rank, total_num_kv_heads, tensor_parallel_size
        )

    def is_gqa_model(self) -> bool:
        """Returns True if this is a Grouped Query Attention model."""
        return self.get_total_num_kv_heads() < self.num_attention_heads

    def get_kv_padding_strategy(self) -> str:
        """Returns the padding strategy for KV heads."""
        if self.is_gqa_model():
            # GQA models should replicate existing kv heads to maintain attention semantics
            return "replicate"
        else:
            # MHA models can use zero padding since all heads are equivalent
            return "zero"

    def log_kv_heads_info(self, tensor_parallel_size: int):
        """Log KV heads configuration information during initialization."""
        original_kv_heads = self.get_total_num_kv_heads()
        kv_heads_per_device = self.get_num_kv_heads(tensor_parallel_size)
        needs_replication = self.needs_kv_head_replication(tensor_parallel_size)
        padding_strategy = self.get_kv_padding_strategy()

        model_type = "GQA" if self.is_gqa_model() else "MHA"

        if needs_replication:
            num_replicas = self.get_num_kv_head_replicas(tensor_parallel_size)
            logger.info(
                f"KV heads replication enabled for {model_type} model: "
                f"original_kv_heads={original_kv_heads}, tp_size={tensor_parallel_size}, "
                f"each device gets {kv_heads_per_device} head(s), "
                f"each original head replicated {num_replicas} times, "
                f"padding_strategy={padding_strategy}"
            )
        else:
            logger.info(
                f"KV heads distribution for {model_type} model: "
                f"original_kv_heads={original_kv_heads}, tp_size={tensor_parallel_size}, "
                f"each device gets {kv_heads_per_device} head(s), no replication needed, "
                f"padding_strategy={padding_strategy}"
            )

    def validate_tensor_parallel_config(self, tensor_parallel_size: int):
        """Validate tensor parallel configuration constraints."""
        # Query heads must be divisible by tensor parallel size
        assert self.num_attention_heads % tensor_parallel_size == 0, (
            f"Number of attention heads ({self.num_attention_heads}) must be divisible by "
            f"tensor parallel size ({tensor_parallel_size}). "
            f"Got remainder: {self.num_attention_heads % tensor_parallel_size}"
        )

    # adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/config.py
    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        if quant_cfg is None:
            # check if is modelopt model -- modelopt doesn't have corresponding field
            # in hf `config.json` but has a standalone `hf_quant_config.json` in the root directory
            # example: https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8/tree/main
            is_local = os.path.exists(self.model_path)
            modelopt_quant_config = {"quant_method": "modelopt"}
            if not is_local:
                from huggingface_hub import HfApi

                hf_api = HfApi()
                if hf_api.file_exists(self.model_path, "hf_quant_config.json"):
                    quant_cfg = modelopt_quant_config
            elif os.path.exists(os.path.join(self.model_path, "hf_quant_config.json")):
                quant_config_file = os.path.join(
                    self.model_path, "hf_quant_config.json"
                )
                with open(quant_config_file) as f:
                    quant_config_dict = json.load(f)
                json_quant_configs = quant_config_dict["quantization"]
                quant_algo = json_quant_configs.get("quant_algo", None)
                if quant_algo == "MIXED_PRECISION":
                    quant_cfg = {"quant_method": "w4afp8"}
                else:
                    quant_cfg = modelopt_quant_config
        return quant_cfg

    def get_hf_eos_token_id(self) -> Optional[Set[int]]:
        eos_ids = getattr(self.hf_config, "eos_token_id", None)
        if eos_ids:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        if eos_ids is None:
            eos_ids = set()
        if self.hf_generation_config:
            generation_eos_ids = getattr(
                self.hf_generation_config, "eos_token_id", None
            )
            if generation_eos_ids:
                generation_eos_ids = (
                    {generation_eos_ids}
                    if isinstance(generation_eos_ids, int)
                    else set(generation_eos_ids)
                )
                eos_ids = eos_ids | generation_eos_ids
        return eos_ids

    def maybe_pull_model_tokenizer_from_remote(self) -> None:
        """
        Pull the model config files to a temporary
        directory in case of remote.

        Args:
            model: The model name or path.

        """
        from sgl_jax.srt.utils.common_utils import is_remote_url

        if is_remote_url(self.model_path):
            raise ValueError(
                f"Remote URLs are not supported in JAX implementation. "
                f"Please use a local path or HuggingFace model name instead: {self.model_path}"
            )


_STR_DTYPE_TO_JAX_DTYPE = {
    "half": jnp.float16,
    "float16": jnp.float16,
    "float": jnp.float32,
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
}


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: Union[str, jnp.dtype],
) -> jnp.dtype:
    config_dtype = getattr(config, "torch_dtype", None)
    if isinstance(config_dtype, str):
        config_dtype = _STR_DTYPE_TO_JAX_DTYPE.get(config_dtype, None)
    elif config_dtype is not None:
        config_dtype = _STR_DTYPE_TO_JAX_DTYPE.get(
            str(config_dtype).replace("torch.", ""), None
        )

    if config_dtype is None:
        config_dtype = jnp.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            jax_dtype = config_dtype
            if config_dtype != jnp.bfloat16:
                logger.warning(
                    f"Model dtype is {config_dtype}. "
                    "On TPU, using non-bfloat16 models may reduce performance."
                )
        else:
            if dtype not in _STR_DTYPE_TO_JAX_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            jax_dtype = _STR_DTYPE_TO_JAX_DTYPE[dtype]
    elif isinstance(dtype, jnp.dtype):
        jax_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    # Verify the dtype.
    if jax_dtype != config_dtype:
        if jax_dtype == jnp.float32:
            # Upcasting to float32 is allowed.
            logger.info("Upcasting %s to %s.", config_dtype, jax_dtype)
            pass
        elif config_dtype == jnp.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info("Downcasting %s to %s.", config_dtype, jax_dtype)
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning("Casting %s to %s.", config_dtype, jax_dtype)
    return jax_dtype


def is_generation_model(model_architectures: List[str], is_embedding: bool = False):
    # We have two ways to determine whether a model is a generative model.
    # 1. Check the model architecture
    # 2. check the `is_embedding` server args

    if (
        "LlamaEmbeddingModel" in model_architectures
        or "MistralModel" in model_architectures
        or "LlamaForSequenceClassification" in model_architectures
        or "LlamaForSequenceClassificationWithNormal_Weights" in model_architectures
        or "InternLM2ForRewardModel" in model_architectures
        or "Qwen2ForRewardModel" in model_architectures
        or "Qwen2ForSequenceClassification" in model_architectures
        or "CLIPModel" in model_architectures
        or "BertModel" in model_architectures
        or "Contriever" in model_architectures
        or "BertForSequenceClassification" in model_architectures
        or "XLMRobertaModel" in model_architectures
        or "XLMRobertaForSequenceClassification" in model_architectures
    ):
        return False
    else:
        return not is_embedding


multimodal_model_archs = [
    "CLIPModel",
    "DeepseekVL2ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForConditionalGeneration",
    "Grok1VForCausalLM",
    "Grok1AForCausalLM",
    "LlavaLlamaForCausalLM",
    "Llama4ForConditionalGeneration",
    "LlavaMistralForCausalLM",
    "LlavaQwenForCausalLM",
    "LlavaForConditionalGeneration",
    "LlavaVidForCausalLM",
    "MiniCPMO",
    "MiniCPMV",
    "Mistral3ForConditionalGeneration",
    "MultiModalityCausalLM",
    "MllamaForConditionalGeneration",
    "Qwen2AudioForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "KimiVLForConditionalGeneration",
    "InternVLChatModel",
    "Phi4MMForCausalLM",
    "VILAForConditionalGeneration",
]


class MockModelConfig(ModelConfig):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        context_len: int,
        num_hidden_layers: int,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.context_len = context_len
        self.num_hidden_layers = num_hidden_layers

    def get_num_kv_heads(self, tensor_parallel_size) -> int:
        return self.num_kv_heads
