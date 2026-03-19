import inspect
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class _EncoderSpec:
    model_class: Any
    stage_sub_dir: str | None
    tokenizer_path: str | None
    model_config: ModelConfig | None = None
    model: Any = None
    jitted_forward: Any = None
    tokenizer: Any = None
    max_length: int | None = None


class EncoderModelRunner(BaseModelRunner):
    def __init__(
        self,
        server_args: ServerArgs = None,
        mesh: jax.sharding.Mesh = None,
        model_class: str | list[str] = None,
        stage_sub_dir: str = None,
        tokenizer: str = None,
    ):
        self.server_args = server_args
        self.mesh = mesh
        self.encoder_specs = self._build_encoder_specs(model_class, stage_sub_dir, tokenizer)
        self.initialize()

    def initialize(self):
        for spec in self.encoder_specs:
            self._load_encoder(spec)
            self._initialize_jit(spec)

    def _build_encoder_specs(
        self,
        model_class: Any | Sequence[Any] | None,
        stage_sub_dir: str | Sequence[str] | None,
        tokenizer: str | Sequence[str] | None,
    ) -> list[_EncoderSpec]:
        model_classes = self._normalize_to_list(model_class)
        stage_sub_dirs = self._normalize_to_list(stage_sub_dir)
        tokenizer_paths = self._normalize_to_list(tokenizer)

        if not model_classes:
            raise ValueError("EncoderModelRunner requires at least one model class.")

        stage_sub_dirs = self._align_optional_list(stage_sub_dirs, len(model_classes))
        tokenizer_paths = self._align_optional_list(tokenizer_paths, len(model_classes))

        specs: list[_EncoderSpec] = []
        for idx, cls in enumerate(model_classes):
            sub_dir = stage_sub_dirs[idx]
            tokenizer_path = tokenizer_paths[idx]
            specs.append(
                _EncoderSpec(
                    model_class=cls,
                    stage_sub_dir=sub_dir,
                    tokenizer_path=tokenizer_path,
                )
            )
        return specs

    def _load_encoder(self, spec: _EncoderSpec):
        model_path = self.server_args.model_path
        tokenizer_source, tokenizer_subdir = self._resolve_tokenizer_source(
            model_path, spec.tokenizer_path
        )
        spec.tokenizer = get_tokenizer(
            tokenizer_source,
            trust_remote_code=getattr(self.server_args, "trust_remote_code", False),
            sub_dir=tokenizer_subdir,
        )
        spec.model_config = ModelConfig(
            model_path=model_path,
            trust_remote_code=getattr(self.server_args, "trust_remote_code", True),
            revision=getattr(self.server_args, "revision", None),
            dtype=getattr(self.server_args, "dtype", "auto"),
            model_impl=getattr(self.server_args, "model_impl", "auto"),
            quantization=getattr(self.server_args, "quantization", None),
            quantization_config_path=getattr(self.server_args, "quantization_config_path", None),
            multimodal=True,
            model_sub_dir=spec.stage_sub_dir,
        )
        spec.model_config.model_class = spec.model_class
        spec.model = get_model_loader(
            load_config=LoadConfig(model_class=spec.model_class),
            mesh=self.mesh,
        ).load_model(model_config=spec.model_config)
        spec.max_length = self._resolve_max_length(spec)

    def _initialize_jit(self, spec: _EncoderSpec):
        model_def, model_state = nnx.split(spec.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)
        model_call = getattr(spec.model, "__call__")
        parameters = inspect.signature(model_call).parameters
        use_input_ids = "input_ids" in parameters
        use_output_hidden_states = "output_hidden_states" in parameters
        use_cache = "use_cache" in parameters

        @partial(
            jax.jit,
            static_argnames=["model_state_def"],
        )
        def forward(
            model_def,
            model_state_def,
            model_state_leaves,
            input_ids,
            attention_mask,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            kwargs = {}
            if use_input_ids:
                kwargs["input_ids"] = input_ids
                kwargs["attention_mask"] = attention_mask
            else:
                kwargs["input_ids"] = input_ids
                kwargs["attention_mask"] = attention_mask
            if use_output_hidden_states:
                kwargs["output_hidden_states"] = True
            if use_cache:
                kwargs["use_cache"] = False
            return model(**kwargs)

        def forward_wrapper(input_ids: jax.Array, attention_mask: jax.Array):
            return forward(
                model_def,
                model_state_def,
                model_state_leaves,
                input_ids,
                attention_mask,
            )

        spec.jitted_forward = forward_wrapper

    def mock_data(self):
        # 初始化随机种子，用于生成逼真的浮点数 Embedding
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        # ---------------------------------------------------------
        # 1. 构造 prompt_embeds_list
        # ---------------------------------------------------------
        # 形状分别是 (1, 768) 和 (1, 512, 4096)
        # 使用正态分布模拟真实的文本特征向量
        embed_1 = jax.random.normal(k1, (1, 768))
        embed_2 = jax.random.normal(k2, (1, 512, 4096))

        prompt_embeds_list = [embed_1, embed_2]

        # ---------------------------------------------------------
        # 2. 构造 prompt_masks_list
        # ---------------------------------------------------------
        # 第一个形状是 (1, 77)，前面是1，后面是0
        # 假设我们模拟一个长度为 30 的有效句子，剩下的 47 个全是 Padding(0)
        valid_length = 30
        # 使用 jnp.arange 生成 0-76 的索引，并与 valid_length 比较，最后 reshape 回 (1, 77)
        mask_1 = jnp.where(jnp.arange(77) < valid_length, 1, 0).reshape(1, 77).astype(jnp.int32)

        # 第二个形状是 (1, 512)，全 1
        mask_2 = jnp.ones((1, 512), dtype=jnp.int32)

        prompt_masks_list = [mask_1, mask_2]

        # ---------------------------------------------------------
        # 3. 构造 pooler_embeds_list
        # ---------------------------------------------------------
        pooler_embeds_list = [jax.random.normal(k3, (1, 768)), None]

        return prompt_embeds_list, prompt_masks_list, pooler_embeds_list
    
    def forward(self, batch: Req) -> Req:
        all_indices = list(range(len(self.encoder_specs)))
        prompt_text = batch.prompt or batch.origin_input_text
        if prompt_text is not None:
            # prompt_embeds_list, prompt_masks_list, pooler_embeds_list = self.encode_text(
            #     prompt_text,
            #     encoder_index=all_indices,
            # )
            prompt_embeds_list, prompt_masks_list, pooler_embeds_list = self.mock_data()
            self._assign_prompt_outputs(
                batch=batch,
                embeds_list=prompt_embeds_list,
                masks_list=prompt_masks_list,
                pooler_embeds_list=pooler_embeds_list,
                is_negative=False,
            )

        if batch.do_classifier_free_guidance and batch.negative_prompt is not None:
            # neg_embeds_list, neg_masks_list, neg_pooler_embeds_list = self.encode_text(
            #     batch.negative_prompt,
            #     encoder_index=all_indices,
            # )
            neg_embeds_list, neg_masks_list, neg_pooler_embeds_list = self.mock_data()
            self._assign_prompt_outputs(
                batch=batch,
                embeds_list=neg_embeds_list,
                masks_list=neg_masks_list,
                pooler_embeds_list=neg_pooler_embeds_list,
                is_negative=True,
            )

        return batch

    def encode_text(
        self,
        text: str | list[str],
        encoder_index: int | list[int] | None = None,
    ):
        if encoder_index is None:
            indices = [0]
        elif isinstance(encoder_index, int):
            indices = [encoder_index]
        else:
            indices = list(encoder_index)

        num_encoders = len(self.encoder_specs)
        for idx in indices:
            if idx < 0 or idx >= num_encoders:
                raise IndexError(f"encoder index {idx} out of range [0, {num_encoders - 1}]")

        embeds_list: list[jax.Array] = []
        pooler_embeds_list: list[jax.Array] = []
        attn_masks_list: list[jax.Array] = []

        for idx in indices:
            spec = self.encoder_specs[idx]
            tokenized = self._tokenize_text(spec, text)
            model_attention_mask = self._get_model_attention_mask(
                spec,
                tokenized["attention_mask"],
                idx,
            )
            outputs = self._forward_encoder(
                spec,
                input_ids=tokenized["input_ids"],
                attention_mask=model_attention_mask,
            )
            hidden_states, pooler_output = self._extract_encoder_outputs(outputs)
            if hidden_states is not None:
                embeds_list.append(hidden_states)
            if pooler_output is not None:
                pooler_embeds_list.append(pooler_output)
            attn_masks_list.append(model_attention_mask)

        return embeds_list, attn_masks_list, pooler_embeds_list
       

    def _assign_prompt_outputs(
        self,
        batch: Req,
        embeds_list: list[jax.Array],
        masks_list: list[jax.Array],
        pooler_embeds_list: list[jax.Array],
        is_negative: bool,
    ) -> None:
        if len(embeds_list) == 1:
            embed_value: jax.Array | list[jax.Array] = embeds_list[0]
        else:
            embed_value = embeds_list

        if is_negative:
            batch.negative_prompt_embeds = embed_value
            if batch.negative_attention_mask is None:
                batch.negative_attention_mask = []
            batch.negative_attention_mask = masks_list
            batch.neg_pooled_embeds = pooler_embeds_list
        else:
            batch.prompt_embeds = embed_value
            if batch.prompt_attention_mask is None:
                batch.prompt_attention_mask = []
            batch.prompt_attention_mask = masks_list
            batch.pooled_embeds = pooler_embeds_list

    def _forward_encoder(
        self,
        spec: _EncoderSpec,
        input_ids: jax.Array,
        attention_mask: jax.Array,
    ):
        if spec.jitted_forward is None:
            raise ValueError(f"Encoder model for {spec.model_class} has not been jitted.")
        return spec.jitted_forward(input_ids, attention_mask)


    def _extract_encoder_outputs(self, outputs: Any) -> tuple[jax.Array | None, jax.Array | None]:
        output_obj = outputs[0] if isinstance(outputs, tuple) else outputs

        if hasattr(output_obj, "last_hidden_state"):
            pooler_output = getattr(output_obj, "pooler_output", None)
            return output_obj.last_hidden_state, pooler_output

        if hasattr(output_obj, "hidden_states"):
            pooler_output = getattr(output_obj, "pooler_output", None)
            if pooler_output is None and self.is_flux_v1():
                pooler_output = getattr(output_obj, "text_embeds", None)
            return output_obj.hidden_states, pooler_output

        if isinstance(output_obj, jax.Array):
            pooler_output = output_obj if self.is_flux_v1() and output_obj.ndim == 2 else None
            return output_obj, pooler_output

        raise TypeError(f"Unsupported encoder output type: {type(output_obj)!r}")

    def _tokenize_text(self, spec: _EncoderSpec, text: str | list[str]) -> dict[str, jax.Array]:
        if spec.max_length is None:
            raise ValueError(f"Encoder max_length is not initialized for {spec.model_class}.")
        encoded_unpadded = spec.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=spec.max_length,
            return_tensors="np",
        )
        actual_length = encoded_unpadded["input_ids"].shape[1]
        padded_max_length = self._select_precompiled_max_length(
            actual_length=actual_length,
            encoder_max_length=spec.max_length,
        )
        encoded = spec.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=padded_max_length,
            return_tensors="np",
        )
        input_ids = jnp.asarray(encoded["input_ids"], dtype=jnp.int32)
        attention_mask = jnp.asarray(encoded["attention_mask"], dtype=jnp.int32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _select_precompiled_max_length(
        self,
        actual_length: int,
        encoder_max_length: int,
    ) -> int:
        token_buckets = [16, 32, 64, 128, 256, 512, 1024]
        for size in token_buckets:
            if actual_length <= size <= encoder_max_length:
                return size
        return encoder_max_length

    def _get_model_attention_mask(
        self,
        spec: _EncoderSpec,
        attention_mask: jax.Array,
        encoder_idx: int,
    ) -> jax.Array:
        if self.is_flux_t5(spec, encoder_idx):
            return jnp.ones_like(attention_mask, dtype=jnp.int32)
        return attention_mask

    def is_flux_t5(self, spec: _EncoderSpec, encoder_idx: int) -> bool:
        is_flux_t5 = self.is_flux_v1() and encoder_idx == 1
        return is_flux_t5

    def is_flux_v1(self) -> bool:
        model_path = (self.server_args.model_path or "").lower()
        return "flux" in model_path and "flux2" not in model_path

    @staticmethod
    def _resolve_tokenizer_source(
        model_path: str,
        tokenizer_path: str | None,
    ) -> tuple[str, str]:
        if not tokenizer_path:
            return model_path, ""

        normalized = tokenizer_path.rstrip("/")
        if os.path.isabs(normalized) or "/" in normalized:
            return normalized, ""

        return model_path, normalized

    @staticmethod
    def _resolve_max_length(spec: _EncoderSpec) -> int:
        tokenizer_max_length = getattr(spec.tokenizer, "model_max_length", None)
        if (
            isinstance(tokenizer_max_length, int)
            and tokenizer_max_length > 0
            and tokenizer_max_length < 1_000_000
        ):
            return tokenizer_max_length

        configs = (
            getattr(spec.model_config, "hf_config", None),
            getattr(spec.model_config, "hf_text_config", None),
        )
        for config in configs:
            if config is None:
                continue
            value = getattr(config, "model_max_length", None)
            if isinstance(value, int) and value > 0:
                return value

        raise ValueError(
            f"Unable to infer max_length for encoder {spec.model_class}. "
            "Please ensure the tokenizer or model config exposes a sequence length."
        )

    @staticmethod
    def _normalize_to_list(value: Any | Sequence[Any] | None) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _align_optional_list(values: list[Any], target_len: int) -> list[Any]:
        if not values:
            return [None] * target_len
        if len(values) == target_len:
            return values
        if len(values) == 1:
            return values * target_len
        raise ValueError(f"Expected 1 or {target_len} values, got {len(values)}.")
