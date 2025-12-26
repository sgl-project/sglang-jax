# misc file, only used for dump hf_logprobs
# adapted from sglang python/sglang/test/runners.py
# Copyright 2023-2024 SGLang Team
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    GenerationConfig,
)

from sgl_jax.srt.hf_transformers_utils import get_tokenizer

NUM_TOP_LOGPROBS = 5

DEFAULT_PROMPTS = [
    "Apple is red. Banana is Yellow. " * 800 + "Apple is",
    "The capital of the United Kingdom is",
    "Today is a sunny day and I like",
    "AI is a field of computer science focused on",
    # the output of gemma-2-2b from SRT is unstable on the commented prompt
    # "The capital of France is",
]


@dataclass
class ModelOutput:
    output_strs: List[str] = None
    output_ids: List[int] = None
    top_input_logprobs: List[torch.Tensor] = None
    top_output_logprobs: List[torch.Tensor] = None
    top_output_logprob_idx: List[List[int]] = None
    embed_logits: List[torch.Tensor] = None
    scores: List[float] = None
    input_token_logprobs_lst: List[List[Tuple[float, int, None]]] = None
    output_token_logprobs_lst: List[List[Tuple[float, int, None]]] = None
    token_ids_input_logprobs: List[torch.Tensor] = None
    token_ids_output_logprobs: List[torch.Tensor] = None
    last_token_logits_list: List[torch.Tensor] = None
    last_layer_hidden_states_list: List[torch.Tensor] = None


class HFRunner:
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype,
        model_type: str = "generation",
        output_str_only: bool = False,
        trust_remote_code: bool = False,
        patch_model_do_sample_false: bool = False,
        matryoshka_dim: Optional[int] = None,
        use_cpu: bool = False,
        num_layers: Optional[int] = None,  # None means all layers, 1 means only first layer
    ):
        self.model_type = model_type
        self.output_str_only = output_str_only
        self.trust_remote_code = trust_remote_code
        self.patch_model_do_sample_false = patch_model_do_sample_false
        self.use_cpu = use_cpu
        self.num_layers = num_layers

        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()

        self.model_proc = mp.Process(
            target=self.start_model_process,
            args=(
                self.in_queue,
                self.out_queue,
                model_path,
                torch_dtype,
                matryoshka_dim,
            ),
        )
        self.model_proc.start()

    def needs_trust_remote_code(self, model_path):
        models_needs_trust_remote = [
            "LxzGordon/URM-LLaMa-3.1-8B",
        ]
        if model_path in models_needs_trust_remote:
            return True
        return False

    def _forward_gme_qwen2_vl(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pooling_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.get_dtype())
                image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw).to(
                    inputs_embeds.device
                )
                image_mask = input_ids == self.model.config.image_token_id
                inputs_embeds[image_mask] = image_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
        )

        embeddings = outputs.hidden_states[-1][:, -1]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def start_model_process(
        self,
        in_queue,
        out_queue,
        model_path,
        torch_dtype,
        matryoshka_dim: Optional[int] = None,
    ):
        # Apply model-specific patches
        monkey_patch_gemma2_sdpa()

        # Determine device
        device = "cpu" if self.use_cpu else "cuda"

        # Load the model and tokenizer
        if self.model_type == "generation":
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=self.trust_remote_code
            )
            if self.trust_remote_code:
                model_cls = AutoModelForCausalLM
            else:
                model_arch = getattr(config, "architectures")[0]
                model_cls = getattr(transformers, model_arch)
            self.base_model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=True,
            ).to(device)
        elif self.model_type == "embedding":
            if "gme-qwen2-vl" in model_path.lower():
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                ).to(device)
                self.processor = AutoProcessor.from_pretrained(model_path)
            elif "clip" in model_path.lower():
                self.model = AutoModel.from_pretrained(model_path).device()
                self.processor = AutoProcessor.from_pretrained(model_path)
            else:
                self.model = _get_sentence_transformer_embedding_model(
                    model_path, torch_dtype, matryoshka_dim=matryoshka_dim, use_cpu=self.use_cpu
                )
        elif self.model_type == "reward" or self.model_type == "cross_encoder":
            from transformers import AutoModelForSequenceClassification

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.needs_trust_remote_code(model_path),
            ).to(device)
        else:
            raise Exception(f"Unrecognized model type {self.model_type}")
        self.tokenizer = get_tokenizer(
            model_path,
            torch_dtype=torch.dtype,
            trust_remote_code=self.trust_remote_code,
        )

        # Run forward
        while True:
            prompts, image_data, max_new_tokens, lora_paths, token_ids_logprob = in_queue.get()
            if lora_paths is not None:
                assert len(prompts) == len(lora_paths)

            if prompts is not None:
                if self.model_type == "generation":
                    out_queue.put(
                        self.forward_generation_raw(
                            base_model=self.base_model,
                            prompts=prompts,
                            max_new_tokens=max_new_tokens,
                            tokenizer=self.tokenizer,
                            lora_paths=lora_paths,
                            torch_dtype=torch_dtype,
                            output_str_only=self.output_str_only,
                            token_ids_logprob=token_ids_logprob,
                            patch_model_do_sample_false=self.patch_model_do_sample_false,
                            use_cpu=self.use_cpu,
                            num_layers=self.num_layers,
                        )
                    )
                elif self.model_type == "cross_encoder":
                    inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                    scores = self.model(**inputs).logits
                    scores = scores.squeeze().tolist()
                    if not isinstance(scores, list):
                        scores = [scores]
                    out_queue.put(ModelOutput(scores=scores))

                elif self.model_type == "reward":
                    scores = []
                    for conv in prompts:
                        conv_formatted = self.tokenizer.apply_chat_template(
                            conv, tokenize=False, return_dict=False
                        )
                        conv_tokenized = self.tokenizer(conv_formatted, return_tensors="pt").to(
                            device
                        )
                        scores.append(float(self.model(**conv_tokenized).logits[0][0].item()))
                    out_queue.put(ModelOutput(scores=scores))
                else:
                    raise Exception(f"Unrecognized model type {self.model_type}")

    def forward(
        self,
        prompts: Union[List[List[str]], List[str], List[torch.Tensor]] = DEFAULT_PROMPTS,
        image_data: Optional[List[str]] = None,
        max_new_tokens: int = 8,
        lora_paths: Optional[List[str]] = None,
        token_ids_logprob: Optional[int] = None,
    ):
        self.in_queue.put((prompts, image_data, max_new_tokens, lora_paths, token_ids_logprob))
        return self.out_queue.get()

    def terminate(self):
        self.model_proc.terminate()
        self.in_queue = self.out_queue = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model_proc.terminate()
        self.in_queue = self.out_queue = None

    @staticmethod
    def forward_generation_raw(
        base_model,
        prompts: Union[List[str], List[torch.Tensor]],
        max_new_tokens: int,
        tokenizer,
        torch_dtype: torch.dtype,
        lora_paths: Optional[List[str]] = None,
        output_str_only: bool = False,
        token_ids_logprob: Optional[int] = None,
        patch_model_do_sample_false: Optional[bool] = False,
        use_cpu: bool = False,
        num_layers: Optional[int] = None,
    ) -> ModelOutput:
        device = "cpu" if use_cpu else "cuda"
        output_strs = []
        top_input_logprobs = []
        top_output_logprobs = []
        if token_ids_logprob is not None:
            token_ids_input_logprobs = []
            token_ids_output_logprobs = []
        else:
            token_ids_input_logprobs = token_ids_output_logprobs = None
        last_token_logits_list=[]
        last_layer_hidden_states_list=[]

        for i, p in enumerate(prompts):
            if isinstance(p, str):
                input_ids = tokenizer.encode(p, return_tensors="pt").to(device)
            else:
                input_ids = torch.tensor([p], device=device)

            if lora_paths is not None and lora_paths[i] is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    base_model,
                    lora_paths[i],
                    torch_dtype=torch_dtype,
                    is_trainable=False,
                )
            else:
                model = base_model

            if patch_model_do_sample_false:
                model.generation_config.do_sample = False
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=(not output_str_only),
                    # make sure to disable compile
                    disable_compile=True,
                ),
            )

            text = tokenizer.decode(outputs[0][0][len(input_ids[0]) :], skip_special_tokens=True)

            # Check if the text is empty or only whitespace.
            if not text.strip():
                raise ValueError(
                    "Received an empty text response. Please verify your input or model configuration."
                )
            output_strs.append(text)

            if not output_str_only:
                # outputs.scores: (num_token, 1, vocab_size)
                top_output_logprobs.append(
                    [
                        get_top_logprobs(logits[0], NUM_TOP_LOGPROBS).tolist()
                        for logits in outputs.scores
                    ]
                )
                if token_ids_logprob is not None:
                    token_ids_output_logprobs.append(
                        [
                            get_token_ids_logprobs(logits[0], token_ids_logprob).tolist()
                            for logits in outputs.scores
                        ]
                    )


                full_sequence_ids = outputs.sequences[:,:-1] # [1, orig_len + actual_new_tokens]
                print(f"[recompute] {full_sequence_ids.shape=}, {full_sequence_ids=}")
                print(f"[recompute] {input_ids.shape=}, {input_ids=}")
                input_ids = full_sequence_ids
                del outputs

                def get_logits_and_hidden_states(num_layers, input_ids):
                    # Forward with limited layers if num_layers is specified
                    if num_layers is not None:
                        # Use output_hidden_states to get intermediate layer outputs
                        outputs = model.forward(input_ids, output_hidden_states=True, return_dict=True)
                        # Current implementation may be not right, so return zero tensor to remind user this is not supported when he/she uses it.
                        # hidden_states: tuple of (num_layers + 1) tensors
                        # Index 0 is embedding, 1 is after first layer, etc.
                        # So num_layers=1 means we want hidden_states[1]
                        #hidden_states = outputs.hidden_states[num_layers]

                        # # Manually apply norm and lm_head to get logits
                        # # Try to find the base model
                        # base = model.base_model if hasattr(model, "base_model") else model
                        # if hasattr(base, "model"):
                        #     base = base.model

                        # # Apply final norm if it exists (try different common names)
                        # if hasattr(base, "norm"):
                        #     hidden_state = base.norm(hidden_state)
                        # elif hasattr(base, "final_layernorm"):
                        #     hidden_state = base.final_layernorm(hidden_state)
                        # elif hasattr(base, "ln_f"):
                        #     hidden_state = base.ln_f(hidden_state)
                        # # If no norm found, skip it (some models might not have it)

                        # # Apply lm_head
                        # if hasattr(model, "lm_head"):
                        #     input_logits = model.lm_head(hidden_state)[0]
                        # elif hasattr(base, "lm_head"):
                        #     input_logits = base.lm_head(hidden_state)[0]
                        # else:
                        #     # If no lm_head found, just use the hidden state as "logits"
                        #     input_logits = hidden_state[0]
                    else:
                        outputs = model.forward(input_ids,output_hidden_states=True, return_dict=True)
                        input_logits = outputs.logits[0]
                        # hidden_states = outputs.hidden_states[0] # Support in the future

                    return input_logits

                input_tokens_logits= get_logits_and_hidden_states(num_layers, input_ids)
                if max_new_tokens > 1:
                    # For full_sequences logits
                    full_tokens_logits = get_logits_and_hidden_states(num_layers, full_sequence_ids)
                else:
                    full_tokens_logits = input_tokens_logits

                last_token_logit = full_tokens_logits[-1,:].detach()
                last_token_logits_list.append(last_token_logit)
                last_layer_hidden_states_list.append(None)
                top_input_logprobs.append(get_top_logprobs(input_tokens_logits, NUM_TOP_LOGPROBS).tolist())
                if token_ids_logprob is not None:
                    token_ids_input_logprobs.append(
                        get_token_ids_logprobs(input_tokens_logits, token_ids_logprob).tolist()
                    )
                del input_tokens_logits

            if lora_paths is not None and lora_paths[i] is not None:
                # Unload the LoRA adapter if it is used
                model.unload()

        return ModelOutput(
            output_strs=output_strs,
            top_input_logprobs=top_input_logprobs,
            top_output_logprobs=top_output_logprobs,
            token_ids_input_logprobs=token_ids_input_logprobs,
            token_ids_output_logprobs=token_ids_output_logprobs,
            last_token_logits_list=last_token_logits_list,
            last_layer_hidden_states_list=last_layer_hidden_states_list,
        )

def monkey_patch_gemma2_sdpa():
    """
    Use sdpa by default to fix the OOM issue.
    Revert this commit:
    https://github.com/huggingface/transformers/commit/975b988bfe6e7ebb47390cd9a1556c6888804883#diff-5f76eac6f18f4b491521314c318a9692318feb4d19228e9576cce7bde4240834R660
    """
    from transformers.models.gemma2.modeling_gemma2 import Gemma2PreTrainedModel

    def _check_and_enable_sdpa(config, hard_check_only: bool = False):
        config._attn_implementation = "sdpa"
        return config

    setattr(Gemma2PreTrainedModel, "_check_and_enable_sdpa", _check_and_enable_sdpa)


def get_dtype_str(torch_dtype):
    if torch_dtype is torch.float16:
        return "float16"
    if torch_dtype is torch.float32:
        return "float32"
    if torch_dtype is torch.bfloat16:
        return "bfloat16"
    else:
        raise NotImplementedError()


def get_top_logprobs(logits, k):
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    del logits
    logprobs, top_indices = torch.topk(logprobs, k=k, dim=-1)
    return logprobs


def get_token_ids_logprobs(logits, token_ids):
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    del logits
    logprobs = logprobs[..., token_ids]
    return logprobs


def _get_sentence_transformer_embedding_model(
    model_path, torch_dtype, matryoshka_dim: Optional[int] = None, use_cpu: bool = False
):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import is_sentence_transformer_model

    device = "cpu" if use_cpu else "cuda"

    if is_sentence_transformer_model(model_path):
        model = SentenceTransformer(
            model_path,
            model_kwargs={"torch_dtype": torch_dtype},
            truncate_dim=matryoshka_dim,
        )
    else:  # if no pre-trained sentence-transformers model
        from sentence_transformers import models

        word_embedding_model = models.Transformer(model_path).to(dtype=torch_dtype)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="lasttoken",
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], truncate_dim=matryoshka_dim
        )

    return model.to(device)
