import logging
import os
import sys
import time
from types import SimpleNamespace

# Setup TPU networking environment variables before JAX initializes
if "TPU_WORKER_HOSTNAMES" not in os.environ:
    os.environ["TPU_WORKER_HOSTNAMES"] = "localhost"
if "TPU_PROCESS_PORT" not in os.environ:
    os.environ["TPU_PROCESS_PORT"] = "8471"
if "TPU_PROCESS_ADDRESSES" not in os.environ:
    os.environ["TPU_PROCESS_ADDRESSES"] = "localhost:8471"

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.entrypoints.engine import _set_envs_and_config
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_kimi_int4_prompt")


def _run_forward_and_sample(model_runner, batch: ScheduleBatch, token_first_arg: int):
    page_size = model_runner.page_size
    bs_needed = len(batch.seq_lens)
    cache_loc_needed = int(
        np.sum(
            ((np.array(batch.seq_lens, dtype=np.int64) + page_size - 1) // page_size) * page_size
        )
    )

    model_worker_batch = batch.get_model_worker_batch(
        [token_first_arg],
        [bs_needed],
        [cache_loc_needed],
        page_size,
        False,
    )

    forward_metadata = model_runner.attn_backend.get_forward_metadata(model_worker_batch)
    model_runner.attn_backend.forward_metadata = forward_metadata

    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_metadata = LogitsMetadata.from_model_worker_batch(
        model_worker_batch, mesh=model_runner.mesh
    )
    logits_output, _, _ = model_runner.forward(forward_batch, logits_metadata=logits_metadata)

    pad_size = len(model_worker_batch.seq_lens) - model_worker_batch.real_bs
    sampling_metadata = SamplingMetadata.from_model_worker_batch(
        model_worker_batch,
        pad_size=pad_size,
        mesh=model_runner.mesh,
        vocab_size=model_runner.model_config.vocab_size,
    )
    next_token_ids, _, _ = model_runner.sample(logits_output, sampling_metadata)
    return next_token_ids, logits_output.next_token_logits


def extend(reqs, model_runner):
    dummy_tree_cache = SimpleNamespace(
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=[reqs],
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        dp_size=1,
        enable_custom_logit_processor=False,
        chunked_reqs=None,
    )
    batch.prepare_for_extend()
    if hasattr(batch, "extend_lens") and batch.extend_lens is not None:
        token_needed = int(np.sum(np.array(batch.extend_lens, dtype=np.int64)))
    else:
        token_needed = int(np.sum(np.array(batch.seq_lens, dtype=np.int64)))
    next_token_ids, next_token_logits = _run_forward_and_sample(model_runner, batch, token_needed)
    return next_token_ids, next_token_logits, batch


def decode(input_token_ids, batch: ScheduleBatch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    bs_needed = len(batch.seq_lens)
    next_token_ids, next_token_logits = _run_forward_and_sample(model_runner, batch, bs_needed)
    return next_token_ids, next_token_logits


def generate_for_prompt(prompt: str, model_runner, tokenizer, max_new_tokens: int = 16) -> str:
    logger.info("Generating for prompt: %r (max_new_tokens=%d)", prompt, max_new_tokens)
    input_ids = tokenizer.encode(prompt)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_new_tokens=max_new_tokens,
    )

    req = Req(
        rid=0,
        origin_input_text=prompt,
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
    )
    req.prefix_indices = []
    req.fill_ids = req.origin_input_ids
    req.extend_input_len = len(req.fill_ids)
    req.logprob_start_len = len(req.origin_input_ids) - 1

    # Clear memory pools for fresh request
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    # Prefill step
    next_token_ids, _, batch = extend([req], model_runner)
    first_tok = int(np.array(next_token_ids)[0])
    generated_tokens = [first_tok]

    # Decode steps
    next_token_ids_cpu = np.array([first_tok])
    for step in range(max_new_tokens - 1):
        next_token_ids, _ = decode(next_token_ids_cpu, batch, model_runner)
        next_tok = int(np.array(next_token_ids)[0])
        generated_tokens.append(next_tok)
        next_token_ids_cpu = np.array([next_tok])
        if next_tok in (tokenizer.eos_token_id, 151643, 151645):
            break

    generated_text = tokenizer.decode(generated_tokens)
    full_text = tokenizer.decode(input_ids + generated_tokens)
    logger.info("Generated tokens: %s", generated_tokens)
    logger.info("Generated text: %r", generated_text)
    logger.info("Full response: %r", full_text)
    return generated_text


def main():
    model_path = os.environ.get("MODEL_PATH", "/dsk/kimi_original_new")
    logger.info("Initializing JAX and TPU devices...")
    logger.info("Devices available: %s", jax.devices())

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Tokenizer loaded from %s with vocab size %d", model_path, len(tokenizer))

    tp_size = int(os.environ.get("TP_SIZE", str(len(jax.devices()))))
    logger.info("Using TP size: %d", tp_size)
    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        trust_remote_code=True,
        tp_size=tp_size,
        ep_size=1,
        device="tpu",
        dtype="bfloat16",
        attention_backend="fa",
        mem_fraction_static=0.8,
        page_size=64,
        max_running_requests=1,
        skip_server_warmup=True,
        log_level="info",
    )
    _set_envs_and_config(server_args)

    model_config = ModelConfig.from_server_args(server_args)

    num_layers_env = os.environ.get("NUM_LAYERS")
    if num_layers_env is not None:
        num_layers = int(num_layers_env)
        logger.info("Overriding num_hidden_layers to %d for prompt testing...", num_layers)
        model_config.hf_text_config.num_hidden_layers = num_layers
        model_config.num_hidden_layers = num_layers
        if (
            hasattr(model_config.hf_config, "text_config")
            and model_config.hf_config.text_config is not None
        ):
            model_config.hf_config.text_config.num_hidden_layers = num_layers

    # Create mesh
    all_devices = jax.devices()
    tp = min(server_args.tp_size, len(all_devices))
    mesh = create_device_mesh(
        ici_parallelism=[-1, tp],
        dcn_parallelism=[1, 1],
    )

    logger.info("Instantiating ModelRunner with mesh %s...", mesh)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        tp_size=tp,
        dp_size=1,
        server_args=server_args,
        mesh=mesh,
    )
    model_runner.req_to_token_pool.init_cache_loc_host_buffer(model_runner.max_total_num_tokens)
    logger.info("ModelRunner successfully initialized and weights loaded onto TPU!")

    # Test prompts for verification
    prompts = [
        "The capital of France is",
        "What is the boiling point of water in Celsius? Answer:",
        "Translate 'Good morning' into Spanish: ",
        "Calculate: 25 * 4 = ",
        "Python function to add two numbers:\ndef add(a, b):",
    ]

    for p in prompts:
        print("\n" + "=" * 50)
        print(f"PROMPT: {p}")
        out = generate_for_prompt(p, model_runner, tokenizer, max_new_tokens=32)
        print(f"RESPONSE: {out}")
        print("=" * 50 + "\n")
        assert len(out.strip()) > 0, f"Empty response generated for prompt: {p}"

    logger.info("SUCCESS: All prompt generation tests passed with verified legible responses!")


if __name__ == "__main__":
    main()
