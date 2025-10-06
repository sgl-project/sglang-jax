"""
Benchmark the latency of running a single static batch without a server.

This script does not launch a server and uses the low-level APIs.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

# Usage (latency test)
## with dummy weights:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run
## run with profiling:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --profile
# Usage (correctness test):
python -m sglang.bench_one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from typing import Tuple

import jax
import numpy as np
from jax import profiler as jax_profiler
from jax.experimental import multihost_utils as jax_mh

from sgl_jax.srt.configs.model_config import ModelConfig

# from sgl_jax.srt.distributed.parallel_state import destroy_distributed_environment
from sgl_jax.srt.entrypoints.engine import _set_envs_and_config
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import PortArgs, ServerArgs

# from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils import (  # require_mlp_sync,; require_mlp_tp_gather,; set_gpu_proc_affinity,; suppress_other_loggers,
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
)


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (8,)
    output_len: Tuple[int] = (2,)
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_filename_prefix: str = "profile"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument("--profile", action="store_true", help="Use JAX Profiler.")
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling output path. The trace will be saved under a directory named "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].tb"',
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, port_args, tp_rank):
    # TODO: pass in tp_size
    # server_args.tp_size = 1
    # suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)

    # Create a mesh that includes both 'data' and 'tensor' axes.
    # Use a size-1 'data' axis and shard across the 'tensor' axis per tp_size.
    all_devices = jax.devices()
    tp = min(server_args.tp_size, len(all_devices))
    devices = all_devices[:tp]
    print(f"devices: {devices}")
    devices_array = np.array(devices, dtype=object).reshape((1, tp))
    mesh = jax.sharding.Mesh(devices_array, ("data", "tensor"))

    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        tp_size=tp,
        server_args=server_args,
        mesh=mesh,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if tp > 1:
        try:
            jax_mh.sync_global_devices("load_model")
        except Exception:
            pass
    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer):
    prompts = [
        "The capital of France is",
        # "The capital of the United Kindom is",
        # "Today is a sunny day and I like",
    ]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=bench_args.output_len[0],
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
    return reqs


def prepare_synthetic_inputs_for_latency_test(batch_size, input_len):
    input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=(
            BenchArgs.output_len[0] if isinstance(BenchArgs.output_len, tuple) else 16
        ),
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return reqs


def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        # spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    # Prepare paddings consistent with Scheduler usage
    page_size = model_runner.page_size
    bs_needed = len(batch.seq_lens)
    # token paddings should cover total extend tokens
    if hasattr(batch, "extend_lens") and batch.extend_lens is not None:
        token_needed = int(np.sum(np.array(batch.extend_lens, dtype=np.int64)))
    else:
        # fallback: derive extend lengths as seq_len - prefix_len per req
        token_needed = int(np.sum(np.array(batch.seq_lens, dtype=np.int64)))
    # cache_loc size must be >= sum of per-seq aligned lengths
    cache_loc_needed = int(
        np.sum(
            ((np.array(batch.seq_lens, dtype=np.int64) + page_size - 1) // page_size)
            * page_size
        )
    )
    model_worker_batch = batch.get_model_worker_batch(
        [token_needed], [bs_needed], [cache_loc_needed], page_size
    )
    # Prepare attention forward metadata (required by FlashAttention backend)
    forward_metadata = model_runner.attn_backend.get_forward_metadata(
        model_worker_batch
    )
    model_runner.attn_backend.forward_metadata = forward_metadata
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_metadata = LogitsMetadata.from_model_worker_batch(
        model_worker_batch, mesh=model_runner.mesh
    )
    logits_output, _ = model_runner.forward(
        forward_batch, logits_metadata=logits_metadata
    )
    pad_size = len(model_worker_batch.seq_lens) - model_worker_batch.real_bs
    sampling_metadata = SamplingMetadata.from_model_worker_batch(
        model_worker_batch, pad_size=pad_size, mesh=model_runner.mesh
    )
    next_token_ids = model_runner.sample(logits_output, sampling_metadata)
    return next_token_ids, logits_output.next_token_logits, batch


def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    # Prepare paddings consistent with Scheduler usage
    page_size = model_runner.page_size
    bs_needed = len(batch.seq_lens)
    cache_loc_needed = int(
        np.sum(
            ((np.array(batch.seq_lens, dtype=np.int64) + page_size - 1) // page_size)
            * page_size
        )
    )
    model_worker_batch = batch.get_model_worker_batch(
        [bs_needed], [bs_needed], [cache_loc_needed], page_size
    )
    # Prepare attention forward metadata (required by FlashAttention backend)
    forward_metadata = model_runner.attn_backend.get_forward_metadata(
        model_worker_batch
    )
    model_runner.attn_backend.forward_metadata = forward_metadata
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_metadata = LogitsMetadata.from_model_worker_batch(
        model_worker_batch, mesh=model_runner.mesh
    )
    logits_output, _ = model_runner.forward(
        forward_batch, logits_metadata=logits_metadata
    )
    pad_size = len(model_worker_batch.seq_lens) - model_worker_batch.real_bs
    sampling_metadata = SamplingMetadata.from_model_worker_batch(
        model_worker_batch, pad_size=pad_size, mesh=model_runner.mesh
    )
    next_token_ids = model_runner.sample(logits_output, sampling_metadata)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    # No-op for JAX bench; MLP sync is not required here
    return


def correctness_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs
    input_ids, reqs = prepare_inputs_for_correctness_test(bench_args, tokenizer)
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # Prepare extend inputs
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def synchronize(device):
    # JAX: submit a tiny computation and wait for completion
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    device,
    log_decode_step,
    profile,
    profile_filename_prefix,
):
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    # Clear the pools.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    tot_latency = 0

    profiler = None
    if profile:
        profile_dir = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}.tb"
        os.makedirs(profile_dir, exist_ok=True)
        jax_profiler.start_trace(profile_dir)

    # Prefill
    synchronize(device)
    tic = time.perf_counter()
    next_token_ids, _, batch = extend(reqs, model_runner)
    synchronize(device)
    prefill_latency = time.perf_counter() - tic
    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    # Decode
    decode_latencies = []
    for i in range(output_len - 1):
        synchronize(device)
        tic = time.perf_counter()
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        synchronize(device)
        latency = time.perf_counter() - tic
        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )

    if profile:
        jax_profiler.stop_trace()
        rank_print(f"JAX profiler trace saved to {profile_dir}")

    # Record decode timing from 2nd output
    if output_len > 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    tp_rank,
):
    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0]
    )

    # Warm up
    rank_print("Warmup ...")
    latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        bench_args.batch_size[0],
        bench_args.input_len[0],
        min(32, bench_args.output_len[0]),  # shorter decoding to speed up the warmup
        server_args.device,
        log_decode_step=0,
        profile=False,
        profile_filename_prefix="",  # not used
    )

    rank_print("Benchmark ...")

    # Run the sweep
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        reqs = prepare_synthetic_inputs_for_latency_test(bs, il)
        ret = latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            ol,
            server_args.device,
            bench_args.log_decode_step,
            bench_args.profile if tp_rank == 0 else None,
            bench_args.profile_filename_prefix,
        )
        if ret is not None:
            result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")


def main(server_args, bench_args):
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)
    server_args.ep_size = 1

    # Constrain static KV allocation for single-device TPU if not user-specified
    if (
        server_args.max_total_tokens is None
        and (server_args.device is None or server_args.device == "tpu")
        and server_args.tp_size == 1
    ):
        bs_max = max(bench_args.batch_size)
        in_max = max(bench_args.input_len)
        out_max = max(bench_args.output_len)

        # If running correctness test, ensure the cap covers the real workload
        # (3 prompts with cut_len prefill and extend, plus a few decode steps),
        # while still being small to avoid compile-time memory blowups.
        if bench_args.correctness_test:
            bs_max = max(bs_max, 3)
            # Prompts here are short but can exceed CLI defaults; pick a safe small bound.
            in_max = max(in_max, max(bench_args.cut_len, 16))
            out_max = max(out_max, bench_args.output_len[0])

        # Total KV tokens needed equals sum of per-seq tokens kept in cache
        tokens_needed = bs_max * (in_max + out_max)
        # Small headroom for alignment/padding
        tokens_needed = int(tokens_needed * 1.1) + server_args.page_size
        # Align to page size (>=1)
        page = max(1, server_args.page_size)
        tokens_needed = (tokens_needed // page) * page
        server_args.max_total_tokens = max(tokens_needed, page)
        logging.info(
            f"Setting max_total_tokens={server_args.max_total_tokens} (bs={bs_max}, in={in_max}, out={out_max}) to limit static KV memory on single TPU"
        )

    # Prefer native attention on single-TPU runs to avoid large FA compile-time temps
    if (
        (server_args.device is None or server_args.device == "tpu")
        and server_args.tp_size == 1
        and getattr(server_args, "attention_backend", "fa") == "fa"
    ):
        server_args.attention_backend = "native"
        logging.info(
            "Switching attention backend to 'native' for single TPU to reduce compile-time memory"
        )

    _set_envs_and_config()

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)
    port_args.nccl_port = 29500

    work_func(server_args, port_args, bench_args, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
