import asyncio
import copy
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import unittest
from collections.abc import Awaitable, Callable, Sequence
from contextlib import nullcontext, suppress
from types import SimpleNamespace

import jax
import numpy as np
import psutil
import requests
from jax._src import mesh_utils

from sgl_jax.bench_serving import run_benchmark
from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import get_bool_env_var, retry

DEFAULT_MODEL_NAME_FOR_TEST = "Qwen/Qwen3-8B"
DEFAULT_SMALL_MODEL_NAME_FOR_TEST = "Qwen/Qwen3-1.7B"
QWEN3_8B = "Qwen/Qwen3-8B"
QWEN_7B = "Qwen/Qwen-7B"

QWEN3_MOE_30B = "Qwen/Qwen3-30B-A3B"
QWEN2_5_7B_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
QWEN3_CODER_30B_A3B_INSTRUCT = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
GEMMA2_2B_IT = "google/gemma-2-2b-it"
bailing_moe="inclusionAI/Ling-flash-2.0"
DEEPSEEK_R1_DISTILL_QWEN_1_5B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 600


def is_in_ci():
    """Return whether it is in CI runner."""
    return get_bool_env_var("SGLANG_JAX_IS_IN_CI")


DEFAULT_PORT_FOR_SRT_TEST_RUNNER = 5000 + 100 if is_in_ci() else 7000 + 100
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"

mesh_axes = [
    "data",  # data parallelism
    "tensor",  # tensor parallelism
]


def create_device_mesh(
    ici_parallelism: Sequence[int],
    dcn_parallelism: Sequence[int],
    devices=None,
    num_slices: int = 1,
    allow_split_physical_axes: bool = True,
    use_explicit_sharding: bool = True,
) -> jax.sharding.Mesh:
    """Create a device mesh"""
    if devices is None:
        devices = jax.devices()

    ici_parallelism = fill_unspecified_parallelism(ici_parallelism, len(devices))
    if num_slices > 1:
        dcn_parallelism = fill_unspecified_parallelism(dcn_parallelism, num_slices)
        devices_array = mesh_utils.create_hybrid_device_mesh(
            ici_parallelism,
            dcn_parallelism,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
        devices_array = mesh_utils.create_device_mesh(
            ici_parallelism,
            devices=devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=allow_split_physical_axes,
        )

    if use_explicit_sharding:
        axis_types = (jax.sharding.AxisType.Explicit,) * len(mesh_axes)
        mesh = jax.sharding.Mesh(devices_array, mesh_axes, axis_types=axis_types)
    else:
        mesh = jax.sharding.Mesh(devices_array, mesh_axes)
    return mesh


def fill_unspecified_parallelism(parallelism: Sequence[int], num_devices: int) -> Sequence[int]:
    if -1 not in parallelism:
        return parallelism

    assert parallelism.count(-1) == 1, "At most one axis can be unspecified."
    unspecified_axis_idx = parallelism.index(-1)
    determined_val = num_devices / np.prod(parallelism) * -1
    assert (
        determined_val >= 1 and determined_val.is_integer
    ), "Unspecified value unable to be determined with the given parallelism values"
    parallelism[unspecified_axis_idx] = int(determined_val)
    return parallelism


def jax_trace_context(log_dir: str):
    """Return a JAX trace context manager with options configured via env vars.

    The following environment variables are honored (all optional):

    1. ``JAX_TRACE_CREATE_PERFETTO_LINK`` – Boolean-like string (``1``, ``0``). Controls ``create_perfetto_link``.

    Example::

        os.environ["JAX_TRACE_HOST_TRACER_LEVEL"] = "2"
        with jax_trace_context("/tmp/trace"):
            ...  # code to profile
    """

    jax_trace_enabled = os.getenv("ENABLE_JAX_TRACE", "1")
    if jax_trace_enabled == "0":
        return nullcontext()

    create_perfetto_link = os.getenv("JAX_TRACE_CREATE_PERFETTO_LINK", "1") == "1"

    return jax.profiler.trace(
        log_dir, create_perfetto_trace=True, create_perfetto_link=create_perfetto_link
    )


class CustomTestCase(unittest.TestCase):
    def _callTestMethod(self, method):
        max_retry = int(os.environ.get("SGLANG_TEST_MAX_RETRY", "1" if is_in_ci() else "0"))
        retry(
            lambda: super(CustomTestCase, self)._callTestMethod(method),
            max_retry=max_retry,
        )


def popen_launch_server(
    model: str,
    base_url: str,
    timeout: float,
    api_key: str | None = None,
    other_args: list[str] | None = None,
    env: dict | None = None,
    return_stdout_stderr: tuple | None = None,
    device: str = "tpu",
    check_cache_miss: bool = True,
):
    """Launch a server process with automatic device detection.

    Args:
        device: Device type ("auto", "cuda", "rocm" or "cpu").
                If "auto", will detect available platforms automatically.
    """
    other_args = list(other_args) if other_args is not None else []
    other_args += ["--device", str(device)]

    _, host, port = base_url.split(":")
    host = host[2:]

    module = "sgl_jax.launch_server"

    module_argv = [
        "-m",
        module,
        "--trust-remote-code",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        *[str(x) for x in other_args],
    ]

    if api_key:
        module_argv += ["--api-key", api_key]

    command = [sys.executable, *module_argv]

    print(f"command={' '.join(command)}")

    # Merge environment variables, avoid overwriting PATH / PYTHONPATH etc
    env_final = os.environ.copy()
    if env:
        env_final.update(env)

    if check_cache_miss:
        env_final.update({"SGLANG_JAX_ENABLE_CACHE_MISS_CHECK": "1"})

    if return_stdout_stderr:
        process = subprocess.Popen(
            command,
            stdout=return_stdout_stderr[0],
            stderr=return_stdout_stderr[1],
            env=env_final,
            text=True,
        )
    else:
        process = subprocess.Popen(command, stdout=None, stderr=None, env=env_final)

    start_time = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start_time < timeout:
            return_code = process.poll()
            if return_code is not None:
                # Server failed to start (non-zero exit code) or crashed
                raise Exception(
                    f"Server process exited with code {return_code}. Check server logs for errors."
                )

            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {api_key}",
                }
                response = session.get(
                    f"{base_url}/health_generate",
                    headers=headers,
                )
                if response.status_code == 200:
                    return process
            except requests.RequestException:
                pass

            return_code = process.poll()
            if return_code is not None:
                raise Exception(
                    f"Server unexpectedly exits ({return_code=}). Usually there will be error logs describing the cause far above this line."
                )

            time.sleep(10)

    kill_process_tree(process.pid)
    raise TimeoutError("Server failed to start within the timeout period.")


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        with suppress(psutil.NoSuchProcess):
            child.kill()

    if include_parent:
        with suppress(psutil.NoSuchProcess):
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)


def generate_server_args() -> ServerArgs:
    return ServerArgs(
        model_path="Qwen/Qwen-7B",
        tokenizer_path="Qwen/Qwen-7B",
        tokenizer_mode="auto",
        skip_tokenizer_init=False,
        load_format="auto",
        model_loader_extra_config="{}",
        trust_remote_code=True,
        context_length=None,
        is_embedding=False,
        revision=None,
        model_impl="auto",
        host="127.0.0.1",
        port=30000,
        skip_server_warmup=True,
        warmups=None,
        dtype="bfloat16",
        quantization=None,
        quantization_param_path=None,
        kv_cache_dtype="auto",
        mem_fraction_static=0.1,
        max_running_requests=None,
        max_total_tokens=None,
        max_prefill_tokens=4096,
        schedule_policy="fcfs",
        schedule_conservativeness=1.0,
        page_size=1,
        swa_full_tokens_ratio=0.8,
        disable_hybrid_swa_memory=False,
        device="tpu",
        tp_size=4,
        stream_interval=1,
        stream_output=False,
        random_seed=3,
        constrained_json_whitespace_pattern=None,
        watchdog_timeout=300,
        dist_timeout=None,
        download_dir="/tmp",
        sleep_on_idle=False,
        dp_size=1,
        log_level="info",
        log_level_http=None,
        log_requests=False,
        log_requests_level=0,
        crash_dump_folder=None,
        show_time_cost=False,
        bucket_time_to_first_token=None,
        bucket_inter_token_latency=None,
        bucket_e2e_request_latency=None,
        decode_log_interval=40,
        enable_request_time_stats_logging=False,
        kv_events_config=None,
        api_key=None,
        served_model_name="Qwen/Qwen-7B",
        file_storage_path="sglang_storage",
        enable_cache_report=False,
        reasoning_parser=None,
        tool_call_parser=None,
        dist_init_addr="0.0.0.0:10011",
        nnodes=1,
        node_rank=0,
        json_model_override_args="{}",
        preferred_sampling_params=None,
        disable_radix_cache=False,
        allow_auto_truncate=False,
        max_seq_len=4096,
        precompile_token_paddings=[1, 8],
        disable_precompile=False,
    )


# note: add fields value as you want, decrease existing fields is forbidden
def generate_schedule_batch(
    bs: int, num_tokens_per_req: int, mode: ForwardMode, model_runner: ModelRunner
) -> ScheduleBatch:
    req_for_1_bs = Req(
        rid="8ec8955e997f43b6aadf2557188e1508",
        origin_input_text="",
        origin_input_ids=[1] * num_tokens_per_req,
        sampling_params=SamplingParams(),
    )
    reqs = [req_for_1_bs] * bs
    input_ids = np.array([1] * num_tokens_per_req * bs, dtype=np.int32)
    extend_lens = [num_tokens_per_req] * bs
    seq_lens = np.array([num_tokens_per_req] * bs, dtype=np.int32)
    req_pool_indices = np.arange(bs, dtype=np.int32)
    return ScheduleBatch(
        reqs=reqs,
        forward_mode=mode,
        extend_lens=extend_lens,
        prefix_lens=[0] * bs,
        input_ids=input_ids,
        out_cache_loc=np.arange(1, sum(extend_lens) + 1, 1, dtype=np.int32),
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token_pool=model_runner.req_to_token_pool,
        mesh=model_runner.mesh,
    )


def get_max_total_num_tokens(tp_worker: ModelWorker) -> int:
    max_total_num_tokens, _, _, _, _, _, _, _, _, _, _ = tp_worker.get_worker_info()
    return max_total_num_tokens


def get_benchmark_args(
    base_url="",
    dataset_name="",
    dataset_path="",
    tokenizer=None,
    num_prompts=500,
    sharegpt_output_len=None,
    random_input_len=4096,
    random_output_len=2048,
    max_concurrency=256,
    random_range_ratio=1,
    sharegpt_context_len=None,
    request_rate=float("inf"),
    disable_stream=False,
    disable_ignore_eos=False,
    seed: int = 0,
    device="auto",
    pd_separated: bool = False,
    lora_name=None,
):
    return SimpleNamespace(
        backend="sgl-jax",
        base_url=base_url,
        host=None,
        port=None,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        model=None,
        tokenizer=tokenizer,
        num_prompts=num_prompts,
        sharegpt_output_len=sharegpt_output_len,
        sharegpt_context_len=sharegpt_context_len,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        max_concurrency=max_concurrency,
        random_range_ratio=random_range_ratio,
        request_rate=request_rate,
        multi=None,
        output_file=None,
        disable_tqdm=False,
        disable_stream=disable_stream,
        return_logprob=False,
        seed=seed,
        disable_ignore_eos=disable_ignore_eos,
        extra_request_body=None,
        apply_chat_template=False,
        profile=None,
        lora_name=lora_name,
        prompt_suffix="",
        device=device,
        pd_separated=pd_separated,
    )


def run_bench_serving(
    model,
    num_prompts,
    request_rate,
    other_server_args,
    dataset_name="random",
    dataset_path="",
    tokenizer=None,
    random_input_len=4096,
    random_output_len=2048,
    max_concurrency=256,
    random_range_ratio=1,
    sharegpt_context_len=None,
    disable_stream=False,
    disable_ignore_eos=False,
    need_warmup=False,
    seed: int = 0,
    device="auto",
    background_task: Callable[[str, asyncio.Event], Awaitable[None]] | None = None,
    lora_name: str | None = None,
):
    if device == "auto":
        device = "tpu"
    # Launch the server
    base_url = DEFAULT_URL_FOR_TEST
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_server_args,
    )

    # Run benchmark
    args = get_benchmark_args(
        base_url=base_url,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        num_prompts=num_prompts,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        max_concurrency=max_concurrency,
        random_range_ratio=random_range_ratio,
        sharegpt_context_len=sharegpt_context_len,
        request_rate=request_rate,
        disable_stream=disable_stream,
        disable_ignore_eos=disable_ignore_eos,
        seed=seed,
        device=device,
        lora_name=lora_name,
    )

    async def _run():
        if need_warmup:
            warmup_args = copy.deepcopy(args)
            warmup_args.num_prompts = 16
            await asyncio.to_thread(run_benchmark, warmup_args)

        start_event = asyncio.Event()
        stop_event = asyncio.Event()
        task_handle = (
            asyncio.create_task(background_task(base_url, start_event, stop_event))
            if background_task
            else None
        )

        try:
            start_event.set()
            result = await asyncio.to_thread(run_benchmark, args)
        finally:
            if task_handle:
                stop_event.set()
                await task_handle

        return result

    try:
        res = asyncio.run(_run())
    finally:
        kill_process_tree(process.pid)
        
        try:
            process.wait(timeout=5) 
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    assert res["completed"] == num_prompts
    return res


def run_bench_one_batch(model, other_args):
    """Launch a offline process with automatic device detection.

    Args:
        device: Device type ("auto", "cuda", "rocm" or "cpu").
                If "auto", will detect available platforms automatically.
    """
    # Auto-detect device if needed

    device = "tpu"
    print(f"Auto-configed device: {device}", flush=True)
    other_args += ["--device", str(device)]

    command = [
        "python3",
        "-m",
        "sgl_jax.bench_one_batch",
        "--batch-size",
        "1",
        "--input",
        "128",
        "--output",
        "8",
        *[str(x) for x in other_args],
    ]
    if model is not None:
        command += ["--model-path", model]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        stdout, stderr = process.communicate()
        output = stdout.decode()
        error = stderr.decode()
        print(f"Output: {output}", flush=True)
        print(f"Error: {error}", flush=True)

        # Return prefill_latency, decode_throughput, decode_latency
        prefill_line = output.split("\n")[-9]
        decode_line = output.split("\n")[-3]
        pattern = r"latency: (?P<latency>\d+\.\d+).*?throughput:\s*(?P<throughput>\d+\.\d+)"
        match = re.search(pattern, prefill_line)
        if match:
            prefill_latency = float(match.group("latency"))
        match = re.search(pattern, decode_line)
        if match:
            decode_latency = float(match.group("latency"))
            decode_throughput = float(match.group("throughput"))
    finally:
        kill_process_tree(process.pid)

    return prefill_latency, decode_throughput, decode_latency


def run_bench_offline_throughput(model, other_args):
    command = [
        "python3",
        "-m",
        "sgl_jax.bench_offline_throughput",
        "--num-prompts",
        "10",
        "--dataset-name",
        "random",
        "--random-input-len",
        "256",
        "--random-output-len",
        "256",
        "--trust-remote-code",
        "--skip-server-warmup",
        "--random-seed",
        "3",
        "--max-prefill-tokens",
        "4096",
        "--download-dir",
        "/tmp/",
        "--dtype",
        "bfloat16",
        "--precompile-bs-paddings",
        "16",
        "--precompile-token-paddings",
        "4096",
        "--page-size",
        "64",
        "--attention-backend",
        "fa",
        "--max-running-requests",
        "16",
        "--model-path",
        model,
        *[str(x) for x in other_args],
    ]

    print(f"{command=}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    output_lines = []
    output_throughput = -1

    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break

            line = line.rstrip()
            if line:
                print(f"[subprocess] {line}", flush=True)
                output_lines.append(line)

                if "Last generation throughput (tok/s):" in line:
                    output_throughput = float(line.split(":")[-1])

        process.wait()
    finally:
        if process.stdout:
            process.stdout.close()
        kill_process_tree(process.pid)

    return output_throughput


def run_bench_one_batch_server(
    model,
    base_url,
    server_args,
    bench_args,
    other_server_args,
    simulate_spec_acc_lens=None,
):
    from sgl_jax.bench_one_batch_server import run_benchmark

    if simulate_spec_acc_lens is not None:
        env = {**os.environ, "SIMULATE_ACC_LEN": str(simulate_spec_acc_lens)}
    else:
        env = None

    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_server_args,
        env=env,
    )
    try:
        run_benchmark(server_args=server_args, bench_args=bench_args)
    finally:
        kill_process_tree(process.pid)


def write_github_step_summary(content):
    if not os.environ.get("GITHUB_STEP_SUMMARY"):
        logging.warning("GITHUB_STEP_SUMMARY environment variable not set")
        return

    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(content)


def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


def calculate_rouge_l(output_strs_list1, output_strs_list2):
    """calculate the ROUGE-L score"""
    rouge_l_scores = []

    for s1, s2 in zip(output_strs_list1, output_strs_list2):
        lcs_len = lcs(s1, s2)
        precision = lcs_len / len(s1) if len(s1) > 0 else 0
        recall = lcs_len / len(s2) if len(s2) > 0 else 0
        fmeasure = (
            (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        )
        rouge_l_scores.append(fmeasure)

    return rouge_l_scores
