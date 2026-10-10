"""
Benchmark the latency of running a single batch with a server.

This script launches a server and uses the HTTP interface.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

Usage:
python3 -m sgl_jax.bench_one_batch_server --model meta-llama/Meta-Llama-3.1-8B --batch-size 1 16 64 --input-len 1024 --output-len 8

python3 -m sgl_jax.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8
python3 -m sgl_jax.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8 --show-report --profile --profile-by-stage

The input/output cost columns in --show-report are derived from the *server's* tp_size
(read from /get_server_info) and the per-chip-hour price of the selected TPU generation
(built-in table from the official Cloud TPU pricing sheet, 3-year commitment by default).
The assumptions are printed above the table.
python3 -m sgl_jax.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8 --show-report --chip ironwood --pricing-tier 3yr
"""

import argparse
import dataclasses
import itertools
import json
import math
import multiprocessing
import os
import time

import requests

from sgl_jax.bench_serving import get_tokenizer, sample_random_requests
from sgl_jax.profiler import run_profile
from sgl_jax.srt.entrypoints import http_server
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import is_in_ci, write_github_step_summary

# ---------------------------------------------------------------------------
# Cloud TPU per-chip-hour pricing, USD.
#
# Source: official Cloud TPU pricing sheet, snapshot taken Oct 2026.
# For each generation we take the cheapest listed US region. The default tier
# used for the cost columns is the 3-year commitment price.
#
# devices_per_chip = number of JAX devices (TensorCores) one *billable chip*
# exposes. The server's tp_size counts JAX devices, so
# num_chips = ceil(tp_size / devices_per_chip).
# ---------------------------------------------------------------------------
TPU_PRICING_SNAPSHOT = "Oct 2026"
TPU_PRICING_SOURCE = "official Cloud TPU pricing sheet"
TPU_PRICING_TIERS = ("3yr", "1yr", "on_demand")
TPU_PRICING_TIER_LABELS = {
    "3yr": "3-year commitment",
    "1yr": "1-year commitment",
    "on_demand": "on-demand",
}
TPU_PRICING = {
    # key: (display name, region, devices_per_chip, {tier: $/chip-hr})
    "ironwood": ("Ironwood (v7x)", "us-central1", 2, {"on_demand": 12.00, "1yr": 8.40, "3yr": 5.40}),
    "trillium": ("Trillium (v6e)", "us-east1", 1, {"on_demand": 2.70, "1yr": 1.89, "3yr": 1.22}),
    "v5p": ("TPU v5p", "us-east5", 1, {"on_demand": 4.20, "1yr": 2.94, "3yr": 1.89}),
    "v5e": ("TPU v5e", "us-central1", 1, {"on_demand": 1.20, "1yr": 0.84, "3yr": 0.54}),
    "v4": ("TPU v4 pod", "us-central2", 1, {"on_demand": 3.22, "1yr": 2.0286, "3yr": 1.449}),
    "v3": ("TPU v3 pod", "europe-west4", 2, {"on_demand": 2.00, "1yr": 1.26, "3yr": 0.90}),
    "v2": ("TPU v2 pod", "us-central1", 2, {"on_demand": 1.50, "1yr": 0.945, "3yr": 0.675}),
}
TPU_CHIP_ALIASES = {"v7x": "ironwood", "v7": "ironwood", "v6e": "trillium"}


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: tuple[int] = (1,)
    input_len: tuple[int] = (1024,)
    output_len: tuple[int] = (16,)
    temperature: float = 0.0
    return_logprob: bool = False
    client_stream_interval: int = 1
    input_len_step_percentage: float = 0.0
    result_filename: str = "result.jsonl"
    base_url: str = ""
    skip_warmup: bool = False
    show_report: bool = False
    profile: bool = False
    profile_by_stage: bool = False
    api_type: str = "native"  # "native" or "openai"
    # Cost model for the --show-report table.
    #   hourly cost = price_per_chip_hour * num_chips
    #   price       = --hourly-cost-per-chip if > 0, else TPU_PRICING[chip][tier]
    #   num_chips   = --num-chips if > 0, else ceil(server tp_size / devices_per_chip)
    chip: str = "ironwood"  # key into TPU_PRICING (or alias)
    pricing_tier: str = "3yr"  # 3yr | 1yr | on_demand
    hourly_cost_per_chip: float = 0.0  # $/chip-hour override; 0 = from TPU_PRICING
    devices_per_chip: int = 0  # JAX devices per billable chip override; 0 = from TPU_PRICING
    num_chips: int = 0  # 0 = derive from server tp_size
    input_util: float = 0.7  # assumed prefill utilization for input cost

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument("--batch-size", type=int, nargs="+", default=BenchArgs.batch_size)
        parser.add_argument("--input-len", type=int, nargs="+", default=BenchArgs.input_len)
        parser.add_argument("--output-len", type=int, nargs="+", default=BenchArgs.output_len)
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument(
            "--client-stream-interval",
            type=int,
            default=BenchArgs.client_stream_interval,
        )
        parser.add_argument(
            "--input-len-step-percentage",
            type=float,
            default=BenchArgs.input_len_step_percentage,
        )
        parser.add_argument("--result-filename", type=str, default=BenchArgs.result_filename)
        parser.add_argument("--base-url", type=str, default=BenchArgs.base_url)
        parser.add_argument("--skip-warmup", action="store_true")
        parser.add_argument("--show-report", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument("--profile-by-stage", action="store_true")
        parser.add_argument(
            "--api-type",
            type=str,
            default=BenchArgs.api_type,
            choices=["native", "openai"],
            help="API type to use: 'native' for /generate or 'openai' for /v1/completions",
        )
        parser.add_argument(
            "--chip",
            type=str,
            default=BenchArgs.chip,
            help="TPU generation used for pricing: "
            + ", ".join(sorted(TPU_PRICING) + sorted(TPU_CHIP_ALIASES))
            + f" (default: {BenchArgs.chip}).",
        )
        parser.add_argument(
            "--pricing-tier",
            type=str,
            default=BenchArgs.pricing_tier,
            choices=TPU_PRICING_TIERS,
            help=f"Which price column of the {TPU_PRICING_SOURCE} to use "
            f"(default: {BenchArgs.pricing_tier} = 3-year commitment).",
        )
        parser.add_argument(
            "--hourly-cost-per-chip",
            type=float,
            default=BenchArgs.hourly_cost_per_chip,
            help="Override price in $/hour of one billable chip. "
            "0 (default) = look up --chip / --pricing-tier in the built-in table.",
        )
        parser.add_argument(
            "--devices-per-chip",
            type=int,
            default=BenchArgs.devices_per_chip,
            help="Override JAX devices (cores) per billable chip, used to convert the "
            "server's tp_size into a chip count. 0 (default) = from the built-in table "
            "(v7x: 2, v6e/v5p/v5e/v4: 1).",
        )
        parser.add_argument(
            "--num-chips",
            type=int,
            default=BenchArgs.num_chips,
            help="Explicit number of billable chips. If 0 (default), derived as "
            "ceil(server tp_size / devices_per_chip).",
        )
        parser.add_argument(
            "--input-util",
            type=float,
            default=BenchArgs.input_util,
            help="Assumed prefill utilization when computing input cost.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(**{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs})


def launch_server_internal(server_args):
    try:
        http_server.launch(server_args)
    except Exception as e:
        raise e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_server_process(server_args: ServerArgs):
    proc = multiprocessing.Process(target=launch_server_internal, args=(server_args,))
    proc.start()
    base_url = f"http://{server_args.host}:{server_args.port}"
    timeout = 600

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
            }
            response = requests.get(f"{base_url}/v1/models", headers=headers)
            if response.status_code == 200:
                return proc, base_url
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")


def run_one_case(
    url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    temperature: float,
    return_logprob: bool,
    stream_interval: int,
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer,
    profile: bool = False,
    profile_by_stage: bool = False,
    api_type: str = "native",
):
    requests.post(url + "/flush_cache")

    # Determine whether to use text or input_ids based on API type
    return_text = api_type == "openai"
    input_requests = sample_random_requests(
        input_len=input_len,
        output_len=output_len,
        num_prompts=batch_size,
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path="",
        random_sample=True,
        return_text=return_text,
    )

    use_structured_outputs = False
    if use_structured_outputs:
        texts = []
        for _ in range(batch_size):
            texts.append(
                "Human: What is the capital city of france? can you give as many trivial information as possible about that city? answer in json.\n"
                * 50
                + "Assistant:"
            )
        json_schema = "$$ANY$$"
    else:
        json_schema = None

    profile_link = None
    if profile:
        profile_link: str = run_profile(url, 3, ["CPU", "GPU"], None, None, profile_by_stage)

    tic = time.perf_counter()

    if api_type == "openai":
        # Use OpenAI API - send requests as a batch with n parameter
        # Convert prompts to a single request with n > 1 if batch_size > 1
        if batch_size == 1:
            request_data = {
                "model": "default",
                "prompt": input_requests[0].prompt,
                "max_tokens": output_len,
                "temperature": temperature,
                "logprobs": 1 if return_logprob else None,
                "stream": True,
            }
        else:
            # For batch, we'll send the first prompt with n=batch_size
            # Note: This assumes all prompts should be the same, which may not be ideal
            # For different prompts, we'd need to send separate requests
            request_data = {
                "model": "default",
                "prompt": [req.prompt for req in input_requests],
                "max_tokens": output_len,
                "temperature": temperature,
                "logprobs": 1 if return_logprob else None,
                "stream": True,
            }

        response = requests.post(
            url + "/v1/completions",
            json=request_data,
            stream=True,
        )

        # Parse OpenAI streaming response
        ttft = 0.0
        first_token_received = False
        total_chunks = 0
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                if "error" in data:
                    raise RuntimeError(f"Request has failed. {data}.")

                # Track first token time (across all choices in batch)
                if data.get("choices") and len(data["choices"]) > 0:
                    # Check if any choice has generated text
                    for choice in data["choices"]:
                        if choice.get("text") and not first_token_received:
                            ttft = time.perf_counter() - tic
                            first_token_received = True
                            break
                    total_chunks += 1
    else:
        # Use native API
        response = requests.post(
            url + "/generate",
            json={
                "input_ids": [req.prompt for req in input_requests],
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": output_len,
                    "ignore_eos": True,
                    "json_schema": json_schema,
                    "stream_interval": stream_interval,
                },
                "return_logprob": return_logprob,
                "stream": True,
            },
            stream=True,
        )

        # The TTFT of the last request in the batch
        ttft = 0.0
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                if "error" in data:
                    raise RuntimeError(f"Request has failed. {data}.")

                assert (
                    data["meta_info"]["finish_reason"] is None
                    or data["meta_info"]["finish_reason"]["type"] == "length"
                )
                if data["meta_info"]["completion_tokens"] == 1:
                    ttft = time.perf_counter() - tic

    latency = time.perf_counter() - tic
    input_throughput = batch_size * input_len / ttft
    output_throughput = batch_size * output_len / (latency - ttft)
    overall_throughput = batch_size * (input_len + output_len) / latency

    server_info = requests.get(url + "/get_server_info").json()
    acc_length = server_info["internal_states"][0].get("avg_spec_accept_length", None)
    last_gen_throughput = server_info["internal_states"][0]["last_gen_throughput"]

    print(f"batch size: {batch_size}")
    print(f"input_len: {input_len}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"ttft: {ttft:.2f} s")
    print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s")

    if result_filename:
        with open(result_filename, "a") as fout:
            res = {
                "run_name": run_name,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "latency": round(latency, 4),
                "output_throughput": round(output_throughput, 2),
                "overall_throughput": round(overall_throughput, 2),
                "last_gen_throughput": round(last_gen_throughput, 2),
            }
            fout.write(json.dumps(res) + "\n")

    return (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        profile_link if profile else None,
    )


def resolve_hourly_cost(server_info: dict, server_args: ServerArgs, bench_args: BenchArgs):
    """Build the CostModel used for the $/1M-token columns in --show-report.

    The number of accelerators is taken from the *running server* (via
    /get_server_info), not from this script's own CLI args. When benchmarking
    an existing server with --base-url, the client-side ``server_args.tp_size``
    is just the default (1), which would silently under-price the deployment.

    On TPUs ``tp_size`` counts JAX devices (cores) while billing is per chip,
    so devices are converted to chips with ``devices_per_chip`` unless
    ``--num-chips`` is given explicitly.
    """
    server_tp_size = None
    tp_source = "server /get_server_info"
    if isinstance(server_info, dict):
        if "tp_size" in server_info:
            server_tp_size = server_info["tp_size"]
        elif "decode" in server_info and server_info["decode"]:
            # PD-disaggregated deployments: price the decode side.
            server_tp_size = server_info["decode"][0].get("tp_size")
            tp_source = "server /get_server_info (decode side)"
        elif "prefill" in server_info and server_info["prefill"]:
            server_tp_size = server_info["prefill"][0].get("tp_size")
            tp_source = "server /get_server_info (prefill side)"

    if server_tp_size is None:
        server_tp_size = server_args.tp_size
        tp_source = "client-side --tp-size (FALLBACK, server did not report tp_size)"
        print(
            "WARNING: could not read tp_size from /get_server_info; falling back to "
            f"client-side --tp-size={server_tp_size}. Cost columns may be wrong; "
            "pass --num-chips to override."
        )

    # Chip / pricing lookup.
    chip_key = bench_args.chip.lower()
    chip_key = TPU_CHIP_ALIASES.get(chip_key, chip_key)
    if chip_key not in TPU_PRICING:
        raise ValueError(
            f"Unknown --chip {bench_args.chip!r}. Known: "
            + ", ".join(sorted(TPU_PRICING) + sorted(TPU_CHIP_ALIASES))
        )
    chip_name, region, table_devices_per_chip, prices = TPU_PRICING[chip_key]
    tier = bench_args.pricing_tier

    if bench_args.hourly_cost_per_chip > 0:
        price_per_chip_hour = bench_args.hourly_cost_per_chip
        price_source = "--hourly-cost-per-chip override"
    else:
        price_per_chip_hour = prices[tier]
        price_source = (
            f"{TPU_PRICING_SOURCE}, {TPU_PRICING_TIER_LABELS[tier]} column, "
            f"{region}, snapshot {TPU_PRICING_SNAPSHOT}"
        )

    if bench_args.devices_per_chip > 0:
        devices_per_chip = bench_args.devices_per_chip
        devices_source = "--devices-per-chip override"
    else:
        devices_per_chip = table_devices_per_chip
        devices_source = f"built-in table for {chip_name}"

    if bench_args.num_chips > 0:
        num_chips = bench_args.num_chips
        chips_source = "--num-chips override"
    else:
        num_chips = max(1, math.ceil(server_tp_size / devices_per_chip))
        chips_source = f"ceil(tp_size {server_tp_size} / {devices_per_chip} devices per chip)"

    return CostModel(
        chip_key=chip_key,
        chip_name=chip_name,
        region=region,
        pricing_tier=tier,
        price_per_chip_hour=price_per_chip_hour,
        price_source=price_source,
        devices_per_chip=devices_per_chip,
        devices_source=devices_source,
        server_tp_size=server_tp_size,
        tp_source=tp_source,
        num_chips=num_chips,
        chips_source=chips_source,
        input_util=bench_args.input_util,
    )


@dataclasses.dataclass
class CostModel:
    """Everything that goes into the $/1M-token columns, with provenance."""

    chip_key: str
    chip_name: str
    region: str
    pricing_tier: str
    price_per_chip_hour: float
    price_source: str
    devices_per_chip: int
    devices_source: str
    server_tp_size: int
    tp_source: str
    num_chips: int
    chips_source: str
    input_util: float

    @property
    def hourly_cost(self) -> float:
        return self.price_per_chip_hour * self.num_chips

    def input_cost_per_1m(self, input_throughput: float) -> float:
        return 1e6 / (input_throughput * self.input_util) / 3600 * self.hourly_cost

    def output_cost_per_1m(self, output_throughput: float) -> float:
        return 1e6 / output_throughput / 3600 * self.hourly_cost

    def assumptions_markdown(self) -> str:
        lines = [
            "**Cost assumptions** (used for the `input cost` / `output cost` columns)",
            "",
            f"- Chip: **{self.chip_name}**, priced per chip-hour.",
            f"- Price: **${self.price_per_chip_hour:.2f}/chip-hr** — {self.price_source}.",
            f"- Devices per chip: {self.devices_per_chip} ({self.devices_source}).",
            f"- Server tp_size: {self.server_tp_size} ({self.tp_source}).",
            f"- Billable chips: **{self.num_chips}** = {self.chips_source}.",
            f"- Hourly cost: **${self.hourly_cost:.2f}/hr** "
            f"= {self.num_chips} chips x ${self.price_per_chip_hour:.2f}/chip-hr.",
            f"- Input cost  = 1e6 / (input_tput x {self.input_util} util) / 3600 x hourly cost.",
            "- Output cost = 1e6 / output_tput / 3600 x hourly cost.",
            "",
            f"> Prices are from the {TPU_PRICING_SOURCE} as of {TPU_PRICING_SNAPSHOT}; "
            "re-check before quoting externally. Override with --chip, --pricing-tier, "
            "--hourly-cost-per-chip, --devices-per-chip, --num-chips.",
        ]
        return "\n".join(lines)


def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    if bench_args.base_url:
        proc, base_url = None, bench_args.base_url
    else:
        proc, base_url = launch_server_process(server_args)

    server_info = requests.get(base_url + "/get_server_info").json()
    if "tokenizer_path" in server_info:
        tokenizer_path = server_info["tokenizer_path"]
    elif "prefill" in server_info:
        tokenizer_path = server_info["prefill"][0]["tokenizer_path"]
    tokenizer = get_tokenizer(tokenizer_path)

    # warmup
    if not bench_args.skip_warmup:
        print("=" * 8 + " Warmup Begin " + "=" * 8)
        run_one_case(
            base_url,
            batch_size=16,
            input_len=1024,
            output_len=16,
            temperature=bench_args.temperature,
            return_logprob=bench_args.return_logprob,
            stream_interval=bench_args.client_stream_interval,
            input_len_step_percentage=bench_args.input_len_step_percentage,
            run_name="",
            result_filename="",
            tokenizer=tokenizer,
            api_type=bench_args.api_type,
        )
        print("=" * 8 + " Warmup End   " + "=" * 8 + "\n")

    # benchmark
    result = []
    bench_result = []
    try:
        for bs, il, ol in itertools.product(
            bench_args.batch_size, bench_args.input_len, bench_args.output_len
        ):
            result.append(
                run_one_case(
                    base_url,
                    bs,
                    il,
                    ol,
                    temperature=bench_args.temperature,
                    return_logprob=bench_args.return_logprob,
                    stream_interval=bench_args.client_stream_interval,
                    input_len_step_percentage=bench_args.input_len_step_percentage,
                    run_name=bench_args.run_name,
                    result_filename=bench_args.result_filename,
                    tokenizer=tokenizer,
                    api_type=bench_args.api_type,
                )
            )

        if bench_args.profile:
            try:
                for bs, il, ol in itertools.product(
                    bench_args.batch_size, bench_args.input_len, bench_args.output_len
                ):
                    bench_result.append(
                        (
                            run_one_case(
                                base_url,
                                bs,
                                il,
                                ol,
                                temperature=bench_args.temperature,
                                return_logprob=bench_args.return_logprob,
                                stream_interval=bench_args.client_stream_interval,
                                input_len_step_percentage=bench_args.input_len_step_percentage,
                                run_name=bench_args.run_name,
                                result_filename=bench_args.result_filename,
                                tokenizer=tokenizer,
                                profile=bench_args.profile,
                                profile_by_stage=bench_args.profile_by_stage,
                                api_type=bench_args.api_type,
                            )[-1],
                        )
                    )
                result = [t1[:-1] + t2 for t1, t2 in zip(result, bench_result)]
            except Exception as e:
                print(f"Error profiling, there will be no profile trace dump: {e}")
    finally:
        if proc:
            kill_process_tree(proc.pid)

    print(f"\nResults are saved to {bench_args.result_filename}")

    if not bench_args.show_report:
        return

    cost = resolve_hourly_cost(server_info, server_args, bench_args)

    summary = "\n" + cost.assumptions_markdown() + "\n\n"
    summary += f"Input lens: {bench_args.input_len}. Output lens: {bench_args.output_len}.\n"
    summary += "| batch size | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) |"

    if bench_args.profile:
        summary += " profile |"

    summary += "\n"
    summary += "| ---------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ |"

    if bench_args.profile:
        summary += "-------------|"
    summary += "\n"

    for (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        trace_link,
    ) in result:
        accept_length = round(acc_length, 2) if acc_length is not None else "n/a"
        line = (
            f"| {batch_size} | "
            f"{latency:.2f} | "
            f"{input_throughput:.2f} | "
            f"{output_throughput:.2f} | "
            f"{accept_length} | "
            f"{1 / (output_throughput / batch_size) * 1000:.2f} | "
            f"{cost.input_cost_per_1m(input_throughput):.2f} | "
            f"{cost.output_cost_per_1m(output_throughput):.2f} |"
        )
        if trace_link:
            line += f" [Profile]({trace_link}) |"
        line += "\n"
        summary += line

    # print metrics table
    print(summary)

    if is_in_ci():
        write_github_step_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    run_benchmark(server_args, bench_args)
