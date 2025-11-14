import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from sgl_jax.srt.utils import kill_process_tree


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60  # in minitues
    test_methods: Optional[List[str]] = (
        None  # Optional: specific test methods to run (e.g., ["TestClass.test_method"])
    )


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = None,
):
    """Run a function with timeout."""
    ret_value = []

    def _target_func():
        ret_value.append(func(*args, **(kwargs or {})))

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError()

    if not ret_value:
        raise RuntimeError()

    return ret_value[0]


def cleanup_model_cache():
    shm_dir = "/dev/shm"
    if os.path.exists(shm_dir):
        for item in os.listdir(shm_dir):
            if item.startswith("model"):
                model_path = os.path.join(shm_dir, item)
                if os.path.isdir(model_path):
                    try:
                        print(f"\nCleaning up model cache at {model_path}...", flush=True)
                        shutil.rmtree(model_path)
                        print(f"Model cache cleaned successfully.\n", flush=True)
                    except Exception as e:
                        print(f"Failed to clean model cache: {e}\n", flush=True)


def run_unittest_files(files: List[TestFile], timeout_per_file: float):
    tic = time.perf_counter()
    success = True

    for i, file in enumerate(files):
        filename, estimated_time = file.name, file.estimated_time
        process = None

        def run_one_file(filename):
            nonlocal process

            filename = os.path.join(os.getcwd(), filename)
            tic = time.perf_counter()

            # Check if specific test methods are specified
            if file.test_methods:
                # Run specific test methods using unittest module syntax
                # Convert file path to module path (e.g., test/srt/test_file.py -> test.srt.test_file)
                module_path = (
                    filename.replace(os.getcwd() + "/", "").replace(".py", "").replace("/", ".")
                )

                print(
                    f".\n.\nBegin ({i}/{len(files) - 1}):\nRunning specific test methods from {filename}\n",
                    flush=True,
                )

                # Run each test method sequentially
                for method in file.test_methods:
                    test_path = f"{module_path}.{method}"
                    print(f"Running: python3 -m unittest {test_path}\n", flush=True)

                    process = subprocess.Popen(
                        ["uv", "run", "python3", "-m", "unittest", test_path],
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        env=os.environ,
                    )
                    process.wait()

                    # If any test fails, return immediately
                    if process.returncode != 0:
                        print(
                            f"Test {test_path} failed with return code {process.returncode}\n",
                            flush=True,
                        )
                        cleanup_model_cache()
                        return process.returncode
            else:
                # Run entire file (existing behavior)
                print(
                    f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {filename}\n.\n.\n",
                    flush=True,
                )

                process = subprocess.Popen(
                    ["uv", "run", "python3", filename],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    env=os.environ,
                )
                process.wait()

            elapsed = time.perf_counter() - tic
            print(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n",
                flush=True,
            )

            cleanup_model_cache()

            return process.returncode

        try:
            ret_code = run_with_timeout(run_one_file, args=(filename,), timeout=timeout_per_file)
            assert ret_code == 0, f"expected return code 0, but {filename} returned {ret_code}"
        except TimeoutError:
            kill_process_tree(process.pid)
            time.sleep(5)
            print(
                f"\nTimeout after {timeout_per_file} seconds when running {filename}\n",
                flush=True,
            )
            success = False
            break

    if success:
        print(f"Success. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)
    else:
        print(f"Fail. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)

    return 0 if success else -1


suites = {
    "nightly": [],
    "sglang_dependency_test": [],
    "kernel-performance-test-tpu-v6e-1": [
        TestFile("benchmark/kernels/flash_attention/bench_flashattention.py", 5),
        TestFile("benchmark/kernels/megablox_gmm/bench_megablox_gmm.py", 2),
        TestFile("benchmark/kernels/update_kv_cache/bench_update_kv_cache.py", 3),
    ],
    "accuracy-test-tpu-v6e-1": [
        TestFile("test/srt/test_eval_accuracy_large.py", 5, ["TestEvalAccuracyLarge.test_mmlu"]),
    ],
    "accuracy-test-tpu-v6e-4": [
        TestFile(
            "test/srt/test_moe_eval_accuracy_large.py", 40, ["TestMoEEvalAccuracyLarge.test_mmlu"]
        ),
    ],
    "performance-test-tpu-v6e-1": [
        TestFile(
            "test/srt/test_bench_serving.py",
            7,
            [
                "TestBenchServing.test_ttft_default",
                "TestBenchServing.test_itl_default",
                "TestBenchServing.test_input_throughput_default",
                "TestBenchServing.test_output_throughput_default",
            ],
        )
    ],
    "performance-test-tpu-v6e-4": [
        TestFile(
            "test/srt/test_bench_serving.py",
            13,
            [
                "TestBenchServing.test_ttft_default_tp_4",
                "TestBenchServing.test_itl_default_tp_4",
                "TestBenchServing.test_input_throughput_default_tp_4",
                "TestBenchServing.test_output_throughput_default_tp_4",
            ],
        ),
        TestFile(
            "test/srt/test_bench_serving.py",
            35,
            [
                "TestBenchServing.test_moe_ttft_default",
                "TestBenchServing.test_moe_itl_default",
                "TestBenchServing.test_moe_input_throughput_default",
                "TestBenchServing.test_moe_output_throughput_default",
            ],
        ),
    ],
    "e2e-test-tpu-v6e-1": [
        # openai_server e2e test
        TestFile("test/srt/openai_server/basic/test_protocol.py", 0.1),
        TestFile("test/srt/openai_server/basic/test_serving_chat.py", 0.1),
        TestFile("test/srt/openai_server/basic/test_serving_completions.py", 0.1),
        TestFile("test/srt/openai_server/basic/test_openai_server.py", 1),
        TestFile("test/srt/test_srt_engine.py", 1),
    ],
    "e2e-test-tpu-v6e-4": [
        TestFile("test/srt/openai_server/basic/test_tool_calls.py", 3),
        TestFile("test/srt/test_features.py", 3),
        TestFile("test/srt/test_chunked_prefill_size.py", 5),
        # TestFile("test/srt/test_sliding_window_attention.py", 30), # add after gpt-oss supported
    ],
}


def auto_partition(files, rank, size):
    """
    Partition files into size sublists with approximately equal sums of estimated times
    using stable sorting, and return the partition for the specified rank.

    Args:
        files (list): List of file objects with estimated_time attribute
        rank (int): Index of the partition to return (0 to size-1)
        size (int): Number of partitions

    Returns:
        list: List of file objects in the specified rank's partition
    """
    weights = [f.estimated_time for f in files]

    if not weights or size <= 0 or size > len(weights):
        return []

    # Create list of (weight, original_index) tuples
    # Using negative index as secondary key to maintain original order for equal weights
    indexed_weights = [(w, -i) for i, w in enumerate(weights)]
    # Stable sort in descending order by weight
    # If weights are equal, larger (negative) index comes first (i.e., earlier original position)
    indexed_weights = sorted(indexed_weights, reverse=True)

    # Extract original indices (negate back to positive)
    indexed_weights = [(w, -i) for w, i in indexed_weights]

    # Initialize partitions and their sums
    partitions = [[] for _ in range(size)]
    sums = [0.0] * size

    # Greedy approach: assign each weight to partition with smallest current sum
    for weight, idx in indexed_weights:
        # Find partition with minimum sum
        min_sum_idx = sums.index(min(sums))
        partitions[min_sum_idx].append(idx)
        sums[min_sum_idx] += weight

    # Return the files corresponding to the indices in the specified rank's partition
    indices = partitions[rank]
    return [files[i] for i in indices]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1800,
        help="The time limit for running one file in seconds.",
    )
    arg_parser.add_argument(
        "--suite",
        type=str,
        default=list(suites.keys())[0],
        choices=list(suites.keys()) + ["all"],
        help="The suite to run",
    )
    arg_parser.add_argument(
        "--range-begin",
        type=int,
        default=0,
        help="The begin index of the range of the files to run.",
    )
    arg_parser.add_argument(
        "--range-end",
        type=int,
        default=None,
        help="The end index of the range of the files to run.",
    )
    arg_parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    arg_parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    args = arg_parser.parse_args()
    print(f"{args=}")

    if args.suite == "all":
        files = glob.glob("**/test_*.py", recursive=True)
    else:
        files = suites[args.suite]

    if args.auto_partition_size:
        files = auto_partition(files, args.auto_partition_id, args.auto_partition_size)
    else:
        files = files[args.range_begin : args.range_end]

    print("The running tests are ", [f.name for f in files])

    exit_code = run_unittest_files(files, args.timeout_per_file)
    exit(exit_code)
