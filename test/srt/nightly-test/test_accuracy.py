import csv
import os
import sys
import unittest
from types import SimpleNamespace

from evalscope import TaskConfig, run_task

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    GEMMA2_2B_IT,
    QWEN2_5_7B_INSTRUCT,
    QWEN3_8B,
    QWEN3_CODER_30B_A3B_INSTRUCT,
    QWEN_7B,
    CustomTestCase,
    bailing_moe,
    popen_launch_server,
)


class TestModelAccuracy(CustomTestCase):
    def test_qwen_7b(self):
        model = QWEN_7B
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_results.csv"

        # launch server
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "1",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                # if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                #     dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)

    def test_qwen3_8b(self):
        model = QWEN3_8B
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_results.csv"

        # launch server
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "1",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                # if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                #     dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)

    def test_DEEPSEEK_R1_DISTILL_QWEN_1_5B(self):
        model = DEEPSEEK_R1_DISTILL_QWEN_1_5B
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_results.csv"
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "1",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
                {"name": "math_500", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                # if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                #     dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)

    def test_GEMMA2_2B_IT(self):
        model = GEMMA2_2B_IT
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_results.csv"
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "1",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "1",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                # if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                #     dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)

    def test_qwen_7b_tp_4(self):
        model = QWEN_7B
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_tp_4_results.csv"
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "4",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                    dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)

    def test_qwen3_8b_tp_4(self):
        model = QWEN3_8B
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_tp_4_results.csv"
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "4",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                    dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)

    def test_GEMMA2_2B_IT_tp_4(self):
        model = GEMMA2_2B_IT
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_tp_4_results.csv"
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "4",
                "--disable-hybrid-swa-memory",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                    dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)

    def test_QWEN3_CODER_30B_A3B_INSTRUCT_tp_4(self):
        model = QWEN3_CODER_30B_A3B_INSTRUCT
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_tp_4_results.csv"
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "2",
                "--ep-size",
                "2",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                    dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)

    def test_DEEPSEEK_R1_DISTILL_QWEN_1_5B_tp_4(self):
        model = DEEPSEEK_R1_DISTILL_QWEN_1_5B
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_tp_4_results.csv"
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "4",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
                {"name": "math_500", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                    dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)

    def test_bailing_moe_tp_2_ep2(self):
        model = bailing_moe
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_tp_2_ep_2results.csv"
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "2",
                "--ep-size",
                "2",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                    dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)

    def test_QWEN3_CODER_30B_A3B_INSTRUCT_tp_2_ep_2(self):
        model = QWEN3_CODER_30B_A3B_INSTRUCT
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        csv_file_path = "./test/nightly_test_output/benchmark/benchmark_tp_2_ep_2results.csv"
        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--max-prefill-tokens",
                "16384",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--max-running-requests",
                "256",
                "--attention-backend",
                "fa",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "2048",
                "--tp-size",
                "2",
                "--ep-size",
                "2",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0},
                {"name": "mmlu", "threshold": 0},
                {"name": "mmlu_pro", "threshold": 0},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {}
                if dataset_name == "mmlu" or dataset_name == "modelscope/mmlu":
                    dataset_args = {"mmlu": {"subset_list": ["global_facts"]}}

                config = TaskConfig(
                    model=model,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="service",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{model} Running eval for {dataset_name}")
                results = run_task(config)
                print(f"SDK Results: {results}")
                if dataset_name in results:
                    report = results[dataset_name]
                    score = report.score
                    try:
                        rows = []
                        fieldnames = ["Model"]

                        if os.path.exists(csv_file_path):
                            with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                if reader.fieldnames:
                                    fieldnames = reader.fieldnames
                                rows = list(reader)

                        if dataset_name not in fieldnames:
                            fieldnames.append(dataset_name)

                        model_found = False
                        for row in rows:
                            if row.get("Model") == model:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": model, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(f"Updated CSV {csv_file_path}: {model} - {dataset_name} = {score}")

                    except Exception as e:
                        print(f"Warning: Failed to update CSV file: {e}")

                    print(f"[{dataset_name}] Final Score: {score}")
                    self.assertGreater(
                        score,
                        threshold,
                        f"{dataset_name} score {score} is too low (target: {threshold})",
                    )
                else:
                    self.fail(f"Dataset {dataset_name} not found in results: {results.keys()}")

        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
