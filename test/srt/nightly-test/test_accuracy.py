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
    BAILING_MOE,
    DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    GEMMA2_2B_IT,
    QWEN2_5_7B_INSTRUCT,
    QWEN3_8B,
    QWEN3_MOE_30B,
    QWEN_7B,
    CustomTestCase,
    popen_launch_server,
)


class TestModelAccuracy(CustomTestCase):
    BASIC_SERVER_ARGS = [
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
        "--mem-fraction-static",
        "0.8",
        "--use-sort-for-toppk-minp",
    ]

    def get_dataset_args(self, dataset_name):
        dataset_configs = {
            "gsm8k": {
                "prompt_template": "Question: {query}\nLet's think step by step\nAnswer:",
                "train_split": None,
            },
            "mmlu": {
                "prompt_template": "Answer the following multiple choice question about {subset_name}. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{query}",
            },
            "mmlu_pro": {
                "prompt_template": 'The following are multiple choice questions (with answers) about {subset_name}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n{query}',
            },
            "aime24": {
                "prompt_template": "The following is a multiple choice question. Choose the correct answer.\n\nQuestion: {query}\nOptions:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n\nAnswer:",
            },
            "aime24": {
                "prompt_template": "{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            },
            "math500": {
                "prompt_template": "{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            },
        }

        if dataset_name not in dataset_configs:
            print(
                f"Warning: No specific config for dataset '{dataset_name}', using default config."
            )
            return {
                "prompt_template": "{query}",
                "few_shot_num": 5,
            }

        return dataset_configs[dataset_name].copy()

    def get_generation_config(self, model_name):
        base_config = {
            "max_tokens": 2048,
            "temperature": 0.0,
        }

        model_configs = {
            "QWEN_7B": {
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 0,
                "do_sample": True,
            },
            "QWEN3_8B": {
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
            },
            "DEEPSEEK_R1_DISTILL_QWEN_1_5B": {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
            },
            "QWEN3_MOE_30B": {
                "do_sample": True,
                "top_p": 0.8,
                "top_k": 20,
                "temperature": 0.7,
            },
        }

        if model_name in model_configs:
            config = {**base_config, **model_configs[model_name]}
        else:
            config_found = False
            for model_prefix, config_values in model_configs.items():
                if model_name.lower().startswith(model_prefix):
                    config = {**base_config, **config_values}
                    config_found = True
                    break

            if not config_found:
                print(
                    f"Warning: No specific config for model '{model_name}', using general config."
                )
                config = {**base_config, **model_configs["general"]}

        return config

    def test_qwen_7b(self):
        # args setting
        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN_7B
        model_dir_name = "QWEN_7B"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)
        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        output_dir = os.getenv("BENCH_OUTPUT_DIR", "./test/nightly_test_output/benchmark/local_run")
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, "qwen_7b_benchmark_results.csv")
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "1",
        ]

        # launch server
        process = popen_launch_server(
            model_path_for_server,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0.41},
                {"name": "mmlu", "threshold": 0.4},
            ]
            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                # dataset_args = {}
                dataset_args = {dataset_name: self.get_dataset_args(dataset_name)}
                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")

                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")

                # generation_config = self.get_generation_config(raw_model_id)

                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                    # generation_config=generation_config,
                )

                # Run the task and get results
                print(f"{raw_model_id} Running eval for {task['name']}")
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
                            if row.get("Model") == raw_model_id:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": raw_model_id, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(
                            f"Updated CSV {csv_file_path}: {raw_model_id} - {dataset_name} = {score}"
                        )

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
        # args setting
        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN3_8B
        model_dir_name = "QWEN3_8B"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)
        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        output_dir = os.getenv("BENCH_OUTPUT_DIR", "./test/nightly_test_output/benchmark/local_run")
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, "qwen3_8b_benchmark_results.csv")
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "1",
        ]
        # launch server
        process = popen_launch_server(
            model_path_for_server,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0.86},
                {"name": "mmlu", "threshold": 0.75},
                {"name": "mmlu_pro", "threshold": 0.58},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                # dataset_args = {}
                dataset_args = {dataset_name: self.get_dataset_args(dataset_name)}
                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")

                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")

                # generation_config = self.get_generation_config(raw_model_id)

                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                    # generation_config=generation_config,
                )

                # Run the task and get results
                print(f"{raw_model_id} Running eval for {task['name']}")
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
                            if row.get("Model") == raw_model_id:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": raw_model_id, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(
                            f"Updated CSV {csv_file_path}: {raw_model_id} - {dataset_name} = {score}"
                        )

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
        # args setting
        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = DEEPSEEK_R1_DISTILL_QWEN_1_5B
        model_dir_name = "DEEPSEEK_R1_DISTILL_QWEN_1_5B"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)
        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        output_dir = os.getenv("BENCH_OUTPUT_DIR", "./test/nightly_test_output/benchmark/local_run")
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(
            output_dir, "deepseek_r1_distill_qwen_1_5b_benchmark_results.csv"
        )
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "1",
        ]

        process = popen_launch_server(
            model_path_for_server,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0.74},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
                {"name": "math_500", "threshold": 0.48},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                # dataset_args = {}
                dataset_args = {dataset_name: self.get_dataset_args(dataset_name)}
                generation_config = {}

                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")
                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")
                    pass
                if dataset_name != "gsm8k":
                    if dataset_name not in dataset_args:
                        dataset_args[dataset_name] = {}
                    dataset_args[dataset_name]["metric_list"] = ["Pass@1"]
                    generation_config = {
                        "max_tokens": 32768,
                        "temperature": 0.6,
                        "top_p": 0.95,
                    }
                # generation_config = self.get_generation_config(raw_model_id)

                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    generation_config=generation_config,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{raw_model_id} Running eval for {task['name']}")
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
                            if row.get("Model") == raw_model_id:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:
                            new_row = {"Model": raw_model_id, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(
                            f"Updated CSV {csv_file_path}: {raw_model_id} - {dataset_name} = {score}"
                        )

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
        # args setting
        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = GEMMA2_2B_IT
        model_dir_name = "GEMMA2_2B_IT"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)
        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        output_dir = os.getenv("BENCH_OUTPUT_DIR", "./test/nightly_test_output/benchmark/local_run")
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, "gemma2_2b_it_benchmark_results.csv")
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "1",
            "--disable-hybrid-swa-memory",
        ]
        # launch server
        process = popen_launch_server(
            model_path_for_server,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": -1},
                {"name": "mmlu", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                # dataset_args = {}
                dataset_args = {dataset_name: self.get_dataset_args(dataset_name)}
                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")

                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")

                # generation_config = self.get_generation_config(raw_model_id)
                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                    # generation_config=generation_config,
                )
                # Run the task and get results
                print(f"{raw_model_id} Running eval for {task['name']}")
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
                            if row.get("Model") == raw_model_id:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": raw_model_id, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(
                            f"Updated CSV {csv_file_path}: {raw_model_id} - {dataset_name} = {score}"
                        )

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
        # args setting
        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN_7B
        model_dir_name = "QWEN_7B"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)
        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        output_dir = os.getenv("BENCH_OUTPUT_DIR", "./test/nightly_test_output/benchmark/local_run")
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, "qwen_7b_benchmark_tp_4_results.csv")
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "4",
        ]
        # launch server
        process = popen_launch_server(
            model_path_for_server,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0.41},
                {"name": "mmlu", "threshold": 0.4},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                # dataset_args = {}
                dataset_args = {dataset_name: self.get_dataset_args(dataset_name)}
                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")

                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")

                # generation_config = self.get_generation_config(raw_model_id)
                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                    # generation_config=generation_config,
                )
                # Run the task and get results
                print(f"{raw_model_id} Running eval for {task['name']}")
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
                            if row.get("Model") == raw_model_id:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": raw_model_id, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(
                            f"Updated CSV {csv_file_path}: {raw_model_id} - {dataset_name} = {score}"
                        )

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
        # args setting
        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN3_8B
        model_dir_name = "QWEN3_8B"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)
        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        output_dir = os.getenv("BENCH_OUTPUT_DIR", "./test/nightly_test_output/benchmark/local_run")
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, "qwen3_8b_benchmark_tp_4_results.csv")
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "4",
        ]
        process = popen_launch_server(
            model_path_for_server,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0.86},
                {"name": "mmlu", "threshold": 0.75},
                {"name": "mmlu_pro", "threshold": 0.59},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                # dataset_args = {}
                dataset_args = {dataset_name: self.get_dataset_args(dataset_name)}
                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")

                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")

                # generation_config = self.get_generation_config(raw_model_id)

                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                    # generation_config=generation_config,
                )
                # Run the task and get results
                print(f"{raw_model_id} Running eval for {task['name']}")
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
                            if row.get("Model") == raw_model_id:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": raw_model_id, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(
                            f"Updated CSV {csv_file_path}: {raw_model_id} - {dataset_name} = {score}"
                        )

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
        # args setting
        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = GEMMA2_2B_IT
        model_dir_name = "GEMMA2_2B_IT"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)
        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        output_dir = os.getenv("BENCH_OUTPUT_DIR", "./test/nightly_test_output/benchmark/local_run")
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, "gemma2_2b_it_benchmark_tp_4_results.csv")
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "4",
            "--disable-hybrid-swa-memory",
        ]
        # launch server
        process = popen_launch_server(
            model_path_for_server,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0.57},
                {"name": "mmlu", "threshold": 0.55},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                # dataset_args = {}
                dataset_args = {dataset_name: self.get_dataset_args(dataset_name)}
                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")

                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")

                # generation_config = self.get_generation_config(raw_model_id)
                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                    # generation_config=generation_config,
                )
                # Run the task and get results
                print(f"{raw_model_id} Running eval for {task['name']}")
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
                            if row.get("Model") == raw_model_id:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": raw_model_id, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(
                            f"Updated CSV {csv_file_path}: {raw_model_id} - {dataset_name} = {score}"
                        )

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
        # args setting
        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = DEEPSEEK_R1_DISTILL_QWEN_1_5B
        model_dir_name = "DEEPSEEK_R1_DISTILL_QWEN_1_5B"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)
        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        output_dir = os.getenv("BENCH_OUTPUT_DIR", "./test/nightly_test_output/benchmark/local_run")
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(
            output_dir, "deepseek_r1_distill_qwen_1_5b_benchmark_tp_4_results.csv"
        )
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "4",
        ]
        process = popen_launch_server(
            model_path_for_server,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0.73},
                {"name": "aime24", "threshold": -1},
                {"name": "aime25", "threshold": -1},
                {"name": "math_500", "threshold": 0.47},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                # dataset_args = {}
                dataset_args = {dataset_name: self.get_dataset_args(dataset_name)}
                generation_config = {}

                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")
                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")
                    pass
                if dataset_name != "gsm8k":
                    if dataset_name not in dataset_args:
                        dataset_args[dataset_name] = {}
                    dataset_args[dataset_name]["metric_list"] = ["Pass@1"]
                    generation_config = {
                        "max_tokens": 32768,
                        "temperature": 0.6,
                        "top_p": 0.95,
                    }
                # generation_config = self.get_generation_config(raw_model_id)

                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    generation_config=generation_config,
                    eval_batch_size=64,
                )
                # Run the task and get results
                print(f"{raw_model_id} Running eval for {task['name']}")
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
                            if row.get("Model") == raw_model_id:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:
                            new_row = {"Model": raw_model_id, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(
                            f"Updated CSV {csv_file_path}: {raw_model_id} - {dataset_name} = {score}"
                        )

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
        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = BAILING_MOE
        model_dir_name = "BAILING_MOE"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)
        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        output_dir = os.getenv("BENCH_OUTPUT_DIR", "./test/nightly_test_output/benchmark/local_run")
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, "bailing_moe_benchmark_tp_2_ep_2_results.csv")
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "2",
            "--ep-size",
            "2",
        ]
        process = popen_launch_server(
            model_path_for_server,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

        # run evalscope tasks
        try:
            tasks = [
                {"name": "mmlu_pro", "threshold": 0.6},
                {"name": "aime25", "threshold": -1},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                # dataset_args = {}
                dataset_args = {dataset_name: self.get_dataset_args(dataset_name)}
                generation_config = {}

                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")
                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")
                    pass
                if dataset_name != "mmlu_pro":
                    if dataset_name not in dataset_args:
                        dataset_args[dataset_name] = {}
                    dataset_args[dataset_name]["metric_list"] = ["Pass@1"]
                    generation_config = {
                        "max_tokens": 16384,
                        "temperature": 0.6,
                        "top_p": 0.95,
                    }
                # generation_config = self.get_generation_config(raw_model_id)

                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    generation_config=generation_config,
                    eval_batch_size=64,
                )

                # Run the task and get results
                print(f"{raw_model_id} Running eval for {task['name']}")
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
                            if row.get("Model") == raw_model_id:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:
                            new_row = {"Model": raw_model_id, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(
                            f"Updated CSV {csv_file_path}: {raw_model_id} - {dataset_name} = {score}"
                        )

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

    def test_QWEN3_30B_A3B_tp_2_ep_2(self):
        # args setting
        MOUNT_ROOT = os.getenv("CI_MOUNT_ROOT", "/models")
        raw_model_id = QWEN3_MOE_30B
        model_dir_name = "QWEN3_MOE_30B"
        cached_model_path = os.path.join(MOUNT_ROOT, "model_scope", model_dir_name)
        print(f"[CI Info] Checking Model Cache at: {cached_model_path}")
        if os.path.exists(cached_model_path):
            print(f"[CI Info] Hit Model Cache: {cached_model_path}")
            model_path_for_server = cached_model_path
        else:
            print(f"[CI Info] Cache Miss, downloading: {raw_model_id}")
            model_path_for_server = raw_model_id
        base_url = DEFAULT_URL_FOR_TEST
        api_url_for_eval = f"{base_url}/v1"
        output_dir = os.getenv("BENCH_OUTPUT_DIR", "./test/nightly_test_output/benchmark/local_run")
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, "qwen3_30b_a3b_benchmark_tp_2_ep_2_results.csv")
        specific_args = self.BASIC_SERVER_ARGS + [
            "--tp-size",
            "2",
            "--ep-size",
            "2",
        ]

        persistent_cache_dir = os.getenv(
            "JAX_COMPILATION_CACHE_DIR", "/model/jax_compilation_cache/QWEN3_MOE_30B"
        )

        if not os.path.exists(persistent_cache_dir):
            try:
                os.makedirs(persistent_cache_dir, exist_ok=True)
                print(f"[CI Info] Created JAX Cache Dir: {persistent_cache_dir}")
            except Exception as e:
                print(f"[CI Error] Failed to create cache dir: {e}. Fallback to /tmp")
                persistent_cache_dir = "/tmp/jax_compilation_cache"

        print(f"[CI Info] Using JAX Cache Dir: {persistent_cache_dir}")

        process = popen_launch_server(
            model_path_for_server,
            DEFAULT_URL_FOR_TEST,
            timeout=3600,
            device="tpu",
            other_args=specific_args,
            env={
                "JAX_COMPILATION_CACHE_DIR": persistent_cache_dir,
            },
        )

        # run evalscope tasks
        try:
            tasks = [
                {"name": "gsm8k", "threshold": 0.90},
                {"name": "mmlu", "threshold": 0.77},
                {"name": "mmlu_pro", "threshold": 0.66},
            ]

            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                # dataset_args = {}
                dataset_args = {dataset_name: self.get_dataset_args(dataset_name)}
                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")

                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")

                # generation_config = self.get_generation_config(raw_model_id)
                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
                    eval_batch_size=64,
                    # generation_config=generation_config,
                )
                # Run the task and get results
                print(f"{raw_model_id} Running eval for {task['name']}")
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
                            if row.get("Model") == raw_model_id:
                                row[dataset_name] = score
                                model_found = True
                                break

                        if not model_found:

                            new_row = {"Model": raw_model_id, dataset_name: score}
                            rows.append(new_row)

                        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)

                        print(
                            f"Updated CSV {csv_file_path}: {raw_model_id} - {dataset_name} = {score}"
                        )

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
