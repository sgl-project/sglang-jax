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
        # "--log-requests",
        # "--log-requests-level",
        # "3",
    ]

    def test_qwen_7b_daily(self):
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
                {"name": "gsm8k", "threshold": 0.39},
            ]
            for task in tasks:
                dataset_name = task["name"]
                threshold = task["threshold"]

                dataset_args = {
                    dataset_name: {
                        "prompt_template": "Question: {query}\nLet's think step by step\nAnswer:",
                        "train_split": None,
                    }
                }
                cached_dataset_path = os.path.join(MOUNT_ROOT, "dataset", dataset_name)
                if os.path.exists(cached_dataset_path):
                    print(f"[CI Info] Hit Dataset Cache: {cached_dataset_path}")

                    dataset_args[dataset_name]["local_path"] = cached_dataset_path
                else:
                    print(f"[CI Info] Dataset Cache Miss: {dataset_name}")

                config = TaskConfig(
                    model=raw_model_id,
                    api_url=api_url_for_eval,
                    api_key="EMPTY",
                    eval_type="openai_api",
                    datasets=[dataset_name],
                    dataset_args=dataset_args,
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


if __name__ == "__main__":
    unittest.main()
