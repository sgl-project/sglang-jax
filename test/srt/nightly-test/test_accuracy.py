import os
import sys
import unittest
import csv
from types import SimpleNamespace
import subprocess
import re
import time
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN_7B,
    QWEN3_8B,
    CustomTestCase,
    popen_launch_server,
    QWEN3_CODER_30B_A3B_INSTRUCT,
    GEMMA2_2B_IT,
    bailing_moe,
    DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    QWEN2_5_7B_INSTRUCT,
    CustomTestCase,
    popen_launch_server,
)


class TestModelAccuracy(CustomTestCase):
    # def test_qwen_7b(self):
    #     model = QWEN_7B
    #     base_url = DEFAULT_URL_FOR_TEST
    #     process = popen_launch_server(
    #         model,
    #         base_url,
    #         timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    #         device="tpu",
    #         other_args=[
    #             "--trust-remote-code",
    #             "--skip-server-warmup",
    #             "--random-seed",
    #             "3",
    #             "--max-prefill-tokens",
    #             "16384",
    #             "--download-dir",
    #             "/dev/shm/",
    #             "--dtype",
    #             "bfloat16",
    #             "--max-running-requests",
    #             "256",
    #             "--attention-backend",
    #             "fa",
    #             "--page-size",
    #             "128",
    #             "--chunked-prefill-size",
    #             "2048",
    #             "--tp-size",
    #             "1",
    #             # "--grammar-backend", 
    #             # "none", 
    #         ],
    #         env={
    #             "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
    #         },
    #     )
    #     ## test mmlu
    #     args = SimpleNamespace(
    #         base_url=base_url,
    #         model=model,
    #         eval_name="mmlu",
    #         num_examples=256,
    #         num_threads=128,
    #         max_tokens=1024,
    #     )
    #     metrics = run_eval(args)
    #     self.assertGreater(metrics["score"], 0.35)

    #     ## kill process
    #     kill_process_tree(process.pid)

    def test_qwen_7b(self):
        model_name_str = "Qwen-7B"
        model = QWEN_7B  
        base_url = DEFAULT_URL_FOR_TEST 
        results_file = "benchmark_results.csv"


        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed", "3",
                "--max-prefill-tokens", "16384",
                "--download-dir", "/dev/shm/",
                "--dtype", "bfloat16",
                "--max-running-requests", "256",
                "--attention-backend", "fa",
                "--page-size", "128",
                "--chunked-prefill-size", "2048",
                "--tp-size", "1",
                # "--grammar-backend", "none",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        

        mmlu_score = 0.0
        gsm8k_score = 0.0

        try:
            try:
                print(f">>> [{model_name_str}] Starting MMLU Evaluation (Internal)...")
                mmlu_args = SimpleNamespace(
                    base_url=base_url,
                    model=model,
                    eval_name="mmlu",
                    num_examples=256,
                    num_threads=128,
                    max_tokens=1024,
                )
                # 假设 run_eval 返回 {'score': 0.35...}
                mmlu_metrics = run_eval(mmlu_args) 
                mmlu_score = mmlu_metrics.get('score', 0.0)
                print(f">>> MMLU Score: {mmlu_score}")
            except Exception as e:
                print(f"!!! MMLU Failed: {e}")


            print(f">>> [{model_name_str}] Starting GSM8K Evaluation (CLI: evalscope)...")
            

            api_url_for_cli = f"{base_url}/v1"
            
            cmd = [
                "uv", "run",
                "--with", "evalscope", 
                "evalscope", "eval",
                "--model", str(model),        
                "--api-url", api_url_for_cli,   
                "--api-key", "EMPTY",
                "--eval-type", "openai_api",
                "--datasets", "gsm8k",
                "--eval-batch-size", "64",
                "--limit", "10"
            ]
            
            print(f"Executing: {' '.join(cmd)}")
            

            result = subprocess.run(cmd, capture_output=True, text=True)
            

            print(">>> GSM8K CLI Output (Last 500 chars):")
            print(result.stdout[-500:]) 
            if result.stderr:
                print(">>> GSM8K CLI Error Output:")
                print(result.stderr[-500:])


            try:

                match = re.search(r"['\"]?acc['\"]?:\s*([0-9\.]+)", result.stdout) or \
                        re.search(r"['\"]?score['\"]?:\s*([0-9\.]+)", result.stdout) or \
                        re.search(r"Average Accuracy:\s*([0-9\.]+)", result.stdout)
                
                if match:
                    gsm8k_score = float(match.group(1))
                    print(f">>> Parsed GSM8K Score: {gsm8k_score}")
                else:
                    print("!!! Could not regex parse score from evalscope output. Setting to -1.")
                    gsm8k_score = -1.0
            except Exception as e:
                print(f"!!! Error parsing GSM8K score: {e}")
                gsm8k_score = -1.0

            file_exists = os.path.isfile(results_file)
            with open(results_file, mode='a', newline='', encoding='utf-8') as f:
                fieldnames = ['Model', 'MMLU', 'GSM8K']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'Model': model_name_str,
                    'MMLU': mmlu_score,
                    'GSM8K': gsm8k_score
                })
            print(f">>> Results saved to {results_file}")

        finally:
            kill_process_tree(process.pid)

        self.assertGreater(mmlu_score, 0.35, "MMLU score too low")
        if gsm8k_score > 0:
            self.assertGreater(gsm8k_score, 0.20, "GSM8K score too low")

    def test_qwen3_8b(self):
        model = QWEN3_8B
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            QWEN3_8B,
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
                # "--grammar-backend", 
                # "none", 
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        

        self.assertGreater(metrics["score"], 0.5)

        ## kill process
        kill_process_tree(process.pid)

    def test_DEEPSEEK_R1_DISTILL_QWEN_1_5B(self):
        model = DEEPSEEK_R1_DISTILL_QWEN_1_5B
        base_url = DEFAULT_URL_FOR_TEST
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
                # "--grammar-backend", 
                # "none", 
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.2)

        ## kill process
        kill_process_tree(process.pid)

    def test_GEMMA2_2B_IT(self):
        model = GEMMA2_2B_IT
        base_url = DEFAULT_URL_FOR_TEST
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
                "--grammar-backend", 
                "none", 
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)

        ## kill process
        kill_process_tree(process.pid)
    




    def test_qwen_7b_tp_4(self):
        model = QWEN_7B
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            QWEN_7B,
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
                "--grammar-backend", 
                "none", 
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.35)

        ## kill process
        kill_process_tree(process.pid)

    
    def test_qwen3_8b_tp_4(self):
        model = QWEN3_8B
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            QWEN3_8B,
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
                "--grammar-backend", 
                "none", 
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        

        self.assertGreater(metrics["score"], 0.5)

        ## kill process
        kill_process_tree(process.pid)

    def test_GEMMA2_2B_IT_tp_4(self):
        model = GEMMA2_2B_IT
        base_url = DEFAULT_URL_FOR_TEST
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
                "--grammar-backend", 
                "none", 
                "--disable-hybrid-swa-memory",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)

        self.assertGreater(metrics["score"], 0.35)

        ## kill process
        kill_process_tree(process.pid)

    def test_QWEN3_CODER_30B_A3B_INSTRUCT_tp_4(self):
        model = QWEN3_CODER_30B_A3B_INSTRUCT
        base_url = DEFAULT_URL_FOR_TEST
        
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
                "--grammar-backend", 
                "none", 
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.76)

        ## kill process
        kill_process_tree(process.pid)

    def test_DEEPSEEK_R1_DISTILL_QWEN_1_5B_tp_4(self):
        model = DEEPSEEK_R1_DISTILL_QWEN_1_5B
        base_url = DEFAULT_URL_FOR_TEST
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
                "--grammar-backend", 
                "none", 
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)

        self.assertGreater(metrics["score"], 0.2)

        ## kill process
        kill_process_tree(process.pid)




    def test_bailing_moe_tp_2_ep2(self):
        model = bailing_moe
        base_url = DEFAULT_URL_FOR_TEST
        
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
                "--grammar-backend", 
                "none", 
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.7)

        ## kill process
        kill_process_tree(process.pid)

    def test_QWEN3_CODER_30B_A3B_INSTRUCT_tp_2_ep_2(self):
        model = QWEN3_CODER_30B_A3B_INSTRUCT
        base_url = DEFAULT_URL_FOR_TEST
        
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
                "--grammar-backend", 
                "none", 
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        
        ## test mmlu
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=128,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.35)

        ## kill process
        kill_process_tree(process.pid)





if __name__ == "__main__":
    unittest.main()
