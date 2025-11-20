import os
import sys
import unittest
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
)


class TestModelAccuracy(CustomTestCase):
    def test_qwen_7b(self):
        model = QWEN_7B
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
        self.assertGreater(metrics["score"], 0.35)

        ## kill process
        kill_process_tree(process.pid)


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
