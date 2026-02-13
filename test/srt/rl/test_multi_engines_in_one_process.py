"""
Test multiple engine instances in one process on tpu-v6e-4.

Usage:
    python3 -m unittest test.srt.rl.multi_engines_in_one_process.TestMultiEnginesInOneProcess.test_multi_engine_modes

Modes:
    1. Single engine: devices [0, 1], single process, warmup enabled.
    2. Two engines:   engineA [0, 1], engineB [2, 3], single process, warmup enabled.
       Only engineA calls generate.
"""

import logging
import time
import unittest

import jax

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

logger = logging.getLogger(__name__)

PROMPT = "The capital of China is"

# model_path = "meta-llama/Llama-3.2-1B-Instruct"


def _make_engine(device_indexes: list[int]) -> Engine:
    return Engine(
        model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        trust_remote_code=True,
        tp_size=len(device_indexes),
        device="tpu",  # use proxy when running in Pathways
        device_indexes=device_indexes,
        enable_single_process=True,
        skip_server_warmup=True,
        random_seed=3,
        node_rank=0,
        mem_fraction_static=0.8,
        chunked_prefill_size=1024,
        download_dir="/tmp",
        dtype="bfloat16",
        precompile_bs_paddings=[8],
        max_running_requests=8,
        attention_backend="fa",
        precompile_token_paddings=[1024],
        page_size=128,
        log_requests=False,
        log_level="info",
    )


class TestMultiEnginesInOneProcess(CustomTestCase):
    def _run_generate(self, engine: Engine, max_new_tokens: int = 1) -> list[dict]:
        sampling_params = engine.get_default_sampling_params()
        sampling_params.max_new_tokens = max_new_tokens
        sampling_params.temperature = 0
        outputs = engine.generate(
            prompt=PROMPT,
            sampling_params=sampling_params.convert_to_dict(),
        )
        return outputs

    def test_01_multi_engine_modes_cache_miss(self):
        print(
            "=== test_01_multi_engine_modes_cache_miss: engineA=[0,1], engineB=[2,3] ===",
            flush=True,
        )
        engine_a = _make_engine(device_indexes=[0, 1])
        engine_b = _make_engine(device_indexes=[2, 3])
        try:
            prefill_output = self._run_generate(engine_a, max_new_tokens=1)
            decode_output = self._run_generate(engine_a, max_new_tokens=2)
            assert prefill_output["meta_info"]["cache_miss_count"] == 0
            assert decode_output["meta_info"]["cache_miss_count"] == 0
        finally:
            engine_a.shutdown()
            engine_b.shutdown()

    def test_02_multi_engine_modes(self):
        # ------------------------------------------------------------------ #
        # Mode 1: single engine, devices [0, 1]                               #
        # ------------------------------------------------------------------ #
        print("=== Mode 1: single engine, device_indexes=[0, 1] ===", flush=True)
        engine = _make_engine(device_indexes=[0, 1])
        try:
            with jax.profiler.trace(
                "/home/gcpuser/aolemila/profile"
            ):  # Use gs://aolemila/rl/profiler/multi_rollout when running in Pathways
                outputs = self._run_generate(engine, max_new_tokens=5)
                print("Mode 1 output: %s", outputs, flush=True)
        finally:
            print(f"engine shutdown in mode 1")
            engine.shutdown()

        print(f"sleep for 30 seconds")
        jax.clear_caches()
        time.sleep(30)
        [print(f"\n", flush=True) for _ in range(5)]

        # ------------------------------------------------------------------ #
        # Mode 2: two engines, engineA [0,1]  engineB [2,3]                   #
        # ------------------------------------------------------------------ #
        jax.clear_caches()
        engine_a = _make_engine(device_indexes=[0, 1])
        engine_b = _make_engine(device_indexes=[2, 3])
        try:
            [print(f"\n") for _ in range(5)]
            with jax.profiler.trace(
                "/home/gcpuser/aolemila/profile"
            ):  # Use gs://aolemila/rl/profiler/multi_rollout when running in Pathways
                outputs = self._run_generate(engine_a, max_new_tokens=5)
                print("Mode 2 engineA output: %s", outputs, flush=True)
        finally:
            engine_a.shutdown()
            engine_b.shutdown()


if __name__ == "__main__":
    unittest.main()
