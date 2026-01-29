import os
import unittest
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import requests

from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

headers = {"Content-Type": "application/json"}


# note this test cost about 20 minutes
class TestQwen3OmniMoeThinkerTextPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
        cls.prefill_data = "/home/gcpuser/qwen3_omni_moe_thinker_text_prefill.txt"
        cls.decode_data = "/home/gcpuser/qwen3_omni_moe_thinker_text_decode.txt"
        cls.toleration = 2e-5
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="cpu",
            check_cache_miss=False,
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
                "float32",
                "--precompile-bs-paddings",
                "16",
                "--precompile-token-paddings",
                "128",
                "--tp-size",
                "1",
                "--nnodes",
                "1",
                "--dist-init-addr",
                "0.0.0.0:10011",
                "--max-running-requests",
                "16",
                "--page-size",
                "64",
                "--disable-precompile",
                "--watchdog-timeout",
                "1200",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
                "DUMP_LAST_LAYER_LOGITS_FILENAMES": cls.prefill_data + "," + cls.decode_data,
                "SGLANG_HEALTH_CHECK_TIMEOUT": "1200",
            },
        )

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("JAX_PLATFORMS", None)
        if os.path.exists(cls.prefill_data):
            os.remove(cls.prefill_data)
        if os.path.exists(cls.decode_data):
            os.remove(cls.decode_data)

    def _get_transformers_output(self):
        # from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
        # import torch
        # thinker = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained("/models/Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype=torch.float32)
        # import json
        # input_ids = torch.tensor([[1474, 25, 220, 14880, 11622, 105321, 104136, 106582, 109539, 115822, 1773, 71703, 25, 220]])
        # output=thinker(input_ids=input_ids)
        # np.savetxt("torch_output.txt", output.logits[0,-1,:].to(torch.float32).detach().numpy().flatten(), fmt="%.15f")
        current_dir = str(Path(__file__).resolve().parent)
        return np.loadtxt(current_dir + "/data/qwen3_omni_moe_thinker_text_prefill.txt")

    def _get_jax_output(self):
        data = {
            "model": "Qwen3-Omni-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "请用一句话解释什么是量子纠缠。"}],
            "max_tokens": 1,
        }
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=1200,
        )
        response.raise_for_status()

        jax_prefill_data = np.loadtxt(self.prefill_data)[:100]
        return jax_prefill_data

    def test_prefill(self):
        jax_prefill_data = self._get_jax_output()
        torch_prefill_data = self._get_transformers_output()
        np.testing.assert_allclose(
            jax_prefill_data, torch_prefill_data, rtol=self.toleration, atol=self.toleration
        )


if __name__ == "__main__":
    unittest.main()
