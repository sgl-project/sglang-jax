import json
import unittest
import os

import requests

from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    LTX2_MODEL,
    CustomTestCase,
    popen_launch_server,
)

headers = {"Content-Type": "application/json"}

class TestLTX2Model(CustomTestCase):
    def test_ltx2_video_generation(self):
        """
        Integration test for the LTX-2 model.
        Launches the server and performs a small generation task.
        """
        process = popen_launch_server(
            LTX2_MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "42",
                "--multimodal",
                "--tokenizer-path", "google/gemma-3-12b-it",
                "--download-dir", "/mnt/disks/persist/hf_cache",
                "--tp-size", "8",
                "--enable-single-process",
                "--max-total-tokens", "8192",
                "--context-length", "1024",
            ],
            env={
                "HF_HOME": "/mnt/disks/persist/hf_cache",
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            },
            multimodal=True,
        )
        
        # Use a small number of frames and steps for a quick integration test
        data = {
            "prompt": "A running horse",
            "num_frames": 9,  # 8*1 + 1
            "height": 256,
            "width": 256,
            "num_inference_steps": 4, 
            "guidance_scale": 3.0,
        }
        
        try:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/api/v1/videos/generation",
                headers=headers,
                json=data,
                timeout=1200, # Allow ample time for JIT compilation + generation
            )
            response.raise_for_status()
            result = response.json()
            print("LTX-2 Test Success!")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        finally:
            # Ensure the server process is terminated
            process.kill()

if __name__ == "__main__":
    unittest.main()
