import json
import unittest
import requests
import os

from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    LTX2_MODEL,
    CustomTestCase,
    popen_launch_server,
)

headers = {"Content-Type": "application/json"}

class TestLTX2EndToEnd(CustomTestCase):
    def test_ltx2_video_generation_e2e(self):
        """
        A proper integration test for the LTX-2 model.
        This test launches the full SGLang server and verifies that a small
        video generation request can be processed end-to-end.
        """
        process = popen_launch_server(
            LTX2_MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed", "42",
                "--multimodal",
                "--tokenizer-path", "google/gemma-3-12b-it",
                "--download-dir", "/mnt/disks/persist/hf_cache",
                "--tp-size", "8",
                "--enable-single-process",
                "--max-total-tokens", "32768",
                "--context-length", "8192",
            ],
            env={
                "HF_HOME": "/mnt/disks/persist/hf_cache",
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            },
            multimodal=True,
        )
        
        # A small, fast request to verify the pipeline runs without crashing.
        data = {
            "prompt": "A horse galloping on a beach",
            "num_frames": 9,  # Minimum valid frame count: 8*1 + 1
            "size": "256*256",
            "num_inference_steps": 4,  # Keep steps low for a fast test
            "guidance_scale": 3.0,
        }
        
        try:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/api/v1/videos/generation",
                headers=headers,
                json=data,
                timeout=1200, 
            )
            response.raise_for_status()
            result = response.json()
            # The most important check is that the API call completes successfully.
            self.assertTrue(result.get("success", False))
            print("SUCCESS: LTX-2 end-to-end integration test passed.")
        finally:
            # Ensure the server process is always terminated
            process.kill()

if __name__ == "__main__":
    unittest.main()
