import json
import unittest

import requests

from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

headers = {"Content-Type": "application/json"}


class TestFLUX1_DEV(CustomTestCase):
    def test_flux1_dev(self):
        process = popen_launch_server(
            "/models/black-forest-labs/FLUX.1-dev/",
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--multimodal",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
            multimodal=True,
        )
        data = {
            "prompt": "A cat holding a sign that says hello world",
            "size": "480*832",
            "num_inference_steps": 5,
        }
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/api/v1/images/generation",
            headers=headers,
            json=data,
            timeout=1200,
        )
        response.raise_for_status()
        result = response.json()
        print("success！")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        process.kill()



if __name__ == "__main__":
    unittest.main()
