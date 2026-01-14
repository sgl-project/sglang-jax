import json
import unittest

import requests

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    WAN2_1_T2V_1_3B,
    WAN2_1_T2V_14B,
    CustomTestCase,
    popen_launch_server,
)

headers = {"Content-Type": "application/json"}


class TestWan2_1Model(CustomTestCase):
    def test_wan2_1_1_3b(self):
        process = popen_launch_server(
            "/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/",
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--multimodal",
                "--disable-precompile",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
            multimodal=True,
        )
        data = {"prompt": "A curious raccoon", "size": "480*832", "num_frames": 41}
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/api/v1/videos/generation",
            headers=headers,
            json=data,
            timeout=1200,
        )
        response.raise_for_status()
        # 解析返回的 JSON 数据
        result = response.json()
        print("请求成功！")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        process.kill()

    def test_wan2_1_14b(self):
        process = popen_launch_server(
            "/models/Wan-AI/Wan2.1-T2V-14B-Diffusers/",
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--multimodal",
                "--disable-precompile",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
            multimodal=True,
        )
        data = {"prompt": "A curious raccoon", "size": "480*832", "num_frames": 5}
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/api/v1/videos/generation",
            headers=headers,
            json=data,
            timeout=1200,
        )
        response.raise_for_status()
        # 解析返回的 JSON 数据
        result = response.json()
        print("请求成功！")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        process.kill()


if __name__ == "__main__":
    unittest.main()
