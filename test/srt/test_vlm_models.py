"""Real-weight VLM smoke and scheduling regression (one TPU by default)."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
from openai import OpenAI
from PIL import Image
from vlm_utils import complete, image_content

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


class TestVLM(unittest.TestCase):
    def test_serving_equivalence(self):
        from nightly.single_host.accuracy_case_runner import (
            load_profile_file,
            profile_server_spec,
        )

        profile = load_profile_file("qwen3-vl-2b-v6e-4.yaml")
        profile.tp_size = int(os.getenv("VLM_TP_SIZE", "1"))
        full = os.getenv("VLM_FULL_REGRESSION") == "1"
        inputs = [
            ([("red", (224, 224))], "red"),
            ([("blue", (224, 224))], "blue"),
            ([("red", (224, 224)), ("blue", (224, 224))], "red"),
            ([("blue", (224, 224)), ("red", (224, 224))], "blue"),
        ]
        if full:
            # 784 vision tokens cross a 256-token chunk; reserved for weekly CI.
            inputs += [
                ([("red", (448, 224))], "red"),
                ([("blue", (896, 896))], "blue"),
            ]
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        *[image_content(Image.new("RGB", size, color)) for color, size in images],
                        {
                            "type": "text",
                            "text": "What color fills the first image? Answer with one color word only.",
                        },
                    ],
                }
            ]
            for images, _ in inputs
        ]
        # Include a text-only request in the same concurrent workload.
        messages.append([{"role": "user", "content": "Reply with exactly the word hello."}])
        expected = [answer for _, answer in inputs] + ["hello"]
        reference = None
        modes = [("256", False, "dp")]
        if full:
            modes.insert(0, ("-1", True, "dp"))
        if full and profile.tp_size > 1:
            modes.append(("256", False, "tp"))
        for chunk, no_cache, encoder in modes:
            with self.subTest(chunk=chunk, no_cache=no_cache, encoder=encoder):
                spec = profile_server_spec(profile)
                # Bound cold compilation to this fixed four-request workload.
                # Four small-image requests contain at most 1536 patches.
                extra = [
                    "--context-length",
                    "1024",
                    "--max-seq-len",
                    "1024",
                    "--max-total-tokens",
                    "4096",
                    "--chunked-prefill-size",
                    chunk,
                    "--vision-encoder-parallel",
                    encoder,
                    "--max-prefill-tokens",
                    "2048" if full else "256",
                    "--precompile-token-paddings",
                    "256",
                    *(["2048"] if full else []),
                    "--max-running-requests",
                    "4",
                    "--precompile-bs-paddings",
                    "4",
                    "--precompile-vision-patch-paddings",
                    "5120" if full else "1536",
                ]
                if no_cache:
                    extra.append("--disable-radix-cache")
                process = popen_launch_server(
                    spec["model"],
                    spec["base_url"],
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=spec["other_args"] + extra,
                )
                try:
                    with OpenAI(
                        base_url=spec["base_url"] + "/v1",
                        api_key="EMPTY",
                        timeout=120,
                        max_retries=0,
                    ) as client:

                        def ask(message):
                            return (
                                complete(client, message, temperature=0, max_tokens=16, seed=42)
                                .lower()
                                .strip(" .!\n")
                            )

                        if full:
                            serial = list(map(ask, messages))
                            self.assertEqual(serial, expected)
                            reference = serial if reference is None else reference
                            self.assertEqual(serial, reference)
                        response = requests.post(spec["base_url"] + "/flush_cache", timeout=30)
                        response.raise_for_status()
                        with ThreadPoolExecutor(max_workers=4) as pool:
                            # Cold mixed batch, then the same workload with cached images/prefixes.
                            for _ in range(2):
                                self.assertEqual(list(pool.map(ask, messages)), expected)
                finally:
                    kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
