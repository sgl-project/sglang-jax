"""Serving-level verification for ``--enable-unified-radix-tree``.

HiCache Stage 1, S1c (#1337); recurrent page=1 backfill, S5a Stage 2 (#1380).

Scope is intentionally flag-on only: this per-PR test proves the enabled
UnifiedRadixCache serving path is active and correct. The Radix / no-cache /
Unified comparison evidence (relative cache-hit and throughput numbers) is
produced separately by a standalone A/B script, not by this file.

Qwen3-1.7B is non-hybrid, so ``--enable-unified-radix-tree`` routes the cache
to UnifiedRadixCache via the registry. The ``/get_server_info`` assertion in
``setUpClass`` confirms the flag actually took effect before any cache
behavior is checked.

``TestUnifiedRadixCacheServingRecurrent`` covers the recurrent (GDN-hybrid)
path at ``--page-size 1``: a Qwen3.5-35B-A3B (qwen3_5_moe) model routes to
UnifiedRadixCache with a recurrent component, and the cache-hit run must stay
byte-identical to a ``--disable-radix-cache`` baseline (PR#1's KL==0 guarantee
at page_size=1). It needs 4 chips and runs in the ``e2e-test-tpu-v6e-4`` suite,
not v6e-1.
"""

import os
import unittest

import requests

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.kits.cache_hit_kit import flush_cache, run_multiturn_cache_hit_test
from sgl_jax.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_PAGE_SIZE = 64

# Recurrent (GDN-hybrid) checkpoint for the page=1 serving test. Overridable so
# CI and the dev VM can each resolve a local path. Pinned to the qwen3_5_moe
# (GDN-hybrid) checkpoint this branch registers; it fits 4 chips at tp=4.
RECURRENT_MODEL_PATH = os.environ.get("SGLANG_RECURRENT_TEST_MODEL", "/models/Qwen3.5-35B-A3B")
_RECURRENT_PAGE_SIZE = 1
_RECURRENT_TP_SIZE = 4

# Do NOT add "--stream-output": the cache-hit kit assumes each chunk carries the
# full accumulated output_ids (only true with the default stream_output=False).
_SERVER_ARGS = [
    "--skip-server-warmup",
    "--dtype",
    "bfloat16",
    "--mem-fraction-static",
    "0.8",
    "--max-running-requests",
    "16",
    "--page-size",
    "64",
    "--random-seed",
    "42",
    "--enable-unified-radix-tree",
]


class TestUnifiedRadixCacheServing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_SERVER_ARGS,
            # Keep check_cache_miss=True: a compile-cache miss on the hit path is
            # a regression signal. Drop to False if it trips on the first TPU run.
        )

        # Fail fast for the whole class if the flag didn't take effect.
        # /get_server_info spreads dataclasses.asdict(server_args) at top level.
        # A raise here aborts setUpClass and skips tearDownClass, so kill the
        # server before re-raising to avoid leaking it on the TPU.
        try:
            resp = requests.get(f"{cls.base_url}/get_server_info", timeout=30)
            resp.raise_for_status()
            info = resp.json()
            assert info["enable_unified_radix_tree"] is True, (
                f"server did not enable unified radix tree: "
                f"enable_unified_radix_tree={info.get('enable_unified_radix_tree')!r}"
            )
            assert (
                info["page_size"] == _PAGE_SIZE
            ), f"server page_size={info.get('page_size')!r}, expected {_PAGE_SIZE}"
        except Exception:
            kill_process_tree(cls.process.pid)
            raise

    @classmethod
    def tearDownClass(cls):
        proc = getattr(cls, "process", None)
        if proc is not None:
            kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=30)
            except Exception:
                pass

    def _generate(self, prompt):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 80},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def test_multiturn_cache_hit(self):
        flush_cache(self.base_url)  # shared server: start from an empty tree
        result = run_multiturn_cache_hit_test(
            base_url=self.base_url,
            model_path=self.model,
            num_clients=4,
            num_rounds=3,
            request_length=256,
            output_length=96,
            miss_tolerance=1,
            max_parallel=4,
        )
        # The kit asserts the exact per-request bound internally; assert the
        # aggregate here so the coverage is visible in this file.
        rounds = result["rounds"]
        self.assertEqual(rounds["round_0"]["total_cached_tokens"], 0)  # cold start
        # Round 1 re-sends each client's grown prefix -> >= one page cached each.
        self.assertGreaterEqual(
            rounds["round_1"]["total_cached_tokens"],
            rounds["round_1"]["request_count"] * _PAGE_SIZE,
        )
        self.assertGreater(result["overall"]["cache_hit_rate"], 0.0)

    def test_hit_flush_determinism(self):
        # paragraph * 6 tokenizes to well over one page.
        paragraph = (
            "In the field of artificial intelligence, large language models have "
            "shown remarkable capabilities in natural language processing. These "
            "models can perform text generation, translation, summarization, and "
            "question answering. "
        )
        prompt = paragraph * 6

        # 1. Cold: empty tree -> nothing cached.
        flush_cache(self.base_url)
        cold = self._generate(prompt)
        self.assertGreaterEqual(
            cold["meta_info"]["prompt_tokens"],
            2 * _PAGE_SIZE,
            "fixture prompt must exceed one page to exercise page-aligned caching",
        )
        self.assertEqual(cold["meta_info"]["cached_tokens"], 0)

        # 2. Hit: re-send -> prefix cached, output unchanged.
        hit = self._generate(prompt)
        prompt_tokens = hit["meta_info"]["prompt_tokens"]
        self.assertLessEqual(_PAGE_SIZE, hit["meta_info"]["cached_tokens"])
        self.assertLessEqual(hit["meta_info"]["cached_tokens"], prompt_tokens)
        # Greedy determinism. The hit path has a different prefill shape than cold;
        # if this flakes on TPU bit-exactness, keep only the step-3 comparison.
        self.assertEqual(hit["text"], cold["text"])

        # 3. Post-flush: tree emptied -> nothing cached; shape-identical to cold.
        flush_cache(self.base_url)
        post_flush = self._generate(prompt)
        self.assertEqual(post_flush["meta_info"]["cached_tokens"], 0)
        self.assertEqual(post_flush["text"], cold["text"])


class TestUnifiedRadixCacheServingRecurrent(CustomTestCase):
    """Recurrent (GDN-hybrid) page=1 serving determinism.

    A Qwen3.5-35B-A3B (qwen3_5_moe) model under ``--enable-unified-radix-tree --page-size 1``
    routes to UnifiedRadixCache with a recurrent component. The cache-hit run
    must be byte-identical to a ``--disable-radix-cache`` baseline, proving
    PR#1's KL==0 guarantee holds for recurrent state at page_size=1.

    Sampler caveat: use greedy + ``--random-seed`` only; ``--page-size 1``
    plus ``--precompile-bs-paddings`` pinned to multiples of tp_size (=4)
    keeps the decode batch divisible by the device count. Do NOT add
    ``--enable-deterministic-sampling`` (it crashes on the sampler ``lax.cond``
    divisibility bug).
    """

    # Shared launch args except the radix-cache toggle (cache vs baseline).
    _COMMON_ARGS = [
        "--trust-remote-code",
        "--skip-server-warmup",
        "--dtype",
        "bfloat16",
        "--tp-size",
        str(_RECURRENT_TP_SIZE),
        "--nnodes",
        "1",
        "--dist-init-addr",
        "0.0.0.0:10011",
        "--mem-fraction-static",
        "0.8",
        "--chunked-prefill-size",
        "512",
        "--page-size",
        str(_RECURRENT_PAGE_SIZE),
        "--max-running-requests",
        "16",
        "--random-seed",
        "42",
        "--disable-overlap-schedule",
        # bs paddings pinned to multiples of tp_size (=4) so the decode batch
        # stays divisible by the device count (sampler lax.cond requirement).
        "--precompile-bs-paddings",
        "4",
        "8",
        "16",
        "--precompile-token-paddings",
        "512",
        "1024",
    ]
    _ENV = {"JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache"}

    @classmethod
    def setUpClass(cls):
        cls.model = RECURRENT_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._COMMON_ARGS + ["--enable-unified-radix-tree"],
            env=cls._ENV,
        )

        # Fail fast for the whole class if the flag or page size didn't take.
        try:
            resp = requests.get(f"{cls.base_url}/get_server_info", timeout=30)
            resp.raise_for_status()
            info = resp.json()
            assert info["enable_unified_radix_tree"] is True, (
                f"server did not enable unified radix tree: "
                f"enable_unified_radix_tree={info.get('enable_unified_radix_tree')!r}"
            )
            assert info["page_size"] == _RECURRENT_PAGE_SIZE, (
                f"server page_size={info.get('page_size')!r}, " f"expected {_RECURRENT_PAGE_SIZE}"
            )
        except Exception:
            kill_process_tree(cls.process.pid)
            raise

    @classmethod
    def tearDownClass(cls):
        proc = getattr(cls, "process", None)
        if proc is not None:
            kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=30)
            except Exception:
                pass

    def _generate(self, prompt):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 80},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def _baseline_text(cls, prompt):
        """Generate the same prompt on a fresh ``--disable-radix-cache`` server.

        The recurrent hit run must reproduce this byte-for-byte. A separate
        process avoids any cache state from the unified-tree server leaking in.
        """
        proc = popen_launch_server(
            cls.model,
            base_url=cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._COMMON_ARGS + ["--disable-radix-cache"],
            env=cls._ENV,
        )
        try:
            resp = requests.post(
                f"{cls.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 80},
                },
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            assert data["meta_info"]["cached_tokens"] == 0, (
                "baseline must run with radix cache disabled: "
                f"cached_tokens={data['meta_info']['cached_tokens']}"
            )
            return data["text"]
        finally:
            kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=30)
            except Exception:
                pass

    def test_recurrent_hit_flush_determinism(self):
        paragraph = (
            "In the field of artificial intelligence, large language models have "
            "shown remarkable capabilities in natural language processing. These "
            "models can perform text generation, translation, summarization, and "
            "question answering. "
        )
        prompt = paragraph * 6

        # 1. Cold: empty tree -> nothing cached.
        flush_cache(self.base_url)
        cold = self._generate(prompt)
        self.assertEqual(cold["meta_info"]["cached_tokens"], 0)

        # 2. Hit: re-send -> recurrent prefix cached, output unchanged.
        hit = self._generate(prompt)
        prompt_tokens = hit["meta_info"]["prompt_tokens"]
        self.assertGreater(hit["meta_info"]["cached_tokens"], 0)
        self.assertLessEqual(hit["meta_info"]["cached_tokens"], prompt_tokens)
        self.assertEqual(hit["text"], cold["text"])

        # 3. Post-flush: tree emptied -> nothing cached; shape-identical to cold.
        flush_cache(self.base_url)
        post_flush = self._generate(prompt)
        self.assertEqual(post_flush["meta_info"]["cached_tokens"], 0)
        self.assertEqual(post_flush["text"], cold["text"])

        # 4. KL==0: the cache-hit output is byte-identical to a no-cache baseline.
        # Restart on the same URL with radix cache disabled, so kill our server
        # first to free the port.
        kill_process_tree(type(self).process.pid)
        try:
            type(self).process.wait(timeout=30)
        except Exception:
            pass
        type(self).process = None
        baseline_text = self._baseline_text(prompt)
        self.assertEqual(hit["text"], baseline_text)


if __name__ == "__main__":
    unittest.main()
