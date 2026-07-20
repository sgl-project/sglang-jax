"""Step-3.5-Flash NEXTN (chain MTP) end-to-end correctness tests.

Ports the two correctness checks upstream sglang runs for Step-3.5-Flash chain
MTP (`test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`) to sgl-jax:

  1. GSM8K accuracy with speculative decoding ON — because NEXTN spec-decode is
     lossless at temperature=0, the score must match the plain-Flash greedy score
     when a baseline is supplied; otherwise it must meet upstream's five-shot
     completion threshold. Optionally also gates mean accept-length >= 2.6
     (upstream's threshold) from the server log, since sgl-jax does not surface
     accept-length over the API yet.

  2. Logprob consistency — generate with chain-MTP spec, then re-score the same
     sequence via prefill-only (no speculation); the two logprob sets must match
     within tolerance. This is a DISTRIBUTION-level check that temperature=0
     token-equality (lossless) does not cover: a subtle chain hidden-state bug
     can keep the argmax token correct while shifting the logprob values.

These need the real multi-host Step-3.5-Flash server (196B), which the test
harness cannot launch, so they run against an externally-managed server:
launch it with the NEXTN spec flags, then set SGLANG_NEXTN_E2E_URL.

Requires the multi-host host-transfer fix (sgl-project/sglang-jax#1410) to be
present in the running server, otherwise the logprob path crashes on multi-host.
"""

import os
import re
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import requests
from run_eval import run_eval

_URL = os.getenv("SGLANG_NEXTN_E2E_URL")
_MODEL = os.getenv("SGLANG_NEXTN_E2E_MODEL", "step-3.5-flash")
# Optional: path to the server stdout/stderr log, to assert mean accept-length.
_SERVER_LOG = os.getenv("SGLANG_SPEC_SERVER_LOG")
# Optional: the plain-Flash (no-spec) GSM8K score to assert losslessness against.
_GSM8K_BASELINE = os.getenv("SGLANG_GSM8K_BASELINE")

# GSM8K knobs (overridable via env for quick/full runs).
_GSM8K_N = int(os.getenv("SGLANG_GSM8K_NUM_EXAMPLES", "200"))
_GSM8K_THREADS = int(os.getenv("SGLANG_GSM8K_THREADS", "32"))
_DEFAULT_GSM8K_MAX_TOKENS = 512
_GSM8K_MAX_TOKENS = int(os.getenv("SGLANG_GSM8K_MAX_TOKENS", str(_DEFAULT_GSM8K_MAX_TOKENS)))
# Upstream's five-shot completion threshold when no greedy baseline is given.
_DEFAULT_GSM8K_FLOOR = 0.83
_GSM8K_FLOOR = float(os.getenv("SGLANG_GSM8K_FLOOR", str(_DEFAULT_GSM8K_FLOOR)))
# Tolerance vs baseline (lossless → near-zero; small slack for eval subset noise).
_GSM8K_TOL = float(os.getenv("SGLANG_GSM8K_TOL", "0.01"))
# Upstream's accept-length threshold for Step-3.5-Flash chain MTP.
_ACCEPT_LEN_THRES = float(os.getenv("SGLANG_ACCEPT_LEN_THRES", "2.6"))


def _mean_accept_length_from_log(path: str, start_offset: int = 0) -> float | None:
    """Mean of `accept-len X.XX` values in the server decode log (None if absent)."""
    vals = []
    pat = re.compile(r"accept-len\s+([0-9.]+)")
    with open(path, encoding="utf-8", errors="ignore") as f:
        f.seek(start_offset)
        for line in f:
            m = pat.search(line)
            if m:
                vals.append(float(m.group(1)))
    return float(np.mean(vals)) if vals else None


def _validate_step3p5_nextn_server(info: dict) -> None:
    expected = {
        "speculative_algorithm": "NEXTN",
        "speculative_eagle_topk": 1,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise AssertionError(f"expected {key}={value!r}, got {info.get(key)!r}")

    mode = info.get("speculative_target_verify_mode")
    if mode not in {"auto", "decode-loop"}:
        raise AssertionError(f"expected decode-equivalent target verify, got mode={mode!r}")
    steps = info.get("speculative_num_steps")
    if info.get("speculative_num_draft_tokens") != steps + 1:
        raise AssertionError("draft-token count must equal speculative steps plus one")
    if info.get("nnodes", 1) <= 1:
        raise AssertionError("Step3p5 MTP E2E requires a multi-host server")

    states = info.get("internal_states") or []
    if not states:
        raise AssertionError("server did not return scheduler internal state")
    for state in states:
        if state.get("model_type") != "step3p5":
            raise AssertionError(f"expected model_type='step3p5', got {state.get('model_type')!r}")
        if "Step3p5ForCausalLM" not in state.get("model_architectures", []):
            raise AssertionError("server is not running Step3p5ForCausalLM")
        if not state.get("spec_multi_layer"):
            raise AssertionError("server did not initialize multi-layer MTP")
        if state.get("enable_overlap"):
            raise AssertionError("decode-loop target verify must run with overlap disabled")


def _build_gsm8k_eval_args() -> SimpleNamespace:
    return SimpleNamespace(
        base_url=_URL,
        model=_MODEL,
        eval_name="gsm8k",
        api="completion",
        num_shots=5,
        num_examples=_GSM8K_N,
        num_threads=_GSM8K_THREADS,
        max_tokens=_GSM8K_MAX_TOKENS,
    )


class TestStep3p5MTPHarness(unittest.TestCase):
    def test_accept_length_reader_uses_current_log_segment(self):
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as log:
            log.write("accept-len 9.0\n")
            log.flush()
            start_offset = log.tell()
            log.write("accept-len 2.0\naccept-len 4.0\n")
            log.flush()

            self.assertEqual(_mean_accept_length_from_log(log.name, start_offset), 3.0)

    def test_server_validation_requires_effective_decode_loop_configuration(self):
        info = {
            "speculative_algorithm": "NEXTN",
            "speculative_eagle_topk": 1,
            "speculative_target_verify_mode": "auto",
            "speculative_num_steps": 3,
            "speculative_num_draft_tokens": 4,
            "nnodes": 4,
            "internal_states": [
                {
                    "model_type": "step3p5",
                    "model_architectures": ["Step3p5ForCausalLM"],
                    "spec_multi_layer": True,
                    "enable_overlap": False,
                }
            ],
        }

        _validate_step3p5_nextn_server(info)
        info["internal_states"][0]["enable_overlap"] = True
        with self.assertRaisesRegex(AssertionError, "overlap disabled"):
            _validate_step3p5_nextn_server(info)


@unittest.skipUnless(
    _URL,
    "Set SGLANG_NEXTN_E2E_URL to a running multi-host Step-3.5-Flash NEXTN server "
    "(launched with --speculative-algorithm NEXTN ...).",
)
class TestStep3p5MTPGsm8k(unittest.TestCase):
    """NEXTN spec-decode must preserve accuracy (lossless) and accept drafts."""

    @classmethod
    def setUpClass(cls):
        requests.get(f"{_URL}/health", timeout=30).raise_for_status()
        info = requests.get(f"{_URL}/get_server_info", timeout=30)
        info.raise_for_status()
        _validate_step3p5_nextn_server(info.json())

    def test_gsm8k_accuracy_lossless(self):
        metrics = run_eval(_build_gsm8k_eval_args())
        score = metrics["score"]
        print(f"[gsm8k] NEXTN spec-decode score={score:.4f} (n={_GSM8K_N})")

        if _GSM8K_BASELINE is not None:
            # Strict losslessness: spec score must match the plain-Flash score.
            base = float(_GSM8K_BASELINE)
            self.assertLessEqual(
                abs(score - base),
                _GSM8K_TOL,
                f"spec gsm8k {score:.4f} deviates from Flash baseline {base:.4f} "
                f"by >{_GSM8K_TOL} — NEXTN is supposed to be lossless at temp=0",
            )
        else:
            self.assertGreaterEqual(
                score,
                _GSM8K_FLOOR,
                f"spec gsm8k {score:.4f} below upstream five-shot floor {_GSM8K_FLOOR} "
                f"(set SGLANG_GSM8K_BASELINE=<flash score> for the strict lossless check)",
            )

    def test_accept_length(self):
        """Mean accept-length >= threshold — proves the MTP draft is genuinely
        accepted (chain working), not just lossless. sgl-jax does not expose
        accept-length over the API, so this drives a few decode-heavy requests
        itself, then reads the server log (self-contained: does not depend on
        another test having generated first / on pytest ordering)."""
        if not _SERVER_LOG:
            self.skipTest(
                "Set SGLANG_SPEC_SERVER_LOG=<server log path> to assert accept-length; "
                "otherwise grep 'accept-len' in the server log manually (want >= "
                f"{_ACCEPT_LEN_THRES})."
            )
        start_offset = os.path.getsize(_SERVER_LOG)
        # Warm up: generate enough decode steps so the scheduler logs 'accept-len'
        # (spec accept-length is only emitted during decode, not prefill).
        for _ in range(4):
            requests.post(
                f"{_URL}/generate",
                json={
                    "text": "Count slowly: one, two, three, four, five, six, seven,",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 64},
                },
                timeout=600,
            ).raise_for_status()

        mean_al = _mean_accept_length_from_log(_SERVER_LOG, start_offset)
        self.assertIsNotNone(
            mean_al,
            "no 'accept-len' lines in server log even after warm-up decode — is this "
            "a spec-decode server, and is SGLANG_SPEC_SERVER_LOG the right log path?",
        )
        print(f"[accept-len] mean={mean_al:.3f} (threshold {_ACCEPT_LEN_THRES})")
        self.assertGreaterEqual(mean_al, _ACCEPT_LEN_THRES)


@unittest.skipUnless(
    _URL,
    "Set SGLANG_NEXTN_E2E_URL to a running multi-host Step-3.5-Flash NEXTN server.",
)
class TestStep3p5MTPLogprobConsistency(unittest.TestCase):
    """Chain-MTP spec-decode logprobs must match prefill-only re-scoring.

    Distribution-level check beyond temp=0 token-equality: a chain hidden-state
    bug can keep the argmax token right while drifting the logprob values.
    Ported from upstream ``TestStep3p5FlashChainMTP.test_logprob_spec_v2_match``.
    """

    # Tolerances from the upstream reference test (TP=8 bf16 + multi-layer EAGLE).
    # The bulk of the distribution must stay tight; the tail (max/p99) is
    # dominated by very-low-prob tokens whose logprobs are extremely sensitive to
    # bf16 + TP logsumexp noise — a real chain bug shifts the MEDIAN, not the tail.
    _CHOSEN_MAX_DIFF = float(os.getenv("SGLANG_LOGPROB_CHOSEN_MAXDIFF", "0.255"))

    @classmethod
    def setUpClass(cls):
        requests.get(f"{_URL}/health", timeout=30).raise_for_status()
        info = requests.get(f"{_URL}/get_server_info", timeout=30)
        info.raise_for_status()
        _validate_step3p5_nextn_server(info.json())

    def test_logprob_spec_matches_prefill(self):
        try:
            requests.get(f"{_URL}/flush_cache", timeout=30)
        except requests.RequestException:
            pass

        top_k = 5
        probe_token_ids = [1, 2, 10, 100, 1000]
        prompts = [
            "The capital of France is",
            "Explain quantum computing in simple terms:",
        ]

        for round_idx, prompt in enumerate(prompts):
            with self.subTest(round=round_idx, prompt=prompt):
                gen_res = requests.post(
                    f"{_URL}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                        "top_logprobs_num": top_k,
                        "token_ids_logprob": probe_token_ids,
                        "logprob_start_len": 0,
                    },
                    timeout=600,
                ).json()

                decode_logprobs = gen_res["meta_info"]["output_token_logprobs"]
                decode_top_logprobs = gen_res["meta_info"]["output_top_logprobs"]
                decode_tid_logprobs = gen_res["meta_info"]["output_token_ids_logprobs"]
                input_token_ids = [t[1] for t in gen_res["meta_info"]["input_token_logprobs"]]
                output_token_ids = [t[1] for t in decode_logprobs]
                num_prompt_tokens = gen_res["meta_info"]["prompt_tokens"]

                score_res = requests.post(
                    f"{_URL}/generate",
                    json={
                        "input_ids": input_token_ids + output_token_ids,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                        "return_logprob": True,
                        "top_logprobs_num": top_k,
                        "token_ids_logprob": probe_token_ids,
                        "logprob_start_len": 0,
                    },
                    timeout=600,
                ).json()

                score_logprobs = score_res["meta_info"]["input_token_logprobs"][num_prompt_tokens:]
                score_top_logprobs = score_res["meta_info"]["input_top_logprobs"][
                    num_prompt_tokens:
                ]
                score_tid_logprobs = score_res["meta_info"]["input_token_ids_logprobs"][
                    num_prompt_tokens:
                ]

                self.assertEqual(len(decode_logprobs), len(score_logprobs))

                # Chosen-token logprob: the tight, decisive check.
                decode_vals = np.array([t[0] for t in decode_logprobs])
                score_vals = np.array([t[0] for t in score_logprobs])
                max_diff = float(np.max(np.abs(decode_vals - score_vals)))
                print(f"[round {round_idx}] chosen-token logprob max_diff={max_diff:.6f}")
                self.assertLess(max_diff, self._CHOSEN_MAX_DIFF)

                # Top-k logprobs: compare on the token ids both sides expose.
                top_diffs = []
                for pos in range(len(decode_logprobs)):
                    dec_top = {t[1]: t[0] for t in decode_top_logprobs[pos]}
                    scr_top = {t[1]: t[0] for t in score_top_logprobs[pos]}
                    common_ids = set(dec_top) & set(scr_top)
                    self.assertGreater(len(common_ids), 0)
                    top_diffs.extend(abs(dec_top[t] - scr_top[t]) for t in common_ids)
                top_diffs = np.array(top_diffs)
                print(
                    f"[round {round_idx}] top-k diffs: n={len(top_diffs)} "
                    f"p50={np.percentile(top_diffs, 50):.4f} mean={top_diffs.mean():.4f} "
                    f"p95={np.percentile(top_diffs, 95):.4f} max={top_diffs.max():.4f}"
                )

                # Probe token-ids logprobs (same ids on both sides).
                self.assertEqual(len(decode_tid_logprobs), len(score_tid_logprobs))
                tid_diffs = []
                for pos in range(len(decode_tid_logprobs)):
                    dec_tid = {t[1]: t[0] for t in decode_tid_logprobs[pos]}
                    scr_tid = {t[1]: t[0] for t in score_tid_logprobs[pos]}
                    self.assertEqual(set(dec_tid), set(scr_tid))
                    tid_diffs.extend(abs(dec_tid[t] - scr_tid[t]) for t in dec_tid)
                tid_diffs = np.array(tid_diffs)
                print(
                    f"[round {round_idx}] token-ids diffs: n={len(tid_diffs)} "
                    f"p50={np.percentile(tid_diffs, 50):.4f} mean={tid_diffs.mean():.4f} "
                    f"p95={np.percentile(tid_diffs, 95):.4f} max={tid_diffs.max():.4f}"
                )

                # Bulk of the distribution must stay tight (a chain bug shifts the
                # median); tails are left loose (bf16 + TP logsumexp sensitivity).
                self.assertLess(np.percentile(top_diffs, 50), 0.1)
                self.assertLess(top_diffs.mean(), 0.2)
                self.assertLess(np.percentile(top_diffs, 95), 0.4)
                self.assertLess(np.percentile(tid_diffs, 50), 0.2)
                self.assertLess(tid_diffs.mean(), 0.4)


if __name__ == "__main__":
    unittest.main()
