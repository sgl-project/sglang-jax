"""CPU tests for the Ling-3 multi-turn Agent serving probe."""

import importlib.util
import os
import sys
import unittest
from unittest import mock

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BENCH_PATH = os.path.join(
    _REPO_ROOT, "benchmark", "hicache", "bench_ling3_multiturn_agent.py"
)


def _load_bench_module():
    name = "bench_ling3_multiturn_agent"
    spec = importlib.util.spec_from_file_location(name, _BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestLing3MultiturnAgent(unittest.TestCase):
    def setUp(self):
        self.bench = _load_bench_module()

    def test_sessions_have_unique_long_case_files(self):
        sessions = self.bench._build_sessions(32)

        self.assertEqual(len({session.codename for session in sessions}), 32)
        self.assertEqual(len({session.owner for session in sessions}), 32)
        self.assertTrue(all(len(session.messages[0]["content"]) > 5000 for session in sessions))

    def test_request_records_quality_cache_and_actual_dp_rank(self):
        session = self.bench._build_sessions(1)[0]
        content = f"CASE_ACCEPTED {session.codename}"
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning_content": "checked the case",
                    },
                    "finish_reason": "stop",
                    "dp_rank": 6,
                }
            ],
            "usage": {
                "prompt_tokens": 900,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 512},
            },
        }

        with mock.patch.object(self.bench.requests, "post", return_value=response):
            result = self.bench._request_turn(
                session,
                0,
                url="http://server",
                model="ling3-tiny",
                max_tokens=64,
            )

        self.assertTrue(result["quality_ok"])
        self.assertEqual(result["cached_tokens"], 512)
        self.assertEqual(result["dp_rank"], 6)
        self.assertEqual(session.messages[-1], {"role": "assistant", "content": content})


if __name__ == "__main__":
    unittest.main()
