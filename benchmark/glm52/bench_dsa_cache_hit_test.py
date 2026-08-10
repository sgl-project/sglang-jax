import json
import unittest
from unittest import mock

from benchmark.glm52.bench_dsa_cache_hit import _run_native_batch


class _FakeResponse:
    def raise_for_status(self):
        return None

    def iter_lines(self, *, decode_unicode):
        assert decode_unicode
        event = {
            "index": 0,
            "meta_info": {
                "completion_tokens": 1,
                "cached_tokens": 128,
                "finish_reason": {"type": "length"},
            },
        }
        yield f"data: {json.dumps(event)}"
        yield "data: [DONE]"


class TestClientTimingPhases(unittest.TestCase):
    def test_native_batch_splits_submission_from_stream_ttft(self):
        with (
            mock.patch(
                "benchmark.glm52.bench_dsa_cache_hit.requests.post",
                return_value=_FakeResponse(),
            ),
            mock.patch(
                "benchmark.glm52.bench_dsa_cache_hit.time.perf_counter",
                side_effect=[10.0, 10.5, 15.0, 15.25],
            ),
            mock.patch(
                "benchmark.glm52.bench_dsa_cache_hit.time.time_ns",
                side_effect=[100, 200, 300],
            ),
        ):
            result = _run_native_batch(
                "http://localhost:30000",
                [[1, 2, 3]],
                1,
                label="timing-test",
            )

        self.assertEqual(result["ttft_s"], [5.0])
        self.assertEqual(result["request_to_headers_s"], [0.5])
        self.assertEqual(result["headers_to_first_token_s"], [4.5])
        self.assertEqual(result["request_start_unix_ns"], [100])
        self.assertEqual(result["response_headers_unix_ns"], [200])
        self.assertEqual(result["first_token_unix_ns"], [300])


if __name__ == "__main__":
    unittest.main()
