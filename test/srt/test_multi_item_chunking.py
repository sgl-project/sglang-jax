import asyncio
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager


class _FakeTokenizerManager:
    score_request = TokenizerManager.score_request
    _build_multi_item_token_sequence = staticmethod(
        TokenizerManager._build_multi_item_token_sequence
    )

    def __init__(self, delimiter: int, chunk_size: int, max_seq_len: int):
        self.server_args = SimpleNamespace(
            multi_item_scoring_delimiter=delimiter,
            multi_item_scoring_chunk_size=chunk_size,
            max_multi_item_seq_len=max_seq_len,
        )
        self.tokenizer = None
        self.captured_requests = []

    async def generate_request(self, req, _request=None):
        self.captured_requests.append(req)
        combined = req.input_ids[0]
        delimiter = self.server_args.multi_item_scoring_delimiter
        num_delimiters = sum(1 for token_id in combined if token_id == delimiter)

        # Emit one logprob row per delimiter. score_request skips the first row
        # and uses one row per item, so this shape matches runtime expectations.
        fake_logprobs = [[(0.0, 1, None), (0.0, 2, None)] for _ in range(num_delimiters)]
        yield [{"meta_info": {"input_token_ids_logprobs": fake_logprobs, "id": "fake"}}]


def test_multi_item_chunking_allows_large_total_by_chunking():
    manager = _FakeTokenizerManager(delimiter=99, chunk_size=2, max_seq_len=10)

    query_tokens = [1]
    # 4 items, each len=3:
    # full sequence would be 1 + 12 + 5 = 18 (>10)
    # chunked with size=2 => each chunk is 1 + 6 + 3 = 10 (valid)
    items = [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]

    scores = asyncio.run(
        manager.score_request(
            query=query_tokens,
            items=items,
            label_token_ids=[1, 2],
            apply_softmax=True,
        )
    )

    assert len(scores) == 4
    assert len(manager.captured_requests) == 2
    for score in scores:
        assert score[0] == pytest.approx(0.5)
        assert score[1] == pytest.approx(0.5)


def test_multi_item_chunking_disabled_keeps_single_pass_validation():
    manager = _FakeTokenizerManager(delimiter=99, chunk_size=0, max_seq_len=10)
    query_tokens = [1]
    items = [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]

    with pytest.raises(ValueError, match="Multi-item combined sequence too long"):
        asyncio.run(
            manager.score_request(
                query=query_tokens,
                items=items,
                label_token_ids=[1, 2],
                apply_softmax=True,
            )
        )
