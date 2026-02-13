import asyncio
import math
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager


class _FakeCreateTokenizedManager:
    _create_tokenized_object = TokenizerManager._create_tokenized_object

    def __init__(self):
        self.preferred_sampling_params = None
        self.tokenizer = None
        self.model_config = SimpleNamespace(vocab_size=32000)


class _FakePrefillManager:
    _prefill_and_cache = TokenizerManager._prefill_and_cache

    def __init__(self):
        self.seen_requests = []

    async def generate_request(self, req, request=None):
        del request
        self.seen_requests.append(req)
        yield {"meta_info": {"id": req.rid}}


class _FakeExtendManager:
    _batched_extend_score = TokenizerManager._batched_extend_score

    def __init__(self):
        self.seen_request = None

    async def generate_request(self, req, request=None):
        del request
        self.seen_request = req
        yield [
            {
                "index": 0,
                "meta_info": {
                    "output_token_ids_logprobs": [
                        [
                            (math.log(0.9), 10, None),
                            (math.log(0.1), 20, None),
                        ]
                    ]
                }
            },
            {
                "index": 1,
                "meta_info": {
                    "output_token_ids_logprobs": [
                        [
                            (math.log(0.2), 10, None),
                            (math.log(0.8), 20, None),
                        ]
                    ]
                }
            },
        ]


class _FakeExtendMissingLogprobsManager:
    _batched_extend_score = TokenizerManager._batched_extend_score

    async def generate_request(self, req, request=None):
        del req, request
        yield [{"index": 0, "meta_info": {"output_token_ids_logprobs": []}}]


def test_create_tokenized_object_keeps_prefill_extend_fields():
    manager = _FakeCreateTokenizedManager()
    req = GenerateReqInput(
        rid="req-1",
        input_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 0},
        return_logprob=True,
        token_ids_logprob=[10, 20],
        cache_for_scoring=True,
        extend_from_cache="cache-handle-1",
        is_single=True,
    )
    req.normalize_batch_and_arguments()

    tokenized = manager._create_tokenized_object(req, input_text=None, input_ids=req.input_ids)

    assert tokenized.cache_for_scoring is True
    assert tokenized.extend_from_cache == "cache-handle-1"


def test_prefill_and_cache_uses_single_request_and_stable_handle():
    manager = _FakePrefillManager()
    handle = asyncio.run(manager._prefill_and_cache([11, 12, 13]))

    assert len(manager.seen_requests) == 1
    req = manager.seen_requests[0]
    assert req.input_ids == [11, 12, 13]
    assert req.is_single is True
    assert req.cache_for_scoring is True
    assert isinstance(req.rid, str)
    assert handle == req.rid


def test_batched_extend_score_passes_cache_handle_and_scores_items():
    manager = _FakeExtendManager()
    scores = asyncio.run(
        manager._batched_extend_score(
            cache_handle="cache-handle-xyz",
            items=[[1], [2]],
            label_token_ids=[10, 20],
            apply_softmax=False,
        )
    )

    assert manager.seen_request is not None
    assert manager.seen_request.extend_from_cache == "cache-handle-xyz"
    assert manager.seen_request.input_ids == [[1], [2]]
    assert manager.seen_request.return_output_logprob_only is True
    assert manager.seen_request.logprob_start_len is None
    assert len(scores) == 2
    assert scores[0] == pytest.approx([0.9, 0.1])
    assert scores[1] == pytest.approx([0.2, 0.8])


def test_batched_extend_score_raises_when_output_logprobs_missing():
    manager = _FakeExtendMissingLogprobsManager()
    with pytest.raises(RuntimeError, match="output_token_ids_logprobs is empty"):
        asyncio.run(
            manager._batched_extend_score(
                cache_handle="cache-handle-xyz",
                items=[[1]],
                label_token_ids=[10, 20],
                apply_softmax=False,
            )
        )
