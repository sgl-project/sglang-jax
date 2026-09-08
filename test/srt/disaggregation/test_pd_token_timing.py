"""First-token timing must follow resolved output, including D's first EXTEND."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sgl_jax.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sgl_jax.srt.disaggregation.req_time_stats import TimeStats
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
    _mark_pd_decode_first_token,
)


def _fixture(role="decode", enabled=True):
    now = [1.0]
    req = SimpleNamespace(
        rid="timed", output_ids=[], pd_time_stats=TimeStats(role, clock=lambda: now[0])
    )
    req.pd_time_stats.mark("bootstrap_start")
    req.pd_time_stats.mark("transfer_entry")
    now[0] = 2.0
    req.pd_time_stats.mark("decode_ready")
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            disaggregation_mode=role, enable_request_time_stats_logging=enabled
        )
    )
    scheduler._pd_mark_time = lambda req, name: SchedulerDisaggregationDecodeMixin._pd_mark_time(
        scheduler, req, name
    )
    return scheduler, req, now


def test_admission_has_no_first_token_or_total():
    _, req, _ = _fixture()
    assert req.pd_time_stats.phases()["kv_wait"] == 1.0
    assert "first_token" not in req.pd_time_stats.marks
    assert "total" not in req.pd_time_stats.phases()


def test_first_output_records_and_logs_once(caplog):
    scheduler, req, now = _fixture()
    caplog.set_level(logging.INFO, logger="sgl_jax.srt.disaggregation.req_time_stats")
    _mark_pd_decode_first_token(scheduler, req)
    assert "first_token" not in req.pd_time_stats.marks
    now[0] = 3.0
    req.output_ids.append(42)
    _mark_pd_decode_first_token(scheduler, req)
    now[0] = 4.0
    req.output_ids.append(43)
    _mark_pd_decode_first_token(scheduler, req)
    assert req.pd_time_stats.marks["first_token"] == 3.0
    assert req.pd_time_stats.phases()["decode_start"] == 1.0
    assert req.pd_time_stats.phases()["total"] == 2.0
    assert len(caplog.records) == 1
    assert "decode_start=1000.0ms" in caplog.text


@pytest.mark.parametrize("role,enabled", [("null", True), ("prefill", True), ("decode", False)])
def test_other_roles_and_disabled_logging_are_unchanged(role, enabled):
    scheduler, req, _ = _fixture(role, enabled)
    scheduler._pd_mark_time = Mock()
    req.output_ids.append(42)
    _mark_pd_decode_first_token(scheduler, req)
    scheduler._pd_mark_time.assert_not_called()
    assert "first_token" not in req.pd_time_stats.marks


class _ReachedFinishCheck(Exception):
    pass


@pytest.mark.parametrize("processor", ["prefill", "decode"])
@pytest.mark.parametrize("overlap", [False, True])
def test_output_processors_mark_resolved_token_before_finish_check(processor, overlap):
    scheduler, req, now = _fixture()
    req.finished = lambda: False
    req.is_retracted = False
    req.is_chunked = 0

    def check_finished(*_):
        # Stop at the lifecycle boundary, before unrelated cache/stream work.
        assert req.output_ids == [42]
        assert req.pd_time_stats.marks["first_token"] == 3.0
        raise _ReachedFinishCheck

    req.check_finished = check_finished
    scheduler.enable_overlap = overlap
    scheduler.spec_algorithm = None
    scheduler.pd = None
    scheduler.is_generation = True
    scheduler.num_generated_tokens = 0
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(free_group_begin=lambda: None)
    logits = SimpleNamespace(next_token_logprobs=None)
    scheduler.tp_worker = SimpleNamespace(resolve_last_batch_result=lambda _: (logits, [42], 0))
    batch = SimpleNamespace(
        dp_size=1,
        per_dp_bs_size=1,
        reqs_info=[SimpleNamespace(reqs=[req])],
        return_logprob=False,
        return_output_logprob_only=False,
        batch_size=lambda: 1,
    )
    result = SimpleNamespace(
        bid=1,
        logits_output=logits,
        next_token_ids=[-1] if overlap else [42],
        cache_miss_count=0,
        extend_input_len_per_req=[1],
        extend_logprob_start_len_per_req=[0],
    )
    now[0] = 3.0
    method = getattr(SchedulerOutputProcessorMixin, f"process_batch_result_{processor}")
    with pytest.raises(_ReachedFinishCheck):
        method(scheduler, batch, result)
