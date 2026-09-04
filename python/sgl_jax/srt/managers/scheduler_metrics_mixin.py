from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from sgl_jax.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
    from sgl_jax.srt.managers.schedule_policy import PrefillAdder
    from sgl_jax.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")

# Keep these queue-latency buckets aligned with upstream SGLang's
# sglang:queue_time_seconds histogram.
QUEUE_TIME_BUCKETS = (
    0.0,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    3,
    4,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1200,
    1400,
    1600,
    1800,
    2000,
    2500,
    3000,
)


def create_queue_time_histogram():
    # prometheus_client chooses its storage backend when it is first imported.
    # Keep this import on the runtime initialization path, after the HTTP server
    # has configured PROMETHEUS_MULTIPROC_DIR.
    from prometheus_client import Histogram

    return Histogram(
        name="sglang:queue_time_seconds",
        documentation="Histogram of queueing time in seconds.",
        labelnames=("dp_rank",),
        buckets=QUEUE_TIME_BUCKETS,
    )


def record_queue_wait_times(reqs: list[Req], metric, now: float | None = None) -> None:
    """Record first-admission queue latency, labeled by DP rank.

    Queue time starts when a request first enters the scheduler (including the
    grammar queue) and ends when it is admitted to its first prefill batch.
    Requests admitted again after retraction retain their original measurement.

    This is the request-scheduling analogue of Linux run-queue latency: the time
    a runnable task spends on the run queue before the scheduler gives it a CPU.
    """
    if now is None:
        now = time.perf_counter()

    for req in reqs:
        if req.queue_time_start is None or req.queue_time_end is not None:
            continue

        req.queue_time_end = now
        if req.dp_rank is not None:
            metric.labels(dp_rank=str(req.dp_rank)).observe(max(0.0, now - req.queue_time_start))


class SchedulerMetricsMixin:
    def init_metrics(self: Scheduler):
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self.spec_num_total_accepted_tokens = 0
        self.spec_num_total_forward_ct = 0
        self.cum_spec_accept_length = 0
        self.cum_spec_accept_count = 0
        self.total_retracted_reqs = 0
        self.queue_time = create_queue_time_histogram() if self.server_args.enable_metrics else None

    def log_prefill_stats(
        self: Scheduler,
        adder: PrefillAdder,
        can_run_list: list[Req],
        running_bs: int,
    ):
        if self.queue_time is not None:
            record_queue_wait_times(can_run_list, self.queue_time)
        gap_latency = time.perf_counter() - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_input_throughput = self.last_prefill_tokens / gap_latency
        self.last_prefill_tokens = adder.log_input_tokens

        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
            token_msg = (
                f"full token usage: {full_token_usage:.2f}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_msg = f"token usage: {token_usage:.2f}, "

        num_new_seq = sum(len(v) for v in adder.can_run_list.values())
        f = (
            f"Prefill batch. "
            f"#new-seq: {num_new_seq}, "
            f"#new-token: {adder.log_input_tokens}, "
            f"#cached-token: {adder.log_hit_tokens}, "
            f"{token_msg}"
        )

        f += f"#running-req: {running_bs}, "
        if self.dp_size > 1:
            per_dp_prefill = [len(adder.can_run_list[i]) for i in range(self.dp_size)]
            per_dp_running = [
                (
                    len(self.running_batch.reqs_info[i].reqs)
                    if self.running_batch.reqs_info[i].reqs
                    else 0
                )
                for i in range(self.dp_size)
            ]
            f += f"#prefill per DP: {per_dp_prefill}, #running per DP: {per_dp_running}, "

        f += f"#queue-req: {len(self.waiting_queue)}, "

        logger.info(f)

    def log_decode_stats(self: Scheduler, running_batch: ScheduleBatch = None):
        batch = running_batch or self.running_batch

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency
        self.num_generated_tokens = 0
        num_running_reqs = batch.batch_size()
        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
            token_msg = (
                f"#full token: {full_num_used}, "
                f"full token usage: {full_token_usage:.2f}, "
                f"#swa token: {swa_num_used}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_msg = f"#token: {num_used}, token usage: {token_usage:.2f}, "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        msg = f"Decode batch. #running-req: {num_running_reqs}, {token_msg}"

        if batch.dp_size > 1:
            per_dp_running = [len(info.reqs) if info.reqs else 0 for info in batch.reqs_info]
            msg += f"#running-req per DP: {per_dp_running}, "

        if (
            self.spec_algorithm is not None
            and not self.spec_algorithm.is_none()
            and self.draft_token > 0
            and self.spec_num_forward_ct > 0
        ):
            accept_ratio = self.accept_token / self.draft_token
            accept_len = self.accept_token / self.spec_num_forward_ct
            self.accept_token = 0
            self.draft_token = 0
            self.spec_num_forward_ct = 0
            msg += f"accept-len {accept_len:.2f}, accept-ratio {accept_ratio:.2f}, "

        msg += (
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )

        if batch.cache_miss_count > 0:
            msg += f"#cache_miss: {batch.cache_miss_count}"

        logger.info(msg)
