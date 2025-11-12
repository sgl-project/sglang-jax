import logging
import time
from collections import defaultdict

from sgl_jax.srt.managers.schedule_policy import PrefillAdder
from sgl_jax.srt.managers.scheduler import Req, ScheduleBatch
from sgl_jax.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")


class SchedulerMetricsMixin:
    def init_metrics(self):
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self.spec_num_total_accepted_tokens = 0
        self.spec_num_total_forward_ct = 0
        self.cum_spec_accept_length = 0
        self.cum_spec_accept_count = 0
        self.total_retracted_reqs = 0

    def log_prefill_stats(
        self,
        adder: PrefillAdder,
        can_run_list: list[Req],
        running_bs: int,
    ):
        gap_latency = time.perf_counter() - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_input_throughput = self.last_prefill_tokens / gap_latency
        self.last_prefill_tokens = adder.log_input_tokens

        num_used, token_usage, _, _ = self._get_token_info()
        token_msg = f"token usage: {token_usage:.2f}, "

        num_new_seq = len(can_run_list)
        f = (
            f"Prefill batch. "
            f"#new-seq: {num_new_seq}, "
            f"#new-token: {adder.log_input_tokens}, "
            f"#cached-token: {adder.log_hit_tokens}, "
            f"{token_msg}"
        )

        f += f"#running-req: {running_bs}, "
        f += f"#queue-req: {len(self.waiting_queue)}, "

        logger.info(f)

    def log_decode_stats(self, running_batch: ScheduleBatch = None):
        batch = running_batch or self.running_batch

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency
        self.num_generated_tokens = 0
        num_running_reqs = len(batch.reqs)
        num_used, token_usage, _, _ = self._get_token_info()
        token_msg = f"#token: {num_used}, token usage: {token_usage:.2f}, "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        msg = f"Decode batch. #running-req: {num_running_reqs}, {token_msg}"

        msg += (
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )

        if not running_batch.spec_algorithm.is_none():
            accept_ratio = (self.accept_token / self.draft_token) * 100
            self.accept_token = 0
            self.draft_token = 0
            msg += f"accept_ratio {accept_ratio:.2f}%"

        msg += f"#cache_miss: {batch.cache_miss_count}"

        logger.info(msg)
