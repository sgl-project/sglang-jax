import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax

from sgl_jax.srt.managers.io_struct import ProfileReq, ProfileReqOutput, ProfileReqType
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)


def _get_stage_from_forward_mode(forward_mode: ForwardMode) -> str | None:
    if forward_mode.is_prefill():
        return "prefill"
    elif forward_mode.is_decode():
        return "decode"
    elif forward_mode.is_idle():
        return None
    else:
        return None


class _StageBasedTrigger:
    """State machine that triggers profiler start/stop based on stage transitions.

    Ported from SGLang V2 (sglang/srt/utils/profile_utils.py:_StageBasedTrigger).
    """

    @dataclass
    class _StageConfig:
        target_count: int

    @dataclass
    class _RunningState:
        curr_stage: str
        curr_count: int

    def __init__(self, on_start: Callable, on_stop: Callable):
        self.on_start = on_start
        self.on_stop = on_stop
        self.running_state: _StageBasedTrigger._RunningState | None = None
        self.stage_configs: dict[str, _StageBasedTrigger._StageConfig] = {}

    def configure(self, num_steps: int, interesting_stages: list[str]):
        assert self.running_state is None
        self.stage_configs = {
            stage: self._StageConfig(target_count=num_steps) for stage in interesting_stages
        }

    @property
    def is_configured(self) -> bool:
        return len(self.stage_configs) > 0

    def step(self, stage: str):
        # Incr counter
        if (s := self.running_state) is not None:
            s.curr_count += 1

        # Maybe stop
        if ((s := self.running_state) is not None) and (
            (s.curr_count > self.stage_configs[s.curr_stage].target_count)
            or (stage != s.curr_stage)
        ):
            del self.stage_configs[s.curr_stage]
            self.running_state = None
            self.on_stop()

        # Maybe start
        if (self.running_state is None) and (stage in self.stage_configs):
            self.running_state = self._RunningState(
                curr_stage=stage,
                curr_count=0,
            )
            self.on_start(stage=stage)

    def reset(self):
        if self.running_state is not None:
            self.running_state = None
            self.on_stop()
        self.stage_configs.clear()


class _ProfileManager:
    """Manages stage-based profiling with JAX profiler backend."""

    def __init__(self):
        self._trigger = _StageBasedTrigger(
            on_start=self._do_start,
            on_stop=self._do_stop,
        )
        self._output_dir: str = ""
        self._host_tracer_level: int | None = None
        self._python_tracer_level: int | None = None
        self._trace_active: bool = False

    def configure(
        self,
        output_dir: str,
        num_steps: int,
        interesting_stages: list[str],
        host_tracer_level: int | None = None,
        python_tracer_level: int | None = None,
    ):
        self._output_dir = output_dir
        self._host_tracer_level = host_tracer_level
        self._python_tracer_level = python_tracer_level
        self._trigger.configure(num_steps=num_steps, interesting_stages=interesting_stages)

    @property
    def is_configured(self) -> bool:
        return self._trigger.is_configured

    def step(self, forward_mode: ForwardMode) -> None:
        if not self._trigger.is_configured:
            return
        stage = _get_stage_from_forward_mode(forward_mode)
        if stage is None:
            return
        self._trigger.step(stage)

    def stop(self):
        """Force stop any active trace (for manual /stop_profile)."""
        self._trigger.reset()

    def _do_start(self, stage: str):
        stage_dir = os.path.join(self._output_dir, stage)
        Path(stage_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Stage-based profiling: starting trace for '%s' -> %s", stage, stage_dir)

        profiler_options = jax.profiler.ProfileOptions()
        if self._host_tracer_level:
            profiler_options.host_tracer_level = self._host_tracer_level
        if self._python_tracer_level:
            profiler_options.python_tracer_level = self._python_tracer_level

        jax.profiler.start_trace(stage_dir, profiler_options=profiler_options)
        self._trace_active = True

    def _do_stop(self):
        if self._trace_active:
            logger.info("Stage-based profiling: stopping trace")
            jax.profiler.stop_trace()
            self._trace_active = False


class SchedulerProfilerMixin:
    def init_profier(self):
        self.profiler_output_dir: str | None = None
        self.profile_id: str | None = None
        self.profiler_start_forward_ct: int | None = None
        self.profiler_target_forward_ct: int | None = None
        self.profile_steps: int | None = None
        self.profile_in_progress: bool = False
        self.host_tracer_level: int | None = None
        self.python_tracer_level: int | None = None
        self._profile_manager = _ProfileManager()

    def start_profile(
        self,
        output_dir: str | None,
        start_step: int | None,
        num_steps: int | None,
        host_tracer_level: int | None,
        python_tracer_level: int | None,
        profile_id: str,
    ) -> ProfileReqOutput:
        if self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is already in progress. Call /stop_profile first.",
            )

        if output_dir is None:
            output_dir = os.getenv("SGLANG_JAX_PROFILER_DIR", "/tmp")

        # check permission for output_dir
        tmp_output_dir = output_dir
        while not os.path.exists(tmp_output_dir):
            tmp_output_dir = os.path.dirname(tmp_output_dir)
        if not os.access(tmp_output_dir, os.W_OK):
            return ProfileReqOutput(
                success=False,
                message=f"no permission to write the {output_dir}",
            )

        self.profiler_output_dir = output_dir
        self.profile_id = profile_id
        self.host_tracer_level = host_tracer_level
        self.python_tracer_level = python_tracer_level

        if start_step:
            self.profiler_start_forward_ct = max(start_step, self.forward_ct + 1)

        if num_steps:
            self.profile_steps = num_steps
            if start_step:
                self.profiler_target_forward_ct = self.profiler_start_forward_ct + num_steps
            else:
                self.profiler_target_forward_ct = self.forward_ct + num_steps
        else:
            self.profiler_target_forward_ct = None

        if start_step:
            return ProfileReqOutput(success=True, message="Succeeded")

        logger.info(
            "Profiling starts. Traces will be saved to: %s (with profile id: %s)",
            self.profiler_output_dir,
            self.profile_id,
        )

        profiler_options = jax.profiler.ProfileOptions()
        if host_tracer_level:
            profiler_options.host_tracer_level = host_tracer_level
        if python_tracer_level:
            profiler_options.python_tracer_level = python_tracer_level

        print(f"profiler_options: {profiler_options}")

        jax.profiler.start_trace(
            self.profiler_output_dir,
            profiler_options=profiler_options,
        )

        self.profile_in_progress = True
        return ProfileReqOutput(success=True, message="Succeeded")

    def _start_stage_profile(self, recv_req: ProfileReq) -> ProfileReqOutput:
        if self.profile_in_progress or self._profile_manager.is_configured:
            return ProfileReqOutput(
                success=False,
                message="Profiling is already in progress. Call /stop_profile first.",
            )

        output_dir = recv_req.output_dir or os.getenv("SGLANG_JAX_PROFILER_DIR", "/tmp")

        # check permission for output_dir
        tmp_output_dir = output_dir
        while not os.path.exists(tmp_output_dir):
            tmp_output_dir = os.path.dirname(tmp_output_dir)
        if not os.access(tmp_output_dir, os.W_OK):
            return ProfileReqOutput(
                success=False,
                message=f"no permission to write {output_dir}",
            )

        num_steps = recv_req.num_steps or 5
        interesting_stages = recv_req.profile_stages or ["prefill", "decode"]

        self._profile_manager.configure(
            output_dir=output_dir,
            num_steps=num_steps,
            interesting_stages=interesting_stages,
            host_tracer_level=recv_req.host_tracer_level,
            python_tracer_level=recv_req.python_tracer_level,
        )

        self.profile_in_progress = True
        self.profile_id = recv_req.profile_id
        self.profiler_output_dir = output_dir

        logger.info(
            "Stage-based profiling configured: stages=%s, num_steps=%d, output=%s",
            interesting_stages,
            num_steps,
            output_dir,
        )
        return ProfileReqOutput(success=True, message="Stage-based profiling configured")

    def stop_profile(self) -> ProfileReqOutput | None:
        # Stop stage-based profiling if active
        if self._profile_manager.is_configured:
            self._profile_manager.stop()
            self.profile_in_progress = False
            logger.info("Stage-based profiling stopped.")
            return ProfileReqOutput(success=True, message="Stage-based profiling stopped.")

        if not self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is not in progress. Call /start_profile first.",
            )

        if not Path(self.profiler_output_dir).exists():
            Path(self.profiler_output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Stop profiling...")
        jax.profiler.stop_trace()

        logger.info(
            "Profiling done. Traces are saved to: %s",
            self.profiler_output_dir,
        )
        self.profile_in_progress = False
        self.profiler_start_forward_ct = None

        return ProfileReqOutput(success=True, message="Succeeded.")

    def _profile_batch_predicate(self, batch):
        # Stage-based profiling path
        if self._profile_manager.is_configured:
            self._profile_manager.step(batch.forward_mode)
            if not self._profile_manager.is_configured:
                self.profile_in_progress = False
                logger.info("Stage-based profiling completed for all stages")
            return

        # Existing step-counting path
        if self.profiler_target_forward_ct and self.profiler_target_forward_ct <= self.forward_ct:
            self.stop_profile()
        if self.profiler_start_forward_ct and self.profiler_start_forward_ct == self.forward_ct:
            self.start_profile(
                self.profiler_output_dir,
                None,
                self.profile_steps,
                self.host_tracer_level,
                self.python_tracer_level,
                self.profile_id,
            )

    def profile(self, recv_req: ProfileReq):
        if recv_req.type == ProfileReqType.START_PROFILE:
            if recv_req.profile_by_stage:
                return self._start_stage_profile(recv_req)
            return self.start_profile(
                recv_req.output_dir,
                recv_req.start_step,
                recv_req.num_steps,
                recv_req.host_tracer_level,
                recv_req.python_tracer_level,
                recv_req.profile_id,
            )
        elif recv_req.type == ProfileReqType.GET_STATUS:
            return self.get_profile_status()
        else:
            return self.stop_profile()

    def get_profile_status(self) -> ProfileReqOutput:
        in_progress = self.profile_in_progress or self._profile_manager.is_configured
        return ProfileReqOutput(
            success=True,
            message="in_progress" if in_progress else "idle",
        )
