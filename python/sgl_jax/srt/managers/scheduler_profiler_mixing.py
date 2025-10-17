import logging
import os
from pathlib import Path

import jax

from sgl_jax.srt.managers.io_struct import ProfileReq, ProfileReqOutput, ProfileReqType

logger = logging.getLogger(__name__)


class SchedulerProfilerMixin:
    def init_profier(self):
        self.profiler_output_dir: str | None = None
        self.profile_id: str | None = None
        self.profiler_start_forward_ct: int | None = None
        self.profiler_target_forward_ct: int | None = None
        self.profile_steps: int | None = None
        self.profile_in_progress: bool = False

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

        if start_step:
            self.profiler_start_forward_ct = max(start_step, self.forward_ct + 1)

        if num_steps:
            self.profile_steps = num_steps
            if start_step:
                self.profiler_target_forward_ct = (
                    self.profiler_start_forward_ct + num_steps
                )
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

    def stop_profile(self) -> ProfileReqOutput | None:
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
        if (
            self.profiler_target_forward_ct
            and self.profiler_target_forward_ct <= self.forward_ct
        ):
            self.stop_profile()
        if (
            self.profiler_start_forward_ct
            and self.profiler_start_forward_ct == self.forward_ct
        ):
            self.start_profile(
                self.profiler_output_dir, None, self.profile_steps, self.profile_id
            )

    def profile(self, recv_req: ProfileReq):
        if recv_req.type == ProfileReqType.START_PROFILE:
            return self.start_profile(
                recv_req.output_dir,
                recv_req.start_step,
                recv_req.num_steps,
                recv_req.host_tracer_level,
                recv_req.python_tracer_level,
                recv_req.profile_id,
            )
        else:
            return self.stop_profile()
