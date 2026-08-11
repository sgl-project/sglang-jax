import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import jax

from sgl_jax.srt.managers.scheduler_profiler_mixing import (
    _make_profiler_options,
    _ProfileManager,
    _should_profile_this_process,
    _StageBasedTrigger,
)


class TestProfilerTracerLevels(unittest.TestCase):
    """An explicit tracer level of 0 must reach jax.profiler.ProfileOptions.

    0 disables the corresponding tracer; None means "unset, keep JAX's default".
    """

    def _captured_options(self, **levels):
        manager = _ProfileManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            manager.configure(
                output_dir=tmpdir,
                num_steps=1,
                interesting_stages=["decode"],
                **levels,
            )
            with mock.patch.object(jax.profiler, "start_trace") as start_trace:
                manager._do_start(stage="decode")
        return start_trace.call_args.kwargs["profiler_options"]

    def _plain_profile_options(self):
        return SimpleNamespace(
            host_tracer_level=0,
            python_tracer_level=0,
            advanced_configuration={},
        )

    def test_explicit_zero_disables_tracers(self):
        options = self._captured_options(host_tracer_level=0, python_tracer_level=0)
        self.assertEqual(options.host_tracer_level, 0)
        self.assertEqual(options.python_tracer_level, 0)

    def test_none_keeps_jax_defaults(self):
        options = self._captured_options()
        defaults = jax.profiler.ProfileOptions()
        self.assertEqual(options.host_tracer_level, defaults.host_tracer_level)
        self.assertEqual(options.python_tracer_level, defaults.python_tracer_level)

    def test_explicit_nonzero_is_applied(self):
        options = self._captured_options(host_tracer_level=3, python_tracer_level=1)
        self.assertEqual(options.host_tracer_level, 3)
        self.assertEqual(options.python_tracer_level, 1)

    def test_profile_max_hosts_limits_pjrt_processes(self):
        with mock.patch.dict(os.environ, {"SGLANG_PROFILE_MAX_HOSTS": "1"}):
            with mock.patch.object(jax, "process_index", return_value=0):
                self.assertTrue(_should_profile_this_process())
            with mock.patch.object(jax, "process_index", return_value=1):
                self.assertFalse(_should_profile_this_process())

    def test_profile_num_chips_per_task_is_applied(self):
        with (
            mock.patch.dict(
                os.environ,
                {"SGLANG_PROFILE_NUM_CHIPS_PER_TASK": "1"},
            ),
            mock.patch.object(
                jax.profiler,
                "ProfileOptions",
                return_value=self._plain_profile_options(),
            ),
        ):
            options = _make_profiler_options(0, 0)

        self.assertEqual(
            options.advanced_configuration["tpu_num_chips_to_profile_per_task"],
            1,
        )
        self.assertFalse(hasattr(options, "tpu_num_chips_to_profile_per_task"))

    def test_profile_sparse_core_limits_are_applied(self):
        with (
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_PROFILE_NUM_SPARSE_CORES_TO_TRACE": "1",
                    "SGLANG_PROFILE_NUM_SPARSE_CORE_TILES_TO_TRACE": "1",
                },
            ),
            mock.patch.object(
                jax.profiler,
                "ProfileOptions",
                return_value=self._plain_profile_options(),
            ),
        ):
            options = _make_profiler_options(0, 0)

        self.assertEqual(
            options.advanced_configuration["tpu_num_sparse_cores_to_trace"],
            1,
        )
        self.assertEqual(
            options.advanced_configuration["tpu_num_sparse_core_tiles_to_trace"],
            1,
        )
        self.assertFalse(hasattr(options, "tpu_num_sparse_cores_to_trace"))
        self.assertFalse(hasattr(options, "tpu_num_sparse_core_tiles_to_trace"))

    def test_profile_num_chips_per_task_must_be_positive(self):
        for value in ("0", "-1", "not-an-integer"):
            with (
                self.subTest(value=value),
                mock.patch.dict(
                    os.environ,
                    {"SGLANG_PROFILE_NUM_CHIPS_PER_TASK": value},
                ),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                _make_profiler_options(0, 0)

    def test_profile_sparse_core_limits_must_be_positive(self):
        for environment_name in (
            "SGLANG_PROFILE_NUM_SPARSE_CORES_TO_TRACE",
            "SGLANG_PROFILE_NUM_SPARSE_CORE_TILES_TO_TRACE",
        ):
            for value in ("0", "-1", "not-an-integer"):
                with (
                    self.subTest(environment_name=environment_name, value=value),
                    mock.patch.dict(os.environ, {environment_name: value}),
                    self.assertRaisesRegex(ValueError, "positive integer"),
                ):
                    _make_profiler_options(0, 0)

    def test_stage_trigger_profiles_exactly_the_requested_number_of_steps(self):
        events = []
        trigger = _StageBasedTrigger(
            on_start=lambda **kwargs: events.append(("start", kwargs["stage"])),
            on_stop=lambda: events.append(("stop", None)),
        )
        trigger.configure(num_steps=3, interesting_stages=["decode"])

        for _ in range(3):
            trigger.step("decode")

        self.assertEqual(events, [("start", "decode")])
        self.assertTrue(trigger.is_configured)

        # The predicate runs immediately before each forward. The fourth call
        # must stop the trace before that forward begins.
        trigger.step("decode")
        self.assertEqual(events, [("start", "decode"), ("stop", None)])
        self.assertFalse(trigger.is_configured)

    def test_stage_profile_synchronizes_before_stopping_trace(self):
        events = []
        manager = _ProfileManager(synchronize=lambda: events.append("synchronize"))
        manager._trace_active = True

        with (
            mock.patch.object(
                jax,
                "effects_barrier",
                side_effect=lambda: events.append("effects_barrier"),
            ),
            mock.patch.object(
                jax.profiler,
                "stop_trace",
                side_effect=lambda: events.append("stop_trace"),
            ),
        ):
            manager._do_stop()

        self.assertEqual(events, ["synchronize", "effects_barrier", "stop_trace"])

    def test_stage_profile_synchronizes_on_process_without_active_trace(self):
        events = []
        manager = _ProfileManager(synchronize=lambda: events.append("synchronize"))

        with (
            mock.patch.object(jax, "effects_barrier") as effects_barrier,
            mock.patch.object(jax.profiler, "stop_trace") as stop_trace,
        ):
            manager._do_stop()

        self.assertEqual(events, ["synchronize"])
        effects_barrier.assert_not_called()
        stop_trace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
