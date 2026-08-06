import os
import tempfile
import unittest
from unittest import mock

import jax

from sgl_jax.srt.managers.scheduler_profiler_mixing import (
    _ProfileManager,
    _make_profiler_options,
    _should_profile_this_process,
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
        with mock.patch.dict(
            os.environ,
            {"SGLANG_PROFILE_NUM_CHIPS_PER_TASK": "1"},
        ):
            options = _make_profiler_options(0, 0)

        self.assertEqual(options.tpu_num_chips_to_profile_per_task, 1)

    def test_profile_num_chips_per_task_must_be_positive(self):
        for value in ("0", "-1", "not-an-integer"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ,
                    {"SGLANG_PROFILE_NUM_CHIPS_PER_TASK": value},
                ):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        _make_profiler_options(0, 0)


if __name__ == "__main__":
    unittest.main()
