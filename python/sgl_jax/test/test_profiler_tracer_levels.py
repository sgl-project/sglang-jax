import tempfile
import unittest
from unittest import mock

import jax

from sgl_jax.srt.managers.scheduler_profiler_mixing import _ProfileManager


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


if __name__ == "__main__":
    unittest.main()
