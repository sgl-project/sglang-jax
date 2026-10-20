"""SGLANG_JAX_SC_ALLREDUCE_MIN_BYTES adds the SparseCore all-reduce size threshold
to decode-shaped executables (AOT decode path and the speculative decode
executables) and leaves prefill untouched; unset = 32 MiB (production), 0 = off."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from unittest import mock

from sgl_jax.srt.model_executor import aot_dispatch
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

AR_KEY = "xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"
GATHER_KEY = "xla_tpu_offload_gather_to_sparsecore"


class _Batch:
    def __init__(self, mode):
        self.forward_mode = mode


def _on_tpu():
    return mock.patch.object(aot_dispatch, "is_tpu_runtime", return_value=True)


def _spec_options():
    # imported lazily: importing draft_extend_fused at collection time changes the
    # order-dependent state test_aot_dispatch.py relies on (pre-existing coupling)
    from sgl_jax.srt.speculative import draft_extend_fused

    return draft_extend_fused._spec_decode_compiler_options()


class ScAllReduceMinBytesTest(unittest.TestCase):
    def test_zero_leaves_decode_options_alone(self):
        with mock.patch.dict(os.environ, {"SGLANG_JAX_SC_ALLREDUCE_MIN_BYTES": "0"}), _on_tpu():
            os.environ.pop("SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD", None)
            self.assertIsNone(aot_dispatch.sc_all_reduce_min_bytes_compiler_options())
            self.assertIsNone(aot_dispatch.decode_compiler_options())
            self.assertIsNone(aot_dispatch.decode_no_sc_gather_compiler_options_fn())
            self.assertIsNone(_spec_options())

    def test_unset_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False), _on_tpu():
            os.environ.pop("SGLANG_JAX_SC_ALLREDUCE_MIN_BYTES", None)
            self.assertIsNone(aot_dispatch.sc_all_reduce_min_bytes_compiler_options())

    def test_set_applies_to_decode_only(self):
        env = {"SGLANG_JAX_SC_ALLREDUCE_MIN_BYTES": str(32 << 20)}
        with mock.patch.dict(os.environ, env), _on_tpu():
            os.environ.pop("SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD", None)
            fn = aot_dispatch.decode_no_sc_gather_compiler_options_fn()
            self.assertEqual(fn((_Batch(ForwardMode.DECODE),)), {AR_KEY: "33554432"})
            self.assertIsNone(fn((_Batch(ForwardMode.EXTEND),)))
            self.assertEqual(_spec_options(), {AR_KEY: "33554432"})

    def test_merges_with_gather_workaround(self):
        env = {
            "SGLANG_JAX_SC_ALLREDUCE_MIN_BYTES": "1048576",
            "SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD": "1",
        }
        with mock.patch.dict(os.environ, env), _on_tpu():
            options = aot_dispatch.decode_compiler_options()
            self.assertEqual(options[AR_KEY], "1048576")
            self.assertEqual(options[GATHER_KEY], "false")
            self.assertEqual(
                options["xla_tpu_offload_all_supported_gathers_to_sparsecore"], "false"
            )
            self.assertEqual(_spec_options(), options)

    def test_not_tpu_or_bad_value_is_ignored(self):
        env = {"SGLANG_JAX_SC_ALLREDUCE_MIN_BYTES": str(32 << 20)}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(aot_dispatch, "is_tpu_runtime", return_value=False),
        ):
            self.assertIsNone(aot_dispatch.sc_all_reduce_min_bytes_compiler_options())
        for bad in ("32MiB", "0", "-1"):
            with mock.patch.dict(os.environ, {"SGLANG_JAX_SC_ALLREDUCE_MIN_BYTES": bad}), _on_tpu():
                self.assertIsNone(aot_dispatch.sc_all_reduce_min_bytes_compiler_options())


if __name__ == "__main__":
    unittest.main()
