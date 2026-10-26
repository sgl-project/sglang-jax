"""SGLANG_JAX_PREFILL_XLA_OPTIONS adds per-executable XLA options to the
prefill-shaped executables (EXTEND / MIXED) and leaves decode-shaped ones
(DECODE, TARGET_VERIFY, DRAFT_EXTEND) untouched; unset = no options
set (environ.py), empty = none. Mirrors SGLANG_JAX_DECODE_XLA_OPTIONS on the
other side of the split."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from unittest import mock

from sgl_jax.srt.model_executor import aot_dispatch
from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

ENV = "SGLANG_JAX_PREFILL_XLA_OPTIONS"
COMBINER = "xla_jf_crs_combiner_threshold_in_bytes"
A2A = "xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all"


class _Batch:
    def __init__(self, mode, num_tokens=None):
        self.forward_mode = mode
        if num_tokens is not None:
            import numpy as np

            self.input_ids = np.zeros((num_tokens,), np.int32)


class PrefillXlaOptionsTest(unittest.TestCase):
    def test_empty_is_none(self):
        with mock.patch.dict(os.environ, {ENV: ""}):
            self.assertIsNone(aot_dispatch.prefill_compiler_options())
            self.assertIsNone(aot_dispatch.prefill_compiler_options_fn())

    def test_unset_is_no_options(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            self.assertIsNone(aot_dispatch.prefill_compiler_options())
            self.assertIsNone(aot_dispatch.prefill_compiler_options_fn())

    def test_production_set_parses(self):
        prod = (
            f"{COMBINER}=1,xla_jf_enable_multi_output_fusion=false,{A2A}=true,"
            "xla_tpu_sparse_core_all_reduce_latency_multiplier=8,"
            "xla_jf_enable_producer_consumer_multi_output_fusion=false,"
            "xla_tpu_scheduling_annotation_deannotate_unsupported_groups=true"
        )
        with mock.patch.dict(os.environ, {ENV: prod}):
            opts = aot_dispatch.prefill_compiler_options()
            self.assertEqual(opts[COMBINER], "1")
            self.assertEqual(opts[A2A], "true")
            self.assertEqual(opts["xla_jf_enable_producer_consumer_multi_output_fusion"], "false")
            self.assertEqual(len(opts), 6)

    def test_parses_flag_list(self):
        with mock.patch.dict(os.environ, {ENV: f"{COMBINER}=1, {A2A}=true"}):
            self.assertEqual(aot_dispatch.prefill_compiler_options(), {COMBINER: "1", A2A: "true"})

    def test_malformed_entry_raises(self):
        with mock.patch.dict(os.environ, {ENV: f"{COMBINER}"}):
            with self.assertRaises(ValueError):
                aot_dispatch.prefill_compiler_options()

    def test_fn_applies_to_prefill_shaped_modes_only(self):
        with mock.patch.dict(os.environ, {ENV: f"{COMBINER}=1"}):
            fn = aot_dispatch.prefill_compiler_options_fn()
            self.assertEqual(fn((_Batch(ForwardMode.EXTEND),)), {COMBINER: "1"})
            self.assertEqual(fn((_Batch(ForwardMode.MIXED),)), {COMBINER: "1"})
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND):
                self.assertIsNone(fn((_Batch(mode),)), mode)

    def test_compilation_manager_merges_by_mode(self):
        env = {ENV: f"{COMBINER}=1", "SGLANG_JAX_SC_ALLREDUCE_MIN_BYTES": "1048576"}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(aot_dispatch, "is_tpu_runtime", return_value=True),
        ):
            os.environ.pop("SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD", None)
            backend = mock.Mock(compiler_options=None)
            extend = CompilationManager.compiler_options(backend, _Batch(ForwardMode.EXTEND))
            decode = CompilationManager.compiler_options(backend, _Batch(ForwardMode.DECODE))
            self.assertEqual(extend.get(COMBINER), "1")
            self.assertNotIn("xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes", extend)
            self.assertNotIn(COMBINER, decode)
            self.assertEqual(
                decode.get("xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"), "1048576"
            )

    def test_is_prefill_shaped(self):
        self.assertTrue(aot_dispatch.is_prefill_shaped(ForwardMode.EXTEND))
        self.assertTrue(aot_dispatch.is_prefill_shaped(ForwardMode.MIXED))
        self.assertFalse(aot_dispatch.is_prefill_shaped(ForwardMode.DECODE))
        self.assertFalse(aot_dispatch.is_prefill_shaped(ForwardMode.DRAFT_EXTEND))
        self.assertFalse(aot_dispatch.is_prefill_shaped(ForwardMode.TARGET_VERIFY))
        self.assertFalse(aot_dispatch.is_prefill_shaped(ForwardMode.IDLE))


if __name__ == "__main__":
    unittest.main()


class SelectRunModelByModeTest(unittest.TestCase):
    def test_selector(self):
        from sgl_jax.srt.model_executor.model_forward import select_run_model_by_mode

        default, prefill = object(), object()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_JAX_PREFILL_XLA_OPTIONS_MIN_TOKENS", None)
            pick = select_run_model_by_mode(default, prefill)
            self.assertIs(pick(_Batch(ForwardMode.EXTEND)), prefill)
            self.assertIs(pick(_Batch(ForwardMode.MIXED)), prefill)
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND):
                self.assertIs(pick(_Batch(mode)), default)
            pick = select_run_model_by_mode(default, None)
            self.assertIs(pick(_Batch(ForwardMode.EXTEND)), default)

    def test_min_tokens_threshold(self):
        from sgl_jax.srt.model_executor.model_forward import select_run_model_by_mode

        default, prefill = object(), object()
        with mock.patch.dict(
            os.environ, {"SGLANG_JAX_PREFILL_XLA_OPTIONS_MIN_TOKENS": "2048", ENV: f"{COMBINER}=1"}
        ):
            pick = select_run_model_by_mode(default, prefill)
            self.assertIs(pick(_Batch(ForwardMode.EXTEND, 1024)), default)
            self.assertIs(pick(_Batch(ForwardMode.EXTEND, 2048)), prefill)
            self.assertIs(pick(_Batch(ForwardMode.EXTEND)), prefill)  # unknown size -> apply
            fn = aot_dispatch.prefill_compiler_options_fn()
            self.assertIsNone(fn((_Batch(ForwardMode.EXTEND, 512),)))
            self.assertEqual(fn((_Batch(ForwardMode.EXTEND, 4096),)), {COMBINER: "1"})
            self.assertEqual(aot_dispatch.prefill_xla_options_min_tokens(), 2048)


class SpecPrefillOptionsTest(unittest.TestCase):
    def test_spec_prefill_jit_gets_prefill_options(self):
        # imported lazily (see test_sc_allreduce_compile_option.py for the order coupling)
        from sgl_jax.srt.speculative import draft_extend_fused

        with mock.patch.dict(os.environ, {ENV: f"{COMBINER}=1"}):
            self.assertEqual(draft_extend_fused._spec_prefill_compiler_options(), {COMBINER: "1"})
        with mock.patch.dict(os.environ, {ENV: ""}):
            self.assertIsNone(draft_extend_fused._spec_prefill_compiler_options())


class SpecPrefillDispatchTest(unittest.TestCase):
    def test_build_prefill_dispatches_by_token_count(self):
        from sgl_jax.srt.speculative import draft_extend_fused as def_

        calls = []

        def fake_build(num_layers, topk, hidden_relay=None, compiler_options=None):
            return lambda *a, **k: calls.append(compiler_options) or compiler_options

        with (
            mock.patch.dict(os.environ, {"SGLANG_JAX_PREFILL_XLA_OPTIONS_MIN_TOKENS": "2048"}),
            mock.patch.object(def_, "_SPEC_PREFILL_COMPILER_OPTIONS", {COMBINER: "1"}),
            mock.patch.object(def_, "_build_prefill_jit", side_effect=fake_build),
        ):
            fn = def_._build_prefill(3, 1)
            self.assertEqual(fn(None, None, None, _Batch(ForwardMode.EXTEND, 1024)), None)
            self.assertEqual(
                fn(None, None, None, _Batch(ForwardMode.EXTEND, 4096)), {COMBINER: "1"}
            )
        with (
            mock.patch.dict(os.environ, {"SGLANG_JAX_PREFILL_XLA_OPTIONS_MIN_TOKENS": "0"}),
            mock.patch.object(def_, "_SPEC_PREFILL_COMPILER_OPTIONS", {COMBINER: "1"}),
            mock.patch.object(def_, "_build_prefill_jit", side_effect=fake_build),
        ):
            fn = def_._build_prefill(3, 1)
            self.assertEqual(fn(None, None, None, _Batch(ForwardMode.EXTEND, 512)), {COMBINER: "1"})
