from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sgl_jax.srt.multimodal.manager.host_processor import resolve_host_processor


class TestMiMoV25HostProcessorResolver(unittest.TestCase):
    def test_matching_processor_failure_is_not_silently_ignored(self):
        module_name = "_fake_mimo_v25_host_processor"
        module = types.ModuleType(module_name)

        class _Processor:
            @staticmethod
            def matches(mm_config):
                return True

            @staticmethod
            def from_hf_processor(*args, **kwargs):
                raise RuntimeError("wrapper failed")

        module.FakeProcessor = _Processor
        sys.modules[module_name] = module
        try:
            with (
                patch(
                    "sgl_jax.srt.multimodal.manager.host_processor._PROCESSOR_SPECS",
                    [(module_name, "FakeProcessor")],
                ),
                self.assertRaisesRegex(RuntimeError, "wrapper failed"),
            ):
                resolve_host_processor(SimpleNamespace(model_type="mimo_v2_5"), "model", object())
        finally:
            sys.modules.pop(module_name, None)

    def test_non_matching_processor_returns_original(self):
        module_name = "_fake_non_matching_host_processor"
        module = types.ModuleType(module_name)

        class _Processor:
            @staticmethod
            def matches(mm_config):
                return False

        module.FakeProcessor = _Processor
        sys.modules[module_name] = module
        hf_processor = object()
        try:
            with patch(
                "sgl_jax.srt.multimodal.manager.host_processor._PROCESSOR_SPECS",
                [(module_name, "FakeProcessor")],
            ):
                resolved = resolve_host_processor(
                    SimpleNamespace(model_type="other"), "model", hf_processor
                )
        finally:
            sys.modules.pop(module_name, None)

        self.assertIs(resolved, hf_processor)


if __name__ == "__main__":
    unittest.main()
