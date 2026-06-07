from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np


def _install_import_stubs():
    jax_stub = types.ModuleType("jax")
    jax_stub.Array = np.ndarray
    sys.modules.setdefault("jax", jax_stub)

    jax_src_stub = types.ModuleType("jax._src")
    mesh_stub = types.ModuleType("jax._src.mesh")
    mesh_stub.Mesh = type("Mesh", (), {})
    jax_src_stub.mesh = mesh_stub
    sys.modules.setdefault("jax._src", jax_src_stub)
    sys.modules.setdefault("jax._src.mesh", mesh_stub)

    server_arg_attrs = {
        "device": None,
        "chunked_prefill_size": None,
        "disable_radix_cache": None,
        "speculative_accept_threshold_single": None,
        "speculative_accept_threshold_acc": None,
        "enable_deterministic_sampling": None,
    }

    modules = {
        "sgl_jax.global_config": {"global_config": object()},
        "sgl_jax.srt.configs.model_config": {"ModelConfig": type("ModelConfig", (), {})},
        "sgl_jax.srt.mem_cache.allocator": {
            "BaseTokenToKVPoolAllocator": type("BaseTokenToKVPoolAllocator", (), {}),
            "SWATokenToKVPoolAllocator": type("SWATokenToKVPoolAllocator", (), {}),
        },
        "sgl_jax.srt.mem_cache.base_prefix_cache": {
            "BasePrefixCache": type("BasePrefixCache", (), {})
        },
        "sgl_jax.srt.mem_cache.chunk_cache": {"ChunkCache": type("ChunkCache", (), {})},
        "sgl_jax.srt.mem_cache.common": {
            "alloc_paged_token_slots_extend": lambda *args, **kwargs: None,
            "alloc_token_slots": lambda *args, **kwargs: None,
            "evict_from_tree_cache": lambda *args, **kwargs: None,
            "release_kv_cache": lambda *args, **kwargs: None,
        },
        "sgl_jax.srt.mem_cache.memory_pool": {
            "HybridReqToTokenPool": type("HybridReqToTokenPool", (), {}),
            "ReqToTokenPool": type("ReqToTokenPool", (), {}),
        },
        "sgl_jax.srt.mem_cache.radix_cache": {"RadixKey": list},
        "sgl_jax.srt.mem_cache.swa_radix_cache": {"SWARadixCache": type("SWARadixCache", (), {})},
        "sgl_jax.srt.precision_tracer": {
            "PrecisionTracerRequestMetadata": type("PrecisionTracerRequestMetadata", (), {}),
            "precision_tracer": object(),
        },
        "sgl_jax.srt.sampling.sampling_batch_info": {
            "SamplingBatchInfo": type("SamplingBatchInfo", (), {})
        },
        "sgl_jax.srt.sampling.sampling_params": {
            "DEFAULT_SAMPLING_SEED": 0,
            "SamplingParams": type(
                "SamplingParams", (), {"__init__": lambda self, *args, **kwargs: None}
            ),
        },
        "sgl_jax.srt.server_args": {"ServerArgs": type("ServerArgs", (), server_arg_attrs)},
        "sgl_jax.srt.utils.common_utils": {
            "get_bool_env_var": lambda *args, **kwargs: False,
            "pad_to_bucket": lambda value, buckets: (value, value),
        },
    }

    class CaptureHiddenMode:
        FULL = 1
        NULL = 0

    modules["sgl_jax.srt.model_executor.forward_batch_info"] = {
        "CaptureHiddenMode": CaptureHiddenMode,
        "ForwardMode": type("ForwardMode", (), {}),
    }

    for module_name, attrs in modules.items():
        module = types.ModuleType(module_name)
        for name, value in attrs.items():
            setattr(module, name, value)
        sys.modules.setdefault(module_name, module)


_install_import_stubs()

from sgl_jax.srt.managers.schedule_batch import (  # noqa: E402
    ScheduleBatch,
    ScheduleReqsInfo,
)


class _ExtendMode:
    def is_extend(self):
        return True

    def is_decode(self):
        return False


class TestMiMoV25ArInputEmbeddingMerge(unittest.TestCase):
    def _new_batch(self, *, req, seq_len=4, prefix_len=1):
        info = ScheduleReqsInfo(
            reqs=[req],
            seq_lens=np.array([seq_len], dtype=np.int32),
            prefix_lens=[prefix_len],
        )
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.reqs_info = [info]
        batch.dp_size = 1
        batch.forward_mode = _ExtendMode()
        return batch

    def test_merge_multimodal_slices_extend_window_into_input_embedding(self):
        embedding = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        req = SimpleNamespace(rid="rid-ok", multimodal_embedding=embedding, mm_inputs={})
        batch = self._new_batch(req=req, seq_len=4, prefix_len=1)

        merged = batch._merge_multimodal(per_dp_token_size=4, total_token_size=4)

        self.assertIsNotNone(merged["input_embedding"])
        np.testing.assert_array_equal(merged["input_embedding"][:3], embedding[1:4])
        np.testing.assert_array_equal(
            merged["input_embedding"][3:], np.zeros((1, 3), dtype=np.float32)
        )
        self.assertIsNone(merged["mrope_positions"])
        self.assertFalse(merged["apply_for_deepstack"])
        self.assertIsNone(merged["deepstack_visual_embedding"])

    def test_merge_multimodal_rejects_short_input_embedding(self):
        embedding = np.ones((3, 3), dtype=np.float32)
        req = SimpleNamespace(rid="rid-short", multimodal_embedding=embedding, mm_inputs={})
        batch = self._new_batch(req=req, seq_len=4, prefix_len=1)

        with self.assertRaisesRegex(ValueError, "multimodal_embedding length mismatch"):
            batch._merge_multimodal(per_dp_token_size=4, total_token_size=4)


if __name__ == "__main__":
    unittest.main()
