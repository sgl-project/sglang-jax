from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np


def _install_import_stubs():
    jax_stub = types.ModuleType("jax")
    jax_stub.Array = np.ndarray

    jax_numpy_stub = types.ModuleType("jax.numpy")
    jax_numpy_stub.bfloat16 = np.float32
    jax_stub.numpy = jax_numpy_stub

    sharding_stub = types.ModuleType("jax.sharding")

    class _NamedSharding:
        def __init__(self, mesh, spec):
            self.mesh = mesh
            self.spec = spec

    class _PartitionSpec:
        def __init__(self, *parts):
            self.parts = parts

    sharding_stub.NamedSharding = _NamedSharding
    sharding_stub.PartitionSpec = _PartitionSpec

    tree_util_stub = types.ModuleType("jax.tree_util")
    tree_util_stub.register_pytree_node_class = lambda cls: cls

    sys.modules.setdefault("jax", jax_stub)
    sys.modules.setdefault("jax.numpy", jax_numpy_stub)
    sys.modules.setdefault("jax.sharding", sharding_stub)
    sys.modules.setdefault("jax.tree_util", tree_util_stub)

    model_config_stub = types.ModuleType("sgl_jax.srt.configs.model_config")
    model_config_stub.need_attention_mask = lambda architectures, is_embedding: False
    sys.modules.setdefault("sgl_jax.srt.configs.model_config", model_config_stub)

    expert_location_stub = types.ModuleType("sgl_jax.srt.eplb.expert_location")
    expert_location_stub.ExpertLocationMetadata = type("ExpertLocationMetadata", (), {})
    expert_location_stub.get_global_expert_location_metadata = lambda: None
    sys.modules.setdefault("sgl_jax.srt.eplb.expert_location", expert_location_stub)

    speculative_stub = types.ModuleType("sgl_jax.srt.speculative.spec_info")
    speculative_stub.SpeculativeAlgorithm = type("SpeculativeAlgorithm", (), {})
    sys.modules.setdefault("sgl_jax.srt.speculative.spec_info", speculative_stub)

    jax_utils_stub = types.ModuleType("sgl_jax.srt.utils.jax_utils")
    jax_utils_stub.device_array = lambda values, sharding=None: tuple(
        np.asarray(value) if value is not None else None for value in values
    )
    sys.modules.setdefault("sgl_jax.srt.utils.jax_utils", jax_utils_stub)


_install_import_stubs()

from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch  # noqa: E402


class TestMiMoV25ForwardBatchInputEmbedding(unittest.TestCase):
    def _new_model_runner(self):
        return SimpleNamespace(
            mesh=object(),
            attn_backend=None,
            model_config=SimpleNamespace(
                is_embedding=False,
                hf_config=SimpleNamespace(architectures=[]),
            ),
        )

    def _new_worker_batch(self, *, input_embedding):
        return SimpleNamespace(
            bid="bid",
            forward_mode=object(),
            input_ids=np.array([1, 2, 3], dtype=np.int32),
            seq_lens=np.array([3], dtype=np.int32),
            out_cache_loc=np.array([0, 1, 2], dtype=np.int32),
            positions=np.array([0, 1, 2], dtype=np.int32),
            req_pool_indices=np.array([0], dtype=np.int32),
            cache_loc=np.array([0, 1, 2], dtype=np.int32),
            extend_prefix_lens=np.array([0], dtype=np.int32),
            extend_seq_lens=np.array([3], dtype=np.int32),
            mrope_positions=None,
            lora_ids=None,
            lora_scalings=None,
            lora_token_indices=None,
            lora_ranks=None,
            spec_info_padded=None,
            spec_algorithm=None,
            capture_hidden_mode=None,
            input_embedding=input_embedding,
            apply_for_deepstack=False,
            deepstack_visual_embedding=None,
            recurrent_indices=None,
        )

    def test_init_new_preserves_and_casts_input_embedding(self):
        input_embedding = np.arange(12, dtype=np.float64).reshape(3, 4)
        batch = self._new_worker_batch(input_embedding=input_embedding)

        forward_batch = ForwardBatch.init_new(
            batch,
            self._new_model_runner(),
        )

        self.assertIsNotNone(forward_batch.input_embedding)
        self.assertEqual(forward_batch.input_embedding.dtype, np.float32)
        np.testing.assert_array_equal(
            forward_batch.input_embedding,
            input_embedding.astype(np.float32),
        )

    def test_init_new_keeps_missing_input_embedding_none(self):
        batch = self._new_worker_batch(input_embedding=None)

        forward_batch = ForwardBatch.init_new(
            batch,
            self._new_model_runner(),
        )

        self.assertIsNone(forward_batch.input_embedding)


if __name__ == "__main__":
    unittest.main()
