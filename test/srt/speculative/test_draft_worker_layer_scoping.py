"""CPU regressions for two #1639 review items about multi-layer draft models.

1. ``_kv_pool_layer_count`` forced every draft worker to one KV layer. That is right for
   NextN / MTP heads (one decoder layer per worker) and wrong for multi-layer drafts such
   as DFlash (Qwen3-8B-DFlash-b16: 5 layers -> ``kv_buffers[i]`` IndexError on the first
   draft prefill). The override is now scoped by architecture.
2. The fused draft-extend path collected weights and pools from ``draft_worker._worker``
   only, which ``MultiLayerDraftWorker`` aliases to ``_workers[0]``: every draft step of a
   three-layer MTP model ran layer 0. ``draft_worker_list`` returns all workers in order.
"""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
    is_single_layer_draft_arch,
)
from sgl_jax.srt.speculative.draft_extend_fused import draft_worker_list
from sgl_jax.test.test_utils import CustomTestCase


def _runner(arch, layers, draft=True):
    """The slice of ModelRunner that ``_kv_pool_layer_count`` reads."""
    return SimpleNamespace(
        is_draft_worker=draft,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch])),
        linear_recurrent_config=None,
        adjust_layer_num=lambda: layers,
    )


def _count(runner):
    return ModelRunnerKVCacheMixin._kv_pool_layer_count(runner)


class TestDraftKvPoolLayerCount(CustomTestCase):
    def test_nextn_and_mtp_heads_keep_one_layer(self):
        for arch in (
            "GlmMoeDsaForCausalLMNextN",
            "DeepseekV3ForCausalLMNextN",
            "MiMoMTPForCausalLM",
            "MiMoV2MTPForCausalLM",
            "Qwen3NextForCausalLMMTP",
        ):
            self.assertTrue(is_single_layer_draft_arch(SimpleNamespace(architectures=[arch])), arch)
            self.assertEqual(_count(_runner(arch, layers=3)), 1, arch)

    def test_multi_layer_draft_keeps_its_layer_count(self):
        # DFlash-style draft: five decoder layers, not a NextN head.
        self.assertFalse(
            is_single_layer_draft_arch(SimpleNamespace(architectures=["Qwen3ForCausalLM"]))
        )
        self.assertEqual(_count(_runner("Qwen3ForCausalLM", layers=5)), 5)
        self.assertEqual(_count(_runner("LlamaForCausalLMEagle3", layers=1)), 1)

    def test_target_worker_is_untouched(self):
        self.assertEqual(_count(_runner("GlmMoeDsaForCausalLM", layers=78, draft=False)), 78)

    def test_missing_architectures_is_not_single_layer(self):
        self.assertFalse(is_single_layer_draft_arch(SimpleNamespace()))
        self.assertFalse(is_single_layer_draft_arch(SimpleNamespace(architectures=[])))


class TestDraftWorkerList(CustomTestCase):
    def test_multi_layer_worker_returns_every_layer_in_order(self):
        w = [SimpleNamespace(layer=i) for i in range(3)]
        dw = SimpleNamespace(_workers=w, _worker=w[0])
        self.assertEqual([x.layer for x in draft_worker_list(dw)], [0, 1, 2])

    def test_single_layer_worker_returns_its_only_worker(self):
        only = SimpleNamespace(layer=0)
        self.assertEqual(draft_worker_list(SimpleNamespace(_worker=only)), [only])
        # an empty _workers list (never built) also falls back to _worker
        self.assertEqual(draft_worker_list(SimpleNamespace(_workers=[], _worker=only)), [only])


if __name__ == "__main__":
    unittest.main()
