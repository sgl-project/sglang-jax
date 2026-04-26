""" model signature migration: outer __call__ takes memory_pools, returns dict.

Static-only tests (inspect-based); no real model load. Each migrated model
adds itself to MIGRATED_MODELS;  asserts all 13 are present.

Schema: (module_path, class_name, expects_dict_in_return)
- expects_dict_in_return=True: outer __call__ returns dict at idx 1 (the
  standard 12 cases)
- expects_dict_in_return=False: signature-only migration; return value
  shape unchanged. Used for umt5 (encoder, never returns layers_kv_fused).
"""

import inspect
import unittest

# Per-model: (module_path, class_name, expects_dict_in_return)
#  fill this list;  verifies length == 13.
# llama_eagle3.LlamaForCausalLMEagle3 inherits __call__ from LlamaForCausalLM
# (D6), so it is NOT listed here even though `grep token_to_kv_pool` hits 14
# files — that 14th hit is on the inner LlamaEagleModel.__call__ which is
# explicitly out of scope per 's "inner Model unchanged" constraint.
MIGRATED_MODELS = [
    #  (callback_flag variant)
    ("sgl_jax.srt.models.llama", "LlamaForCausalLM", True),
    ("sgl_jax.srt.models.qwen", "QWenLMHeadModel", True),
    ("sgl_jax.srt.models.qwen2", "Qwen2ForCausalLM", True),
    ("sgl_jax.srt.models.qwen3", "Qwen3ForCausalLM", True),
    #  (topk_ids variant)
    ("sgl_jax.srt.models.bailing_moe", "BailingMoEForCausalLM", True),
    ("sgl_jax.srt.models.deepseek_v3", "DeepseekV3ForCausalLM", True),
    ("sgl_jax.srt.models.gemma2", "Gemma2ForCausalLM", True),
    ("sgl_jax.srt.models.grok", "Grok1ForCausalLM", True),
    ("sgl_jax.srt.models.mimo_v2_flash", "MiMoV2FlashForCausalLM", True),
    ("sgl_jax.srt.models.qwen2_moe", "Qwen2MoeForCausalLM", True),
    ("sgl_jax.srt.models.qwen3_moe", "Qwen3MoeForCausalLM", True),
    #  (special)
    ("sgl_jax.srt.models.mimo_mtp", "MiMoMTPForCausalLM", True),
    ("sgl_jax.srt.models.umt5", "UMT5ForConditionalGeneration", False),  # signature-only
]


def _import_class(module_path, class_name):
    mod = __import__(module_path, fromlist=[class_name])
    return getattr(mod, class_name)


class TestMigratedModelSignatures(unittest.TestCase):
    """Each migrated model's outer __call__:
    1. has parameter named 'memory_pools' (not 'token_to_kv_pool')
    2. function source contains 'memory_pools.token_to_kv_pool' (the unpack line)
    3. if expects_dict_in_return: function source contains '"token_to_kv_pool":'
       (dict construction in return; suppressed for umt5-style signature-only
       migrations)
    """

    def test_signatures(self):
        for module_path, class_name, _expects in MIGRATED_MODELS:
            with self.subTest(model=f"{module_path}:{class_name}"):
                cls = _import_class(module_path, class_name)
                sig = inspect.signature(cls.__call__)
                params = list(sig.parameters.keys())
                self.assertIn(
                    "memory_pools",
                    params,
                    f"{class_name}.__call__ must take 'memory_pools'",
                )
                self.assertNotIn(
                    "token_to_kv_pool",
                    params,
                    f"{class_name}.__call__ must not take 'token_to_kv_pool' anymore",
                )

    def test_unpack_in_source(self):
        for module_path, class_name, _expects in MIGRATED_MODELS:
            with self.subTest(model=f"{module_path}:{class_name}"):
                cls = _import_class(module_path, class_name)
                src = inspect.getsource(cls.__call__)
                self.assertIn(
                    "memory_pools.token_to_kv_pool",
                    src,
                    f"{class_name}.__call__ must unpack via memory_pools.token_to_kv_pool",
                )

    def test_dict_return_in_source(self):
        """Only enforced for models that return dict (skip umt5-style signature-only)."""
        for module_path, class_name, expects_dict in MIGRATED_MODELS:
            if not expects_dict:
                continue
            with self.subTest(model=f"{module_path}:{class_name}"):
                cls = _import_class(module_path, class_name)
                src = inspect.getsource(cls.__call__)
                self.assertIn(
                    '"token_to_kv_pool":',
                    src,
                    f"{class_name}.__call__ must return dict containing 'token_to_kv_pool' key",
                )


class TestAll13ModelsMigrated(unittest.TestCase):
    """Lock-in:  must end with exactly 13 entries in MIGRATED_MODELS.

    Note: `grep token_to_kv_pool` in srt/models/ hits 14 files, but
    llama_eagle3.LlamaForCausalLMEagle3 inherits __call__ from
    LlamaForCausalLM (D6) — so it is migrated transparently by  and
    NOT counted here. The 14th hit is on the inner LlamaEagleModel.__call__
    which is out of scope per 's "inner Model unchanged" constraint.
    """

    def test_exactly_13(self):
        names = [c for _, c, *_ in MIGRATED_MODELS]
        self.assertEqual(
            len(MIGRATED_MODELS),
            13,
            f"Expected 13 migrated outer model classes, got {len(MIGRATED_MODELS)}: {names}",
        )


class TestEagle3InheritsMigratedCall(unittest.TestCase):
    """Lock-in: llama_eagle3.LlamaForCausalLMEagle3 inherits the migrated
    __call__ from LlamaForCausalLM (D6). Verify by signature-only — no
    source check (inspect.getsource would just return the parent's source)."""

    def test_eagle3_outer_call_inherits_memory_pools(self):
        from sgl_jax.srt.models.llama import LlamaForCausalLM
        from sgl_jax.srt.models.llama_eagle3 import LlamaForCausalLMEagle3

        # Same function object via MRO inheritance.
        self.assertIs(
            LlamaForCausalLMEagle3.__call__,
            LlamaForCausalLM.__call__,
            "Eagle3 must inherit __call__ from LlamaForCausalLM, not override it",
        )


if __name__ == "__main__":
    unittest.main()
