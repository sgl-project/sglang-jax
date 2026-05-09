import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

TOKENIZER_MODEL = "Qwen/Qwen3-0.6B"
TEST_TEXTS = [
    "Hello, world!",
    "SGLang JAX tokenizer parity.\n" * 2048,
]

try:
    import fastokens  # noqa: F401

    HAS_FASTOKENS = True
except ImportError:
    HAS_FASTOKENS = False


@unittest.skipUnless(HAS_FASTOKENS, "fastokens package not installed")
class TestFastokensBackend(unittest.TestCase):
    def test_01_fastokens_matches_huggingface_token_ids(self):
        from fastokens._compat import _TokenizerShim

        from sgl_jax.srt.hf_transformers_utils import get_tokenizer

        # Load the Hugging Face reference before enabling fastokens, because
        # fastokens.patch_transformers() is process-wide.
        hf_tokenizer = get_tokenizer(TOKENIZER_MODEL)
        hf_token_ids = [hf_tokenizer.encode(text, add_special_tokens=False) for text in TEST_TEXTS]

        fastokens_tokenizer = get_tokenizer(
            TOKENIZER_MODEL,
            tokenizer_backend="fastokens",
        )
        fastokens_backend = getattr(fastokens_tokenizer, "_tokenizer", None)
        self.assertIsInstance(fastokens_backend, _TokenizerShim)

        fastokens_token_ids = [
            fastokens_tokenizer.encode(text, add_special_tokens=False) for text in TEST_TEXTS
        ]

        self.assertEqual(fastokens_token_ids, hf_token_ids)
        self.assertGreater(len(fastokens_token_ids[-1]), 1000)
        self.assertEqual(
            fastokens_tokenizer.decode(fastokens_token_ids[0], skip_special_tokens=True),
            TEST_TEXTS[0],
        )

    def test_02_fastokens_load_error_is_actionable(self):
        from sgl_jax.srt import hf_transformers_utils

        hf_transformers_utils._ensure_fastokens_patched()

        with (
            TemporaryDirectory() as tokenizer_dir,
            patch.object(
                hf_transformers_utils.AutoTokenizer,
                "from_pretrained",
                side_effect=RuntimeError("unsupported tokenizer"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "Use tokenizer_backend='huggingface' to use the default backend",
            ),
        ):
            hf_transformers_utils.get_tokenizer(
                tokenizer_dir,
                tokenizer_backend="fastokens",
            )


class TestTokenizerBackendValidation(unittest.TestCase):
    def test_invalid_tokenizer_backend_is_rejected(self):
        from sgl_jax.srt.hf_transformers_utils import get_tokenizer

        with self.assertRaisesRegex(ValueError, "Unsupported tokenizer_backend"):
            get_tokenizer(TOKENIZER_MODEL, tokenizer_backend="not-a-backend")


if __name__ == "__main__":
    unittest.main()
