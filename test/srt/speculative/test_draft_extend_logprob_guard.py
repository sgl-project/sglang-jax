"""prepare_for_extend_after_verify must not route a speculative decode batch (return_logprob set,
per-request logprob lists None) through LogitsMetadata.from_model_worker_batch: gp224/gp225 crashed
with "'NoneType' object is not iterable" at logits_processor.from_model_worker_batch on the first
MTP + return_logprob request."""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from sgl_jax.srt.speculative.eagle_info import batch_has_extend_logprob_lists


class DraftExtendLogprobGuardTest(unittest.TestCase):
    def test_spec_decode_batch_with_none_lists_is_rejected(self):
        b = SimpleNamespace(
            return_logprob=True,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
        )
        self.assertFalse(batch_has_extend_logprob_lists(b))

    def test_no_logprob_batch_is_rejected(self):
        b = SimpleNamespace(
            return_logprob=False,
            top_logprobs_nums=[0, 0],
            token_ids_logprobs=[None, None],
            extend_logprob_start_lens=[0, 0],
        )
        self.assertFalse(batch_has_extend_logprob_lists(b))

    def test_extend_batch_with_lists_is_accepted(self):
        b = SimpleNamespace(
            return_logprob=True,
            top_logprobs_nums=[5, 0],
            token_ids_logprobs=[None, None],
            extend_logprob_start_lens=[0, 3],
        )
        self.assertTrue(batch_has_extend_logprob_lists(b))

    def test_spec_decode_batch_with_top_lists_but_no_start_lens_is_rejected(self):
        # _get_spec_decode_mwb_dp now carries top_logprobs_nums / token_ids_logprobs
        # (for verify-row logprobs); without extend_logprob_start_lens it is still
        # a decode batch and must not build extend-logprob metadata.
        b = SimpleNamespace(
            return_logprob=True,
            top_logprobs_nums=[5, 0],
            token_ids_logprobs=[None, [1, 2]],
            extend_logprob_start_lens=None,
        )
        self.assertFalse(batch_has_extend_logprob_lists(b))

    def test_partial_lists_are_rejected(self):
        b = SimpleNamespace(
            return_logprob=True,
            top_logprobs_nums=[5],
            token_ids_logprobs=[None],
            extend_logprob_start_lens=None,
        )
        self.assertFalse(batch_has_extend_logprob_lists(b))


if __name__ == "__main__":
    unittest.main()
