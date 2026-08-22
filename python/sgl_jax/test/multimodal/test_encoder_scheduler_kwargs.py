import unittest
from unittest import mock

from sgl_jax.srt.multimodal.manager.scheduler.encoder_scheduler import EncoderScheduler

WORKER = "sgl_jax.srt.multimodal.manager.scheduler.encoder_scheduler.EncoderModelWorker"


class _FakeServerArgs:
    disable_precompile = True


class TestEncoderSchedulerConstructorContract(unittest.TestCase):
    """EncoderScheduler.__init__ contract after removing **kwargs (issue #1326).

    The failure cases raise during argument binding / at the top of __init__,
    before the (heavy) EncoderModelWorker is built, so they run on CPU without
    devices or model weights. The accepted cases mock the worker and profiler
    init to keep construction on CPU.
    """

    def test_tokenizer_typo_fails_at_construction(self):
        # `tokenizer` (singular) is the classic YAML typo for `tokenizers`; it must
        # now fail instead of being silently swallowed and defaulted.
        with self.assertRaises(TypeError):
            EncoderScheduler(None, None, None, tokenizer="/tok")

    def test_unknown_scheduler_param_fails_at_construction(self):
        with self.assertRaises(TypeError):
            EncoderScheduler(None, None, None, some_unsupported_param=1)

    def test_non_empty_precompile_params_is_rejected(self):
        with self.assertRaises(ValueError):
            EncoderScheduler(None, None, None, precompile_params={"mode": "eager"})

    @mock.patch.object(EncoderScheduler, "init_profier")
    @mock.patch(WORKER)
    def test_none_precompile_params_accepted_and_tokenizers_forwarded(
        self, mock_worker, _mock_init_profier
    ):
        # None is the uniform forward from stage.py and must be accepted; the
        # tokenizers value must reach the worker.
        scheduler = EncoderScheduler(
            _FakeServerArgs(),
            mesh=None,
            communication_backend="backend",
            model_class="EncoderModel",
            stage_sub_dir="sub",
            tokenizers="/path/to/tok",
            precompile_params=None,
        )
        self.assertIs(scheduler.encoder_worker, mock_worker.return_value)
        self.assertEqual(mock_worker.call_args.kwargs["tokenizer"], "/path/to/tok")

    @mock.patch.object(EncoderScheduler, "init_profier")
    @mock.patch(WORKER)
    def test_empty_precompile_params_is_accepted(self, _mock_worker, _mock_init_profier):
        # Empty means "nothing configured" -> must not raise.
        EncoderScheduler(_FakeServerArgs(), None, "backend", precompile_params={})


if __name__ == "__main__":
    unittest.main()
