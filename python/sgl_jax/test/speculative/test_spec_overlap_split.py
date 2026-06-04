import numpy as np

from sgl_jax.srt.managers.scheduler import SpecVerifyPhaseResult
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput


def test_spec_verify_phase_result_keeps_dp_padded_accept_layout():
    per_dp_bs = 4
    dp_size = 2
    stride = 4
    total_bs = per_dp_bs * dp_size
    accept_lens = np.array([4, 2, 0, 0, 3, 1, 0, 0], dtype=np.int32)
    next_token_ids = np.arange(total_bs * stride, dtype=np.int32)
    draft = EagleDraftInput(
        verified_id=np.arange(total_bs, dtype=np.int32),
        new_seq_lens=np.arange(total_bs, dtype=np.int32) + 100,
        allocate_lens=np.arange(total_bs, dtype=np.int32) + 128,
        hidden_states=np.zeros((total_bs, 8), dtype=np.float32),
    )

    result = SpecVerifyPhaseResult(
        logits_output=None,
        next_token_ids=next_token_ids,
        accept_lens=accept_lens,
        allocate_lens=draft.allocate_lens,
        scheduler_next_draft_input=draft,
        draft_extend_state={"stride": stride},
        bid=7,
        cache_miss_count=0,
    )

    assert result.accept_lens.shape == (total_bs,)
    assert result.next_token_ids.shape == (total_bs * stride,)
    assert result.scheduler_next_draft_input.new_seq_lens.shape == (total_bs,)


def test_split_phase_entrypoints_import():
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
    from sgl_jax.srt.speculative.draft_extend_fused import spec_decode_verify_phase

    assert callable(spec_decode_verify_phase)
    assert hasattr(BaseSpecWorker, "forward_batch_speculative_verify_phase")
