from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sgl_jax.srt.speculative.draft_extend_fused import (
    _prepare_eagle_overlap_verify,
    _prepare_mtp_overlap_verify,
    _spec_decode_fused_chain_overlap,
)


def test_fused_chain_overlap_uses_common_relay_envelope():
    published_new_seq_lens = Mock()
    next_draft_input = SimpleNamespace(
        future_indices=None,
        new_seq_lens=np.array([7, 8, 9], dtype=np.int32),
    )
    batch_output = SimpleNamespace(
        next_draft_input=next_draft_input,
        published_new_seq_lens=published_new_seq_lens,
    )
    draft_worker = object()
    spec_worker = SimpleNamespace(
        draft_worker=draft_worker,
        spec_relay_buffers="old-buffers",
    )
    model_worker_batch = SimpleNamespace(
        req_pool_indices=np.array([4, 5, 0, 8, 0, 0], dtype=np.int32),
        logits_indices_selector=np.array([0, 1, 3], dtype=np.int32),
        real_bs_per_dp=[2, 1],
        per_dp_bs_size=3,
    )
    prepare_verify = Mock(return_value=("token-map", True))
    launch_draft = Mock(return_value=SimpleNamespace(updated_relay_buffers="new-buffers"))

    with patch(
        "sgl_jax.srt.speculative.draft_extend_fused.spec_decode_verify",
        return_value=batch_output,
    ) as verify:
        result, published = _spec_decode_fused_chain_overlap(
            spec_worker,
            model_worker_batch,
            np.array([16, 24, 32], dtype=np.int32),
            prepare_verify=prepare_verify,
            launch_draft=launch_draft,
        )

    assert result is batch_output
    assert published is published_new_seq_lens
    published_new_seq_lens.copy_to_host_async.assert_called_once_with()
    prepare_verify.assert_called_once_with(draft_worker, model_worker_batch)
    verify.assert_called_once()
    assert verify.call_args.kwargs["draft_to_target_token_ids"] == "token-map"
    assert verify.call_args.kwargs["draft_padding_prepared"] is True
    launch_draft.assert_called_once()
    np.testing.assert_array_equal(
        launch_draft.call_args.kwargs["relay_future_indices"],
        np.array([4, 5, 0, 8, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        launch_draft.call_args.kwargs["relay_valid_mask"],
        np.array([True, True, False, True, False, False]),
    )
    np.testing.assert_array_equal(
        next_draft_input.future_indices,
        np.array([4, 5, 8], dtype=np.int32),
    )
    assert next_draft_input.new_seq_lens is None
    assert spec_worker.spec_relay_buffers == "new-buffers"


def test_overlap_verify_strategies_keep_algorithm_specific_bootstrap():
    eagle_worker = SimpleNamespace(
        hot_token_ids="hot-token-map",
        prepare_for_fused_verify=Mock(return_value="bootstrap-token-map"),
    )
    relay_batch = SimpleNamespace(
        spec_info_padded=SimpleNamespace(
            future_indices=np.array([3], dtype=np.int32),
            topk_index=None,
        )
    )
    bootstrap_batch = SimpleNamespace(
        spec_info_padded=SimpleNamespace(
            future_indices=None,
            topk_index=np.array([[11]], dtype=np.int32),
        )
    )

    assert _prepare_eagle_overlap_verify(eagle_worker, relay_batch) == (
        "hot-token-map",
        False,
    )
    assert _prepare_eagle_overlap_verify(eagle_worker, bootstrap_batch) == (
        "bootstrap-token-map",
        True,
    )

    mtp_worker = SimpleNamespace(prepare_for_fused_verify=Mock())
    assert _prepare_mtp_overlap_verify(mtp_worker, relay_batch) == (None, False)
    assert _prepare_mtp_overlap_verify(mtp_worker, bootstrap_batch) == (None, True)
    mtp_worker.prepare_for_fused_verify.assert_called_once_with(bootstrap_batch)
