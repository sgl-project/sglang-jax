"""CPU-side shape tests for the MLA tuner workload presets."""

import numpy as np
import pytest

from benchmark.kernels.mla.get_block_spec_config_mla import (
    _SCENARIOS,
    _candidate_failure_label,
    _decode_trace_scope,
    _enum_decode_candidates,
)
from benchmark.kernels.mla.utils import create_mla_mixed_uniform_data


def test_glm52_dp16_scenario_uses_local_shapes():
    scenario = _SCENARIOS["glm52-dp16-128k"]

    assert scenario["num_q_heads"] == (64,)
    assert scenario["page_sizes"] == (64,)
    assert scenario["decode_mnt"] == (2,)
    assert scenario["mixed_mnt"] == (2048,)
    assert scenario["mixed_num_seqs"] == 2
    assert scenario["mixed_kv_len"] == 132096
    assert scenario["decode_kv_len"] == 133120


def test_decode_scope_uses_tail_when_fallback_batch_exceeds_mnt():
    assert (
        _decode_trace_scope(
            max_num_tokens=2,
            bkv_p=3,
            page_size=64,
            decode_batch_size=4,
        )
        == "MLA-d-bq_1-bkvp_3-p_64-bsz_1"
    )


def test_decode_scope_uses_batched_kernel_for_glm_candidate():
    assert (
        _decode_trace_scope(
            max_num_tokens=2,
            bkv_p=8,
            page_size=64,
            decode_batch_size=2,
        )
        == "MLA-bd-bq_1-bkvp_8-p_64-bsz_2"
    )
    assert {candidate[2] for candidate in _enum_decode_candidates(2)} == {1, 2}


def test_glm52_mixed_inputs_model_two_local_requests():
    inputs = create_mla_mixed_uniform_data(
        max_num_tokens=8,
        num_q_heads=2,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        page_size=4,
        kv_len=12,
        num_seqs=2,
    )

    np.testing.assert_array_equal(np.asarray(inputs["cu_q_lens"]), [0, 4, 8])
    np.testing.assert_array_equal(np.asarray(inputs["cu_kv_lens"]), [0, 12, 24])
    np.testing.assert_array_equal(np.asarray(inputs["distribution"]), [0, 0, 2])
    assert inputs["kv_lens"].shape == (2,)


def test_mixed_inputs_require_uniform_query_lengths():
    with pytest.raises(ValueError, match="must be divisible"):
        create_mla_mixed_uniform_data(
            max_num_tokens=7,
            num_q_heads=2,
            kv_lora_rank=8,
            qk_rope_head_dim=4,
            page_size=4,
            kv_len=12,
            num_seqs=2,
        )


def test_expected_vmem_rejection_is_reported_as_skip():
    error = RuntimeError(
        "RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem.\n"
        "very large compiler allocation report"
    )
    assert _candidate_failure_label(error) == "SKIP_VMEM"
