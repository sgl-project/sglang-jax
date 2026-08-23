"""Correctness contract for fused-RS pre-operation A/B diagnostics."""

from __future__ import annotations


DEFAULT_FINAL_REL_L2_THRESHOLD = 0.01


def evaluate_preop_variant_contract(
    *,
    hidden_gather_all_finite: bool,
    hidden_gather_max_abs: float,
    full_all_finite: bool,
    full_rel_l2: float,
    padded_all_finite: bool,
    padded_rel_l2: float,
    invalid_padding_max_abs: float,
    final_rel_l2_threshold: float = DEFAULT_FINAL_REL_L2_THRESHOLD,
) -> dict[str, bool]:
    """Separate collective semantics from downstream low-precision drift.

    A BF16 AllGather must preserve its payload exactly.  The fused W8A8 MoE
    output is allowed the same 1% relative-L2 tolerance used by its independent
    oracle checks because changing collective placement can change compiler
    layout and low-precision reduction order without changing the collective's
    logical values.
    """
    if final_rel_l2_threshold <= 0:
        raise ValueError("final_rel_l2_threshold must be positive")

    hidden_gather_exact = hidden_gather_all_finite and hidden_gather_max_abs == 0.0
    final_output_ok = (
        full_all_finite
        and padded_all_finite
        and full_rel_l2 <= final_rel_l2_threshold
        and padded_rel_l2 <= final_rel_l2_threshold
        and invalid_padding_max_abs == 0.0
    )
    return {
        "hidden_gather_exact": hidden_gather_exact,
        "final_output_ok": final_output_ok,
        "contract_ok": hidden_gather_exact and final_output_ok,
    }
