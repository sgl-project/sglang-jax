"""Correctness contract for the fused-RS FP8 Hidden AllGather path."""

DEFAULT_REL_L2_THRESHOLD = 0.01


def evaluate_fp8_hidden_ag_contract(
    *,
    full_all_finite: bool,
    full_rel_l2: float,
    padded_all_finite: bool,
    padded_rel_l2: float,
    padding_invariance_rel_l2: float,
    invalid_padding_max_abs: float,
    rel_l2_threshold: float = DEFAULT_REL_L2_THRESHOLD,
) -> dict[str, bool | float]:
    if rel_l2_threshold <= 0:
        raise ValueError("rel_l2_threshold must be positive")
    output_ok = bool(
        full_all_finite
        and padded_all_finite
        and full_rel_l2 <= rel_l2_threshold
        and padded_rel_l2 <= rel_l2_threshold
        and padding_invariance_rel_l2 <= rel_l2_threshold
        and invalid_padding_max_abs == 0.0
    )
    return {
        "contract_ok": output_ok,
        "full_output_ok": bool(full_all_finite and full_rel_l2 <= rel_l2_threshold),
        "padded_output_ok": bool(padded_all_finite and padded_rel_l2 <= rel_l2_threshold),
        "padding_invariance_ok": bool(
            padding_invariance_rel_l2 <= rel_l2_threshold
            and invalid_padding_max_abs == 0.0
        ),
        "rel_l2_threshold": float(rel_l2_threshold),
    }
