"""Cross-model MTP KV-cache sharing: maps Gemma4 MTP draft layers to
their target (verifier) model KV-cache layers.

Gemma4 MTP draft attention is Q-only — K/V come from the target model's
KV cache. Which target layer each draft layer reads from is determined by
matching ``layer_types`` (full vs sliding attention) between the two
HF configs.
"""

from __future__ import annotations

from collections import defaultdict


def compute_mtp_kv_share_map(draft_config, target_config) -> dict[str, str]:
    """Return ``{'draft_layer.i': 'layer.j'}`` mapping speculative MTP layers
    to their cross-model KV sharing target verifier layer cache array indices.

    Each draft layer's Q-only attention reads K/V from a specific target
    verifier layer of the same attention type (full or sliding). This
    mapping is consumed at runtime to redirect draft attention to the
    corresponding target KV cache slot.

    Target layers that are themselves KV-shared (the last
    ``num_kv_shared_layers``) are excluded — only the non-shared source
    layer is a valid cache owner.
    """
    draft_text_config = getattr(draft_config, "text_config", draft_config)
    target_text_config = getattr(target_config, "text_config", target_config)

    target_layer_types = getattr(target_text_config, "layer_types", [])
    target_num_kv_shared = getattr(target_text_config, "num_kv_shared_layers", 0)
    num_non_shared = len(target_layer_types) - target_num_kv_shared

    # Group verifier non-shared layers by attention type
    type_to_target_indices = defaultdict(list)
    for idx, lt in enumerate(target_layer_types[:num_non_shared]):
        type_to_target_indices[lt].append(idx)

    draft_layer_types = getattr(draft_text_config, "layer_types", [])
    draft_num_layers = getattr(draft_text_config, "num_hidden_layers", 4)

    redirects = {}
    for draft_idx in range(draft_num_layers):
        draft_layer_type = (
            draft_layer_types[draft_idx] if draft_idx < len(draft_layer_types) else "full_attention"
        )
        candidates = type_to_target_indices.get(draft_layer_type, [])
        if candidates:
            target_idx = candidates[-1]
            redirects[f"draft_layer.{draft_idx}"] = f"layer.{target_idx}"
        else:
            raise ValueError(
                f"MTP cross-model KV-share: draft layer {draft_idx} of type "
                f"{draft_layer_type!r} has no matching same-type verifier layer "
                f"in target configuration (candidates are empty)."
            )
    return redirects
