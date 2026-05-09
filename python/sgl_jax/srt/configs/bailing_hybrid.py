from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BailingHybridLinearConfig:
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    linear_layer_ids: list[int]
    full_attention_layer_ids: list[int]

    @property
    def linear_attn_config(self) -> dict[str, Any]:
        from sgl_jax.srt.mem_cache.recurrent_state_pool import recurrent_state_dtype

        return {
            "kda_layers": self.linear_layer_ids,
            "num_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "short_conv_kernel_size": 1,
            "dtype": recurrent_state_dtype(),
        }


def get_bailing_hybrid_config(hf_config: Any) -> BailingHybridLinearConfig | None:
    if not _is_bailing_hybrid_config(hf_config):
        return None

    num_hidden_layers = int(hf_config.num_hidden_layers)
    num_attention_heads = int(hf_config.num_attention_heads)
    head_dim = int(hf_config.head_dim)
    linear_layer_ids, full_attention_layer_ids = _get_layer_ids(hf_config, num_hidden_layers)
    if not linear_layer_ids:
        return None

    return BailingHybridLinearConfig(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        linear_layer_ids=linear_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
    )


def _is_bailing_hybrid_config(hf_config: Any) -> bool:
    if getattr(hf_config, "model_type", None) == "bailing_hybrid":
        return True
    architectures = getattr(hf_config, "architectures", None) or []
    return any(str(arch).startswith("BailingMoeV2_") for arch in architectures)


def _get_layer_ids(hf_config: Any, num_hidden_layers: int) -> tuple[list[int], list[int]]:
    layers_block_type = getattr(hf_config, "layers_block_type", None)
    if layers_block_type is not None:
        if len(layers_block_type) != num_hidden_layers:
            raise ValueError(
                f"layers_block_type length ({len(layers_block_type)}) must match "
                f"num_hidden_layers ({num_hidden_layers})"
            )
        linear_layer_ids = []
        full_attention_layer_ids = []
        for layer_id, block_type in enumerate(layers_block_type):
            normalized = str(block_type).lower()
            if normalized == "linear_attention":
                linear_layer_ids.append(layer_id)
            elif normalized in {"attention", "full_attention"}:
                full_attention_layer_ids.append(layer_id)
            else:
                raise ValueError(f"Unsupported Bailing hybrid layer block type: {block_type}")
        return linear_layer_ids, full_attention_layer_ids

    layer_group_size = int(hf_config.layer_group_size)
    if layer_group_size <= 0:
        raise ValueError(f"layer_group_size must be positive, got {layer_group_size}")

    linear_layer_ids = []
    full_attention_layer_ids = []
    for layer_id in range(num_hidden_layers):
        if (layer_id + 1) % layer_group_size == 0:
            full_attention_layer_ids.append(layer_id)
        else:
            linear_layer_ids.append(layer_id)
    return linear_layer_ids, full_attention_layer_ids


__all__ = ["BailingHybridLinearConfig", "get_bailing_hybrid_config"]
