"""DFlash speculative decoding utilities.

Minimal-runnable scope (Qwen3 target, greedy, no overlap/DP/tree).

DFlash draft config field names follow the PyTorch sglang DFlash convention:
``dflash_config.{block_size, target_layer_ids, mask_token, mask_token_id}``.
The exact keys of a given draft checkpoint may differ — see
:func:`parse_dflash_draft_config`, which is the single place to adjust once the
real checkpoint's ``config.json`` is known.

Terminology (kept identical to the PyTorch implementation):
- ``block_size`` == number of tokens in one verify block == ``speculative_num_draft_tokens``.
  candidates = [verified_id (seed), d_1, ..., d_{block_size-1}].
- ``K`` == ``len(target_layer_ids)`` == number of captured target layers whose
  hidden states are concatenated and projected by the draft's ``fc`` (fc in-dim
  = ``K * target_hidden_size``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"
DEFAULT_DFLASH_BLOCK_SIZE = 16


@dataclass(frozen=True)
class DFlashDraftConfig:
    """Parsed DFlash draft-model configuration.

    All fields are resolved from the draft model's HF config. ``target_hidden_size``
    defaults to the draft ``hidden_size`` when the checkpoint does not record the
    target's hidden size separately (common for same-family Qwen3 draft/target).
    """

    num_hidden_layers: int
    hidden_size: int
    target_hidden_size: int
    block_size: int
    target_layer_ids: Optional[List[int]]
    num_injection_layers: int
    mask_token: str
    mask_token_id: Optional[int]


def _get_dflash_subconfig(hf_config) -> dict:
    """Return the ``dflash_config`` sub-dict from an HF config, or {}.

    Supports both an attribute (`hf_config.dflash_config`) and a nested dict.
    """
    cfg = getattr(hf_config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    # transformers PretrainedConfig-like nested object
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    try:
        return dict(cfg.__dict__)
    except Exception:
        return {}


def _cfg_get(dflash: dict, hf_config, key: str, default=None):
    """Prefer dflash_config[key], fall back to a top-level hf_config attribute."""
    if key in dflash and dflash[key] is not None:
        return dflash[key]
    return getattr(hf_config, key, default)


def parse_dflash_draft_config(
    model_path: str,
    revision: Optional[str] = None,
    trust_remote_code: bool = True,
) -> DFlashDraftConfig:
    """Load and parse a DFlash draft model's HF config.

    NOTE: field names follow the PyTorch DFlash convention. When wiring a real
    checkpoint, verify the actual keys via::

        from transformers import AutoConfig
        print(AutoConfig.from_pretrained(<draft_path>))

    and adjust the ``_cfg_get`` keys below if they differ.
    """
    from sgl_jax.srt.hf_transformers_utils import get_config

    hf_config = get_config(
        model_path, trust_remote_code=trust_remote_code, revision=revision
    )
    dflash = _get_dflash_subconfig(hf_config)

    num_hidden_layers = int(getattr(hf_config, "num_hidden_layers", 1))
    hidden_size = int(getattr(hf_config, "hidden_size"))
    target_hidden_size = int(
        _cfg_get(dflash, hf_config, "target_hidden_size", hidden_size)
    )

    block_size = _cfg_get(dflash, hf_config, "block_size", None)
    if block_size is None:
        logger.warning(
            "DFLASH: draft config has no block_size; defaulting to %d.",
            DEFAULT_DFLASH_BLOCK_SIZE,
        )
        block_size = DEFAULT_DFLASH_BLOCK_SIZE
    block_size = int(block_size)
    if block_size <= 1:
        raise ValueError(f"DFLASH block_size must be > 1, got {block_size}.")

    raw_layer_ids = _cfg_get(dflash, hf_config, "target_layer_ids", None)
    if raw_layer_ids is not None:
        target_layer_ids = [int(x) for x in raw_layer_ids]
        if not target_layer_ids:
            raise ValueError("DFLASH target_layer_ids must be non-empty when set.")
    else:
        # Fall back to num_target_layers if provided; otherwise leave None so the
        # caller uses set_eagle3_layers_to_capture's default. K must ultimately
        # match the draft fc in-dim, so a real checkpoint should always specify this.
        target_layer_ids = None
        logger.warning(
            "DFLASH: draft config has no target_layer_ids; capture layers will "
            "fall back to defaults, which may mismatch the draft fc in-dim."
        )

    num_injection_layers = (
        len(target_layer_ids)
        if target_layer_ids is not None
        else int(_cfg_get(dflash, hf_config, "num_target_layers", num_hidden_layers))
    )

    mask_token = _cfg_get(dflash, hf_config, "mask_token", DEFAULT_DFLASH_MASK_TOKEN)
    mask_token_id = _cfg_get(dflash, hf_config, "mask_token_id", None)
    if mask_token_id is not None:
        mask_token_id = int(mask_token_id)

    return DFlashDraftConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        target_hidden_size=target_hidden_size,
        block_size=block_size,
        target_layer_ids=target_layer_ids,
        num_injection_layers=int(num_injection_layers),
        mask_token=str(mask_token),
        mask_token_id=mask_token_id,
    )


def resolve_mask_token_id(
    dflash_config: DFlashDraftConfig,
    tokenizer,
    vocab_size: int,
) -> int:
    """Resolve the mask token id used to fill draft block positions.

    Priority: explicit dflash_config.mask_token_id -> tokenizer vocab lookup of
    mask_token. The id must be within the target vocab (DFlash reuses the target
    embedding; no vocab resizing is supported in minimal mode).
    """
    resolved = dflash_config.mask_token_id
    if resolved is None:
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires either dflash_config.mask_token_id or an "
                "initialized tokenizer to resolve mask_token "
                f"{dflash_config.mask_token!r}."
            )
        vocab = tokenizer.get_vocab()
        resolved = vocab.get(dflash_config.mask_token, None)
        if resolved is None:
            raise ValueError(
                f"DFLASH mask_token {dflash_config.mask_token!r} not found in "
                "tokenizer vocab and no mask_token_id given."
            )
    resolved = int(resolved)
    if resolved < 0 or resolved >= int(vocab_size):
        raise ValueError(
            f"DFLASH mask_token_id={resolved} outside target vocab_size={vocab_size}. "
            "Vocab resizing is not supported in minimal mode."
        )
    return resolved
