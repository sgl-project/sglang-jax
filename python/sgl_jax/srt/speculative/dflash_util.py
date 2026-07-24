from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"
DEFAULT_DFLASH_BLOCK_SIZE = 16


@dataclass(frozen=True)
class DFlashDraftConfig:
    """DFlash settings consumed outside the draft model implementation."""

    block_size: int
    target_layer_ids: list[int] | None
    mask_token: str
    mask_token_id: int | None


def _get_dflash_subconfig(hf_config) -> dict:
    """Return ``dflash_config`` as a dict."""
    cfg = getattr(hf_config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
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
    revision: str | None = None,
    trust_remote_code: bool = True,
) -> DFlashDraftConfig:
    """Load a DFlash draft model's HF config."""
    from sgl_jax.srt.hf_transformers_utils import get_config

    hf_config = get_config(model_path, trust_remote_code=trust_remote_code, revision=revision)
    dflash = _get_dflash_subconfig(hf_config)

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
        target_layer_ids = None
        logger.warning(
            "DFLASH: draft config has no target_layer_ids; capture layers will "
            "fall back to defaults, which may mismatch the draft fc in-dim."
        )

    mask_token = _cfg_get(dflash, hf_config, "mask_token", DEFAULT_DFLASH_MASK_TOKEN)
    mask_token_id = _cfg_get(dflash, hf_config, "mask_token_id", None)
    if mask_token_id is not None:
        mask_token_id = int(mask_token_id)

    return DFlashDraftConfig(
        block_size=block_size,
        target_layer_ids=target_layer_ids,
        mask_token=str(mask_token),
        mask_token_id=mask_token_id,
    )


def resolve_mask_token_id(
    dflash_config: DFlashDraftConfig,
    tokenizer,
    vocab_size: int,
) -> int:
    """Resolve and validate the draft mask token id."""
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
