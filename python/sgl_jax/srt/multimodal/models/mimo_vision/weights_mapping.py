import glob
import os
import warnings

import jax.numpy as jnp
from flax import nnx
from safetensors import safe_open

from sgl_jax.srt.utils.weight_utils import WeightMapping

CONV3D_TORCH_TO_JAX = (2, 3, 4, 1, 0)


def to_mappings(
    config,
    source_prefix: str = "visual",
    target_prefix: str = "",
) -> dict[str, WeightMapping]:
    mappings: dict[str, WeightMapping] = {
        f"{source_prefix}.patch_embed.proj.weight": WeightMapping(
            target_path=f"{target_prefix}patch_embed.proj.kernel",
            transpose_axes=CONV3D_TORCH_TO_JAX,
        ),
        f"{source_prefix}.merger.ln_q.weight": WeightMapping(
            target_path=f"{target_prefix}merger.ln_q.scale",
            sharding=(),
        ),
        f"{source_prefix}.merger.ln_q.bias": WeightMapping(
            target_path=f"{target_prefix}merger.ln_q.bias",
            sharding=(),
        ),
        f"{source_prefix}.merger.mlp.0.weight": WeightMapping(
            target_path=f"{target_prefix}merger.mlp_fc1.kernel",
            transpose=True,
        ),
        f"{source_prefix}.merger.mlp.0.bias": WeightMapping(
            target_path=f"{target_prefix}merger.mlp_fc1.bias",
            sharding=(),
        ),
        f"{source_prefix}.merger.mlp.2.weight": WeightMapping(
            target_path=f"{target_prefix}merger.mlp_fc2.kernel",
            transpose=True,
        ),
        f"{source_prefix}.merger.mlp.2.bias": WeightMapping(
            target_path=f"{target_prefix}merger.mlp_fc2.bias",
            sharding=(),
        ),
    }

    for block_idx in range(int(config.depth)):
        source = f"{source_prefix}.blocks.{block_idx}"
        target = f"{target_prefix}blocks.{block_idx}"
        mappings.update(
            {
                f"{source}.norm1.weight": WeightMapping(
                    target_path=f"{target}.norm1.scale",
                    sharding=(),
                ),
                f"{source}.norm2.weight": WeightMapping(
                    target_path=f"{target}.norm2.scale",
                    sharding=(),
                ),
                f"{source}.attn.qkv.weight": WeightMapping(
                    target_path=f"{target}.attn.qkv.kernel",
                    transpose=True,
                ),
                f"{source}.attn.qkv.bias": WeightMapping(
                    target_path=f"{target}.attn.qkv.bias",
                    sharding=(),
                ),
                f"{source}.attn.proj.weight": WeightMapping(
                    target_path=f"{target}.attn.proj.kernel",
                    transpose=True,
                ),
                f"{source}.attn.proj.bias": WeightMapping(
                    target_path=f"{target}.attn.proj.bias",
                    sharding=(),
                ),
                f"{source}.attn.sinks": WeightMapping(
                    target_path=f"{target}.attn.sinks",
                    sharding=(),
                ),
                f"{source}.mlp.gate_proj.weight": WeightMapping(
                    target_path=f"{target}.mlp.gate_proj.kernel",
                    transpose=True,
                ),
                f"{source}.mlp.gate_proj.bias": WeightMapping(
                    target_path=f"{target}.mlp.gate_proj.bias",
                    sharding=(),
                ),
                f"{source}.mlp.up_proj.weight": WeightMapping(
                    target_path=f"{target}.mlp.up_proj.kernel",
                    transpose=True,
                ),
                f"{source}.mlp.up_proj.bias": WeightMapping(
                    target_path=f"{target}.mlp.up_proj.bias",
                    sharding=(),
                ),
                f"{source}.mlp.down_proj.weight": WeightMapping(
                    target_path=f"{target}.mlp.down_proj.kernel",
                    transpose=True,
                ),
                f"{source}.mlp.down_proj.bias": WeightMapping(
                    target_path=f"{target}.mlp.down_proj.bias",
                    sharding=(),
                ),
            }
        )
    return mappings


def load_weights_from_safetensors(model: nnx.Module, model_path: str, config) -> None:
    weight_index = _index_safetensors(model_path)
    mappings = to_mappings(config)
    zero_filled: list[str] = []
    non_bias_missing: list[str] = []
    for hf_key, mapping in mappings.items():
        if hf_key not in weight_index:
            if not _has_param(model, mapping.target_path):
                continue
            if not hf_key.endswith(".bias"):
                non_bias_missing.append(hf_key)
                continue
            _zero_param(model, mapping.target_path)
            zero_filled.append(hf_key)
            continue
        weight = _load_weight(weight_index[hf_key], hf_key)
        if mapping.transpose_axes is not None:
            weight = jnp.transpose(weight, mapping.transpose_axes)
        elif mapping.transpose:
            weight = jnp.transpose(weight, (1, 0))
        _set_param(model, mapping.target_path, weight)

    if non_bias_missing:
        raise AssertionError(
            f"Missing non-bias MiMo vision weights from checkpoint: {non_bias_missing}"
        )
    if zero_filled:
        warnings.warn(
            f"MiMo vision: zero-filled {len(zero_filled)} missing bias weights "
            f"to mirror HF behavior: {zero_filled}",
            stacklevel=2,
        )


def _index_safetensors(model_path: str) -> dict[str, str]:
    index = {}
    for filename in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(filename, framework="np", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118
                index[key] = filename
    return index


def _load_weight(filename: str, key: str) -> jnp.ndarray:
    with safe_open(filename, framework="np", device="cpu") as handle:
        return jnp.asarray(handle.get_tensor(key))


def _set_param(model: nnx.Module, target_path: str | list[str], weight: jnp.ndarray) -> None:
    target = _resolve_param(model, target_path)
    target[...] = weight.astype(target.dtype)


def _zero_param(model: nnx.Module, target_path: str | list[str]) -> None:
    target = _resolve_param(model, target_path)
    target[...] = jnp.zeros_like(target[...])


def _resolve_param(model: nnx.Module, target_path: str | list[str]):
    if not isinstance(target_path, str):
        raise TypeError(f"MiMo vision loader expects a single target path, got {target_path}")

    target = model
    for part in target_path.split("."):
        target = target[int(part)] if part.isdigit() else getattr(target, part)

    if not isinstance(target, nnx.Variable):
        raise TypeError(f"{target_path} does not point to an NNX variable")
    return target


def _has_param(model: nnx.Module, target_path: str | list[str]) -> bool:
    try:
        _resolve_param(model, target_path)
    except (AttributeError, TypeError, IndexError, KeyError):
        return False
    return True
