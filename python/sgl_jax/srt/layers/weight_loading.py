"""Loading groups for layer-owned layouts, independent of checkpoint storage."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.model_loader.weights import WeightSpec


def _put(value, mesh, axes):
    return jax.make_array_from_callback(
        value.shape, NamedSharding(mesh, P(*axes)), lambda index: value[index]
    )


def _absorb_mla(inputs, *, mesh, rank, heads, key_dim, value_dim, dtype, block):
    weight = _put(inputs[0], mesh, ())
    if len(inputs) == 2:
        scale = _put(inputs[1], mesh, ()).astype(jnp.float32)
        if block is not None:
            # Checkpoint axes are [out, in]. Preserve FP32 multiplication and
            # the BF16 rounding boundary of the former projection post-load.
            scale = jnp.repeat(jnp.repeat(scale, block[0], axis=0), block[1], axis=1)
            scale = scale[: weight.shape[0], : weight.shape[1]]
        else:
            scale = scale.reshape(-1, 1)
        weight = (weight.astype(jnp.float32) * scale).astype(jnp.bfloat16)
    else:
        weight = weight.astype(dtype)
    weight = weight.T.reshape(rank, heads, key_dim + value_dim)
    sharding = NamedSharding(mesh, P(None, "tensor", None))
    return tuple(
        jax.sharding.reshard(part, sharding)
        for part in (weight[:, :, :key_dim], weight[:, :, key_dim:])
    )


def prepare_absorbed_mla(module, loader, mappings, prefix):
    if not module.use_absorbed or module.kv_b_proj is None:
        return mappings
    target = prefix + ".kv_b_proj."
    selected = {
        spec.target_path: source
        for source, spec in mappings.items()
        if isinstance(spec.target_path, str) and spec.target_path.startswith(target)
    }
    weight = selected.get(target + "weight", selected.get(target + "weight_q"))
    if weight is None:
        return mappings  # Another loader invocation owns this module (e.g. vision).
    sources = [weight]
    scale = selected.get(target + "weight_scale")
    if scale is not None:
        sources.append(scale)
    if len(selected) != len(sources):
        raise ValueError(f"Unsupported absorbed MLA checkpoint inputs: {tuple(selected)}")
    mappings = {key: spec for key, spec in mappings.items() if key not in sources}
    mappings[weight] = WeightSpec(
        [prefix + ".w_uk", prefix + ".w_uv"],
        sources=tuple(sources),
        recipe=partial(
            _absorb_mla,
            mesh=module.mesh,
            rank=module.kv_lora_rank,
            heads=module.num_heads,
            key_dim=module.qk_nope_head_dim,
            value_dim=module.v_head_dim,
            dtype=module.w_uk.value.dtype,
            block=getattr(
                getattr(loader.model_config, "quantization_config", None),
                "weight_block_size",
                None,
            ),
        ),
    )
    module.kv_b_proj = None
    return mappings
