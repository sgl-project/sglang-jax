"""MiMo checkpoint recipes. Raw quantized inputs live for one projection group."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
from sgl_jax.srt.model_loader.weights import WeightSpec


def _put(value, mesh, axes):
    value = np.ascontiguousarray(value, dtype=ml_dtypes.bfloat16)
    return jax.make_array_from_callback(
        value.shape, NamedSharding(mesh, P(*axes)), lambda index: value[index]
    )


def _replicate(value, head_dim, heads):
    original = value.shape[1] // head_dim
    if original == heads:
        return value
    if original <= 0 or heads % original:
        raise ValueError(f"Cannot replicate {original} KV heads to {heads}")
    return np.repeat(
        value.reshape(value.shape[0], original, head_dim), heads // original, axis=1
    ).reshape(value.shape[0], heads * head_dim)


def _block_dequant(weight, scale, block_size):
    rows, cols = weight.shape
    padded = scale.shape[0] * block_size
    weight = np.pad(weight, ((0, padded - rows), (0, 0)))
    return (
        (
            weight.astype(np.float32).reshape(
                scale.shape[0], block_size, scale.shape[1], block_size
            )
            * scale[:, None, :, None]
        )
        .reshape(padded, cols)[:rows]
        .astype(ml_dtypes.bfloat16)
    )


def load_fused_kv(inputs, *, head_dim, v_head_dim, block_size, kv_heads, mesh):
    k, ks, v, vs = inputs
    heads, cols = k.shape[0] // head_dim, k.shape[1]
    k_blocks = math.ceil(head_dim / block_size)
    per_head = ks.shape[0] == heads * k_blocks and heads * k_blocks != math.ceil(
        k.shape[0] / block_size
    )
    if per_head:
        blocks = math.ceil((head_dim + v_head_dim) / block_size)
        fused = np.concatenate(
            (k.reshape(heads, head_dim, cols), v.reshape(heads, v_head_dim, cols)),
            axis=1,
        )
        scales = np.concatenate(
            (ks.reshape(heads, k_blocks, -1), vs.reshape(heads, blocks - k_blocks, -1)),
            axis=1,
        )
        fused = np.pad(fused, ((0, 0), (0, blocks * block_size - head_dim - v_head_dim), (0, 0)))
        full = (
            (
                fused.astype(np.float32).reshape(
                    heads, blocks, block_size, cols // block_size, block_size
                )
                * scales[:, :, None, :, None]
            )
            .reshape(heads, blocks * block_size, cols)
            .astype(ml_dtypes.bfloat16)
        )
        k = full[:, :head_dim].reshape(heads * head_dim, cols)
        v = full[:, head_dim : head_dim + v_head_dim].reshape(heads * v_head_dim, cols)
    else:
        k, v = _block_dequant(k, ks, block_size), _block_dequant(v, vs, block_size)
    return tuple(
        _put(_replicate(w.T, hd, kv_heads), mesh, (None, "tensor"))
        for w, hd in ((k, head_dim), (v, v_head_dim))
    )


def _infer_qkv_shards(
    total_out_dim,
    total_scale_blocks,
    num_heads,
    num_kv_heads,
    head_dim,
    v_head_dim,
    block_size,
):
    # Preserve checkpoint TP inference: prefer the largest compatible divisor.
    for kv in sorted(
        (d for d in range(1, num_kv_heads + 1) if num_kv_heads % d == 0), reverse=True
    ):
        for tp in sorted((d for d in range(1, num_heads + 1) if num_heads % d == 0), reverse=True):
            if kv % tp:
                continue
            size = num_heads // tp * head_dim + kv // tp * (head_dim + v_head_dim)
            if (
                size * tp == total_out_dim
                and math.ceil(size / block_size) * tp == total_scale_blocks
            ):
                return tp, kv
    raise ValueError(
        f"Cannot infer QKV shard count: weight={total_out_dim}, scales={total_scale_blocks}"
    )


def load_fused_qkv(inputs, *, head_dim, v_head_dim, num_heads, kv_heads, block_size, mesh):
    weight, scale = inputs
    count, original_kv = _infer_qkv_shards(
        weight.shape[0],
        scale.shape[0],
        num_heads,
        kv_heads,
        head_dim,
        v_head_dim,
        block_size,
    )
    sizes = (
        num_heads // count * head_dim,
        original_kv // count * head_dim,
        original_kv // count * v_head_dim,
    )
    rows, cols = sum(sizes), weight.shape[1]
    blocks = math.ceil(rows / block_size)
    parts = ([], [], [])
    for shard in range(count):
        w = weight[shard * rows : (shard + 1) * rows]
        s = scale[shard * blocks : (shard + 1) * blocks]
        w = np.pad(w, ((0, blocks * block_size - rows), (0, 0)))
        # Keep FP32 until each checkpoint shard has been split. Rounding earlier
        # changes the existing FP8 -> BF16 conversion contract.
        full = (
            w.astype(np.float32).reshape(blocks, block_size, cols // block_size, block_size)
            * s[:, None, :, None]
        ).reshape(blocks * block_size, cols)[:rows]
        for dst, part in zip(parts, np.split(full, np.cumsum(sizes)[:-1], axis=0)):
            dst.append(part.T.copy())
    outputs = []
    for i, group in enumerate(parts):
        value = np.concatenate(group, axis=1).astype(ml_dtypes.bfloat16)
        group.clear()
        if i:
            value = _replicate(value, head_dim if i == 1 else v_head_dim, kv_heads)
        outputs.append(_put(value, mesh, (None, "tensor")))
    return tuple(outputs)


def _load_linear(inputs, *, scale_out, head_dim, kernel_axes, mesh, block_size, target_heads=None):
    weight, scale = inputs
    weight = weight.T.astype(np.float32)
    if scale.ndim == 1 or (scale.ndim == 2 and scale.shape[-1] == 1 and block_size is None):
        result = weight * scale.reshape(1, -1)
    else:
        # Same expansion and per-head indexing as the former post-load path.
        expanded = np.repeat(scale.T, block_size, axis=1)[:, :scale_out]
        if weight.shape[1] != expanded.shape[1] and head_dim is not None:
            indices = (
                (np.arange(weight.shape[1]) // head_dim) * math.ceil(head_dim / 128)
                + (np.arange(weight.shape[1]) % head_dim) // 128
            ) * 128
            expanded = expanded[:, indices]
        elif weight.shape[1] > expanded.shape[1]:
            expanded = np.tile(expanded, (1, weight.shape[1] // expanded.shape[1]))
        result = (
            weight.reshape(expanded.shape[0], -1, weight.shape[1]) * expanded[:, None, :]
        ).reshape(weight.shape)
    result = result.astype(ml_dtypes.bfloat16)
    if target_heads is not None:
        result = _replicate(result, head_dim, target_heads)
    return (_put(result, mesh, kernel_axes),)


def fused_qkv_spec(source, target, attn, mesh, block_size=128):
    return WeightSpec(
        [f"{target}.{name}.weight" for name in ("q_proj", "k_proj", "v_proj")],
        sources=(source + ".weight", source + ".weight_scale_inv"),
        recipe=partial(
            load_fused_qkv,
            head_dim=attn.head_dim,
            v_head_dim=attn.v_head_dim,
            num_heads=attn.q_head_num,
            kv_heads=attn.k_head_num,
            block_size=block_size,
            mesh=mesh,
        ),
    )


def fused_kv_spec(source, target, attn, mesh, block_size=128):
    return WeightSpec(
        [f"{target}.{name}.weight" for name in ("k_proj", "v_proj")],
        sources=tuple(
            f"{source}.{name}.{suffix}"
            for name in ("k_proj", "v_proj")
            for suffix in ("weight", "weight_scale_inv")
        ),
        recipe=partial(
            load_fused_kv,
            head_dim=attn.head_dim,
            v_head_dim=attn.v_head_dim,
            kv_heads=attn.k_head_num,
            block_size=block_size,
            mesh=mesh,
        ),
    )


def prepare_mimo(model, loader, mappings, layers):
    """Fix BF16 module structure before the loader captures parameter references."""
    if not loader.is_static_quant:
        return mappings
    mappings = dict(mappings)
    quant = loader.model_config.quantization_config
    block = getattr(quant, "weight_block_size", None)
    for prefix, layer, dense in layers:
        selected = [
            ("self_attn." + proj, hd)
            for proj, hd in (
                ("q_proj", layer.self_attn.head_dim),
                ("k_proj", layer.self_attn.head_dim),
                ("v_proj", layer.self_attn.v_head_dim),
            )
        ]
        if dense:
            selected.extend(("mlp." + proj, None) for proj in ("gate_proj", "up_proj", "down_proj"))
        for relative, hd in selected:
            parent_name, attr = relative.split(".")
            parent = getattr(layer, parent_name)
            linear = getattr(parent, attr)
            if not isinstance(linear, QuantizedLinear):
                continue
            path = prefix + "." + relative
            outs, ins = linear.weight_q.value.shape
            axes = linear.kernel_axes
            sources = {
                spec.target_path: key
                for key, spec in mappings.items()
                if isinstance(spec.target_path, str)
            }
            wk, sk = sources.get(path + ".weight_q"), sources.get(path + ".weight_scale")
            if wk is not None:
                if sk is None:
                    raise ValueError(f"Missing quantized scale declaration: {path}")
                del mappings[wk], mappings[sk]
                scale_shape = linear.weight_scale.value.shape
                mappings[wk] = WeightSpec(
                    path + ".weight",
                    sources=(wk, sk),
                    recipe=partial(
                        _load_linear,
                        scale_out=scale_shape[-1],
                        head_dim=hd,
                        kernel_axes=axes,
                        mesh=loader.mesh,
                        block_size=block[0] if block else None,
                        target_heads=(
                            layer.self_attn.k_head_num if attr in ("k_proj", "v_proj") else None
                        ),
                    ),
                )
            if attr in ("k_proj", "v_proj"):
                outs = layer.self_attn.k_head_num * hd
            # Only abstract leaves are created here; no weight I/O or H2D.
            with jax.set_mesh(loader.mesh):
                replacement = nnx.eval_shape(
                    lambda ins=ins, outs=outs, axes=axes, linear=linear: LinearBase(
                        input_size=ins,
                        output_size=outs,
                        kernel_axes=axes,
                        use_bias=linear.bias is not None,
                        params_dtype=jnp.bfloat16,
                        mesh=loader.mesh,
                    )
                )
            if linear.bias is not None:
                replacement.bias = linear.bias
            setattr(parent, attr, replacement)
    return mappings
