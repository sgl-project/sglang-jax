"""JAX primitives lowered to TT attention custom calls."""

from __future__ import annotations

import jax.numpy as jnp
from jax._src.interpreters import mlir
from jax.extend import core
from jax.extend.mlir import ir


def _primitive(name):
    primitive = core.Primitive(name.replace(".", "_"))
    primitive.def_abstract_eval(lambda result, *_operands, **_attrs: result)

    def lowering(_ctx, *operands, **attrs):
        extra_attributes = None
        if attrs:
            extra_attributes = {
                "mhlo.frontend_attributes": ir.DictAttr.get(
                    {
                        key: ir.StringAttr.get(value)
                        for key, value in attrs.items()
                    }
                )
            }

        return mlir.custom_call(
            name,
            result_types=[operands[0].type],
            operands=operands,
            extra_attributes=extra_attributes,
        ).results

    mlir.register_lowering(primitive, lowering, platform="tt")
    return primitive


_paged_scaled_dot_product_attention_decode = _primitive(
    "tt.paged_scaled_dot_product_attention_decode"
)
_paged_update_cache = _primitive("tt.paged_update_cache")
_paged_fill_cache = _primitive("tt.paged_fill_cache")
_chunked_scaled_dot_product_attention = _primitive(
    "tt.chunked_scaled_dot_product_attention"
)
_weight_dtype_override = _primitive("tt.weight_dtype_override")


def chunked_scaled_dot_product_attention(
    query, key_cache, value_cache, page_table, chunk_start, *, scale=None
):
    attributes = {} if scale is None else {"scale": str(scale)}
    return _chunked_scaled_dot_product_attention.bind(
        query,
        key_cache,
        value_cache,
        page_table,
        chunk_start,
        **attributes,
    )


def paged_scaled_dot_product_attention_decode(
    query, key_cache, value_cache, page_table, positions
):
    return _paged_scaled_dot_product_attention_decode.bind(
        query,
        key_cache,
        value_cache,
        page_table,
        positions,
        is_causal="True",
        has_attention_mask="False",
        has_cur_pos_tensor="True",
        has_attention_sink="False",
    )


def paged_update_cache(cache, value, positions, page_table):
    return _paged_update_cache.bind(cache, value, positions, page_table)


def paged_fill_cache(cache, value, page_table, batch_indices):
    return _paged_fill_cache.bind(cache, value, page_table, batch_indices)


def annotate_weight_dtype(tensor, dtype):
    if dtype not in {"bf16", "bfp_bf8", "bfp_bf4"}:
        raise ValueError(f"Unsupported TT weight dtype: {dtype}")

    original_shape = tensor.shape
    if tensor.ndim < 3:
        tensor = jnp.reshape(tensor, (1,) * (3 - tensor.ndim) + original_shape)
    tensor = _weight_dtype_override.bind(
        tensor, **{"ttcore.weight_dtype": dtype}
    )
    return jnp.reshape(tensor, original_shape)
