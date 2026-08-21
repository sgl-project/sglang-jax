"""JAX primitives lowered to TT attention custom calls."""

from __future__ import annotations

import jax.numpy as jnp
from jax.extend import core
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import stablehlo
from jax.interpreters import mlir


def custom_call(
    name,
    operands,
    result_types,
    *,
    backend_config="",
    frontend_attributes=None,
):
    i32 = ir.IntegerType.get_signless(32)
    op = stablehlo.CustomCallOp(
        result_types,
        operands,
        ir.StringAttr.get(name),
        backend_config=ir.StringAttr.get(backend_config),
        api_version=ir.IntegerAttr.get(i32, 2),
    )
    if frontend_attributes is not None:
        op.operation.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
            {
                key: ir.StringAttr.get(value)
                for key, value in frontend_attributes.items()
            }
        )
    return op.results


def _primitive(name):
    primitive = core.Primitive(name.replace(".", "_"))
    primitive.def_abstract_eval(lambda result, *_operands, **_attrs: result)

    def lowering(_ctx, *operands, **attrs):
        return custom_call(
            name,
            operands,
            [operands[0].type],
            frontend_attributes=attrs,
        )

    mlir.register_lowering(primitive, lowering, platform="tt")
    return primitive


_scaled_dot_product_attention = _primitive("tt.scaled_dot_product_attention")
_paged_scaled_dot_product_attention_decode = _primitive(
    "tt.paged_scaled_dot_product_attention_decode"
)
_paged_update_cache = _primitive("tt.paged_update_cache")
_paged_fill_cache = _primitive("tt.paged_fill_cache")
_weight_dtype_override = _primitive("tt.weight_dtype_override")


def scaled_dot_product_attention(query, key, value):
    return _scaled_dot_product_attention.bind(query, key, value)


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
