"""Pool-independent, explicitly sharded HCA array execution."""

from __future__ import annotations

import jax
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.hca.hca import HCAMetadata, fused_projection_weight, hca_step
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime


def _metadata_partition_spec(metadata: HCAMetadata) -> HCAMetadata:
    """Every metadata leaf rides the leading data axis, like SGLang batch fields.

    Derived rather than hand-listed so a new field cannot silently miss a spec.
    """
    return jax.tree.map(lambda _: P("data"), metadata)


def _check_constants(
    wkv,
    wgate,
    ape,
    norm_weight,
    cos,
    sin,
    attention_sink,
    fused_weight,
    max_context_len,
    *,
    head_dim,
    compressor_hidden_size,
    compress_ratio,
    num_heads,
) -> None:
    """Validate fixed model shapes at trace time without changing runtime state."""
    if not is_tpu_runtime():
        raise RuntimeError("production HCABackend requires a TPU backend")
    weight_shape = (head_dim, compressor_hidden_size)
    if wkv.shape != weight_shape or wgate.shape != weight_shape:
        raise ValueError(f"wkv and wgate must both have shape {weight_shape}")
    if ape.shape != (compress_ratio, head_dim):
        raise ValueError("ape must be [128,512]")
    if norm_weight.shape != (head_dim,):
        raise ValueError("norm_weight must be [512]")
    if attention_sink.shape != (num_heads,):
        raise ValueError("attention_sink must be [64]")
    if cos.ndim != 2 or cos.shape[1] != 32 or sin.shape != cos.shape:
        raise ValueError("production HCA RoPE tables must both be [positions,32]")
    # Boundary emission gathers the row at each group start; a shorter table
    # would silently rotate records with clamped or filled frequencies.
    min_rope_rows = max(1, max_context_len - compress_ratio + 1)
    if cos.shape[0] < min_rope_rows:
        raise ValueError(
            f"RoPE tables cover {cos.shape[0]} positions but max_context_len="
            f"{max_context_len} requires at least {min_rope_rows}"
        )
    if fused_weight is not None and fused_weight.shape != (
        compressor_hidden_size,
        2 * head_dim,
    ):
        raise ValueError(f"fused_weight must have shape {(compressor_hidden_size, 2 * head_dim)}")


def run_hca(
    q,
    k,
    v,
    *,
    mesh,
    page_size,
    max_context_len,
    positions,
    forward_mode,
    state_arg,
    window_arg,
    compressed_arg,
    metadata,
    compressor_input,
    wkv,
    wgate,
    ape,
    norm_weight,
    cos,
    sin,
    attention_sink,
    fused_weight=None,
    softmax_scale=None,
    norm_eps=1e-6,
    compressor_hidden_size=4096,
):
    """Execute HCA on explicit cache arrays; return output and replacement arrays.

    JAX threads cache state through the compiled model instead of mutating pools
    as the PyTorch backend does. Allocation and layer lookup belong to callers.
    """
    num_heads, head_dim = 64, 512
    compress_ratio, window_size = 128, 128
    if metadata.kernel is None:
        raise RuntimeError("HCABackend.forward_metadata has not been prepared")
    if metadata.schedule is None:
        raise RuntimeError("HCABackend has no HCA kernel schedule")
    # Only the per-token shapes can change between steps; the model
    # constants are validated in _check_constants below.
    if q.ndim != 3 or q.shape[1:] != (num_heads, head_dim):
        raise ValueError("q must be [T,num_attn_heads,head_dim]")
    if k.ndim == 3 and k.shape[1] == 1:
        new_kv = k[:, 0]
    elif k.ndim == 2:
        new_kv = k
    else:
        raise ValueError("HCA k/v must be [T,D] or [T,1,D]")
    if v.shape != k.shape or new_kv.shape != (q.shape[0], head_dim):
        raise ValueError("HCA k and v must share the same KV shape")
    if compressor_input.shape != (q.shape[0], compressor_hidden_size):
        raise ValueError(f"compressor_input must be [T,{compressor_hidden_size}]")
    _check_constants(
        wkv,
        wgate,
        ape,
        norm_weight,
        cos,
        sin,
        attention_sink,
        fused_weight,
        max_context_len,
        head_dim=head_dim,
        compressor_hidden_size=compressor_hidden_size,
        compress_ratio=compress_ratio,
        num_heads=num_heads,
    )

    kernel_options = {
        "softmax_scale": head_dim**-0.5 if softmax_scale is None else float(softmax_scale),
        "compress_ratio": compress_ratio,
        "norm_eps": norm_eps,
        "head_dim": head_dim,
        "window_size": window_size,
        "page_size": page_size,
        # Native 4-D pools carry geometry; only flat views need P / ratio.
        "compressed_page_size": (
            compressed_arg.shape[1] * compressed_arg.shape[2]
            if compressed_arg.ndim == 4
            else max(1, page_size // compress_ratio)
        ),
        "schedule": metadata.schedule,
    }
    fused_weight = fused_projection_weight(wkv, wgate, fused_weight)

    if forward_mode.is_decode():
        kernel_options["mode"] = "decode"
    elif forward_mode.is_extend():
        kernel_options["mode"] = "uniform" if metadata.use_uniform_prefill_fast_path else "ragged"
    else:
        raise ValueError(f"unsupported HCA forward mode: {forward_mode}")

    # Built inline like the MLA and GDN backends: under the model's outer
    # jit this is traced once, so a cached callable buys nothing.
    def rank_local(
        x,
        q_,
        new_kv_,
        state,
        window,
        compressed,
        wkv_,
        wgate_,
        ape_,
        norm_,
        cos_,
        sin_,
        positions_,
        sink_,
        md,
        fused,
    ):
        output, state, window, compressed = hca_step(
            x,
            q_,
            new_kv_,
            state,
            window,
            compressed,
            wkv_,
            wgate_,
            ape_,
            norm_,
            cos_,
            sin_,
            positions_,
            sink_,
            md,
            fused_weight=fused,
            **kernel_options,
        )
        return output.reshape(output.shape[0], -1), state, window, compressed

    output, state, window, compressed = jax.shard_map(
        rank_local,
        mesh=mesh,
        in_specs=(
            P("data", None),  # compressor_input [T, hidden]
            P("data", "tensor", None),  # q                [T, H/tp, D]
            P("data", None),  # new_kv           [T, D]
            _data_spec(state_arg),  # recurrent state pool (rank follows the buffer)
            _data_spec(window_arg),  # window cache
            _data_spec(compressed_arg),  # compressed cache
            P(None, None),  # wkv
            P(None, None),  # wgate
            P(None, None),  # ape
            P(None),  # norm_weight
            P(None, None),  # cos
            P(None, None),  # sin
            P("data"),  # positions        [T]
            P("tensor"),  # attention_sink   [H/tp]
            _metadata_partition_spec(metadata.kernel),
            P(None, None),  # fused_weight
        ),
        out_specs=(
            P("data", "tensor"),  # output [T, H/tp*D]
            _data_spec(state_arg),  # state pool
            _data_spec(window_arg),  # window cache
            _data_spec(compressed_arg),  # compressed cache
        ),
        check_vma=False,
    )(
        compressor_input,
        q,
        new_kv,
        state_arg,
        window_arg,
        compressed_arg,
        wkv,
        wgate,
        ape,
        norm_weight,
        cos,
        sin,
        positions,
        attention_sink,
        metadata.kernel,
        fused_weight,
    )
    return output.astype(q.dtype), (state, window, compressed)


def _data_spec(array) -> P:
    """``P("data", None, ...)`` matching the array rank (flat or 4D pool views)."""
    return P("data", *([None] * (array.ndim - 1)))
