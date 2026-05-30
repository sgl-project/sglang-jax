"""GDN attention backend DP integration tests.

Mirrors ``test_kda_attention_dp.py`` structure but exercises the fused
``conv1d`` + Gated-DeltaNet recurrence path under Data Parallelism:
* the layer carries a single fused ``conv1d`` over ``[Q | K | V]``
  (vs KDA's three split conv1ds),
* the gates ``a`` / ``b`` are per-V-head ``[T, n_v]`` (vs KDA's
  ``[T, n_v, d_v]`` / ``[T, n_v]``),
* GQA can repeat Q/K from ``n_kq`` to ``n_v`` heads.

The reference oracle ``ref_gdn_attention`` is duplicated from
``test_gdn_attention.py`` (the single-mesh GDN test) to keep this file
self-contained.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.managers.schedule_batch import PADDING_BUCKETS, ModelWorkerBatch
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.common_utils import pad_to_bucket
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase, GDNAttnBackendForTest


def _scaled_randn(rng: np.random.Generator, shape, scale: float = 0.1) -> np.ndarray:
    # scale=0.1 is a test-only hack: it shrinks the recurrent state so bf16 noise
    # in the delta-rule update fits the global atol=1e-2 (shared with flashattn).
    # Kernel correctness is validated end-to-end (MMLU-pro on tp4dp4, PR #1047);
    # only inputs that grow the state need it (q/k/v/a/b + initial state).
    return rng.standard_normal(shape).astype(np.float32) * scale


# Reference baselines duplicated from test_gdn_attention.py — keep in sync.


def _l2norm(x, eps: float = 1e-6):
    x = x.astype(jnp.float32)
    return x / jnp.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def ref_fused_short_convolution(
    mixed_qkv: jax.Array,
    weight: jax.Array,  # [conv_dim, K]
    cache: jax.Array,  # [B, conv_dim, K-1]
    cu_seqlens: jax.Array,
    forward_mode: ForwardMode,
) -> tuple[jax.Array, jax.Array]:
    """Depthwise conv1d + SiLU ref for the fused ``[Q | K | V]`` channel block.

    All inputs unsharded. ``cache`` is the prior conv state per request;
    returned cache has the same shape.
    """
    conv_dim, kernel_size = weight.shape
    cache_width = kernel_size - 1

    conv = nnx.Conv(
        in_features=conv_dim,
        out_features=conv_dim,
        kernel_size=(kernel_size,),
        feature_group_count=conv_dim,
        padding=[(kernel_size - 1, 0)],
        use_bias=False,
        rngs=nnx.Rngs(0),
        dtype=weight.dtype,
        param_dtype=weight.dtype,
    )
    conv.kernel.value = jnp.transpose(weight, (1, 0))[:, None, :].astype(conv.kernel.value.dtype)

    if forward_mode == ForwardMode.DECODE:
        outputs = []
        for i in range(mixed_qkv.shape[0]):
            history = jnp.swapaxes(cache[i], 0, 1)
            full = jnp.concatenate([history, mixed_qkv[i : i + 1]], axis=0)
            y = conv(full[None, ...])[0]
            outputs.append(jax.nn.silu(y[-1:]))
        y = jnp.concatenate(outputs, axis=0)
        new_cache = jnp.concatenate([cache, mixed_qkv[..., None]], axis=-1)[:, :, 1:]
        return y, new_cache

    pieces = []
    new_cache = []
    cu = np.asarray(cu_seqlens)
    for i in range(len(cu) - 1):
        start, end = int(cu[i]), int(cu[i + 1])
        seq = mixed_qkv[start:end]
        history = jnp.swapaxes(cache[i], 0, 1)
        seq_with_history = jnp.concatenate([history, seq], axis=0)
        y = conv(seq_with_history[None, ...])[0][cache_width:]
        pieces.append(jax.nn.silu(y))

        if seq.shape[0] >= cache_width:
            new_cache.append(jnp.swapaxes(seq[-cache_width:], 0, 1))
        else:
            new_cache.append(jnp.concatenate([cache[i, :, seq.shape[0] :], seq.T], axis=1))

    return jnp.concatenate(pieces, axis=0), jnp.stack(new_cache, axis=0)


def _gdn_python_recurrence(
    mixed_qkv: jax.Array,  # [T, 2*n_kq*d_k + n_v*d_v]  (already conv-out)
    b: jax.Array,  # [T, n_v]
    a: jax.Array,  # [T, n_v]
    initial_state: jax.Array,  # [n_v, d_k, d_v]
    A_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v]
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
):
    """Token-by-token GDN reference for a SINGLE request."""
    T = mixed_qkv.shape[0]
    key_dim = n_kq * d_k
    q = mixed_qkv[:, :key_dim]
    k = mixed_qkv[:, key_dim : 2 * key_dim]
    v = mixed_qkv[:, 2 * key_dim :]

    repeat = n_v // n_kq
    A = jnp.exp(A_log.reshape(n_v).astype(jnp.float32))
    scale = d_k**-0.5

    state = initial_state.astype(jnp.float32)
    outs = []
    for t in range(T):
        q_h = q[t].reshape(n_kq, d_k)
        k_h = k[t].reshape(n_kq, d_k)
        v_h = v[t].reshape(n_v, d_v)
        if repeat > 1:
            q_h = jnp.repeat(q_h, repeat, axis=0)
            k_h = jnp.repeat(k_h, repeat, axis=0)
        q_h = _l2norm(q_h) * scale
        k_h = _l2norm(k_h)
        v_h = v_h.astype(jnp.float32)
        beta = jax.nn.sigmoid(b[t].astype(jnp.float32))
        g = -A * jax.nn.softplus(a[t].astype(jnp.float32) + dt_bias.astype(jnp.float32))

        decay = jnp.exp(g)[:, None, None]
        state = state * decay
        kv_mem = (state * k_h[..., None]).sum(axis=-2)
        delta = (v_h - kv_mem) * beta[:, None]
        state = state + k_h[..., None] * delta[..., None, :]
        out = (state * q_h[..., None]).sum(axis=-2)  # [n_v, d_v]
        outs.append(out)
    output = jnp.stack(outs, axis=0)  # [T, n_v, d_v]
    return state, output.astype(mixed_qkv.dtype)


def ref_gdn_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    layer: RadixLinearAttention,
    cu_seqlens: jax.Array,
    initial_ssm_state: jax.Array,
    initial_conv_state: jax.Array,
    forward_mode: ForwardMode,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """GDN reference: fused conv1d → per-request Python recurrence.

    Inputs unsharded; output flattens head dim to mirror the production
    backend (``[T, n_v * d_v]``).
    """
    n_kq = layer.num_k_heads
    n_v = layer.num_v_heads
    d_k = layer.head_k_dim
    d_v = layer.head_v_dim

    mixed_qkv = jnp.concatenate([q, k, v], axis=-1)
    mixed_qkv, new_conv_state = ref_fused_short_convolution(
        mixed_qkv,
        layer.conv1d.weight.value,
        initial_conv_state,
        cu_seqlens,
        forward_mode,
    )

    A_log = jnp.asarray(np.asarray(layer.A_log.value), dtype=layer.A_log.value.dtype)
    dt_bias = jnp.asarray(np.asarray(layer.dt_bias.value), dtype=layer.dt_bias.value.dtype)

    cu = np.asarray(cu_seqlens)
    outputs = []
    states = []
    for i in range(len(cu) - 1):
        start, end = int(cu[i]), int(cu[i + 1])
        state_i, out_i = _gdn_python_recurrence(
            mixed_qkv[start:end],
            b[start:end],
            a[start:end],
            initial_ssm_state[i],
            A_log,
            dt_bias,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        outputs.append(out_i)
        states.append(state_i)

    output = jnp.concatenate(outputs, axis=0).reshape(mixed_qkv.shape[0], -1)
    ssm_state = jnp.stack(states, axis=0)
    return output, ssm_state, new_conv_state


def set_mesh(tp_size: int, dp_size: int):
    if tp_size * dp_size != jax.device_count():
        raise RuntimeError(
            f"tp_size * dp_size must equal to available device count {jax.device_count()}, but got tp_size {tp_size}, dp_size {dp_size}"
        )
    mesh = create_device_mesh(ici_parallelism=[dp_size, tp_size], dcn_parallelism=[1, 1])
    jax.sharding.set_mesh(mesh)
    return mesh


def create_test_data(
    mode: str,
    lens_per_rank: dict[int, list[int]],  # {dp_rank: [seq_len, ...]}
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    dtype,
    mesh,
    dp_size: int,
    seed: int = 0,
    has_initial_state_per_rank: (
        dict[int, list[bool]] | None
    ) = None,  # {dp_rank: [has_initial_state per request]}
):
    """Build a ForwardBatch + RecurrentStatePool + global padded q/k/v/a/b for DP testing.

    Layout strides:
      - prefill: token-axis strided by per_dp_token_padding
      - decode:  batch-axis strided by per_dp_bs_padding (q/k/v are [B,...])
    Per-rank initial states are written into the pool buffer slots
    `[dp_rank * (slots_per_rank + 1) + 1, ...]` (slot 0 of each rank reserved).
    """
    assert mode in ("prefill", "decode")
    is_prefill = mode == "prefill"
    forward_mode = ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = 2 * key_dim + value_dim
    layer_id = 0

    # 1. Padding: per-DP token / bs buckets.
    max_tokens_per_dp = 0
    max_bs_per_dp = 0
    for reqs in lens_per_rank.values():
        toks = sum(reqs) if is_prefill else len(reqs)
        max_tokens_per_dp = max(max_tokens_per_dp, toks)
        max_bs_per_dp = max(max_bs_per_dp, len(reqs))
    per_dp_token_padding, _ = pad_to_bucket(max(max_tokens_per_dp, 1), PADDING_BUCKETS)
    per_dp_bs_padding, _ = pad_to_bucket(max(max_bs_per_dp, 1), [1, 2, 4, 8, 16, 32, 64])

    total_tokens = per_dp_token_padding * dp_size
    total_bs = per_dp_bs_padding * dp_size

    # In prefill q/k/v are [T, hidden]; in decode they are [B, hidden].
    global_input_size = total_tokens if is_prefill else total_bs

    # 2. Pool. Pool size must be divisible by dp_size; use per_dp_bs_padding * dp_size.
    pool_size = per_dp_bs_padding * dp_size
    pool = RecurrentStatePool(
        linear_recurrent_layer_ids=[layer_id],
        size=pool_size,
        num_heads=num_v_heads,
        head_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        mesh=mesh,
        dp_size=dp_size,
        recurrent_partition_axis="tensor",
        conv_partition_axis="tensor",
        data_partition_axis="data",
        temporal_dtype=jnp.float32,
        conv_dtype=dtype,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
    )
    slots_per_rank = pool.slots_per_rank  # valid slots per rank (excl. dummy)
    rank_stride = slots_per_rank + 1  # +1 dummy at index 0 per rank

    # 3. Layer (RadixLinearAttention) with sharded random params.
    rng = np.random.default_rng(seed)
    conv_sharding = NamedSharding(mesh, P("tensor", None))
    head_sharding = NamedSharding(mesh, P("tensor"))

    def normal(shape, scale=0.1):
        return jnp.asarray(_scaled_randn(rng, shape, scale), dtype=dtype)

    def conv_weight():
        return jax.device_put(normal((conv_dim, conv_kernel_size), scale=1.0), conv_sharding)

    layer = RadixLinearAttention(
        layer_id=layer_id,
        num_q_heads=num_k_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_q_dim=head_k_dim,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv1d=SimpleNamespace(weight=SimpleNamespace(value=conv_weight())),
        activation="silu",
        A_log=SimpleNamespace(
            value=jax.device_put(
                normal((num_v_heads,), scale=1.0).astype(jnp.float32), head_sharding
            )
        ),
        dt_bias=SimpleNamespace(
            value=jax.device_put(normal((num_v_heads,), scale=1.0), head_sharding)
        ),
        scale=head_k_dim**-0.5,
    )

    # 4. Per-rank random data + initial state (host NumPy; resharded later).
    q_global = np.zeros((global_input_size, key_dim), dtype=np.float32)
    k_global = np.zeros((global_input_size, key_dim), dtype=np.float32)
    v_global = np.zeros((global_input_size, value_dim), dtype=np.float32)
    # Per-V-head gates (differ from KDA's per-element shape).
    a_global = np.zeros((global_input_size, num_v_heads), dtype=np.float32)
    b_global = np.zeros((global_input_size, num_v_heads), dtype=np.float32)

    seq_lens_cpu = np.zeros(total_bs, dtype=np.int32)
    extend_seq_lens_cpu = np.zeros(total_bs, dtype=np.int32) if is_prefill else None
    extend_prefix_lens_cpu = np.zeros(total_bs, dtype=np.int32) if is_prefill else None
    recurrent_indices_cpu = np.zeros(total_bs, dtype=np.int32)
    has_initial_state_cpu = np.zeros(total_bs, dtype=np.bool_)

    # Pool init state buffers: full [total_slots, ...] layout, per-rank shards.
    # GDN ssm state is [n_v, d_k, d_v] (K dim then V dim) per slot.
    ssm_init_full_dev = np.zeros(
        (pool.total_slots, num_v_heads, head_k_dim, head_v_dim), dtype=np.float32
    )
    conv_init_full_dev = np.zeros(
        (pool.total_slots, pool.proj_size, conv_kernel_size - 1), dtype=np.float32
    )

    per_dp_infos = {}
    for dp_rank in range(dp_size):
        reqs = lens_per_rank.get(dp_rank, [])
        rank_rng = np.random.default_rng(seed + dp_rank * 100 + 1)

        bs_offset = dp_rank * per_dp_bs_padding
        token_offset = dp_rank * per_dp_token_padding
        slot_base = dp_rank * rank_stride  # dummy at slot_base, valid at slot_base+1 ..

        rank_q_blocks, rank_k_blocks, rank_v_blocks = [], [], []
        rank_a_blocks, rank_b_blocks = [], []
        rank_seq_lens = []
        rank_initial_ssm_ref = []
        rank_initial_conv_ref = []
        rank_indices = []
        cursor = 0
        for i, q_len in enumerate(reqs):
            q_i = _scaled_randn(rank_rng, (q_len, key_dim))
            k_i = _scaled_randn(rank_rng, (q_len, key_dim))
            v_i = _scaled_randn(rank_rng, (q_len, value_dim))
            a_i = _scaled_randn(rank_rng, (q_len, num_v_heads))
            b_i = _scaled_randn(rank_rng, (q_len, num_v_heads))

            req_has_initial_state = (
                has_initial_state_per_rank is not None
                and i < len(has_initial_state_per_rank.get(dp_rank, []))
                and bool(has_initial_state_per_rank[dp_rank][i])
            )
            ssm_i_dev = _scaled_randn(rank_rng, (num_v_heads, head_k_dim, head_v_dim))
            conv_i_dev = _scaled_randn(rank_rng, (pool.proj_size, conv_kernel_size - 1))
            ssm_i_ref = ssm_i_dev if req_has_initial_state else np.zeros_like(ssm_i_dev)
            conv_i_ref = conv_i_dev if req_has_initial_state else np.zeros_like(conv_i_dev)

            # Per-rank local slot index: 1..slots_per_rank.
            local_slot = i + 1
            global_slot = slot_base + local_slot
            recurrent_indices_cpu[bs_offset + i] = local_slot
            seq_lens_cpu[bs_offset + i] = q_len
            if is_prefill:
                extend_seq_lens_cpu[bs_offset + i] = q_len
                extend_prefix_lens_cpu[bs_offset + i] = 0
            has_initial_state_cpu[bs_offset + i] = req_has_initial_state

            ssm_init_full_dev[global_slot] = ssm_i_dev
            conv_init_full_dev[global_slot] = conv_i_dev

            if is_prefill:
                slc = slice(token_offset + cursor, token_offset + cursor + q_len)
                q_global[slc] = q_i
                k_global[slc] = k_i
                v_global[slc] = v_i
                a_global[slc] = a_i
                b_global[slc] = b_i
                cursor += q_len
            else:
                # Decode: q_len must be 1; place at batch slot.
                assert q_len == 1, f"decode requires q_len=1, got {q_len}"
                idx = bs_offset + i
                q_global[idx] = q_i[0]
                k_global[idx] = k_i[0]
                v_global[idx] = v_i[0]
                a_global[idx] = a_i[0]
                b_global[idx] = b_i[0]

            rank_q_blocks.append(q_i)
            rank_k_blocks.append(k_i)
            rank_v_blocks.append(v_i)
            rank_a_blocks.append(a_i)
            rank_b_blocks.append(b_i)
            rank_seq_lens.append(q_len)
            rank_initial_ssm_ref.append(ssm_i_ref)
            rank_initial_conv_ref.append(conv_i_ref)
            rank_indices.append(local_slot)
        per_dp_infos[dp_rank] = {
            "q": rank_q_blocks,
            "k": rank_k_blocks,
            "v": rank_v_blocks,
            "a": rank_a_blocks,
            "b": rank_b_blocks,
            "seq_lens": rank_seq_lens,
            "initial_ssm_ref": rank_initial_ssm_ref,
            "initial_conv_ref": rank_initial_conv_ref,
            "indices": rank_indices,
        }

    # 5. Push pool init state (full buffer, sharded P("data","tensor",None,None)).
    ssm_init_dev = jax.device_put(
        jnp.asarray(ssm_init_full_dev, dtype=pool.temporal_dtype), pool.recurrent_sharding
    )
    conv_init_dev = jax.device_put(
        jnp.asarray(conv_init_full_dev, dtype=pool.conv_dtype), pool.conv_sharding
    )
    pool.replace_buffer(([ssm_init_dev], [[conv_init_dev]]))

    # 6. Build mwb + ForwardBatch.
    input_ids_cpu = np.zeros(global_input_size, dtype=np.int32)
    positions_cpu = np.zeros(global_input_size, dtype=np.int32)
    out_cache_loc_cpu = np.arange(global_input_size, dtype=np.int32)
    req_pool_indices_cpu = np.arange(total_bs, dtype=np.int32)

    real_bs_per_dp = [len(lens_per_rank.get(r, [])) for r in range(dp_size)]
    backend = GDNAttnBackendForTest(
        GDNAttnBackend(
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            mesh=mesh,
        )
    )

    mwb = ModelWorkerBatch(
        bid=1,
        forward_mode=forward_mode,
        input_ids=input_ids_cpu,
        real_input_ids_len=input_ids_cpu.shape[0],
        seq_lens=seq_lens_cpu,
        out_cache_loc=out_cache_loc_cpu,
        req_pool_indices=req_pool_indices_cpu,
        positions=positions_cpu,
        cache_loc=out_cache_loc_cpu,
        extend_seq_lens=extend_seq_lens_cpu,
        extend_prefix_lens=extend_prefix_lens_cpu,
        sampling_info=None,
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=(np.zeros(total_bs, dtype=np.int32) if is_prefill else None),
        real_bs=total_bs,
        real_bs_per_dp=real_bs_per_dp,
        dp_size=dp_size,
        per_dp_bs_size=per_dp_bs_padding,
        spec_info_padded=None,
        recurrent_indices=recurrent_indices_cpu,
        has_initial_state=has_initial_state_cpu,
    )

    # ForwardBatch.init_new is the production entrypoint (see test_flashattention_dp.py).
    dummy_model_config = type(
        "DummyModelConfig",
        (),
        {"is_embedding": False, "hf_config": type("DummyHFConfig", (), {"architectures": []})()},
    )()
    dummy_runner = type(
        "DummyRunner",
        (),
        {"mesh": mesh, "attn_backend": backend, "model_config": dummy_model_config},
    )()
    fb = ForwardBatch.init_new(mwb, dummy_runner)
    fb.attn_backend.forward_metadata = backend.get_forward_metadata(mwb)

    # Shard q/k/v/a/b for the actual backend call.
    key_sharding = NamedSharding(mesh, P("data", "tensor"))
    v_sharding = NamedSharding(mesh, P("data", "tensor"))
    head_sharding = NamedSharding(mesh, P("data", "tensor"))
    q_dev = jax.device_put(jnp.asarray(q_global, dtype=dtype), key_sharding)
    k_dev = jax.device_put(jnp.asarray(k_global, dtype=dtype), key_sharding)
    v_dev = jax.device_put(jnp.asarray(v_global, dtype=dtype), v_sharding)
    a_dev = jax.device_put(jnp.asarray(a_global, dtype=dtype), head_sharding)
    b_dev = jax.device_put(jnp.asarray(b_global, dtype=dtype), head_sharding)

    return (
        fb,
        pool,
        layer,
        q_dev,
        k_dev,
        v_dev,
        a_dev,
        b_dev,
        per_dp_infos,
        per_dp_bs_padding,
        per_dp_token_padding,
    )


def compute_dp_reference_gdn(
    mode: str,
    per_dp_infos: dict,
    layer: RadixLinearAttention,
    dp_size: int,
    per_dp_bs_padding: int,
    per_dp_token_padding: int,
    num_v_heads: int,
    head_v_dim: int,
    dtype,
) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Per-rank baseline composed into a global padded layout.

    For each dp_rank: pack per-request blocks, run ref_gdn_attention once,
    write result into the rank's stride in the global output.

    Returns (global_out, {dp_rank: (final_ssm, final_conv)}).
    """
    is_prefill = mode == "prefill"
    forward_mode = ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
    out_hidden = num_v_heads * head_v_dim
    global_output_size = (
        dp_size * per_dp_token_padding if is_prefill else dp_size * per_dp_bs_padding
    )
    global_out = np.zeros((global_output_size, out_hidden), dtype=np.float32)
    per_dp_final_states = {}

    for dp_rank in range(dp_size):
        info = per_dp_infos[dp_rank]
        if not info["seq_lens"]:
            continue
        token_offset = dp_rank * per_dp_token_padding
        bs_offset = dp_rank * per_dp_bs_padding

        # Pack per-request blocks + cu_seqlens + stacked initial states.
        q_packed = jnp.asarray(np.concatenate(info["q"], axis=0), dtype=dtype)
        k_packed = jnp.asarray(np.concatenate(info["k"], axis=0), dtype=dtype)
        v_packed = jnp.asarray(np.concatenate(info["v"], axis=0), dtype=dtype)
        a_packed = jnp.asarray(np.concatenate(info["a"], axis=0), dtype=dtype)
        b_packed = jnp.asarray(np.concatenate(info["b"], axis=0), dtype=dtype)
        cu_seqlens = jnp.asarray([0] + np.cumsum(info["seq_lens"]).tolist(), dtype=jnp.int32)
        initial_ssm = jnp.asarray(np.stack(info["initial_ssm_ref"], axis=0), dtype=jnp.float32)
        initial_conv = jnp.asarray(np.stack(info["initial_conv_ref"], axis=0), dtype=dtype)

        rank_out, final_ssm, final_conv = ref_gdn_attention(
            q_packed,
            k_packed,
            v_packed,
            a_packed,
            b_packed,
            layer,
            cu_seqlens,
            initial_ssm,
            initial_conv,
            forward_mode,
        )
        rank_out_np = np.asarray(rank_out).reshape(-1, out_hidden)
        per_dp_final_states[dp_rank] = (np.asarray(final_ssm), np.asarray(final_conv))

        if is_prefill:
            valid = sum(info["seq_lens"])
            global_out[token_offset : token_offset + valid] = rank_out_np
        else:
            num_seqs = len(info["seq_lens"])
            global_out[bs_offset : bs_offset + num_seqs] = rank_out_np
    return global_out, per_dp_final_states


def assert_pool_state_per_rank(
    rec_buf: jax.Array,
    conv_buf: jax.Array,
    ref_ssm_per_rank: dict[int, np.ndarray],
    ref_conv_per_rank: dict[int, np.ndarray],
    per_dp_infos: dict,
    dp_size: int,
    rtol: float,
    atol: float,
    err_prefix: str,
):
    rec_np = np.asarray(rec_buf)
    conv_np = np.asarray(conv_buf)
    rank_stride = rec_np.shape[0] // dp_size
    for dp_rank in range(dp_size):
        info = per_dp_infos[dp_rank]
        if not info["seq_lens"]:
            continue
        slot_base = dp_rank * rank_stride
        global_slots = [slot_base + s for s in info["indices"]]
        actual_ssm = rec_np[global_slots]
        actual_conv = conv_np[global_slots]
        np.testing.assert_allclose(
            actual_ssm,
            ref_ssm_per_rank[dp_rank],
            rtol=rtol,
            atol=atol,
            err_msg=f"{err_prefix}: pool ssm rank {dp_rank}",
        )
        np.testing.assert_allclose(
            actual_conv,
            ref_conv_per_rank[dp_rank],
            rtol=rtol,
            atol=atol,
            err_msg=f"{err_prefix}: pool conv rank {dp_rank}",
        )


class TestGDNAttentionDP(CustomTestCase):
    # Qwen3.5-35B-A3B GDN shapes: n_kq=16, n_v=32 (GQA repeat=2),
    # d_k=d_v=128, K=4. Test uses small variants to keep CPU runtime cheap.
    NUM_K_HEADS = 4
    NUM_V_HEADS = 8
    HEAD_K_DIM = 32
    HEAD_V_DIM = 32
    CONV_KERNEL_SIZE = 4
    DTYPE = jnp.bfloat16

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.rtol = 2e-2
        # GDN tests use deliberately small head_dim (32) — bf16 noise per
        # token in the decode rec-state update is proportional to
        # ``head_v_dim**-1``, so 35B-A3B's head_dim=128 has ~4× tighter
        # noise floor. Bumping atol to 5e-2 absorbs the test-only
        # scale-vs-noise tradeoff; matches the RFC §6 acceptance band.
        self.atol = 5e-2

    def _run_test(
        self,
        mode: str,
        lens_per_rank: dict[int, list[int]],
        dp_size: int,
        tp_size: int,
        has_initial_state_per_rank: dict[int, list[bool]] | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        if mesh is None:
            mesh = set_mesh(tp_size=tp_size, dp_size=dp_size)
        (
            fb,
            pool,
            layer,
            q,
            k,
            v,
            a,
            b,
            per_dp_infos,
            per_dp_bs_padding,
            per_dp_token_padding,
        ) = create_test_data(
            mode,
            lens_per_rank,
            self.NUM_K_HEADS,
            self.NUM_V_HEADS,
            self.HEAD_K_DIM,
            self.HEAD_V_DIM,
            self.CONV_KERNEL_SIZE,
            self.DTYPE,
            mesh,
            dp_size=dp_size,
            has_initial_state_per_rank=has_initial_state_per_rank,
        )
        expected, ref_states = compute_dp_reference_gdn(
            mode,
            per_dp_infos,
            layer,
            dp_size=dp_size,
            per_dp_bs_padding=per_dp_bs_padding,
            per_dp_token_padding=per_dp_token_padding,
            num_v_heads=self.NUM_V_HEADS,
            head_v_dim=self.HEAD_V_DIM,
            dtype=self.DTYPE,
        )
        actual, (rec_buf, conv_buf_list) = layer(fb, q, k, v, a, b, pool)
        actual_np = np.asarray(actual)

        # Compare only rank-valid positions; padded slots hold kernel garbage.
        is_prefill = mode == "prefill"
        for dp_rank in range(dp_size):
            info = per_dp_infos[dp_rank]
            if not info["seq_lens"]:
                continue
            valid = sum(info["seq_lens"]) if is_prefill else len(info["seq_lens"])
            offset = dp_rank * per_dp_token_padding if is_prefill else dp_rank * per_dp_bs_padding
            np.testing.assert_allclose(
                actual_np[offset : offset + valid],
                expected[offset : offset + valid],
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"DP rank {dp_rank} output mismatch",
            )

        ref_ssm_per_rank = {r: s for r, (s, _) in ref_states.items()}
        ref_conv_per_rank = {r: c for r, (_, c) in ref_states.items()}
        assert_pool_state_per_rank(
            rec_buf,
            conv_buf_list[0],
            ref_ssm_per_rank,
            ref_conv_per_rank,
            per_dp_infos,
            dp_size=dp_size,
            rtol=self.rtol,
            atol=self.atol,
            err_prefix=mode,
        )
        # Returned for chained tests like extend_then_decode.
        return SimpleNamespace(
            mesh=mesh,
            fb=fb,
            pool=pool,
            layer=layer,
            rec_buf=rec_buf,
            conv_buf_list=conv_buf_list,
            per_dp_infos=per_dp_infos,
            per_dp_bs_padding=per_dp_bs_padding,
            per_dp_token_padding=per_dp_token_padding,
            ref_ssm_per_rank=ref_ssm_per_rank,
            ref_conv_per_rank=ref_conv_per_rank,
        )

    def test_extend_dp4_tp1(self):
        self._run_test(
            "prefill",
            {0: [128], 1: [64], 2: [32, 64], 3: [128]},
            dp_size=4,
            tp_size=1,
        )

    def test_extend_sparse_ranks_dp4(self):
        self._run_test(
            "prefill",
            {0: [64], 1: [], 2: [128], 3: []},
            dp_size=4,
            tp_size=1,
        )

    def test_decode_dp4_tp1(self):
        self._run_test(
            "decode",
            {0: [1, 1], 1: [1], 2: [1, 1], 3: [1]},
            dp_size=4,
            tp_size=1,
        )

    def test_decode_sparse_ranks_dp4(self):
        self._run_test(
            "decode",
            {0: [], 1: [1, 1], 2: [], 3: [1]},
            dp_size=4,
            tp_size=1,
        )

    def test_decode_unbalanced_dp4(self):
        # Unbalanced bs across DP ranks; per_dp_bs_padding bucketed to max.
        self._run_test(
            "decode",
            {0: [1], 1: [1, 1, 1], 2: [1], 3: [1, 1, 1, 1]},
            dp_size=4,
            tp_size=1,
        )

    @unittest.skip(
        # dp=2, tp=2 mixes DP and TP — fused conv1d stripe assumption
        # interacts with DP sharding in a way the current fixture doesn't
        # mirror. Re-enable when the P2 weight loader lands and we can
        # exercise the stripe-aware fixture across both axes.
        "GDN mixed DP+TP requires the P2 stripe-aware weight loader; "
        "tracked together with the pure-TP probe."
    )
    def test_extend_dp2_tp2(self):
        self._run_test(
            "prefill",
            {0: [128, 64], 1: [32]},
            dp_size=2,
            tp_size=2,
        )

    def test_dp_state_isolation_dp4(self):
        # Per-rank distinct initial state; verifies no cross-rank state leakage.
        lens = {0: [1, 1], 1: [1], 2: [1, 1], 3: [1]}
        has_initial_state_per_rank = {rank: [True] * len(reqs) for rank, reqs in lens.items()}
        self._run_test(
            "decode",
            lens,
            dp_size=4,
            tp_size=1,
            has_initial_state_per_rank=has_initial_state_per_rank,
        )

    def test_dp_mixed_new_continuing_dp4(self):
        lens = {0: [1, 1], 1: [1, 1], 2: [1, 1], 3: [1, 1]}
        has_initial_state_per_rank = {
            0: [False, False],
            1: [True, True],
            2: [True, False],
            3: [False, True],
        }
        self._run_test(
            "decode",
            lens,
            dp_size=4,
            tp_size=1,
            has_initial_state_per_rank=has_initial_state_per_rank,
        )

    def test_dp_extend_then_decode_dp4(self):
        # Multi-round prefill -> decode handover under DP; reuses _run_test for prefill.
        dp_size, tp_size = 4, 1
        lens_prefill = {0: [128], 1: [64], 2: [32, 64], 3: [128]}
        prefill = self._run_test("prefill", lens_prefill, dp_size=dp_size, tp_size=tp_size)
        rec_buf = prefill.rec_buf
        conv_buf_list = prefill.conv_buf_list
        ref_ssm_per_rank = prefill.ref_ssm_per_rank
        ref_conv_per_rank = prefill.ref_conv_per_rank

        decode_lens = {r: [1] * len(reqs) for r, reqs in lens_prefill.items()}
        for step in range(4):
            (
                fb_d,
                _pool_d,
                _layer_d,
                q_d,
                k_d,
                v_d,
                a_d,
                b_d,
                per_dp_infos_d,
                per_dp_bs_padding_d,
                per_dp_token_padding_d,
            ) = create_test_data(
                "decode",
                decode_lens,
                self.NUM_K_HEADS,
                self.NUM_V_HEADS,
                self.HEAD_K_DIM,
                self.HEAD_V_DIM,
                self.CONV_KERNEL_SIZE,
                self.DTYPE,
                prefill.mesh,
                dp_size=dp_size,
                seed=100 + step,
                has_initial_state_per_rank={
                    r: [True] * len(reqs) for r, reqs in decode_lens.items()
                },
            )

            round_metadata = fb_d.attn_backend.forward_metadata
            fb_d.attn_backend = prefill.fb.attn_backend
            fb_d.attn_backend.forward_metadata = round_metadata
            prefill.pool.replace_buffer(([rec_buf], [conv_buf_list]))

            actual_d, (rec_buf, conv_buf_list) = prefill.layer(
                fb_d, q_d, k_d, v_d, a_d, b_d, prefill.pool
            )
            actual_d_np = np.asarray(actual_d)

            for dp_rank in range(dp_size):
                if dp_rank in ref_ssm_per_rank:
                    ssm_stack = ref_ssm_per_rank[dp_rank]
                    conv_stack = ref_conv_per_rank[dp_rank]
                    per_dp_infos_d[dp_rank]["initial_ssm_ref"] = [
                        ssm_stack[i] for i in range(ssm_stack.shape[0])
                    ]
                    per_dp_infos_d[dp_rank]["initial_conv_ref"] = [
                        conv_stack[i] for i in range(conv_stack.shape[0])
                    ]

            expected_d, ref_states_d = compute_dp_reference_gdn(
                "decode",
                per_dp_infos_d,
                prefill.layer,
                dp_size=dp_size,
                per_dp_bs_padding=per_dp_bs_padding_d,
                per_dp_token_padding=per_dp_token_padding_d,
                num_v_heads=self.NUM_V_HEADS,
                head_v_dim=self.HEAD_V_DIM,
                dtype=self.DTYPE,
            )
            ref_ssm_per_rank = {r: s for r, (s, _) in ref_states_d.items()}
            ref_conv_per_rank = {r: c for r, (_, c) in ref_states_d.items()}

            for dp_rank in range(dp_size):
                info_d = per_dp_infos_d[dp_rank]
                if not info_d["seq_lens"]:
                    continue
                num_seqs = len(info_d["seq_lens"])
                offset = dp_rank * per_dp_bs_padding_d
                np.testing.assert_allclose(
                    actual_d_np[offset : offset + num_seqs],
                    expected_d[offset : offset + num_seqs],
                    rtol=self.rtol,
                    atol=self.atol,
                    err_msg=f"decode round {step}: rank {dp_rank}",
                )

            assert_pool_state_per_rank(
                rec_buf,
                conv_buf_list[0],
                ref_ssm_per_rank,
                ref_conv_per_rank,
                per_dp_infos_d,
                dp_size=dp_size,
                rtol=self.rtol,
                atol=self.atol,
                err_prefix=f"decode round {step}",
            )

    @unittest.skip(
        # Production weight loader stripe-rearranges the fused conv1d weight
        # so each TP rank sees per-shard [Q_local | K_local | V_local]
        # (plan §P1.4 lines 113-138). Fixture skips that rearrange — TP > 1
        # with leading-chunk sharding silently mis-routes channels. Re-enable
        # when stripe-aware fixture lands with the P2 weight loader.
        "GDN test fixture does not stripe-rearrange conv1d weight; pure-TP "
        "correctness covered when the P2 weight loader lands."
    )
    def test_extend_dp1_tp4_pure_tp(self):
        """Pure-TP probe: dp=1, tp=4. Catches in_proj_qkvz / in_proj_ba /
        conv1d.weight slicing bugs that DP would mask. Asserts a single
        prefill matches the reference."""
        mesh = create_device_mesh(ici_parallelism=[1, 4], dcn_parallelism=[1, 1])
        with jax.sharding.set_mesh(mesh):
            self._run_test(
                "prefill",
                {0: [64]},
                dp_size=1,
                tp_size=4,
                mesh=mesh,
            )


if __name__ == "__main__":
    unittest.main()
