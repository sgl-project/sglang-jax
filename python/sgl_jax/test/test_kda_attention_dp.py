import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.kda import naive_recurrent_kda
from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend, l2_normalize
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.managers.schedule_batch import PADDING_BUCKETS, ModelWorkerBatch
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.common_utils import pad_to_bucket
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase, KDAAttnBackendForTest


def _scaled_randn(rng: np.random.Generator, shape, scale: float = 0.1) -> np.ndarray:
    # scale=0.1 is a test-only hack: it shrinks the recurrent state so bf16 noise
    # in the delta-rule update fits the global atol=1e-2 (shared with flashattn).
    # Kernel correctness is validated end-to-end (MMLU-pro on tp4dp4, PR #1047);
    # only inputs that grow the state need it (q/k/v/a/b + initial state).
    return rng.standard_normal(shape).astype(np.float32) * scale


# Reference baselines duplicated from test_kda_attention.py — keep in sync.


def make_nnx_depthwise_conv(weight_dk: jax.Array) -> nnx.Conv:
    """Build an nnx.Conv matching short_convolution's [D, K] depthwise kernel."""
    weight = jnp.asarray(np.asarray(weight_dk), dtype=weight_dk.dtype)
    hidden, kernel_size = weight.shape
    conv = nnx.Conv(
        in_features=hidden,
        out_features=hidden,
        kernel_size=(kernel_size,),
        feature_group_count=hidden,
        padding=[(kernel_size - 1, 0)],
        use_bias=False,
        rngs=nnx.Rngs(0),
        dtype=weight.dtype,
        param_dtype=weight.dtype,
    )
    conv.kernel.value = jnp.transpose(weight, (1, 0))[:, None, :].astype(conv.kernel.value.dtype)
    return conv


def ref_short_convolution(
    x: jax.Array,
    weight: jax.Array,
    cache: jax.Array,
    cu_seqlens: jax.Array,
    forward_mode: ForwardMode,
) -> tuple[jax.Array, jax.Array]:
    """nnx.Conv baseline. All inputs must be unsharded host tensors."""
    conv = make_nnx_depthwise_conv(weight)
    kernel_size = conv.kernel.value.shape[0]
    cache_width = kernel_size - 1

    if forward_mode == ForwardMode.DECODE:
        outputs = []
        for i in range(x.shape[0]):
            history = jnp.swapaxes(cache[i], 0, 1)
            full = jnp.concatenate([history, x[i : i + 1]], axis=0)
            y = conv(full[None, ...])[0]
            outputs.append(jax.nn.silu(y[-1:]))
        y = jnp.concatenate(outputs, axis=0)
        new_cache = jnp.concatenate([cache, x[..., None]], axis=-1)[:, :, 1:]
        return y, new_cache

    pieces = []
    new_cache = []
    cu = np.asarray(cu_seqlens)
    for i in range(len(cu) - 1):
        start, end = int(cu[i]), int(cu[i + 1])
        seq = x[start:end]
        history = jnp.swapaxes(cache[i], 0, 1)
        seq_with_history = jnp.concatenate([history, seq], axis=0)
        y = conv(seq_with_history[None, ...])[0][cache_width:]
        pieces.append(jax.nn.silu(y))

        if seq.shape[0] >= cache_width:
            new_cache.append(jnp.swapaxes(seq[-cache_width:], 0, 1))
        else:
            new_cache.append(jnp.concatenate([cache[i, :, seq.shape[0] :], seq.T], axis=1))

    return jnp.concatenate(pieces, axis=0), jnp.stack(new_cache, axis=0)


def ref_kda_attention(
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
    """KDA reference. All tensor inputs (q/k/v/a/b/initial_*) must be
    unsharded host arrays. Layer params are unsharded locally.
    """
    num_heads = layer.num_q_heads
    head_dim = layer.head_q_dim

    a_log = jnp.asarray(np.asarray(layer.A_log.value), dtype=layer.A_log.value.dtype)
    dt_bias = jnp.asarray(np.asarray(layer.dt_bias.value), dtype=layer.dt_bias.value.dtype)

    q_state, k_state, v_state = jnp.split(initial_conv_state, 3, axis=1)
    q, q_state = ref_short_convolution(
        q,
        layer.q_conv1d.weight.value,
        q_state,
        cu_seqlens,
        forward_mode,
    )
    k, k_state = ref_short_convolution(
        k,
        layer.k_conv1d.weight.value,
        k_state,
        cu_seqlens,
        forward_mode,
    )
    v, v_state = ref_short_convolution(
        v,
        layer.v_conv1d.weight.value,
        v_state,
        cu_seqlens,
        forward_mode,
    )

    q = l2_normalize(q.reshape(q.shape[0], num_heads, head_dim))
    k = l2_normalize(k.reshape(k.shape[0], num_heads, head_dim))
    v = v.reshape(v.shape[0], num_heads, head_dim)
    g = a.reshape(a.shape[0], num_heads, head_dim)
    g = -jnp.exp(a_log.reshape(num_heads, 1).astype(jnp.float32)) * jax.nn.softplus(
        g.astype(jnp.float32) + dt_bias.reshape(num_heads, head_dim).astype(jnp.float32)
    )
    g = g.astype(q.dtype)

    cu = np.asarray(cu_seqlens)
    outputs = []
    states = []
    for i in range(len(cu) - 1):
        start, end = int(cu[i]), int(cu[i + 1])
        o_i, state_i = naive_recurrent_kda(
            q[start:end][None],
            k[start:end][None],
            v[start:end][None],
            g[start:end][None],
            b[start:end][None],
            scale=layer.scale,
            initial_state=initial_ssm_state[i : i + 1],
            output_final_state=True,
        )
        outputs.append(o_i[0])
        states.append(state_i[0])

    output = jnp.concatenate(outputs, axis=0).reshape(q.shape[0], -1)
    ssm_state = jnp.stack(states, axis=0)
    conv_state = jnp.concatenate([q_state, k_state, v_state], axis=1)
    return output, ssm_state, conv_state


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
    num_heads: int,
    head_dim: int,
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
    hidden = num_heads * head_dim
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
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
        mesh=mesh,
        dp_size=dp_size,
        recurrent_partition_axis="tensor",
        conv_partition_axis="tensor",
        data_partition_axis="data",
        temporal_dtype=jnp.float32,
        conv_dtype=dtype,
    )
    slots_per_rank = pool.slots_per_rank  # valid slots per rank (excl. dummy)
    rank_stride = slots_per_rank + 1  # +1 dummy at index 0 per rank

    # 3. Layer (RadixLinearAttention) with sharded random params.
    rng = np.random.default_rng(seed)
    conv_sharding = NamedSharding(mesh, P("tensor", None))
    param_head_sharding = NamedSharding(mesh, P("tensor", None))

    def normal(shape, scale=0.1):
        return jnp.asarray(_scaled_randn(rng, shape, scale), dtype=dtype)

    def conv_weight():
        return jax.device_put(normal((hidden, conv_kernel_size), scale=1.0), conv_sharding)

    layer = RadixLinearAttention(
        layer_id=layer_id,
        num_q_heads=num_heads,
        num_k_heads=num_heads,
        num_v_heads=num_heads,
        head_q_dim=head_dim,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        q_conv1d=SimpleNamespace(weight=SimpleNamespace(value=conv_weight())),
        k_conv1d=SimpleNamespace(weight=SimpleNamespace(value=conv_weight())),
        v_conv1d=SimpleNamespace(weight=SimpleNamespace(value=conv_weight())),
        activation="silu",
        A_log=SimpleNamespace(
            value=jax.device_put(normal((num_heads, 1), scale=1.0), param_head_sharding)
        ),
        dt_bias=SimpleNamespace(
            value=jax.device_put(normal((num_heads, head_dim), scale=1.0), param_head_sharding)
        ),
        scale=head_dim**-0.5,
    )

    # 4. Per-rank random data + initial state (host NumPy; resharded later).
    q_global = np.zeros((global_input_size, hidden), dtype=np.float32)
    k_global = np.zeros((global_input_size, hidden), dtype=np.float32)
    v_global = np.zeros((global_input_size, hidden), dtype=np.float32)
    a_global = np.zeros((global_input_size, hidden), dtype=np.float32)
    b_global = np.zeros((global_input_size, num_heads), dtype=np.float32)

    seq_lens_cpu = np.zeros(total_bs, dtype=np.int32)
    extend_seq_lens_cpu = np.zeros(total_bs, dtype=np.int32) if is_prefill else None
    extend_prefix_lens_cpu = np.zeros(total_bs, dtype=np.int32) if is_prefill else None
    recurrent_indices_cpu = np.zeros(total_bs, dtype=np.int32)
    has_initial_state_cpu = np.zeros(total_bs, dtype=np.bool_)

    # Pool init state buffers: full [total_slots, ...] layout, per-rank shards.
    ssm_init_full_dev = np.zeros(
        (pool.total_slots, num_heads, head_dim, head_dim), dtype=np.float32
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
            q_i = _scaled_randn(rank_rng, (q_len, hidden))
            k_i = _scaled_randn(rank_rng, (q_len, hidden))
            v_i = _scaled_randn(rank_rng, (q_len, hidden))
            a_i = _scaled_randn(rank_rng, (q_len, hidden))
            b_i = _scaled_randn(rank_rng, (q_len, num_heads))

            req_has_initial_state = (
                has_initial_state_per_rank is not None
                and i < len(has_initial_state_per_rank.get(dp_rank, []))
                and bool(has_initial_state_per_rank[dp_rank][i])
            )
            ssm_i_dev = _scaled_randn(rank_rng, (num_heads, head_dim, head_dim))
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
    backend = KDAAttnBackendForTest(KDAAttnBackend(mesh=mesh))

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
    hidden_sharding = NamedSharding(mesh, P("data", "tensor"))
    head_sharding = NamedSharding(mesh, P("data", "tensor"))
    q_dev = jax.device_put(jnp.asarray(q_global, dtype=dtype), hidden_sharding)
    k_dev = jax.device_put(jnp.asarray(k_global, dtype=dtype), hidden_sharding)
    v_dev = jax.device_put(jnp.asarray(v_global, dtype=dtype), hidden_sharding)
    a_dev = jax.device_put(jnp.asarray(a_global, dtype=dtype), hidden_sharding)
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


def compute_dp_reference_kda(
    mode: str,
    per_dp_infos: dict,
    layer: RadixLinearAttention,
    dp_size: int,
    per_dp_bs_padding: int,
    per_dp_token_padding: int,
    num_heads: int,
    head_dim: int,
    dtype,
) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Per-rank baseline composed into a global padded layout.

    For each dp_rank: pack per-request blocks, run ref_kda_attention once,
    write result into the rank's stride in the global output.

    Returns (global_out, {dp_rank: (final_ssm, final_conv)}).
    """
    is_prefill = mode == "prefill"
    forward_mode = ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
    hidden = num_heads * head_dim
    global_output_size = (
        dp_size * per_dp_token_padding if is_prefill else dp_size * per_dp_bs_padding
    )
    global_out = np.zeros((global_output_size, hidden), dtype=np.float32)
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

        rank_out, final_ssm, final_conv = ref_kda_attention(
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
        rank_out_np = np.asarray(rank_out).reshape(-1, hidden)
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


class TestKDAAttentionDP(CustomTestCase):
    NUM_HEADS = 32
    HEAD_DIM = 128
    CONV_KERNEL_SIZE = 4
    DTYPE = jnp.bfloat16

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.rtol = 2e-2
        self.atol = 2e-2

    def _run_test(
        self,
        mode: str,
        lens_per_rank: dict[int, list[int]],
        dp_size: int,
        tp_size: int,
        has_initial_state_per_rank: dict[int, list[bool]] | None = None,
    ):
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
            self.NUM_HEADS,
            self.HEAD_DIM,
            self.CONV_KERNEL_SIZE,
            self.DTYPE,
            mesh,
            dp_size=dp_size,
            has_initial_state_per_rank=has_initial_state_per_rank,
        )
        expected, ref_states = compute_dp_reference_kda(
            mode,
            per_dp_infos,
            layer,
            dp_size=dp_size,
            per_dp_bs_padding=per_dp_bs_padding,
            per_dp_token_padding=per_dp_token_padding,
            num_heads=self.NUM_HEADS,
            head_dim=self.HEAD_DIM,
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
                self.NUM_HEADS,
                self.HEAD_DIM,
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

            expected_d, ref_states_d = compute_dp_reference_kda(
                "decode",
                per_dp_infos_d,
                prefill.layer,
                dp_size=dp_size,
                per_dp_bs_padding=per_dp_bs_padding_d,
                per_dp_token_padding=per_dp_token_padding_d,
                num_heads=self.NUM_HEADS,
                head_dim=self.HEAD_DIM,
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


if __name__ == "__main__":
    unittest.main()
