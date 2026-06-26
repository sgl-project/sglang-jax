"""GDN attention backend integration tests (single-mesh).

Mirrors ``test_kda_attention.py`` structure but exercises the fused
``conv1d`` + Gated-DeltaNet recurrence path:
* the layer carries a single fused ``conv1d`` over ``[Q | K | V]``
  (vs KDA's three split conv1ds),
* the gates ``a`` / ``b`` are per-V-head ``[T, n_v]`` (vs KDA's
  ``[T, n_v, d_v]`` / ``[T, n_v]``),
* GQA can repeat Q/K from ``n_kq`` to ``n_v`` heads.

The reference oracle ``ref_gdn_attention`` lifts
``_python_reference`` from the (P1.11-deleted)
``test_ragged_gated_delta_rule_ref.py`` so this file remains the
single source of truth for GDN numerics after P1.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import GDNAttnBackendForTest

# Single-mesh test fixture uses one TPU chip (dp=1, tp=1). GDN's fused
# conv1d requires the model loader to stripe-rearrange weights into
# per-shard ``[Q_local | K_local | V_local]`` layout under TP > 1
# (mirrors ``MergedColumnParallelLinear``'s ``in_proj_qkv`` convention) —
# this test fixture skips that rearrange to keep the math reference
# simple. TP/DP combinations (including the pure-TP probe that
# exercises the stripe contract) live in ``test_gdn_attention_dp.py``.
mesh = create_device_mesh(
    ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
)
jax.sharding.set_mesh(mesh)


def _scaled_randn(rng: np.random.Generator, shape, scale: float = 0.1) -> np.ndarray:
    # Same rationale as KDA test: shrink the recurrent state so bf16 noise in
    # the delta-rule update fits the global atol=1e-2.
    return rng.standard_normal(shape).astype(np.float32) * scale


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

    ``mixed_qkv`` / ``cache`` are unsharded host tensors. ``weight`` may
    be sharded; we force-unshard via ``jnp.asarray(np.asarray(...))`` to
    avoid the ``conv_general_dilated`` requiring ``out_sharding=`` under
    ``set_mesh`` (same workaround KDA's ``make_nnx_depthwise_conv``
    applies).
    """
    from flax import nnx

    weight = jnp.asarray(np.asarray(weight), dtype=weight.dtype)
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
    """Token-by-token GDN reference for a SINGLE request.

    Lifted (and lightly broadened — handles ``A_log`` shape ``[n_v]`` or
    ``[n_v, 1]``) from
    ``test_ragged_gated_delta_rule_ref._python_reference``.
    """
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


def create_random_states(
    batch_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    dtype,
    rng: np.random.Generator,
) -> tuple[jax.Array, jax.Array]:
    conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    ssm = jnp.asarray(
        _scaled_randn(rng, (batch_size, num_v_heads, head_k_dim, head_v_dim)),
        dtype=jnp.float32,
    )
    conv = jnp.asarray(
        _scaled_randn(rng, (batch_size, conv_dim, conv_kernel_size - 1)),
        dtype=dtype,
    )
    return ssm, conv


def write_initial_state(
    pool: RecurrentStatePool,
    layer_id: int,
    recurrent_indices: np.ndarray,
    ssm_state: jax.Array,
    conv_state: jax.Array,
) -> None:
    recurrent_buffer, conv_buffer_list = pool.get_linear_recurrent_layer_cache(layer_id)
    recurrent_buffer = recurrent_buffer.at[recurrent_indices].set(
        ssm_state, out_sharding=pool.recurrent_sharding
    )
    conv_buffer = (
        conv_buffer_list[0].at[recurrent_indices].set(conv_state, out_sharding=pool.conv_sharding)
    )
    pool.replace_buffer(([recurrent_buffer], [[conv_buffer]]))


def gather_ssm(pool: RecurrentStatePool, recurrent_buffer: jax.Array, indices: np.ndarray):
    return recurrent_buffer.at[indices].get(out_sharding=pool.recurrent_sharding)


def gather_conv(pool: RecurrentStatePool, conv_buffer: jax.Array, indices: np.ndarray):
    return conv_buffer.at[indices].get(out_sharding=pool.conv_sharding)


def create_test_data(
    mode: str,
    seq_lens: list[int],
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    dtype,
    rng: np.random.Generator,
    test_mesh: jax.sharding.Mesh = mesh,
    layer_id: int = 0,
    all_have_initial_state: bool | list[bool] = False,
    initial_ssm_state: jax.Array | None = None,
    initial_conv_state: jax.Array | None = None,
):
    assert mode in ("prefill", "decode")
    forward_mode = ForwardMode.EXTEND if mode == "prefill" else ForwardMode.DECODE
    batch_size = len(seq_lens)

    if isinstance(all_have_initial_state, bool):
        has_initial_state_per_req = [all_have_initial_state] * batch_size
    else:
        assert len(all_have_initial_state) == batch_size
        has_initial_state_per_req = list(all_have_initial_state)

    total_tokens = sum(seq_lens)
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = 2 * key_dim + value_dim

    conv_sharding = NamedSharding(test_mesh, P("tensor", None))
    head_sharding = NamedSharding(test_mesh, P("tensor"))

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
    pool = RecurrentStatePool(
        linear_recurrent_layer_ids=[layer_id],
        size=batch_size,
        num_heads=num_v_heads,
        head_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        mesh=test_mesh,
        dp_size=1,
        recurrent_partition_axis="tensor",
        conv_partition_axis="tensor",
        data_partition_axis="data",
        temporal_dtype=jnp.float32,
        conv_dtype=dtype,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
    )
    q = normal((total_tokens, key_dim))
    k = normal((total_tokens, key_dim))
    v = normal((total_tokens, value_dim))
    # Per-V-head gates (differ from KDA's per-element shape).
    a = normal((total_tokens, num_v_heads))
    b = normal((total_tokens, num_v_heads))

    recurrent_indices = np.arange(1, batch_size + 1, dtype=np.int32)

    if initial_ssm_state is None or initial_conv_state is None:
        random_ssm, random_conv = create_random_states(
            batch_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            dtype,
            rng,
        )
        if initial_ssm_state is None:
            initial_ssm_state = random_ssm
        if initial_conv_state is None:
            initial_conv_state = random_conv

    initial_ssm_state_sharded = jax.device_put(initial_ssm_state, pool.recurrent_sharding)
    initial_conv_state_sharded = jax.device_put(initial_conv_state, pool.conv_sharding)
    write_initial_state(
        pool,
        layer_id,
        recurrent_indices,
        initial_ssm_state_sharded,
        initial_conv_state_sharded,
    )

    seq_lens_np = np.asarray(seq_lens, dtype=np.int32)
    input_ids = np.arange(total_tokens, dtype=np.int32)
    positions = np.arange(total_tokens, dtype=np.int32)
    out_cache_loc = np.arange(total_tokens, dtype=np.int32)
    req_pool_indices = np.arange(batch_size, dtype=np.int32)
    extend_seq_lens = seq_lens_np if mode == "prefill" else None
    extend_prefix_lens = np.zeros(batch_size, dtype=np.int32) if mode == "prefill" else None
    has_initial_state_np = np.asarray(has_initial_state_per_req, dtype=np.bool_)

    backend = GDNAttnBackendForTest(
        GDNAttnBackend(
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            mesh=test_mesh,
        )
    )

    mwb = ModelWorkerBatch(
        bid=1,
        forward_mode=forward_mode,
        input_ids=input_ids,
        real_input_ids_len=input_ids.shape[0],
        seq_lens=seq_lens_np,
        out_cache_loc=out_cache_loc,
        req_pool_indices=req_pool_indices,
        sampling_info=None,
        positions=positions,
        cache_loc=out_cache_loc,
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=np.cumsum(seq_lens_np) - 1 if mode == "prefill" else None,
        real_bs=batch_size,
        real_bs_per_dp=[batch_size],
        dp_size=1,
        per_dp_bs_size=batch_size,
        spec_info_padded=None,
        recurrent_indices=recurrent_indices,
        has_initial_state=has_initial_state_np,
    )

    dummy_model_config = type(
        "DummyModelConfig",
        (),
        {"is_embedding": False, "hf_config": type("DummyHFConfig", (), {"architectures": []})()},
    )()
    dummy_runner = type(
        "DummyRunner",
        (),
        {"mesh": test_mesh, "attn_backend": backend, "model_config": dummy_model_config},
    )()
    fb = ForwardBatch.init_new(mwb, dummy_runner)
    fb.attn_backend.forward_metadata = backend.get_forward_metadata(mwb)

    mask = jnp.asarray(has_initial_state_per_req, dtype=jnp.bool_)
    initial_ssm_ref = jnp.where(
        mask[:, None, None, None], initial_ssm_state, jnp.zeros_like(initial_ssm_state)
    )
    initial_conv_ref = jnp.where(
        mask[:, None, None], initial_conv_state, jnp.zeros_like(initial_conv_state)
    )
    return fb, pool, layer, q, k, v, a, b, initial_ssm_ref, initial_conv_ref, recurrent_indices


class TestGDNAttention(unittest.TestCase):
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
        self.rng = np.random.default_rng(0)
        self.rtol = 2e-2
        self.atol = 1e-2

    def run_test(
        self,
        mode,
        seq_lens,
        test_mesh: jax.sharding.Mesh = mesh,
        all_have_initial_state: bool | list[bool] = False,
        initial_ssm_state: jax.Array | None = None,
        initial_conv_state: jax.Array | None = None,
    ):
        (
            forward_batch,
            pool,
            layer,
            q,
            k,
            v,
            a,
            b,
            initial_ssm_ref,
            initial_conv_ref,
            recurrent_indices,
        ) = create_test_data(
            mode,
            seq_lens,
            self.NUM_K_HEADS,
            self.NUM_V_HEADS,
            self.HEAD_K_DIM,
            self.HEAD_V_DIM,
            self.CONV_KERNEL_SIZE,
            self.DTYPE,
            self.rng,
            test_mesh=test_mesh,
            all_have_initial_state=all_have_initial_state,
            initial_ssm_state=initial_ssm_state,
            initial_conv_state=initial_conv_state,
        )

        expected, expected_ssm, expected_conv = ref_gdn_attention(
            q,
            k,
            v,
            a,
            b,
            layer,
            forward_batch.attn_backend.forward_metadata.cu_q_lens,
            initial_ssm_ref,
            initial_conv_ref,
            forward_batch.forward_mode,
        )

        key_sharding = NamedSharding(test_mesh, P("data", "tensor"))
        v_sharding = NamedSharding(test_mesh, P("data", "tensor"))
        head_sharding = NamedSharding(test_mesh, P("data", "tensor"))
        q_dev = jax.device_put(q, key_sharding)
        k_dev = jax.device_put(k, key_sharding)
        v_dev = jax.device_put(v, v_sharding)
        a_dev = jax.device_put(a, head_sharding)
        b_dev = jax.device_put(b, head_sharding)
        actual, (recurrent_buffer, conv_buffer_list) = layer(
            forward_batch, q_dev, k_dev, v_dev, a_dev, b_dev, pool
        )
        actual_ssm = gather_ssm(pool, recurrent_buffer, recurrent_indices)
        actual_conv = gather_conv(pool, conv_buffer_list[0], recurrent_indices)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), self.rtol, self.atol)
        np.testing.assert_allclose(
            np.asarray(actual_ssm), np.asarray(expected_ssm), self.rtol, self.atol
        )
        np.testing.assert_allclose(
            np.asarray(actual_conv), np.asarray(expected_conv), self.rtol, self.atol
        )
        return layer, pool, recurrent_indices, recurrent_buffer, conv_buffer_list

    def random_states(self, batch_size: int):
        return create_random_states(
            batch_size,
            self.NUM_K_HEADS,
            self.NUM_V_HEADS,
            self.HEAD_K_DIM,
            self.HEAD_V_DIM,
            self.CONV_KERNEL_SIZE,
            self.DTYPE,
            self.rng,
        )

    def test_single_seq_extend_no_shard(self):
        one_device_mesh = create_device_mesh(
            ici_parallelism=[1, 1],
            dcn_parallelism=[1, 1],
            devices=[jax.devices()[0]],
        )
        with jax.sharding.set_mesh(one_device_mesh):
            self.run_test("prefill", [128], test_mesh=one_device_mesh)

    def test_single_seq_extend(self):
        self.run_test("prefill", [128])

    def test_multi_seq_extend(self):
        self.run_test("prefill", [64, 128, 32])

    def test_extend_short_seqs(self):
        self.run_test("prefill", [3, 7, 1])

    def test_extend_with_initial_state(self):
        ssm, conv = self.random_states(batch_size=1)
        self.run_test(
            "prefill",
            [64],
            all_have_initial_state=True,
            initial_ssm_state=ssm,
            initial_conv_state=conv,
        )

    def test_extend_short_request_no_state_ignores_poisoned_slot(self):
        """Fresh prefill (has_initial_state=False) of a request shorter than K-1
        must zero-pad its conv final-state, not leak the slot. We poison the slot
        and assert the sentinel never appears (run_test also compares the gathered
        state to the zero-masked reference).
        """
        K = self.CONV_KERNEL_SIZE
        T = K - 2  # shorter than the K-1 conv window
        conv_dim = 2 * self.NUM_K_HEADS * self.HEAD_K_DIM + self.NUM_V_HEADS * self.HEAD_V_DIM
        poison = 1.0e4
        poison_conv = jnp.full((1, conv_dim, K - 1), poison, dtype=self.DTYPE)

        _, pool, recurrent_indices, _, conv_buffer_list = self.run_test(
            "prefill",
            [T],
            all_have_initial_state=False,
            initial_conv_state=poison_conv,
        )

        final_conv = np.asarray(
            gather_conv(pool, conv_buffer_list[0], recurrent_indices), dtype=np.float32
        )
        # Real state holds the request's own tokens (~0.1); the 1e4 sentinel must not appear.
        self.assertFalse(
            bool(np.any(np.abs(final_conv) > poison / 2)),
            "stale poisoned slot leaked into the conv final-state",
        )

    def test_single_step_decode(self):
        ssm, conv = self.random_states(batch_size=1)
        self.run_test(
            "decode",
            [1],
            all_have_initial_state=True,
            initial_ssm_state=ssm,
            initial_conv_state=conv,
        )

    def test_multi_request_decode(self):
        ssm, conv = self.random_states(batch_size=3)
        self.run_test(
            "decode",
            [1, 1, 1],
            all_have_initial_state=True,
            initial_ssm_state=ssm,
            initial_conv_state=conv,
        )

    def test_decode_mixed_new_continuing(self):
        ssm, conv = self.random_states(batch_size=4)
        self.run_test(
            "decode",
            [1, 1, 1, 1],
            all_have_initial_state=[True, False, True, False],
            initial_ssm_state=ssm,
            initial_conv_state=conv,
        )

    def test_extend_then_decode(self):
        layer, pool, recurrent_indices, recurrent_buffer, conv_buffer_list = self.run_test(
            "prefill", [128]
        )
        pool.replace_buffer(([recurrent_buffer], [conv_buffer_list]))

        ssm_ref = gather_ssm(pool, recurrent_buffer, recurrent_indices)
        conv_ref = gather_conv(pool, conv_buffer_list[0], recurrent_indices)
        ssm_ref = jnp.asarray(np.asarray(ssm_ref), dtype=ssm_ref.dtype)
        conv_ref = jnp.asarray(np.asarray(conv_ref), dtype=conv_ref.dtype)
        for _ in range(4):
            (
                forward_batch,
                _pool,
                _layer,
                q,
                k,
                v,
                a,
                b,
                _initial_ssm_ref,
                _initial_conv_ref,
                _recurrent_indices,
            ) = create_test_data(
                "decode",
                [1],
                self.NUM_K_HEADS,
                self.NUM_V_HEADS,
                self.HEAD_K_DIM,
                self.HEAD_V_DIM,
                self.CONV_KERNEL_SIZE,
                self.DTYPE,
                self.rng,
                all_have_initial_state=True,
                initial_ssm_state=ssm_ref,
                initial_conv_state=conv_ref,
            )
            forward_batch.attn_backend.forward_metadata.recurrent_indices = jax.device_put(
                jnp.asarray(recurrent_indices),
                NamedSharding(mesh, P("data")),
            )
            expected, ssm_ref, conv_ref = ref_gdn_attention(
                q,
                k,
                v,
                a,
                b,
                layer,
                forward_batch.attn_backend.forward_metadata.cu_q_lens,
                ssm_ref,
                conv_ref,
                ForwardMode.DECODE,
            )
            key_sharding = NamedSharding(mesh, P("data", "tensor"))
            v_sharding = NamedSharding(mesh, P("data", "tensor"))
            head_sharding = NamedSharding(mesh, P("data", "tensor"))
            q_dev = jax.device_put(q, key_sharding)
            k_dev = jax.device_put(k, key_sharding)
            v_dev = jax.device_put(v, v_sharding)
            a_dev = jax.device_put(a, head_sharding)
            b_dev = jax.device_put(b, head_sharding)
            actual, (recurrent_buffer, conv_buffer_list) = layer(
                forward_batch, q_dev, k_dev, v_dev, a_dev, b_dev, pool
            )
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), self.rtol, self.atol
            )
            np.testing.assert_allclose(
                np.asarray(gather_ssm(pool, recurrent_buffer, recurrent_indices)),
                np.asarray(ssm_ref),
                self.rtol,
                self.atol,
            )
            np.testing.assert_allclose(
                np.asarray(gather_conv(pool, conv_buffer_list[0], recurrent_indices)),
                np.asarray(conv_ref),
                self.rtol,
                self.atol,
            )
            pool.replace_buffer(([recurrent_buffer], [conv_buffer_list]))

    def test_recurrent_cow_copy_equivalence(self):
        """Recurrent CoW: copy_slots clones a prefilled recurrent+conv state
        into another slot bitwise-identically (KL==0 at page_size=1)."""
        layer, pool, rec_idx, rec_buf, conv_buf_list = self.run_test("prefill", [128])
        pool.replace_buffer(([rec_buf], [conv_buf_list]))

        src = int(np.asarray(rec_idx).reshape(-1)[0])
        dst = src + 1 if src + 1 < pool.total_slots else 1
        data_sh = NamedSharding(mesh, P("data"))
        src_arr = jax.device_put(np.array([src], dtype=np.int32), data_sh)
        dst_arr = jax.device_put(np.array([dst], dtype=np.int32), data_sh)
        new_rec, new_conv = jax.jit(pool.copy_slots)(src_arr, dst_arr)
        pool.replace_buffer((new_rec, new_conv))

        src_idx = np.array([src], dtype=np.int32)
        dst_idx = np.array([dst], dtype=np.int32)
        for layer_idx in range(pool.num_linear_recurrent_layers):
            np.testing.assert_array_equal(
                np.asarray(gather_ssm(pool, pool.recurrent_buffers[layer_idx], dst_idx)),
                np.asarray(gather_ssm(pool, pool.recurrent_buffers[layer_idx], src_idx)),
            )
            np.testing.assert_array_equal(
                np.asarray(gather_conv(pool, pool.conv_buffers[layer_idx][0], dst_idx)),
                np.asarray(gather_conv(pool, pool.conv_buffers[layer_idx][0], src_idx)),
            )


if __name__ == "__main__":
    unittest.main()
