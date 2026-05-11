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
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


def _scaled_randn(rng: np.random.Generator, shape, scale: float = 0.1) -> np.ndarray:
    # scale=0.1 keeps bf16 mantissa precision in the delta-rule recurrence
    # S = S*exp(g) + beta*k*(v - k^T S); larger values overflow.
    return rng.standard_normal(shape).astype(np.float32) * scale


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


def create_random_states(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    dtype,
    rng: np.random.Generator,
) -> tuple[jax.Array, jax.Array]:
    hidden = num_heads * head_dim
    ssm = jnp.asarray(
        _scaled_randn(rng, (batch_size, num_heads, head_dim, head_dim)),
        dtype=jnp.float32,
    )
    conv = jnp.asarray(
        _scaled_randn(rng, (batch_size, hidden * 3, conv_kernel_size - 1)),
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
    num_heads: int,
    head_dim: int,
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
        assert len(all_have_initial_state) == batch_size, (
            f"has_initial_state list length {len(all_have_initial_state)} "
            f"!= batch_size {batch_size}"
        )
        has_initial_state_per_req = list(all_have_initial_state)

    total_tokens = sum(seq_lens)
    hidden = num_heads * head_dim
    conv_sharding = NamedSharding(test_mesh, P("tensor", None))
    param_head_sharding = NamedSharding(test_mesh, P("tensor", None))

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
    pool = RecurrentStatePool(
        linear_recurrent_layer_ids=[layer_id],
        size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
        mesh=test_mesh,
        dp_size=1,
        recurrent_partition_axis="tensor",
        conv_partition_axis="tensor",
        data_partition_axis="data",
        temporal_dtype=jnp.float32,
        conv_dtype=dtype,
    )
    # q/k/v/a/b unsharded for ref; reshard in run_test for the backend call.
    q = normal((total_tokens, hidden))
    k = normal((total_tokens, hidden))
    v = normal((total_tokens, hidden))
    a = normal((total_tokens, hidden))
    b = normal((total_tokens, num_heads))

    recurrent_indices = np.arange(1, batch_size + 1, dtype=np.int32)
    if initial_ssm_state is None or initial_conv_state is None:
        random_ssm, random_conv = create_random_states(
            batch_size, num_heads, head_dim, conv_kernel_size, dtype, rng
        )
        zero_ssm = jnp.zeros_like(random_ssm)
        zero_conv = jnp.zeros_like(random_conv)
        all_true = all(has_initial_state_per_req)
        if initial_ssm_state is None:
            initial_ssm_state = zero_ssm if all_true else random_ssm
        if initial_conv_state is None:
            initial_conv_state = zero_conv if all_true else random_conv

    # Reshard initial state for the (sharded) pool; keep host copies for ref.
    initial_ssm_state_sharded = jax.device_put(initial_ssm_state, pool.recurrent_sharding)
    initial_conv_state_sharded = jax.device_put(initial_conv_state, pool.conv_sharding)
    write_initial_state(
        pool, layer_id, recurrent_indices, initial_ssm_state_sharded, initial_conv_state_sharded
    )

    seq_lens_np = np.asarray(seq_lens, dtype=np.int32)
    input_ids = np.arange(total_tokens, dtype=np.int32)
    positions = np.arange(total_tokens, dtype=np.int32)
    out_cache_loc = np.arange(total_tokens, dtype=np.int32)
    req_pool_indices = np.arange(batch_size, dtype=np.int32)
    extend_seq_lens = seq_lens_np if mode == "prefill" else None
    extend_prefix_lens = np.zeros(batch_size, dtype=np.int32) if mode == "prefill" else None
    has_initial_state_np = np.asarray(has_initial_state_per_req, dtype=np.bool_)

    backend = KDAAttnBackend(mesh=test_mesh)

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
        spec_info=None,
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

    # Mirrors backend's get_state mask: True rows pass through, False rows zero.
    mask = jnp.asarray(has_initial_state_per_req, dtype=jnp.bool_)
    baseline_ssm = jnp.where(
        mask[:, None, None, None], initial_ssm_state, jnp.zeros_like(initial_ssm_state)
    )
    baseline_conv = jnp.where(
        mask[:, None, None], initial_conv_state, jnp.zeros_like(initial_conv_state)
    )
    return fb, pool, layer, q, k, v, a, b, baseline_ssm, baseline_conv, recurrent_indices


class TestKDAAttention(unittest.TestCase):
    NUM_HEADS = 32
    HEAD_DIM = 128
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
        mode_args=None,
        test_mesh: jax.sharding.Mesh = mesh,
        all_have_initial_state: bool | list[bool] = False,
        initial_ssm_state: jax.Array | None = None,
        initial_conv_state: jax.Array | None = None,
    ):
        if mode_args is None:
            mode_args = (self.NUM_HEADS, self.HEAD_DIM, self.CONV_KERNEL_SIZE, self.DTYPE)
        num_heads, head_dim, conv_kernel_size, dtype = mode_args

        (
            forward_batch,
            pool,
            layer,
            q,
            k,
            v,
            a,
            b,
            baseline_ssm,
            baseline_conv,
            recurrent_indices,
        ) = create_test_data(
            mode,
            seq_lens,
            num_heads,
            head_dim,
            conv_kernel_size,
            dtype,
            self.rng,
            test_mesh=test_mesh,
            all_have_initial_state=all_have_initial_state,
            initial_ssm_state=initial_ssm_state,
            initial_conv_state=initial_conv_state,
        )

        expected, expected_ssm, expected_conv = ref_kda_attention(
            q,
            k,
            v,
            a,
            b,
            layer,
            forward_batch.attn_backend.forward_metadata.cu_q_lens,
            baseline_ssm,
            baseline_conv,
            forward_batch.forward_mode,
        )

        # Shard q/k/v/a/b for the actual backend call.
        hidden_sharding = NamedSharding(test_mesh, P("data", "tensor"))
        head_sharding = NamedSharding(test_mesh, P("data", "tensor"))
        q_dev = jax.device_put(q, hidden_sharding)
        k_dev = jax.device_put(k, hidden_sharding)
        v_dev = jax.device_put(v, hidden_sharding)
        a_dev = jax.device_put(a, hidden_sharding)
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
            self.NUM_HEADS,
            self.HEAD_DIM,
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
        # Override module-level mesh so the 1-device pool sharding matches.
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
        # Pool gather is sharded; ref needs unsharded.
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
                _baseline_ssm,
                _baseline_conv,
                _recurrent_indices,
            ) = create_test_data(
                "decode",
                [1],
                self.NUM_HEADS,
                self.HEAD_DIM,
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
            expected, ssm_ref, conv_ref = ref_kda_attention(
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
            hidden_sharding = NamedSharding(mesh, P("data", "tensor"))
            head_sharding = NamedSharding(mesh, P("data", "tensor"))
            q_dev = jax.device_put(q, hidden_sharding)
            k_dev = jax.device_put(k, hidden_sharding)
            v_dev = jax.device_put(v, hidden_sharding)
            a_dev = jax.device_put(a, hidden_sharding)
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


if __name__ == "__main__":
    unittest.main()
