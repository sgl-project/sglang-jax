"""TT GDN parity with the upstream JAX backend and device state ownership."""

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.hardware_backend.tt.attention import ops
from sgl_jax.srt.hardware_backend.tt.attention.gdn_backend import TTGDNAttnBackend, TTGDNMetadata
from sgl_jax.srt.hardware_backend.tt.attention.tt_backend import TTAttention
from sgl_jax.srt.kernels.gdn.gated_delta import (
    _gated_delta_step,
    _scatter_idx0_safe,
    jax_causal_conv1d_update,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend


@pytest.fixture(autouse=True)
def isolated_mesh():
    # Some model tests install an explicit mesh at module import time.
    with jax.set_mesh(jax.sharding.Mesh(np.empty((), dtype=object), ())):
        yield


@pytest.fixture
def tt_device():
    if "tt" not in os.environ.get("JAX_PLATFORMS", "").split(","):
        pytest.skip("requires JAX_PLATFORMS=tt,cpu and a Tenstorrent device")
    return jax.devices("tt")[0]


def test_weight_precision_policy(monkeypatch):
    annotations = []

    def annotate(value, dtype):
        annotations.append((value.shape, dtype))
        return value

    monkeypatch.setattr(ops, "annotate_weight_dtype", annotate)
    leaves = (
        jnp.ones((128, 64), jnp.bfloat16),  # Projection weights.
        jnp.ones((128,), jnp.float32),  # Decay, bias, and normalization parameters.
        jnp.ones((), jnp.float32),
        jnp.ones((32,), jnp.int32),
    )
    backend = HybridLinearAttnBackend(
        full_attn_backend=TTAttention.__new__(TTAttention),
        linear_attn_backend=TTGDNAttnBackend.__new__(TTGDNAttnBackend),
        full_attn_layers=[0],
    )
    prepared = backend.prepare_model_state(leaves)
    assert all(a is b for a, b in zip(prepared, leaves))
    assert annotations == [((128, 64), "bfp_bf8"), ((128,), "bf16")]
    # A global weight override could also affect unannotated recurrent matmuls.
    assert "experimental_weight_dtype" not in backend.compiler_options


def reference_chunk(q, k, v, gate, beta, state):
    def step(state, inputs):
        return _gated_delta_step(state, *inputs)

    state, out = jax.lax.scan(step, state, tuple(x.swapaxes(0, 1) for x in (q, k, v, gate, beta)))
    return state, out.swapaxes(0, 1)


def reference_decode(state, q, k, v, b, a, A_log, dt_bias, indices, initial):
    active = jnp.where(initial[:, None, None, None], state[indices], 0)
    gate = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
        a.astype(jnp.float32) + dt_bias.astype(jnp.float32)
    )
    active, out = _gated_delta_step(active, q, k, v, gate, jax.nn.sigmoid(b.astype(jnp.float32)))
    return _scatter_idx0_safe(state, indices, active), out


def reference_conv(state, value, weight, indices, initial):
    out, state = jax_causal_conv1d_update(
        value, state, indices, weight, activation="silu", has_initial_state=initial
    )
    return state, out


def make_backend(device, metadata, num_k_heads=2, num_v_heads=4, backend_cls=TTGDNAttnBackend):
    backend = backend_cls(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        mesh=jax.sharding.Mesh(np.array([[device]]), ("data", "tensor")),
        dtype=jnp.bfloat16,
        prefill_impl="chunked_jax",
    )
    backend.forward_metadata = metadata
    return backend


def inputs(count, num_k_heads=2, num_v_heads=4, seed=35, slots=3):
    rng = np.random.default_rng(seed)
    channels = (2 * num_k_heads + num_v_heads) * 128

    def rand(shape, scale=0.1, dtype=jnp.bfloat16):
        return jnp.asarray(rng.normal(0, scale, shape), dtype=dtype)

    x = rand((count, channels))
    # The scheduler reserves an immutable zero slot for dummy/fresh requests.
    conv = rand((slots, channels, 3)).at[0].set(0)
    state = rand((slots, num_v_heads, 128, 128), dtype=jnp.float32).at[0].set(0)
    return (
        x,
        conv,
        state,
        rand((count, num_v_heads)),
        rand((count, num_v_heads)),
        rand((channels, 4)),
        rand((num_v_heads,), dtype=jnp.float32),
        rand((num_v_heads,), dtype=jnp.float32),
    )


@partial(jax.jit, static_argnames=("num_k_heads", "num_v_heads", "decode"))
def reference(args, metadata, num_k_heads=2, num_v_heads=4, decode=False):
    native = make_backend(jax.devices("cpu")[0], metadata, num_k_heads, num_v_heads, GDNAttnBackend)
    if decode:
        return native.forward_decode(*args)
    return native.forward_extend(*args, seq_lens=None)


@pytest.fixture
def reference_ops(monkeypatch):
    monkeypatch.setattr(ops, "gated_delta_rule", reference_chunk)
    monkeypatch.setattr(ops, "state_pool_update", _scatter_idx0_safe)
    monkeypatch.setattr(ops, "gated_delta_decode", reference_decode)
    monkeypatch.setattr(ops, "causal_conv1d_update", reference_conv)


@pytest.mark.parametrize("length", [1, 5, 32, 47, 64])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("metadata_size", [1, 4])
def test_prefill(reference_ops, length, initial, metadata_size):
    with jax.default_device(jax.devices("cpu")[0]):
        meta = LinearRecurrentAttnBackendMetadata(
            cu_q_lens=jnp.array([0] + [length] * metadata_size, jnp.int32),
            recurrent_indices=jnp.array([1] + [0] * (metadata_size - 1), jnp.int32),
            has_initial_state=jnp.array([initial] + [False] * (metadata_size - 1)),
        )
        backend = make_backend(jax.devices("cpu")[0], meta)
        args = inputs((length + 31) // 32 * 32)
        expected = reference(args, backend.forward_metadata)
        actual = backend.forward_extend(*args, seq_lens=None)
        # The reference leaves padded outputs unspecified; only live tokens
        # and the complete saved state are part of the serving contract.
        expected = (expected[0][:length], *expected[1:])
        np.testing.assert_array_equal(np.asarray(actual[0][length:]), 0)
        actual = (actual[0][:length], *actual[1:])
        for result, wanted in zip(actual, expected):
            np.testing.assert_allclose(
                np.asarray(result, dtype=np.float32),
                np.asarray(wanted, dtype=np.float32),
                rtol=0.01,
                atol=1e-5,
            )


@pytest.mark.parametrize(
    "indices,initial",
    [([1], [False]), ([1], [True]), ([4, 2, 0, 1], [False, True, False, True])],
)
def test_decode(reference_ops, indices, initial):
    with jax.default_device(jax.devices("cpu")[0]):
        meta = LinearRecurrentAttnBackendMetadata(
            recurrent_indices=jnp.array(indices, jnp.int32),
            has_initial_state=jnp.array(initial),
        )
        backend = make_backend(jax.devices("cpu")[0], meta)
        args = inputs(len(indices), slots=max(indices) + 2)
        expected = reference(args, backend.forward_metadata, decode=True)
        actual = backend.forward_decode(*args)
        for result, wanted in zip(actual, expected):
            np.testing.assert_allclose(
                np.asarray(result, dtype=np.float32),
                np.asarray(wanted, dtype=np.float32),
                rtol=0.01,
                atol=1e-5,
            )


@pytest.mark.parametrize("decode,batch", [(False, 1), (False, 4), (True, 1), (True, 4)])
def test_explicit_serving_mesh(decode, batch):
    cpu = jax.devices("cpu")[0]
    mesh = jax.sharding.Mesh(
        np.array([[cpu]]), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    P = jax.sharding.PartitionSpec
    specs = (P("data", "tensor"),) * 5 + (P("tensor"),) * 3
    with jax.default_device(cpu):
        operands = inputs(batch if decode else 32, slots=batch + 2)
    operands = tuple(
        jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))
        for x, spec in zip(operands, specs)
    )

    metadata = LinearRecurrentAttnBackendMetadata(
        recurrent_indices=(
            np.arange(1, batch + 1, dtype=np.int32)
            if decode
            else np.array([1] + [0] * (batch - 1), np.int32)
        ),
        has_initial_state=np.zeros(batch, bool),
        cu_q_lens=(
            np.arange(batch + 1, dtype=np.int32)
            if decode
            else np.array([0] + [5] * batch, np.int32)
        ),
    )
    metadata = jax.device_put(metadata, jax.sharding.NamedSharding(mesh, P("data")))

    def forward(metadata, *args):
        backend = make_backend(cpu, metadata)
        backend.mesh = mesh
        if decode:
            return backend.forward_decode(*args)
        return backend.forward_extend(*args, seq_lens=None)

    # Check JAX's explicit-sharding rules without executing the TT FFI on CPU.
    with jax.set_mesh(mesh):
        result = jax.eval_shape(forward, metadata, *operands)
    assert result[1].shape == operands[1].shape
    assert result[2].shape == operands[2].shape


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize(
    "heads,batch",
    [(heads, batch) for heads in [(2, 4), (16, 32), (20, 40)] for batch in [1, 4]] + [((2, 4), 32)],
)
def test_device_state_handoff(tt_device, trace, heads, batch):
    """Real kernels: warmup, replay, chunk continuation, slot reuse and padding."""
    cpu = jax.devices("cpu")[0]

    def compile_forward(decode):
        def forward(metadata, *args):
            backend = make_backend(tt_device, metadata, *heads)
            if decode:
                return backend.forward_decode(*args)
            return backend.forward_extend(*args, seq_lens=None)

        return jax.jit(
            forward,
            donate_argnums=(2, 3),
            compiler_options={"optimization_level": "1", "enable_trace": str(trace).lower()},
        )

    compiled = {decode: compile_forward(decode) for decode in (False, True)}

    def to_device(tree):
        return jax.tree.map(lambda x: jax.device_put(np.asarray(x), tt_device), tree)

    slots = batch + 2
    with jax.default_device(cpu):
        base = inputs(1, *heads, slots=slots)
    host_states, device_states = base[1:3], to_device(base[1:3])
    previous_device_states = tuple(np.asarray(x, dtype=np.float32) for x in host_states)
    weights = to_device(base[5:])
    if batch == 1:
        # (decode, live length, slot, has initial state)
        cases = [(False, 5, 1, False)] + [(True, 1, 1, True)] * 4
        cases += [(False, 17, 1, True)] + [(True, 1, 1, True)] * 4
        cases += [(False, 31, 2, False), (True, 1, 2, True), (True, 1, 0, False)]
        cases += [(False, 47, 2, True), (True, 1, 2, True), (False, 64, 1, False)]
        cases = [(decode, [length], [slot], [initial]) for decode, length, slot, initial in cases]
    elif batch == 4:
        # Ragged prefill, chunk continuation, slot reuse, and interspersed dummies.
        cases = [(False, [5, 2, 17, 0], [4, 2, 1, 0], [False] * 4)]
        cases += [(True, [1] * 4, [4, 2, 0, 1], [True, True, False, True])] * 4
        cases += [(False, [7, 25, 1, 0], [1, 4, 2, 0], [True, True, False, False])]
        cases += [(True, [1] * 4, [2, 4, 1, 0], [True, True, True, False])] * 4
        cases += [(False, [0, 17, 0, 1], [0, 3, 0, 2], [False] * 4)]
        cases += [(True, [1] * 4, [3, 2, 1, 4], [True] * 4)]
        cases += [(False, [65, 2, 0, 161], [1, 2, 0, 4], [True, True, False, True])]
        cases += [(True, [1] * 4, [1, 4, 3, 0], [True, True, True, False])]
    elif batch == 32:
        indices = list(range(1, batch + 1))
        cases = [(False, [1] * batch, indices, [False] * batch)]
        cases += [(True, [1] * batch, indices, [True] * batch)] * 4
    for step, (decode, lengths, indices, initial) in enumerate(cases):
        length = sum(lengths)
        with jax.default_device(cpu):
            count = batch if decode else (length + 31) // 32 * 32
            sample = inputs(count, *heads, seed=35 + step, slots=slots)
            host = (sample[0], *host_states, *sample[3:5], *base[5:])
            metadata = TTGDNMetadata(
                cu_q_lens=jnp.asarray(np.cumsum([0, *lengths]), jnp.int32),
                recurrent_indices=jnp.array(indices, jnp.int32),
                has_initial_state=jnp.array(initial),
                max_prefill_len=1 << (max(max(lengths), 32) - 1).bit_length(),
            )
            expected = reference(host, metadata, *heads, decode=decode)
        dynamic = to_device((sample[0], *sample[3:5]))
        actual = compiled[decode](
            to_device(metadata),
            dynamic[0],
            *device_states,
            *dynamic[1:],
            *weights,
        )
        actual_host = jax.tree.map(lambda x: np.asarray(x, dtype=np.float32), actual)
        for i, (result, wanted) in enumerate(zip(actual_host, expected)):
            wanted = np.asarray(wanted, dtype=np.float32)
            if i == 0:
                # Dummy-request and padded output is not part of the contract.
                live = np.repeat(np.asarray(indices) != 0, lengths)
                result, wanted = result[:length][live], wanted[:length][live]
            np.testing.assert_allclose(
                result,
                wanted,
                rtol=0.06,
                atol=3e-5 if i == 0 else 1e-3,
                err_msg=f"trace={trace}, step={step}, decode={decode}, output={i}",
            )
            if i > 0:
                untouched = [s for s in range(slots) if s == 0 or s not in indices]
                np.testing.assert_array_equal(
                    result[untouched], previous_device_states[i - 1][untouched]
                )
        previous_device_states = actual_host[1:]
        host_states, device_states = expected[1:], actual[1:]


@pytest.mark.parametrize("clear", [False, True])
def test_device_pool_layers_are_independent(tt_device, clear):
    from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

    mesh = jax.sharding.Mesh(np.array([[tt_device]]), ("data", "tensor"))
    pool = RecurrentStatePool([0, 1], 1, 4, 128, 4, mesh, num_k_heads=2)
    if clear:
        pool.clear()
    update = jax.jit(
        ops.state_pool_update, donate_argnums=(0,), compiler_options={"enable_trace": "false"}
    )
    indices = jax.device_put(np.array([1], np.int32), tt_device)
    for buffers in (pool.recurrent_buffers, [x[0] for x in pool.conv_buffers]):
        values = jax.device_put(
            np.ones((1, *buffers[0].shape[1:]), np.dtype(buffers[0].dtype)), tt_device
        )
        changed = update(buffers[0], indices, values)
        np.testing.assert_array_equal(np.asarray(changed)[1], 1)
        np.testing.assert_array_equal(np.asarray(buffers[1]), 0)
