import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.fla.group_rmsnorm import GroupRMSNorm

HIDDEN_SIZE = 8192
NUM_GROUPS = 8
GROUP_SIZE = HIDDEN_SIZE // NUM_GROUPS  # 1024
EPSILON = 1e-6
SEED = 42
BATCH_SIZE = 4
SEQ_LEN = 8

# Config: HIDDEN_SIZE=8192, NUM_GROUPS=8, GROUP_SIZE=1024
FP32_ATOL = 8.6e-6
FP32_RTOL = 1.4e-6


def _numpy_group_rmsnorm_fp64(hidden_states, weight, num_groups, eps):
    """fp64 ground truth reference implementation."""
    x = hidden_states.astype(np.float64)
    w = weight.astype(np.float64)
    orig_shape = x.shape
    group_size = orig_shape[-1] // num_groups
    x = x.reshape(*orig_shape[:-1], num_groups, group_size)
    variance = np.mean(x**2, axis=-1, keepdims=True)
    x = x / np.sqrt(variance + eps)
    x = x.reshape(orig_shape)
    return w * x


def _make_input(rng, shape):
    """Generate random float32 input data."""
    return rng.standard_normal(shape).astype(np.float32)


def _make_weight(rng, hidden_size=HIDDEN_SIZE):
    """Generate random float32 weight vector."""
    return rng.standard_normal(hidden_size).astype(np.float32)


def _make_jax_model(hidden_size=HIDDEN_SIZE, num_groups=NUM_GROUPS, weight=None):
    """Create a JAX GroupRMSNorm model, optionally with custom weight."""
    model = GroupRMSNorm(hidden_size, num_groups=num_groups, epsilon=EPSILON)
    if weight is not None:
        model.weight[...] = jnp.array(weight)
    return model


def _run_jax(model, input_np, dtype=jnp.float32):
    """Run JAX model and return numpy array."""
    return np.array(model(jnp.array(input_np, dtype=dtype)))


class TestGroupRMSNorm:
    """GroupRMSNorm unit tests for JAX implementation."""

    def test_output_shape_matches_input(self):
        """Output shape must match input shape."""
        rng = np.random.default_rng(SEED)
        input_data = jnp.array(_make_input(rng, (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)))

        model = _make_jax_model()
        output = model(input_data)

        assert output.shape == input_data.shape

    def test_groups_are_independent(self):
        """Modifying one group must not affect other groups' outputs."""
        rng = np.random.default_rng(SEED)

        input_original = _make_input(rng, (1, 1, HIDDEN_SIZE))
        input_modified = input_original.copy()
        input_modified[..., :GROUP_SIZE] = _make_input(rng, (GROUP_SIZE,))  # perturb group 0 only

        model = _make_jax_model()
        output_original = _run_jax(model, input_original)
        output_modified = _run_jax(model, input_modified)

        # Groups 1~7 should be identical.
        np.testing.assert_allclose(
            output_original[..., GROUP_SIZE:],
            output_modified[..., GROUP_SIZE:],
            rtol=FP32_RTOL,
            atol=FP32_ATOL,
        )
        # Group 0 should differ.
        assert not np.allclose(
            output_original[..., :GROUP_SIZE],
            output_modified[..., :GROUP_SIZE],
        )

    def test_weight_participates_in_computation(self):
        """Weight parameter must participate in computation correctly."""
        rng = np.random.default_rng(SEED)
        input_data = _make_input(rng, (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))
        weight = _make_weight(rng)

        model = _make_jax_model(weight=weight)
        jax_output = _run_jax(model, input_data)
        expected = _numpy_group_rmsnorm_fp64(input_data, weight, NUM_GROUPS, EPSILON)

        np.testing.assert_allclose(jax_output, expected, rtol=FP32_RTOL, atol=FP32_ATOL)
