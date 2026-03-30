import jax.numpy as jnp
import numpy as np
import torch
from torch import nn

from sgl_jax.srt.layers.layernorm import GroupRMSNorm

HIDDEN_SIZE = 128
NUM_GROUPS = 8
EPSILON = 1e-6
SEED = 42
BATCH_SIZE = 2
SEQ_LEN = 16
RTOL = 1e-6


class BailingMoeV2_5GroupRMSNorm(nn.Module):
    def __init__(self, hidden_size, group_norm_size, eps=1e-6):
        """
        BailingMoeV2_5RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_norm_size = group_norm_size
        assert (
            hidden_size % group_norm_size == 0
        ), "hidden_size must be divisible by group_norm_size"
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.size()
        group_input_shape = input_shape[:-1] + (
            self.group_norm_size,
            input_shape[-1] // self.group_norm_size,
        )
        hidden_states = hidden_states.view(group_input_shape)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype).view(input_shape)


def _numpy_group_rmsnorm(hidden_states, weight, num_groups, eps):
    """Numpy reference implementation of GroupRMSNorm."""
    orig_shape = hidden_states.shape
    grouped = hidden_states.reshape(*orig_shape[:-1], num_groups, orig_shape[-1] // num_groups)
    variance = np.mean(grouped**2, axis=-1, keepdims=True)
    normalized = grouped / np.sqrt(variance + eps)
    return weight * normalized.reshape(orig_shape)


def _make_input(rng, shape):
    """Generate random float32 input data."""
    return rng.standard_normal(shape).astype(np.float32)


def _make_weight(rng):
    """Generate random float32 weight vector."""
    return rng.standard_normal(HIDDEN_SIZE).astype(np.float32)


def _make_jax_model(weight=None):
    """Create a JAX GroupRMSNorm model, optionally with custom weight."""
    model = GroupRMSNorm(HIDDEN_SIZE, num_groups=NUM_GROUPS, epsilon=EPSILON)
    if weight is not None:
        model.weight[...] = jnp.array(weight)
    return model


def _make_torch_model(weight=None):
    """Create a PyTorch GroupRMSNorm model, optionally with custom weight."""
    model = BailingMoeV2_5GroupRMSNorm(HIDDEN_SIZE, NUM_GROUPS, eps=EPSILON)
    if weight is not None:
        model.weight = torch.nn.Parameter(torch.from_numpy(weight))
    return model


def _run_jax(model, input_np):
    """Run JAX model and return numpy array."""
    return np.array(model(jnp.array(input_np)))


def _run_torch(model, input_np):
    """Run PyTorch model and return numpy array."""
    return model(torch.from_numpy(input_np)).detach().numpy()


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
        group_size = HIDDEN_SIZE // NUM_GROUPS

        input_original = _make_input(rng, (1, 1, HIDDEN_SIZE))
        input_modified = input_original.copy()
        input_modified[..., :group_size] = _make_input(rng, (group_size,))  # perturb group 0 only

        model = _make_jax_model()
        output_original = _run_jax(model, input_original)
        output_modified = _run_jax(model, input_modified)

        # Groups 1~7 should be identical.
        np.testing.assert_allclose(
            output_original[..., group_size:],
            output_modified[..., group_size:],
            rtol=RTOL,
        )
        # Group 0 should differ.
        assert not np.allclose(
            output_original[..., :group_size],
            output_modified[..., :group_size],
        )

    def test_weight_participates_in_computation(self):
        """Weight parameter must participate in computation correctly."""
        rng = np.random.default_rng(SEED)
        input_data = _make_input(rng, (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))
        weight = _make_weight(rng)

        model = _make_jax_model(weight)
        jax_output = _run_jax(model, input_data)
        expected = _numpy_group_rmsnorm(input_data, weight, NUM_GROUPS, EPSILON)

        np.testing.assert_allclose(jax_output, expected, rtol=RTOL)


class TestCrossFramework:
    """Cross-framework consistency tests between JAX and PyTorch."""

    def test_output_shapes_match(self):
        """JAX and PyTorch output shapes must both equal input shape."""
        rng = np.random.default_rng(SEED)
        input_data = _make_input(rng, (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))

        jax_model = _make_jax_model()
        torch_model = _make_torch_model()
        jax_output = _run_jax(jax_model, input_data)
        torch_output = _run_torch(torch_model, input_data)

        assert jax_output.shape == input_data.shape
        assert torch_output.shape == input_data.shape
        assert jax_output.shape == torch_output.shape

    def test_values_match_with_default_weight(self):
        """JAX and PyTorch must produce identical output with default weights (ones)."""
        rng = np.random.default_rng(SEED)
        input_data = _make_input(rng, (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))

        jax_output = _run_jax(_make_jax_model(), input_data)
        torch_output = _run_torch(_make_torch_model(), input_data)

        np.testing.assert_allclose(jax_output, torch_output, rtol=RTOL)

    def test_values_match_with_random_weight(self):
        """JAX and PyTorch must produce identical output with random weights."""
        rng = np.random.default_rng(123)
        input_data = _make_input(rng, (BATCH_SIZE * 2, SEQ_LEN // 2, HIDDEN_SIZE))
        weight = _make_weight(rng)

        jax_output = _run_jax(_make_jax_model(weight), input_data)
        torch_output = _run_torch(_make_torch_model(weight), input_data)

        np.testing.assert_allclose(jax_output, torch_output, rtol=RTOL)

    def test_values_match_with_2d_input(self):
        """Cross-framework consistency with 2D input (batch, hidden)."""
        rng = np.random.default_rng(99)
        input_data = _make_input(rng, (BATCH_SIZE * 4, HIDDEN_SIZE))
        weight = _make_weight(rng)

        jax_output = _run_jax(_make_jax_model(weight), input_data)
        torch_output = _run_torch(_make_torch_model(weight), input_data)

        assert jax_output.shape == torch_output.shape
        np.testing.assert_allclose(jax_output, torch_output, rtol=RTOL)

    def test_precision_report(self):
        """Report precision difference between JAX and PyTorch (no assertion)."""
        rng = np.random.default_rng(123)
        input_data = _make_input(rng, (BATCH_SIZE * 2, SEQ_LEN // 2, HIDDEN_SIZE))
        weight = _make_weight(rng)

        jax_output = _run_jax(_make_jax_model(weight), input_data)
        torch_output = _run_torch(_make_torch_model(weight), input_data)

        abs_diff = np.abs(jax_output - torch_output)
        print(f"\n{'='*50}")
        print("  Precision diff: JAX vs PyTorch (float32)")
        print(f"{'='*50}")
        print(f"  max  abs diff: {abs_diff.max():.6e}")
        print(f"  mean abs diff: {abs_diff.mean():.6e}")
        print(f"  min  abs diff: {abs_diff.min():.6e}")
        print(f"{'='*50}")
        assert abs_diff.max() < RTOL, f"max abs diff {abs_diff.max():.6e} exceeds {RTOL} threshold"
