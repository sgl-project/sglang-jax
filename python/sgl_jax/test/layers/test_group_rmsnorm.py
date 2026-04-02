import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import nn

from sgl_jax.srt.layers.attention.fla.layernorm_gated import GroupRMSNorm


@pytest.fixture(autouse=True, scope="session")
def _enforce_cpu():
    """Ensure both JAX and PyTorch run on CPU."""
    jax_backend = jax.default_backend()
    torch_device = torch.tensor(0).device
    print(f"\n[unit test device] JAX backend: {jax_backend}, PyTorch device: {torch_device}")
    assert jax_backend == "cpu", f"Expected JAX backend 'cpu', got '{jax_backend}'"
    assert torch.device("cpu") == torch_device, "PyTorch default device is not CPU"


HIDDEN_SIZE = 8192
NUM_GROUPS = 8
GROUP_SIZE = HIDDEN_SIZE // NUM_GROUPS  # 1024
EPSILON = 1e-6
SEED = 42
BATCH_SIZE = 4
SEQ_LEN = 8

# (200 trials, 3x safety margin)
# Config: HIDDEN_SIZE=8192, NUM_GROUPS=8, GROUP_SIZE=1024
FP32_ATOL = 8.6e-6
FP32_RTOL = 1.4e-6

# bf16: fp32 normalize → cast bf16 → weight mul
# (200 trials, 3x safety margin)
BF16_ATOL = 9.9e-2
BF16_RTOL = 2.4e-2

# Theoretical single-framework error bound for fp64 ground truth verification
# Computation graph: square(1) → sum(GROUP_SIZE) → div(1) → rsqrt(1) → mul(1) → weight_mul(1)
_eps_fp32 = float(jnp.finfo(jnp.float32).eps)  # 2^-23
_n_ops = GROUP_SIZE + 5
_single_framework_bound = _n_ops * _eps_fp32


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


def _make_torch_model(hidden_size=HIDDEN_SIZE, num_groups=NUM_GROUPS, weight=None):
    """Create a PyTorch GroupRMSNorm model, optionally with custom weight."""
    model = BailingMoeV2_5GroupRMSNorm(hidden_size, num_groups, eps=EPSILON)
    if weight is not None:
        model.weight = torch.nn.Parameter(torch.from_numpy(weight))
    return model


def _run_jax(model, input_np, dtype=jnp.float32):
    """Run JAX model and return numpy array."""
    return np.array(model(jnp.array(input_np, dtype=dtype)))


def _run_torch(model, input_np, dtype=torch.float32, device="cpu"):
    """Run PyTorch model and return numpy array."""
    model = model.to(device)
    return model(torch.tensor(input_np, dtype=dtype, device=device)).detach().float().cpu().numpy()


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

    _DTYPE_PARAMS = [
        (jnp.float32, torch.float32, FP32_ATOL, FP32_RTOL),
        (jnp.bfloat16, torch.bfloat16, BF16_ATOL, BF16_RTOL),
    ]
    _DTYPE_IDS = ["fp32", "bf16"]

    @pytest.mark.parametrize(
        ("jax_dtype", "torch_dtype", "atol", "rtol"), _DTYPE_PARAMS, ids=_DTYPE_IDS
    )
    def test_values_match_with_default_weight(self, jax_dtype, torch_dtype, atol, rtol):
        """JAX and PyTorch must produce identical output with default weights (ones)."""
        rng = np.random.default_rng(SEED)
        input_data = _make_input(rng, (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))

        jax_output = _run_jax(_make_jax_model(), input_data, jax_dtype)
        torch_output = _run_torch(_make_torch_model(), input_data, torch_dtype)

        np.testing.assert_allclose(jax_output, torch_output, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("jax_dtype", "torch_dtype", "atol", "rtol"), _DTYPE_PARAMS, ids=_DTYPE_IDS
    )
    def test_values_match_with_random_weight(self, jax_dtype, torch_dtype, atol, rtol):
        """JAX and PyTorch must produce identical output with random weights."""
        rng = np.random.default_rng(123)
        input_data = _make_input(rng, (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))
        weight = _make_weight(rng)

        jax_output = _run_jax(_make_jax_model(weight=weight), input_data, jax_dtype)
        torch_output = _run_torch(_make_torch_model(weight=weight), input_data, torch_dtype)

        np.testing.assert_allclose(jax_output, torch_output, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("jax_dtype", "torch_dtype", "atol", "rtol"), _DTYPE_PARAMS, ids=_DTYPE_IDS
    )
    def test_values_match_with_2d_input(self, jax_dtype, torch_dtype, atol, rtol):
        """Cross-framework consistency with 2D input (batch, hidden)."""
        rng = np.random.default_rng(99)
        input_data = _make_input(rng, (BATCH_SIZE * 4, HIDDEN_SIZE))
        weight = _make_weight(rng)

        jax_output = _run_jax(_make_jax_model(weight=weight), input_data, jax_dtype)
        torch_output = _run_torch(_make_torch_model(weight=weight), input_data, torch_dtype)

        assert jax_output.shape == torch_output.shape
        np.testing.assert_allclose(jax_output, torch_output, atol=atol, rtol=rtol)


class TestErrorBound:
    """Verify each framework stays within theoretical error bound vs fp64 ground truth."""

    def test_error_bound_verification(self):
        """Both JAX and PyTorch must stay within theoretical relative error bound."""
        max_abs_jax = 0.0
        max_abs_torch = 0.0
        max_rel_jax = 0.0
        max_rel_torch = 0.0
        sum_abs_jax = 0.0
        sum_abs_torch = 0.0
        sum_rel_jax = 0.0
        sum_rel_torch = 0.0
        n_trials = 100

        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            input_data = _make_input(rng, (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))
            weight = _make_weight(rng)

            fp64_out = _numpy_group_rmsnorm_fp64(input_data, weight, NUM_GROUPS, EPSILON)
            jax_out = _run_jax(_make_jax_model(weight=weight), input_data, jnp.float32).astype(
                np.float64
            )
            torch_out = _run_torch(
                _make_torch_model(weight=weight), input_data, torch.float32
            ).astype(np.float64)

            # Absolute error
            abs_jax = np.max(np.abs(jax_out - fp64_out))
            abs_torch = np.max(np.abs(torch_out - fp64_out))
            max_abs_jax = max(max_abs_jax, abs_jax)
            max_abs_torch = max(max_abs_torch, abs_torch)
            sum_abs_jax += abs_jax
            sum_abs_torch += abs_torch

            # Relative error: |impl - truth| / |truth|
            nonzero = np.abs(fp64_out) > 1e-12
            rel_jax = np.max(
                np.abs(jax_out[nonzero] - fp64_out[nonzero]) / np.abs(fp64_out[nonzero])
            )
            rel_torch = np.max(
                np.abs(torch_out[nonzero] - fp64_out[nonzero]) / np.abs(fp64_out[nonzero])
            )
            max_rel_jax = max(max_rel_jax, rel_jax)
            max_rel_torch = max(max_rel_torch, rel_torch)
            sum_rel_jax += rel_jax
            sum_rel_torch += rel_torch

        print(f"\n{'=' * 50}")
        print(f"  Error bound verification ({n_trials} trials)")
        print(f"{'=' * 50}")
        print(f"  theoretical bound:   {_single_framework_bound:.6e}")
        print(f"  JAX  max  abs err:   {max_abs_jax:.6e}")
        print(f"  JAX  mean abs err:   {sum_abs_jax / n_trials:.6e}")
        print(f"  JAX  max  rel err:   {max_rel_jax:.6e}")
        print(f"  JAX  mean rel err:   {sum_rel_jax / n_trials:.6e}")
        print(f"  Torch max  abs err:  {max_abs_torch:.6e}")
        print(f"  Torch mean abs err:  {sum_abs_torch / n_trials:.6e}")
        print(f"  Torch max  rel err:  {max_rel_torch:.6e}")
        print(f"  Torch mean rel err:  {sum_rel_torch / n_trials:.6e}")
        print(f"{'=' * 50}")

        np.testing.assert_array_less(
            max_rel_jax,
            _single_framework_bound,
            err_msg=f"JAX exceeded bound: {max_rel_jax:.6e}",
        )
        np.testing.assert_array_less(
            max_rel_torch,
            _single_framework_bound,
            err_msg=f"Torch exceeded bound: {max_rel_torch:.6e}",
        )
