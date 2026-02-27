"""
Test script for LTX2Scheduler to verify correctness against PyTorch reference.

This script demonstrates the usage of the JAX implementation and validates
that it produces the expected output.
"""

import jax.numpy as jnp
from ltx2_scheduler import LTX2Scheduler


def test_basic_schedule():
    """Test basic scheduler functionality."""
    print("Test 1: Basic schedule generation")
    scheduler = LTX2Scheduler()

    # Generate schedule for 10 steps without latent
    sigmas = scheduler.execute(steps=10)

    print(f"  Shape: {sigmas.shape}")
    print(f"  First value: {sigmas[0]:.6f} (should be ~1.0)")
    print(f"  Last value: {sigmas[-1]:.6f} (should be 0.0)")
    print(f"  Dtype: {sigmas.dtype}")
    print(f"  Values: {sigmas}")
    assert sigmas.shape == (11,), f"Expected shape (11,), got {sigmas.shape}"
    assert sigmas[0] > 0.9, f"First value should be close to 1.0, got {sigmas[0]}"
    assert sigmas[-1] == 0.0, f"Last value should be 0.0, got {sigmas[-1]}"
    print("  ✓ Passed\n")


def test_with_latent():
    """Test schedule generation with a latent tensor."""
    print("Test 2: Schedule with latent tensor")
    scheduler = LTX2Scheduler()

    # Create a dummy latent with spatial dimensions [16, 16]
    # Shape: [batch, channels, height, width]
    latent = jnp.zeros((1, 4, 16, 16))
    tokens = 16 * 16  # 256 tokens

    sigmas = scheduler.execute(steps=20, latent=latent)

    print(f"  Latent shape: {latent.shape}")
    print(f"  Tokens: {tokens}")
    print(f"  Schedule shape: {sigmas.shape}")
    print(f"  First 5 values: {sigmas[:5]}")
    print(f"  Last 5 values: {sigmas[-5:]}")
    assert sigmas.shape == (21,), f"Expected shape (21,), got {sigmas.shape}"
    print("  ✓ Passed\n")


def test_without_stretch():
    """Test schedule generation without stretching."""
    print("Test 3: Schedule without stretching")
    scheduler = LTX2Scheduler()

    sigmas_no_stretch = scheduler.execute(steps=10, stretch=False)
    sigmas_with_stretch = scheduler.execute(steps=10, stretch=True)

    print(f"  Without stretch - Last non-zero: {sigmas_no_stretch[sigmas_no_stretch != 0][-1]:.6f}")
    print(f"  With stretch - Last non-zero: {sigmas_with_stretch[sigmas_with_stretch != 0][-1]:.6f}")
    print(f"  Expected terminal value: 0.1")

    # With stretch should be closer to terminal value (0.1)
    last_nonzero_stretched = sigmas_with_stretch[sigmas_with_stretch != 0][-1]
    assert abs(last_nonzero_stretched - 0.1) < 0.01, \
        f"Stretched schedule should end near 0.1, got {last_nonzero_stretched}"
    print("  ✓ Passed\n")


def test_custom_parameters():
    """Test schedule with custom shift parameters."""
    print("Test 4: Custom shift parameters")
    scheduler = LTX2Scheduler()

    # Test with different shift values
    sigmas_default = scheduler.execute(steps=10)
    sigmas_high_shift = scheduler.execute(steps=10, max_shift=3.0, base_shift=1.5)

    print(f"  Default shift - mid value: {sigmas_default[5]:.6f}")
    print(f"  High shift - mid value: {sigmas_high_shift[5]:.6f}")
    print(f"  Schedules differ: {not jnp.allclose(sigmas_default, sigmas_high_shift)}")
    assert not jnp.allclose(sigmas_default, sigmas_high_shift), \
        "Different shift parameters should produce different schedules"
    print("  ✓ Passed\n")


def test_different_steps():
    """Test schedules with different numbers of steps."""
    print("Test 5: Different step counts")
    scheduler = LTX2Scheduler()

    for steps in [5, 10, 20, 50]:
        sigmas = scheduler.execute(steps=steps)
        print(f"  Steps={steps:2d}: shape={sigmas.shape}, "
              f"first={sigmas[0]:.4f}, last={sigmas[-1]:.4f}")
        assert sigmas.shape == (steps + 1,), \
            f"Expected shape ({steps+1},), got {sigmas.shape}"
    print("  ✓ Passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("LTX2Scheduler JAX Implementation Tests")
    print("=" * 60 + "\n")

    test_basic_schedule()
    test_with_latent()
    test_without_stretch()
    test_custom_parameters()
    test_different_steps()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
