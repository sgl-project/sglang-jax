import jax
import jax.numpy as jnp

from sgl_jax.srt.layers.gmm.megablox_gmm_backend import gmm


def test_gmm_with_padding():
    """Test GMM behavior when sum(group_sizes) != lhs.shape[0]"""

    # Case 1: sum(group_sizes) == lhs.shape[0] (expected behavior)
    print("=== Case 1: Matching sizes ===")
    lhs1 = jnp.ones((30, 128), dtype=jnp.bfloat16)  # 30 tokens
    rhs1 = jnp.ones((4, 128, 256), dtype=jnp.bfloat16)  # 4 experts
    group_sizes1 = jnp.array([10, 8, 7, 5], dtype=jnp.int32)  # sum = 30
    print(f"lhs shape: {lhs1.shape}")
    print(f"group_sizes: {group_sizes1}, sum: {jnp.sum(group_sizes1)}")

    try:
        out1 = gmm(lhs1, rhs1, group_sizes1, preferred_element_type=jnp.bfloat16)
        print(f"Output shape: {out1.shape}")
        print(f"Success!\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Case 2: sum(group_sizes) < lhs.shape[0] (has padding - original bug)
    print("=== Case 2: Has padding (sum < lhs size) ===")
    lhs2 = jnp.ones((100, 128), dtype=jnp.bfloat16)  # 100 tokens (with padding)
    # But only 30 are valid
    lhs2 = lhs2.at[30:].set(0.0)  # Zero out padding
    rhs2 = jnp.ones((4, 128, 256), dtype=jnp.bfloat16)
    group_sizes2 = jnp.array([10, 8, 7, 5], dtype=jnp.int32)  # sum = 30 < 100
    print(f"lhs shape: {lhs2.shape}")
    print(f"group_sizes: {group_sizes2}, sum: {jnp.sum(group_sizes2)}")

    try:
        out2 = gmm(lhs2, rhs2, group_sizes2, preferred_element_type=jnp.bfloat16)
        print(f"Output shape: {out2.shape}")
        print(f"Output[0:5]: {out2[0:5, 0]}")
        print(f"Output[30:35] (padding region): {out2[30:35, 0]}")
        print(f"Success (but may be incorrect)!\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Case 3: Add padding group to make sum == lhs.shape[0] (our fix)
    print("=== Case 3: Add padding group (our fix) ===")
    lhs3 = jnp.ones((100, 128), dtype=jnp.bfloat16)
    lhs3 = lhs3.at[30:].set(0.0)  # Zero out padding
    rhs3_pad = jnp.zeros((1, 128, 256), dtype=jnp.bfloat16)  # Dummy expert
    rhs3 = jnp.concatenate([rhs2, rhs3_pad], axis=0)  # 5 experts now
    group_sizes3 = jnp.array([10, 8, 7, 5, 70], dtype=jnp.int32)  # sum = 100
    print(f"lhs shape: {lhs3.shape}")
    print(f"group_sizes: {group_sizes3}, sum: {jnp.sum(group_sizes3)}")

    try:
        out3 = gmm(lhs3, rhs3, group_sizes3, preferred_element_type=jnp.bfloat16)
        print(f"Output shape: {out3.shape}")
        print(f"Output[0:5]: {out3[0:5, 0]}")
        print(f"Output[30:35] (padding region): {out3[30:35, 0]}")
        print(f"Success!\n")
    except Exception as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    test_gmm_with_padding()
