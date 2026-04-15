import argparse
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# Copy from https://docs.jax.dev/en/latest/pallas/tpu/matmul.html
def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@functools.partial(
    jax.jit, static_argnames=["bm", "bk", "bn", "reduction_axis_dimension_semantics"]
)
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    reduction_axis_dimension_semantics: str = "parallel",
):
    M, K = x.shape
    _, N = y.shape
    pad_m = (bm - (M % bm)) % bm
    pad_k = (bk - (K % bk)) % bk
    pad_n = (bn - (N % bn)) % bn

    x_padded = jnp.pad(x, ((0, pad_m), (0, pad_k)))
    y_padded = jnp.pad(y, ((0, pad_k), (0, pad_n)))

    M_padded, K_padded = x_padded.shape
    _, N_padded = y_padded.shape

    grid_m = M_padded // bm
    grid_n = N_padded // bn
    grid_k = K_padded // bk
    n_steps = grid_k

    return pl.pallas_call(
        functools.partial(matmul_kernel, nsteps=n_steps),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(grid_m, grid_n, grid_k),
        ),
        out_shape=jax.ShapeDtypeStruct((M_padded, N_padded), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", reduction_axis_dimension_semantics)
        ),
    )(x_padded, y_padded)


# --- Testing Code ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test matmul with Pallas")
    parser.add_argument(
        "--reduction-axis-dimension-semantics",
        type=str,
        default="parallel",
        help="Dimension semantics for reduction axis (default: parallel)",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="simple",
        choices=["simple", "full"],
    )
    parser.add_argument(
        "--block-configs-mode",
        type=str,
        default="fixed",
        choices=["fixed", "mixed"],
    )
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)

    mode_to_block_configs = {
        "fixed": ([128], [128], [128]),
        "mixed": ([8, 128, 256, 1024], [128, 256, 512, 1024], [256, 512, 1024, 1152]),
    }
    mode_to_MNK = {
        "full": (
            [1, 2, 4, 5, 8, 16, 32, 64, 100, 128, 256, 500, 1024, 2000, 4096, 8192],  # M
            [1024, 2048, 4096],  # K,
            [1024, 8192, 151936],  # N
        ),
        "simple": (
            [1, 2, 4, 6, 8, 16, 32, 64, 128, 256],
            [4096],
            [151936],
        ),
    }

    bm_list, bk_list, bn_list = mode_to_block_configs[args.block_configs_mode]
    M_list, K_list, N_list = mode_to_MNK[args.data_mode]
    batch_size_list = M_list

    block_configs = list(zip(bm_list, bk_list, bn_list))

    print("Starting matmul tests with parameter sweep...")
    print(f"Block configurations: {len(block_configs)}")
    total_test_cases_num = len(K_list) * len(N_list) * len(M_list) * len(block_configs)
    print(
        f"Total test cases: {len(K_list)} * {len(N_list)} * {len(M_list)} * {len(block_configs)} = {total_test_cases_num}"
    )

    results_by_kn = {}

    print(f"Begin to generate K_input_no_padding and KN_weights, ")
    K_input_no_padding = {
        K: jax.random.normal(key, shape=(1, K), dtype=jnp.bfloat16) for K in K_list
    }
    KN_weights = {
        (K, N): jax.random.normal(key, shape=(K, N), dtype=jnp.bfloat16)
        for K, N in itertools.product(K_list, N_list)
    }
    print(
        f"Complete generating K_input_no_padding and KN_weights, {len(K_input_no_padding)=}, {len(KN_weights)=}",
        flush=True,
    )

    # Generate all combinations
    all_configs = itertools.product(K_list, N_list, M_list, block_configs)

    count = 0
    KN_failed_count = {(K, N): 0 for K, N in itertools.product(K_list, N_list)}
    for K, N, M, (bm, bk, bn) in all_configs:
        count += 1
        print(
            f"[{count}/{total_test_cases_num}]Testing: K={K}, N={N}, M={M}, bm={bm}, bk={bk}, bn={bn}",
            flush=True,
        )

        input_no_padding = K_input_no_padding[K]
        weight_matrix = KN_weights[(K, N)]

        if M == 1:
            input_matrix = input_no_padding
        else:
            input_matrix_tmp = jax.random.normal(key, shape=(M - 1, K), dtype=jnp.bfloat16)
            input_matrix = jnp.concat([input_no_padding, input_matrix_tmp], axis=0)

        try:
            # Run matmul
            result = matmul(
                input_matrix,
                weight_matrix,
                bm=bm,
                bk=bk,
                bn=bn,
                reduction_axis_dimension_semantics=args.reduction_axis_dimension_semantics,
            )
            result_cpu = jax.device_get(result)

            # Extract first row (result[0])
            extracted_res = result_cpu[0, :N].reshape(-1)

            # Store result with nested tuple structure
            kn_key = (K, N)
            if kn_key not in results_by_kn:
                results_by_kn[kn_key] = {}

            m_block_key = (M, (bm, bk, bn))
            results_by_kn[kn_key][m_block_key] = extracted_res

            print(f"✓ Completed: {result.shape=}, {extracted_res.shape=}", flush=True)

        except Exception as e:
            print(f"✗ Failed with error: {e}", flush=True)
            KN_failed_count[(K, N)] += 1
            continue

    print(f"====================================={KN_failed_count=}\n", flush=True)

    # Compare results varying across M, bm, bk, bn for each (K, N)
    print("\n" + "=" * 80)
    print("Comparing Results Across M and Block Configurations", flush=True)
    print("=" * 80 + "\n")

    passed_count = 0
    failed_count = 0

    for kn_key, m_block_results in results_by_kn.items():
        K, N = kn_key

        if len(m_block_results) < 2:
            continue

        print(f"\nConfiguration: K={K}, N={N}", flush=True)
        print(f"  Comparing {len(m_block_results)} combinations of M and (bm, bk, bn)", flush=True)

        # Compare all pairs
        m_block_keys = list(m_block_results.keys())
        for i in range(len(m_block_keys)):
            for j in range(i + 1, len(m_block_keys)):
                key_i = m_block_keys[i]
                key_j = m_block_keys[j]
                M_i, (bm_i, bk_i, bn_i) = key_i
                M_j, (bm_j, bk_j, bn_j) = key_j
                result_i = m_block_results[key_i]
                result_j = m_block_results[key_j]

                try:
                    np.testing.assert_array_equal(result_i, result_j)
                    print(
                        f"  \033[92m✓ M={M_i},({bm_i},{bk_i},{bn_i}) vs M={M_j},({bm_j},{bk_j},{bn_j}): PASSED\033[0m",
                        flush=True,
                    )
                    passed_count += 1
                except AssertionError:
                    print(
                        f"  \033[91m✗ M={M_i},({bm_i},{bk_i},{bn_i}) vs M={M_j},({bm_j},{bk_j},{bn_j}): FAILED\033[0m",
                        flush=True,
                    )
                    # Print first few differences for debugging
                    diff_mask = result_i != result_j
                    num_diffs = np.sum(diff_mask)
                    print(f"    Number of differences: {num_diffs}/{len(result_i)}", flush=True)
                    failed_count += 1

    print("\n" + "=" * 80)
    print(f"Summary: {passed_count} passed, {failed_count} failed")
    print("=" * 80)
    print("\nDone.")
