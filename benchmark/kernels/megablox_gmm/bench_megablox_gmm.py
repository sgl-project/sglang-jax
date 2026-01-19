import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.test.test_utils import CustomTestCase, is_in_ci


def create_gmm_test_data(
    m: int,
    k: int,
    n: int,
    num_groups: int,
    group_sizes: jax.Array,
    dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 42,
):
    """Create test data for megablox gmm benchmark."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)

    lhs = jax.random.normal(keys[0], (m, k), dtype=dtype)
    rhs = jax.random.normal(keys[1], (num_groups, k, n), dtype=dtype)

    return lhs, rhs


def benchmark_backend(
    m: int,
    k: int,
    n: int,
    num_groups: int,
    group_sizes: jax.Array,
    backend_type: str = "megablox",
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] = (128, 128, 128),
    dtype: jnp.dtype = jnp.bfloat16,
):
    """Benchmark megablox gmm with given parameters."""

    if backend_type == "megablox":
        lhs, rhs = create_gmm_test_data(m, k, n, num_groups, group_sizes, dtype)

        @functools.partial(
            jax.jit,
            static_argnames=[
                "preferred_element_type",
                "tiling",
            ],
        )
        def jitted_gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type,
            tiling,
        ):
            return gmm(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type=preferred_element_type,
                tiling=tiling,
            )

        gmm_fn = functools.partial(
            jitted_gmm,
            lhs,
            rhs,
            group_sizes,
            preferred_element_type,
            tiling,
        )
    else:
        raise ValueError(f"Invalid backend type: {backend_type}")

    # Benchmark
    # warm up
    out = gmm_fn()
    jax.block_until_ready(out)

    # start benchmark
    times = []
    for i in range(3):
        start = time.perf_counter()
        output = gmm_fn()
        jax.block_until_ready(output)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    return avg_time


def create_uniform_group_sizes(num_groups: int, group_size: int) -> jax.Array:
    """Create uniform group sizes array."""
    return jnp.array([group_size] * num_groups, dtype=jnp.int32)


def full_benchmark():
    """Main benchmark function."""
    # Configuration ranges
    m_config = [512, 1024, 2048, 4096]
    k_config = [512, 1024, 2048, 4096]
    n_config = [512, 1024, 2048, 4096]
    num_groups_config = [2, 4, 8, 16]
    group_size_config = [128, 256, 512]

    print("MEGABLOX GMM BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    results = []
    config_count = 0
    valid_config_count = 0

    for m in m_config:
        for k in k_config:
            for n in n_config:
                for num_groups in num_groups_config:
                    for group_size in group_size_config:
                        config_count += 1

                        # Make sure m is divisible by group_size * num_groups for consistent benchmarking
                        total_required_m = group_size * num_groups
                        if m < total_required_m:
                            # Skip this configuration if m is too small
                            continue

                        # Adjust m to be exactly divisible
                        adjusted_m = (m // total_required_m) * total_required_m
                        if adjusted_m == 0:
                            continue

                        valid_config_count += 1
                        group_sizes = create_uniform_group_sizes(
                            num_groups, adjusted_m // num_groups
                        )

                        print(
                            f"Config {valid_config_count}: m={adjusted_m}, k={k}, n={n}, groups={num_groups}, group_size={adjusted_m // num_groups}"
                        )

                        try:
                            megablox_time = benchmark_backend(
                                adjusted_m,
                                k,
                                n,
                                num_groups,
                                group_sizes,
                                backend_type="megablox",
                            )

                            results.append(
                                {
                                    "config": f"M{adjusted_m}_K{k}_N{n}_G{num_groups}",
                                    "megablox_ms": megablox_time * 1000,
                                    "m": adjusted_m,
                                    "k": k,
                                    "n": n,
                                    "num_groups": num_groups,
                                }
                            )

                            print(f"  Time: {megablox_time * 1000:.2f} ms")

                        except Exception as e:
                            print(f"  ERROR: {e}")

                        print()

    print("=" * 80)
    print("SUMMARY OF ALL RESULTS")
    print("-" * 80)
    print(f"{'Config':<25} {'Time (ms)':<12} {'M':<6} {'K':<6} {'N':<6} {'Groups':<8}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['config']:<25} {r['megablox_ms']:<12.2f} {r['m']:<6} {r['k']:<6} {r['n']:<6} {r['num_groups']:<8}"
        )

    # Find best and worst performing configs
    if results:
        best_config = min(results, key=lambda x: x["megablox_ms"])
        worst_config = max(results, key=lambda x: x["megablox_ms"])

        print("-" * 80)
        print(f"Best performance:  {best_config['config']} - {best_config['megablox_ms']:.2f} ms")
        print(f"Worst performance: {worst_config['config']} - {worst_config['megablox_ms']:.2f} ms")
        print(f"Speedup ratio: {worst_config['megablox_ms'] / best_config['megablox_ms']:.2f}x")


class TestPerformance(CustomTestCase):
    def test_megablox_gmm_performance(self, floating_threshold: int = 0.15):
        """
        Args:
            floating_threshold: the ratio of expected results
        """
        # Key: (m, k, n, num_groups)
        # Value: expected cost-time (baseline) in ms
        test_cases = {
            (1024, 1024, 4096, 16): 1.9645499999872604,
            (1024, 4096, 4096, 16): 7.200843333369751,
            (4096, 1024, 4096, 16): 3.762219999998706,
            (4096, 4096, 1024, 8): 3.689633333351594,
            (4096, 4096, 4096, 16): 14.21690999995917,
        }

        for case, baseline in test_cases.items():
            m, k, n, num_groups = case
            group_sizes = create_uniform_group_sizes(num_groups, m // num_groups)
            megablox_time = benchmark_backend(
                m,
                k,
                n,
                num_groups,
                group_sizes,
                backend_type="megablox",
            )
            expected_result = baseline * (1 + floating_threshold)
            print(f"{case}, res={megablox_time*1000}ms, {expected_result=}ms")
            self.assertLess(
                megablox_time * 1000,
                expected_result,
                f"Run ragged_paged_attention performance test failed, {case=}",
            )


if __name__ == "__main__":
    if is_in_ci():
        print("Run megablox gmm performance test...")
        TestPerformance().test_megablox_gmm_performance()
    else:
        print("Run megablox gmm full benchmark...")
        full_benchmark
