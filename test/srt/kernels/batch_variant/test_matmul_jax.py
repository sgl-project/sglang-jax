# This test is based on https://github.com/jax-ml/jax/issues/34080. But just simplify it.
import jax
import jax.numpy as jnp
import numpy as np

key = jax.random.PRNGKey(0)
input_no_padding = jax.random.normal(key, shape=(1, 4096), dtype=jnp.bfloat16)
weight = jax.random.normal(key, shape=(4096, 151936), dtype=jnp.bfloat16)


#batch_size_list = [1, 2, 4, 6, 8, 16, 32, 64, 128, 256]
#batch_size_list = [1, 2, 4, 5, 8, 16, 32, 64, 100, 128, 256, 500, 1024, 2000, 4096, 8192, 10000, 35423]
#batch_size_list = [1,2,4,16]
batch_size_list = [1]
batch_size_result = []

print(f"Begin to prepare data",flush=True)

for bs in batch_size_list:
    input_with_padding_tmp = jax.random.normal(key, shape=(bs - 1, 4096), dtype=jnp.bfloat16)
    input_with_padding = jnp.concat([input_no_padding, input_with_padding_tmp], axis=0)
    result_with_padding = jnp.dot(input_with_padding, weight, preferred_element_type=jnp.float32)
    extracted_res = jax.device_get(result_with_padding[0, ...]).reshape(-1)
    batch_size_result.append(extracted_res)

print(f"Complete preparing data",flush=True)

# Compare all batch_size_results with each other
print("\n" + "=" * 60)
print("Comparing batch_size_results across all batch sizes")
print("=" * 60 + "\n")

for i in range(len(batch_size_list)):
    for j in range(i + 1, len(batch_size_list)):
        bs_i = batch_size_list[i]
        bs_j = batch_size_list[j]
        result_i = batch_size_result[i]
        result_j = batch_size_result[j]

        try:
            np.testing.assert_array_equal(result_i, result_j)
            print(f"\033[92m✓ batch_size {bs_i} vs {bs_j}: PASSED\033[0m", flush=True)
        except AssertionError as e:
            print(f"\033[91m✗ batch_size {bs_i} vs {bs_j}: FAILED\033[0m", flush=True)
            print(f"{e=}")


print("\n" + "=" * 60)
print("Batch comparison complete")
print("=" * 60 + "\n")
