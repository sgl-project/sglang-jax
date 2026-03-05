try:
    import pathwaysutils
    pathwaysutils.initialize()
    print("In Pathways")
except Exception as e:
    print(f"Not in Pathways")

import jax
import time

y = jax.numpy.ones((1000, 1000))
print(f"y locates on {y.device}")
start_y = time.perf_counter()
y_cpu=jax.device_get(y)
print(f"Use device_get: {(time.perf_counter()-start_y)*1e3:.3f}ms")

# Create array on device
x = jax.numpy.ones((1000, 1000))

# Time the async initiation (should be fast)
start = time.perf_counter()
x._arrays[0]._copy_single_device_array_to_host_async()
async_time = time.perf_counter() - start
print(f"Async initiation: {async_time*1e3:.3f}ms")

# Time the actual blocking wait
start = time.perf_counter()
result = x._value  # This blocks until data arrives
blocking_time = time.perf_counter() - start
print(f"Blocking wait: {blocking_time*1e3:.3f}ms")

