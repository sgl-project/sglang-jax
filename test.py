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