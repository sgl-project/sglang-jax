import sys

sys.path.insert(0, ".")
import time
from concurrent.futures import ThreadPoolExecutor

import loader as L
import ml_dtypes
import numpy as np


class T(L.EngramTable):
    def __init__(self, R=200000, D=256):
        rng = np.random.default_rng(0)
        self.w8 = rng.integers(0, 256, (R, D), dtype=np.uint8)
        self.s8 = rng.integers(100, 130, (R, D // 32), dtype=np.uint8)


t = T()
pool = ThreadPoolExecutor(8)
rng = np.random.default_rng(1)
for shape in [(16, 512, 16), (4, 512, 16), (7, 5, 16), (3, 1, 16)]:
    ids = rng.integers(0, 200000, shape)
    a1, b1 = t.raw(ids)
    a2, b2 = t.raw_threaded(ids, pool)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2) and a1.shape == a2.shape, shape
ids = rng.integers(0, 200000, (16, 512, 16))
t0 = time.time()
[t.raw(ids) for _ in range(3)]
t1 = time.time()
[t.raw_threaded(ids, pool) for _ in range(3)]
t2 = time.time()
print(
    f"identical for all shapes | single {1000*(t1-t0)/3:.0f} ms vs threaded {1000*(t2-t1)/3:.0f} ms (16x512 tokens, one layer, in-RAM table)"
)
