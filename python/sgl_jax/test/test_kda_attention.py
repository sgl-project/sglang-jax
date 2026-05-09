import unittest

'''
TODO: test KDAAttnBackend with dp in sgl_jax/srt/layers/attention/linear/kda_backend.py
    test structure: sgl_jax/test/test_flashattention.py
    ref short conv: ./test_short_conv.py
    ref kda kernel: _forward_extend_naive
    compare kda kernel: _forward_extend_pallas
    memory pool

NOTE: test_flashattention.py uses unique_in_original_order(cache_loc // page_size) to
    build page_table, but since slots are linearly allocated, cache_loc[::page_size]
    suffices (as test_flashattention_dp.py does). Use the stride approach here.
'''

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


if __name__ == "__main__":
    unittest.main()
