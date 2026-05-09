import unittest

'''
TODO: test KDAAttnBackend with dp in sgl_jax/srt/layers/attention/linear/kda_backend.py
    test structure: sgl_jax/test/test_flashattention_dp.py
    ref short conv: ./test_short_conv.py
    ref kda kernel: _forward_extend_naive
    compare kda kernel: _forward_extend_pallas

NOTE: test_flashattention_dp.py copies unique_in_original_order but never calls it
    (dead code). Use cache_loc[::page_size] for page_table construction.
'''

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


if __name__ == "__main__":
    unittest.main()
