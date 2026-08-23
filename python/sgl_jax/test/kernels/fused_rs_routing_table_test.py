import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.fused_moe.fused_rs.gmm_fused_rs_nodedup import (
    _build_packed_index_tile_table,
    _build_packed_index_tile_table_pallas,
)


@pytest.mark.parametrize("tile_m", [128, 256, 384])
def test_pallas_index_tile_table_matches_jax_reference(tile_m):
    group_sizes = jnp.asarray([65, 511, 0, 256], dtype=jnp.int32)
    packed = jnp.arange(int(group_sizes.sum()), dtype=jnp.int32) * 7 + 3
    kwargs = {
        "num_local_groups": 2,
        "tile_m": tile_m,
        "size_lhs_sublane": 8,
        "max_num_gm": 8,
    }
    expected = _build_packed_index_tile_table(
        packed,
        group_sizes,
        jnp.asarray([0], dtype=jnp.int32),
        **kwargs,
    )
    actual = _build_packed_index_tile_table_pallas(
        packed,
        group_sizes,
        jnp.asarray([0], dtype=jnp.int32),
        interpret=True,
        **kwargs,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
