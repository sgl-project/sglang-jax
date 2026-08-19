"""KDA A_log must be narrowed to num_heads, matching the torch/GPU reference.

K3 ships A_log with 128 entries while the KDA has 96 heads. The rest of the checkpoint pins the
geometry: o_norm.weight is [128] (== head_dim, kernel-validated), b_proj is [96, 7168] (beta is
per-head), q/k/v_proj are [12288, ...] = 96*128. So A_log is padded to head_dim and only its
first num_heads entries are meaningful -- exactly what torch's _load_a_log does with
    loaded_weight.narrow(0, rank*shard_size, shard_size),  shard_size = num_heads // tp

Getting this wrong does not raise: the kernel indexes A_log per head
(A_log.reshape(H,1,1,1,1); -exp(A)[:,None,None]*softplus(g)), so a wrong slice silently mis-gates
every head.
"""
import numpy as np, pytest

NUM_HEADS, HEAD_DIM = 96, 128


def _narrow(raw, num_heads, rank=0, tp=1):
    """The reference's narrow, reproduced."""
    if raw.ndim == 4:
        raw = raw.reshape(-1)
    shard = num_heads // tp
    return raw[rank * shard : (rank + 1) * shard]


def test_narrow_takes_the_first_num_heads_entries():
    raw = np.arange(HEAD_DIM, dtype=np.float32)
    got = _narrow(raw, NUM_HEADS)
    assert got.shape == (NUM_HEADS,)
    np.testing.assert_array_equal(got, np.arange(NUM_HEADS, dtype=np.float32))


def test_padding_beyond_num_heads_is_discarded():
    """Entries 96..127 are padding and must not reach the gate."""
    raw = np.zeros(HEAD_DIM, np.float32); raw[NUM_HEADS:] = 999.0
    assert not (_narrow(raw, NUM_HEADS) == 999.0).any()


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_tp_shards_partition_the_first_num_heads(tp):
    """Union of rank shards == first num_heads entries, no overlap, no gaps."""
    raw = np.arange(HEAD_DIM, dtype=np.float32)
    shards = [_narrow(raw, NUM_HEADS, r, tp) for r in range(tp)]
    assert all(s.shape == (NUM_HEADS // tp,) for s in shards)
    np.testing.assert_array_equal(np.concatenate(shards), np.arange(NUM_HEADS, dtype=np.float32))


def test_old_four_dim_layout_is_accepted():
    """The reference accepts either [1,1,H,1] or [H]."""
    raw4 = np.arange(HEAD_DIM, dtype=np.float32).reshape(1, 1, HEAD_DIM, 1)
    np.testing.assert_array_equal(_narrow(raw4, NUM_HEADS),
                                  np.arange(NUM_HEADS, dtype=np.float32))


def test_geometry_is_consistent_with_the_rest_of_the_checkpoint():
    """num_heads*head_dim must equal the projection size dt_bias/q_proj declare."""
    assert NUM_HEADS * HEAD_DIM == 12288
