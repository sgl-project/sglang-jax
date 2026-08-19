"""Tests for K3's AttnRes depth protocol, against a numpy transcription of KimiDecoderLayer.forward."""
import jax.numpy as jnp, numpy as np, pytest
from sgl_jax.srt.models.kimi_k3_residual import (
    initial_block_residuals, residual_state_transition, n_candidates_at_depth)
from sgl_jax.srt.models.kimi_k3_layers import attention_residual_apply


def _run_depth_numpy(n_layers, block_size, hidden, attn_out, nscale, pk, eps):
    """Transcription of the reference forward's residual bookkeeping over n_layers."""
    toks = attn_out.shape[1]
    br = np.zeros((toks, 0, hidden), np.float32)
    prefix = np.zeros((toks, hidden), np.float32)
    seen = []
    for i in range(n_layers):
        if br.shape[-2] > 0:
            seen.append(br.shape[-2])          # candidates AttnRes#1 sees
        else:
            seen.append(0)
        if i % block_size == 0:
            br = np.concatenate((br, prefix[:, None, :]), axis=-2)
            prefix = None
        a = attn_out[i]
        prefix = a if prefix is None else prefix + a
    return br, prefix, seen


def test_checkpoint_resets_prefix_sum():
    """At a boundary the running sum must RESTART (None sentinel), not carry across."""
    hidden, toks = 8, 3
    br = initial_block_residuals(toks, hidden, jnp.float32)
    ps = jnp.ones((toks, hidden), jnp.float32)
    br2, ps2 = residual_state_transition(0, 4, ps, br)      # 0 % 4 == 0 -> checkpoint
    assert ps2 is None, "prefix_sum must be reset at a checkpoint boundary"
    assert br2.shape[-2] == 1, br2.shape
    br3, ps3 = residual_state_transition(1, 4, ps, br2)     # not a boundary
    assert ps3 is not None and br3.shape[-2] == 1


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_candidate_count_matches_reference_bookkeeping(block_size):
    n_layers, hidden, toks = 16, 8, 2
    rng = np.random.default_rng(0)
    attn = rng.normal(size=(n_layers, toks, hidden)).astype(np.float32)
    br, _, _ = _run_depth_numpy(n_layers, block_size, hidden, attn, None, None, 1e-6)
    # after n_layers, the reference has checkpointed once per boundary
    assert br.shape[-2] == sum(1 for i in range(n_layers) if i % block_size == 0)
    # and our shape oracle agrees at every depth
    for d in range(n_layers):
        assert n_candidates_at_depth(d, block_size) == d // block_size + 1


def test_state_transition_matches_reference_over_full_depth():
    """Drive the JAX transition and the numpy reference in lockstep; states must agree."""
    n_layers, block_size, hidden, toks = 12, 3, 8, 2
    rng = np.random.default_rng(1)
    attn = rng.normal(size=(n_layers, toks, hidden)).astype(np.float32)

    br_j = initial_block_residuals(toks, hidden, jnp.float32)
    ps_j = jnp.zeros((toks, hidden), jnp.float32)
    br_n = np.zeros((toks, 0, hidden), np.float32)
    ps_n = np.zeros((toks, hidden), np.float32)

    for i in range(n_layers):
        br_j, ps_j = residual_state_transition(i, block_size, ps_j, br_j)
        if i % block_size == 0:
            br_n = np.concatenate((br_n, ps_n[:, None, :]), axis=-2); ps_n = None
        a = attn[i]
        ps_j = jnp.asarray(a) if ps_j is None else ps_j + jnp.asarray(a)
        ps_n = a if ps_n is None else ps_n + a
        np.testing.assert_allclose(np.asarray(br_j), br_n, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(ps_j), ps_n, rtol=1e-6, atol=1e-6)


def test_attnres_consumes_the_growing_candidate_axis():
    """AttnRes must accept the depth-dependent candidate count without broadcasting silently."""
    hidden, toks, eps = 16, 4, 1e-6
    rng = np.random.default_rng(2)
    nscale = jnp.asarray(rng.normal(size=(hidden,)).astype(np.float32))
    pk = jnp.asarray((rng.normal(size=(hidden, 1)) * 0.05).astype(np.float32))
    for n_cand in (1, 2, 5):
        br = jnp.asarray(rng.normal(size=(toks, n_cand, hidden)).astype(np.float32))
        ps = jnp.asarray(rng.normal(size=(toks, hidden)).astype(np.float32))
        out = attention_residual_apply(ps, br, nscale, pk, eps)
        assert out.shape == (toks, hidden), (n_cand, out.shape)
