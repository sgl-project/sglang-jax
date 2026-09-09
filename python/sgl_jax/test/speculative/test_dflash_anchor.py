"""Exercise the real draft JIT with a small position-encoding backbone on CPU."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
from sgl_jax.srt.speculative.dflash_info import DFlashVerifyInput
from sgl_jax.srt.speculative.dflash_worker import DFlashWorker


@jax.tree_util.register_dataclass
@dataclass
class Backend:
    forward_metadata: object
    page_size: int = field(default=4, metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass
class Batch:
    input_ids: object
    positions: object
    seq_lens: object
    out_cache_loc: object
    spec_info: object
    attn_backend: object
    input_embedding: object = None


class PositionBackbone(nnx.Module):
    def __call__(self, batch, pools, logits_metadata):
        # A position t predicts token t+1, independently of embedding values.
        hidden = jax.nn.one_hot(batch.positions + 1, 16)
        updates = {
            "query_positions": batch.positions,
            "query_cache": batch.out_cache_loc,
            "query_lens": batch.attn_backend.forward_metadata.cu_q_lens,
            "kv_lens": batch.attn_backend.forward_metadata.seq_lens,
            "pages": batch.attn_backend.forward_metadata.page_indices,
            "spec_width": jnp.asarray(batch.spec_info.draft_token_num),
        }
        return SimpleNamespace(hidden_states=hidden), updates, [], None


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("relay", [False, True])
def test_draft_jit_keeps_verify_layout_and_uses_requested_query_layout(
    monkeypatch, anchor, relay, dp_size
):
    from sgl_jax.srt.speculative import dflash_worker

    # Isolate relay lookup; run the actual draft JIT and metadata repacking.
    monkeypatch.setattr(
        dflash_worker, "gather_dflash_relay_buffers", lambda buffers, *a, **k: buffers
    )
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    with jax.set_mesh(mesh):

        def put(x):
            return jax.device_put(jnp.asarray(x, dtype=jnp.int32), NamedSharding(mesh, P("data")))

        model_def, state = nnx.split(PositionBackbone())
        leaves, state_def = jax.tree_util.tree_flatten(state)
        runner = SimpleNamespace(_model_def=model_def, _model_state_def=state_def, mesh=mesh)
        worker = object.__new__(DFlashWorker)
        worker._worker = SimpleNamespace(model_runner=runner)
        worker.block_size = 4
        worker.sample_from_anchor = anchor
        worker.draft_query_tokens = 3 if anchor else 4
        worker._target_vocab_size = 16
        worker._mask_token_id = 15
        worker._init_jit_draft_block()

        prefix = put([1, 2])
        positions = put([1, 2, 3, 4, 2, 3, 4, 5])
        ids = put([1, 15, 15, 15, 2, 15, 15, 15])
        allocated = prefix + (8 if relay else 4)
        pages = put([10, 11, 12, 20, 21, 22, 0, 0] if relay else [10, 11, 20, 21, 0, 0, 0, 0])
        if dp_size == 2:
            pages = put([10, 11, 12, 0, 20, 21, 22, 0] if relay else [10, 11, 0, 0, 20, 21, 0, 0])
        verify_cu_q = [0, 4, 8] if dp_size == 1 else [0, 4, 0, 4]
        metadata = FlashAttentionMetadata(
            cu_q_lens=put(verify_cu_q),
            cu_kv_lens=put([0, 8, 16]),
            page_indices=pages,
            seq_lens=prefix + 4,
            distribution=put([0, 2, 2]),
        )
        batch = Batch(
            ids,
            positions,
            prefix,
            put([4, 5, 6, 7, 8, 9, 10, 11]),
            DFlashVerifyInput(ids, 4),
            Backend(metadata),
        )
        embed = jax.device_put(jnp.eye(16), NamedSharding(mesh, P("data", "tensor")))
        result = worker._jit_draft_block(
            leaves,
            batch,
            None,
            embed,
            embed,
            (put([1, 2]), prefix + 1),
            put([0, 1]),
            put([1, 1]).astype(jnp.bool_),
            allocated,
            prefix,
            use_relay_state=relay,
            dp_size=dp_size,
        )
        updates, candidates, resolved_prefix, verify_positions, verify_cache = result
        q = 3 if anchor else 4
        np.testing.assert_array_equal(
            updates["query_positions"], np.array(positions).reshape(2, 4)[:, :q].reshape(-1)
        )
        np.testing.assert_array_equal(
            updates["query_cache"],
            np.array([4, 5, 6, 7, 8, 9, 10, 11]).reshape(2, 4)[:, :q].reshape(-1),
        )
        np.testing.assert_array_equal(
            updates["query_lens"], [0, q, 2 * q] if dp_size == 1 else [0, q, 0, q]
        )
        np.testing.assert_array_equal(updates["kv_lens"], np.array([1, 2]) + q)
        assert int(updates["spec_width"]) == q
        np.testing.assert_array_equal(verify_positions, positions)
        np.testing.assert_array_equal(verify_cache, [4, 5, 6, 7, 8, 9, 10, 11])
        np.testing.assert_array_equal(resolved_prefix, prefix)
        np.testing.assert_array_equal(
            candidates, [1, 2, 3, 4, 2, 3, 4, 5] if anchor else [1, 3, 4, 5, 2, 4, 5, 6]
        )
        if anchor:
            if dp_size == 1:
                np.testing.assert_array_equal(updates["pages"][:3], [10, 20, 21])
            else:
                np.testing.assert_array_equal(updates["pages"], [10, 0, 0, 0, 20, 21, 0, 0])
        # The target's original metadata must not be mutated by tracing the draft.
        np.testing.assert_array_equal(batch.attn_backend.forward_metadata.cu_q_lens, verify_cu_q)
