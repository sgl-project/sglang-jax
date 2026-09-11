"""End-to-end forward tests for the Gemma4 FROZEN_KV_MTP draft model.

These drive the *real* pieces the speculative loop depends on — a real
``MHATokenToKVPool``, the native attention backend, the real ``LogitsProcessor``
— rather than shapes alone, because the defects this guards against are all
invisible at construction time:

* the draft's dummy zero K/V reaching the target's shared KV pool
* the draft's captured hidden leaving in the wrong vector space
* an empty aux-hidden list crashing the target's verify forward

A full Scheduler run needs real checkpoints and TPU kernels, so these stop at
the model/pool boundary, which is exactly where the frozen-KV contract lives.

Run on CPU:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1 \\
      python -m pytest python/sgl_jax/test/speculative/test_gemma4_mtp_e2e.py -v
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.layers.kv_share import compute_mtp_kv_share_map
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.models.gemma4_mtp import Gemma4AssistantForCausalLM
from sgl_jax.srt.speculative.base_worker import replicate_to_mesh
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

# Pinned to a single device rather than create_device_mesh's "use everything":
# these tests assert on KV-pool contents and hidden-state shapes, not on
# sharding, and a mesh whose width depends on XLA_FLAGS makes them pass alone
# but fail inside the wider suite.
MESH = jax.sharding.Mesh(
    np.asarray(jax.devices()[:1]).reshape(1, 1),
    ("data", "tensor"),
    axis_types=(jax.sharding.AxisType.Explicit,) * 2,
)

# Snapshotted from google/gemma-4-12B-it{,-assistant}, dims shrunk for CPU
# (real value in the comment). Relationships are preserved, not the magnitudes.
#
# Unlike the unit tests, every layer here is full_attention: MHATokenToKVPool
# has one head geometry for all layers, whereas a real hybrid target uses
# SWAKVPool to give sliding and full layers different widths. These tests are
# about the frozen-KV contract and the hidden-state space, not per-type
# geometry (test_gemma4_mtp.py covers that), so the single-geometry pool is the
# cheaper fixture. The values below are therefore the *global* (full-attention)
# ones from the real config.
BACKBONE_HIDDEN = 384  # real 3840 = the target's hidden_size
DRAFT_HIDDEN = 128  # real 1024; must stay != BACKBONE_HIDDEN
# 128-aligned: merge_kv pads head_dim up to 128, so an unaligned head_dim makes
# the pool wider than the attention layer and o_proj mismatches. The real
# global_head_dim (512) is already aligned.
HEAD_DIM = 128  # real global_head_dim 512
NUM_HEADS = 2  # real 16
NUM_KV_HEADS = 1  # real num_global_key_value_heads 1 -- verbatim
VOCAB = 256  # real 262144
NUM_DRAFT_LAYERS = 2  # real 4
# The target has more layers than the draft — that gap is what makes a
# positional KV write-back land on the wrong layers. Real ratio is 48:4.
NUM_TARGET_LAYERS = 6  # real 48
POOL_SIZE = 128
DTYPE = jnp.float32
# merge_kv aligns head_dim up to 128, so the pool is allocated at the aligned
# width even though attention runs at HEAD_DIM. Production does the same thing
# in ModelRunnerKVCacheMixin (`head_dim=(head_dim + 127) // 128 * 128`).
POOL_HEAD_DIM = (HEAD_DIM + 127) // 128 * 128


def test_target_verify_compacts_reserved_pages_between_requests():
    """TARGET_VERIFY must not interpret request A's reserve page as request B's KV.

    ``padding_for_decode`` packs the source cache by ``allocate_lens``.  Here,
    each request owns two physical pages but target verification needs only its
    first page.  The ``cu_kv_lens`` rows therefore require the compact page
    sequence ``[A0, B0]``.  Without target-verify compaction the backend passes
    ``[A0, A1, B0, B1]`` through unchanged, so B reads A's reserve page whenever
    both requests are in one batch.
    """
    page_size = 8
    backend = FlashAttention(1, 1, 8, page_size=page_size, mesh=MESH)

    def _page(page_id: int) -> np.ndarray:
        return np.arange(page_id * page_size, (page_id + 1) * page_size, dtype=np.int32)

    # cache_loc is in the same allocation-packed form produced by
    # EagleDraftWorker.padding_for_decode: [A0, A1, B0, B1].
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        cache_loc=np.concatenate([_page(10), _page(11), _page(20), _page(21)]),
        seq_lens=np.array([4, 4], dtype=np.int32),  # prepare_for_verify already subtracted 1
        dp_size=1,
        per_dp_bs_size=2,
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        spec_info_padded=SimpleNamespace(
            custom_mask=None,
            draft_token_num=4,
            allocate_lens=np.array([9, 9], dtype=np.int32),
        ),
    )

    metadata = backend.get_eagle_forward_metadata(batch)
    # Verification length is seq_len + 4 == 8, i.e. one page per request.
    # The first two physical page IDs consumed by its two cu_kv rows must be
    # A0 then B0, not A0 then A1.
    np.testing.assert_array_equal(np.asarray(metadata.page_indices)[:2], [10, 20])


def test_frozen_draft_steps_remap_target_pages_to_swa_pool():
    """Every recurrent draft step must address the target's SWA sub-pool."""
    page_size = 8
    backend = FlashAttention(1, 1, 8, page_size=page_size, mesh=MESH)
    mapping = np.zeros(1024, dtype=np.int32)
    mapping[10 * page_size] = 100 * page_size
    mapping[20 * page_size] = 200 * page_size
    backend.swa_index_mapping = mapping

    def _page(page_id: int) -> np.ndarray:
        return np.arange(page_id * page_size, (page_id + 1) * page_size, dtype=np.int32)

    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        cache_loc=np.concatenate([_page(10), _page(11), _page(20), _page(21)]),
        seq_lens=np.array([4, 4], dtype=np.int32),
        dp_size=1,
        per_dp_bs_size=2,
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
        spec_info_padded=EagleDraftInput(allocate_lens=np.array([9, 9], dtype=np.int32)),
    )

    metadata = backend.get_eagle_multi_step_metadata(batch)

    assert len(metadata) == 3
    for step_metadata in metadata:
        np.testing.assert_array_equal(
            np.asarray(step_metadata.swa_page_indices)[:2],
            np.array([100, 200], dtype=np.int32),
        )


def _draft_config() -> PretrainedConfig:
    return PretrainedConfig(
        hidden_size=DRAFT_HIDDEN,
        intermediate_size=2 * DRAFT_HIDDEN,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        swa_head_dim=HEAD_DIM,
        vocab_size=VOCAB,
        num_hidden_layers=NUM_DRAFT_LAYERS,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=10000.0,
        layer_types=["full_attention"] * NUM_DRAFT_LAYERS,
        sliding_window=0,
        tie_word_embeddings=True,
        backbone_hidden_size=BACKBONE_HIDDEN,
        use_ordered_embeddings=False,
    )


def _target_config() -> PretrainedConfig:
    return PretrainedConfig(
        layer_types=["full_attention"] * NUM_TARGET_LAYERS,
        num_kv_shared_layers=0,
        num_hidden_layers=NUM_TARGET_LAYERS,
    )


def _make_pool(seed: int = 0) -> MHATokenToKVPool:
    """A target-shaped KV pool pre-filled with a distinctive per-layer pattern.

    Non-zero, layer-distinguishable contents are the point: zero-filled buffers
    would make a clobbering write indistinguishable from a correct no-op.
    """
    with jax.set_mesh(MESH):
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=1,
            dtype=DTYPE,
            head_num=NUM_KV_HEADS,
            head_dim=POOL_HEAD_DIM,
            layer_num=NUM_TARGET_LAYERS,
            mesh=MESH,
        )
        for layer_id in range(NUM_TARGET_LAYERS):
            buf = pool.kv_buffer[layer_id]
            pool.kv_buffer[layer_id] = jnp.full_like(buf, float(seed + layer_id + 1))
    return pool


def _make_model() -> Gemma4AssistantForCausalLM:
    with jax.set_mesh(MESH):
        model = Gemma4AssistantForCausalLM(config=_draft_config(), mesh=MESH, dtype=DTYPE)
        # The draft embeds tokens with the TARGET's embedding (backbone-dim).
        model.set_embed_and_head(
            nnx.Param(
                jax.random.normal(jax.random.PRNGKey(1), (VOCAB, BACKBONE_HIDDEN), dtype=DTYPE)
                * 0.02
            ),
            None,
        )
    return model


def _redirect_layer_ids(model: Gemma4AssistantForCausalLM) -> dict[str, str]:
    """Apply the same layer_id redirection MultiLayerDraftWorker._init_gemma4_mtp does."""
    share_map = compute_mtp_kv_share_map(_draft_config(), _target_config())
    for i, layer in enumerate(model.layers):
        target_idx = int(share_map[f"draft_layer.{i}"].split(".")[-1])
        layer.self_attn.attn.layer_id = target_idx
    return share_map


def _make_forward_batch(num_tokens: int, seq_len: int, hidden: jax.Array) -> ForwardBatch:
    """An EXTEND batch for a single request whose prefix is already in the pool."""
    backend = NativeAttention(num_attn_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, mesh=MESH)
    spec_info = EagleDraftInput(hidden_states=hidden)
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=jnp.arange(num_tokens, dtype=jnp.int32) % VOCAB,
        req_pool_indices=jnp.zeros((1,), dtype=jnp.int32),
        seq_lens=jnp.array([seq_len], dtype=jnp.int32),
        out_cache_loc=jnp.arange(seq_len - num_tokens, seq_len, dtype=jnp.int32),
        positions=jnp.arange(seq_len - num_tokens, seq_len, dtype=jnp.int32),
        attn_backend=backend,
        cache_loc=jnp.arange(seq_len, dtype=jnp.int32),
        extend_prefix_lens=jnp.array([seq_len - num_tokens], dtype=jnp.int32),
        extend_seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
        spec_info=spec_info,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )


def _logits_metadata(num_tokens: int) -> LogitsMetadata:
    # The serving path constructs both arrays with P("data") sharding.  JAX
    # 0.10 validates shard_map inputs strictly, so a default-replicated test
    # array is not a faithful CPU stand-in for the production metadata.
    data_sharding = jax.sharding.NamedSharding(MESH, jax.sharding.PartitionSpec("data"))
    return LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        capture_hidden_mode=CaptureHiddenMode.LAST,
        extend_seq_lens=jax.device_put(jnp.array([num_tokens], dtype=jnp.int32), data_sharding),
        logits_indices=jax.device_put(jnp.array([num_tokens - 1], dtype=jnp.int32), data_sharding),
    )


def _run_draft(model, pool, num_tokens=4, seq_len=8, hidden=None):
    if hidden is None:
        hidden = (
            jax.random.normal(jax.random.PRNGKey(2), (num_tokens, BACKBONE_HIDDEN), dtype=DTYPE)
            * 0.1
        )
    fb = _make_forward_batch(num_tokens, seq_len, hidden)
    with jax.set_mesh(MESH):
        return model(fb, MemoryPools(token_to_kv_pool=pool), _logits_metadata(num_tokens))


@pytest.fixture(scope="module")
def wired():
    model = _make_model()
    share_map = _redirect_layer_ids(model)
    return model, share_map


class TestKvShareWiring:
    def test_layer_ids_point_into_the_target_range(self, wired):
        model, share_map = wired
        for i, layer in enumerate(model.layers):
            expected = int(share_map[f"draft_layer.{i}"].split(".")[-1])
            assert layer.self_attn.attn.layer_id == expected
            # Redirected ids must address the TARGET pool, not the draft's own
            # 0..NUM_DRAFT_LAYERS-1 range.
            assert 0 <= layer.self_attn.attn.layer_id < NUM_TARGET_LAYERS

    def test_redirect_actually_leaves_the_draft_index_space(self, wired):
        _, share_map = wired
        targets = {int(v.split(".")[-1]) for v in share_map.values()}
        assert max(targets) >= NUM_DRAFT_LAYERS, (
            "Test is not exercising redirection: every draft layer already maps "
            "to an index inside the draft's own range."
        )


class TestFrozenKvContract:
    """The draft must read the target's cache and never write to it."""

    def test_pool_update_covers_every_target_layer(self, wired):
        """A short list is the positional-write bug: it silently overwrites
        target layers 0..len-1 instead of the layers the draft actually read."""
        model, _ = wired
        pool = _make_pool()
        _, pool_updates, _, _ = _run_draft(model, pool)
        assert len(pool_updates["token_to_kv_pool"]) == NUM_TARGET_LAYERS

    def test_pool_update_is_the_targets_own_buffers(self, wired):
        model, _ = wired
        pool = _make_pool()
        before = [np.asarray(b) for b in pool.kv_buffer]
        _, pool_updates, _, _ = _run_draft(model, pool)
        for layer_id, updated in enumerate(pool_updates["token_to_kv_pool"]):
            np.testing.assert_array_equal(
                np.asarray(updated),
                before[layer_id],
                err_msg=f"draft modified KV for target layer {layer_id}",
            )

    def test_target_kv_survives_replace_all(self, wired):
        """The B1 regression: run the real write-back and assert the target's
        verified KV is bitwise untouched."""
        model, _ = wired
        pool = _make_pool()
        pools = MemoryPools(token_to_kv_pool=pool)
        before = [np.asarray(b) for b in pool.kv_buffer]

        _, pool_updates, _, _ = _run_draft(model, pool)
        pools.replace_all(pool_updates)

        for layer_id in range(NUM_TARGET_LAYERS):
            np.testing.assert_array_equal(
                np.asarray(pool.kv_buffer[layer_id]),
                before[layer_id],
                err_msg=(
                    f"target KV layer {layer_id} changed after a draft forward — "
                    "frozen-KV draft wrote into the shared pool"
                ),
            )

    def test_pool_round_trips_through_a_jitted_donated_call(self, wired):
        """Replays ModelRunner's real dispatch: jax.jit with the pools donated,
        then replace_all() from the returned updates.

        What this pins: the model is jit-compatible with MemoryPools as a
        donated pytree argument, and the refill round-trips to the same buffers.
        The eager tests above never enter jit at all.

        What it does NOT pin, despite the donation: on the CPU backend XLA
        declines to donate ("Some donated buffers were not usable") and the
        inputs survive, so the deleted-buffer failure mode this pass-through
        exists to prevent only reproduces on TPU. The real guard against a
        short/empty update list is test_pool_update_covers_every_target_layer.
        """
        model, _ = wired
        pool = _make_pool()
        pools = MemoryPools(token_to_kv_pool=pool)
        before = [np.asarray(b) for b in pool.kv_buffer]

        graphdef, state = nnx.split(model)
        leaves, state_def = jax.tree_util.tree_flatten(state)

        @partial(jax.jit, donate_argnames=["memory_pools"], static_argnames=["state_def"])
        def jitted(graphdef, state_def, leaves, forward_batch, memory_pools, meta):
            m = nnx.merge(graphdef, jax.tree_util.tree_unflatten(state_def, leaves))
            return m(forward_batch, memory_pools, meta)

        num_tokens = 4
        hidden = jnp.ones((num_tokens, BACKBONE_HIDDEN), dtype=DTYPE) * 0.1
        fb = _make_forward_batch(num_tokens, 8, hidden)
        with jax.set_mesh(MESH):
            _, pool_updates, _, _ = jitted(
                graphdef, state_def, leaves, fb, pools, _logits_metadata(num_tokens)
            )
            pools.replace_all(pool_updates)

        updates = pool_updates["token_to_kv_pool"]
        assert len(updates) == NUM_TARGET_LAYERS, (
            f"jitted draft returned {len(updates)} pool updates for a "
            f"{NUM_TARGET_LAYERS}-layer target; on TPU the unreturned buffers "
            f"are donated away and the pool is left holding deleted arrays"
        )
        for layer_id in range(NUM_TARGET_LAYERS):
            buf = pool.kv_buffer[layer_id]
            assert not buf.is_deleted(), f"target KV layer {layer_id} is deleted"
            # The refill must have installed the returned arrays, not left the
            # pool on its pre-call ones.
            assert buf is updates[layer_id]
            np.testing.assert_array_equal(np.asarray(buf), before[layer_id])

    def test_no_layer_was_zeroed(self, wired):
        """Sharpest form of the clobber: dummy K/V are zeros, the pattern is not."""
        model, _ = wired
        pool = _make_pool()
        pools = MemoryPools(token_to_kv_pool=pool)
        _run_draft(model, pool)
        pools.replace_all(_run_draft(model, pool)[1])
        for layer_id in range(NUM_TARGET_LAYERS):
            buf = np.asarray(pool.kv_buffer[layer_id])
            assert np.count_nonzero(buf) == buf.size, (
                f"target KV layer {layer_id} contains zeros — the draft's dummy "
                "K/V reached the shared pool"
            )


class TestHiddenStateSpace:
    """The captured hidden is the next draft step's input; it must be backbone-dim."""

    def test_output_hidden_is_backbone_dim(self, wired):
        model, _ = wired
        pool = _make_pool()
        output, _, _, _ = _run_draft(model, pool)
        assert output.hidden_states is not None
        assert output.hidden_states.shape[-1] == BACKBONE_HIDDEN, (
            f"captured hidden is {output.hidden_states.shape[-1]}-dim; the next "
            f"draft step concatenates it into pre_projection(2*{BACKBONE_HIDDEN})"
        )

    def test_logits_still_span_the_full_vocab(self, wired):
        model, _ = wired
        pool = _make_pool()
        output, _, _, _ = _run_draft(model, pool)
        assert output.next_token_logits.shape[-1] == VOCAB

    def test_captured_hidden_feeds_the_next_draft_step(self, wired):
        """The B3 regression: step N's output must be a valid step N+1 input.

        This is the multi-step autoregressive loop EagleDraftWorker.draft_forward
        runs; a hidden left in draft space raises on the concatenate.
        """
        model, _ = wired
        pool = _make_pool()
        output, _, _, _ = _run_draft(model, pool, num_tokens=4, seq_len=8)

        # EagleDraftWorker.draft_forward replicates the captured hidden before
        # feeding it to the next step; do the same so this exercises the real
        # sequence rather than a sharding it would never see.
        fed_back = replicate_to_mesh(MESH, output.hidden_states)
        # draft_forward feeds one hidden per active token; broadcast the captured
        # last-token hidden to the next step's token count the same way.
        next_tokens = 4
        if fed_back.shape[0] != next_tokens:
            fed_back = jnp.broadcast_to(
                fed_back.reshape(1, -1)[:1], (next_tokens, fed_back.shape[-1])
            )

        output2, _, _, _ = _run_draft(
            model, pool, num_tokens=next_tokens, seq_len=12, hidden=fed_back
        )
        assert output2.next_token_logits.shape[-1] == VOCAB
        assert output2.hidden_states.shape[-1] == BACKBONE_HIDDEN

    def test_outputs_are_finite(self, wired):
        model, _ = wired
        pool = _make_pool()
        output, _, _, _ = _run_draft(model, pool)
        assert jnp.all(jnp.isfinite(output.next_token_logits))
        assert jnp.all(jnp.isfinite(output.hidden_states))


class TestReadsRedirectedLayer:
    def test_output_depends_on_the_mapped_target_layer(self, wired):
        """Changing the cache the draft was redirected to must change its output.

        If the redirect were ignored the draft would read layer 0 and this would
        produce identical logits.
        """
        model, share_map = wired
        mapped = sorted({int(v.split(".")[-1]) for v in share_map.values()})

        pool_a = _make_pool(seed=0)
        out_a, _, _, _ = _run_draft(model, pool_a)

        pool_b = _make_pool(seed=0)
        with jax.set_mesh(MESH):
            for layer_id in mapped:
                buf = pool_b.kv_buffer[layer_id]
                pool_b.kv_buffer[layer_id] = jnp.full_like(buf, 42.0)
        out_b, _, _, _ = _run_draft(model, pool_b)

        assert not np.allclose(
            np.asarray(out_a.next_token_logits), np.asarray(out_b.next_token_logits)
        ), "draft output ignored the KV-share redirect"

    def test_output_ignores_unmapped_target_layers(self, wired):
        """The complement: layers the draft never reads must not affect it."""
        model, share_map = wired
        mapped = {int(v.split(".")[-1]) for v in share_map.values()}
        unmapped = [i for i in range(NUM_TARGET_LAYERS) if i not in mapped]
        assert unmapped, "test needs at least one unread target layer"

        pool_a = _make_pool(seed=0)
        out_a, _, _, _ = _run_draft(model, pool_a)

        pool_b = _make_pool(seed=0)
        with jax.set_mesh(MESH):
            for layer_id in unmapped:
                buf = pool_b.kv_buffer[layer_id]
                pool_b.kv_buffer[layer_id] = jnp.full_like(buf, -7.0)
        out_b, _, _, _ = _run_draft(model, pool_b)

        np.testing.assert_allclose(
            np.asarray(out_a.next_token_logits),
            np.asarray(out_b.next_token_logits),
            rtol=1e-5,
            atol=1e-5,
            err_msg="draft read a target layer outside its KV-share map",
        )


class TestAuxHiddenStatesCapture:
    """B2: an empty aux list must not crash the target's verify forward."""

    def _processor(self):
        with jax.set_mesh(MESH):
            processor = LogitsProcessor(VOCAB, mesh=MESH)
            head = Embed(
                num_embeddings=VOCAB,
                features=DRAFT_HIDDEN,
                param_dtype=DTYPE,
                dtype=DTYPE,
            )
        return processor, head

    def _run(self, aux, mode):
        processor, head = self._processor()
        hidden = jnp.ones((3, DRAFT_HIDDEN), dtype=DTYPE)
        meta = LogitsMetadata(forward_mode=ForwardMode.DECODE, capture_hidden_mode=mode)
        with jax.set_mesh(MESH):
            return processor(hidden, head, meta, aux_hidden_states=aux)

    @pytest.mark.parametrize("mode", [CaptureHiddenMode.LAST, CaptureHiddenMode.FULL])
    def test_empty_aux_list_falls_back_to_final_hidden(self, mode):
        """Gemma4Model always returns a list, so [] reaches here whenever no
        layer matched. Both capture modes must fall back, not concat([])."""
        out = self._run([], mode)
        assert out.hidden_states is not None
        assert out.hidden_states.shape[-1] == DRAFT_HIDDEN

    def test_none_aux_still_works(self):
        out = self._run(None, CaptureHiddenMode.LAST)
        assert out.hidden_states.shape[-1] == DRAFT_HIDDEN

    def test_populated_aux_is_still_concatenated(self):
        """The guard must not disable real EAGLE3-style aux capture."""
        aux = [jnp.ones((3, DRAFT_HIDDEN), dtype=DTYPE) for _ in range(2)]
        out = self._run(aux, CaptureHiddenMode.LAST)
        assert out.hidden_states.shape[-1] == 2 * DRAFT_HIDDEN
