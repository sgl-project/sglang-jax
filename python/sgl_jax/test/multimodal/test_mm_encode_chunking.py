"""L-k OOM guard: tests for the batched-encode patch-budget chunker that decouples the in-model
vision encode's peak HBM from concurrency.

The ViT does dense full attention (activation ~ (Σ patches)^2), so the original batched path
concatenated ALL co-scheduled requests' images into ONE encode jit call and OOM'd under continuous
batching (RESOURCE_EXHAUSTED jit_jitted_encode_mm). The fix chunks the encode so a single jit call
never exceeds a patch bound, running multiple sequential calls when a step is large. Decode/forward
still batches the whole step (untouched throughput); only the encode peak is bounded.

These cover (a) the PURE pieces (no jax / TPU needed): the bound resolution and the deterministic
chunker, and (b) the end-to-end ``_encode_mm_batched`` routing/bit-identity with stubbed
encode/merge jits (jax-on-CPU, no TPU). The e2e tests reproduce the multi-chunk correctness
regression: a chunked batch must produce, for EACH request, the SAME merged embedding as the
single-call path AND each request must receive its OWN image's features (no cross-request misroute,
no vision-dropped placeholders).

Run on the project's env (Python >=3.12):
    python -m pytest python/sgl_jax/test/multimodal/test_mm_encode_chunking.py
"""

import types
import unittest

import numpy as np

from sgl_jax.srt.model_executor.model_runner import (
    _DEFAULT_MAX_PATCHES_PER_ENCODE,
    ModelRunner,
    _chunk_by_patch_budget,
    _resolve_max_patches_per_encode,
)


class _SA:
    """Minimal server_args stand-in."""

    def __init__(self, per_encode=0, max_patches=0):
        self.vision_max_patches_per_encode = per_encode
        self.vision_max_patches = max_patches


class TestResolveBound(unittest.TestCase):
    def test_explicit_wins(self):
        self.assertEqual(
            _resolve_max_patches_per_encode(_SA(per_encode=2048, max_patches=9999)), 2048
        )

    def test_falls_back_to_vision_max_patches(self):
        # No explicit per-encode bound -> use the worst-case single-image count the G1 reserve is
        # already sized for, so one chunk fits the reserved HBM.
        self.assertEqual(_resolve_max_patches_per_encode(_SA(max_patches=8192)), 8192)

    def test_default_when_unset(self):
        self.assertEqual(_resolve_max_patches_per_encode(_SA()), _DEFAULT_MAX_PATCHES_PER_ENCODE)

    def test_always_positive(self):
        self.assertGreater(_resolve_max_patches_per_encode(_SA()), 0)


class TestChunkByPatchBudget(unittest.TestCase):
    def test_single_chunk_when_under_budget(self):
        # 4 images, 1000 patches each, bound 8192 -> one chunk == the single-call path (no chunking,
        # so the chunked path is bit-identical to before by construction).
        chunks = _chunk_by_patch_budget([1000, 1000, 1000, 1000], 8192)
        self.assertEqual(chunks, [(0, 4)])

    def test_splits_when_over_budget(self):
        # 8 images of 4000 patches, bound 8192 -> 2 per chunk (8000 <= 8192, +4000 would be 12000).
        chunks = _chunk_by_patch_budget([4000] * 8, 8192)
        self.assertEqual(chunks, [(0, 2), (2, 4), (4, 6), (6, 8)])

    def test_oversized_single_item_gets_own_chunk(self):
        # An item bigger than the bound can't be subdivided -> its own (over-budget) chunk. Admission
        # control (--vision-max-patches) rejects this upstream; the chunker must not infinite-loop.
        chunks = _chunk_by_patch_budget([10000, 1000], 8192)
        self.assertEqual(chunks, [(0, 1), (1, 2)])

    def test_covers_all_items_in_order(self):
        # Every item appears exactly once, contiguous, in order -> concatenating chunk features back
        # reproduces the original per-image order (the merge's per-req slicing depends on this).
        counts = [3000, 3000, 3000, 100, 9000, 50, 50]
        chunks = _chunk_by_patch_budget(counts, 8192)
        covered = []
        for s, e in chunks:
            self.assertLess(s, e)  # no empty chunks
            covered.extend(range(s, e))
        self.assertEqual(covered, list(range(len(counts))))
        # No chunk exceeds the bound unless it is a single oversized item.
        for s, e in chunks:
            total = sum(counts[s:e])
            self.assertTrue(total <= 8192 or (e - s) == 1)

    def test_empty(self):
        self.assertEqual(_chunk_by_patch_budget([], 8192), [])

    def test_deterministic_same_inputs_same_chunks(self):
        # Multi-host lockstep: identical inputs -> identical chunk boundaries (pure function).
        counts = [4000, 4000, 4000, 4000, 4000]
        a = _chunk_by_patch_budget(counts, 8192)
        b = _chunk_by_patch_budget(list(counts), 8192)
        self.assertEqual(a, b)

    def test_exact_fit_boundary(self):
        # Sum exactly at the bound stays in one chunk (uses <=, not <).
        self.assertEqual(_chunk_by_patch_budget([4096, 4096], 8192), [(0, 2)])
        self.assertEqual(_chunk_by_patch_budget([4096, 4097], 8192), [(0, 1), (1, 2)])


# --------------------------------------------------------------------------------------------------
# End-to-end _encode_mm_batched routing / bit-identity (jax-on-CPU stubs, no TPU).
#
# These reproduce the measured multi-chunk correctness regression. The bug: a SECOND, duplicated
# per-req merge loop ran after the first, reusing the SAME ``cum`` cursor without resetting it -- so
# the second loop sliced ``vfeats[m][total : total + n]`` (an empty [0, hidden] block) for every
# request and OVERWROTE the correct embeddings from the first loop, dropping all vision features.
# It corrupts EVERY request whenever ``plan`` is non-empty (independent of chunk count), which is why
# the chunked path collapsed to near-random accuracy. The single-chunk path is bit-identical to the
# single-call oracle only once the duplicate loop is gone.
#
# The stubs faithfully model the two contracts the merge depends on:
#   * encode: each image's feature rows are a per-image-distinguishable constant derived from the
#     pixel rows (which carry the image id), so a cross-request misroute is detectable.
#   * merge: writes EXACTLY ``image.shape[0]`` rows into the placeholder positions (matching
#     mm_core.merge's ``size=feats.shape[0]`` scatter), so an empty feature block leaves the
#     placeholder rows un-filled -- exactly the bug's observable effect.
HIDDEN = 4
IMG_TOK = 100
PLACEHOLDER_BASE = -7.0  # text-embed value at a placeholder row before vision is scattered in


class _Req:
    """Minimal request stand-in: distinguishable images, clean input_ids with IMG_TOK placeholders.

    Image ``iid`` occupies ``rows`` patch rows; every pixel row holds the float ``iid`` so the stub
    encoder can produce a per-image-unique feature (``iid + 1000``). grid_thw is (1, 1, rows) so the
    per-image patch count is ``rows`` (lets the budget force one image per chunk)."""

    def __init__(self, rid, images):  # images: list[(iid, rows)]
        self.rid = rid
        px, grids = [], []
        for iid, rows in images:
            px.append(np.full((rows, 1), float(iid), dtype=np.float32))
            grids.append((1, 1, rows))
        self.images = images
        self._pixel_values = np.concatenate(px, 0)
        self._grids = grids
        n_ph = sum(rows for _, rows in images)
        # input_ids: n_ph placeholder rows (== IMG_TOK) then one trailing text token.
        self.origin_input_ids = np.array([IMG_TOK] * n_ph + [1], dtype=np.int32)
        # _encode_mm_batched encodes over origin_input_ids + output_ids (resume-after-retract
        # coverage); on the initial prefill output_ids is empty -> unchanged behavior here.
        self.output_ids = []
        self.mm_inputs = {"_rid": rid}  # opaque; the stub assemble keys off the bound req
        self.multimodal_embedding = None


def _make_runner(reqs, max_pp):
    """Build a fake bound to ModelRunner._encode_mm_batched: only the attributes that method reads."""
    import jax.numpy as jnp

    by_rid = {r.mm_inputs["_rid"]: r for r in reqs}

    def assemble(mm_inputs):
        r = by_rid[mm_inputs["_rid"]]
        return {
            "pixel_values_images": r._pixel_values,
            "image_grid_thw": r._grids,
            "pixel_values_videos": None,
            "video_grid_thw": None,
            "audio_codes": None,
        }

    def _put(x, bf16=False):
        return None if x is None else np.asarray(x)

    def _thw(rows):
        return tuple(tuple(int(v) for v in row) for row in rows) if rows else None

    def jitted_encode_mm(mm_pixel_values=None, mm_grid_thw=None, **_):
        # Each pixel row carries its image id -> feature row = id + 1000 (per-image-unique). The ViT
        # segments per image, so concatenating a chunk's images then encoding == encoding each alone;
        # the stub reproduces that (purely row-wise) so chunk boundaries are bit-neutral.
        px = np.asarray(mm_pixel_values).reshape(-1)
        feats = (px[:, None] + 1000.0) * np.ones((1, HIDDEN), dtype=np.float64)
        return {"image": jnp.asarray(feats)}

    def jitted_merge_mm(input_ids, image=None, video=None, audio=None):
        ids = np.asarray(input_ids)
        seq = ids.shape[0]
        fused = np.full((seq, HIDDEN), PLACEHOLDER_BASE, dtype=np.float64)
        if image is not None:
            feats = np.asarray(image)
            # Scatter EXACTLY feats.shape[0] rows into the placeholder positions, in order
            # (mm_core.merge contract: size = feats.shape[0]). Empty feats -> nothing scattered.
            pos = np.flatnonzero(ids == IMG_TOK)[: feats.shape[0]]
            fused[pos] = feats
        return jnp.asarray(fused), None, None

    return (
        types.SimpleNamespace(
            model=types.SimpleNamespace(
                image_token_id=IMG_TOK, video_token_id=None, audio_token_id=None
            ),
            server_args=types.SimpleNamespace(
                vision_max_patches_per_encode=max_pp, vision_max_patches=0
            ),
            jitted_encode_mm=jitted_encode_mm,
            jitted_merge_mm=jitted_merge_mm,
        ),
        assemble,
        _put,
        _thw,
    )


def _expected_embedding(req):
    """The correct merged embedding for ``req``: placeholder rows filled with this req's own images'
    features (id+1000), trailing text row left at PLACEHOLDER_BASE."""
    rows = []
    for iid, rr in req.images:
        rows += [iid + 1000.0] * rr
    exp = np.full((len(req.origin_input_ids), HIDDEN), PLACEHOLDER_BASE, dtype=np.float64)
    exp[: len(rows)] = np.array(rows)[:, None]
    return exp


class TestEncodeBatchedRouting(unittest.TestCase):
    """_encode_mm_batched must route each req its OWN features and be bit-identical to single-call."""

    def _run(self, reqs, max_pp):
        runner, assemble, _put, _thw = _make_runner(reqs, max_pp)
        ModelRunner._encode_mm_batched(runner, reqs, assemble, _put, _thw)

    def test_multi_chunk_one_image_per_chunk(self):
        # 3 images of DIFFERENT grid sizes; budget = 1 patch forces one image per chunk (the CONC>1
        # regression shape). Each req must get its own image's features.
        reqs = [_Req(0, [(0, 4)]), _Req(1, [(1, 3)]), _Req(2, [(2, 5)])]
        self._run(reqs, max_pp=1)
        for r in reqs:
            np.testing.assert_array_equal(r.multimodal_embedding, _expected_embedding(r))

    def test_chunked_equals_single_call(self):
        # Bit-identity: chunked (budget forces splits) == single-call (budget covers everything).
        spec = [_Req(0, [(0, 4)]), _Req(1, [(1, 3)]), _Req(2, [(2, 5)])]
        chunked = [_Req(r.rid, r.images) for r in spec]
        single = [_Req(r.rid, r.images) for r in spec]
        self._run(chunked, max_pp=2)  # 4/3/5 patches -> multiple chunks
        self._run(single, max_pp=10_000)  # one chunk
        for rc, rs in zip(chunked, single):
            np.testing.assert_array_equal(rc.multimodal_embedding, rs.multimodal_embedding)

    def test_multi_image_per_req_routing(self):
        # A req with two images interleaved with single-image reqs; budget splits across image
        # boundaries inside and across reqs. Each req still gets its own images' features in order.
        reqs = [_Req(0, [(0, 2), (1, 3)]), _Req(1, [(2, 4)]), _Req(2, [(3, 2), (4, 2)])]
        self._run(reqs, max_pp=3)
        for r in reqs:
            np.testing.assert_array_equal(r.multimodal_embedding, _expected_embedding(r))

    def test_single_chunk_matches_expected(self):
        # The single-chunk path (CONC=1-like) must also be correct (the duplicate loop corrupted it
        # too -- it overwrote the one req's embedding with an empty slice).
        reqs = [_Req(0, [(0, 4)])]
        self._run(reqs, max_pp=10_000)
        np.testing.assert_array_equal(reqs[0].multimodal_embedding, _expected_embedding(reqs[0]))


if __name__ == "__main__":
    unittest.main()
