"""Output logprobs under speculative decoding (MTP / EAGLE + return_logprob).

gp230 (10-03): after the verify-width and draft-extend-metadata fixes, the first
MTP + return_logprob request still crashed in
``add_logprob_return_values: output.next_token_logprobs[i]`` because spec prefill
ran the greedy ``argmax`` shortcut (``skip_sample=True``, no logprobs computed)
and spec decode never produced per-accepted-token logprobs. These tests pin the
pieces added by ``speculative/spec_logprob.py``:

* ``spec_prefill_can_skip_sample`` keeps the shortcut only for no-logprob batches;
* ``compute_spec_output_logprobs`` / ``attach_spec_output_logprobs`` turn the
  gathered verify rows into token / top-k / token-id logprobs;
* ``append_spec_output_logprobs`` appends exactly ``accept_len`` entries per
  request, trimmed to the request's own ``top_logprobs_num`` / ``token_ids_logprob``;
* ``draft_extend_logits_metadata`` builds the draft-model extend metadata with
  every extend-logprob switch off while keeping ``logits_indices``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()

import unittest
from types import SimpleNamespace

import jax
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.speculative.spec_logprob import (
    append_spec_output_logprobs,
    attach_spec_output_logprobs,
    compute_spec_output_logprobs,
    draft_extend_logits_metadata,
    materialize_spec_output_logprobs,
    spec_prefill_can_skip_sample,
)


def _mesh(dp=1):
    devices = np.array(jax.devices()[:4]).reshape(dp, 4 // dp)
    return jax.sharding.Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def _fake_req(*, return_logprob=True, return_output_logprob_only=False, top_k=0, token_ids=None):
    return SimpleNamespace(
        return_logprob=return_logprob,
        return_output_logprob_only=return_output_logprob_only,
        top_logprobs_num=top_k,
        token_ids_logprob=token_ids,
        output_token_logprobs_val=[],
        output_token_logprobs_idx=[],
        output_top_logprobs_val=[] if return_logprob else None,
        output_top_logprobs_idx=[] if return_logprob else None,
        output_token_ids_logprobs_val=[] if return_logprob else None,
        output_token_ids_logprobs_idx=[] if return_logprob else None,
    )


class PrefillSkipSampleTest(unittest.TestCase):
    def _batch(self, greedy=True, return_logprob=False, only=False):
        return SimpleNamespace(
            sampling_info=SimpleNamespace(is_all_greedy=greedy),
            return_logprob=return_logprob,
            return_output_logprob_only=only,
        )

    def test_greedy_without_logprob_skips(self):
        self.assertTrue(spec_prefill_can_skip_sample(self._batch(), legacy_non_overlap=False))

    def test_logprob_requests_must_sample(self):
        self.assertFalse(
            spec_prefill_can_skip_sample(self._batch(return_logprob=True), legacy_non_overlap=False)
        )
        self.assertFalse(
            spec_prefill_can_skip_sample(self._batch(only=True), legacy_non_overlap=False)
        )

    def test_non_greedy_or_legacy_never_skips(self):
        self.assertFalse(
            spec_prefill_can_skip_sample(self._batch(greedy=False), legacy_non_overlap=False)
        )
        self.assertFalse(spec_prefill_can_skip_sample(self._batch(), legacy_non_overlap=True))


class ComputeSpecOutputLogprobsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.logits = rng.normal(size=(6, 11)).astype(np.float32)
        self.tokens = rng.integers(0, 11, size=(6,)).astype(np.int32)

    def test_matches_numpy_log_softmax_and_topk(self):
        tok, top_v, top_i, full = compute_spec_output_logprobs(
            self.logits, self.tokens, None, top_k=3, need_full=True
        )
        ref = _log_softmax(self.logits)
        np.testing.assert_allclose(
            np.asarray(tok), ref[np.arange(6), self.tokens], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(np.asarray(full), ref, rtol=1e-5, atol=1e-5)
        ref_idx = np.argsort(-ref, axis=-1)[:, :3]
        np.testing.assert_array_equal(np.asarray(top_i), ref_idx)
        np.testing.assert_allclose(
            np.asarray(top_v), np.take_along_axis(ref, ref_idx, -1), rtol=1e-5, atol=1e-5
        )

    def test_no_topk_no_full(self):
        tok, top_v, top_i, full = compute_spec_output_logprobs(
            self.logits, self.tokens, None, top_k=0, need_full=False
        )
        self.assertEqual(tok.shape, (6,))
        self.assertIsNone(top_v)
        self.assertIsNone(top_i)
        self.assertIsNone(full)

    def test_temperature_scales_like_regular_sampler(self):
        temps = np.full((6,), 0.5, dtype=np.float32)
        tok, _, _, _ = compute_spec_output_logprobs(
            self.logits, self.tokens, temps, top_k=0, need_full=False
        )
        ref = _log_softmax(self.logits / 0.5)
        np.testing.assert_allclose(
            np.asarray(tok), ref[np.arange(6), self.tokens], rtol=1e-5, atol=1e-5
        )


class AttachAndAppendTest(unittest.TestCase):
    """bs=2, steps=2 -> width 3 rows per request; request 0 accepts 2, request 1 accepts 3."""

    def setUp(self):
        self.mesh = _mesh()
        rng = np.random.default_rng(1)
        self.width = 3
        self.vocab = 13
        self.logits = rng.normal(size=(2 * self.width, self.vocab)).astype(np.float32)
        self.verified = rng.integers(0, self.vocab, size=(2 * self.width,)).astype(np.int32)
        rep = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
        self.out = LogitsProcessorOutput(
            next_token_logits=jax.device_put(self.logits, rep), hidden_states=None
        )
        self.batch = SimpleNamespace(
            return_logprob=True,
            return_output_logprob_only=False,
            top_logprobs_nums=[5, 2],
            token_ids_logprobs=[None, [1, 4]],
            sampling_info=SimpleNamespace(is_all_greedy=True),
        )

    def test_attach_then_append_per_request(self):
        attach_spec_output_logprobs(
            self.out, self.verified, self.batch, self.mesh, width=self.width
        )
        self.assertEqual(self.out.next_token_logprobs.shape, (6,))
        self.assertEqual(self.out.next_token_top_logprobs_val.shape, (6, 5))
        self.assertEqual(self.out.next_token_token_ids_logprobs_val.shape, (6, self.vocab))
        host = materialize_spec_output_logprobs(self.out, self.width)
        ref = _log_softmax(self.logits)

        r0 = _fake_req(top_k=5)
        acc0 = self.verified[0:2].tolist()
        append_spec_output_logprobs(r0, host, 0, acc0)
        self.assertEqual(r0.output_token_logprobs_idx, acc0)
        np.testing.assert_allclose(
            r0.output_token_logprobs_val, ref[[0, 1], acc0], rtol=1e-5, atol=1e-5
        )
        self.assertEqual([len(v) for v in r0.output_top_logprobs_val], [5, 5])
        self.assertEqual(r0.output_top_logprobs_idx[1], np.argsort(-ref[1])[:5].tolist())
        self.assertEqual(r0.output_token_ids_logprobs_val, [])

        r1 = _fake_req(top_k=2, token_ids=[1, 4])
        acc1 = self.verified[3:6].tolist()
        append_spec_output_logprobs(r1, host, 1, acc1)
        self.assertEqual(len(r1.output_token_logprobs_val), 3)
        np.testing.assert_allclose(
            r1.output_token_logprobs_val, ref[[3, 4, 5], acc1], rtol=1e-5, atol=1e-5
        )
        self.assertEqual([len(v) for v in r1.output_top_logprobs_val], [2, 2, 2])
        self.assertEqual(r1.output_token_ids_logprobs_idx, [[1, 4]] * 3)
        np.testing.assert_allclose(
            r1.output_token_ids_logprobs_val[2], ref[5, [1, 4]], rtol=1e-5, atol=1e-5
        )
        self.assertTrue(all(isinstance(v, float) for v in r1.output_token_logprobs_val))
        self.assertTrue(all(isinstance(i, int) for i in r1.output_top_logprobs_idx[0]))

    def test_output_logprob_only_gets_token_logprobs_only(self):
        batch = SimpleNamespace(
            return_logprob=False,
            return_output_logprob_only=True,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            sampling_info=SimpleNamespace(is_all_greedy=True),
        )
        attach_spec_output_logprobs(self.out, self.verified, batch, self.mesh, width=self.width)
        self.assertIsNone(self.out.next_token_top_logprobs_val)
        self.assertIsNone(self.out.next_token_token_ids_logprobs_val)
        host = materialize_spec_output_logprobs(self.out, self.width)
        req = _fake_req(return_logprob=False, return_output_logprob_only=True)
        append_spec_output_logprobs(req, host, 1, self.verified[3:4].tolist())
        self.assertEqual(len(req.output_token_logprobs_val), 1)
        self.assertIsNone(req.output_top_logprobs_val)

    def test_zero_accept_appends_nothing_and_overflow_raises(self):
        attach_spec_output_logprobs(
            self.out, self.verified, self.batch, self.mesh, width=self.width
        )
        host = materialize_spec_output_logprobs(self.out, self.width)
        req = _fake_req(top_k=5)
        append_spec_output_logprobs(req, host, 0, [])
        self.assertEqual(req.output_token_logprobs_val, [])
        with self.assertRaises(ValueError):
            append_spec_output_logprobs(req, host, 1, [1, 2, 3, 4])

    def test_missing_logprobs_materialize_to_none(self):
        self.assertIsNone(materialize_spec_output_logprobs(self.out, self.width))
        self.assertIsNone(materialize_spec_output_logprobs(None, self.width))

    def test_non_greedy_rows_use_request_temperature(self):
        batch = SimpleNamespace(
            return_logprob=True,
            return_output_logprob_only=False,
            top_logprobs_nums=[0, 0],
            token_ids_logprobs=[None, None],
            sampling_info=SimpleNamespace(
                is_all_greedy=False, temperatures=np.array([[1.0], [0.5]], dtype=np.float32)
            ),
        )
        attach_spec_output_logprobs(self.out, self.verified, batch, self.mesh, width=self.width)
        got = np.asarray(self.out.next_token_logprobs)
        ref0 = _log_softmax(self.logits[:3])[np.arange(3), self.verified[:3]]
        ref1 = _log_softmax(self.logits[3:] / 0.5)[np.arange(3), self.verified[3:]]
        np.testing.assert_allclose(got[:3], ref0, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(got[3:], ref1, rtol=1e-5, atol=1e-5)


class DraftExtendLogitsMetadataTest(unittest.TestCase):
    def test_extend_logprob_switches_off_but_logits_indices_kept(self):
        mesh = _mesh()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            return_logprob=True,
            return_output_logprob_only=False,
            extend_seq_lens=np.array([4, 4], dtype=np.int32),
            extend_logprob_start_lens=np.array([0, 2], dtype=np.int32),
            top_logprobs_nums=[5, 0],
            token_ids_logprobs=[None, None],
            logits_indices=np.array([3, 7], dtype=np.int32),
            extend_input_logprob_token_ids=np.arange(6, dtype=np.int32),
            input_logprob_indices=np.arange(6, dtype=np.int32),
            spec_info_padded=None,
        )
        md = draft_extend_logits_metadata(batch, mesh)
        self.assertFalse(md.extend_return_logprob)
        self.assertFalse(md.extend_return_top_logprob)
        self.assertFalse(md.extend_token_ids_logprob)
        self.assertIsNone(md.top_logprobs_nums)
        self.assertIsNone(md.token_ids_logprobs)
        self.assertIsNone(md.extend_logprob_start_lens_cpu)
        self.assertIsNone(md.input_logprob_indices_device)
        np.testing.assert_array_equal(np.asarray(md.logits_indices), [3, 7])
        np.testing.assert_array_equal(np.asarray(md.extend_seq_lens), [4, 4])


if __name__ == "__main__":
    unittest.main()
