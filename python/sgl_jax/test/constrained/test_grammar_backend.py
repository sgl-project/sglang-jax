import unittest
from concurrent.futures import Future
from contextlib import contextmanager
from threading import Event
from unittest.mock import patch

import tiktoken
from llguidance.tiktoken import lltokenizer_from_encoding

from sgl_jax.srt.constrained.base_grammar_backend import INVALID_GRAMMAR_OBJ
from sgl_jax.srt.constrained.llguidance_backend import GuidanceBackend
from sgl_jax.srt.managers.schedule_batch import FINISH_ABORT, Req
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class TestGrammarBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A byte-level vocabulary keeps these tests independent of model downloads.
        encoding = tiktoken.Encoding(
            name="grammar-test",
            pat_str=r".",
            mergeable_ranks={bytes([i]): i for i in range(256)},
            special_tokens={"<|endoftext|>": 256},
        )
        cls.tokenizer = lltokenizer_from_encoding(encoding, eos_token=256)

    def setUp(self):
        self.backend = GuidanceBackend(self.tokenizer, num_threads=1)
        self.addCleanup(self.backend.executor.shutdown, wait=True)
        self.key = ("regex", "coherent|broken")

    @contextmanager
    def hold_worker(self):
        started = Event()
        release = Event()

        def block():
            started.set()
            if not release.wait(timeout=10):
                raise TimeoutError("Test worker was not released")

        blocker = self.backend.executor.submit(block)
        try:
            self.assertTrue(started.wait(timeout=10))
            yield
        finally:
            release.set()
            blocker.result(timeout=10)

    def assert_independent(self, first, second):
        self.assertIsNot(first, second)
        self.assertIsNot(first.ll_matcher, second.ll_matcher)
        for token in b"co":
            first.accept_token(token)
        for token in b"br":
            second.accept_token(token)
        for token in b"herent":
            first.accept_token(token)
        first.accept_token(self.tokenizer.eos_token)
        self.assertTrue(first.finished)
        self.assertFalse(second.finished)
        for token in b"oken":
            second.accept_token(token)
        second.accept_token(self.tokenizer.eos_token)
        self.assertTrue(second.finished)

    def test_cache_hits_have_independent_state(self):
        template = self.backend.dispatch_regex(self.key[1])
        self.backend.set_cache(self.key, template)

        first, first_hit = self.backend.get_cached_or_future_value(self.key)
        second, second_hit = self.backend.get_cached_or_future_value(self.key)

        self.assertTrue(first_hit)
        self.assertTrue(second_hit)
        self.assert_independent(first, second)
        self.assertFalse(template.finished)
        self.assertIs(self.backend.cache[self.key], template)
        later, cache_hit = self.backend.get_cached_or_future_value(self.key)
        self.assertTrue(cache_hit)
        self.assert_independent(later, template.copy())

    def test_cache_hits_do_not_recompile_grammar(self):
        template = self.backend.dispatch_regex(self.key[1])
        self.backend.set_cache(self.key, template)

        with patch(
            "sgl_jax.srt.constrained.llguidance_backend.LLMatcher",
            side_effect=AssertionError("Cache hits must not compile a new matcher"),
        ):
            first, first_hit = self.backend.get_cached_or_future_value(self.key)
            second, second_hit = self.backend.get_cached_or_future_value(self.key)

        self.assertTrue(first_hit)
        self.assertTrue(second_hit)
        self.assert_independent(first, second)

    def make_scheduler_with_ready_grammar(self, grammar):
        req = Req(
            rid="grammar-test",
            origin_input_text="",
            origin_input_ids=[1],
            sampling_params=SamplingParams(regex=self.key[1]),
        )
        req.grammar = Future()
        req.grammar.set_result(grammar)
        req.grammar_key = self.key
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.grammar_backend = self.backend
        scheduler.grammar_queue = [req]
        scheduler.waiting_queue = []
        return scheduler, req

    def test_ready_grammar_keeps_cache_template_untouched(self):
        grammar = self.backend.dispatch_regex(self.key[1])
        scheduler, req = self.make_scheduler_with_ready_grammar(grammar)

        scheduler.move_ready_grammar_requests()

        self.assertEqual(scheduler.grammar_queue, [])
        self.assertEqual(scheduler.waiting_queue, [req])
        self.assertIs(req.grammar, grammar)
        self.assertIsNot(self.backend.cache[self.key], grammar)
        for token in b"co":
            req.grammar.accept_token(token)

        later, cache_hit = self.backend.get_cached_or_future_value(self.key)
        self.assertTrue(cache_hit)
        for token in b"broken":
            later.accept_token(token)
        later.accept_token(self.tokenizer.eos_token)
        for token in b"herent":
            req.grammar.accept_token(token)
        req.grammar.accept_token(self.tokenizer.eos_token)
        self.assertTrue(later.finished)
        self.assertTrue(req.grammar.finished)

    def test_ready_invalid_grammar_is_cached_and_aborted(self):
        scheduler, req = self.make_scheduler_with_ready_grammar(INVALID_GRAMMAR_OBJ)

        scheduler.move_ready_grammar_requests()

        self.assertEqual(scheduler.grammar_queue, [])
        self.assertEqual(scheduler.waiting_queue, [req])
        self.assertIsNone(req.grammar)
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(req.to_finish.status_code, 400)
        self.assertIn("Invalid grammar request", req.to_finish.message)
        value, cache_hit = self.backend.get_cached_or_future_value(self.key)
        self.assertTrue(cache_hit)
        self.assertIs(value, INVALID_GRAMMAR_OBJ)

    def test_pending_compilations_have_independent_state(self):
        with self.hold_worker():
            first, first_hit = self.backend.get_cached_or_future_value(self.key)
            second, second_hit = self.backend.get_cached_or_future_value(self.key)
            self.assertFalse(first.done())
            self.assertFalse(second.done())

        self.assertFalse(first_hit)
        self.assertFalse(second_hit)
        self.assertIsNot(first, second)
        self.assert_independent(first.result(timeout=10), second.result(timeout=10))

    def test_completed_future_is_not_reused_before_caching(self):
        first, first_hit = self.backend.get_cached_or_future_value(self.key)
        first_grammar = first.result(timeout=10)
        second, second_hit = self.backend.get_cached_or_future_value(self.key)

        self.assertFalse(first_hit)
        self.assertFalse(second_hit)
        self.assert_independent(first_grammar, second.result(timeout=10))

    def test_cancelling_one_request_does_not_cancel_another(self):
        with self.hold_worker():
            first, _ = self.backend.get_cached_or_future_value(self.key)
            second, _ = self.backend.get_cached_or_future_value(self.key)
            self.assertTrue(first.cancel())
            self.assertFalse(second.cancelled())

        grammar = second.result(timeout=10)
        self.assert_independent(grammar, grammar.copy())

    def test_cached_invalid_grammar_keeps_sentinel_identity(self):
        self.backend.set_cache(self.key, INVALID_GRAMMAR_OBJ)

        for _ in range(2):
            grammar, cache_hit = self.backend.get_cached_or_future_value(self.key)
            self.assertTrue(cache_hit)
            self.assertIs(grammar, INVALID_GRAMMAR_OBJ)

    def test_reset_clears_cached_grammar(self):
        template = self.backend.dispatch_regex(self.key[1])
        self.backend.set_cache(self.key, template)
        self.backend.reset()

        future, cache_hit = self.backend.get_cached_or_future_value(self.key)

        self.assertFalse(cache_hit)
        self.assert_independent(future.result(timeout=10), template)


if __name__ == "__main__":
    unittest.main()
