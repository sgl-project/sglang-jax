"""Scheduler bootstrap protocol tests; these do not launch a model or a TPU."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sgl_jax.srt.managers.scheduler import Scheduler


class TestLoRAOverlapBootstrap(unittest.TestCase):
    def make_scheduler(self):
        scheduler = SimpleNamespace(
            enable_overlap=True,
            process_batch_result_decode=Mock(),
            process_batch_result_prefill=Mock(),
        )
        scheduler.set_next_batch_sampling_info_done = (
            Scheduler.set_next_batch_sampling_info_done.__get__(scheduler)
        )
        return scheduler

    def make_batch(self, sampling_info, *, dummy=True):
        mode = SimpleNamespace(
            is_decode=lambda: not dummy,
            is_extend=lambda: False,
            is_idle=lambda: False,
            is_dummy_first=lambda: dummy,
        )
        return SimpleNamespace(forward_mode=mode, next_batch_sampling_info=sampling_info)

    def test_sampling_readiness_precedes_launch_wait(self):
        events = []
        sampling = SimpleNamespace(
            grammars=object(),
            update_grammar_vocab_mask=lambda: events.append("mask"),
            sampling_info_done=SimpleNamespace(set=lambda: events.append("ready")),
        )
        launch_done = SimpleNamespace(wait=lambda: events.append("wait"))

        Scheduler.process_batch_result(
            self.make_scheduler(), self.make_batch(sampling), None, launch_done
        )

        self.assertEqual(events, ["mask", "ready", "wait"])

    def test_next_preparation_waits_for_capture_after_idle(self):
        scheduler = self.make_scheduler()
        # Each bootstrap after idle must establish the same ordering.
        for idle_cycle in range(2):
            with self.subTest(idle_cycle=idle_cycle):
                self.assert_capture_before_replacement(scheduler)

    def assert_capture_before_replacement(self, scheduler):
        ready = threading.Event()
        allow_capture = threading.Event()
        scheduler_checkpoint = threading.Event()
        prepared_b = threading.Event()
        errors = []
        captured = []
        weights_a = object()
        weights_b = object()
        state = SimpleNamespace(weights=weights_a)

        class LaunchEvent(threading.Event):
            def wait(self, timeout=None):
                scheduler_checkpoint.set()
                return super().wait(timeout)

        launch_done = LaunchEvent()
        sampling = SimpleNamespace(grammars=None, sampling_info_done=ready)
        batch = self.make_batch(sampling)

        def prepare_next():
            try:
                Scheduler.process_batch_result(scheduler, batch, None, launch_done)
                state.weights = weights_b
                prepared_b.set()
            except Exception as error:
                errors.append(error)
            finally:
                scheduler_checkpoint.set()

        def dispatch():
            try:
                if not ready.wait(5):
                    raise AssertionError("Sampling readiness was not signalled")
                if not allow_capture.wait(5):
                    raise AssertionError("Test did not release weight capture")
                captured.append(state.weights)
            except Exception as error:
                errors.append(error)
            finally:
                launch_done.set()

        prepare_thread = threading.Thread(target=prepare_next, daemon=True)
        worker_thread = threading.Thread(target=dispatch, daemon=True)
        prepare_thread.start()
        worker_thread.start()
        try:
            self.assertTrue(scheduler_checkpoint.wait(5), "No scheduler checkpoint")
            self.assertTrue(ready.is_set(), "Waiting before readiness can deadlock")
            self.assertFalse(prepared_b.is_set(), "B replaced A before weight capture")
            self.assertIs(state.weights, weights_a)
        finally:
            allow_capture.set()
            # Release readiness on a failing test so no worker is left behind.
            ready.set()
            worker_thread.join(5)
            launch_done.set()
            prepare_thread.join(5)

        self.assertFalse(worker_thread.is_alive())
        self.assertFalse(prepare_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(captured, [weights_a])
        self.assertTrue(prepared_b.is_set())
        self.assertIs(state.weights, weights_b)

    def test_no_launch_event_still_signals_sampling_readiness(self):
        ready = threading.Event()
        sampling = SimpleNamespace(grammars=None, sampling_info_done=ready)
        Scheduler.process_batch_result(self.make_scheduler(), self.make_batch(sampling), None)
        self.assertTrue(ready.is_set())

    def test_already_launched_batch_does_not_need_sampling_info(self):
        launch_done = threading.Event()
        launch_done.set()
        Scheduler.process_batch_result(
            self.make_scheduler(), self.make_batch(None), None, launch_done
        )

    def test_decode_keeps_existing_result_handler(self):
        scheduler = self.make_scheduler()
        batch = self.make_batch(None, dummy=False)
        result, launch_done = object(), Mock()
        Scheduler.process_batch_result(scheduler, batch, result, launch_done)
        scheduler.process_batch_result_decode.assert_called_once_with(batch, result, launch_done)
        launch_done.wait.assert_not_called()

    def test_mask_failure_is_not_hidden_by_wait(self):
        launch_done = Mock()
        sampling = SimpleNamespace(
            grammars=object(),
            update_grammar_vocab_mask=Mock(side_effect=ValueError("invalid mask")),
            sampling_info_done=Mock(),
        )
        with self.assertRaisesRegex(ValueError, "invalid mask"):
            Scheduler.process_batch_result(
                self.make_scheduler(), self.make_batch(sampling), None, launch_done
            )
        sampling.sampling_info_done.set.assert_not_called()
        launch_done.wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
