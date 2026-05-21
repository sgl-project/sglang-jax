"""Unit tests for the tokenizer manager event-wakeup race fix (#1091)."""

import asyncio
import unittest
from types import SimpleNamespace

import pytest

from sgl_jax.srt.managers.tokenizer_manager import ReqState, TokenizerManager

pytestmark = pytest.mark.cpu_only


def _make_state(loop):
    """Build a minimal ReqState for testing _notify_state_event /
    _wait_one_response in isolation."""
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=None,
        created_time=0.0,
        event_loop=loop,
    )


def _make_tm():
    """Build a minimal TokenizerManager skeleton without running __init__,
    just enough for the methods under test."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.wait_timeout = 0.05
    tm.log_requests = False
    tm.dump_requests_folder = None
    tm.crash_dump_folder = None
    return tm


class TestTokenizerManagerEvent(unittest.IsolatedAsyncioTestCase):
    """Regression tests for the same-loop event race fixed by #1091."""

    async def test_same_loop_notify_is_synchronous(self):
        """``_notify_state_event`` in same-loop mode must set the event
        synchronously so the consumer's ``clear()`` cannot race against a
        deferred ``set`` deferred by ``call_soon_threadsafe``.
        """
        loop = asyncio.get_running_loop()
        state = _make_state(loop)
        tm = _make_tm()

        self.assertFalse(state.event.is_set())
        tm._notify_state_event(state)
        self.assertTrue(
            state.event.is_set(),
            msg=(
                "Same-loop notify must be synchronous; a deferred "
                "call_soon_threadsafe set is the root cause of #1091."
            ),
        )

    async def test_atomic_drain_handles_stale_wakeup(self):
        """A stale event ``set`` with an empty ``out_list`` (the cross-loop
        fallback path can still defer past ``clear()``) must hit the
        ``if not out_list: continue`` guard, not raise ``IndexError`` on
        ``out_list[-1]``.
        """
        loop = asyncio.get_running_loop()
        state = _make_state(loop)
        tm = _make_tm()

        # Simulate a stale set arriving after a prior drain emptied the list.
        state.event.set()
        self.assertEqual(state.out_list, [])

        obj = SimpleNamespace(rid="test_rid", stream=True)
        gen = tm._wait_one_response(obj, state, request=None)

        # First wake: stale (event set + out_list empty) -> drain block
        #   hits ``if not out_list: continue``.
        # Second wait: no producer ever sets again -> internal ``wait_for``
        #   times out after ``tm.wait_timeout``, ``request`` is None ->
        #   inner ``continue`` keeps looping.
        # The generator never produces output and (critically) never raises
        # ``IndexError``. We bound the test by wrapping in an outer
        # ``wait_for`` that times out, proving we did not crash and we did
        # not return.
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(gen.__anext__(), timeout=0.3)


if __name__ == "__main__":
    unittest.main()
