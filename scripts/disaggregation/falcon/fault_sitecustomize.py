"""Test-only delayed Raiden completion notification; never used for benchmarks."""

import json
import os
import threading
import time
from pathlib import Path

_delay = float(os.environ.get("PD_TEST_HOLD_COMPLETION_S", "0"))
if _delay > 0:
    from sgl_jax.raiden import preload_raiden

    preload_raiden()
    from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper

    _original = RaidenTransferWrapper.poll_stats
    _lock = threading.Lock()
    _pending = []
    _directory = Path(os.environ["PD_TEST_EVENT_DIR"])
    _directory.mkdir(parents=True, exist_ok=True)
    _events = _directory / f"completions-{os.getpid()}.jsonl"

    def delayed(self):
        with _lock:
            sent, received, failed = _original(self)
            now = time.monotonic()
            if sent or received:
                with _events.open("a") as file:
                    file.write(
                        json.dumps(
                            {
                                "at": time.time(),
                                "sent": sent,
                                "received": received,
                                "delay_s": _delay,
                            }
                        )
                        + "\n"
                    )
                _pending.append((now + _delay, sent, received))
            out_sent, out_received = [], []
            while _pending and _pending[0][0] <= now:
                _, ready_sent, ready_received = _pending.pop(0)
                out_sent.extend(ready_sent)
                out_received.extend(ready_received)
            return out_sent, out_received, failed

    RaidenTransferWrapper.poll_stats = delayed
