import threading
import time
from queue import Queue

from sgl_jax.srt.managers.communication import QueueBackend


def test_queue_deadlock():
    in_q = Queue()
    out_q = Queue()
    backend = QueueBackend(in_q, out_q)

    # 1. Simulate fast spin (Idle)
    # Reset stats
    backend.consecutive_empty_count = 10
    backend.last_activity_time = time.time() - 4.0  # > 3s ago
    backend.last_call_time = time.time()

    print("Step 1: Testing Idle Spin (should block)")
    start = time.time()
    # Call immediately implies small interval
    reqs = backend.recv_requests()
    dur = time.time() - start

    # It should have blocked for at least timeout (0.01s)
    # Because interval was small (effectively 0 since we set last_call_time to now)
    print(f"  Idle call duration: {dur:.4f}s")
    assert dur >= 0.01, "Should block when idle!"
    assert reqs == []

    # 2. Simulate working loop (Busy)
    print("Step 2: Testing Busy Loop (should NOT block)")

    # Reset stats again
    backend.consecutive_empty_count = 10
    backend.last_activity_time = time.time() - 4.0  # > 3s ago

    # Simulate processing time
    time.sleep(0.02)  # 20ms processing

    start = time.time()
    reqs = backend.recv_requests()
    dur = time.time() - start

    print(f"  Busy call duration: {dur:.4f}s")
    # It should NOT block, so duration should be negligible
    assert dur < 0.01, "Should NOT block when busy!"
    assert reqs == []

    print("Test passed!")


if __name__ == "__main__":
    test_queue_deadlock()
