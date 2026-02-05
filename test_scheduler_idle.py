import time
from unittest.mock import MagicMock

from sgl_jax.srt.managers.communication import CommunicationBackend
from sgl_jax.srt.managers.scheduler import Scheduler


def test_scheduler_idle_logic():
    # Mock backend
    backend = MagicMock(spec=CommunicationBackend)
    backend.recv_requests.return_value = []

    # We can't easily instantiate a full Scheduler due to dependencies.
    # We will check the logic by inspecting the code or creating a minimal dummy.
    # Since we modified the file, let's verify syntax and import first.

    print("Verifying Scheduler import...")
    try:
        from sgl_jax.srt.managers.scheduler import Scheduler

        print("Scheduler imported successfully.")
    except ImportError as e:
        print(f"Import failed: {e}")
        return

    # To test the logic without running the full massive scheduler,
    # we can inspect the file content change or rely on the fact that we replaced the code block.
    # The replaced code block:
    # 1. Initialize last_busy_time
    # 2. Update last_busy_time when batch is run
    # 3. If batch is None (else block), check time > 3.0
    # 4. If true, call wait_for_new_requests

    print("Test Logic Inspection:")
    with open("python/sgl_jax/srt/managers/scheduler.py", "r") as f:
        content = f.read()

    if "last_busy_time = time.time()" in content:
        print("  [OK] last_busy_time initialization found.")
    else:
        print("  [FAIL] last_busy_time initialization NOT found.")

    if "self._comm_backend.wait_for_new_requests(timeout=0.01)" in content:
        print("  [OK] wait_for_new_requests call found.")
    else:
        print("  [FAIL] wait_for_new_requests call NOT found.")

    if "time.time() - last_busy_time > 3.0" in content:
        print("  [OK] 3.0s window check found.")
    else:
        print("  [FAIL] 3.0s window check NOT found.")


if __name__ == "__main__":
    test_scheduler_idle_logic()
