import multiprocessing
import os
import select
import time


def test_select_on_queue():
    q = multiprocessing.Queue()

    # Get FD
    try:
        fd = q._reader.fileno()
        print(f"Queue FD: {fd}")
    except Exception as e:
        print(f"Failed to get FD: {e}")
        return

    print("Step 1: Select on empty queue (should timeout)")
    start = time.time()
    r, w, x = select.select([fd], [], [], 1.0)
    dur = time.time() - start
    print(f"  Duration: {dur:.4f}s, Ready: {r}")
    assert not r

    print("Step 2: Put item and select")
    q.put("hello")
    # Give time for pipe to fill
    time.sleep(0.1)

    start = time.time()
    r, w, x = select.select([fd], [], [], 1.0)
    dur = time.time() - start
    print(f"  Duration: {dur:.4f}s, Ready: {r}")
    assert fd in r

    item = q.get()
    print(f"  Got item: {item}")

    print("Test passed!")


if __name__ == "__main__":
    test_select_on_queue()
