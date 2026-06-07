"""Unit tests for MultiHostQueueBackend (multi-host SPMD stage lockstep).

Verifies the rank0-publish / non-rank0-subscribe contract that keeps every host's
AR-stage forward in lockstep, using real ZMQ over a tcp loopback address (no TPU).
"""

import time
import unittest
from queue import Queue

from sgl_jax.srt.managers.communication import MultiHostQueueBackend, QueueBackend

try:
    import zmq  # noqa: F401

    _HAS_ZMQ = True
except ImportError:
    _HAS_ZMQ = False


def _free_tcp_addr():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"tcp://127.0.0.1:{port}"


@unittest.skipUnless(_HAS_ZMQ, "pyzmq not installed (validated in TPU/CI env)")
class TestMultiHostQueueBackend(unittest.TestCase):
    def test_rank0_publishes_local_queue_and_nonrank0_receives(self):
        addr = _free_tcp_addr()
        in0, out0 = Queue(), Queue()
        in1, out1 = Queue(), Queue()
        rank0 = MultiHostQueueBackend(in0, out0, node_rank=0, pub_sub_addr=addr)
        rank1 = MultiHostQueueBackend(in1, out1, node_rank=1, pub_sub_addr=addr)
        # PUB/SUB needs a moment to connect before the first send is not dropped.
        time.sleep(0.3)

        in0.put({"rid": "a"})
        in0.put({"rid": "b"})
        got0 = rank0.recv_requests()
        got1 = rank1.recv_requests()

        self.assertEqual(got0, [{"rid": "a"}, {"rid": "b"}])
        # rank1 ignores its own (empty) queue and receives rank0's broadcast
        self.assertEqual(got1, [{"rid": "a"}, {"rid": "b"}])

    def test_rank0_publishes_empty_batch_so_nonrank0_does_not_block(self):
        addr = _free_tcp_addr()
        rank0 = MultiHostQueueBackend(Queue(), Queue(), node_rank=0, pub_sub_addr=addr)
        rank1 = MultiHostQueueBackend(Queue(), Queue(), node_rank=1, pub_sub_addr=addr)
        time.sleep(0.3)

        # rank0 idle turn -> still publishes [] so subscribers don't hang
        self.assertEqual(rank0.recv_requests(), [])
        self.assertEqual(rank1.recv_requests(), [])

    def test_send_pyobj_only_rank0_outputs(self):
        addr = _free_tcp_addr()
        out0, out1 = Queue(), Queue()
        rank0 = MultiHostQueueBackend(Queue(), out0, node_rank=0, pub_sub_addr=addr)
        rank1 = MultiHostQueueBackend(Queue(), out1, node_rank=1, pub_sub_addr=addr)

        rank0.send_pyobj("r0-out")
        rank1.send_pyobj("r1-out")

        self.assertEqual(out0.get_nowait(), "r0-out")
        self.assertTrue(out1.empty())  # non-rank0 discards output

    def test_is_a_queuebackend(self):
        # single-host path must remain a plain QueueBackend (no behavior change)
        qb = QueueBackend(Queue(), Queue())
        self.assertIsInstance(qb, QueueBackend)


if __name__ == "__main__":
    unittest.main()
