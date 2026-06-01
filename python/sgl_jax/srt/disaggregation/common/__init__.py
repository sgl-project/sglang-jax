"""Shared infrastructure for PD disaggregation backends.

Re-exports the common KV manager base and side-channel notifier.
"""

from sgl_jax.srt.disaggregation.common.core import (
    CommonKVManager,
    TerminalTransferRecord,
)
from sgl_jax.srt.disaggregation.common.zmq_notifier import ZmqPullNotifier

__all__ = [
    "CommonKVManager",
    "TerminalTransferRecord",
    "ZmqPullNotifier",
]
