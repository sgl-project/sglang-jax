"""Shared infrastructure for PD disaggregation backends.

Re-exports the domain-agnostic transport core and side-channel notifier.
"""

from sgl_jax.srt.disaggregation.common.core import (
    RequestTransportCore,
    TerminalTransferRecord,
)
from sgl_jax.srt.disaggregation.common.zmq_notifier import ZmqPullNotifier

__all__ = [
    "RequestTransportCore",
    "TerminalTransferRecord",
    "ZmqPullNotifier",
]
