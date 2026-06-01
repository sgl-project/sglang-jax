"""Domain-agnostic transport layer for request-scoped transfers.

Public surface:
* :class:`RequestTransportCore` — lifecycle, reaper, terminal records
* :class:`TerminalTransferRecord` — frozen record of a terminal state
* :class:`TransferBackend` — abstract tensor publish / pull / release
* :class:`JaxTransferBackend` — concrete backend over ``jax.experimental.transfer``
"""

from sgl_jax.srt.disaggregation.transport.backend import (
    JaxTransferBackend,
    TransferBackend,
)
from sgl_jax.srt.disaggregation.transport.core import (
    RequestTransportCore,
    TerminalTransferRecord,
)

__all__ = [
    "JaxTransferBackend",
    "RequestTransportCore",
    "TerminalTransferRecord",
    "TransferBackend",
]
