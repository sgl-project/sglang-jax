"""Prefill sender frees its device payload after D2H staging.

Path A (D2H staging) copies the gather output to a host buffer and
registers the HOST arrays for pull, so the device KV is dead weight once
``send()`` returns. The sender must drop its ``_payload`` ref there —
otherwise every sender still queued for the decode ack keeps its device
gather output alive, accumulating until prefill OOMs (the conc48 root
cause). Path B registers the HBM arrays directly, so the payload must
survive ``send()`` until the ack releases it.
"""
from __future__ import annotations

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import JaxTransferKVSender


class _Status:
    def __init__(self):
        self.sub_uuids = ("req-a:kv",)

    def on_done(self):
        pass


class _Notifier:
    def __init__(self):
        self.registered = []
        self.unregistered = []

    def register_callback(self, uuid_bytes, cb):
        self.registered.append(uuid_bytes)

    def unregister_callback(self, uuid_bytes):
        self.unregistered.append(uuid_bytes)


class _Mgr:
    def __init__(self):
        self._notifier = _Notifier()
        self.handoff_calls = []

    @property
    def zmq_notifier(self):
        return self._notifier

    def producer_handoff(self, uuid, payload, *, use_d2h_staging, buffer_id):
        self.handoff_calls.append((uuid, use_d2h_staging, buffer_id))
        return _Status()


def _make_sender():
    mgr = _Mgr()
    sender = JaxTransferKVSender(mgr, "req-a")
    sender.init(kv_indices=None, transfer_id="req-a")
    return mgr, sender


def test_staging_send_drops_device_payload():
    mgr, sender = _make_sender()
    sender.attach_payload({"kv": [object()]}, use_d2h_staging=True, buffer_id=3)

    sender.send()

    # Staging registered the host copy, so the device payload is freed.
    assert sender._payload is None
    assert sender.poll() == KVPoll.TRANSFERRING
    assert mgr.handoff_calls == [("req-a", True, 3)]


def test_path_b_send_keeps_device_payload():
    mgr, sender = _make_sender()
    payload = {"kv": [object()]}
    sender.attach_payload(payload, use_d2h_staging=False, buffer_id=None)

    sender.send()

    # Path B pulls straight from HBM; the payload must stay alive until ack.
    assert sender._payload is payload
    assert sender.poll() == KVPoll.TRANSFERRING


def test_staging_send_handoff_failure_unregisters_and_keeps_payload():
    mgr, sender = _make_sender()

    def _boom(*a, **k):
        raise RuntimeError("handoff boom")

    mgr.producer_handoff = _boom
    payload = {"kv": [object()]}
    sender.attach_payload(payload, use_d2h_staging=True, buffer_id=1)

    try:
        sender.send()
        raised = False
    except RuntimeError:
        raised = True

    assert raised
    # Failed handoff: callback rolled back, payload NOT dropped (no transfer).
    assert mgr.zmq_notifier.unregistered
    assert sender._payload is payload
