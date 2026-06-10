"""producer_handoff path A uses a pre-reserved slot and registers the pytree."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from sgl_jax.srt.disaggregation.jax_transfer.conn import JaxTransferKVManager
from sgl_jax.srt.mem_cache.host_kv_pool import StagedData


class _FakeWrapper:
    def __init__(self):
        self.registered = {}
        self.released = []

    def register_pull(self, uuid, data):
        self.registered[uuid] = data

    def release(self, uuid):
        self.released.append(uuid)


class _FakePool:
    def __init__(self, raise_on_call=None):
        self.copy_calls = []
        self.released = []
        self._raise_on_call = raise_on_call

    def copy_from_device(self, layers, buffer_id):
        self.copy_calls.append((layers, buffer_id))
        if self._raise_on_call is not None and len(self.copy_calls) == self._raise_on_call:
            raise RuntimeError("simulated D2H failure")
        return StagedData(buffer_id=buffer_id, array_pytree=layers)

    def release(self, buffer_id):
        self.released.append(buffer_id)

    def put_buffer(self, buffer_id):
        self.released.append(buffer_id)


def test_path_a_uses_reserved_buffer_and_no_op_on_done():
    w = _FakeWrapper()
    pool = _FakePool()
    mgr = JaxTransferKVManager.__new__(JaxTransferKVManager)
    mgr._wrapper = w
    mgr._host_pool = pool
    layers = [jnp.ones((2, 3)), jnp.ones((2, 3))]
    status = mgr.producer_handoff("uuid-1", {"kv": layers}, use_d2h_staging=True, buffer_id=5)
    # copy_from_device called once with the reserved buffer id
    assert pool.copy_calls == [(layers, 5)]
    # registered under the sub-uuid as the pytree
    assert status.sub_uuids == ("uuid-1:kv",)
    assert "uuid-1:kv" in w.registered
    # on_done must NOT release the pool slot (scheduler owns release)
    status.on_done()
    assert pool.released == []


def test_path_a_rollback_releases_only_registered_sub_uuids_no_double_free():
    w = _FakeWrapper()
    # raise on the SECOND copy_from_device call
    pool = _FakePool(raise_on_call=2)
    mgr = JaxTransferKVManager.__new__(JaxTransferKVManager)
    mgr._wrapper = w
    mgr._host_pool = pool
    layers = [jnp.ones((2, 3))]
    # dicts preserve insertion order: "a" succeeds, "b" raises
    with pytest.raises(RuntimeError, match="simulated D2H failure"):
        mgr.producer_handoff(
            "uuid-1", {"a": layers, "b": layers}, use_d2h_staging=True, buffer_id=7
        )
    # only the sub-uuid registered before the failure ("a") was rolled back
    assert w.released == ["uuid-1:a"]
    # pool slot was NEVER released (scheduler prefill-terminal callback owns it)
    assert pool.released == []
