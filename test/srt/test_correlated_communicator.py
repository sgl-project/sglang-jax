# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from dataclasses import dataclass

from sgl_jax.srt.managers.tokenizer_manager import _CorrelatedCommunicator


@dataclass
class _Req:
    rid: str | None


@dataclass
class _Resp:
    rid: str | None
    value: int


class _DummySender:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, obj):
        self.sent.append(obj)


def test_correlated_communicator_allows_concurrent_calls():
    async def run():
        sender = _DummySender()
        communicator = _CorrelatedCommunicator(sender, fan_out=1)

        task_a = asyncio.create_task(communicator(_Req(rid="a"), timeout=1.0))
        task_b = asyncio.create_task(communicator(_Req(rid="b"), timeout=1.0))
        await asyncio.sleep(0)

        communicator.handle_recv(_Resp(rid="b", value=2))
        communicator.handle_recv(_Resp(rid="a", value=1))

        out_a = await task_a
        out_b = await task_b

        assert [resp.value for resp in out_a] == [1]
        assert [resp.value for resp in out_b] == [2]
        assert [req.rid for req in sender.sent] == ["a", "b"]

    asyncio.run(run())


def test_correlated_communicator_requires_non_empty_rid():
    async def run():
        communicator = _CorrelatedCommunicator(_DummySender(), fan_out=1)
        try:
            await communicator(_Req(rid=None), timeout=0.1)
        except ValueError as exc:
            assert "non-empty `rid`" in str(exc)
        else:
            raise AssertionError("Expected ValueError for missing rid.")

    asyncio.run(run())


def test_correlated_communicator_waits_for_fanout():
    async def run():
        communicator = _CorrelatedCommunicator(_DummySender(), fan_out=2)
        task = asyncio.create_task(communicator(_Req(rid="x"), timeout=1.0))
        await asyncio.sleep(0)

        communicator.handle_recv(_Resp(rid="x", value=10))
        await asyncio.sleep(0)
        assert not task.done()

        communicator.handle_recv(_Resp(rid="x", value=11))
        out = await task
        assert [resp.value for resp in out] == [10, 11]

    asyncio.run(run())
