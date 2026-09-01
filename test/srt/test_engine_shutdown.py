from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from sgl_jax.srt.entrypoints import engine as engine_module
from sgl_jax.srt.entrypoints.engine import Engine


class _FakeTokenizerManager:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _make_engine(loop) -> Engine:
    engine = object.__new__(Engine)
    engine.loop = loop
    engine.tokenizer_manager = _FakeTokenizerManager()
    engine.server_args = SimpleNamespace(enable_single_process=False)
    engine.send_to_rpc = None
    return engine


def test_sync_shutdown_rejects_waiting_on_its_own_event_loop():
    async def run() -> None:
        engine = _make_engine(asyncio.get_running_loop())
        with pytest.raises(RuntimeError, match="async_shutdown"):
            engine.shutdown()
        assert not engine.tokenizer_manager.closed

    asyncio.run(run())


def test_async_shutdown_closes_on_current_event_loop(monkeypatch):
    killed = []
    monkeypatch.setattr(
        engine_module,
        "kill_process_tree",
        lambda pid, *, include_parent: killed.append((pid, include_parent)),
    )

    async def run() -> None:
        engine = _make_engine(asyncio.get_running_loop())
        await engine.async_shutdown()
        assert engine.tokenizer_manager.closed

    asyncio.run(run())
    assert len(killed) == 1
    assert killed[0][1] is False
