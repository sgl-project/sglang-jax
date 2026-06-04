"""Endpoint-level tests for ``/get_server_info``.

Regression guard that validates the response schema of ``/get_server_info`` —
the introspection surface that monitoring tools and routers scrape.  Uses stub
injection (no model server needed), runs on CPU.

Reference: sglang GPU ``test/registered/unit/entrypoints/test_server_info.py``
"""

import asyncio
import dataclasses
import unittest
from types import SimpleNamespace

from sgl_jax.srt.entrypoints import http_server
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.test.test_utils import CustomTestCase


def _call_server_info_with(server_args: ServerArgs) -> dict:
    """Invoke ``http_server.get_server_info()`` against a stub global state.

    Bypasses the FastAPI HTTP layer: the handler is an ``async def`` that reads
    the module-level ``_global_state``, so wiring a ``SimpleNamespace`` stub via
    ``set_global_state`` and awaiting the coroutine directly is enough to
    exercise the handler logic without booting a model server.
    """

    async def _fake_internal_state():
        return [{"last_gen_throughput": 0.0, "memory_usage": {"kvcache": 0.0, "token_capacity": 0}}]

    stub_state = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            server_args=server_args,
            get_internal_state=_fake_internal_state,
        ),
        scheduler_info={"max_req_input_len": 1024},
    )
    prior_state = http_server._global_state
    http_server.set_global_state(stub_state)
    try:
        return asyncio.run(http_server.get_server_info())
    finally:
        http_server._global_state = prior_state


class TestServerInfoFieldsPreserved(CustomTestCase):
    """Regression guard: every field existing consumers depend on must remain
    visible in the ``/get_server_info`` response.
    """

    def test_every_server_args_field_appears_in_response(self):
        args = ServerArgs(model_path="dummy")
        info = _call_server_info_with(args)
        for field in dataclasses.fields(ServerArgs):
            self.assertIn(
                field.name,
                info,
                f"ServerArgs field '{field.name}' missing from /get_server_info response",
            )

    def test_internal_states_and_version_keys_preserved(self):
        args = ServerArgs(model_path="dummy")
        info = _call_server_info_with(args)
        self.assertIn("internal_states", info)
        self.assertIsInstance(info["internal_states"], list)
        self.assertGreater(len(info["internal_states"]), 0)
        self.assertIn("version", info)

    def test_scheduler_info_fields_merged(self):
        args = ServerArgs(model_path="dummy")
        info = _call_server_info_with(args)
        self.assertIn(
            "max_req_input_len",
            info,
            "scheduler_info field 'max_req_input_len' missing — scheduler_info not merged",
        )


if __name__ == "__main__":
    unittest.main()
