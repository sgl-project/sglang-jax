"""Tests for mini_lb router: helpers, generate, backend fetch, admission, launch args, OpenAI PD fields."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sgl_jax.srt.disaggregation.mini_lb as m
from sgl_jax.srt.disaggregation.mini_lb import MiniLoadBalancer, app, fetch_backend_json
from sgl_jax.srt.disaggregation.mini_lb_helpers import (
    ensure_request_identity_fields,
    generate_bootstrap_room,
    get_request_batch_size,
    inject_bootstrap_fields,
    maybe_wrap_ipv6_address,
)
from sgl_jax.srt.disaggregation.router_args import RouterArgs
from sgl_jax.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)

# ---- from test_mini_lb_helpers.py ----


class TestMaybeWrapIpv6:
    def test_ipv4_unchanged(self):
        assert maybe_wrap_ipv6_address("10.0.0.1") == "10.0.0.1"

    def test_hostname_unchanged(self):
        assert maybe_wrap_ipv6_address("myhost.local") == "myhost.local"

    def test_ipv6_wrapped(self):
        assert maybe_wrap_ipv6_address("::1") == "[::1]"

    def test_full_ipv6_wrapped(self):
        assert maybe_wrap_ipv6_address("2001:db8::1") == "[2001:db8::1]"


class TestGenerateBootstrapRoom:
    def test_room_is_int(self):
        room = generate_bootstrap_room()
        assert isinstance(room, int)

    def test_room_in_range(self):
        for _ in range(100):
            room = generate_bootstrap_room()
            assert 0 <= room < 2**63


class TestGetRequestBatchSize:
    def test_single_text(self):
        assert get_request_batch_size({"text": "hello"}) is None

    def test_batch_text(self):
        assert get_request_batch_size({"text": ["hello", "world"]}) == 2

    def test_single_input_ids(self):
        assert get_request_batch_size({"input_ids": [1, 2, 3]}) is None

    def test_batch_input_ids(self):
        assert get_request_batch_size({"input_ids": [[1, 2], [3, 4], [5, 6]]}) == 3

    def test_batch_prompt_strings(self):
        assert get_request_batch_size({"prompt": ["hello", "world"]}) == 2

    def test_batch_prompt_token_ids(self):
        assert get_request_batch_size({"prompt": [[1, 2], [3, 4], [5, 6]]}) == 3

    def test_no_prompt_field(self):
        assert get_request_batch_size({"other": "value"}) is None


class TestEnsureRequestIdentityFields:
    def test_generates_rid_and_transfer_id(self):
        result = ensure_request_identity_fields({"text": "hello"})
        assert "rid" in result
        assert "disagg_transfer_id" in result
        assert result["rid"] == result["disagg_transfer_id"]
        assert len(result["rid"]) == 32  # uuid4 hex

    def test_preserves_existing_rid(self):
        result = ensure_request_identity_fields({"text": "hello", "rid": "abc123"})
        assert result["rid"] == "abc123"
        assert result["disagg_transfer_id"] == "abc123"

    def test_preserves_existing_transfer_id(self):
        result = ensure_request_identity_fields({"text": "hello", "disagg_transfer_id": "tid123"})
        assert result["rid"] == "tid123"
        assert result["disagg_transfer_id"] == "tid123"

    def test_batch_generates_list(self):
        result = ensure_request_identity_fields({"text": ["a", "b", "c"]})
        assert isinstance(result["rid"], str)
        assert result["disagg_transfer_id"] == [
            f"{result['rid']}_0",
            f"{result['rid']}_1",
            f"{result['rid']}_2",
        ]

    def test_completion_prompt_batch_generates_aligned_transfer_ids(self):
        result = ensure_request_identity_fields({"prompt": ["a", "b"], "rid": "cmpl"})
        assert result["rid"] == "cmpl"
        assert result["disagg_transfer_id"] == ["cmpl_0", "cmpl_1"]

    def test_does_not_mutate_input(self):
        original = {"text": "hello"}
        result = ensure_request_identity_fields(original)
        assert "rid" not in original
        assert "rid" in result


class TestInjectBootstrapFields:
    def test_single_request(self):
        result = inject_bootstrap_fields(
            {"text": "hello"},
            prefill_server="http://10.0.0.1:30100",
            bootstrap_port=8998,
        )
        assert result["bootstrap_host"] == "10.0.0.1"
        assert result["bootstrap_port"] == 8998
        assert isinstance(result["bootstrap_room"], int)
        assert "rid" in result

    def test_batch_request(self):
        result = inject_bootstrap_fields(
            {"text": ["a", "b"]},
            prefill_server="http://10.0.0.1:30100",
            bootstrap_port=8998,
        )
        assert result["bootstrap_host"] == ["10.0.0.1", "10.0.0.1"]
        assert result["bootstrap_port"] == [8998, 8998]
        assert len(result["bootstrap_room"]) == 2
        assert result["bootstrap_room"][1] == result["bootstrap_room"][0] + 1

    def test_bootstrap_host_override(self):
        result = inject_bootstrap_fields(
            {"text": "hello"},
            prefill_server="http://localhost:30100",
            bootstrap_port=8998,
            bootstrap_host_override="10.31.0.1",
        )
        assert result["bootstrap_host"] == "10.31.0.1"

    def test_ipv6_server(self):
        result = inject_bootstrap_fields(
            {"text": "hello"},
            prefill_server="http://[::1]:30100",
            bootstrap_port=8998,
        )
        assert result["bootstrap_host"] == "[::1]"

    def test_none_bootstrap_port(self):
        result = inject_bootstrap_fields(
            {"text": "hello"},
            prefill_server="http://10.0.0.1:30100",
            bootstrap_port=None,
        )
        assert result["bootstrap_port"] is None


# ---- from test_mini_lb_generate.py ----


@pytest.fixture
def router_args():
    return RouterArgs(
        host="127.0.0.1",
        port=30000,
        mini_lb=True,
        pd_disaggregation=True,
        prefill_urls=[("http://prefill:30100", 8998)],
        decode_urls=["http://decode:30200"],
    )


@pytest.fixture
def lb_instance(router_args):
    import sgl_jax.srt.disaggregation.mini_lb as mini_lb_module

    instance = MiniLoadBalancer(router_args)
    mini_lb_module.lb = instance
    yield instance
    mini_lb_module.lb = None


@pytest.fixture
def client(lb_instance):
    from fastapi.testclient import TestClient

    return TestClient(app)


class TestValidation:
    def test_requires_pd_disaggregation(self):
        with pytest.raises(ValueError, match="PD disaggregation"):
            MiniLoadBalancer(
                RouterArgs(
                    mini_lb=True,
                    pd_disaggregation=False,
                    prefill_urls=[("http://p:30100", 8998)],
                    decode_urls=["http://d:30200"],
                )
            )

    def test_requires_prefill_urls(self):
        with pytest.raises(ValueError, match="at least one"):
            MiniLoadBalancer(
                RouterArgs(
                    mini_lb=True,
                    pd_disaggregation=True,
                    prefill_urls=[],
                    decode_urls=["http://d:30200"],
                )
            )

    def test_requires_decode_urls(self):
        with pytest.raises(ValueError, match="at least one"):
            MiniLoadBalancer(
                RouterArgs(
                    mini_lb=True,
                    pd_disaggregation=True,
                    prefill_urls=[("http://p:30100", 8998)],
                    decode_urls=[],
                )
            )


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200


class TestSelectPair:
    def test_select_pair_returns_tuple(self, lb_instance):
        p_url, bootstrap_port, d_url = lb_instance.select_pair()
        assert p_url == "http://prefill:30100"
        assert bootstrap_port == 8998
        assert d_url == "http://decode:30200"

    def test_select_pair_multi(self):
        args = RouterArgs(
            mini_lb=True,
            pd_disaggregation=True,
            prefill_urls=[
                ("http://p1:30100", 8998),
                ("http://p2:30100", 9998),
            ],
            decode_urls=["http://d1:30200", "http://d2:30200"],
        )
        lb = MiniLoadBalancer(args)
        seen_prefills = set()
        seen_decodes = set()
        for _ in range(100):
            p, _, d = lb.select_pair()
            seen_prefills.add(p)
            seen_decodes.add(d)
        assert len(seen_prefills) == 2
        assert len(seen_decodes) == 2


class TestGenerateEndpoint:
    @patch("aiohttp.ClientSession")
    def test_generate_non_stream(self, mock_session_cls, client, lb_instance):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"text": "hello world"})

        mock_post = AsyncMock(return_value=mock_response)
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        response = client.post(
            "/generate",
            json={
                "text": "hello",
                "sampling_params": {"max_new_tokens": 16},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data == {"text": "hello world"}


class TestV1ChatCompletionsEndpoint:
    @patch("aiohttp.ClientSession")
    def test_chat_completions(self, mock_session_cls, client, lb_instance):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"choices": [{"message": {"content": "hi"}}]})

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200


class TestV1CompletionsEndpoint:
    @patch("aiohttp.ClientSession")
    def test_completions(self, mock_session_cls, client, lb_instance):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"choices": [{"text": "world"}]})

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        response = client.post(
            "/v1/completions",
            json={
                "model": "test",
                "prompt": "hello",
            },
        )
        assert response.status_code == 200

    def test_parallel_sampling_rejected(self, client):
        response = client.post(
            "/v1/completions",
            json={
                "model": "test",
                "prompt": "hello",
                "n": 2,
            },
        )
        assert response.status_code == 400
        assert "n > 1" in response.json()["detail"]


class TestBootstrapInjection:
    @patch("aiohttp.ClientSession")
    def test_generate_injects_bootstrap_fields(self, mock_session_cls, client, lb_instance):
        captured_requests = []

        async def capture_post(url, json=None):
            captured_requests.append({"url": url, "json": json})
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"text": "ok"})
            return mock_resp

        mock_session = AsyncMock()
        mock_session.post = capture_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        response = client.post("/generate", json={"text": "hello"})
        assert response.status_code == 200

        assert len(captured_requests) == 2
        prefill_req = captured_requests[0]["json"]
        assert "bootstrap_host" in prefill_req
        assert "bootstrap_port" in prefill_req
        assert "bootstrap_room" in prefill_req
        assert prefill_req["bootstrap_port"] == 8998

    @patch("aiohttp.ClientSession")
    def test_bootstrap_host_override(self, mock_session_cls, client):
        import sgl_jax.srt.disaggregation.mini_lb as mini_lb_module

        args = RouterArgs(
            mini_lb=True,
            pd_disaggregation=True,
            prefill_urls=[("http://localhost:30100", 8998)],
            decode_urls=["http://decode:30200"],
            prefill_bootstrap_host="10.31.0.1",
        )
        instance = MiniLoadBalancer(args)
        mini_lb_module.lb = instance

        captured_requests = []

        async def capture_post(url, json=None):
            captured_requests.append({"url": url, "json": json})
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"text": "ok"})
            return mock_resp

        mock_session = AsyncMock()
        mock_session.post = capture_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        from fastapi.testclient import TestClient

        test_client = TestClient(app)
        response = test_client.post("/generate", json={"text": "hello"})
        assert response.status_code == 200

        prefill_req = captured_requests[0]["json"]
        assert prefill_req["bootstrap_host"] == "10.31.0.1"

        mini_lb_module.lb = None


# ---- from test_mini_lb_backend_fetch.py ----


class _FakeResponse:
    def __init__(self, status, json_data=None, text=""):
        self.status = status
        self._json = json_data
        self._text = text

    async def json(self):
        return self._json

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse]):
        self._responses = responses

    def get(self, url: str):
        for endpoint, resp in self._responses.items():
            if url.endswith(f"/{endpoint}"):
                return resp
        return _FakeResponse(404, text="not found")


def test_first_candidate_succeeds():
    session = _FakeSession(
        {
            "get_server_info": _FakeResponse(200, {"model": "test"}),
        }
    )

    async def _run():
        return await fetch_backend_json(
            session, "http://host:8000", ("get_server_info", "server_info")
        )

    result = asyncio.run(_run())
    assert result == {"model": "test"}


def test_fallback_to_second_candidate():
    session = _FakeSession(
        {
            "get_server_info": _FakeResponse(404, text="not found"),
            "server_info": _FakeResponse(200, {"model": "fallback"}),
        }
    )

    async def _run():
        return await fetch_backend_json(
            session, "http://host:8000", ("get_server_info", "server_info")
        )

    result = asyncio.run(_run())
    assert result == {"model": "fallback"}


def test_all_candidates_fail_raises():
    from fastapi import HTTPException

    session = _FakeSession(
        {
            "get_server_info": _FakeResponse(500, text="error1"),
            "server_info": _FakeResponse(503, text="error2"),
        }
    )

    async def _run():
        return await fetch_backend_json(
            session, "http://host:8000", ("get_server_info", "server_info")
        )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_run())
    assert exc_info.value.status_code == 502


def test_single_candidate():
    session = _FakeSession(
        {
            "server_info": _FakeResponse(200, {"status": "ok"}),
        }
    )

    async def _run():
        return await fetch_backend_json(session, "http://host:8000", ("server_info",))

    result = asyncio.run(_run())
    assert result == {"status": "ok"}


# ---- from test_mini_lb_admission.py ----


@pytest.fixture(autouse=True)
def reset_sem():
    saved = m._admission_sem
    m._admission_sem = None
    yield
    m._admission_sem = saved


def test_no_cap_passthrough(monkeypatch):
    """No cap -> semaphore is None and the request passes straight through."""

    async def fake_forward(req, ep):
        return "OK"

    monkeypatch.setattr(m, "_do_forward", fake_forward)
    assert m._admission_sem is None
    result = asyncio.run(m._forward_to_backend({}, "generate"))
    assert result == "OK"


def test_cap_serializes_excess(monkeypatch):
    """cap=1: the second request stays pending until the first releases, then
    both complete successfully (no error)."""

    m._admission_sem = asyncio.Semaphore(1)
    gate = None  # created inside the running loop
    started = []
    finished = []

    async def fake_forward(req, ep):
        started.append(req["id"])
        await gate.wait()
        finished.append(req["id"])
        return f"done-{req['id']}"

    monkeypatch.setattr(m, "_do_forward", fake_forward)

    async def scenario():
        nonlocal gate
        gate = asyncio.Event()

        t1 = asyncio.create_task(m._forward_to_backend({"id": 1}, "generate"))
        t2 = asyncio.create_task(m._forward_to_backend({"id": 2}, "generate"))

        # Let the tasks run; only the first should enter _do_forward (cap=1),
        # the second is pending on the semaphore.
        await asyncio.sleep(0.05)
        assert started == [1]
        assert t2.done() is False

        # Release the first; the second then acquires the permit and runs.
        gate.set()
        r1, r2 = await asyncio.gather(t1, t2)
        return r1, r2

    r1, r2 = asyncio.run(scenario())
    assert {r1, r2} == {"done-1", "done-2"}
    assert sorted(finished) == [1, 2]


def test_stream_holds_until_drained(monkeypatch):
    """Streaming: the permit is released only after body_iterator is fully
    drained, not when the StreamingResponse object is first returned."""
    from fastapi.responses import StreamingResponse

    m._admission_sem = asyncio.Semaphore(1)

    async def chunks():
        yield b"a"
        yield b"b"

    async def fake_forward(req, ep):
        return StreamingResponse(chunks())

    monkeypatch.setattr(m, "_do_forward", fake_forward)

    async def scenario():
        resp = await m._forward_to_backend({"stream": True}, "generate")
        # Permit still held right after return (stream not drained yet).
        assert m._admission_sem.locked() is True

        collected = [chunk async for chunk in resp.body_iterator]
        # Drained -> permit released.
        assert m._admission_sem.locked() is False
        return collected

    collected = asyncio.run(scenario())
    assert collected == [b"a", b"b"]


def test_no_client_error_on_excess(monkeypatch):
    """Many concurrent requests over a small cap all succeed (never 4xx/5xx)."""

    m._admission_sem = asyncio.Semaphore(2)

    async def fake_forward(req, ep):
        await asyncio.sleep(0.01)
        return f"ok-{req['id']}"

    monkeypatch.setattr(m, "_do_forward", fake_forward)

    async def scenario():
        tasks = [m._forward_to_backend({"id": i}, "generate") for i in range(10)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(scenario())
    assert sorted(results) == sorted(f"ok-{i}" for i in range(10))


# ---- from test_launch_router_args.py ----


def _parse(argv: list[str]) -> RouterArgs:
    import argparse

    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    return RouterArgs.from_cli_args(parser.parse_args(argv))


class TestPrefillUrlParsing:
    def test_single_prefill_with_bootstrap_port_space_separated(self):
        args = _parse(["--prefill", "http://p1:30100", "8998"])
        assert args.prefill_urls == [("http://p1:30100", 8998)]

    def test_single_prefill_with_bootstrap_port_comma_separated(self):
        args = _parse(["--prefill", "http://p1:30100,8998"])
        assert args.prefill_urls == [("http://p1:30100", 8998)]

    def test_single_prefill_without_bootstrap_port(self):
        args = _parse(["--prefill", "http://p1:30100"])
        assert args.prefill_urls == [("http://p1:30100", None)]

    def test_prefill_bootstrap_port_none_keyword(self):
        args = _parse(["--prefill", "http://p1:30100", "none"])
        assert args.prefill_urls == [("http://p1:30100", None)]

    def test_prefill_bootstrap_port_none_keyword_comma(self):
        args = _parse(["--prefill", "http://p1:30100,none"])
        assert args.prefill_urls == [("http://p1:30100", None)]

    def test_multiple_prefill_urls(self):
        args = _parse(
            [
                "--prefill",
                "http://p1:30100",
                "8998",
                "--prefill",
                "http://p2:30100",
                "9998",
            ]
        )
        assert args.prefill_urls == [
            ("http://p1:30100", 8998),
            ("http://p2:30100", 9998),
        ]


class TestDecodeUrlParsing:
    def test_single_decode_url(self):
        args = _parse(["--decode", "http://d1:30200"])
        assert args.decode_urls == ["http://d1:30200"]

    def test_multiple_decode_urls(self):
        args = _parse(
            [
                "--decode",
                "http://d1:30200",
                "--decode",
                "http://d2:30200",
            ]
        )
        assert args.decode_urls == ["http://d1:30200", "http://d2:30200"]

    def test_no_decode_urls(self):
        args = _parse([])
        assert args.decode_urls == []


class TestOtherArgs:
    def test_defaults(self):
        args = _parse([])
        assert args.host == "0.0.0.0"
        assert args.port == 30000
        assert args.mini_lb is False
        assert args.pd_disaggregation is False
        assert args.policy == "random"
        assert args.request_timeout_secs == 1800
        assert args.prefill_bootstrap_host is None

    def test_override_host_port(self):
        args = _parse(["--host", "127.0.0.1", "--port", "8080"])
        assert args.host == "127.0.0.1"
        assert args.port == 8080

    def test_flags(self):
        args = _parse(["--mini-lb", "--pd-disaggregation", "--test-external-dp-routing"])
        assert args.mini_lb is True
        assert args.pd_disaggregation is True
        assert args.test_external_dp_routing is True

    def test_prefill_bootstrap_host_override(self):
        args = _parse(["--prefill-bootstrap-host", "10.0.0.1"])
        assert args.prefill_bootstrap_host == "10.0.0.1"


# ---- from test_openai_pd_fields.py ----


class TestProtocolFields:
    def test_chat_completion_request_has_bootstrap_fields(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            bootstrap_host="10.0.0.1",
            bootstrap_port=8998,
            bootstrap_room=12345,
            disagg_transfer_id="tid-001",
        )
        assert req.bootstrap_host == "10.0.0.1"
        assert req.bootstrap_port == 8998
        assert req.bootstrap_room == 12345
        assert req.disagg_transfer_id == "tid-001"

    def test_completion_request_has_bootstrap_fields(self):
        req = CompletionRequest(
            model="test",
            prompt="hello",
            bootstrap_host="10.0.0.2",
            bootstrap_port=9998,
            bootstrap_room=67890,
            disagg_transfer_id="tid-002",
        )
        assert req.bootstrap_host == "10.0.0.2"
        assert req.bootstrap_port == 9998
        assert req.bootstrap_room == 67890
        assert req.disagg_transfer_id == "tid-002"

    def test_completion_request_accepts_batch_bootstrap_fields(self):
        req = CompletionRequest(
            model="test",
            prompt=["hello", "world"],
            bootstrap_host=["10.0.0.1", "10.0.0.1"],
            bootstrap_port=[8998, 8998],
            bootstrap_room=[1, 2],
            disagg_transfer_id=["tid-0", "tid-1"],
        )
        assert req.bootstrap_host == ["10.0.0.1", "10.0.0.1"]
        assert req.bootstrap_port == [8998, 8998]
        assert req.bootstrap_room == [1, 2]
        assert req.disagg_transfer_id == ["tid-0", "tid-1"]

    def test_bootstrap_fields_default_none(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.bootstrap_host is None
        assert req.bootstrap_port is None
        assert req.bootstrap_room is None
        assert req.disagg_transfer_id is None


class TestServingChatPassthrough:
    def test_chat_passes_bootstrap_fields(self):
        from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        mock_tokenizer_manager = MagicMock()
        mock_tokenizer_manager.model_config = None
        mock_tokenizer_manager.server_args = MagicMock()
        mock_tokenizer_manager.server_args.multimodal = False

        mock_template_manager = MagicMock()
        serving = OpenAIServingChat(mock_tokenizer_manager, mock_template_manager)

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
            bootstrap_host="10.0.0.1",
            bootstrap_port=8998,
            bootstrap_room=42,
        )

        with patch.object(serving, "_process_messages") as mock_process:
            mock_process.return_value = MagicMock(
                prompt="hello",
                prompt_ids="hello",
                image_data=None,
                video_data=None,
                audio_data=None,
                stop=[],
                tool_call_constraint=None,
            )
            with patch.object(serving, "_build_sampling_params", return_value={}):
                adapted, _ = serving._convert_to_internal_request(request)

        assert adapted.bootstrap_host == "10.0.0.1"
        assert adapted.bootstrap_port == 8998
        assert adapted.bootstrap_room == 42


class TestServingCompletionsPassthrough:
    def test_completions_passes_bootstrap_fields(self):
        from sgl_jax.srt.entrypoints.openai.serving_completions import (
            OpenAIServingCompletion,
        )

        mock_tokenizer_manager = MagicMock()
        mock_template_manager = MagicMock()
        mock_template_manager.completion_template_name = None
        serving = OpenAIServingCompletion(mock_tokenizer_manager, mock_template_manager)

        request = CompletionRequest(
            model="test",
            prompt="hello",
            bootstrap_host="10.0.0.2",
            bootstrap_port=9998,
            bootstrap_room=99,
        )

        adapted, _ = serving._convert_to_internal_request(request)

        assert adapted.bootstrap_host == "10.0.0.2"
        assert adapted.bootstrap_port == 9998
        assert adapted.bootstrap_room == 99
