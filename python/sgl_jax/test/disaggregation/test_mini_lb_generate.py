"""Tests for mini_lb.py FastAPI endpoints using TestClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sgl_jax.srt.disaggregation.mini_lb import MiniLoadBalancer, app
from sgl_jax.srt.disaggregation.router_args import RouterArgs


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
            MiniLoadBalancer(RouterArgs(
                mini_lb=True,
                pd_disaggregation=False,
                prefill_urls=[("http://p:30100", 8998)],
                decode_urls=["http://d:30200"],
            ))

    def test_requires_prefill_urls(self):
        with pytest.raises(ValueError, match="at least one"):
            MiniLoadBalancer(RouterArgs(
                mini_lb=True,
                pd_disaggregation=True,
                prefill_urls=[],
                decode_urls=["http://d:30200"],
            ))

    def test_requires_decode_urls(self):
        with pytest.raises(ValueError, match="at least one"):
            MiniLoadBalancer(RouterArgs(
                mini_lb=True,
                pd_disaggregation=True,
                prefill_urls=[("http://p:30100", 8998)],
                decode_urls=[],
            ))


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

        response = client.post("/generate", json={
            "text": "hello",
            "sampling_params": {"max_new_tokens": 16},
        })
        assert response.status_code == 200
        data = response.json()
        assert data == {"text": "hello world"}


class TestV1ChatCompletionsEndpoint:
    @patch("aiohttp.ClientSession")
    def test_chat_completions(self, mock_session_cls, client, lb_instance):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "hi"}}]
        })

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        response = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 200


class TestV1CompletionsEndpoint:
    @patch("aiohttp.ClientSession")
    def test_completions(self, mock_session_cls, client, lb_instance):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"text": "world"}]
        })

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        response = client.post("/v1/completions", json={
            "model": "test",
            "prompt": "hello",
        })
        assert response.status_code == 200


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
