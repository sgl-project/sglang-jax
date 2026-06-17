from __future__ import annotations

import asyncio
import logging
import random
import warnings
from http import HTTPStatus
from itertools import chain

logger = logging.getLogger(__name__)

AIOHTTP_STREAM_READ_CHUNK_SIZE = 1024 * 64


class MiniLoadBalancer:
    def __init__(self, router_args):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.prefill_urls = [url[0] for url in router_args.prefill_urls]
        self.prefill_bootstrap_ports = [url[1] for url in router_args.prefill_urls]
        self.decode_urls = router_args.decode_urls
        self.test_external_dp_routing = router_args.test_external_dp_routing
        self.prefill_bootstrap_host = router_args.prefill_bootstrap_host
        self.prefill_dp_size = None
        self.decode_dp_size = None
        self.max_concurrent_requests = getattr(router_args, "max_concurrent_requests", None)

    def _validate_router_args(self, router_args) -> None:
        if getattr(router_args, "policy", "random") != "random":
            logger.warning("[MiniLB] Overriding policy to random")
            router_args.policy = "random"

        if not getattr(router_args, "pd_disaggregation", False):
            raise ValueError("MiniLB only supports PD disaggregation mode")
        if len(router_args.prefill_urls) == 0 or len(router_args.decode_urls) == 0:
            raise ValueError("MiniLB requires at least one prefill and one decode server")

    def start(self) -> None:
        import uvicorn

        global lb, _admission_sem
        lb = self
        if self.max_concurrent_requests:
            _admission_sem = asyncio.Semaphore(self.max_concurrent_requests)
        uvicorn.run(app, host=self.host, port=self.port)

    async def _ensure_dp_sizes(self) -> None:
        if self.prefill_dp_size is not None:
            return

        import aiohttp

        async with aiohttp.ClientSession() as session:
            prefill_info = await fetch_backend_json(
                session,
                self.prefill_urls[0],
                ("get_server_info", "server_info"),
            )
            decode_info = await fetch_backend_json(
                session,
                self.decode_urls[0],
                ("get_server_info", "server_info"),
            )
        self.prefill_dp_size = len(prefill_info.get("internal_states", [1]))
        self.decode_dp_size = len(decode_info.get("internal_states", [1]))
        logger.info(
            "[MiniLB] DP sizes: prefill=%s, decode=%s",
            self.prefill_dp_size,
            self.decode_dp_size,
        )

    def _fork_dp_requests(self, request: dict):
        p_rank = random.randint(0, self.prefill_dp_size - 1)
        d_rank = random.randint(0, self.decode_dp_size - 1)

        prefill_req = request.copy()
        decode_req = request.copy()
        prefill_req["routed_dp_rank"] = p_rank
        decode_req["routed_dp_rank"] = d_rank
        decode_req["disagg_prefill_dp_rank"] = p_rank

        return prefill_req, decode_req, d_rank

    def select_pair(self) -> tuple[str, int | None, str]:
        pidx = random.randint(0, len(self.prefill_urls) - 1)
        didx = random.randint(0, len(self.decode_urls) - 1)
        return (
            self.prefill_urls[pidx],
            self.prefill_bootstrap_ports[pidx],
            self.decode_urls[didx],
        )

    async def generate(
        self,
        modified_request: dict,
        prefill_server: str,
        decode_server: str,
        endpoint: str,
    ):
        import aiohttp
        from fastapi.responses import ORJSONResponse

        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        expected_decode_dp_rank = None
        if self.test_external_dp_routing:
            await self._ensure_dp_sizes()
            (
                prefill_req,
                decode_req,
                expected_decode_dp_rank,
            ) = self._fork_dp_requests(modified_request)
        else:
            prefill_req = modified_request
            decode_req = modified_request

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            tasks = [
                session.post(f"{prefill_server}/{endpoint}", json=prefill_req),
                session.post(f"{decode_server}/{endpoint}", json=decode_req),
            ]
            prefill_response, decode_response = await asyncio.gather(*tasks)

            if "return_logprob" in modified_request:
                prefill_json = await prefill_response.json()
                ret_json = await decode_response.json()
                if "meta_info" in ret_json and "input_token_logprobs" in ret_json["meta_info"]:
                    ret_json["meta_info"]["input_token_logprobs"] = (
                        prefill_json["meta_info"]["input_token_logprobs"]
                        + ret_json["meta_info"]["input_token_logprobs"]
                    )
            else:
                ret_json = await decode_response.json()

            if expected_decode_dp_rank is not None:
                actual = ret_json.get("meta_info", {}).get("dp_rank")
                if actual != expected_decode_dp_rank:
                    return ORJSONResponse(
                        content={
                            "error": (
                                f"DP rank mismatch: expected {expected_decode_dp_rank}, "
                                f"got {actual}"
                            )
                        },
                        status_code=500,
                    )

            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

    async def generate_stream(
        self,
        modified_request: dict,
        prefill_server: str,
        decode_server: str,
        endpoint: str = "generate",
    ):
        import aiohttp
        import orjson
        from fastapi.responses import StreamingResponse

        if self.test_external_dp_routing:
            warnings.warn(
                "--test-external-dp-routing is not supported with streaming",
                stacklevel=2,
            )

        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                tasks = [
                    session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                    session.post(f"{decode_server}/{endpoint}", json=modified_request),
                ]
                prefill_response, decode_response = await asyncio.gather(*tasks)

                if modified_request.get("return_logprob", False):
                    prefill_chunks = []
                    async for chunk in prefill_response.content:
                        prefill_chunks.append(chunk)

                    first_prefill_chunk = prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                    first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                    async for chunk in decode_response.content:
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                first_prefill_chunk_json["meta_info"]["input_token_logprobs"]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )
                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in decode_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        yield chunk

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import ORJSONResponse, Response, StreamingResponse

    from sgl_jax.srt.disaggregation.mini_lb_helpers import inject_bootstrap_fields

    app = FastAPI()
    lb: MiniLoadBalancer | None = None
    _admission_sem: asyncio.Semaphore | None = None

    async def fetch_backend_json(
        session,
        server_url: str,
        endpoint_candidates: tuple[str, ...],
    ) -> dict:
        last_status = None
        last_error_text = ""
        for endpoint in endpoint_candidates:
            async with session.get(f"{server_url}/{endpoint}") as response:
                if response.status == 200:
                    return await response.json()
                last_status = response.status
                last_error_text = await response.text()
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=(
                f"Failed to get {endpoint_candidates[0]} from {server_url}. "
                f"Last status: {last_status}, Response: {last_error_text}"
            ),
        )

    @app.get("/health")
    async def health_check():
        return Response(status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            tasks = []
            for server in chain(lb.prefill_urls, lb.decode_urls):
                tasks.append(session.get(f"{server}/health_generate"))
            for response in asyncio.as_completed(tasks):
                await response
        return Response(status_code=200)

    @app.post("/flush_cache")
    async def flush_cache():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            tasks = []
            for server in chain(lb.prefill_urls, lb.decode_urls):
                tasks.append(session.post(f"{server}/flush_cache"))
            for response in asyncio.as_completed(tasks):
                await response
        return Response(status_code=200)

    @app.get("/server_info")
    @app.get("/get_server_info")
    async def get_server_info():
        import aiohttp

        prefill_infos = []
        decode_infos = []
        all_internal_states = []

        async with aiohttp.ClientSession() as session:
            for server in lb.prefill_urls:
                prefill_infos.append(
                    await fetch_backend_json(
                        session,
                        server,
                        ("get_server_info", "server_info"),
                    )
                )
            for server in lb.decode_urls:
                info_json = await fetch_backend_json(
                    session,
                    server,
                    ("get_server_info", "server_info"),
                )
                decode_infos.append(info_json)
                if "internal_states" in info_json:
                    all_internal_states.extend(info_json["internal_states"])

        return {
            "internal_states": (
                all_internal_states
                if all_internal_states
                else [{"last_gen_throughput": 0.0, "avg_spec_accept_length": None}]
            ),
            "prefill": prefill_infos,
            "decode": decode_infos,
        }

    async def _get_model_info_impl():
        import aiohttp

        if not lb or not lb.prefill_urls:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="There is no server registered",
            )

        target_server_url = lb.prefill_urls[0]
        async with aiohttp.ClientSession() as session:
            return ORJSONResponse(
                content=await fetch_backend_json(
                    session,
                    target_server_url,
                    ("get_model_info", "model_info"),
                )
            )

    @app.get("/model_info")
    async def model_info():
        return await _get_model_info_impl()

    @app.get("/get_model_info")
    async def get_model_info():
        return await _get_model_info_impl()

    async def _do_forward(request_data: dict, endpoint_name: str):
        prefill_server, bootstrap_port, decode_server = lb.select_pair()
        modified_request = inject_bootstrap_fields(
            request_data,
            prefill_server=prefill_server,
            bootstrap_port=bootstrap_port,
            bootstrap_host_override=lb.prefill_bootstrap_host,
        )
        if request_data.get("stream", False):
            return await lb.generate_stream(
                modified_request,
                prefill_server,
                decode_server,
                endpoint=endpoint_name,
            )
        return await lb.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )

    async def _forward_to_backend(request_data: dict, endpoint_name: str):
        if _admission_sem is None:
            return await _do_forward(request_data, endpoint_name)

        # Pending admission: hold the permit while the request runs. Excess
        # requests await the semaphore (held pending at the proxy) and are
        # never rejected or aborted.
        await _admission_sem.acquire()
        released = False
        try:
            resp = await _do_forward(request_data, endpoint_name)
            if isinstance(resp, StreamingResponse):
                # Streaming returns immediately; the real work happens while the
                # body is drained. Transfer permit ownership to the iterator so
                # it is released only after the stream fully completes.
                original_iter = resp.body_iterator

                async def _release_after_stream():
                    try:
                        async for chunk in original_iter:
                            yield chunk
                    finally:
                        _admission_sem.release()

                resp.body_iterator = _release_after_stream()
                released = True
            return resp
        finally:
            if not released:
                _admission_sem.release()

    @app.post("/generate")
    async def handle_generate_request(request_data: dict):
        return await _forward_to_backend(request_data, "generate")

    @app.post("/v1/chat/completions")
    async def handle_chat_completion_request(request_data: dict):
        return await _forward_to_backend(request_data, "v1/chat/completions")

    @app.post("/v1/completions")
    async def handle_completion_request(request_data: dict):
        return await _forward_to_backend(request_data, "v1/completions")

    @app.get("/v1/models")
    async def get_models():
        import aiohttp

        prefill_server = lb.prefill_urls[0]
        async with aiohttp.ClientSession() as session:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())

except ModuleNotFoundError as exc:  # pragma: no cover
    logger.warning("mini_lb web deps unavailable: %s", exc)
    app = None
    lb = None
