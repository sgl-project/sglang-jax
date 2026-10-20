from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Any

import httpx
import zmq

from sgl_jax.srt.disaggregation.encoder.dispatcher import create_part_req_id
from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
    unpack_embedding_data,
)
from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput

if TYPE_CHECKING:
    from sgl_jax.srt.disaggregation.encoder.raiden_receiver import (
        RaidenReceiverBackend,
        RaidenReceiveSession,
    )

logger = logging.getLogger(__name__)


def plan_encoder_registrations(
    request: TokenizedGenerateReqInput,
) -> list[tuple[str, str]]:
    """Register exactly the modality parts dispatched by the tokenizer."""
    if not isinstance(request.rid, str):
        raise ValueError("encoder request requires a single rid")
    encoder_urls = request.encoder_urls
    if not encoder_urls:
        raise ValueError("encoder_urls is required")

    if request.num_items_assigned is None:
        raise ValueError("num_items_assigned is required")

    registrations: list[tuple[str, str]] = []
    for modality, assignments in request.num_items_assigned.items():
        if len(assignments) != len(encoder_urls):
            raise ValueError(
                f"{modality.name} has {len(assignments)} assignments for "
                f"{len(encoder_urls)} encoders"
            )
        for encoder_idx, count in enumerate(assignments):
            if count < 0:
                raise ValueError("num_items_assigned cannot contain negative values")
            if count == 0:
                continue
            part_idx = len(registrations)
            registrations.append(
                (
                    encoder_urls[encoder_idx],
                    create_part_req_id(request.rid, part_idx),
                )
            )

    if not registrations:
        raise ValueError("num_items_assigned does not assign any multimodal items")
    return registrations


def register_scheduler_receiver(
    registration: tuple[str, str],
    receive_url: str,
    client: httpx.Client,
) -> None:
    encoder_url, req_id = registration
    payload = {
        "req_id": req_id,
        "receive_url": receive_url,
    }
    response = client.post(
        f"{encoder_url.rstrip('/')}/scheduler_receive_url",
        json=payload,
    )
    response.raise_for_status()


class EncoderMetadataRouter:
    """Route one scheduler-wide metadata socket to pending encoder requests."""

    def __init__(self, host: str) -> None:
        self._receiver = zmq.Context.instance().socket(zmq.PULL)
        self._receiver.setsockopt(zmq.LINGER, 0)
        port = self._receiver.bind_to_random_port(f"tcp://{host}")
        self.receive_url = f"{host}:{port}"
        self._request_queues: dict[str, deque[Any]] = {}
        self._lock = threading.Lock()

    def register(self, req_ids: tuple[str, ...]) -> None:
        routes = set(req_ids)
        with self._lock:
            if len(routes) != len(req_ids) or not routes.isdisjoint(self._request_queues):
                raise ValueError(f"duplicate encoder metadata routes: {req_ids}")
            self._request_queues.update((req_id, deque()) for req_id in req_ids)

    def drain(self) -> None:
        while True:
            try:
                payload = self._receiver.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                data = unpack_embedding_data(payload)
            except (ValueError, TypeError):
                logger.warning("Discarding invalid encoder metadata")
                continue
            with self._lock:
                queue = self._request_queues.get(data.req_id)
                if queue is not None:
                    queue.append(data)

    def pop(self, req_ids: tuple[str, ...]) -> Any | None:
        with self._lock:
            for req_id in req_ids:
                queue = self._request_queues.get(req_id)
                if queue:
                    return queue.popleft()
        return None

    def unregister(self, req_ids: tuple[str, ...]) -> None:
        with self._lock:
            for req_id in req_ids:
                self._request_queues.pop(req_id, None)

    def close(self) -> None:
        with self._lock:
            self._request_queues.clear()
        self._receiver.close()


@dataclass(slots=True)
class PendingEncoderRequest:
    recv_req: TokenizedGenerateReqInput
    started_at: float
    metadata_router: EncoderMetadataRouter
    metadata_req_ids: tuple[str, ...]
    registration_futures: tuple[Future[None], ...]
    accumulator: MultiModalEmbeddingData
    backend: RaidenReceiverBackend
    apply_result: Callable[[TokenizedGenerateReqInput, dict[str, Any]], None]
    release_encoder: Callable[[], None]
    # Keep each part's metadata alongside its in-flight transfer session so the
    # completed embedding can later be assembled with the correct modality and grid.
    sessions: dict[int, tuple[EmbeddingData, Future[RaidenReceiveSession]]] = field(
        default_factory=dict
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _result: dict[str, Any] | None = None
    _error: Exception | None = None
    _closed: bool = False
    _claimed: bool = False
    preparation: Future[None] | None = None

    def poll(self) -> dict[str, Any] | None:
        with self._lock:
            if self.preparation is None or not self.preparation.done():
                return None
            self.preparation.result()
            self._claimed = True
            return self._result

    def progress(self, *, backend_progressed: bool = False) -> bool:
        """Advance one request from a dedicated receiver progress thread."""
        with self._lock:
            if self._closed or self._result is not None or self._error is not None:
                return True
            try:
                self._result = self._poll_once(backend_progressed=backend_progressed)
            except Exception as exc:
                self._error = exc
            return self._result is not None or self._error is not None

    def prepare_result(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._error is not None:
                raise self._error
            result = self._result
            self.apply_result(self.recv_req, result)

    def _poll_once(self, *, backend_progressed: bool = False) -> dict[str, Any] | None:
        for future in self.registration_futures:
            if future.done():
                future.result()  # error re-thrown to the scheduler main thread

        # The ZMQ message contains EmbeddingData metadata (part identity,
        # shape/dtype, and transfer endpoints); the backend pulls the actual
        # embedding separately through the receiver backend.
        data = self.metadata_router.pop(self.metadata_req_ids)
        if data is not None:
            if (
                data.num_parts != len(self.metadata_req_ids)
                or not 0 <= data.part_idx < data.num_parts
            ):
                raise ValueError("inconsistent encoder part metadata")
            if data.part_idx in self.sessions or self.accumulator.has_part(data.part_idx):
                raise ValueError(f"duplicate part_idx: {data.part_idx}")
            if data.error_msg is not None:
                raise RuntimeError(data.error_msg)
            self.sessions[data.part_idx] = (data, self.backend.start(data))

        for part_idx, (part_data, future) in list(self.sessions.items()):
            if not future.done():
                continue
            session = future.result()
            embedding = session.poll(refresh_backend=not backend_progressed)
            if embedding is None:
                continue
            self.accumulator.add(part_data, embedding)
            self.sessions.pop(part_idx)
            session.close()

        if self.accumulator.ready:
            return {
                "embeddings": self.accumulator.get_embedding(),
                **self.accumulator.get_mm_extra_meta(),
            }
        return None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for future in self.registration_futures:
                future.cancel()
            for _, future in self.sessions.values():
                if not future.cancel():
                    future.add_done_callback(self._close_session)
            self.sessions.clear()
            if not self._claimed:
                self.accumulator.release()
                self.release_encoder()
            self.metadata_router.unregister(self.metadata_req_ids)

    @staticmethod
    def _close_session(future: Future[RaidenReceiveSession]) -> None:
        try:
            future.result().close()
        except Exception:
            logger.exception("Encoder receiver setup failed during cleanup")


class EncoderClient:
    def __init__(
        self,
        host: str,
        backend: RaidenReceiverBackend,
        apply_result: Callable[[TokenizedGenerateReqInput, dict[str, Any]], None],
        registration_workers: int,
        registration_timeout: float | None,
        progress_interval_s: float = 0.001,
    ) -> None:
        self._backend = backend
        self._registration_executor = ThreadPoolExecutor(max_workers=max(1, registration_workers))
        self._registration_client = httpx.Client(timeout=registration_timeout)
        self._apply_result = apply_result
        self._progress_interval_s = max(0.0001, float(progress_interval_s))
        self._requests: dict[int, PendingEncoderRequest] = {}
        self._requests_lock = threading.Lock()
        self._completed_queue: SimpleQueue[PendingEncoderRequest] = SimpleQueue()
        self._prepare_executor = ThreadPoolExecutor(
            max_workers=min(2, max(1, registration_workers)),
            thread_name_prefix="encoder-language-prepare",
        )
        self._progress_stop = threading.Event()
        self._startup_done = threading.Event()
        self._startup_error: Exception | None = None
        self._metadata_router: EncoderMetadataRouter | None = None
        self._progress_thread = threading.Thread(
            target=self._progress_loop,
            args=(host,),
            name="encoder-receiver-progress",
            daemon=True,
        )
        self._progress_thread.start()
        if not self._startup_done.wait(30):
            self.close()
            raise TimeoutError("timed out starting encoder receiver progress thread")
        if self._startup_error is not None:
            self.close()
            raise RuntimeError(
                "failed to start encoder receiver progress thread"
            ) from self._startup_error

    @property
    def _router(self) -> EncoderMetadataRouter:
        if self._metadata_router is None:
            raise RuntimeError("encoder metadata router is not initialized")
        return self._metadata_router

    def drain_completed(self) -> list[PendingEncoderRequest]:
        completed = []
        while True:
            try:
                completed.append(self._completed_queue.get_nowait())
            except Empty:
                return completed

    def has_completed(self) -> bool:
        return not self._completed_queue.empty()

    def receive(self, request: TokenizedGenerateReqInput) -> PendingEncoderRequest:
        registrations = plan_encoder_registrations(request)
        metadata_req_ids = tuple(registration[1] for registration in registrations)
        router = self._router
        router.register(metadata_req_ids)
        registration_futures = []
        try:
            for registration in registrations:
                registration_futures.append(
                    self._registration_executor.submit(
                        register_scheduler_receiver,
                        registration,
                        router.receive_url,
                        self._registration_client,
                    )
                )
        except Exception:
            for future in registration_futures:
                future.cancel()
            router.unregister(metadata_req_ids)
            raise
        pending = PendingEncoderRequest(
            recv_req=request,
            started_at=time.monotonic(),
            metadata_router=router,
            metadata_req_ids=metadata_req_ids,
            registration_futures=tuple(registration_futures),
            accumulator=MultiModalEmbeddingData(len(metadata_req_ids)),
            backend=self._backend,
            apply_result=self._apply_result,
            release_encoder=lambda: self._registration_executor.submit(
                self._release_encoders, registrations
            ),
        )
        with self._requests_lock:
            self._requests[id(pending)] = pending
        return pending

    def _release_encoders(self, registrations: list[tuple[str, str]]) -> None:
        for part_idx, (url, req_id) in enumerate(registrations):
            try:
                response = self._registration_client.post(
                    f"{url.rstrip('/')}/release",
                    json={"req_id": req_id, "part_idx": part_idx},
                )
                response.raise_for_status()
            except Exception:
                logger.exception("Encoder release failed. req_id=%s", req_id)

    def _progress_loop(self, host: str) -> None:
        try:
            self._metadata_router = EncoderMetadataRouter(host)
        except Exception as exc:
            self._startup_error = exc
            self._startup_done.set()
            return
        self._startup_done.set()
        try:
            while not self._progress_stop.wait(self._progress_interval_s):
                try:
                    self._router.drain()
                except Exception:
                    logger.exception("Failed to drain encoder metadata")

                backend_progressed = False
                try:
                    backend_progressed = self._backend.progress()
                except Exception:
                    logger.exception("Failed to progress encoder receive backend")

                with self._requests_lock:
                    pending = list(self._requests.items())
                for key, request in pending:
                    if request.preparation is None:
                        if request.progress(backend_progressed=backend_progressed):
                            request.preparation = self._prepare_executor.submit(
                                request.prepare_result
                            )
                    elif request.preparation.done():
                        with self._requests_lock:
                            self._requests.pop(key, None)
                        self._completed_queue.put(request)
        finally:
            self._router.close()

    def close(self) -> None:
        self._progress_stop.set()
        self._progress_thread.join()
        with self._requests_lock:
            pending = list(self._requests.values())
            self._requests.clear()
        for request in pending:
            request.close()
        self._prepare_executor.shutdown(cancel_futures=True)
        self._backend.close()
        self._registration_executor.shutdown(wait=True)
        self._registration_client.close()


def create_encoder_client(server_args, model_runner, token_buckets, apply_result):
    """Select the receive backend and prepare its storage before starting the client."""
    from sgl_jax.raiden import require_raiden_preloaded
    from sgl_jax.srt.disaggregation.encoder.embedding_data import (
        precompile_received_embeddings,
    )
    from sgl_jax.srt.disaggregation.encoder.raiden_pool import create_encoder_pool
    from sgl_jax.srt.disaggregation.encoder.raiden_receiver import RaidenReceiverBackend
    from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip

    require_raiden_preloaded()
    transfer_timeout = server_args.encoder_send_timeout_seconds
    if transfer_timeout <= 0:
        raise ValueError("Raiden requires a positive encoder send timeout")
    host = resolve_host_ip(server_args.disaggregation_host_ip)
    channel_number = max(1, int(server_args.disaggregation_channel_number))
    pool = create_encoder_pool(server_args, model_runner.model_config, model_runner.mesh)
    if not server_args.disable_precompile:
        precompile_received_embeddings(pool.buffer, model_runner.model, token_buckets)
    backend = RaidenReceiverBackend(
        host=host,
        pool=pool,
        parallelism=channel_number,
        pool_size=server_args.encoder_transfer_pool_size,
        transfer_timeout_s=transfer_timeout,
    )
    control_timeout = server_args.encoder_control_timeout_seconds

    return EncoderClient(
        host=host,
        backend=backend,
        apply_result=apply_result,
        registration_workers=channel_number,
        registration_timeout=None if control_timeout <= 0 else control_timeout,
    )
