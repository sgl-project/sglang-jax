import contextlib
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def _is_jax_array(obj):
    return hasattr(obj, "shape") and hasattr(obj, "dtype")


class TensorJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if _is_jax_array(obj):
            try:
                return {
                    "__tensor_type__": "jax",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "data": (
                        obj.tolist()
                        if obj.size < 100
                        else f"<array too large: {obj.shape}>"
                    ),
                }
            except Exception:
                return {
                    "__tensor_type__": "jax",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "data": f"<cannot serialize array: {obj.shape}>",
                }
        try:
            return str(obj)
        except Exception:
            return f"<non-serializable object: {type(obj).__name__}>"


class PrecisionTracerRequestMetadata:
    def __init__(self, request_id, request_input_ids, forward_mode):
        self.request_id = request_id
        self.input_hash = hashlib.md5(
            str(request_input_ids).encode("utf-8")
        ).hexdigest()[:16]
        self.request_input_ids = request_input_ids
        self.input_len = len(request_input_ids)
        self.forward_mode = forward_mode

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "input_hash": self.input_hash,
            "input_len": self.input_len,
            "forward_mode": self.forward_mode,
        }


class PrecisionTracerRecord:
    def __init__(
        self,
        bid,
        request_id,
        request_idx,
        start_time,
        precision_records,
        status,
        content_hash,
        process_id,
    ):
        self.bid = bid
        self.request_id = request_id
        self.request_idx = request_idx
        self.start_time = start_time
        self.end_time = None
        self.duration = None
        self.precision_records = {
            "prefill": [],
            "decode": [],
        }
        self.status = status
        self.content_hash = content_hash
        self.process_id = process_id

    def to_dict(self):
        return {
            "bid": self.bid,
            "request_id": self.request_id,
            "request_idx": self.request_idx,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "precision_records": self.precision_records,
            "status": self.status,
            "content_hash": self.content_hash,
            "process_id": self.process_id,
        }


class PrecisionTracer:
    def __init__(self):
        self.lock = threading.Lock()

        # setting
        self._trace_active = False
        self._trace_output_file = None
        self._verbose_logging = False
        self._enable_precision_tracer = False

        # counter
        self._max_requests = 0
        self._completed_requests_count = 0
        self._request_counter = 0

        # metadata
        self._current_batch_id = None
        self._records: dict[str, PrecisionTracerRecord] = {}
        self._batch_requests_mapping: dict[
            int, list[PrecisionTracerRequestMetadata]
        ] = {}

        self._token_counters: dict[str, int] = {}
        self._last_forward_pass_id: dict[str, int] = {}
        self._current_forward_pass_id: int = -1

    def set_enable_precision_tracer(self, enabled: bool):
        self._enable_precision_tracer = enabled
        logger.info(
            "Precision tracer globally %s", "enabled" if enabled else "disabled"
        )

    def get_trace_active(self):
        with self.lock:
            return self._trace_active

    def get_max_requests(self):
        with self.lock:
            return self._max_requests

    def get_request_counter(self):
        with self.lock:
            return self._request_counter

    def add_request_counter(self):
        with self.lock:
            self._request_counter += 1

    def get_completed_requests_count(self):
        with self.lock:
            return self._completed_requests_count

    def add_completed_requests_count(self):
        with self.lock:
            self._completed_requests_count += 1

    def set_request_status_to_completed(self, request_id: str):
        with self.lock:
            self._records[request_id].status = "completed"

    def set_end_time_and_duration(self, request_id: str):
        with self.lock:
            self._records[request_id].end_time = time.time()
            self._records[request_id].duration = (
                self._records[request_id].end_time
                - self._records[request_id].start_time
            )

    def add_request_to_batch_requests_mapping(
        self, batch_id: int, request_metadata: PrecisionTracerRequestMetadata
    ):
        with self.lock:
            if batch_id not in self._batch_requests_mapping:
                self._batch_requests_mapping[batch_id] = []
            self._batch_requests_mapping[batch_id].append(request_metadata)

    def start_trace(
        self,
        req_num: int | None = None,
        output_file: str | None = None,
        verbose_logging: bool = False,
    ):
        if not self._enable_precision_tracer:
            logger.warning(
                "Precision tracer is disabled. Enable with --enable-precision-tracer"
            )
            return

        if self._trace_active:
            return

        logger.info("set trace activate is true")
        self._trace_active = True
        self._verbose_logging = verbose_logging

        self._max_requests = req_num
        self._completed_requests_count = 0
        self._request_counter = 0

        self._current_batch_id = None
        self._records: dict[str, PrecisionTracerRecord] = {}
        self._batch_requests_mapping: dict[
            int, list[PrecisionTracerRequestMetadata]
        ] = {}
        self._token_counters: dict[str, int] = {}
        self._last_forward_pass_id: dict[str, int] = {}

        if not output_file:
            raise ValueError("output_file is required")

        self._trace_output_file = output_file
        logger.info("Trace output file: %s", self._trace_output_file)

        os.makedirs(os.path.dirname(self._trace_output_file), exist_ok=True)

        with open(self._trace_output_file, "w") as _:
            pass

        logger.info("Request tracing started. Output: %s", self._trace_output_file)
        if req_num:
            logger.info("Will trace up to %s requests", req_num)
        if not verbose_logging:
            logger.info("Verbose console logging disabled during tracing")

    def stop_trace(self):
        if not self._trace_active:
            logger.info("No active request tracing")
            return

        self._trace_active = False
        output_file = self._trace_output_file
        self._trace_output_file = None

        try:
            with open(output_file, "a", encoding="utf-8") as f:
                for _, record in self._records.items():
                    record_dict = record.to_dict()
                    json.dump(record_dict, f, cls=TensorJSONEncoder, ensure_ascii=False)
                    f.write("\n")

            logger.info(
                "Saved %s request traces to: %s", len(self._records), output_file
            )

        except Exception as e:
            logger.error("Error saving traces to %s: %s", output_file, e)

        self._records.clear()
        self._batch_requests_mapping.clear()
        self._token_counters.clear()
        self._last_forward_pass_id.clear()
        logger.info("Request tracing stopped. Traces saved to: %s", output_file)
        return output_file

    def start_batch_trace(self, batch_id: int):
        if not self._trace_active:
            return

        with self.lock:
            requests_in_batch = self._batch_requests_mapping.get(batch_id, [])

            if len(requests_in_batch) == 0:
                logger.warning("Batch %s has no requests to trace", batch_id)
                return

            self._current_batch_id = batch_id
            process_pid = os.getpid()
            for idx, request_metadata in enumerate(requests_in_batch):
                # prefill
                if request_metadata.forward_mode == 1:
                    request_idx = self._completed_requests_count + idx
                    record = PrecisionTracerRecord(
                        bid=batch_id,
                        request_id=request_metadata.request_id,
                        request_idx=request_idx,
                        start_time=time.time(),
                        precision_records=None,
                        status="active",
                        content_hash=request_metadata.input_hash,
                        process_id=process_pid,
                    )
                    self._records[request_metadata.request_id] = record
                    self._token_counters[request_metadata.request_id] = 0
                    self._last_forward_pass_id[request_metadata.request_id] = -1

    def set_current_forward_pass_id(self, forward_pass_id: int):
        """Set the current forward pass ID for tracking inference steps"""
        if not self._trace_active:
            return
        self._current_forward_pass_id = forward_pass_id

    def jit_pure_callback_record(
        self, tensor: Any, name: str, stage: str, layer_id: int | None = None
    ) -> bool:
        if self._enable_precision_tracer:
            full_stage = (
                f"{stage}_layer_id_{layer_id}" if layer_id is not None else stage
            )

            def trace_callback(tensor):
                # Debug logging to check what stage is being passed
                if self._trace_active:
                    logger.debug(
                        "Recording tensor %s with stage: %s, layer_id: %s",
                        name,
                        full_stage,
                        layer_id,
                    )
                precision_tracer.record(tensor, name, full_stage)
                return True

            # pure_callback must have a return value, otherwise, it will be removed by the jit compiler's optimization
            callback_flag = jax.pure_callback(trace_callback, jnp.bool_(True), tensor)

            return callback_flag
        else:
            return jnp.bool_(True)

    def record(
        self,
        tensor: Any,
        name: str,
        stage: str = "",
    ):
        if not self._enable_precision_tracer or not self._trace_active:
            return

        if tensor is None:
            logger.info("[%s] %s: None", stage, name)
            return

        with self.lock:
            request_in_batch = self._batch_requests_mapping.get(
                self._current_batch_id, []
            )
            current_batch_id = self._current_batch_id

        if len(request_in_batch) == 0:
            logger.warning("Batch %s has no requests to trace", current_batch_id)
            return

        prisicion_infos = self._calculate_tensor_pricision_info(
            tensor, name, stage, request_in_batch
        )

        with self.lock:
            for req_id, data in prisicion_infos.items():
                if req_id in self._records:
                    forward_mode = request_in_batch[0].forward_mode
                    category = "prefill" if forward_mode == 1 else "decode"

                    precision_records = self._records[req_id].precision_records

                    if category not in precision_records:
                        precision_records[category] = []

                    if category == "prefill":
                        current_token_group = None
                        if len(precision_records[category]) > 0:
                            current_token_group = precision_records[category][-1]

                        if current_token_group is None:
                            current_token_group = {
                                "token_idx": 0,
                                "category": category,
                                "records": [],
                            }
                            precision_records[category].append(current_token_group)

                            if "sequence_length" in data:
                                self._token_counters[req_id] = data["sequence_length"]
                            else:
                                self._token_counters[req_id] = 1

                        record_with_metadata = data.copy()
                        current_token_group["records"].append(record_with_metadata)

                    else:
                        # For decode, use forward_pass_id to determine when to start new token group
                        current_token_idx = self._token_counters.get(req_id, 0)
                        last_forward_pass_id = self._last_forward_pass_id.get(
                            req_id, -1
                        )

                        # Check if this is a new forward pass (new inference step)
                        is_new_forward_pass = (
                            hasattr(self, "_current_forward_pass_id")
                            and self._current_forward_pass_id != last_forward_pass_id
                        )

                        # If this is a new forward pass, increment token counter
                        if is_new_forward_pass and last_forward_pass_id != -1:
                            current_token_idx = self._token_counters.get(req_id, 0) + 1
                            self._token_counters[req_id] = current_token_idx
                            self._last_forward_pass_id[req_id] = (
                                self._current_forward_pass_id
                            )

                        # Look for existing token group at current position
                        current_token_group = None
                        for token_group in precision_records[category]:
                            if token_group["token_idx"] == current_token_idx:
                                current_token_group = token_group
                                break

                        # If no existing token group, create one
                        if current_token_group is None:
                            current_token_group = {
                                "token_idx": current_token_idx,
                                "category": category,
                                "records": [],
                            }
                            precision_records[category].append(current_token_group)
                            # Update forward pass id for first record of this token
                            if hasattr(self, "_current_forward_pass_id"):
                                self._last_forward_pass_id[req_id] = (
                                    self._current_forward_pass_id
                                )

                        # Add record to the token group
                        record_with_metadata = data.copy()
                        current_token_group["records"].append(record_with_metadata)

                else:
                    logger.warning("Request %s not found in records", req_id)
                    continue

        for req_id, data in prisicion_infos.items():
            if req_id in self._records:
                stats_with_req_id = data.copy()
                stats_with_req_id["request_id"] = req_id
                self._verbose_logging_console(stats_with_req_id)

    def _calculate_tensor_pricision_info(
        self,
        tensor: Any,
        name: str,
        stage: str,
        request_in_batch: list[PrecisionTracerRequestMetadata],
    ) -> dict[str, Any]:
        try:
            try:
                test_scalar = jnp.array(1.0)
                _ = test_scalar.item()
                can_concretize = True
            except Exception:
                can_concretize = False
            layer_id, module_type = self._parse_layer_and_module(stage)

            if not hasattr(tensor, "shape") or len(tensor.shape) == 0:
                raise ValueError("Tensor has no valid shape")

            total_batch_size = tensor.shape[0]
            current_idx = 0
            result = {}

            for idx, req_meta in enumerate(request_in_batch):
                req_id = req_meta.request_id
                seq_len = req_meta.input_len if req_meta.forward_mode == 1 else 1

                if current_idx + seq_len > total_batch_size:
                    logger.error(
                        "[TENSOR_DEBUG] ERROR: Request %s requires %s tokens, but only %s left in tensor of shape %s. Current position: %s",
                        req_id,
                        seq_len,
                        total_batch_size - current_idx,
                        tensor.shape,
                        current_idx,
                    )
                    continue

                slice_tensor = jnp.take(
                    tensor, jnp.arange(current_idx, current_idx + seq_len), axis=0
                )
                current_idx += seq_len

                is_prefill = req_meta.forward_mode == 1

                if can_concretize:
                    stats = self._compute_stats(
                        slice_tensor, name, stage, layer_id, module_type, is_prefill
                    )
                else:
                    stats = self._traced_stats(
                        name,
                        stage,
                        slice_tensor.shape,
                        str(slice_tensor.dtype),
                        layer_id,
                        module_type,
                    )

                result[req_id] = stats

            return result

        except Exception as e:
            result = {}
            for req_meta in request_in_batch:
                result[req_meta.request_id] = {
                    "framework": "jax",
                    "name": name,
                    "stage": stage,
                    "shape": (
                        tuple(tensor.shape) if hasattr(tensor, "shape") else "unknown"
                    ),
                    "dtype": (
                        str(tensor.dtype) if hasattr(tensor, "dtype") else "unknown"
                    ),
                    "error": str(e),
                    "layer_id": "unknown",
                    "module_type": "unknown",
                }
            return result

    def _parse_layer_and_module(self, stage: str):
        layer_id = "unknown"
        module_type = "unknown"

        if self._trace_active:
            logger.debug("Parsing stage: '%s'", stage)

        if "_layer_id_" in stage:
            parts = stage.split("_layer_id_")
            if len(parts) >= 2:
                try:
                    layer_id = int(parts[1].split("_")[0])
                    module_type = parts[0]
                    if self._trace_active:
                        logger.debug(
                            "Parsed from _layer_id_ format: layer_id=%s, module_type=%s",
                            layer_id,
                            module_type,
                        )
                    return layer_id, module_type
                except (ValueError, IndexError) as e:
                    if self._trace_active:
                        logger.debug("Failed to parse _layer_id_ format: %s", e)
                    pass

        if stage:
            stage_lower = stage.lower()
            if "attention" in stage_lower or "attn" in stage_lower:
                module_type = "attention"
            elif "mlp" in stage_lower:
                module_type = "mlp"
            elif "block" in stage_lower:
                module_type = "block"
            elif "transformer" in stage_lower:
                module_type = "transformer"
                layer_id = "all"

            import re

            # Look for layer_id in various formats
            layer_match = re.search(r"layer_id[_-](\d+)", stage, re.IGNORECASE)
            if not layer_match:
                # Also try to find layer id in other formats like "layer_0", "L0", etc.
                layer_match = re.search(r"layer[_-]?(\d+)", stage, re.IGNORECASE)
            if not layer_match:
                # Try "L" followed by digits
                layer_match = re.search(r"L(\d+)", stage, re.IGNORECASE)

            if layer_match:
                with contextlib.suppress(ValueError):
                    layer_id = int(layer_match.group(1))

        return layer_id, module_type

    def _compute_stats(
        self,
        tensor,
        name: str,
        stage: str,
        layer_id,
        module_type,
        is_prefill: bool = False,
    ):
        try:
            shape = tensor.shape
            dtype = str(tensor.dtype)

            std_val = float(jnp.std(tensor, ddof=0).item()) if tensor.size > 1 else 0.0

            stats = {
                "framework": "jax",
                "name": name,
                "stage": stage,
                "shape": tuple(shape),
                "dtype": dtype,
                "min": float(jnp.min(tensor).item()),
                "max": float(jnp.max(tensor).item()),
                "mean": float(jnp.mean(tensor).item()),
                "std": std_val,
                "has_nan": bool(jnp.any(jnp.isnan(tensor)).item()),
                "has_inf": bool(jnp.any(jnp.isinf(tensor)).item()),
                "layer_id": layer_id,
                "module_type": module_type,
            }

            if len(shape) >= 2 and shape[0] > 1 and not is_prefill:
                seq_len = shape[0]

                sample_indices = list(range(seq_len))

                token_stats = []
                for idx in sample_indices:
                    t = tensor[idx]
                    if t.size > 1:
                        token_stats.append(
                            {
                                "token_idx": idx,
                                "min": float(jnp.min(t).item()),
                                "max": float(jnp.max(t).item()),
                                "mean": float(jnp.mean(t).item()),
                                "std": float(jnp.std(t, ddof=0).item()),
                                "has_nan": bool(jnp.any(jnp.isnan(t)).item()),
                                "has_inf": bool(jnp.any(jnp.isinf(t)).item()),
                            }
                        )
                    else:
                        val = t.item()
                        token_stats.append(
                            {
                                "token_idx": idx,
                                "value": float(val),
                                "has_nan": bool(jnp.isnan(t).item()),
                                "has_inf": bool(jnp.isinf(t).item()),
                            }
                        )

                stats["token_stats"] = token_stats
                stats["sequence_length"] = seq_len
            elif len(shape) >= 2 and shape[0] > 1 and is_prefill:
                stats["sequence_length"] = shape[0]

            return stats

        except Exception as e:
            return {
                "framework": "jax",
                "name": name,
                "stage": stage,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "error": str(e),
                "layer_id": layer_id,
                "module_type": module_type,
            }

    def _traced_stats(self, name: str, stage: str, shape, dtype, layer_id, module_type):
        return {
            "framework": "jax",
            "name": name,
            "stage": stage,
            "shape": shape,
            "dtype": dtype,
            "min": "traced",
            "max": "traced",
            "mean": "traced",
            "std": "traced",
            "has_nan": "traced",
            "has_inf": "traced",
            "layer_id": layer_id,
            "module_type": module_type,
            "tracing_context": True,
        }

    def _verbose_logging_console(self, stats: dict[str, Any]):
        if self._trace_active and not self._verbose_logging:
            return

        req_info = ""
        if "request_id" in stats:
            req_id_short = (
                stats["request_id"][:8]
                if len(stats["request_id"]) > 8
                else stats["request_id"]
            )
            req_info = f"[Req:{req_id_short}]"

            if "batch_request_count" in stats:
                req_info += f"[Batch:{stats['batch_request_count']}]"

        if "error" in stats:
            print(
                f"{req_info}[{stats['stage']}] {stats['name']}: shape={stats['shape']}, dtype={stats['dtype']}, error={stats['error']}"
            )
        elif stats.get("tracing_context", False):
            framework = stats["framework"].upper()
            extra = f" {stats.get('extra_info', '')}" if stats.get("extra_info") else ""
            print(
                f"{req_info}[{framework}][{stats['stage']}] {stats['name']}: shape={stats['shape']}, "
                f"dtype={stats['dtype']}, TRACED_CONTEXT{extra}"
            )
        else:
            framework = stats["framework"].upper()
            extra = f" {stats.get('extra_info', '')}" if stats.get("extra_info") else ""
            nan_inf = ""
            if stats["has_nan"]:
                nan_inf += ", HAS_NAN"
            if stats["has_inf"]:
                nan_inf += ", HAS_INF"

            print(
                f"{req_info}[{framework}][{stats['stage']}] {stats['name']}: shape={stats['shape']}, "
                f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
                f"mean={stats['mean']:.6f}, std={stats['std']:.6f}{nan_inf}{extra}"
            )


precision_tracer = PrecisionTracer()
