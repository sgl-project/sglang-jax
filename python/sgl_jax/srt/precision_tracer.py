import hashlib
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp

logger = logging.getLogger(__name__)


def _is_jax_array(obj):
    if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
        return False
    return True


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


class PrecisionTracer:
    def __init__(self):
        self.records = {}
        self.lock = threading.Lock()

        # Request coloring and precision tracking
        self._request_traces = {}
        self._current_request_id = None
        self._request_counter = 0
        self._completed_requests_count = 0
        self._request_id_to_number = {}  # Map request_id to request_number
        self._trace_active = False
        self._trace_output_file = None
        self._verbose_logging = False  # Control console output during tracing
        self._enable_precision_tracer = False  # Global enable/disable switch

    def set_enable_precision_tracer(self, enabled: bool):
        """Set the global enable/disable switch for precision tracer"""
        self._enable_precision_tracer = enabled
        logger.info(f"Precision tracer globally {'enabled' if enabled else 'disabled'}")

    def start_trace(
        self,
        req_num: Optional[int] = None,
        output_file: Optional[str] = None,
        verbose_logging: bool = False,
    ):
        """Start tracing requests with optional request number limit and console logging control"""
        if not self._enable_precision_tracer:
            logger.warning(
                "Precision tracer is disabled. Enable with --enable-precision-tracer"
            )
            return None

        if self._trace_active:
            print("Request tracing already active, stopping current trace first...")
            self.stop_trace()

        self._trace_active = True
        self._request_traces = {}
        self._request_counter = 0
        self._completed_requests_count = 0
        self._request_id_to_number = {}  # Reset request ID to number mapping
        self._max_requests = req_num
        self._verbose_logging = verbose_logging

        # Setup output file
        if output_file:
            self._trace_output_file = output_file
        else:
            timestamp = int(time.time())
            self._trace_output_file = f"debug_outputs/request_traces_{timestamp}.jsonl"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self._trace_output_file), exist_ok=True)

        # Clear existing file
        with open(self._trace_output_file, "w") as f:
            pass

        logger.info(f"Request tracing started. Output: {self._trace_output_file}")
        if req_num:
            logger.info(f"Will trace up to {req_num} requests")
        if not verbose_logging:
            logger.info("Verbose console logging disabled during tracing")

    def stop_trace(self):
        """Stop request tracing"""
        if not self._trace_active:
            logger.info("No active request tracing")
            return None

        self._trace_active = False
        output_file = self._trace_output_file
        logger.info(f"Request tracing stopped. Traces saved to: {output_file}")
        return output_file

    def _compute_request_hash(self, req_content):
        """Compute a reproducible hash for request content for comparison purposes"""
        if isinstance(req_content, str):
            content_str = req_content
        elif hasattr(req_content, "origin_input_text"):
            # Handle Req object - always use input_ids as primary identifier for reproducibility
            if (
                hasattr(req_content, "origin_input_ids")
                and req_content.origin_input_ids
            ):
                content_str = str(req_content.origin_input_ids)
            else:
                content_str = req_content.origin_input_text or ""

            # Also include sampling params for uniqueness
            if hasattr(req_content, "sampling_params"):
                content_str += f"_temp{req_content.sampling_params.temperature if req_content.sampling_params else 0}"
                content_str += f"_maxlen{req_content.sampling_params.max_new_tokens if req_content.sampling_params else 0}"
        elif hasattr(req_content, "origin_input_ids"):
            # Handle cases where we have input_ids but no text
            content_str = str(req_content.origin_input_ids)
        else:
            content_str = str(req_content)

        hash_result = hashlib.md5(content_str.encode("utf-8")).hexdigest()[:8]
        return hash_result

    def start_request_trace(self, request_id: Optional[str] = None, req_content=None):
        """Start tracing a new request"""

        if not self._trace_active:
            return None

        # Check if we've completed enough requests FIRST
        if self._max_requests and self._completed_requests_count >= self._max_requests:
            self.stop_trace()
            return None

        # Use provided request_id if available (from scheduler), otherwise generate unique one
        if request_id is None:
            if req_content is not None:
                request_id = str(req_content.rid)  # 直接使用 rid，不加前缀
            else:
                request_id = str(uuid.uuid4())

        # Check if we already have this request in active traces
        if request_id in self._request_traces:
            return request_id

        # Counter is now incremented at scheduler level, not here
        self._current_request_id = request_id

        # Get request number from scheduler assignment
        request_number = self._request_id_to_number.get(request_id, "unknown")

        pid = os.getpid()

        self._request_traces[request_id] = {
            "request_id": request_id,
            "request_number": request_number,
            "start_time": time.time(),
            "precision_records": [],
            "status": "active",
            "content_hash": (
                self._compute_request_hash(req_content) if req_content else None
            ),
            "process_id": pid,
        }

        return request_id

    def end_request_trace(self, request_id: Optional[str] = None):
        """End tracing for a request and save to JSONL"""
        if not self._trace_active:
            return

        if request_id is None:
            request_id = self._current_request_id

        if request_id and request_id in self._request_traces:
            trace_data = self._request_traces[request_id]
            trace_data["end_time"] = time.time()
            trace_data["duration"] = trace_data["end_time"] - trace_data["start_time"]
            trace_data["status"] = "completed"

            # Save to JSONL file
            try:
                with open(self._trace_output_file, "a", encoding="utf-8") as f:
                    json.dump(trace_data, f, cls=TensorJSONEncoder, ensure_ascii=False)
                    f.write("\n")
            except Exception as e:
                logger.error(f"Error saving request trace: {e}")

            # Clean up and increment completed count
            del self._request_traces[request_id]
            self._completed_requests_count += 1
            logger.info(
                f"Request trace completed ({self._completed_requests_count}/{self._max_requests}): {request_id}"
            )
            if request_id == self._current_request_id:
                self._current_request_id = None

    def record(
        self,
        tensor: Any,
        name: str,
        stage: str = "",
        extra_info: str = "",
        request_id: Optional[str] = None,
        forward_batch: Optional[Any] = None,
    ):
        if not self._enable_precision_tracer or not self._trace_active:
            return

        if tensor is None:
            logger.info(f"[{stage}] {name}: None")
            return

        key = f"{stage}_{name}" if stage else name

        # Only handle JAX arrays now
        stats = self._compute_jax_stats(tensor, name, stage, extra_info)

        # Add request coloring - handle batch scenarios
        request_ids_to_process = []

        # Get request IDs from various sources
        if request_id:
            request_ids_to_process = [request_id]
        elif (
            forward_batch
            and hasattr(forward_batch, "trace_request_ids")
            and forward_batch.trace_request_ids
        ):
            request_ids_to_process = forward_batch.trace_request_ids
        elif self._current_request_id:
            request_ids_to_process = [self._current_request_id]

        # If no specific request IDs found, add to all active traces (for JIT callbacks)
        if not request_ids_to_process and self._request_traces:
            request_ids_to_process = list(self._request_traces.keys())
            logger.debug(
                f"No specific request_ids, using all active traces: {request_ids_to_process}"
            )

        # Add stats to all relevant requests
        for req_id in request_ids_to_process:
            if req_id and req_id in self._request_traces:
                # Create a copy of stats for each request
                req_stats = stats.copy()
                req_stats["request_id"] = req_id
                self._request_traces[req_id]["precision_records"].append(req_stats)
            elif req_id:
                logger.warning(
                    f"Request ID mismatch: {req_id} not in traces {list(self._request_traces.keys())}"
                )

        # Set the first request ID for display purposes
        if request_ids_to_process:
            stats["request_id"] = request_ids_to_process[0]
            if len(request_ids_to_process) > 1:
                stats["batch_request_count"] = len(request_ids_to_process)

        with self.lock:
            if key not in self.records:
                self.records[key] = []
            self.records[key].append(stats)

        self._record_stats(stats, key)

    def _compute_jax_stats(
        self, tensor: Any, name: str, stage: str, extra_info: str
    ) -> Dict[str, Any]:
        try:
            try:
                test_scalar = jnp.array(1.0)
                _ = test_scalar.item()
                can_concretize = True
            except Exception:
                can_concretize = False

            if can_concretize:
                if tensor.size > 1:
                    std_val = float(jnp.std(tensor, ddof=0).item())
                else:
                    std_val = 0.0

                stats = {
                    "framework": "jax",
                    "name": name,
                    "stage": stage,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": float(jnp.min(tensor).item()),
                    "max": float(jnp.max(tensor).item()),
                    "mean": float(jnp.mean(tensor).item()),
                    "std": std_val,
                    "has_nan": bool(jnp.any(jnp.isnan(tensor)).item()),
                    "has_inf": bool(jnp.any(jnp.isinf(tensor)).item()),
                    "extra_info": extra_info,
                }

                # Add per-token statistics for sequence data (when first dim > 1)
                if len(tensor.shape) >= 2 and tensor.shape[0] > 1:
                    # Assume first dimension is sequence length (batch size or tokens)
                    seq_len = tensor.shape[0]
                    token_stats = []

                    # Sample a few tokens for detailed analysis (avoid too much data)
                    sample_indices = (
                        [0, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1]
                        if seq_len > 4
                        else list(range(seq_len))
                    )
                    sample_indices = [i for i in sample_indices if 0 <= i < seq_len]

                    for token_idx in sample_indices:
                        token_tensor = tensor[token_idx]
                        if token_tensor.size > 1:
                            token_stats.append(
                                {
                                    "token_idx": token_idx,
                                    "min": float(jnp.min(token_tensor).item()),
                                    "max": float(jnp.max(token_tensor).item()),
                                    "mean": float(jnp.mean(token_tensor).item()),
                                    "std": float(jnp.std(token_tensor, ddof=0).item()),
                                    "has_nan": bool(
                                        jnp.any(jnp.isnan(token_tensor)).item()
                                    ),
                                    "has_inf": bool(
                                        jnp.any(jnp.isinf(token_tensor)).item()
                                    ),
                                }
                            )
                        else:
                            token_stats.append(
                                {
                                    "token_idx": token_idx,
                                    "value": float(token_tensor.item()),
                                    "has_nan": bool(jnp.isnan(token_tensor).item()),
                                    "has_inf": bool(jnp.isinf(token_tensor).item()),
                                }
                            )

                    stats["token_stats"] = token_stats
                    stats["sequence_length"] = seq_len
            else:
                stats = {
                    "framework": "jax",
                    "name": name,
                    "stage": stage,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": "traced",
                    "max": "traced",
                    "mean": "traced",
                    "std": "traced",
                    "has_nan": "traced",
                    "has_inf": "traced",
                    "extra_info": extra_info,
                    "tracing_context": True,
                }

            # Extract layer_id and module_type
            layer_id = "unknown"
            module_type = "unknown"

            if "_layer_id_" in stage:
                parts = stage.split("_layer_id_")
                if len(parts) >= 2:
                    try:
                        layer_id = int(parts[1].split("_")[0])
                        module_type = parts[0]
                    except (ValueError, IndexError):
                        pass
            elif stage:
                stage_lower = stage.lower()
                if "attention" in stage_lower:
                    module_type = "attention"
                elif "mlp" in stage_lower:
                    module_type = "mlp"
                elif "block" in stage_lower:
                    module_type = "block"
                elif "transformer" in stage_lower:
                    module_type = "transformer"
                    layer_id = "all"

                # Extract layer_id (if exists)
                import re

                layer_match = re.search(r"layer_id[_-](\d+)", stage, re.IGNORECASE)
                if layer_match:
                    try:
                        layer_id = int(layer_match.group(1))
                    except ValueError:
                        pass

            stats["layer_id"] = layer_id
            stats["module_type"] = module_type

        except Exception as e:
            stats = {
                "framework": "jax",
                "name": name,
                "stage": stage,
                "shape": tuple(tensor.shape) if hasattr(tensor, "shape") else (),
                "dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else "unknown",
                "extra_info": extra_info,
                "layer_id": "unknown",
                "module_type": "unknown",
                "error": str(e),
            }

        return stats

    def _record_stats(self, stats: Dict[str, Any], key: str):
        # Skip console output if verbose logging is disabled during tracing
        if self._trace_active and not self._verbose_logging:
            return

        # Add request ID info if available
        req_info = ""
        if "request_id" in stats:
            req_id_short = (
                stats["request_id"][:8]
                if len(stats["request_id"]) > 8
                else stats["request_id"]
            )
            req_info = f"[Req:{req_id_short}]"

            # Add batch info if multiple requests
            if "batch_request_count" in stats:
                req_info += f"[Batch:{stats['batch_request_count']}]"

        if "error" in stats:
            print(
                f"{req_info}[{stats['stage']}] {stats['name']}: shape={stats['shape']}, dtype={stats['dtype']}, error={stats['error']}"
            )
        elif stats.get("tracing_context", False):
            # Special handling in JAX tracing context
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

    def get_records(self, key: str = None) -> Union[Dict[str, List], List]:
        with self.lock:
            if key is None:
                return dict(self.records)
            return self.records.get(key, [])

    def clear_records(self):
        with self.lock:
            self.records.clear()


precision_tracer = PrecisionTracer()
