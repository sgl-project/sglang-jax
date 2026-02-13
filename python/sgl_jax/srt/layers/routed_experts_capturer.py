# Adapted from https://github.com/sgl-project/sglang/pull/12162/files#diff-8e61cb3c05ca6a5e195f011e21ea7544f9f7e08163e3ce4ffa0bacb4b5735259.
# Copyright 2025 The SGLang Authors. All rights reserved.
# Note:
# 1. Remove _RoutedExpertsDeviceCache and _RoutedExpertsHostCache due to at.set error
# 2. The following codes are modified to Jax version according to SGLang codes.

import contextlib
import csv
import datetime
import logging
import os
import threading
import time
from abc import ABC, abstractmethod

import jax
import numpy as np
import pybase64

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_array_size_bytes(t: np.ndarray):
    return np.prod(t.shape) * t.dtype.itemsize


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_tokens: int,
        max_padding: int,
        ep_size: int,
        *,
        enable_balance_debug: bool = False,
        balance_segment_counter: int = 100,
        balance_output_file: str | None = None,
        enable_dist_recorder: bool = False,
        dist_recorder_buffer_size: int = 100,
        dist_recorder_output_file: str | None = None,
        physical_expert_counts: int = 256,
    ):
        if enable or enable_balance_debug or enable_dist_recorder:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_padding=max_padding,
                ep_size=ep_size,
                enable_host_buffer=enable,
                enable_balance_debug=enable_balance_debug,
                balance_segment_counter=balance_segment_counter,
                balance_output_file=balance_output_file,
                enable_dist_recorder=enable_dist_recorder,
                dist_recorder_buffer_size=dist_recorder_buffer_size,
                dist_recorder_output_file=dist_recorder_output_file,
                physical_expert_counts=physical_expert_counts,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    @abstractmethod
    def _sync_fwd_experts_buffer_DtoH(
        self,
        topk_ids_cpu: list[np.ndarray],
        model_worker_batch: ModelWorkerBatch,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        bid: int,
    ):
        raise NotImplementedError

    @abstractmethod
    def on_forward_end(self, topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_padding: int,
        ep_size: int,
        *,
        enable_host_buffer: bool,
        enable_balance_debug: bool,
        balance_segment_counter: int,
        balance_output_file: str | None,
        enable_dist_recorder: bool,
        dist_recorder_buffer_size: int,
        dist_recorder_output_file: str | None,
        physical_expert_counts: int,
    ):
        self.enable_host_buffer = enable_host_buffer
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
        self.num_tokens = num_tokens
        self.max_padding = max_padding

        if self.enable_host_buffer:
            self.host_buffer = np.zeros(
                (
                    self.num_hidden_layers,
                    self.num_tokens,
                    self.num_experts_per_tok,
                ),
                dtype=np.int32,
            )
            # Note: self.dummy_expert_ids is used to models whose some of layers are not MoE, like inclusionAI/Ling-mini-2.0
            self.dummy_experts_ids = np.full(
                (self.max_padding, self.num_experts_per_tok), fill_value=-1, dtype=np.int32
            )
        else:
            self.host_buffer = None
            self.dummy_experts_ids = None

        self._balance_analyzer = None
        if enable_balance_debug:
            if not balance_output_file:
                logger.warning(
                    "Expert balance debug enabled, but output file is not set. "
                    "Disabling expert balance debug."
                )
            else:
                self._balance_analyzer = _ExpertBalanceAnalyzer(
                    num_layers=self.num_hidden_layers,
                    num_experts=physical_expert_counts,
                    topk=self.num_experts_per_tok,
                    ep_size=ep_size,
                    segment_counter=balance_segment_counter,
                    output_file=balance_output_file,
                )

        self._dist_recorder = None
        if enable_dist_recorder:
            if not dist_recorder_output_file:
                logger.warning(
                    "Expert distribution recorder enabled, but output file is not set. "
                    "Disabling recorder."
                )
            else:
                self._dist_recorder = _ExpertDistributionRecorder(
                    num_layers=self.num_hidden_layers,
                    buffer_size=dist_recorder_buffer_size,
                    output_file=dist_recorder_output_file,
                    physical_expert_counts=physical_expert_counts,
                )

        self._balance_missing_topk_warned = False
        self.bid = None

        """Common logging and memory usage computation for captured experts buffers."""
        if self.enable_host_buffer:
            buffer_size_GB = self.get_buffer_size_bytes() / _GB
            logger.info(
                "Routing experts host buffer allocated. #tokens: %d, size: %.2f GB",
                self.num_tokens,
                buffer_size_GB,
            )
        if self._balance_analyzer is not None:
            logger.info(
                "Expert balance debug enabled. Segment size: %d, output: %s",
                self._balance_analyzer.segment_counter,
                self._balance_analyzer.output_file,
            )

    def get_buffer_size_bytes(self):
        if self.host_buffer is None:
            return 0
        return get_array_size_bytes(self.host_buffer)

    def _sync_fwd_experts_buffer_DtoH(
        self,
        topk_ids_cpu: list[np.ndarray],  # padded topk_ids
        model_worker_batch: ModelWorkerBatch,
    ):
        if not self.enable_host_buffer:
            return
        unpadded_input_len = model_worker_batch.get_original_input_len()
        valid_out_cache_loc_cpu = model_worker_batch.out_cache_loc[:unpadded_input_len]
        for layer_idx, ids_cpu in enumerate(topk_ids_cpu):
            if ids_cpu is None:
                valid_ids = self.dummy_experts_ids[:unpadded_input_len]
            else:
                valid_ids = ids_cpu[:unpadded_input_len, : self.num_experts_per_tok]
            self.host_buffer[layer_idx, valid_out_cache_loc_cpu, :] = valid_ids

        self.bid = model_worker_batch.bid

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        bid: int,
    ):
        if not self.enable_host_buffer:
            raise RuntimeError("Host buffer is disabled. enable_return_routed_experts is required.")
        cache_pool_idx = req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1]
        while True:
            if self.bid >= bid:
                return self.host_buffer[:, cache_pool_idx, :]
            else:
                time.sleep(0.001)

    def on_forward_end(self, topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch):
        if not self.enable_host_buffer and self._balance_analyzer is None:
            return
        topk_ids_cpu = jax.device_get(topk_ids)
        if self.enable_host_buffer:
            self._sync_fwd_experts_buffer_DtoH(
                topk_ids_cpu=topk_ids_cpu,
                model_worker_batch=model_worker_batch,
            )
        if self._balance_analyzer is not None:
            if topk_ids_cpu and all(ids is None for ids in topk_ids_cpu):
                if not self._balance_missing_topk_warned:
                    logger.warning(
                        "Expert balance debug is enabled, but topk_ids are None. "
                        "This usually means fused MoE is enabled; no expert balance stats will be recorded."
                    )
                    self._balance_missing_topk_warned = True
                return
            if model_worker_batch.forward_mode.is_decode():
                self._balance_analyzer.add_decode_step(
                    topk_ids_cpu,
                    model_worker_batch.real_bs,
                )
        if self._dist_recorder is not None:
            self._dist_recorder.add_topk_ids(
                topk_ids_cpu,
                model_worker_batch.real_bs,
            )

    def reset(self):
        if self._balance_analyzer is not None:
            # Note: _ExpertBalanceAnalyzer doesn't have a reset but it fills 0 on flush.
            # We could add one if needed.
            pass
        if self._dist_recorder is not None:
            self._dist_recorder.reset()


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def _sync_fwd_experts_buffer_DtoH(
        self,
        topk_ids_cpu: list[np.ndarray],
        model_worker_batch: ModelWorkerBatch,
    ):
        pass

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        bid: int,
    ):
        pass

    def on_forward_end(self, topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch):
        pass

    def reset(self):
        pass


_global_expert_capturer: RoutedExpertsCapturer | None = _RoutedExpertsCapturerNoop()


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer


def extract_routed_experts_from_meta_info(data):
    # To solve the performance issue, we return the experts_ids in base64
    # We left this function for user to change it back to normal int32
    # See detokenizer_manager::_extract_routed_experts
    routed_experts_base64 = data["meta_info"].get("routed_experts", None)
    if routed_experts_base64 is not None:
        routed_experts = np.frombuffer(
            pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=np.int32
        )
        return routed_experts
    return None


class _ExpertBalanceAnalyzer:
    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        topk: int,
        ep_size: int,
        segment_counter: int,
        output_file: str,
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts  # use physical expert counts
        self.topk = topk
        self.ep_size = ep_size
        self.segment_counter = segment_counter
        self.output_file = output_file

        self._counts = np.zeros((num_layers, num_experts), dtype=np.int64)
        self._segment_progress = 0
        self._segment_idx = 0
        self._lock = threading.Lock()
        self._segment_decode_steps = 0
        self._segment_padding_tokens_sum = 0

        self._device_metrics_enabled = False
        self._experts_per_device = None
        if self.ep_size and self.ep_size > 0 and self.num_experts % self.ep_size == 0:
            self._experts_per_device = self.num_experts // self.ep_size
            self._device_metrics_enabled = True
        else:
            logger.warning(
                "Device balance metrics disabled: invalid ep_size=%s for num_experts=%s",
                self.ep_size,
                self.num_experts,
            )

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._header = [
            "timestamp",
            "segment_idx",
            "segment_counter",
            "segment_decode_steps",
            "avg_padding_tokens",
            "layer",
            "num_experts",
            "topk",
            "ep_size",
            "total_assignments",
            "experts_count",
        ]
        self._file = None
        self._writer = None
        self._open_writer(write_header=True)

    def _open_writer(self, write_header: bool):
        if self._file is not None and not self._file.closed:
            with contextlib.suppress(Exception):
                self._file.close()
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._file = open(self.output_file, "a", encoding="utf-8", newline="")  # noqa: SIM115
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(self._header)
            self._file.flush()

    def _ensure_writer(self):
        if self._file is None or self._file.closed:
            self._open_writer(write_header=True)
            return
        if not os.path.exists(self.output_file):
            logger.warning(
                "Expert balance output file missing. Recreating: %s",
                self.output_file,
            )
            self._open_writer(write_header=True)

    def add_decode_step(self, topk_ids_cpu: list[np.ndarray], real_bs: int):
        with self._lock:
            self._segment_decode_steps += 1
            if real_bs:
                flat = topk_ids_cpu[:real_bs, : self.topk].reshape(-1)
                counts = np.bincount(flat, minlength=self.num_experts)
                self._counts += counts

            self._segment_padding_tokens_sum += len(topk_ids_cpu)
            self._segment_progress += 1
            if self._segment_progress >= self.segment_counter:
                self._flush_segment()
                self._counts.fill(0)
                self._segment_progress = 0
                self._segment_idx += 1
                self._segment_decode_steps = 0
                self._segment_padding_tokens_sum = 0

    def _flush_segment(self):
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        self._ensure_writer()
        for layer_idx in range(self.num_layers):
            counts = self._counts[layer_idx]
            total = int(counts.sum())
            experts_count = ",".join(map(str, counts.tolist()))
            row = [
                timestamp,
                self._segment_idx,
                self.segment_counter,
                self._segment_decode_steps,
                (
                    self._segment_padding_tokens_sum / self._segment_decode_steps
                    if self._segment_decode_steps > 0
                    else 0
                ),
                layer_idx,
                self.num_experts,
                self.topk,
                self.ep_size,
                total,
                experts_count,
            ]
            try:
                self._writer.writerow(row)
            except Exception as exc:
                logger.warning(
                    "Failed to write expert balance row. Reopening file: %s (%s)",
                    self.output_file,
                    exc,
                )
                self._open_writer(write_header=True)
                try:
                    self._writer.writerow(row)
                except Exception as exc2:
                    logger.error(
                        "Failed to write expert balance row after reopen: %s",
                        exc2,
                    )
                    continue
        try:
            self._file.flush()
        except Exception as exc:
            logger.warning(
                "Failed to flush expert balance output file. Reopening: %s (%s)",
                self.output_file,
                exc,
            )
            self._open_writer(write_header=True)


class _ExpertDistributionRecorder:
    def __init__(
        self,
        *,
        num_layers: int,
        buffer_size: int,
        output_file: str,
        physical_expert_counts: int,
    ):
        self.num_layers = num_layers
        self.buffer_size = buffer_size
        self.output_file = output_file
        self.physical_expert_counts = physical_expert_counts

        self._physical_counts = np.zeros(
            (self.num_layers, self.physical_expert_counts), dtype=np.int64
        )
        self._steps_accumulated = 0
        self._lock = threading.Lock()

    def add_topk_ids(
        self,
        topk_ids_cpu: list[np.ndarray],
        real_bs: int,
    ):
        if not topk_ids_cpu:
            return

        with self._lock:
            for layer_idx, ids_cpu in enumerate(topk_ids_cpu):
                if ids_cpu is None:
                    continue
                if real_bs <= 0:
                    continue
                ids_chunk = ids_cpu[:real_bs, :].flatten()
                if ids_chunk.size > 0:
                    counts = np.bincount(ids_chunk, minlength=self.physical_expert_counts)
                    self._physical_counts[layer_idx] += counts

            self._steps_accumulated += 1
            if self._steps_accumulated >= self.buffer_size:
                self.dump()
                self._physical_counts.fill(0)
                self._steps_accumulated = 0

    def reset(self):
        with self._lock:
            self._physical_counts.fill(0)
            self._steps_accumulated = 0

    def dump(self):
        from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata

        metadata = get_global_expert_location_metadata()
        if metadata is None:
            logger.warning("No expert location metadata found. Dumping physical counts.")
            logical_counts = self._physical_counts
        else:
            # Move JAX arrays to host once to avoid repeated D2H sync in loop
            phy_to_log_map = jax.device_get(metadata.physical_to_logical_map)
            num_logical = metadata.logical_to_all_physical_map.shape[1]
            logical_counts = np.zeros((self.num_layers, num_logical), dtype=np.int64)

            for layer_idx in range(self.num_layers):
                phy_to_log = phy_to_log_map[layer_idx]
                for p_idx, l_idx in enumerate(phy_to_log):
                    if p_idx < self._physical_counts.shape[1]:
                        logical_counts[layer_idx, l_idx] += self._physical_counts[layer_idx, p_idx]

        # Only process 0 saves to file
        # if jax.process_index() == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        base_path = self.output_file
        if base_path.endswith(".npy"):
            filename = base_path.replace(".npy", f"_{timestamp}.npy")
        else:
            filename = f"{base_path}_{timestamp}.npy"

        output_data = {
            "logical_count": logical_counts,
            "timestamp": timestamp,
        }
        np.save(filename, output_data)
        logger.info("Expert distribution dumped to %s", filename)
