# Adapted from https://github.com/sgl-project/sglang/pull/12162/files#diff-8e61cb3c05ca6a5e195f011e21ea7544f9f7e08163e3ce4ffa0bacb4b5735259.
# Copyright 2025 The SGLang Authors. All rights reserved.
# Note:
# 1. Remove _RoutedExpertsDeviceCache and _RoutedExpertsHostCache due to at.set error
# 2. The following codes are modified to Jax version according to SGLang codes.

import logging
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
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_padding=max_padding,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    @abstractmethod
    def _sync_fwd_experts_buffer_DtoH(
        self,
        topk_ids: list[jax.Array],
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


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_padding: int,
    ):
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
        self.num_tokens = num_tokens
        self.max_padding = max_padding

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
        self.bid = None

        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_GB = self.get_buffer_size_bytes() / _GB
        logger.info(
            "Routing experts host buffer allocated. #tokens: %d, size: %.2f GB",
            self.num_tokens,
            buffer_size_GB,
        )

    def get_buffer_size_bytes(self):
        assert hasattr(self, "host_buffer")
        return get_array_size_bytes(self.host_buffer)

    def _sync_fwd_experts_buffer_DtoH(
        self,
        topk_ids: list[jax.Array],  # padded topk_ids
        model_worker_batch: ModelWorkerBatch,
    ):
        unpadded_input_len = model_worker_batch.get_original_input_len()
        valid_out_cache_loc_cpu = model_worker_batch.out_cache_loc[:unpadded_input_len]
        topk_ids_cpu = jax.device_get(topk_ids)
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
        cache_pool_idx = req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1]
        while True:
            if self.bid >= bid:
                return self.host_buffer[:, cache_pool_idx, :]
            else:
                time.sleep(0.001)

    def on_forward_end(self, topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch):
        self._sync_fwd_experts_buffer_DtoH(
            topk_ids=topk_ids,
            model_worker_batch=model_worker_batch,
        )


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def _sync_fwd_experts_buffer_DtoH(
        self,
        topk_ids: list[jax.Array],
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
