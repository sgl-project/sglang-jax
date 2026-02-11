"""Audio Backbone Model Worker for MiMo Audio."""

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import MiMoSamplerConfig
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.audio.audio_backbone_model_runner import (
    AudioBackboneModelRunner,
)
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

MIMO_GROUP_SIZE = 4


class AudioBackboneModelWorker:
    def __init__(self, server_args: ServerArgs = None, mesh=None, model_class=None):
        self.mesh = mesh
        self.model_runner = AudioBackboneModelRunner(server_args, mesh, model_class=model_class)
        self._cache_locations: dict[str, np.ndarray] = {}
        self._total_seq_lens: dict[str, int] = {}

    def reset_cache_state(self, rid: str = None):
        if rid is not None:
            self._cache_locations.pop(rid, None)
            self._total_seq_lens.pop(rid, None)
        else:
            self._cache_locations.clear()
            self._total_seq_lens.clear()

    def _create_forward_batch(
        self,
        input_ids: jax.Array,
        rid: str,
        positions_override: jax.Array = None,
        is_prefill: bool = True,
    ) -> ForwardBatch:
        B, _, seq_len = input_ids.shape
        T_groups = seq_len // MIMO_GROUP_SIZE

        existing_cache_locs = self._cache_locations.get(rid, np.array([], dtype=np.int32))
        existing_seq_len = self._total_seq_lens.get(rid, 0)

        new_cache_loc = self.model_runner.token_to_kv_pool_allocator.alloc(T_groups * B)
        if new_cache_loc is None:
            raise RuntimeError("Failed to allocate KV cache slots")

        all_cache_locs = np.concatenate([existing_cache_locs, new_cache_loc])
        total_seq_len = existing_seq_len + T_groups

        self._cache_locations[rid] = all_cache_locs
        self._total_seq_lens[rid] = total_seq_len

        if positions_override is not None:
            positions = positions_override
            if existing_seq_len == 0:
                forward_mode = ForwardMode.EXTEND
                extend_start_loc = jnp.array([0], dtype=jnp.int32)
                extend_seq_lens = jnp.array([T_groups] * B, dtype=jnp.int32)
                extend_prefix_lens = jnp.zeros(B, dtype=jnp.int32)
            else:
                forward_mode = ForwardMode.DECODE
                extend_start_loc = None
                extend_seq_lens = None
                extend_prefix_lens = None
        elif is_prefill:
            positions = jnp.arange(T_groups, dtype=jnp.int32)
            forward_mode = ForwardMode.EXTEND
            extend_start_loc = jnp.array([0], dtype=jnp.int32)
            extend_seq_lens = jnp.array([T_groups] * B, dtype=jnp.int32)
            extend_prefix_lens = jnp.zeros(B, dtype=jnp.int32)
        else:
            positions = jnp.array([existing_seq_len], dtype=jnp.int32)
            forward_mode = ForwardMode.DECODE
            extend_start_loc = None
            extend_seq_lens = None
            extend_prefix_lens = None

        req_pool_indices = np.arange(B, dtype=np.int32)
        seq_lens_arr = np.array([total_seq_len] * B, dtype=np.int32)

        return ForwardBatch(
            bid=0,
            forward_mode=forward_mode,
            batch_size=B,
            input_ids=None,
            positions=positions,
            req_pool_indices=jnp.array(req_pool_indices),
            seq_lens=jnp.array(seq_lens_arr),
            out_cache_loc=jnp.array(new_cache_loc),
            cache_loc=jnp.array(all_cache_locs),
            extend_start_loc=extend_start_loc,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            attn_backend=self.model_runner.attn_backend,
        )

    def _create_logits_metadata(
        self,
        batch_size: int,
        seq_len: int,
        is_prefill: bool = True,
    ) -> LogitsMetadata:
        if is_prefill:
            return LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                extend_seq_lens=jnp.array([seq_len] * batch_size, dtype=jnp.int32),
                extend_seq_lens_cpu=[seq_len] * batch_size,
                top_logprobs_nums=[0] * batch_size,
            )
        else:
            return LogitsMetadata(
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                top_logprobs_nums=[0] * batch_size,
            )

    def forward(
        self,
        batch: Req,
        cache: Optional[list] = None,
        **kwargs,
    ):
        input_ids = batch.input_ids
        rid = batch.rid

        if input_ids is None:
            raise ValueError("input_ids must be provided (should be pre-aggregated by Req)")

        if kwargs.get("reset_cache", False):
            self.reset_cache_state(rid)

        if not jnp.issubdtype(input_ids.dtype, jnp.integer):
            input_ids = input_ids.astype(jnp.int32)

        if input_ids.ndim == 2:
            input_ids = input_ids[None, :, :]

        seq_len = input_ids.shape[2]
        if seq_len % MIMO_GROUP_SIZE != 0:
            pad_len = MIMO_GROUP_SIZE - (seq_len % MIMO_GROUP_SIZE)
            input_ids = jnp.pad(
                input_ids,
                ((0, 0), (0, 0), (0, pad_len)),
                constant_values=0,
            )

        B, _, padded_seq_len = input_ids.shape
        T_groups = padded_seq_len // MIMO_GROUP_SIZE

        positions_override = kwargs.get("positions", None)
        existing_seq_len = self._total_seq_lens.get(rid, 0)
        is_prefill = existing_seq_len == 0

        forward_batch = self._create_forward_batch(input_ids, rid, positions_override, is_prefill)
        logits_metadata = self._create_logits_metadata(B, T_groups, forward_batch.forward_mode.is_extend())

        return self.model_runner.forward(input_ids, forward_batch, logits_metadata, **kwargs)

    def patch_decode(
        self,
        local_embeds: jax.Array,
        key: jax.Array,
        sampler_config: Optional[MiMoSamplerConfig] = None,
    ):
        return self.model_runner.patch_decode(local_embeds, key, sampler_config)
