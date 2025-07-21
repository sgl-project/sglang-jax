# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ModelRunner runs the forward passes of the models."""

import jax
import logging
from typing import Optional, Tuple, Union

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.mem_cache.memory_pool import (
    ReqToTokenPool,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        server_args: ServerArgs,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
    ):
        pass

    def initialize(self, min_per_gpu_memory: float):
        pass

    def model_specific_adjustment(self):
        pass

    def load_model(self):
        pass

    def profile_max_num_token(self, total_gpu_memory: int):
        pass

    def init_memory_pool(
        self,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        pass

    def forward_decode(
        self, forward_batch: ForwardBatch
    ) -> LogitsProcessorOutput:
        pass

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
    ) -> LogitsProcessorOutput:
        pass

    def forward_idle(
        self, forward_batch: ForwardBatch
    ) -> LogitsProcessorOutput:
        pass

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
    ) -> Tuple[Union[LogitsProcessorOutput], bool]:
        pass

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
    ) -> Tuple[Union[LogitsProcessorOutput], bool]:
        pass

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        pass

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> jax.Array:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        pass
