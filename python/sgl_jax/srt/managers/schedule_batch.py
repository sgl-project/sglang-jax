from __future__ import annotations

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
"""
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.

TODO(lmzheng): ModelWorkerBatch seems a bit redundant and we consider removing it in the future.
"""

import jax
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

@dataclasses.dataclass
class ModelWorkerBatch:
    # The batch id
    bid: int
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: jax.Array
    # The sequence length
    seq_lens: jax.Array
    # cache loc
    cache_loc: jax.Array
    # The indices of output tokens in the token_to_kv_pool_allocator
    out_cache_loc: jax.Array
    # Position information [total_tokens]
    positions: jax.Array = None
    # Start position for each sequence in extend mode [batch_size]
    extend_start_loc: jax.Array = None
