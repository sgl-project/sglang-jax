# MTP Refactor Phase 1 PR1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the PR1 architecture skeleton for the MTP refactor by replacing the old monolithic `EAGLEWorker` with explicit spec-worker and draft-worker boundaries.

**Architecture:** `EAGLEWorker` becomes a composition-only spec worker that owns target-side orchestration and delegates all draft model work to `EagleDraftWorker`. `BaseSpecWorker` and `BaseDraftWorker` define the shared contract that later PRs will reuse for `MultiLayerEAGLEWorker` and `MultiLayerDraftWorker` without preserving the old EAGLE implementation as a compatibility path.

**Tech Stack:** Python 3.11, JAX, Flax NNX, pytest/unittest, existing `sgl_jax.srt` scheduler/model-worker abstractions.

---

## Scope and Constraints

- PR1 does not implement runnable `mimo-v2.5-pro` MTP.
- PR1 does not add scheduler overlap support.
- PR1 does not preserve old `EAGLEWorker` internal behavior as a second path.
- PR1 keeps the public scheduler entrypoint `forward_batch_speculative_generation(model_worker_batch)`.
- PR1 adds `verify(model_worker_batch, verify_input, cur_allocate_lens=None)` to `BaseSpecWorker` as a PR1 refinement of the RFC so target verify/data contract is explicit and reusable by later MultiLayer work.
- PR1 keeps current `EagleDraftInput` and `EagleVerifyInput` field names, including existing `retrive_*` spellings.
- PR1 should keep current non-overlap speculative precompile entrypoints callable: `run_spec_decode_precompile()`, `precompile_spec_extend()`, and `precompile_spec_decode()`.

## File Structure

- Create: `python/sgl_jax/srt/speculative/base_spec_worker.py`
  - Owns abstract `BaseDraftWorker` and `BaseSpecWorker` interfaces.
  - Contains no JAX logic and no scheduler logic.

- Create: `python/sgl_jax/srt/speculative/eagle_draft_worker.py`
  - Owns draft worker initialization, draft model embedding/head setup, draft prefill, draft decode, draft extend, draft precompile, and draft utility calls.
  - Imports draft selection helpers from `spec_utils.py`, not from `eagle_worker.py`, to avoid circular imports.

- Create: `python/sgl_jax/srt/speculative/spec_utils.py`
  - Owns reusable speculative decode helpers that are not data structures: `topk_probs_from_logits`, `fast_topk`, `update_eagle_lists`, `update_forward_batch_info`, and `select_top_k_tokens*`.
  - Mirrors upstream SGLang's `sglang.srt.speculative.spec_utils` responsibility split.
  - May depend on JAX/NumPy and low-level types only; must not import `eagle_worker`, `EAGLEWorker`, or `EagleDraftWorker`.

- Modify: `python/sgl_jax/srt/speculative/eagle_worker.py`
  - Becomes a `BaseSpecWorker` implementation.
  - Owns target prefill, target verify, result assembly, logprob post-processing, and delegation to `EagleDraftWorker`.
  - Removes direct `ModelWorker` inheritance and direct draft runner ownership.

- Modify: `python/sgl_jax/srt/speculative/eagle_util.py`
  - Adds small contract helpers to `EagleDraftInput` and `EagleVerifyInput` so scheduler/model code can distinguish draft and verify inputs without worker-type checks.
  - Keeps data shape semantics stable for PR1.
  - Documents PR1 DP contract semantics: draft-local order remains inside draft state; target verify/result fields must be treated as explicit order/layout boundaries even though full DP attention support lands later.

- Modify: `python/sgl_jax/srt/speculative/spec_info.py`
  - Adds common `SpecInput`/`SpecInputType` base contract mirroring the upstream SPEC_V2 shape enough for PR1.

- Modify: `python/sgl_jax/srt/managers/scheduler.py`
  - Keeps scheduler selection pointed at `EAGLEWorker`.
  - No old/new worker branching.

- Test: `python/sgl_jax/test/speculative/test_base_spec_worker.py`
  - Verifies abstract contracts and minimal concrete fakes.

- Test: `python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py`
  - Verifies importability, inheritance/delegation shape, and contract methods without requiring TPU runtime.

---

### Task 1: Add base worker contract tests

**Files:**
- Create: `python/sgl_jax/test/speculative/test_base_spec_worker.py`
- Later modify: `python/sgl_jax/srt/speculative/base_spec_worker.py`

- [ ] **Step 1: Write the failing test**

Create `python/sgl_jax/test/speculative/test_base_spec_worker.py`:

```python
import pytest

from sgl_jax.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker


class ConcreteDraftWorker(BaseDraftWorker):
    def draft(self, model_worker_batch):
        return ("draft", model_worker_batch)

    def draft_extend_for_prefill(self, model_worker_batch, target_hidden_states, next_token_ids):
        return ("prefill", model_worker_batch, target_hidden_states, next_token_ids)

    def draft_extend_for_decode(self, model_worker_batch, batch_output):
        return ("decode", model_worker_batch, batch_output)


class ConcreteSpecWorker(BaseSpecWorker):
    def __init__(self):
        self._target_worker = object()
        self._draft_worker = ConcreteDraftWorker()

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def forward_batch_speculative_generation(self, model_worker_batch):
        return self.draft_worker.draft(model_worker_batch)

    def verify(self, model_worker_batch, verify_input, cur_allocate_lens=None):
        return ("verify", model_worker_batch, verify_input, cur_allocate_lens)


def test_base_draft_worker_requires_public_methods():
    with pytest.raises(TypeError):
        BaseDraftWorker()


def test_base_spec_worker_requires_public_methods():
    with pytest.raises(TypeError):
        BaseSpecWorker()


def test_concrete_workers_expose_contract():
    worker = ConcreteSpecWorker()

    assert worker.target_worker is worker._target_worker
    assert isinstance(worker.draft_worker, BaseDraftWorker)
    assert worker.forward_batch_speculative_generation("batch") == ("draft", "batch")
    assert worker.verify("batch", "verify_input", "alloc") == (
        "verify",
        "batch",
        "verify_input",
        "alloc",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_base_spec_worker.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sgl_jax.srt.speculative.base_spec_worker'`.

- [ ] **Step 3: Implement the base contract**

Create `python/sgl_jax/srt/speculative/base_spec_worker.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.managers.tp_worker import ModelWorker
    from sgl_jax.srt.speculative.eagle_util import EagleVerifyInput


class BaseDraftWorker(ABC):
    @abstractmethod
    def draft(self, model_worker_batch: ModelWorkerBatch) -> EagleVerifyInput:
        raise NotImplementedError

    @abstractmethod
    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        target_hidden_states,
        next_token_ids,
    ):
        raise NotImplementedError

    @abstractmethod
    def draft_extend_for_decode(
        self,
        model_worker_batch: ModelWorkerBatch,
        batch_output: GenerationBatchResult,
    ) -> None:
        raise NotImplementedError


class BaseSpecWorker(ABC):
    @property
    @abstractmethod
    def target_worker(self) -> ModelWorker:
        raise NotImplementedError

    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        raise NotImplementedError

    @abstractmethod
    def forward_batch_speculative_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> GenerationBatchResult:
        raise NotImplementedError

    @abstractmethod
    def verify(
        self,
        model_worker_batch: ModelWorkerBatch,
        verify_input: EagleVerifyInput,
        cur_allocate_lens=None,
    ) -> GenerationBatchResult:
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_base_spec_worker.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/sgl_jax/srt/speculative/base_spec_worker.py python/sgl_jax/test/speculative/test_base_spec_worker.py
git commit -m "refactor: add speculative worker base contracts"
```

---

### Task 2: Add SpecInput type contract

**Files:**
- Modify: `python/sgl_jax/srt/speculative/spec_info.py`
- Modify: `python/sgl_jax/srt/speculative/eagle_util.py`
- Create: `python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py`

- [ ] **Step 1: Write the failing contract test**

Create `python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py`:

```python
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput
from sgl_jax.srt.speculative.spec_info import SpecInput, SpecInputType


def test_eagle_draft_input_is_spec_input():
    draft_input = EagleDraftInput.create_idle_input(
        hidden_size=16,
        dtype=np.float32,
        topk=1,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )

    assert isinstance(draft_input, SpecInput)
    assert draft_input.spec_input_type == SpecInputType.EAGLE_DRAFT
    assert draft_input.is_draft_input()
    assert not draft_input.is_verify_input()
    assert draft_input.get_spec_adjust_token_coefficient() == (1, 1)


def test_eagle_verify_input_is_spec_input():
    verify_input = EagleVerifyInput(
        draft_token=jnp.empty((0,), dtype=jnp.int32),
        custom_mask=jnp.empty((0,), dtype=jnp.bool_),
        positions=jnp.empty((0,), dtype=jnp.int32),
        retrive_index=jnp.empty((0, 0), dtype=jnp.int32),
        retrive_next_token=jnp.empty((0, 0), dtype=jnp.int32),
        retrive_next_sibling=jnp.empty((0, 0), dtype=jnp.int32),
        retrive_cum_len=None,
        seq_lens_cpu=np.empty((0,), dtype=np.int32),
        spec_steps=3,
        topk=1,
        draft_token_num=4,
        seq_lens_sum=0,
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )

    assert isinstance(verify_input, SpecInput)
    assert verify_input.spec_input_type == SpecInputType.EAGLE_VERIFY
    assert not verify_input.is_draft_input()
    assert verify_input.is_verify_input()
    assert verify_input.get_spec_adjust_token_coefficient() == (4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py -q
```

Expected: FAIL with `ImportError` for `SpecInput` or `SpecInputType`.

- [ ] **Step 3: Add SpecInput and SpecInputType**

Modify `python/sgl_jax/srt/speculative/spec_info.py` so it contains these definitions after `SpeculativeAlgorithm`:

```python
from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Tuple


class SpecInputType(IntEnum):
    EAGLE_DRAFT = auto()
    EAGLE_VERIFY = auto()


class SpecInput(ABC):
    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type

    def is_draft_input(self) -> bool:
        return self.spec_input_type == SpecInputType.EAGLE_DRAFT

    def is_verify_input(self) -> bool:
        return self.spec_input_type == SpecInputType.EAGLE_VERIFY

    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        raise NotImplementedError
```

If `spec_info.py` already imports from `enum`, merge the import rather than duplicating it.

- [ ] **Step 4: Wire EagleDraftInput to SpecInput**

Modify imports near the top of `python/sgl_jax/srt/speculative/eagle_util.py`:

```python
from sgl_jax.srt.speculative.spec_info import SpecInput, SpecInputType
```

Change the class definition and add `__post_init__` plus token coefficient:

```python
@register_pytree_node_class
@dataclass
class EagleDraftInput(SpecInput):
    ALLOC_LEN_PER_DECODE: ClassVar[int] = None

    def __post_init__(self):
        SpecInput.__init__(self, SpecInputType.EAGLE_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> tuple[int, int]:
        return 1, 1
```

Keep the existing fields and methods exactly as they are after `ALLOC_LEN_PER_DECODE`; insert `__post_init__` and `get_spec_adjust_token_coefficient()` before `tree_flatten()`.

- [ ] **Step 5: Preserve `spec_input_type` through pytree unflatten for EagleDraftInput**

In `EagleDraftInput.tree_unflatten()`, set `spec_input_type` on the constructed object:

```python
obj.spec_input_type = SpecInputType.EAGLE_DRAFT
```

Place it immediately after `obj = cls.__new__(cls)`.

- [ ] **Step 6: Wire EagleVerifyInput to SpecInput**

Change the class definition in `python/sgl_jax/srt/speculative/eagle_util.py`:

```python
@register_pytree_node_class
@dataclass
class EagleVerifyInput(SpecInput):
    draft_token: jax.Array
```

Insert these methods before `tree_flatten()`:

```python
    def __post_init__(self):
        SpecInput.__init__(self, SpecInputType.EAGLE_VERIFY)

    def get_spec_adjust_token_coefficient(self) -> tuple[int, int]:
        return self.draft_token_num, self.draft_token_num
```

In `EagleVerifyInput.tree_unflatten()`, set:

```python
obj.spec_input_type = SpecInputType.EAGLE_VERIFY
```

Place it immediately after `obj = cls.__new__(cls)`.

- [ ] **Step 7: Run tests**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py python/sgl_jax/test/speculative/test_base_spec_worker.py -q
```

Expected: PASS.

- [ ] **Step 8: Add PR1 DP contract notes**

In `python/sgl_jax/srt/speculative/eagle_util.py`, add one-line docstrings to the input classes:

```python
@dataclass
class EagleDraftInput(SpecInput):
    """Cross-round draft state kept in draft-local request order."""
```

```python
@dataclass
class EagleVerifyInput(SpecInput):
    """Target verify input; fields define the explicit verify token/layout boundary."""
```

These docstrings are the PR1 DP contract marker: full DP padded layout handling is not implemented in PR1, but consumers must not infer request order from worker type.

- [ ] **Step 9: Commit**

```bash
git add python/sgl_jax/srt/speculative/spec_info.py python/sgl_jax/srt/speculative/eagle_util.py python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py
git commit -m "refactor: add speculative input contracts"
```

---

### Task 3: Extract EagleDraftWorker shell and initialization

**Files:**
- Modify: `python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py`
- Create: `python/sgl_jax/srt/speculative/eagle_draft_worker.py`
- Modify: `python/sgl_jax/srt/speculative/eagle_worker.py`

- [ ] **Step 1: Add failing import and inheritance test**

Append to `python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py`:

```python
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker


def test_eagle_draft_worker_implements_base_contract():
    assert issubclass(EagleDraftWorker, BaseDraftWorker)
    assert EagleDraftWorker.__mro__.index(ModelWorker) < EagleDraftWorker.__mro__.index(BaseDraftWorker)
    for name in ("draft", "draft_extend_for_prefill", "draft_extend_for_decode"):
        assert callable(getattr(EagleDraftWorker, name))


def test_eagle_worker_implements_base_contract():
    assert issubclass(EAGLEWorker, BaseSpecWorker)
    for name in ("target_worker", "draft_worker", "forward_batch_speculative_generation", "verify"):
        assert hasattr(EAGLEWorker, name)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `eagle_draft_worker` or `AssertionError` because `EAGLEWorker` is not a `BaseSpecWorker`.

- [ ] **Step 3: Create EagleDraftWorker with initialization only**

Create `python/sgl_jax/srt/speculative/eagle_draft_worker.py`:

```python
import logging

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.speculative.base_spec_worker import BaseDraftWorker
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)
RETURN_ORIGINAL_LOGPROB = get_bool_env_var("RETURN_ORIGINAL_LOGPROB")


class EagleDraftWorker(ModelWorker, BaseDraftWorker):
    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self._target_worker = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()
        self.hot_token_ids = None
        self.num_new_pages_per_topk = None
        self.extend_lens = None

        ModelWorker.__init__(
            self,
            server_args,
            target_worker.mesh,
            req_to_token_pool=self.req_to_token_pool,
            is_draft_worker=True,
        )
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )
        self._init_embed_and_head()
        self.model_runner.initialize_jit()
        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = self._target_worker.get_precompile_paddings()
        self.precompile_bs_paddings = precompile_bs_paddings
        self.precompile_cache_loc_paddings = precompile_cache_loc_paddings
        self.precompile_token_paddings = precompile_token_paddings

    @property
    def target_worker(self):
        return self._target_worker

    def _init_embed_and_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            if self.draft_model_runner.model.hot_token_ids is not None:
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(
                        NamedSharding(self.model_runner.mesh, P())
                        if jax.process_count() == 1
                        else None
                    ),
                )
        else:
            if self.hot_token_ids is not None:
                head = head.clone()
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(
                        NamedSharding(self.model_runner.mesh, P())
                        if jax.process_count() == 1
                        else None
                    ),
                )
                head.data = head.data[self.hot_token_ids]
            self.draft_model_runner.model.set_embed_and_head(embed, head)

    def draft(self, model_worker_batch):
        raise NotImplementedError

    def draft_extend_for_prefill(self, model_worker_batch, target_hidden_states, next_token_ids):
        raise NotImplementedError

    def draft_extend_for_decode(self, model_worker_batch, batch_output):
        raise NotImplementedError
```

- [ ] **Step 4: Make EAGLEWorker implement BaseSpecWorker and delegate draft ownership**

At the top of `python/sgl_jax/srt/speculative/eagle_worker.py`, add imports:

```python
from sgl_jax.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
```

Change the class declaration:

```python
class EAGLEWorker(BaseSpecWorker):
```

Replace the start of `__init__` with:

```python
    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self._target_worker = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()
        self._draft_worker = EagleDraftWorker(server_args, target_worker)
        self.precompile_bs_paddings = self._draft_worker.precompile_bs_paddings
        self.precompile_cache_loc_paddings = self._draft_worker.precompile_cache_loc_paddings
        self.precompile_token_paddings = self._draft_worker.precompile_token_paddings
```

Add properties after `__init__`:

```python
    @property
    def target_worker(self) -> ModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> BaseDraftWorker:
        return self._draft_worker

    @property
    def mesh(self):
        return self.target_worker.mesh

    @property
    def model_config(self):
        return self.target_worker.model_config
```

Remove the old `super().__init__(..., is_draft_worker=True)` block and old draft embedding/head initialization from `EAGLEWorker.__init__` because `EagleDraftWorker` now owns those.

- [ ] **Step 5: Run import contract tests**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sgl_jax/srt/speculative/eagle_draft_worker.py python/sgl_jax/srt/speculative/eagle_worker.py python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py
git commit -m "refactor: introduce eagle draft worker"
```

---

### Task 4: Move draft prefill into EagleDraftWorker

**Files:**
- Modify: `python/sgl_jax/srt/speculative/eagle_draft_worker.py`
- Modify: `python/sgl_jax/srt/speculative/eagle_worker.py`

- [ ] **Step 1: Add method presence test**

Append to `python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py`:

```python
def test_eagle_worker_does_not_own_draft_prefill_method():
    assert hasattr(EagleDraftWorker, "draft_extend_for_prefill")
    assert not hasattr(EAGLEWorker, "forward_draft_extend")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py::test_eagle_worker_does_not_own_draft_prefill_method -q
```

Expected: FAIL because `EAGLEWorker.forward_draft_extend` still exists.

- [ ] **Step 3: Move draft prefill imports to eagle_draft_worker.py**

Ensure `python/sgl_jax/srt/speculative/eagle_draft_worker.py` imports:

```python
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
```

- [ ] **Step 4: Implement draft_extend_for_prefill**

Move the body of current `EAGLEWorker.forward_draft_extend()` into `EagleDraftWorker.draft_extend_for_prefill()` and adjust `self.draft_model_runner` references to remain on `EagleDraftWorker`:

```python
    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        target_hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ) -> EagleDraftInput:
        model_worker_batch.spec_info = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids[: model_worker_batch.real_bs],
            num_tokens_per_batch=np.asarray(1, dtype=jnp.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=jnp.int32),
            allocate_lens=model_worker_batch.seq_lens,
        )
        model_worker_batch.return_hidden_states = False
        model_worker_batch.spec_info.prepare_for_extend_after_target_prefill(
            model_worker_batch=model_worker_batch
        )
        model_worker_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.return_logprob = False
        forward_metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )
        self.draft_model_runner.attn_backend.forward_metadata = forward_metadata
        forward_batch.forward_mode = ForwardMode.EXTEND

        logits_output, _, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(
                model_worker_batch, self.mesh
            ),
        )
        logits_output.next_token_logits = logits_output.next_token_logits[
            : model_worker_batch.real_bs, :
        ]
        if len(logits_output.hidden_states.shape) == 1:
            logits_output.hidden_states = jnp.expand_dims(logits_output.hidden_states, axis=0)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.allocate_lens = model_worker_batch.seq_lens[
            : model_worker_batch.real_bs
        ]

        self.capture_for_decode(logits_output, forward_batch.spec_info)
        return forward_batch.spec_info
```

If `self.mesh` is not defined on `EagleDraftWorker`, add:

```python
    @property
    def mesh(self):
        return self.target_worker.mesh
```

- [ ] **Step 5: Delegate prefill from EAGLEWorker**

In `EAGLEWorker.forward_batch_speculative_generation()`, replace:

```python
            self.forward_draft_extend(
                model_worker_batch, logits_output.hidden_states, next_token_ids
            )
```

with:

```python
            next_draft_input = self.draft_worker.draft_extend_for_prefill(
                model_worker_batch, logits_output.hidden_states, next_token_ids
            )
```

Then change `GenerationBatchResult` construction:

```python
                next_draft_input=next_draft_input,
```

Delete `EAGLEWorker.forward_draft_extend()` after delegation is in place.

- [ ] **Step 6: Run contract tests**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/sgl_jax/srt/speculative/eagle_draft_worker.py python/sgl_jax/srt/speculative/eagle_worker.py python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py
git commit -m "refactor: move eagle draft prefill"
```

---

### Task 5: Move draft decode helpers into spec_utils and draft decode into EagleDraftWorker

**Files:**
- Create: `python/sgl_jax/srt/speculative/spec_utils.py`
- Modify: `python/sgl_jax/srt/speculative/eagle_draft_worker.py`
- Modify: `python/sgl_jax/srt/speculative/eagle_worker.py`

- [ ] **Step 1: Add method ownership test**

Append to `python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py`:

```python
def test_eagle_worker_does_not_own_draft_decode_methods():
    for name in ("draft", "draft_forward", "padding_for_decode", "capture_for_decode"):
        assert hasattr(EagleDraftWorker, name)
        assert not hasattr(EAGLEWorker, name)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py::test_eagle_worker_does_not_own_draft_decode_methods -q
```

Expected: FAIL because these methods still live on `EAGLEWorker`.

- [ ] **Step 3: Move draft decode imports**

Ensure `eagle_draft_worker.py` imports:

```python
import functools

from sgl_jax.srt.speculative.eagle_util import (
    EagleDraftInput,
    EagleVerifyInput,
    build_tree_kernel_efficient,
    build_tree_mask_for_draft_decode,
)
```

Move helper functions `topk_probs_from_logits`, `fast_topk`, `select_top_k_tokens`, `select_top_k_tokens_step_0`, `select_top_k_tokens_step_greater_0`, `update_eagle_lists`, and `update_forward_batch_info` from `eagle_worker.py` into a new `python/sgl_jax/srt/speculative/spec_utils.py`. `EagleDraftWorker` must import these helpers from `sgl_jax.srt.speculative.spec_utils`; do not import helper functions from `eagle_worker.py`.

- [ ] **Step 4: Move these methods from EAGLEWorker to EagleDraftWorker unchanged except self references**

Move the following methods from `python/sgl_jax/srt/speculative/eagle_worker.py` into `python/sgl_jax/srt/speculative/eagle_draft_worker.py`:

```text
copy_model_worker_batch_to_cpu
capture_for_decode
get_padding_bs
padding_for_decode
draft
draft_forward
run_spec_decode_precompile
precompile_spec_extend
precompile_spec_decode
```

After moving, ensure `draft()` returns `EagleVerifyInput` and does not mutate `EAGLEWorker` state. The method body should keep this final construction:

```python
        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            seq_lens_cpu=seq_lens_cpu,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            seq_lens_sum=model_worker_batch.seq_lens_sum,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
```

- [ ] **Step 5: Delegate decode draft from EAGLEWorker**

In `EAGLEWorker.forward_batch_speculative_generation()` decode branch, replace:

```python
            self.draft(model_worker_batch)

            batch_output = self.verify(model_worker_batch, cur_allocate_lens)
```

with:

```python
            verify_input = self.draft_worker.draft(model_worker_batch)
            model_worker_batch.spec_info = verify_input

            batch_output = self.verify(model_worker_batch, verify_input, cur_allocate_lens)
```

Adjust `EAGLEWorker.verify()` signature to:

```python
    def verify(
        self,
        model_worker_batch: ModelWorkerBatch,
        verify_input: EagleVerifyInput,
        cur_allocate_lens: jax.Array,
    ):
```

Inside `verify()`, remove the local assignment from `model_worker_batch.spec_info` and use the explicit argument:

```python
        spec_info = verify_input
```

- [ ] **Step 6: Delegate precompile from scheduler-compatible methods**

Keep these methods on `EAGLEWorker` as delegating wrappers so scheduler does not need to change in PR1:

```python
    def run_spec_decode_precompile(self):
        return self.draft_worker.run_spec_decode_precompile()

    def precompile_spec_extend(self):
        return self.draft_worker.precompile_spec_extend()

    def precompile_spec_decode(self):
        return self.draft_worker.precompile_spec_decode()
```

- [ ] **Step 7: Run contract tests**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py -q
```

Expected: PASS.

- [ ] **Step 8: Run existing speculative utility tests**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_utils.py python/sgl_jax/test/speculative/test_eagle_tree_build.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add python/sgl_jax/srt/speculative/spec_utils.py python/sgl_jax/srt/speculative/eagle_draft_worker.py python/sgl_jax/srt/speculative/eagle_worker.py python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py
git commit -m "refactor: move eagle draft decode helpers"
```

---

### Task 6: Move draft extend after verify into EagleDraftWorker

**Files:**
- Modify: `python/sgl_jax/srt/speculative/eagle_draft_worker.py`
- Modify: `python/sgl_jax/srt/speculative/eagle_worker.py`

- [ ] **Step 1: Add method ownership test**

Append to `python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py`:

```python
def test_eagle_worker_does_not_own_draft_extend_after_verify():
    assert hasattr(EagleDraftWorker, "draft_extend_for_decode")
    assert not hasattr(EAGLEWorker, "draft_extend_after_verify")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py::test_eagle_worker_does_not_own_draft_extend_after_verify -q
```

Expected: FAIL because `EAGLEWorker.draft_extend_after_verify` still exists.

- [ ] **Step 3: Move draft extend after verify**

Move the body of current `EAGLEWorker.draft_extend_after_verify()` into `EagleDraftWorker.draft_extend_for_decode()`.

Use this signature:

```python
    def draft_extend_for_decode(
        self, model_worker_batch: ModelWorkerBatch, batch_output
    ) -> None:
```

Keep the body behavior the same, including construction of the next `EagleDraftInput` and updating `batch_output.next_draft_input.topk_p`, `topk_index`, and `hidden_states`.

- [ ] **Step 4: Delegate from EAGLEWorker decode branch**

Replace:

```python
            self.draft_extend_after_verify(model_worker_batch, batch_output)
```

with:

```python
            self.draft_worker.draft_extend_for_decode(model_worker_batch, batch_output)
```

Delete `EAGLEWorker.draft_extend_after_verify()`.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py python/sgl_jax/test/speculative/test_eagle_utils.py python/sgl_jax/test/speculative/test_eagle_tree_build.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sgl_jax/srt/speculative/eagle_draft_worker.py python/sgl_jax/srt/speculative/eagle_worker.py python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py
git commit -m "refactor: move eagle draft extend"
```

---

### Task 7: Clean EAGLEWorker into orchestration-only class

**Files:**
- Modify: `python/sgl_jax/srt/speculative/eagle_worker.py`
- Modify: `python/sgl_jax/srt/speculative/eagle_draft_worker.py`
- Modify: `python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py`

- [ ] **Step 1: Add source-level boundary test**

Append to `python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py`:

```python
import inspect


def test_eagle_worker_is_orchestration_only():
    source = inspect.getsource(EAGLEWorker)

    forbidden = (
        "ModelWorker)",
        "is_draft_worker=True",
        "self.draft_model_runner.forward",
        "build_tree_kernel_efficient(",
        "select_top_k_tokens(",
    )
    for text in forbidden:
        assert text not in source

    required = (
        "self.draft_worker.draft_extend_for_prefill",
        "self.draft_worker.draft(",
        "self.draft_worker.draft_extend_for_decode",
        "self.target_worker.forward_batch_generation",
    )
    for text in required:
        assert text in source
```

- [ ] **Step 2: Run test to verify it fails if cleanup is incomplete**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py::test_eagle_worker_is_orchestration_only -q
```

Expected: PASS if previous tasks fully removed draft logic; FAIL listing the remaining forbidden text otherwise.

- [ ] **Step 3: Remove unused draft imports from EAGLEWorker**

In `python/sgl_jax/srt/speculative/eagle_worker.py`, remove imports that only support draft internals after extraction:

```python
import functools
import itertools
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm
from sgl_jax.srt.utils.jax_utils import device_array
```

Keep imports still needed by target verify, logprob handling, and delegating wrappers:

```python
import logging
import jax
import jax.numpy as jnp
import numpy as np
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput, EagleVerifyOutput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
```

- [ ] **Step 4: Verify utility functions moved to spec_utils**

Ensure these helper functions no longer live in `eagle_worker.py` and are importable from `python/sgl_jax/srt/speculative/spec_utils.py`:

```text
topk_probs_from_logits
fast_topk
update_eagle_lists
update_forward_batch_info
select_top_k_tokens
select_top_k_tokens_step_0
select_top_k_tokens_step_greater_0
```

`EagleDraftWorker` should import them with:

```python
from sgl_jax.srt.speculative.spec_utils import (
    select_top_k_tokens,
    topk_probs_from_logits,
)
```

Only import the helper names actually used by `EagleDraftWorker`. Do not import `EagleDraftWorker` from `spec_utils.py` and do not import helper functions from `eagle_worker.py`.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py python/sgl_jax/test/speculative/test_eagle_utils.py python/sgl_jax/test/speculative/test_eagle_tree_build.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/sgl_jax/srt/speculative/eagle_worker.py python/sgl_jax/srt/speculative/eagle_draft_worker.py python/sgl_jax/srt/speculative/spec_utils.py python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py
git commit -m "refactor: simplify eagle worker orchestration"
```

---

### Task 8: Verify scheduler and precompile integration

**Files:**
- Modify if needed: `python/sgl_jax/srt/managers/scheduler.py`
- Test: existing speculative tests and import checks

- [ ] **Step 1: Inspect scheduler integration point**

Open `python/sgl_jax/srt/managers/scheduler.py` around the speculative worker initialization. The code should continue to instantiate `EAGLEWorker` for `spec_algorithm.is_eagle()`:

```python
        if self.spec_algorithm is not None and self.spec_algorithm.is_eagle():
            from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker

            self.draft_worker = EAGLEWorker(
                server_args=server_args,
                target_worker=self.tp_worker,
            )
```

- [ ] **Step 2: Keep scheduler unchanged unless it references draft-only attributes**

If scheduler references `self.draft_worker.speculative_num_draft_tokens`, keep that attribute on `EAGLEWorker` by preserving:

```python
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
```

If scheduler calls `self.draft_worker.run_spec_decode_precompile()`, keep the delegating wrapper from Task 5.

- [ ] **Step 3: Run import check for scheduler path**

Run:

```bash
python - <<'PY'
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
print(Scheduler.__name__, EAGLEWorker.__name__, EagleDraftWorker.__name__)
PY
```

Expected output:

```text
Scheduler EAGLEWorker EagleDraftWorker
```

- [ ] **Step 4: Run focused test suite**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_base_spec_worker.py python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py python/sgl_jax/test/speculative/test_eagle_utils.py python/sgl_jax/test/speculative/test_eagle_tree_build.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit only if scheduler changed**

If `scheduler.py` changed:

```bash
git add python/sgl_jax/srt/managers/scheduler.py
git commit -m "refactor: keep scheduler on eagle spec worker"
```

If `scheduler.py` did not change, do not create an empty commit.

---

### Task 9: Optionally sync RFC implementation status

**Files:**
- Optional modify: `docs/design/rfc-eagle-mtp-refactoring-plan.md`

Run this task only if the PR is expected to include docs updates. If the RFC was already updated before implementation, skip this task and do not create a docs-only commit.

- [ ] **Step 1: Add PR1 status note if needed**

In `docs/design/rfc-eagle-mtp-refactoring-plan.md`, under `### PR1 骨架重写：Base worker + 单层 EAGLE 新路径`, add:

```markdown
PR1 implementation note: this PR lands only the worker boundary rewrite and contract tests. Runnable MultiLayer MTP remains in PR2; DP attention support remains contract-only unless touched by the worker boundary changes.
```

- [ ] **Step 2: Run markdown grep for stale wording**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('docs/design/rfc-eagle-mtp-refactoring-plan.md').read_text()
for phrase in ['旧/新双路径', '保持旧 EAGLE 兼容', 'PR1 中实现可运行的多层 MTP']:
    if phrase in text:
        raise SystemExit(f'stale phrase: {phrase}')
print('RFC wording check passed')
PY
```

Expected output:

```text
RFC wording check passed
```

- [ ] **Step 3: Commit only if the RFC changed in this task**

```bash
git add docs/design/rfc-eagle-mtp-refactoring-plan.md
git commit -m "docs: clarify mtp phase1 pr1 scope"
```

If the RFC did not change in this task, do not create an empty commit.

---

### Task 10: Final verification for PR1

**Files:**
- No code changes expected.

- [ ] **Step 1: Run unit and contract tests**

Run:

```bash
python -m pytest python/sgl_jax/test/speculative/test_base_spec_worker.py python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py python/sgl_jax/test/speculative/test_eagle_utils.py python/sgl_jax/test/speculative/test_eagle_tree_build.py -q
```

Expected: PASS.

- [ ] **Step 2: Run import smoke test**

Run:

```bash
python - <<'PY'
from sgl_jax.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput
print(BaseDraftWorker.__name__, BaseSpecWorker.__name__, EagleDraftWorker.__name__, EAGLEWorker.__name__, EagleDraftInput.__name__, EagleVerifyInput.__name__)
PY
```

Expected output:

```text
BaseDraftWorker BaseSpecWorker EagleDraftWorker EAGLEWorker EagleDraftInput EagleVerifyInput
```

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: no uncommitted files except intentionally uncommitted local artifacts.

- [ ] **Step 4: Capture PR summary**

Use this PR summary:

```markdown
## Summary
- Added explicit speculative worker base contracts for PR1.
- Replaced monolithic EAGLEWorker draft ownership with EagleDraftWorker delegation.
- Clarified PR1 RFC scope: no old EAGLE compatibility path and no MultiLayer MTP runtime yet.

## Test plan
- python -m pytest python/sgl_jax/test/speculative/test_base_spec_worker.py python/sgl_jax/test/speculative/test_eagle_worker_refactor_contract.py python/sgl_jax/test/speculative/test_eagle_utils.py python/sgl_jax/test/speculative/test_eagle_tree_build.py -q
- python import smoke test for BaseSpecWorker, EagleDraftWorker, EAGLEWorker, EagleDraftInput, EagleVerifyInput
```

---

## Self-Review

**Spec coverage:**
- PR1 base worker abstraction is covered by Tasks 1 and 3.
- `EagleDraftWorker` extraction is covered by Tasks 3-6.
- `EAGLEWorker` orchestration-only rewrite is covered by Tasks 3, 5, 6, and 7.
- Explicit `SpecInput`/draft/verify/result contract is covered by Task 2; `GenerationBatchResult` remains structurally unchanged in PR1 because the scheduler already consumes its current fields, but Task 2 records DP/order boundary semantics on the spec inputs.
- Scheduler entrypoint stability is covered by Task 8.
- Optional RFC status sync is covered by Task 9 when docs updates are included in the PR.
- Verification is covered by Task 10.

**Placeholder scan:**
- The plan contains no `TBD`, no `TODO`, and no unspecified implementation steps.

**Type consistency:**
- `BaseDraftWorker.draft()` returns `EagleVerifyInput`.
- `BaseSpecWorker.verify()` accepts `EagleVerifyInput` and optional `cur_allocate_lens` for the current JAX decode path.
- `EagleDraftWorker` method names match the RFC: `draft`, `draft_extend_for_prefill`, `draft_extend_for_decode`.
- `EAGLEWorker` keeps scheduler-facing `forward_batch_speculative_generation()` and `run_spec_decode_precompile()`.
