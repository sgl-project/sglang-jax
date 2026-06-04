"""Host-neutral nightly case definitions.

These dataclasses describe *what* to run (a dataset/model accuracy check or a
serving perf check) and carry no host-topology fields — no nnodes, node rank,
or dist-init address. They are imported by both the single-host runner
(``test/srt/nightly/single_host/accuracy_case_runner.py``) and the multi-host runner
(``test/srt/nightly/multi_host/``), so the case contract lives in exactly one place.

Multi-host-only types (``RuntimeConfig``, ``ModelRun``, ``MultiHostSuite``, the
launch-profile validators) stay in ``test/srt/nightly/multi_host/multi_host_suite.py``.
"""

from dataclasses import dataclass, field
from typing import Any


class SuiteError(Exception):
    """Tagged error consumed by suite_runner.py exit-code mapping."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class PerfCase:
    name: str
    input_len: int
    output_len: int
    num_prompts: int
    max_concurrency: int
    request_rate: float = float("inf")
    seed: int = 42
    flush_cache: bool = False


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    dataset: str
    model_id: str
    eval_batch_size: int = 32
    generation_config: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    timeout: int | None = None
    score_threshold: float | None = None
