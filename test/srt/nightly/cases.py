"""Host-neutral nightly case definitions.

These dataclasses describe *what* to run (a dataset/model accuracy check or a
serving perf check) and carry no host-topology fields — no nnodes, node rank,
or dist-init address. They are imported by both the single-host runner
(``test/srt/nightly/single_host/accuracy_case_runner.py``) and the multi-host runner
(``test/srt/nightly/multi_host/``), so the case contract lives in exactly one place.

Multi-host-only types (``RuntimeConfig``, ``ModelRun``, ``MultiHostSuite``, the
launch-profile validators) stay in ``test/srt/nightly/multi_host/multi_host_suite.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


class SuiteError(Exception):
    """Tagged error consumed by suite_runner.py exit-code mapping."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class AbsolutePerfBaseline:
    value: float
    tolerance: float

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError(f"baseline value must be positive, got {self.value}")
        if not 0 <= self.tolerance < 1:
            raise ValueError(f"baseline tolerance must be in [0, 1), got {self.tolerance}")


@dataclass(frozen=True)
class PerfCase:
    name: str
    input_len: int
    output_len: int
    num_prompts: int
    max_concurrency: int
    workload: Literal["random", "generated-shared-prefix"] = "random"
    request_rate: float = float("inf")
    seed: int = 42
    warmup_requests: int = 0
    flush_cache: bool = False
    gsp_num_groups: int = 64
    gsp_prompts_per_group: int = 16
    gsp_system_prompt_len: int = 2048
    gsp_question_len: int = 128
    gsp_range_ratio: float = 1.0
    # Absolute-floor gate: {csv_column: min_value}. Set only on the points that
    # have a floor (the representative point); the gate skips it when None.
    floors: dict[str, float] | None = None
    absolute_baselines: dict[str, AbsolutePerfBaseline] | None = None
    use_trailing_baseline: bool = True
    expose_in_caselist: bool = False
    # Set on the one representative point that captures an xprof trace.
    capture_trace: bool = False
    profile_num_steps: int | None = None

    def __post_init__(self):
        if self.workload == "generated-shared-prefix":
            expected = self.gsp_num_groups * self.gsp_prompts_per_group
            if self.num_prompts != expected:
                raise ValueError(
                    f"{self.name}: num_prompts={self.num_prompts} must equal "
                    f"gsp_num_groups*gsp_prompts_per_group={expected}"
                )


@dataclass(frozen=True)
class PerfParams:
    """Sweep + gate config for one perf model/config, à la sglang's
    ``PerformanceTestParams``. The sweep is split into two intents: prefill points
    (``output_len=1``, stress prefill — gated on in_tps + TTFT) and decode points
    (``output_len`` long, stress decode — gated on in_tps + out_tps + ITL).
    Defaults give a KV-safe 6-point grid; the decode concurrencies a given server
    can actually hold are KV-bound (see the perf suite registration), so override
    ``decode_concurrencies``/``profile_point`` per model. Override only the fields
    you need at suite registration so the config lives next to the case, not in a
    separate file::

        PerfParams(floors={"out_tps": 657})                  # default grid + a floor
        PerfParams(decode_concurrencies=(16, 32),            # dense (KV ceiling ~41)
                   profile_point=(32, 4096, 1024),
                   floors={"out_tps": 657})

    ``profile_point`` is the one point that carries the floor + xprof trace; it
    must exist in the grid (else nothing is floored/traced — a ValueError).

    ``prefill_floor_point`` is the prefill representative point that carries an
    absolute ``in_tps`` floor (``prefill_floors``), symmetric to the decode
    ``out_tps`` floor on ``profile_point`` but with no trace. If set it must also
    exist in the grid (else a ValueError).

    Every ``decode_concurrencies`` entry must be ``<=`` the server's KV decode
    ceiling (model + config dependent; see the measured ceilings at the perf
    suite registration). A point above the ceiling cannot be held concurrently —
    it queues instead of saturating, so it under-reports throughput rather than
    failing fast. The ceiling can't be known statically, so this is a convention,
    not a validated constraint.
    """

    # Prefill points (output_len=1): prefill throughput + TTFT.
    prefill_concurrencies: tuple[int, ...] = (8, 32)
    prefill_input_lens: tuple[int, ...] = (4096, 8192)
    # Decode points (output_len=decode_output_len): adds output throughput + ITL.
    # Default kept KV-safe (both v6e-4 configs hold c64); per-model overrides at
    # registration push it up to each server's verified ceiling.
    decode_concurrencies: tuple[int, ...] = (32, 64)
    decode_input_lens: tuple[int, ...] = (4096,)
    decode_output_len: int = 1024
    # The single point that captures a trace (profiling perturbs the measurement,
    # so it is not enabled everywhere) and how many steps to profile.
    profile_point: tuple[int, int, int] = (64, 4096, 1024)
    profile_num_steps: int = 10
    # Absolute-floor gate for the representative point, {csv_column: min_value}.
    floors: dict[str, float] | None = None
    # Prefill representative: in_tps (prefill throughput) absolute floor. No trace
    # (trace stays on profile_point/decode). Set per model at registration.
    prefill_floor_point: tuple[int, int, int] | None = None
    prefill_floors: dict[str, float] | None = None


def _perf_num_prompts(concurrency: int, output_len: int) -> int:
    # Enough queued requests to hold the target concurrency in steady state
    # (request_rate=inf): a prefill point finishes each request fast, so it needs
    # a deeper queue than a decode point to stay saturated.
    if output_len <= 1:
        return max(concurrency * 8, 256)
    return max(concurrency * 4, 128)


def perf_sweep_cases(prefix: str, params: PerfParams | None = None) -> list["PerfCase"]:
    """Build the PerfCase points for one model/config from ``params``.

    ``prefix`` names the model+config (e.g. ``qwen3-32b``); each point is
    ``<prefix>-c<concurrency>-i<input_len>-o<output_len>``. The representative
    point (``params.profile_point``) carries the floors + trace flag, so the
    whole config is declared in one place at suite registration.
    """
    p = params or PerfParams()
    # Two intents: prefill (output_len=1) + decode (output_len=decode_output_len).
    points = [(c, i, 1) for c in p.prefill_concurrencies for i in p.prefill_input_lens]
    points += [
        (c, i, p.decode_output_len) for c in p.decode_concurrencies for i in p.decode_input_lens
    ]
    if p.profile_point not in points:
        raise ValueError(
            f"{prefix}: profile_point {p.profile_point} is not in the sweep grid; "
            "no point would carry floors/trace"
        )
    if p.prefill_floor_point is not None and p.prefill_floor_point not in points:
        raise ValueError(f"{prefix}: prefill_floor_point {p.prefill_floor_point} not in grid")
    cases: list[PerfCase] = []
    for concurrency, input_len, output_len in points:
        pt = (concurrency, input_len, output_len)
        is_repr = pt == p.profile_point  # decode repr: trace + out_tps floor
        is_prefill_repr = p.prefill_floor_point is not None and pt == p.prefill_floor_point
        point_floors = p.floors if is_repr else (p.prefill_floors if is_prefill_repr else None)
        cases.append(
            PerfCase(
                name=f"{prefix}-c{concurrency}-i{input_len}-o{output_len}",
                input_len=input_len,
                output_len=output_len,
                num_prompts=_perf_num_prompts(concurrency, output_len),
                max_concurrency=concurrency,
                floors=point_floors,
                capture_trace=is_repr,  # trace only at decode repr
                profile_num_steps=p.profile_num_steps if is_repr else None,
            )
        )
    return cases


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


# Default gsm8k sampling (greedy). Lives here (host-neutral, stdlib-only) so the
# suite catalog can build cases without importing the jax-backed case runners.
GSM8K_GENERATION_CONFIG = {"temperature": 0.0, "max_tokens": 2048}
