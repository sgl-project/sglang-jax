"""Fast discrete-event mock for spec decode scheduler overlap.

This does not model target-model math. It models the dependency topology that
decides whether the fake TPU lane goes idle between fused verify launches.
Use it to iterate on scheduler/worker ordering before running full 4-rank TPU
profiles.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Strategy = Literal["current", "target_future_relay", "same_batch_device_chain"]


@dataclass(frozen=True)
class MockDurations:
    verify_tpu_ms: float = 23.0
    phase_a_to_draft_dispatch_ms: float = 2.4
    draft_extend_tpu_ms: float = 1.6
    materialize_phase_b_ms: float = 0.03
    run_batch_ms: float = 0.7
    metadata_ms: float = 1.1
    device_put_ms: float = 4.0
    verify_dispatch_ms: float = 1.0
    future_enqueue_ms: float = 0.5
    future_metadata_ms: float = 0.8
    phase_a_batch_ms: float = 3.5


@dataclass(frozen=True)
class MockEvent:
    lane: str
    name: str
    start_ms: float
    duration_ms: float
    step: int

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms


@dataclass(frozen=True)
class MockSpecOverlapResult:
    strategy: Strategy
    events: list[MockEvent]
    verify_to_verify_gaps_ms: list[float]
    verify_to_verify_idle_ms: list[float]

    @property
    def max_verify_gap_ms(self) -> float:
        return max(self.verify_to_verify_gaps_ms, default=0.0)

    @property
    def max_verify_idle_ms(self) -> float:
        return max(self.verify_to_verify_idle_ms, default=0.0)


def _add(events: list[MockEvent], lane: str, name: str, start: float, dur: float, step: int):
    events.append(MockEvent(lane, name, round(start, 6), round(dur, 6), step))
    return start + dur


def _verify_events(events: list[MockEvent]) -> list[MockEvent]:
    return [
        event
        for event in sorted(events, key=lambda event: event.start_ms)
        if event.lane == "tpu" and event.name == "jit_fused_greedy_verify"
    ]


def _verify_gap_metrics(events: list[MockEvent]) -> tuple[list[float], list[float]]:
    verify = _verify_events(events)
    tpu_events = [
        event for event in sorted(events, key=lambda event: event.start_ms) if event.lane == "tpu"
    ]
    gaps: list[float] = []
    idle: list[float] = []
    for prev_verify, next_verify in zip(verify, verify[1:]):
        gap_start = prev_verify.end_ms
        gap_end = next_verify.start_ms
        active = 0.0
        for event in tpu_events:
            if event.name == "jit_fused_greedy_verify":
                continue
            overlap = min(event.end_ms, gap_end) - max(event.start_ms, gap_start)
            active += max(0.0, overlap)
        gaps.append(round(gap_end - gap_start, 6))
        idle.append(round(max(0.0, gap_end - gap_start - active), 6))
    return gaps, idle


def _run_current(steps: int, durations: MockDurations) -> list[MockEvent]:
    events: list[MockEvent] = []
    tpu_available = 0.0
    next_verify_ready = 0.0
    for step in range(steps):
        verify_start = max(tpu_available, next_verify_ready)
        verify_end = _add(
            events,
            "tpu",
            "jit_fused_greedy_verify",
            verify_start,
            durations.verify_tpu_ms,
            step,
        )

        phase_a_done = _add(
            events,
            "worker",
            "publish_phase_a_then_prepare_draft_extend",
            verify_end,
            durations.phase_a_to_draft_dispatch_ms,
            step,
        )
        draft_end = _add(
            events,
            "tpu",
            "jit_fused_draft_extend",
            phase_a_done,
            durations.draft_extend_tpu_ms,
            step,
        )
        post = _add(
            events,
            "worker",
            "materialize_predispatched_spec_draft_extend_phase",
            draft_end,
            durations.materialize_phase_b_ms,
            step,
        )
        post = _add(events, "scheduler", "run_batch", post, durations.run_batch_ms, step)
        post = _add(
            events,
            "scheduler",
            "get_eagle_forward_metadata",
            post,
            durations.metadata_ms,
            step,
        )
        post = _add(events, "scheduler", "batched_device_put", post, durations.device_put_ms, step)
        post = _add(
            events,
            "scheduler",
            "PjitFunction(fused_greedy_verify)",
            post,
            durations.verify_dispatch_ms,
            step,
        )

        tpu_available = draft_end
        next_verify_ready = post
    return events


def _run_target_future_relay(steps: int, durations: MockDurations) -> list[MockEvent]:
    events: list[MockEvent] = []
    tpu_available = 0.0
    scheduler_available = 0.0
    next_verify_ready = 0.0
    for step in range(steps):
        verify_start = max(tpu_available, next_verify_ready)
        future_done = _add(
            events,
            "worker",
            "enqueue_future_draft_extend",
            verify_start,
            durations.future_enqueue_ms,
            step,
        )
        _add(
            events,
            "scheduler",
            "enqueue_future_verify_metadata",
            future_done,
            durations.future_metadata_ms,
            step,
        )
        verify_end = _add(
            events,
            "tpu",
            "jit_fused_greedy_verify",
            verify_start,
            durations.verify_tpu_ms,
            step,
        )
        draft_end = _add(
            events,
            "tpu",
            "jit_fused_draft_extend",
            verify_end,
            durations.draft_extend_tpu_ms,
            step,
        )
        cpu_start = max(scheduler_available, verify_end)
        scheduler_available = _add(
            events,
            "scheduler",
            "process_phase_a_and_batch",
            cpu_start,
            durations.phase_a_batch_ms,
            step,
        )

        tpu_available = draft_end
        next_verify_ready = draft_end
    return events


def _run_same_batch_device_chain(steps: int, durations: MockDurations) -> list[MockEvent]:
    events: list[MockEvent] = []
    tpu_available = 0.0
    scheduler_available = 0.0
    next_verify_ready = 0.0
    for step in range(steps):
        verify_start = max(tpu_available, next_verify_ready)
        verify_end = _add(
            events,
            "tpu",
            "jit_fused_greedy_verify",
            verify_start,
            durations.verify_tpu_ms,
            step,
        )
        draft_end = _add(
            events,
            "tpu",
            "jit_fused_draft_extend",
            verify_end,
            durations.draft_extend_tpu_ms,
            step,
        )
        _add(
            events,
            "worker",
            "launch_next_verify_from_same_batch_state",
            draft_end,
            durations.future_enqueue_ms,
            step,
        )
        scheduler_start = max(scheduler_available, verify_end)
        scheduler_available = _add(
            events,
            "scheduler",
            "process_phase_a_catch_up",
            scheduler_start,
            durations.phase_a_batch_ms,
            step,
        )
        tpu_available = draft_end
        next_verify_ready = draft_end
    return events


def run_mock_spec_overlap(
    strategy: Strategy,
    *,
    steps: int = 5,
    durations: MockDurations | None = None,
) -> MockSpecOverlapResult:
    durations = durations or MockDurations()
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if strategy == "current":
        events = _run_current(steps, durations)
    elif strategy == "target_future_relay":
        events = _run_target_future_relay(steps, durations)
    elif strategy == "same_batch_device_chain":
        events = _run_same_batch_device_chain(steps, durations)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    gaps, idle = _verify_gap_metrics(events)
    return MockSpecOverlapResult(
        strategy=strategy,
        events=events,
        verify_to_verify_gaps_ms=gaps,
        verify_to_verify_idle_ms=idle,
    )


def write_chrome_trace(result: MockSpecOverlapResult, path: str | Path) -> None:
    path = Path(path)
    pid_by_lane = {"scheduler": 1, "worker": 2, "tpu": 3}
    trace_events = []
    for lane, pid in pid_by_lane.items():
        trace_events.append({"ph": "M", "pid": pid, "name": "process_name", "args": {"name": lane}})
    for event in result.events:
        trace_events.append(
            {
                "ph": "X",
                "pid": pid_by_lane[event.lane],
                "tid": 1,
                "name": event.name,
                "ts": int(event.start_ms * 1000),
                "dur": int(event.duration_ms * 1000),
                "args": {"step": event.step},
            }
        )
    path.write_text(json.dumps({"traceEvents": trace_events}, indent=2), encoding="utf-8")


def summarize_chrome_trace(path: str | Path) -> dict[str, list[float]]:
    """Summarize verify-to-verify gap/idle from a mock chrome trace."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    events: list[MockEvent] = []
    lane_by_pid = {1: "scheduler", 2: "worker", 3: "tpu"}
    for raw in payload.get("traceEvents", []):
        if raw.get("ph") != "X":
            continue
        lane = lane_by_pid.get(raw.get("pid"))
        if lane is None:
            continue
        events.append(
            MockEvent(
                lane=lane,
                name=raw["name"],
                start_ms=float(raw["ts"]) / 1000.0,
                duration_ms=float(raw.get("dur", 0)) / 1000.0,
                step=int(raw.get("args", {}).get("step", -1)),
            )
        )
    gaps, idle = _verify_gap_metrics(events)
    return {
        "verify_to_verify_gaps_ms": gaps,
        "verify_to_verify_idle_ms": idle,
    }


def _print_summary(result: MockSpecOverlapResult) -> None:
    print(f"strategy={result.strategy}")
    print(f"verify_to_verify_gaps_ms={result.verify_to_verify_gaps_ms}")
    print(f"verify_to_verify_idle_ms={result.verify_to_verify_idle_ms}")
    print(f"max_verify_gap_ms={result.max_verify_gap_ms:.3f}")
    print(f"max_verify_idle_ms={result.max_verify_idle_ms:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=("current", "target_future_relay", "both"),
        default="both",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()

    strategies = ("current", "target_future_relay") if args.strategy == "both" else (args.strategy,)
    for strategy in strategies:
        result = run_mock_spec_overlap(strategy, steps=args.steps)
        _print_summary(result)
        if args.trace is not None:
            trace_path = args.trace
            if len(strategies) > 1:
                trace_path = args.trace.with_name(
                    f"{args.trace.stem}_{strategy}{args.trace.suffix}"
                )
            write_chrome_trace(result, trace_path)
            print(f"trace={trace_path}")


if __name__ == "__main__":
    main()
