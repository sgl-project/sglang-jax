# Copyright 2026 The SGLang-JAX Authors.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from .config import (
    DEFAULT_Q_FRAGMENTS,
    DEFAULT_Q_ROWS,
    FRAGMENT_ROWS,
    KV_FRAGMENTS,
    KV_ROWS,
    MAX_HEAD_GROUP,
    MAX_Q_FRAGMENTS,
    MIN_HEAD_GROUP,
    MRB_ADDRESS_STRIDE,
    MRB_SLOTS,
    PIPELINE_PERIOD,
    PROBABILITY_SLOTS,
    PV_ISSUE_DELAY,
    PV_POP_DELAY,
    QK_POP_DELAY,
    RESIDENT_HEAD_ROWS,
    RHS_PREFETCH_LEAD,
    RHS_SLOTS,
    TEMPLATE_CACHE_SIZE,
    ProbabilityCarry,
    ScalarIndex,
)

if TYPE_CHECKING:
    from .kernel import AttentionOps


@dataclass(frozen=True)
class FragmentLayout:
    """Static description of one KV's work, in head/fragment order."""

    fragments: int = DEFAULT_Q_FRAGMENTS
    row_base: int = 0
    initialize: bool = False
    masked: bool = True
    heads: int = MAX_HEAD_GROUP

    def __post_init__(self) -> None:
        if self.fragments not in (
            KV_FRAGMENTS,
            DEFAULT_Q_FRAGMENTS,
            MAX_Q_FRAGMENTS,
        ) or self.heads not in (MIN_HEAD_GROUP, MAX_HEAD_GROUP):
            raise ValueError("Unsupported static fragment/head layout")
        if (
            self.row_base not in (0, KV_ROWS, DEFAULT_Q_ROWS, (3 * KV_ROWS))
            or self.row_base + FRAGMENT_ROWS * self.fragments > RESIDENT_HEAD_ROWS // self.heads
        ):
            raise ValueError("Layout must fit a resident Q tile")

    @property
    def periods(self) -> int:
        return self.heads * self.fragments // PIPELINE_PERIOD

    def coordinate(self, period: ScalarIndex, offset: int) -> tuple[ScalarIndex, int]:

        return (
            period * (PIPELINE_PERIOD // self.fragments) + offset // self.fragments,
            self.row_base + (offset % self.fragments) * FRAGMENT_ROWS,
        )


@dataclass(frozen=True)
class PipelineEvent:
    """One scheduled action; a negative work_index refers to the previous period."""

    tick: int
    kind: str
    work_index: int
    mxu: int
    mrb_address: int
    rhs_slot: int | None = None


@dataclass(frozen=True)
class Pipeline:
    """Compile-time MRB/RHS schedule; distances require lifecycle validation."""

    qk_pop: int = QK_POP_DELAY
    pv_issue: int = PV_ISSUE_DELAY
    pv_pop: int = PV_POP_DELAY
    period: int = PIPELINE_PERIOD
    mrb_slots: int = MRB_SLOTS
    rhs_lead: int = RHS_PREFETCH_LEAD

    def __post_init__(self) -> None:
        if (
            self.qk_pop,
            self.pv_issue,
            self.pv_pop,
            self.period,
            self.mrb_slots,
            self.rhs_lead,
        ) != (
            QK_POP_DELAY,
            PV_ISSUE_DELAY,
            PV_POP_DELAY,
            PIPELINE_PERIOD,
            MRB_SLOTS,
            RHS_PREFETCH_LEAD,
        ):
            raise ValueError("This emitter implements audited distances 0/28/32/60")

    def mrb_address(self, work_index: int) -> int:
        return (work_index % self.mrb_slots) * MRB_ADDRESS_STRIDE

    @lru_cache(maxsize=TEMPLATE_CACHE_SIZE)  # noqa: B019
    def template(
        self, current: FragmentLayout | None, previous: FragmentLayout | None
    ) -> tuple[PipelineEvent, ...]:
        """Return static events; RHS staging stays within each period."""
        if current is None and previous is None:
            raise ValueError("An empty pipeline template has no work")
        stop = self.period if current is not None else self.pv_pop
        events: dict[int, list[PipelineEvent]] = {i: [] for i in range(stop)}

        def layout_at(work_index: int) -> FragmentLayout | None:
            return previous if work_index < 0 else current

        for mxu, delay in ((0, 0), (1, self.pv_issue)):
            for tick in range(stop):
                work_index = tick - delay
                layout = layout_at(work_index)
                if layout is None or (current is None and work_index >= 0):
                    continue
                if (work_index % self.period) % layout.fragments == 0:
                    rhs_slot = ((work_index % self.period) // layout.fragments) % RHS_SLOTS
                    push_at = max(0, tick - self.rhs_lead)
                    events[push_at].append(
                        PipelineEvent(
                            tick=push_at,
                            kind="rhs",
                            work_index=work_index,
                            mxu=mxu,
                            mrb_address=0,
                            rhs_slot=rhs_slot,
                        )
                    )

        for tick in range(stop):
            # Consume P before producing its replacement.
            for kind, delay, mxu in (
                ("qk", 0, 0),
                ("pv", PV_ISSUE_DELAY, 1),
                ("acc", PV_POP_DELAY, 1),
                ("prob", QK_POP_DELAY, 0),
            ):
                work_index = tick - delay
                layout = layout_at(work_index)
                if layout is None or (current is None and work_index >= 0):
                    continue
                rhs_slot = None
                if kind in ("qk", "pv") and (work_index % PIPELINE_PERIOD) % layout.fragments == 0:
                    rhs_slot = ((work_index % PIPELINE_PERIOD) // layout.fragments) % RHS_SLOTS
                events[tick].append(
                    PipelineEvent(
                        tick=tick,
                        kind=kind,
                        work_index=work_index,
                        mxu=mxu,
                        mrb_address=self.mrb_address(work_index),
                        rhs_slot=rhs_slot,
                    )
                )

        return tuple(e for tick in range(stop) for e in events[tick])

    def emit(
        self,
        attention_ops: AttentionOps,
        current: FragmentLayout | None,
        previous: FragmentLayout | None,
        kv: ScalarIndex,
        previous_kv: ScalarIndex,
        period: ScalarIndex,
        previous_period: ScalarIndex,
        pending_probs: ProbabilityCarry,
    ) -> ProbabilityCarry:
        """Trace the small static template against numerical operations."""
        values = list(pending_probs)
        coords = {}
        for event in self.template(current, previous):
            old = event.work_index < 0
            layout = cast(FragmentLayout, previous if old else current)
            p = previous_period if old else period
            tile = previous_kv if old else kv
            offset = event.work_index % PIPELINE_PERIOD
            # Reuse head expressions across fragments; avoid dynamic div/mod.
            key = (old, offset // layout.fragments)
            if key not in coords:
                coords[key] = p * (PIPELINE_PERIOD // layout.fragments) + offset // layout.fragments
            head = coords[key]
            row = layout.row_base + (offset % layout.fragments) * FRAGMENT_ROWS
            if event.kind == "rhs":
                attention_ops.load_rhs(
                    kv=tile, head_index=head, mxu=event.mxu, rhs_slot=event.rhs_slot
                )
            elif event.kind == "qk":
                attention_ops.issue_qk(
                    head_index=head,
                    row=row,
                    mrb_address=event.mrb_address,
                    rhs_slot=event.rhs_slot,
                )
            elif event.kind == "pv":
                attention_ops.issue_pv(
                    p=values[event.work_index % PROBABILITY_SLOTS],
                    mrb_address=event.mrb_address,
                    rhs_slot=event.rhs_slot,
                )
            elif event.kind == "prob":
                values[event.work_index % PROBABILITY_SLOTS] = attention_ops.compute_probabilities(
                    kv=tile,
                    head_index=head,
                    row=row,
                    mrb_address=event.mrb_address,
                    initialize=layout.initialize,
                    masked=layout.masked,
                )
            else:
                attention_ops.accumulate_output(
                    head_index=head,
                    row=row,
                    mrb_address=event.mrb_address,
                    initialize=layout.initialize,
                )
        return tuple(values)

    def emit_tile(
        self,
        attention_ops: AttentionOps,
        current: FragmentLayout,
        previous: FragmentLayout | None,
        kv: ScalarIndex,
        previous_kv: ScalarIndex,
        pending_probs: ProbabilityCarry,
    ) -> ProbabilityCarry:
        """Emit a full KV tile without draining its final in-flight work."""
        state = self.emit(
            attention_ops=attention_ops,
            current=current,
            previous=previous,
            kv=kv,
            previous_kv=previous_kv,
            period=0,
            previous_period=previous.periods - 1 if previous is not None else 0,
            pending_probs=pending_probs,
        )
        for period in range(1, current.periods):
            state = self.emit(
                attention_ops=attention_ops,
                current=current,
                previous=current,
                kv=kv,
                previous_kv=kv,
                period=period,
                previous_period=period - 1,
                pending_probs=state,
            )
        return state

    def drain(
        self,
        attention_ops: AttentionOps,
        previous: FragmentLayout,
        tiles: ScalarIndex,
        pending_probs: ProbabilityCarry,
    ) -> ProbabilityCarry:
        """Finish the last tile; no Q reads remain in this epilogue."""
        return self.emit(
            attention_ops=attention_ops,
            current=None,
            previous=previous,
            kv=tiles,
            previous_kv=tiles - 1,
            period=0,
            previous_period=previous.periods - 1,
            pending_probs=pending_probs,
        )
