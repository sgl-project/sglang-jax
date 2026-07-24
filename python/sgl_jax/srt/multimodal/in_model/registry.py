"""Registry mapping HF architecture names to in-model multimodal plan builders.

The registry is deliberately modality-agnostic: it stores a factory that turns a
:class:`ModelConfig` into an object exposing ``build(reqs_info, dp_size,
per_dp_token, tp_size)``. Encoder models register through
``encoder_plan_builder.register_encoder``; this module never sees their
modality-specific details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.managers.schedule_batch import ScheduleReqsInfo
    from sgl_jax.srt.multimodal.in_model.plan import MultimodalEmbedPlan


class InModelPlanBuilder(Protocol):
    def build(
        self,
        reqs_info: list[ScheduleReqsInfo] | None,
        dp_size: int,
        per_dp_token: int,
        tp_size: int,
    ) -> MultimodalEmbedPlan | None: ...


_BUILDER_FACTORIES: dict[str, Any] = {}


def register_in_model_plan_builder(arch: str, factory: Any) -> None:
    """Register a ``factory(model_config) -> InModelPlanBuilder`` for ``arch``."""
    _BUILDER_FACTORIES[arch] = factory


def resolve_in_model_plan_builder(
    model_config: ModelConfig,
    input_buckets: Any | None = None,
    merge_buckets: Any | None = None,
) -> InModelPlanBuilder | None:
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    factory = _BUILDER_FACTORIES.get(architectures[0]) if architectures else None
    if factory is None:
        return None
    builder = factory(model_config)
    # Bucket kwargs are optional so text-only builders (which never define these
    # attributes) stay untouched; encoder builders pad input_capacity/merge to them.
    if input_buckets is not None and hasattr(builder, "input_buckets"):
        builder.input_buckets = tuple(input_buckets)
    if merge_buckets is not None and hasattr(builder, "merge_buckets"):
        builder.merge_buckets = tuple(merge_buckets)
    return builder
