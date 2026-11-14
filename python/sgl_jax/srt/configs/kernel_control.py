# ruff: noqa: UP035, UP037
"""Programmer-facing controls for performance-critical kernel scheduling.

The kernel functions expose low-level static arguments, but application code should
not have to call those functions directly or hardcode one configuration for every
shape. ``KernelControlPolicy`` is the stable control plane between server/model code
and those arguments.

Configuration is accepted as a Python mapping, a JSON string, or a JSON file. Each
kernel family has optional defaults and ordered shape rules; later matching rules
override earlier ones::

    {
      "kda": {
        "default": {"intra_block_size": 16},
        "rules": [
          {
            "when": {"head_dim": 64, "max_sequence_length": 512},
            "set": {"state_dim_alignment": 64, "state_block_chunks": 2}
          }
        ]
      },
      "gla": {
        "rules": [
          {
            "when": {"min_sequence_length": 1024},
            "set": {"chunk_size": 256, "compact_alignment": true}
          }
        ]
      }
    }

The policy is intentionally independent of JAX so it can be validated at server
startup and inspected by AKT's capability-exposure gate.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

PROGRAMMER_CONTROL_REGISTRY = {
    "kda": (
        "chunk_size",
        "intra_block_size",
        "scalar_intra_solve",
        "compute_block_chunks",
        "state_block_chunks",
        "state_dim_alignment",
        "pipeline_impl",
    ),
    "gla": (
        "chunk_size",
        "compact_alignment",
        "output_value_tiles",
        "enable__chunk_fwd_o_pl_variant",
        "enable_chunk_fwd_h_kernel_varlen_variant",
        "output_impl",
        "chunk_impl",
        "walk_impl",
    ),
    "rpa_v3": ("d_bkv_sz", "p_bkv_sz"),
    "gmm": ("tk",),
    "fused_moe": (
        "vmem_2d_tile_relayout",
        "vmem_packed_sublane_transpose",
    ),
}


@dataclass(frozen=True)
class KernelControlContext:
    """Static kernel-call context used to select and validate controls.

    Shapes are the per-device shapes seen by the kernel after sharding.
    """

    sequence_length: int
    num_sequences: int
    num_heads: int
    head_dim: int
    value_dim: int
    has_initial_state: bool
    output_final_state: bool
    device_kind: str = ""

    def __post_init__(self):
        for name in (
            "sequence_length",
            "num_sequences",
            "num_heads",
            "head_dim",
            "value_dim",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        for name in ("has_initial_state", "output_final_state"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.device_kind, str):
            raise ValueError("device_kind must be a string")


def _require_bool(name: str, value: Any) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean; got {value!r}")


def _require_choice(name: str, value: Any, choices: tuple[Any, ...]) -> None:
    if value not in choices or type(value) is not type(choices[0]):
        raise ValueError(f"{name} must be one of {choices}; got {value!r}")


@dataclass(frozen=True)
class KDAKernelControls:
    chunk_size: int = 64
    intra_block_size: int = 16
    scalar_intra_solve: bool = False
    compute_block_chunks: int = 1
    state_block_chunks: int = 1
    state_dim_alignment: int = 128
    # Which pipeline executes the chunked recurrence. "incumbent" is the shipped
    # four-launch path; "fused_resident" runs the intra solve, the state
    # recurrence and the output as one pass with the chunk state resident.
    pipeline_impl: str = "incumbent"
    # Internal execution flags, deliberately absent from the programmer registry:
    # serving observes the final recurrent state, so these must remain disabled.
    single_chunk_state_elision: bool = False
    zero_state_output_elision: bool = False

    def validate(self, context: KernelControlContext | None = None) -> None:
        _require_choice("kda.chunk_size", self.chunk_size, (16, 32, 64, 128, 256))
        _require_choice("kda.intra_block_size", self.intra_block_size, (8, 16, 32))
        _require_choice("kda.compute_block_chunks", self.compute_block_chunks, (1, 2))
        _require_choice("kda.state_block_chunks", self.state_block_chunks, (1, 2, 4))
        _require_choice("kda.state_dim_alignment", self.state_dim_alignment, (64, 128))
        _require_choice("kda.pipeline_impl", self.pipeline_impl, ("incumbent", "fused_resident"))
        _require_bool("kda.scalar_intra_solve", self.scalar_intra_solve)
        _require_bool("kda.single_chunk_state_elision", self.single_chunk_state_elision)
        _require_bool("kda.zero_state_output_elision", self.zero_state_output_elision)
        if self.chunk_size % self.intra_block_size:
            raise ValueError(
                "kda.intra_block_size must divide kda.chunk_size; "
                f"got {self.intra_block_size} and {self.chunk_size}"
            )
        if self.zero_state_output_elision and not self.single_chunk_state_elision:
            raise ValueError("kda.zero_state_output_elision requires single_chunk_state_elision")
        if context is None:
            return
        if self.single_chunk_state_elision and (
            context.num_sequences != 1
            or context.sequence_length != self.chunk_size
            or context.has_initial_state
            or context.output_final_state
        ):
            raise ValueError(
                "kda.single_chunk_state_elision requires one full-sequence chunk, "
                "no initial state, and no requested final state"
            )

    def as_kernel_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GLAKernelControls:
    chunk_size: int = 64
    compact_alignment: bool = False
    # Internal execution flags, deliberately absent from the programmer registry:
    # serving observes the final recurrent state, so these must remain disabled.
    single_chunk_state_elision: bool = False
    zero_state_output_elision: bool = False
    output_value_tiles: int = 1
    enable__chunk_fwd_o_pl_variant: bool = False
    enable_chunk_fwd_h_kernel_varlen_variant: bool = False
    # Which handle executes the output stage. "incumbent" is the shipped
    # chunk_fwd_o path with its schedule toggles; "batched_value_tiles" runs the
    # standalone batched_value_tile_fwd_o handle, whose value-tile axis is a
    # batch dimension of every contraction instead of a Python loop bound.
    output_impl: str = "incumbent"
    # Which handle executes the WHOLE chunked recurrence. "incumbent" is the
    # shipped two-stage pipeline (chunk-state stage, entrance-state buffer,
    # output stage); "decay_rescaled" runs the standalone
    # decay_rescaled_chunk_fwd handle, which folds the scalar gate into a
    # rescaling of k and a deferred row scaling of the output, so one k^T v
    # contraction serves both the sub-chunk states and the chunk carry and the
    # state stage disappears entirely.
    chunk_impl: str = "incumbent"
    # Which handle executes the whole chunked recurrence as a GATE-COMPLETE walk.
    # "incumbent" changes nothing. "fused_walk" runs the standalone
    # fused_walk_chunk_fwd handle, which carries both halves of the factored gate
    # (and `scale`) inside the staged q/k operands, so no contraction result is
    # rescaled afterwards, and which issues the sub-chunk prefix and the chunk
    # carry as ONE contraction against the shared k^T v; the whole walk runs at
    # the reduced contraction pass count. "fused_walk_exact_state" is the same
    # walk with the merged state/carry operator -- the one contraction whose
    # result propagates across chunks and into decode -- kept at the full pass
    # count.
    walk_impl: str = "incumbent"

    def validate(self, context: KernelControlContext | None = None) -> None:
        _require_choice(
            "gla.chunk_size",
            self.chunk_size,
            (16, 32, 64, 128, 256, 512, 1024, 2048),
        )
        _require_choice("gla.output_value_tiles", self.output_value_tiles, (1, 2, 4, 8))
        _require_choice("gla.output_impl", self.output_impl, ("incumbent", "batched_value_tiles"))
        _require_choice("gla.chunk_impl", self.chunk_impl, ("incumbent", "decay_rescaled"))
        _require_choice(
            "gla.walk_impl",
            self.walk_impl,
            ("incumbent", "fused_walk", "fused_walk_exact_state"),
        )
        _require_bool("gla.compact_alignment", self.compact_alignment)
        _require_bool("gla.enable__chunk_fwd_o_pl_variant", self.enable__chunk_fwd_o_pl_variant)
        _require_bool(
            "gla.enable_chunk_fwd_h_kernel_varlen_variant",
            self.enable_chunk_fwd_h_kernel_varlen_variant,
        )
        _require_bool("gla.single_chunk_state_elision", self.single_chunk_state_elision)
        _require_bool("gla.zero_state_output_elision", self.zero_state_output_elision)
        if self.zero_state_output_elision and not self.single_chunk_state_elision:
            raise ValueError("gla.zero_state_output_elision requires single_chunk_state_elision")
        if self.chunk_impl != "incumbent":
            if (
                self.output_impl != "incumbent"
                or self.enable__chunk_fwd_o_pl_variant
                or self.enable_chunk_fwd_h_kernel_varlen_variant
                or self.single_chunk_state_elision
                or self.zero_state_output_elision
            ):
                raise ValueError(
                    "gla.chunk_impl owns the state stage and the output stage "
                    "together, so it consumes neither stage's schedule toggle "
                    "nor the state-elision toggles"
                )
            if self.chunk_size <= 128 or self.chunk_size % 128:
                raise ValueError(
                    "gla.chunk_impl='decay_rescaled' walks a chunk in 128-row "
                    f"sub-chunks; got chunk_size={self.chunk_size}"
                )
        if self.walk_impl != "incumbent":
            if (
                self.chunk_impl != "incumbent"
                or self.output_impl != "incumbent"
                or self.enable__chunk_fwd_o_pl_variant
                or self.enable_chunk_fwd_h_kernel_varlen_variant
                or self.single_chunk_state_elision
                or self.zero_state_output_elision
            ):
                raise ValueError(
                    "gla.walk_impl owns the state stage and the output stage "
                    "together, so it consumes neither stage's schedule toggle, "
                    "neither state-elision toggle, and no other whole-recurrence "
                    "handle"
                )
            if self.chunk_size <= 128 or self.chunk_size % 128:
                raise ValueError(
                    "gla.walk_impl walks a chunk in 128-row sub-chunks; got "
                    f"chunk_size={self.chunk_size}"
                )
        if context is None:
            return
        if context.head_dim % 128 or context.value_dim % 128:
            raise ValueError(
                "simple GLA controls require head_dim and value_dim divisible by 128; "
                f"got {context.head_dim}/{context.value_dim}"
            )
        value_tiles = context.num_heads * (context.value_dim // 128)
        if value_tiles % self.output_value_tiles:
            raise ValueError(
                "gla.output_value_tiles must divide the per-device head/value tile count; "
                f"got group={self.output_value_tiles}, tiles={value_tiles}"
            )
        if self.single_chunk_state_elision and (
            context.num_sequences != 1
            or context.sequence_length != self.chunk_size
            or not self.compact_alignment
            or context.has_initial_state
            or context.output_final_state
        ):
            raise ValueError(
                "gla.single_chunk_state_elision requires compact alignment, one "
                "full-sequence chunk, no initial state, and no requested final state"
            )

    def as_kernel_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RPAV3KernelControls:
    """Block-size controls for the ragged-paged-attention v3 kernel.

    Every field defaults to ``None``, which preserves the kernel's incumbent
    selection path byte-identically (explicit ``*_block_sizes`` 4-tuple if the
    caller passed one, else the TPU-v7 tuned table, else the heuristic in
    ``get_default_block_sizes``). Explicit values are validated against the
    certified deployable domain from the HierEvo 2026-08-10 TPU v6e-8
    elevation run (see ``akt/elevations/2026-08-10-tpu-v6e8/``).
    """

    # Decode-stage KV block size (element ``bkv_sz`` of ``d_block_sizes``).
    # Kept default: None = incumbent tuned-table/heuristic selection, which on
    # the elevation campaign shape resolves to 2048. Certified domain for
    # explicit values: (256, 512, 1024, 2048). Provenance: HierEvo elevation
    # rpa_v3.d_bkv_sz, campaign_009_qwen-bs128-ctx4k-ps64 round 2, +3.25%
    # suite improvement against limiter "paged-attn -> qk-softmax".
    d_bkv_sz: int | None = None
    # Prefill-stage KV block size (element ``bkv_sz`` of ``p_block_sizes``).
    # Kept default: None = incumbent tuned-table/heuristic selection, which on
    # the elevation campaign shape resolves to 1024. Certified domain for
    # explicit values: (256, 1024) (api_deduplication minimized the kept
    # domain). Provenance: HierEvo elevation rpa_v3.p_bkv_sz,
    # campaign_088_moe-mixed-ps128 round 4, +2.36% suite improvement against
    # limiter "qwen-prefill-2048-ps128-moe -> qkv-proj -> int8-matmul ->
    # dependency-bound". Liveness: the kernel's PREFILL pallas stage only
    # executes when the caller passes a non-None chunk_prefill_size; the
    # production FlashAttention call does not today, so prefill sequences are
    # remapped into the MIXED stage and this control is dormant until chunk
    # prefill is enabled.
    p_bkv_sz: int | None = None

    def validate(self, context: KernelControlContext | None = None) -> None:
        if self.d_bkv_sz is not None:
            _require_choice("rpa_v3.d_bkv_sz", self.d_bkv_sz, (256, 512, 1024, 2048))
        if self.p_bkv_sz is not None:
            _require_choice("rpa_v3.p_bkv_sz", self.p_bkv_sz, (256, 1024))

    def as_kernel_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GMMKernelControls:
    """Tiling controls for the megablox grouped-matmul (EPMoE) backend."""

    # k-dimension tile of the grouped matmul. Kept default: None = incumbent
    # selection (gmm v1: tuned table then heuristic; gmm v2: the
    # calculate_tiling auto-tiler), which on the elevation campaign shape
    # resolves to 1408 (full-k for moe_intermediate_size=1408). Certified
    # domain for explicit values: (128, 1408). Provenance: HierEvo elevation
    # gmm.tk, campaign_019_qwen-bs128-ctx4k-ps64 round 2, +4.35% suite
    # improvement against limiter "paged-attn -> qk-softmax". Scope: an
    # explicit tk reaches all three EPMoE grouped matmuls (GEMM1 w0/w1 and
    # GEMM2 wo) on the epmoe backend only (fused/fused_v2 do not use gmm);
    # gmm v1 may still silently coerce tk to the weight quant_block_size when
    # neither divides the other.
    tk: int | None = None

    def validate(self, context: KernelControlContext | None = None) -> None:
        if self.tk is not None:
            _require_choice("gmm.tk", self.tk, (128, 1408))

    def as_kernel_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusedMoEKernelControls:
    """VMEM sublane layout & cross-slice staging residency controls for fused EP-MoE.

    Exposes programmer control over how 3D packed token/residual tiles
    ``(tokens, t_packing, dim_per_pack)`` and 2D intermediate accumulators
    ``(tokens, bf)`` are laid out and staged in VMEM across intermediate-dimension
    slices (``bf_id``). When ``vmem_2d_tile_relayout=True`` (default), the kernel:
    1. Allocates ``b_acc_vmem`` as native 3D ``(2, a2a_max_tokens, bf)`` (physical
       layout ``T(8, 128)``) instead of 4D ``(2, a2a_max_tokens, 1, bf)``
       (``T(1, 128)``) and fuses ``activation_fn`` at the end of ``dynamic_ffn1``
       when ``num_bd1 == 1``, eliminating sublane rotation/combine/masked-store
       sequences (``vrot.slane``, ``vcombine``, ``vst.msk``) on accumulator I/O.
    2. Relayouts packed token tiles once on ``bf_id == 0`` into a 2D-major VMEM
       cache ``b_x_2d_vmem`` of shape ``(t_packing, a2a_max_tokens, bd1_per_pack)``
       (physical layout ``T(8, 128)(2, 1)``) and accumulates FFN2 partial outputs
       across ``bf_id < num_bf - 1`` in ``b_res_2d_vmem`` of shape
       ``(t_packing, a2a_max_tokens, bd2_per_pack)`` before packing to the A2A
       staging buffer on ``bf_id == num_bf - 1``, eliminating redundant HBM
       round-trip DMAs and repeated ``(bts, 2, 1024) <-> (bts, 1024)`` sublane
       ``vrot.slane``/``vunpack``/``vsel`` relayouts on subsequent ``bf_id`` slices.
    """

    vmem_2d_tile_relayout: bool = True
    vmem_packed_sublane_transpose: bool = True

    def validate(self, context: KernelControlContext | None = None) -> None:
        _require_bool("fused_moe.vmem_2d_tile_relayout", self.vmem_2d_tile_relayout)
        _require_bool(
            "fused_moe.vmem_packed_sublane_transpose",
            self.vmem_packed_sublane_transpose,
        )

    def as_kernel_kwargs(self) -> dict[str, Any]:
        return asdict(self)


_CONTROL_TYPES = {
    "kda": KDAKernelControls,
    "gla": GLAKernelControls,
    "rpa_v3": RPAV3KernelControls,
    "gmm": GMMKernelControls,
    "fused_moe": FusedMoEKernelControls,
}
_DEFAULTS = {
    family: {field.name: field.default for field in fields(control_type)}
    for family, control_type in _CONTROL_TYPES.items()
}
_CONTEXT_FIELDS = frozenset(field.name for field in fields(KernelControlContext))


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object; got {type(value).__name__}")
    return dict(value)


def _validate_control_override(
    family: str,
    override: Mapping[str, Any],
    label: str,
    *,
    base: Mapping[str, Any] | None = None,
) -> None:
    unknown = set(override) - set(PROGRAMMER_CONTROL_REGISTRY[family])
    if unknown:
        raise ValueError(f"{label} has unknown controls: {sorted(unknown)}")
    candidate = dict(_DEFAULTS[family])
    candidate.update(base or {})
    candidate.update(override)
    controls = _CONTROL_TYPES[family](**candidate)
    controls.validate()


def _validate_match(match: Mapping[str, Any], label: str) -> None:
    for key, value in match.items():
        base = key
        if key.startswith("min_") or key.startswith("max_"):
            base = key[4:]
            if base not in _CONTEXT_FIELDS - {"device_kind"}:
                raise ValueError(f"{label} has unknown range match {key!r}")
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label}.{key} must be a positive integer")
        elif key not in _CONTEXT_FIELDS:
            raise ValueError(f"{label} has unknown match key {key!r}")
        elif key == "device_kind":
            if not isinstance(value, str):
                raise ValueError(f"{label}.{key} must be a string")
        elif key in {"has_initial_state", "output_final_state"}:
            if type(value) is not bool:
                raise ValueError(f"{label}.{key} must be boolean")
        elif type(value) is not int or value <= 0:
            raise ValueError(f"{label}.{key} must be a positive integer")


def _matches(match: Mapping[str, Any], context: KernelControlContext) -> bool:
    for key, expected in match.items():
        if key.startswith("min_"):
            if getattr(context, key[4:]) < expected:
                return False
        elif key.startswith("max_"):
            if getattr(context, key[4:]) > expected:
                return False
        elif getattr(context, key) != expected:
            return False
    return True


def _load_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, KernelControlPolicy):
        return value.to_dict()
    if isinstance(value, str):
        source = value.strip()
        if source.startswith("{"):
            value = json.loads(source)
        else:
            value = json.loads(Path(source).expanduser().read_text())
    return _as_mapping(value, "kernel control configuration")


class KernelControlPolicy:
    """Validated defaults and ordered shape rules for production kernels."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        raw = _as_mapping(config or {}, "kernel control configuration")
        unknown_families = set(raw) - set(PROGRAMMER_CONTROL_REGISTRY)
        if unknown_families:
            raise ValueError(f"unknown kernel control families: {sorted(unknown_families)}")

        normalized = {}
        for family, family_value in raw.items():
            family_config = _as_mapping(family_value, family)
            structured = bool({"default", "rules"} & set(family_config))
            if structured:
                extra = set(family_config) - {"default", "rules"}
                if extra:
                    raise ValueError(f"{family} mixes default/rules with controls: {sorted(extra)}")
                default = _as_mapping(family_config.get("default", {}), f"{family}.default")
                rules_value = family_config.get("rules", [])
            else:
                default = family_config
                rules_value = []
            _validate_control_override(family, default, f"{family}.default")
            if not isinstance(rules_value, list):
                raise ValueError(f"{family}.rules must be an array")
            rules = []
            for index, rule_value in enumerate(rules_value):
                rule = _as_mapping(rule_value, f"{family}.rules[{index}]")
                if set(rule) != {"when", "set"}:
                    raise ValueError(
                        f"{family}.rules[{index}] must contain exactly 'when' and 'set'"
                    )
                match = _as_mapping(rule["when"], f"{family}.rules[{index}].when")
                override = _as_mapping(rule["set"], f"{family}.rules[{index}].set")
                _validate_match(match, f"{family}.rules[{index}].when")
                _validate_control_override(
                    family,
                    override,
                    f"{family}.rules[{index}].set",
                    base=default,
                )
                rules.append({"when": match, "set": override})
            normalized[family] = {"default": default, "rules": rules}
        self._config = normalized

    @classmethod
    def from_config(cls, value: Any = None) -> "KernelControlPolicy":
        if isinstance(value, cls):
            return value
        return cls(_load_mapping(value))

    @classmethod
    def supported_controls(cls, family: str) -> tuple[str, ...]:
        try:
            return PROGRAMMER_CONTROL_REGISTRY[family]
        except KeyError as error:
            raise ValueError(f"unknown kernel family {family!r}") from error

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    def resolve(
        self,
        family: str,
        context: KernelControlContext | Mapping[str, Any],
        *,
        base: Mapping[str, Any] | None = None,
    ) -> (
        KDAKernelControls
        | GLAKernelControls
        | RPAV3KernelControls
        | GMMKernelControls
        | FusedMoEKernelControls
    ):
        if family not in _CONTROL_TYPES:
            raise ValueError(f"unknown kernel family {family!r}")
        if not isinstance(context, KernelControlContext):
            context = KernelControlContext(**_as_mapping(context, "kernel context"))
        values = dict(_DEFAULTS[family])
        if base:
            base = _as_mapping(base, f"{family} base controls")
            _validate_control_override(family, base, f"{family} base controls")
            values.update(base)
        family_config = self._config.get(family, {"default": {}, "rules": []})
        values.update(family_config["default"])
        for rule in family_config["rules"]:
            if _matches(rule["when"], context):
                values.update(rule["set"])
        controls = _CONTROL_TYPES[family](**values)
        controls.validate(context)
        return controls

    def resolve_kda(
        self,
        context: KernelControlContext | Mapping[str, Any],
    ) -> KDAKernelControls:
        controls = self.resolve("kda", context)
        assert isinstance(controls, KDAKernelControls)
        return controls

    def resolve_gla(
        self,
        context: KernelControlContext | Mapping[str, Any],
        *,
        chunk_size: int = 64,
    ) -> GLAKernelControls:
        controls = self.resolve("gla", context, base={"chunk_size": chunk_size})
        assert isinstance(controls, GLAKernelControls)
        return controls

    def resolve_rpa_v3(
        self,
        context: KernelControlContext | Mapping[str, Any],
    ) -> RPAV3KernelControls:
        controls = self.resolve("rpa_v3", context)
        assert isinstance(controls, RPAV3KernelControls)
        return controls

    def resolve_gmm(
        self,
        context: KernelControlContext | Mapping[str, Any],
    ) -> GMMKernelControls:
        controls = self.resolve("gmm", context)
        assert isinstance(controls, GMMKernelControls)
        return controls

    def resolve_fused_moe(
        self,
        context: KernelControlContext | Mapping[str, Any],
    ) -> FusedMoEKernelControls:
        controls = self.resolve("fused_moe", context)
        assert isinstance(controls, FusedMoEKernelControls)
        return controls


def normalize_kernel_control_config(value: Any = None) -> dict[str, Any]:
    """Load and validate a JSON/file/mapping value for storage in ``ServerArgs``."""

    return KernelControlPolicy.from_config(value).to_dict()


__all__ = [
    "FusedMoEKernelControls",
    "GLAKernelControls",
    "GMMKernelControls",
    "KDAKernelControls",
    "KernelControlContext",
    "KernelControlPolicy",
    "PROGRAMMER_CONTROL_REGISTRY",
    "RPAV3KernelControls",
    "normalize_kernel_control_config",
]
