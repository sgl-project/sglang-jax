"""GMM-based Expert-Parallel MoE layer and weight mapping utilities."""

import logging
import math
from functools import partial

import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.collectives.oneshot_allreduce import (
    fused_moe_enabled,
    maybe_oneshot_psum,
    reduce_deferred,
)
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2_fused_glu import (
    gmm_v2_fused_glu,
)
from sgl_jax.srt.kernels.sparse_core.moe_permute import (
    moe_sc_permute_enabled_by_env,
    sc_combine,
    sc_dispatch_gather,
    should_use_sparse_core,
)

# Re-export for backward compatibility: external code imports from this module.
from sgl_jax.srt.layers.fused_moe import FusedEPMoE, FusedEPMoEV2  # noqa: F401
from sgl_jax.srt.layers.gate import GateLogit, TopK  # noqa: F401
from sgl_jax.srt.model_loader.weights import WeightSpec
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import (
    quantize_tensor,
    quantize_tensor_simple,
)

logger = logging.getLogger(__name__)

# For tokens * topk <= PERMUTE_KERNEL_MAX_ROWS the routing indices (stable argsort / bincount /
# inverse) come from one small Pallas call instead of three XLA sort/reduce programs (bs1 census
# 10-05: 0.53 ms/step of 1.7-2.4 us ops). Bit-exact.
from sgl_jax.srt.kernels.moe_permute_small.kernel import (  # noqa: E402
    PERMUTE_KERNEL_MAX_ROWS as _PERMUTE_KERNEL_MAX_ROWS,
)
from sgl_jax.srt.kernels.moe_permute_small.kernel import (  # noqa: E402
    permute_indices_small as _permute_indices_small,
)

# fuse_gate_up() interleaves gate/up columns in 256-wide pairs and GEMM1 applies silu(gate) * up
# in the gmm_v2 epilogue (output already [tokens, N]); removes the XLA activation fusion between
# GEMM1 and GEMM2 (silu activation only).
_GLU_EPILOGUE_CHUNK = 256
_GLU_LOGGED: list = []


def glu_epilogue_chunk(activation: str, intermediate_per_device: int) -> int:
    """Interleave chunk for the gmm_v2 SwiGLU epilogue, or 0 (plain gate|up concat, activation
    applied by XLA) when the activation is not silu or the per-device intermediate width is not
    a multiple of the chunk (``validate_glu_chunk`` needs ``2N % (2 * chunk) == 0``)."""
    if activation != "silu" or intermediate_per_device % _GLU_EPILOGUE_CHUNK != 0:
        return 0
    return _GLU_EPILOGUE_CHUNK


def _explicit_spec(x: jax.Array):
    """The operand's PartitionSpec padded to its rank when it carries an explicit-axes
    sharding (reshapes of a sharded dim then need ``out_sharding``), else None."""
    try:
        sharding = jax.typeof(x).sharding
    except Exception:  # noqa: BLE001 - concrete arrays without a typed sharding
        return None
    spec = getattr(sharding, "spec", None)
    mesh = getattr(sharding, "mesh", None)
    if spec is None or mesh is None or getattr(mesh, "empty", True):
        return None
    if not all(t == jax.sharding.AxisType.Explicit for t in mesh.axis_types):
        return None
    spec = tuple(spec) + (None,) * (x.ndim - len(spec))
    return spec


def interleave_gate_up(w0: jax.Array, w1: jax.Array, chunk: int) -> jax.Array:
    """[..., N] x2 -> [..., 2N] laid out as [w0[:chunk] | w1[:chunk] | w0[chunk:2chunk] | ...].
    Pairs with ``gmm_v2(glu_chunk=chunk)`` / ``glu_from_interleaved``. Under explicit sharding
    the split / merged last dim keeps the operand's own axis (a size-1 ``tensor`` axis is still
    an annotation the reshape has to be told about)."""
    n = w0.shape[-1]
    if w1.shape != w0.shape or n % chunk != 0:
        raise ValueError(f"interleave_gate_up: shapes {w0.shape}/{w1.shape}, chunk {chunk}")
    lead = w0.shape[:-1]
    spec = _explicit_spec(w0)
    if spec is None:
        a = w0.reshape(*lead, n // chunk, 1, chunk)
        b = w1.reshape(*lead, n // chunk, 1, chunk)
        return jnp.concatenate([a, b], axis=-2).reshape(*lead, 2 * n)
    split_spec = P(*spec[:-1], spec[-1], None, None)
    a = w0.reshape(*lead, n // chunk, 1, chunk, out_sharding=split_spec)
    b = w1.reshape(*lead, n // chunk, 1, chunk, out_sharding=split_spec)
    return jnp.concatenate([a, b], axis=-2).reshape(*lead, 2 * n, out_sharding=P(*spec))


# The down-projection memset is skipped on the SparseCore-combine path (dispatch set
# ``local_range`` / ``valid_mask``, i.e.
# should_use_sparse_core() was true for this batch): there the combine is sc_combine ->
# ragged_gather_reduce, which gathers only valid rows (SC kernel: stable partition of valid rows;
# TC fallback: jnp.where after the multiply), so the unvisited 7/8 of out[M_total, K] is never
# read. The XLA combine (small batches, no SparseCore) keeps the memset: its einsum multiplies
# every gathered row by its weight and a stale NaN row would poison the sum.
# Kernel benchmark (v7x, 8k two-window shape m_32768): down gmm 198.1 -> 152.4 us (-23 %),
# visited rows bitwise identical. Two-call (prefill) path only; the fused GLU+down path is untouched.
_MOE_DOWN_ZERO_INIT_SKIP_LOGGED = False
# Fused GLU+down: gate_up -> SwiGLU -> down as ONE pallas_call
# (kernels/gmm/megablox_gmm_kernel/gmm_v2_fused_glu.py) when the GLU-epilogue gmm_v2 path is active,
# activations are quantized and both halves would tile with one grid step per expert (GLM-5.2 v7x int4
# W4A8 decode). Bit-identical to the two calls; saves one pallas_call + one grid step per expert + the
# HBM round trip of the intermediate. Falls back to the two-call path whenever the fused wrapper declines
# the shapes. Row cap (rows per device after dispatch): decode/verify batches on GLM-5.2 2x2x1 are
# <= 2048 rows (cc64 x 4 draft x top-8); prefill is far above it and keeps the two-call path.
_MOE_FUSED_GLU_DOWN_MAX_ROWS = 2048
# Row tile for the fused GLU+down kernel, applied when M > tile_m (decode shapes: one grid step per
# expert at 32 rows; kernel default would be min(M, 128)).
_MOE_FUSED_TILE_M = 32


class EPMoE(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        layer_id: int = 0,
        quantization_config=None,
        physical_to_logical_map: "jax.Array | None" = None,
        pre_gather_quant_dtype=None,
        moe_dp_size: int = 1,
        use_sc_permute: bool | None = None,
    ):
        self.num_experts_per_tok = num_experts_per_tok
        # Opt-in SparseCore permute/unpermute kernels; ``None`` defers to the env flag.
        self.use_sc_permute = (
            moe_sc_permute_enabled_by_env() if use_sc_permute is None else bool(use_sc_permute)
        )
        self.physical_to_logical_map = physical_to_logical_map
        self.pre_gather_quant_dtype = pre_gather_quant_dtype
        # set by fuse_gate_up(): wi_0 / wi_0_scale hold [E, K, 2N] (gate ++ up), wi_1 unused
        self.fused_gate_up = False
        # set by fuse_gate_up() (silu experts): chunk width of the
        # interleaved [gate | up] layout; GEMM1 then returns silu(gate) * up directly
        self.glu_chunk = 0
        self.moe_dp_size = moe_dp_size
        self.replicate_experts = self.moe_dp_size > 1

        metadata = None if self.replicate_experts else get_global_expert_location_metadata()
        if metadata is not None and layer_id is not None:
            self.num_experts = metadata.num_physical_experts
        else:
            self.num_experts = num_experts

        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype  # original dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.original_mesh = mesh
        self.mesh = mesh
        self.activation = activation
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.hidden_size = hidden_size

        # Get quantization settings from config
        self.quantized_dtype = (
            quantization_config.get_moe_weight_dtype() if quantization_config else None
        )
        self.activation_quantized_dtype = (
            quantization_config.get_moe_activation_dtype() if quantization_config else None
        )
        if quantization_config is None:
            self.weight_block_size = None
        elif hasattr(quantization_config, "get_moe_weight_block_size"):
            self.weight_block_size = quantization_config.get_moe_weight_block_size()
        else:
            self.weight_block_size = getattr(quantization_config, "weight_block_size", None)

        if self.moe_dp_size < 1:
            raise ValueError(f"moe_dp_size must be at least 1, got {self.moe_dp_size}")
        if self.replicate_experts and self.ep_size != 1:
            raise ValueError(f"replicated EPMoE requires ep_size=1, got ep_size={self.ep_size}")
        if self.replicate_experts and (
            self.quantized_dtype is not None or self.activation_quantized_dtype is not None
        ):
            raise NotImplementedError(
                "replicated EPMoE currently supports unquantized experts only"
            )
        if not self.replicate_experts and self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by ep_size ({self.ep_size})"
            )
        if self.replicate_experts:
            if "data" not in self.mesh.axis_names or "tensor" not in self.mesh.axis_names:
                raise ValueError(
                    "replicated EPMoE requires a model mesh with ('data', 'tensor') axes; "
                    f"got {self.mesh.axis_names}"
                )
            if self.mesh.shape["data"] != self.moe_dp_size:
                raise ValueError(
                    "replicated EPMoE requires moe_dp_size to match the model mesh data axis; "
                    f"got moe_dp_size={self.moe_dp_size}, data={self.mesh.shape['data']}"
                )
            self.tp_size = self.mesh.shape["tensor"]
            self.experts_per_device = self.num_experts
            self.moe_mesh = self.mesh
            self.updated_mesh = self.mesh.abstract_mesh
            wi_sharding = P(None, None, "tensor")
            wo_sharding = P(None, "tensor", None)
        else:
            world_size = math.prod(self.mesh.shape.values())
            self.tp_size = world_size // self.ep_size
            self.experts_per_device = self.num_experts // self.ep_size

            devices = self.mesh.devices.flatten()
            self.moe_mesh = jax.sharding.Mesh(
                devices.reshape(self.ep_size, self.tp_size),
                axis_names=("expert", "tensor"),
                axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
            )

            abstract_mesh = self.mesh.abstract_mesh
            self.updated_mesh = abstract_mesh.update(
                axis_sizes=(self.ep_size, self.tp_size), axis_names=("expert", "tensor")
            )
            wi_sharding = P("expert", None, "tensor")
            wo_sharding = P("expert", "tensor", None)

        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            # MOE weights' shape is (num_experts, k, n)
            self.wi_0 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=wi_sharding,
                )
            )

            self.wi_1 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=wi_sharding,
                )
            )

            self.wo = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, intermediate_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=wo_sharding,
                )
            )

            # Scales are None by default - only set by quantize_weights() if quantization is enabled
            # gmm kernel handles None scales properly (no scaling applied)
            self.wi_0_scale = None
            self.wi_1_scale = None
            self.wo_scale = None

    def _detect_device_capabilities(self):
        try:
            devices = jax.devices()
            is_cpu_only = all(device.platform == "cpu" for device in devices)
            can_use_ragged = not is_cpu_only and hasattr(jax.lax, "ragged_all_to_all")

            device_types = [device.platform for device in devices]
            primary_device = device_types[0] if device_types else "unknown"

            return can_use_ragged, primary_device
        except Exception as _:
            return False, "cpu"

    def fuse_gate_up(self) -> bool:
        """Concatenate the gate (wi_0) and up (wi_1) expert weights along the output dim so the
        forward runs one grouped matmul [E, K, 2N] instead of two [E, K, N] calls per layer.
        Per-channel / block scales concatenate the same way (kernel-ready layout). Only for
        moe_mesh tensor == 1 (the output dim is not sharded) and static weights."""
        if self.fused_gate_up or self.wi_1 is None:
            return False
        moe_tp = int(self.moe_mesh.shape.get("tensor", 1))
        if moe_tp != 1:
            return False
        if self.weight_block_size is not None:
            # block-scaled experts (FP8 block-wise checkpoints) keep the two-call path: the fused
            # [E, K, 2N] weight / scale layouts are only built for per-channel or unquantized experts
            return False
        w0, w1 = self.wi_0.value, self.wi_1.value
        # scale normalisation and the concat need the moe mesh context (PartitionSpec use)
        with jax.set_mesh(self.moe_mesh), jax.sharding.use_abstract_mesh(self.updated_mesh):
            s0 = self._normalize_scale_for_gmm(
                self.wi_0_scale.value if self.wi_0_scale is not None else None,
                w0,
                scale_name="wi_0_scale",
            )
            s1 = self._normalize_scale_for_gmm(
                self.wi_1_scale.value if self.wi_1_scale is not None else None,
                w1,
                scale_name="wi_1_scale",
            )
            glu_chunk = glu_epilogue_chunk(self.activation, int(w0.shape[-1]))
            if glu_chunk:
                fused = interleave_gate_up(w0, w1, glu_chunk)
            else:
                fused = jnp.concatenate([w0, w1], axis=-1)
            # replicated experts (moe_dp_size > 1) have no "expert" mesh axis: keep the layout
            # the unfused parameters were created with
            ex = None if self.replicate_experts else "expert"
            self.wi_0 = nnx.Param(fused, out_sharding=P(ex, None, "tensor"))
            self.wi_1 = None
            if s0 is not None and s1 is not None:
                if glu_chunk:
                    fs = interleave_gate_up(s0, s1, glu_chunk)
                else:
                    fs = jnp.concatenate([s0, s1], axis=-1)
                self.wi_0_scale = nnx.Param(fs, out_sharding=P(ex, None, None, "tensor"))
                self.wi_1_scale = None
        self.fused_gate_up = True
        self.glu_chunk = glu_chunk
        if glu_chunk and not _GLU_LOGGED:
            _GLU_LOGGED.append(True)
            logger.info(
                "EPMoE GLU epilogue active: gate/up interleaved in "
                "%d-column chunks, GEMM1 emits silu(gate)*up from the gmm_v2 epilogue",
                glu_chunk,
            )
        return True

    def _get_wo_scale_sharding(
        self, scale: jax.Array | None = None, is_block: bool | None = None
    ) -> P:
        """Derive the sharding spec for wo_scale.

        When contracting-dimension block quantization is used (k_blocks > 1),
        wo_scale's K-dimension is partitioned across the tensor axis.
        """
        if is_block is None:
            if scale is not None:
                is_block = scale.shape[1] > 1 if scale.ndim == 4 else scale.shape[-1] > 1
            elif self.wo_scale is not None:
                wo_scale_val = (
                    self.wo_scale.value if hasattr(self.wo_scale, "value") else self.wo_scale
                )
                is_block = wo_scale_val.shape[1] > 1 if wo_scale_val is not None else False
            else:
                is_block = False
        return P("expert", "tensor" if is_block else None, None, None)

    def _normalize_scale_for_gmm(
        self,
        scale: jax.Array | None,
        weight: jax.Array,
        *,
        scale_name: str,
    ) -> jax.Array | None:
        """Normalize offline/runtime scale tensors to GMM's 4D layout.

        Accepted inputs intentionally cover the layouts we see in practice:

        - per-channel: ``[E, out_dim]``
        - already-kernel-ready: ``[E, k_blocks, 1, out_dim]``
        - sub-channel / block-channel: ``[E, out_dim, k_blocks]`` or
          ``[E, k_blocks, out_dim]``
        - offline 2D block quant: ``[E, out_blocks, k_blocks]``

        The returned tensor always matches the GMM contract
        ``[E, k_blocks, 1, out_dim]``.
        """
        if scale is None:
            return None

        # Weight layout is [E, k, n] where k=contraction dim, n=output dim.
        num_experts, in_dim, out_dim = weight.shape

        scale_gmm: jax.Array | None = None

        if scale.ndim == 4:
            if scale.shape[0] != num_experts or scale.shape[2] != 1 or scale.shape[3] != out_dim:
                raise ValueError(
                    f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                    "Expected 4D GMM scale layout [E, k_blocks, 1, out_dim]."
                )
            if self.weight_block_size is None:
                if scale.shape[1] != 1:
                    raise ValueError(
                        f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                        "Per-channel 4D GMM scales must have k_blocks=1."
                    )
            else:
                block_size_k = int(self.weight_block_size[1])
                expected_k_blocks = (in_dim + block_size_k - 1) // block_size_k
                if scale.shape[1] not in (1, expected_k_blocks):
                    raise ValueError(
                        f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                        f"Expected k_blocks dimension to be 1 or {expected_k_blocks}."
                    )
            scale_gmm = scale

        elif scale.ndim == 2 and scale.shape == (num_experts, out_dim):
            scale_gmm = scale[:, None, None, :]

        elif scale.ndim == 3:
            if scale.shape == (num_experts, 1, out_dim):
                scale_gmm = scale[:, :, None, :]
            # Support offline 2D block quant checkpoints whose scales are stored as
            # [num_experts, out_blocks, in_blocks]. GMM expects [E, k_blocks, 1, out_dim].
            elif (
                self.weight_block_size is not None
                and isinstance(self.weight_block_size, (list, tuple))
                and len(self.weight_block_size) == 2
            ):
                block_size_out = int(self.weight_block_size[0])
                block_size_k = int(self.weight_block_size[1])
                expected_out_blocks = (out_dim + block_size_out - 1) // block_size_out
                expected_k_blocks = (in_dim + block_size_k - 1) // block_size_k

                if scale.shape == (num_experts, out_dim, expected_k_blocks):
                    scale_gmm = jnp.transpose(scale, (0, 2, 1))[:, :, None, :]
                elif scale.shape == (num_experts, expected_out_blocks, expected_k_blocks):
                    scale_per_out_sharding = (
                        P("expert", None, None)
                        if scale_name == "wo_scale"
                        else P("expert", "tensor", None)
                    )
                    scale_per_out = scale.at[
                        :, jnp.arange(out_dim, dtype=jnp.int32) // block_size_out, :
                    ].get(out_sharding=scale_per_out_sharding)
                    scale_gmm = jnp.transpose(scale_per_out, (0, 2, 1))[:, :, None, :]
                elif scale.shape == (num_experts, expected_k_blocks, out_dim):
                    scale_gmm = scale[:, :, None, :]

        if scale_gmm is None:
            raise ValueError(
                f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                "Expected one of: [E, out_dim], [E, 1, out_dim], [E, k_blocks, 1, out_dim], "
                "or offline block format [E, out_blocks, k_blocks]."
            )

        final_scale_sharding = (
            self._get_wo_scale_sharding(scale_gmm)
            if scale_name == "wo_scale"
            else P("expert", None, None, "tensor")
        )
        return jax.sharding.reshard(scale_gmm, final_scale_sharding)

    def quantize_weights(self, is_static: bool = False, *, abstract: bool = False):
        """Quantize MoE weights in-place or initialize params for static loading."""
        if self.quantized_dtype is None:
            return

        def _get_block_size_k(
            *,
            hidden_size: int,
            intermediate_dim: int,
            weight_block_size: list[int] | tuple[int, int] | None,
        ) -> int | None:
            """Extract the contracting-dimension block size for MoE weights.

            EPMoE only block-quantizes along the GEMM ``K`` dimension, so for a
            configured ``(block_n, block_k)`` we consume only ``block_k`` here.
            The divisibility checks keep the later GMM scale layout well-defined.
            """
            if weight_block_size is None:
                return None
            if not (isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2):
                raise ValueError(
                    f"EPMoE weight_block_size must be a 2-element list [block_n, block_k], "
                    f"got {weight_block_size}"
                )

            block_size_k = int(weight_block_size[1])
            if block_size_k <= 0:
                raise ValueError(f"EPMoE weight_block_size[1] must be > 0, got {block_size_k}")
            if hidden_size % block_size_k != 0:
                raise ValueError(
                    f"EPMoE hidden_size={hidden_size} not divisible by block_size_k={block_size_k}"
                )
            if intermediate_dim % block_size_k != 0:
                raise ValueError(
                    f"EPMoE intermediate_dim={intermediate_dim} not divisible by block_size_k={block_size_k}"
                )
            return block_size_k

        mesh_context = (
            jax.sharding.use_abstract_mesh(self.moe_mesh.abstract_mesh)
            if abstract
            else jax.set_mesh(self.moe_mesh)
        )
        with mesh_context:
            if is_static:
                if abstract or isinstance(self.wi_0.value, jax.ShapeDtypeStruct):
                    # Prepare quantized placeholders for checkpoint and dummy
                    # loading without replacing already loaded concrete weights.
                    for name in ("wi_0", "wi_1", "wo"):
                        param = getattr(self, name)
                        param.value = jax.ShapeDtypeStruct(
                            param.value.shape,
                            self.quantized_dtype,
                            sharding=jax.sharding.NamedSharding(
                                self.moe_mesh, param.value.sharding.spec
                            ),
                        )
                # Static checkpoints will load real scale tensors later, but the
                # placeholders must already satisfy expert sharding shape rules.
                num_experts = self.wi_0.value.shape[0]
                # [E, k, n] layout: wi_0=[E, hidden_size, intermediate_dim],
                #                    wo=[E, intermediate_dim, hidden_size]
                hidden_size = self.wi_0.value.shape[1]
                intermediate_dim = self.wo.value.shape[1]

                # Compute k_blocks for block quant placeholders.
                # weight_block_size = [hf_out_block, hf_in_block] (HF convention).
                # EPMoE quantizes along axis=1 (k/contraction dim).
                block_size_k = _get_block_size_k(
                    hidden_size=hidden_size,
                    intermediate_dim=intermediate_dim,
                    weight_block_size=self.weight_block_size,
                )
                k_blocks_wi = (hidden_size // block_size_k) if block_size_k else 1
                k_blocks_wo = (intermediate_dim // block_size_k) if block_size_k else 1
                wi_scale_sharding = P("expert", None, None, "tensor")
                wo_scale_sharding = self._get_wo_scale_sharding(is_block=(k_blocks_wo > 1))
                is_abstract = isinstance(self.wi_0.value, jax.ShapeDtypeStruct)
                # Per-channel / block scales stay f32 regardless of the weight dtype: the
                # gmm kernels consume f32 scales, so a bf16 scale parameter costs one
                # convert + copy per expert layer per step (3 per MoE layer in the AOT
                # programs) and rounds the loaded scale. This also applies to int4
                # checkpoints, whose placeholders #1637 created in the model dtype.
                scale_dtype = jnp.float32

                def _make_param(shape, dtype, sharding_spec):
                    if is_abstract:
                        return nnx.Param(
                            jax.ShapeDtypeStruct(
                                shape,
                                dtype=dtype,
                                sharding=jax.sharding.NamedSharding(self.moe_mesh, sharding_spec),
                            )
                        )
                    return nnx.Param(
                        jnp.zeros(shape, dtype=dtype, out_sharding=sharding_spec),
                        out_sharding=sharding_spec,
                    )

                if hasattr(self, "wi_0_scale"):
                    del self.wi_0_scale
                self.wi_0_scale = _make_param(
                    (num_experts, k_blocks_wi, 1, intermediate_dim),
                    scale_dtype,
                    wi_scale_sharding,
                )

                if hasattr(self, "wi_1_scale"):
                    del self.wi_1_scale
                self.wi_1_scale = _make_param(
                    (num_experts, k_blocks_wi, 1, intermediate_dim),
                    scale_dtype,
                    wi_scale_sharding,
                )

                if hasattr(self, "wo_scale"):
                    del self.wo_scale
                self.wo_scale = _make_param(
                    (num_experts, k_blocks_wo, 1, hidden_size),
                    scale_dtype,
                    wo_scale_sharding,
                )

                return

            # Quantize weights along k-dim (axis=1 in [g, k, n] layout)
            # wi_0=[E, hidden_size, intermediate_dim], wo=[E, intermediate_dim, hidden_size]
            hidden_size = self.wi_0.value.shape[1]
            intermediate_dim = self.wo.value.shape[1]
            block_size_k = _get_block_size_k(
                hidden_size=hidden_size,
                intermediate_dim=intermediate_dim,
                weight_block_size=self.weight_block_size,
            )
            w0_value, w0_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_0.value,
                axis=1,
                block_size=block_size_k,
            )
            w1_value, w1_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_1.value,
                axis=1,
                block_size=block_size_k,
            )
            wo_value, wo_scale = quantize_tensor(
                self.quantized_dtype,
                self.wo.value,
                axis=1,
                block_size=block_size_k,
            )

            self.wi_0 = nnx.Param(w0_value, out_sharding=P("expert", None, "tensor"))
            self.wi_1 = nnx.Param(w1_value, out_sharding=P("expert", None, "tensor"))
            self.wo = nnx.Param(wo_value, out_sharding=P("expert", "tensor", None))

            if block_size_k is not None:
                # axis=1 quantization on [g, k, n] gives scale [g, k_blocks, n]
                # → expand to [g, k_blocks, 1, n]
                w0_scale = w0_scale[:, :, None, :]
                w1_scale = w1_scale[:, :, None, :]
                wo_scale = wo_scale[:, :, None, :]
            else:
                w0_scale = w0_scale.reshape(w0_scale.shape[0], 1, 1, w0_scale.shape[1])
                w1_scale = w1_scale.reshape(w1_scale.shape[0], 1, 1, w1_scale.shape[1])
                wo_scale = wo_scale.reshape(wo_scale.shape[0], 1, 1, wo_scale.shape[1])

            if hasattr(self, "wi_0_scale"):
                del self.wi_0_scale
            self.wi_0_scale = nnx.Param(
                w0_scale,
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wi_1_scale"):
                del self.wi_1_scale
            self.wi_1_scale = nnx.Param(
                w1_scale,
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wo_scale"):
                del self.wo_scale
            self.wo_scale = nnx.Param(
                wo_scale,
                out_sharding=self._get_wo_scale_sharding(wo_scale),
            )

    @named_scope
    def __call__(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        *,
        out_sharding: jax.sharding.NamedSharding | None = None,
    ) -> jax.Array:
        if self.replicate_experts:
            if out_sharding is None:
                out_sharding = jax.sharding.NamedSharding(
                    self.mesh,
                    P("data", *([None] * (hidden_states.ndim - 1))),
                )
            return self._call_replicated(
                hidden_states,
                topk_weights,
                topk_ids,
                out_sharding=out_sharding,
            )

        if out_sharding is None:
            out_sharding = jax.sharding.NamedSharding(self.mesh, P(*([None] * hidden_states.ndim)))

        # Translate the caller's target sharding (on self.mesh: data,tensor)
        # into shard_map out_specs (on self.moe_mesh: expert,tensor). Only
        # 'tensor' is shared between the two meshes; everything else is
        # irrelevant inside the per-expert shard_map context.
        out_specs = P(
            *[
                ("tensor" if (s == "tensor" or (isinstance(s, tuple) and "tensor" in s)) else None)
                for s in out_sharding.spec
            ]
        )
        scatter_on_tensor = "tensor" in out_specs

        # Run MoE computation on the expert-parallel mesh
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hidden_states_reshard = jax.sharding.reshard(hidden_states, P(None))
            topk_weights_reshard = jax.sharding.reshard(topk_weights, P(None))
            topk_ids_reshard = jax.sharding.reshard(topk_ids, P(None))

            # Normalize scales to GMM's 4D layout [E, k_blocks, 1, out_dim]
            w0_scale = self._normalize_scale_for_gmm(
                self.wi_0_scale.value if self.wi_0_scale is not None else None,
                self.wi_0.value,
                scale_name="wi_0_scale",
            )
            if self.fused_gate_up:
                w1_scale = w0_scale  # placeholder: the fused forward ignores the w1 operands
            else:
                w1_scale = self._normalize_scale_for_gmm(
                    self.wi_1_scale.value if self.wi_1_scale is not None else None,
                    self.wi_1.value,
                    scale_name="wi_1_scale",
                )
            wo_scale = self._normalize_scale_for_gmm(
                self.wo_scale.value if self.wo_scale is not None else None,
                self.wo.value,
                scale_name="wo_scale",
            )

            result = shard_map(
                partial(self._forward, scatter_on_tensor=scatter_on_tensor),
                mesh=self.moe_mesh,
                in_specs=(
                    P(None),
                    P(None),
                    P(None),
                    # weights [g, k, n]
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", "tensor", None),
                    # scales [g, 1, 1, n]
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, "tensor"),
                    self._get_wo_scale_sharding(wo_scale),
                    # biases [g, 1, n] (unused)
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", None, None),
                ),
                out_specs=out_specs,
                check_vma=False,
            )(
                hidden_states_reshard,
                topk_weights_reshard,
                topk_ids_reshard,
                self.wi_0.value,
                self.wi_0.value if self.fused_gate_up else self.wi_1.value,
                self.wo.value,
                w0_scale,
                w1_scale,
                wo_scale,
                None,
                None,
                None,
            )

        # The shard_map ran under updated_mesh (expert, tensor); land back on
        # the original mesh so downstream ops (residual add, layernorm) see a
        # consistent context.
        return jax.sharding.reshard(result, out_sharding)

    def _call_replicated(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        *,
        out_sharding: jax.sharding.NamedSharding,
    ) -> jax.Array:
        token_spec = P("data", *([None] * (hidden_states.ndim - 1)))
        routing_spec = P("data", *([None] * (topk_ids.ndim - 1)))
        out_spec = out_sharding.spec
        token_axis = out_spec[0] if len(out_spec) > 0 else None
        token_axes = token_axis if isinstance(token_axis, tuple) else (token_axis,)
        if self.mesh.shape["data"] > 1 and "data" not in token_axes:
            raise ValueError(
                "replicated EPMoE output must shard the token dimension over the data axis"
            )
        scatter_on_tensor = "tensor" in token_axes

        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hidden_states = jax.sharding.reshard(hidden_states, token_spec)
            topk_weights = jax.sharding.reshard(topk_weights, routing_spec)
            topk_ids = jax.sharding.reshard(topk_ids, routing_spec)
            result = shard_map(
                partial(self._forward, scatter_on_tensor=scatter_on_tensor),
                mesh=self.moe_mesh,
                in_specs=(
                    token_spec,
                    routing_spec,
                    routing_spec,
                    P(None, None, "tensor"),
                    P(None, None, "tensor"),
                    P(None, "tensor", None),
                ),
                out_specs=out_spec,
                check_vma=False,
            )(
                hidden_states,
                topk_weights,
                topk_ids,
                self.wi_0[...],
                self.wi_0[...] if self.fused_gate_up else self.wi_1[...],
                self.wo[...],
            )

        return jax.sharding.reshard(result, out_sharding)

    def _forward(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        w0_weights,
        w1_weights,
        wo_weights,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
        *,
        scatter_on_tensor: bool = False,
    ):
        expert_shard_id = (
            jnp.array(0, dtype=jnp.int32)
            if self.replicate_experts
            else jax.lax.axis_index("expert")
        )
        disp = self._dispatch_local(hidden_states, topk_ids, expert_shard_id)
        return self._experts_local(
            disp,
            topk_weights,
            w0_weights,
            w1_weights,
            wo_weights,
            w0_kernel_scale,
            w1_kernel_scale,
            wo_kernel_scale,
            w0_kernel_bias,
            w1_kernel_bias,
            wo_kernel_bias,
            scatter_on_tensor=scatter_on_tensor,
        )

    def _dispatch_local(self, hidden_states, topk_ids, expert_shard_id):
        """Per-device first half of ``_forward``: token sort, expert-shard offsets and (SparseCore
        path) the local row range / valid mask. Returns a dict consumed by ``_experts_local``."""
        inputs_2d, token_indices, sorted_selected_experts, group_sizes, permute_inverse = (
            self._permute(hidden_states, topk_ids)
        )

        group_sizes = group_sizes.astype(jnp.int32)

        group_offset = self._dispatch(group_sizes, expert_shard_id)

        local_range = None
        valid_mask = None
        # Trace-time gate: small (decode-sized) batches keep the exact XLA path so
        # their HLO is identical to the flag-off build.
        if self.use_sc_permute and should_use_sparse_core(
            token_indices.shape[0], inputs_2d.shape[-1], self.dtype
        ):
            # Sorted-row range owned by this expert shard, and which routed slots
            # (token-major order) landed on a local expert.
            csum = jnp.cumsum(group_sizes)
            start = jnp.where(group_offset == 0, 0, csum[jnp.maximum(group_offset - 1, 0)])
            end = csum[group_offset + self.experts_per_device - 1]
            if self.ep_size > 1:
                local_range = (start.astype(jnp.int32), end.astype(jnp.int32))
            flat_experts = jnp.ravel(topk_ids)
            valid_mask = (flat_experts >= group_offset) & (
                flat_experts < group_offset + self.experts_per_device
            )
        return dict(
            inputs_2d=inputs_2d,
            token_indices=token_indices,
            sorted_selected_experts=sorted_selected_experts,
            permute_inverse=permute_inverse,
            group_sizes=group_sizes,
            group_offset=group_offset,
            local_range=local_range,
            valid_mask=valid_mask,
            x_gathered=None,
        )

    def _experts_local(
        self,
        disp,
        topk_weights,
        w0_weights,
        w1_weights,
        wo_weights,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
        *,
        scatter_on_tensor: bool = False,
        combine_pin=None,
    ):
        """Per-device second half of ``_forward``: grouped matmuls, unpermute (SparseCore
        combine on the SC path) and the tensor/expert reductions. ``combine_pin`` (micro-batch
        PIN=4) adds a zero-valued data edge from a scalar to the combine input so the SparseCore
        combine cannot be scheduled ahead of that scalar's producer."""
        intermediate_output = self._gmm_compute(
            disp["inputs_2d"],
            disp["token_indices"],
            disp["group_sizes"],
            w0_weights,
            w1_weights,
            wo_weights,
            disp["group_offset"],
            w0_kernel_scale,
            w1_kernel_scale,
            wo_kernel_scale,
            w0_kernel_bias,
            w1_kernel_bias,
            wo_kernel_bias,
            local_range=disp["local_range"],
            x_gathered=disp["x_gathered"],
        )
        if combine_pin is not None:
            tiny = combine_pin.astype(jnp.float32).reshape(-1)[0]
            zero = jnp.where(jnp.isfinite(tiny), tiny * 0, 0.0)
            # The zero-valued edge goes on the routing weights ([tokens, top_k], a direct operand
            # of the SparseCore combine), not on intermediate_output ([tokens*top_k, hidden]):
            # that add could not fuse into the combine custom call and ran as its own pass
            # (4k trace: a broadcast_add_fusion bf16[16384,6144] cost 9.6 ms/chunk).
            topk_weights = topk_weights + zero.astype(topk_weights.dtype)

        output = self._unpermute(
            intermediate_output,
            disp["sorted_selected_experts"],
            topk_weights,
            valid_mask=disp["valid_mask"],
            inverse=disp.get("permute_inverse"),
        )
        return self._reduce_expert_output(output, scatter_on_tensor)

    def _reduce_expert_output(self, output, scatter_on_tensor):
        """Tail of ``_experts_local`` after the combine: tensor-axis RS/AR and the expert-axis reduce."""

        # Reduce on the "tensor" axis. RS (psum_scatter) when caller asked
        # for SP layout on the token dim, AR (psum) otherwise. The matching
        # out_specs is set in __call__ from the same source of truth.
        if self.tp_size > 1:
            if scatter_on_tensor:
                output = jax.lax.psum_scatter(output, "tensor", scatter_dimension=0, tiled=True)
            else:
                # inside the moe_mesh shard_map: the kernel indexes the ("expert", "tensor") axes
                output = maybe_oneshot_psum(output, "tensor", self.moe_mesh.axis_names, site="moe")
        if self.ep_size > 1:
            output = self._combine(output)

        return output

    # ── two-stage call (prefill micro-batch PIN=4) ────────────────────────────
    # dispatch_stage() runs the permute and the SparseCore dispatch gather as its own
    # shard_map and hands back the per-device gathered rows plus a tiny anchor derived
    # from them; experts_stage() finishes the layer. The model uses the anchor to pin
    # the OTHER window's all-reduce operand (so that all-reduce enters the SparseCore
    # stream after this window's dispatch gather) and ``combine_pin`` to keep this
    # window's SparseCore combine behind the other window's all-reduce result. Values
    # are identical to __call__ (same ops per device, only split across two calls).
    def _stage_specs(self, wo_scale=None):
        return (
            # weights [g, k, n]
            P("expert", None, "tensor"),
            P("expert", None, "tensor"),
            P("expert", "tensor", None),
            # scales [g, k_blocks, 1, n]: the wo scale is K-sharded when it is block-scaled
            # (same rule as __call__; the per-channel form is replicated)
            P("expert", None, None, "tensor"),
            P("expert", None, None, "tensor"),
            self._get_wo_scale_sharding(wo_scale),
            # biases [g, 1, n] (unused)
            P("expert", None, "tensor"),
            P("expert", None, "tensor"),
            P("expert", None, None),
        )

    def _stage_weights(self):
        w0_scale = self._normalize_scale_for_gmm(
            self.wi_0_scale.value if self.wi_0_scale is not None else None,
            self.wi_0.value,
            scale_name="wi_0_scale",
        )
        if self.fused_gate_up:
            w1_scale = w0_scale
        else:
            w1_scale = self._normalize_scale_for_gmm(
                self.wi_1_scale.value if self.wi_1_scale is not None else None,
                self.wi_1.value,
                scale_name="wi_1_scale",
            )
        wo_scale = self._normalize_scale_for_gmm(
            self.wo_scale.value if self.wo_scale is not None else None,
            self.wo.value,
            scale_name="wo_scale",
        )
        return (
            self.wi_0.value,
            self.wi_0.value if self.fused_gate_up else self.wi_1.value,
            self.wo.value,
            w0_scale,
            w1_scale,
            wo_scale,
            None,
            None,
            None,
        )

    def _sc_stage(self, num_tokens: int) -> bool:
        return bool(
            self.use_sc_permute
            and should_use_sparse_core(
                num_tokens * self.num_experts_per_tok, self.hidden_size, self.dtype
            )
        )

    def _dispatch_stage_local(self, hidden_states, topk_ids, *, sc: bool):
        expert_shard_id = jax.lax.axis_index("expert")
        disp = self._dispatch_local(hidden_states, topk_ids, expert_shard_id)
        inputs_2d, token_indices = disp["inputs_2d"], disp["token_indices"]
        if sc and disp["local_range"] is not None:
            x = sc_dispatch_gather(inputs_2d, token_indices, *disp["local_range"])
            start, end = disp["local_range"]
        else:
            x = inputs_2d[token_indices]
            start = end = jnp.zeros((), jnp.int32)
        x = x.astype(self.dtype)
        valid_mask = disp["valid_mask"]
        if valid_mask is None:
            valid_mask = jnp.ones((token_indices.shape[0],), jnp.bool_)
        offs = jnp.stack([disp["group_offset"].astype(jnp.int32), start, end])
        # anchor: one element of the gathered rows (a data edge on the dispatch gather's done)
        anchor = x[:1, :1].astype(jnp.float32)
        return (
            x[None],
            token_indices[None],
            disp["sorted_selected_experts"][None],
            disp["group_sizes"][None],
            offs[None],
            valid_mask[None],
            anchor,
        )

    def dispatch_stage(self, hidden_states, topk_ids):
        """Stage 1 of the two-stage call: returns ``(state, anchor)``; ``state`` holds the
        per-device gathered rows and routing metadata (leading ``expert`` axis), ``anchor`` is a
        replicated ``f32[1, 1]`` on the model mesh that depends on the dispatch gather."""
        if self.replicate_experts:
            raise NotImplementedError("dispatch_stage requires expert parallelism (ep_size > 1)")
        sc = self._sc_stage(hidden_states.shape[0])
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hs = jax.sharding.reshard(hidden_states, P(None))
            ti = jax.sharding.reshard(topk_ids, P(None))
            outs = shard_map(
                partial(self._dispatch_stage_local, sc=sc),
                mesh=self.moe_mesh,
                in_specs=(P(None), P(None)),
                out_specs=(
                    P("expert"),
                    P("expert"),
                    P("expert"),
                    P("expert"),
                    P("expert"),
                    P("expert"),
                    P(None),
                ),
                check_vma=False,
            )(hs, ti)
        state = dict(
            zip(("x", "token_indices", "sorted", "group_sizes", "offs", "valid_mask"), outs[:6])
        )
        state["sc"] = sc
        anchor = jax.sharding.reshard(
            outs[6], jax.sharding.NamedSharding(self.mesh, P(*([None] * outs[6].ndim)))
        )
        return state, anchor

    def _experts_stage_local(
        self,
        x,
        token_indices,
        sorted_selected_experts,
        group_sizes,
        offs,
        valid_mask,
        topk_weights,
        combine_pin,
        *weights,
        sc: bool,
        scatter_on_tensor: bool,
    ):
        x, token_indices, sorted_selected_experts, group_sizes, offs, valid_mask = (
            x[0],
            token_indices[0],
            sorted_selected_experts[0],
            group_sizes[0],
            offs[0],
            valid_mask[0],
        )
        disp = dict(
            inputs_2d=x,  # unused once x_gathered is given (shape/dtype only)
            token_indices=token_indices,
            sorted_selected_experts=sorted_selected_experts,
            group_sizes=group_sizes,
            group_offset=offs[0],
            local_range=(offs[1], offs[2]) if (sc and self.ep_size > 1) else None,
            valid_mask=valid_mask if sc else None,
            x_gathered=x,
        )
        return self._experts_local(
            disp,
            topk_weights,
            *weights,
            scatter_on_tensor=scatter_on_tensor,
            combine_pin=combine_pin,
        )

    def experts_stage(
        self,
        state,
        topk_weights,
        *,
        combine_pin=None,
        out_sharding: jax.sharding.NamedSharding | None = None,
    ):
        """Stage 2 of the two-stage call (same output as ``__call__``). ``combine_pin``: any array
        on the model mesh; its first element pins the SparseCore combine (zero-valued edge)."""
        if out_sharding is None:
            out_sharding = jax.sharding.NamedSharding(self.mesh, P(None, None))
        out_specs = P(
            *[
                ("tensor" if (s == "tensor" or (isinstance(s, tuple) and "tensor" in s)) else None)
                for s in out_sharding.spec
            ]
        )
        scatter_on_tensor = "tensor" in out_specs
        pin_src = (
            combine_pin[tuple(slice(0, 1) for _ in range(combine_pin.ndim))].astype(jnp.float32)
            if combine_pin is not None
            else jnp.zeros((1, 1), jnp.float32)
        )
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            tw = jax.sharding.reshard(topk_weights, P(None))
            pin = jax.sharding.reshard(pin_src, P(None))
            stage_weights = self._stage_weights()
            result = shard_map(
                partial(
                    self._experts_stage_local, sc=state["sc"], scatter_on_tensor=scatter_on_tensor
                ),
                mesh=self.moe_mesh,
                in_specs=(
                    P("expert"),
                    P("expert"),
                    P("expert"),
                    P("expert"),
                    P("expert"),
                    P("expert"),
                    P(None),
                    P(None),
                )
                + self._stage_specs(stage_weights[5]),
                out_specs=out_specs,
                check_vma=False,
            )(
                state["x"],
                state["token_indices"],
                state["sorted"],
                state["group_sizes"],
                state["offs"],
                state["valid_mask"],
                tw,
                pin,
                *stage_weights,
            )
        return jax.sharding.reshard(result, out_sharding)

    def _gmm_compute(
        self,
        inputs_2d,
        token_indices,
        group_sizes,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        group_offset,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
        local_range=None,
        x_gathered=None,
    ):
        if token_indices.shape[0] == 0:
            return jnp.zeros((0, wo_kernel.shape[-1]), dtype=inputs_2d.dtype)

        # indexed_gmm: gather sorted_inputs here instead of in _permute,
        # so XLA can fuse the gather with the matmul and avoid materializing
        # the full [M*top_k, D] sorted_inputs tensor at peak memory.
        pre_gather_q = getattr(self, "pre_gather_quant_dtype", None)
        if x_gathered is not None:
            # dispatch_stage() already gathered the sorted rows (two-stage micro-batch path)
            x = x_gathered.astype(self.dtype)
        elif pre_gather_q is not None:
            x_q, x_scale = quantize_tensor_simple(inputs_2d, pre_gather_q, dim=-1)
            x = x_q[token_indices]
            x_scale = x_scale[token_indices]
            x = (x.astype(jnp.float32) * x_scale).astype(self.dtype)
        elif local_range is not None:
            # SparseCore ragged gather: only rows [start, end) (local experts) are
            # materialized; gmm never reads the others thanks to group_offset.
            x = sc_dispatch_gather(inputs_2d, token_indices, *local_range).astype(self.dtype)
        else:
            x = inputs_2d[token_indices].astype(self.dtype)

        # NOTE: do NOT pad LHS / bump group_sizes here. The megablox backend
        # ``gmm`` (sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py:67-73)
        # already pads ``lhs`` to its required alignment (32 for v2, 128 for
        # v1), bumps ``group_sizes[-1]`` accordingly, and slices the output
        # back to the original ``m`` afterwards. An outer pre-pad is at best
        # redundant; in practice the previous workaround pre-padded to a
        # hard-coded ``128`` which forced v2 (alignment=32) into a 4x larger
        # tile, hit a kernel auto-tiler edge case at decode bs=8 / top_k=8
        # (m=64 -> 128) and triggered an on-device SparseCore halt.
        group_sizes = group_sizes.astype(jnp.int32)
        act_q_dtype = self.activation_quantized_dtype

        gmm_kwargs = dict(
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            group_offset=group_offset,
            maybe_quantize_lhs=act_q_dtype is not None,
            acc_dtype=jnp.float32,
        )

        if (
            x.shape[0] <= _MOE_FUSED_GLU_DOWN_MAX_ROWS
            and self.fused_gate_up
            and self.glu_chunk
            and act_q_dtype is not None
            and w0_kernel_bias is None
            and wo_kernel_bias is None
            and is_tpu_runtime()
        ):
            fused_out = self._gmm_fused_glu_down(
                x,
                w0_kernel,
                w0_kernel_scale,
                wo_kernel,
                wo_kernel_scale,
                group_sizes,
                group_offset,
            )
            if fused_out is not None:
                return fused_out

        # === GEMM1: x @ w0 and x @ w1 ===
        if self.fused_gate_up:
            # one [E, K, 2N] grouped matmul (gate ++ up), see fuse_gate_up()
            layer_w01 = gmm(
                lhs=x,
                rhs=w0_kernel,
                rhs_scale=w0_kernel_scale,
                rhs_bias=w0_kernel_bias,
                zero_initialize=False,
                activation_quantized_dtype=act_q_dtype,
                glu_chunk=self.glu_chunk or None,
                **gmm_kwargs,
            )
            if not self.glu_chunk:
                half = layer_w01.shape[-1] // 2
                layer_w0, layer_w1 = layer_w01[:, :half], layer_w01[:, half:]
        else:
            layer_w0 = gmm(
                lhs=x,
                rhs=w0_kernel,
                rhs_scale=w0_kernel_scale,
                rhs_bias=w0_kernel_bias,
                zero_initialize=False,
                activation_quantized_dtype=act_q_dtype,
                **gmm_kwargs,
            )
            layer_w1 = gmm(
                lhs=x,
                rhs=w1_kernel,
                rhs_scale=w1_kernel_scale,
                rhs_bias=w1_kernel_bias,
                zero_initialize=False,
                activation_quantized_dtype=act_q_dtype,
                **gmm_kwargs,
            )

        # === Activation ===
        if self.glu_chunk:
            # silu(gate) * up was applied in the gmm_v2 epilogue (interleaved layout)
            intermediate_layer = layer_w01
        elif self.activation == "silu":
            intermediate_layer = jax.nn.silu(layer_w0) * layer_w1
        elif self.activation == "gelu":
            intermediate_layer = jax.nn.gelu(layer_w0) * layer_w1
        elif self.activation == "swigluoai":
            gate = jnp.clip(layer_w0, max=self.swiglu_limit)
            up = jnp.clip(layer_w1, -self.swiglu_limit, self.swiglu_limit)
            intermediate_layer = (up + 1.0) * gate * jax.nn.sigmoid(gate * self.swiglu_alpha)
        else:
            raise ValueError(f"Unsupported activation function {self.activation}")

        # === GEMM2: intermediate @ wo ===
        return gmm(
            lhs=intermediate_layer,
            rhs=wo_kernel,
            rhs_scale=wo_kernel_scale,
            rhs_bias=wo_kernel_bias,
            zero_initialize=self._down_zero_initialize(local_range, intermediate_layer.dtype),
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )

    @staticmethod
    def _down_zero_initialize(local_range, dtype) -> bool:
        """zero_initialize for the two-call down gmm: False on the SparseCore-combine path
        (see the note above). The TensorCore fallback of ``sc_combine`` (non-bf16
        intermediates) masks the weights, not the products, so it keeps the memset."""
        global _MOE_DOWN_ZERO_INIT_SKIP_LOGGED
        if local_range is not None and dtype == jnp.bfloat16:
            if not _MOE_DOWN_ZERO_INIT_SKIP_LOGGED:
                _MOE_DOWN_ZERO_INIT_SKIP_LOGGED = True
                logger.info(
                    "EPMoE down gmm: zero_initialize skipped on the SparseCore-combine path"
                )
            return False
        return True

    def _gmm_fused_glu_down(
        self,
        x,
        w0_kernel,
        w0_kernel_scale,
        wo_kernel,
        wo_kernel_scale,
        group_sizes,
        group_offset,
    ):
        """gate_up -> SwiGLU -> down in one pallas_call; None when the shapes do not qualify (caller falls back).
        Mirrors the megablox backend's 32-row alignment padding / un-padding around gmm_v2."""
        m = x.shape[0]
        pad_size = (-m) % 32
        lhs = x
        if pad_size:
            lhs = jax.lax.pad(x, jnp.zeros((), dtype=x.dtype), ((0, pad_size, 0), (0, 0, 0)))
            group_sizes = group_sizes.at[-1].add(pad_size)
        out = gmm_v2_fused_glu(
            lhs,
            w0_kernel,
            w0_kernel_scale,
            wo_kernel,
            wo_kernel_scale,
            group_sizes,
            group_offset,
            glu_chunk=int(self.glu_chunk),
            preferred_element_type=self.dtype,
            acc_dtype=jnp.float32,
            zero_initialize=True,
            tile_m=_MOE_FUSED_TILE_M if lhs.shape[0] > _MOE_FUSED_TILE_M else None,
        )
        if out is None:
            return None
        return out[:m] if pad_size else out

    def _dispatch(self, group_sizes, expert_shard_id):
        if self.ep_size <= 1:
            return jnp.array(0, dtype=jnp.int32)
        group_offset = jnp.array(expert_shard_id * self.experts_per_device, dtype=jnp.int32)
        return group_offset

    def _get_all_to_all_params(
        self,
        tokens_group: jax.Array,
        shard_id: jax.Array,
        start_idx: jax.Array,
        *,
        ep_size: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        input_offsets = jnp.full(ep_size, start_idx, dtype=tokens_group.dtype)
        send_sizes = jnp.repeat(tokens_group[shard_id], ep_size)
        output_offset = jnp.concatenate(
            (jnp.array([0], dtype=tokens_group.dtype), jnp.cumsum(tokens_group[:-1]))
        )[shard_id]
        output_offsets = jnp.repeat(output_offset, ep_size)
        recv_sizes = tokens_group

        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _combine(self, data):
        if reduce_deferred() and fused_moe_enabled():
            return data  # partial; the model reduces routed+shared with one fused_all_reduce
        return jax.lax.psum(data, "expert")

    def _permute(self, inputs, top_k_indices):
        if inputs.ndim != 2:
            raise ValueError(
                "EPMoE._permute expects 2-D hidden states [tokens, hidden], "
                f"got shape {inputs.shape}"
            )

        expected_indices_shape = (inputs.shape[0], self.num_experts_per_tok)
        if top_k_indices.shape != expected_indices_shape:
            raise ValueError(
                "EPMoE._permute expects routing indices with shape "
                f"{expected_indices_shape}, got shape {top_k_indices.shape}"
            )

        n_rows = inputs.shape[0] * self.num_experts_per_tok
        if 0 < n_rows <= _PERMUTE_KERNEL_MAX_ROWS:
            sorted_selected_experts, token_indices, group_sizes, inverse = _permute_indices_small(
                top_k_indices, self.num_experts
            )
            return inputs, token_indices, sorted_selected_experts, group_sizes, inverse
        flatten_selected_experts = jnp.ravel(top_k_indices)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts, stable=True)
        # token_indices: maps each sorted position to the original token index.
        # Pass to _gmm_compute so the gather happens there (indexed_gmm pattern),
        # avoiding a full [M*top_k, D] materialization in _permute.
        token_indices = sorted_selected_experts // self.num_experts_per_tok

        group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)

        return (
            inputs,
            token_indices,
            sorted_selected_experts,
            group_sizes,
            None,
        )

    def _unpermute(
        self, intermediate, sorted_selected_experts, weights, valid_mask=None, inverse=None
    ):
        top_k = self.num_experts_per_tok
        if weights.ndim != 2 or weights.shape[1] != top_k:
            raise ValueError(
                "EPMoE._unpermute expects 2-D routing weights "
                f"[tokens, {top_k}], got shape {weights.shape}"
            )

        expected_tokens = weights.shape[0] * top_k
        if sorted_selected_experts.ndim != 1 or sorted_selected_experts.shape[0] != expected_tokens:
            raise ValueError(
                "EPMoE._unpermute expects 1-D sorted routing indices with "
                f"{expected_tokens} entries, got shape {sorted_selected_experts.shape}"
            )

        actual_tokens = intermediate.shape[0]

        if actual_tokens != expected_tokens:
            if actual_tokens > expected_tokens:
                intermediate = intermediate[:expected_tokens]
            else:
                padding_size = expected_tokens - actual_tokens
                padding = jnp.zeros((padding_size, intermediate.shape[1]), dtype=intermediate.dtype)
                intermediate = jnp.concatenate([intermediate, padding], axis=0)

        # Un-permute map: the small-batch routing-index kernel output (bit-exact) or the inverse
        # permutation via a second sort -- XLA lowers the equivalent 1-D scatter
        # (zeros.at[sorted].set(arange)) to ~6 select fusions per layer (10 ms per 4k prefill
        # on v7x); an int32 key sort of the same length is ~8x cheaper and yields the identical
        # permutation (keys are unique).
        argsort_indices = inverse if inverse is not None else jnp.argsort(sorted_selected_experts)
        if self.use_sc_permute and valid_mask is not None:
            # SparseCore fused gather + top-k weighted reduce (falls back to XLA when
            # SparseCore is absent or the problem is too small to benefit).
            return sc_combine(
                intermediate,
                argsort_indices,
                jnp.reshape(weights, (-1,)),
                valid_mask,
                self.num_experts_per_tok,
            ).astype(self.dtype)

        weights_fp32 = weights.astype(jnp.float32)

        # Static token-count branch: small (decode) batches avoid the unrolled
        # per-k gather overhead; large (prefill) batches avoid the fp32
        # (tokens, top_k, hidden) intermediate. Benchmarks in the PR description.
        if weights.shape[0] <= 256:
            # argsort_indices is a permutation of range(expected_tokens): skip the
            # gather's bounds-mask fusions (compare_and / select), keep the gather.
            unsort_intermediate = intermediate.at[argsort_indices].get(mode="promise_in_bounds")
            reshaped_intermediate = jnp.reshape(
                unsort_intermediate,
                (weights.shape[0], top_k, -1),
            )
            output = jnp.einsum(
                "BKE,BK -> BE",
                reshaped_intermediate.astype(jnp.float32),
                weights_fp32,
            )
        else:
            grouped_indices = jnp.reshape(argsort_indices, (weights.shape[0], top_k))
            output = None
            for k in range(top_k):
                contribution = (
                    jnp.take(intermediate, indices=grouped_indices[:, k], axis=0).astype(
                        jnp.float32
                    )
                    * weights_fp32[:, k, None]
                )
                output = contribution if output is None else output + contribution

        final_output = output.astype(self.dtype)

        return final_output


# create_moe_weights_mapping is utility function to generate weight mapping for MOE layers
def create_moe_weights_mapping(
    prefix: str,
    target_prefix: str,
    num_experts: int,  # num logical experts
    expert_type_names: tuple[str, str, str] = (
        "gate_proj",
        "up_proj",
        "down_proj",
    ),  # expert source names [gate, up, down]
    expert_concat_axis_map: dict[
        str, int
    ] = None,  # Map from source weight name to its concatenation axis (default is None)
    moe_backend: str = "epmoe",
    moe_path: str = "mlp",
    source_expert_pattern: str = "experts.{i}",
    physical_to_logical_map=None,  # np.ndarray shape (num_physical,) or None
    weight_suffix: str = "weight",
) -> dict:
    """Generate a unified mapping dictionary for MoE layer expert weights."""
    if moe_backend == "epmoe":
        expert_type_map = {
            expert_type_names[0]: "wi_0",
            expert_type_names[1]: "wi_1",
            expert_type_names[2]: "wo",
        }
    elif moe_backend in ("fused", "fused_v2"):
        expert_type_map = {
            expert_type_names[0]: "w1",
            expert_type_names[1]: "w3",
            expert_type_names[2]: "w2",
        }
    else:
        raise ValueError(f"Unsupported MoE backend: {moe_backend}")

    if expert_concat_axis_map is None:
        expert_concat_axis_map = {}

    mappings = {}
    for source_name, target_name in expert_type_map.items():
        # Target path for JAX model parameters (matching EPMoE internal variables)
        target_path_base = f"{target_prefix}.{moe_path}.{target_name}"

        # Source weight paths for logical experts only
        expert_keys = [
            f"{prefix}.{moe_path}.{source_expert_pattern.format(i=i)}.{source_name}.{weight_suffix}"
            for i in range(num_experts)
        ]

        if moe_backend == "epmoe":
            # Weights are transposed from HF [n, k] to [k, n], stacked to [g, k, n].
            # wi_0/wi_1: [g, hidden_size, intermediate_dim] -> P("expert", None, "tensor")
            # wo:        [g, intermediate_dim, hidden_size] -> P("expert", "tensor", None)
            sharding = (
                ("expert", "tensor", None) if target_name == "wo" else ("expert", None, "tensor")
            )
            transpose = True
        elif moe_backend in ("fused", "fused_v2"):
            # Fused MoE kernel shards experts across the full EP mesh, i.e. the
            # product of ("data", "tensor"). Shard expert dim (axis=0) across
            # both mesh axes so each device owns a disjoint expert slice.
            sharding = (("data", "tensor"), None, None)
            transpose = True
        else:
            raise ValueError(f"Unsupported MoE backend: {moe_backend}")

        concat_axis = expert_concat_axis_map.get(source_name)

        # Use  prefix to indicate aggregated MoE weight loading
        mappings[f"{target_path_base}"] = WeightSpec(
            target_path=target_path_base,
            sources=tuple(expert_keys),
            sharding=sharding,
            transpose=transpose,
            concat_axis=concat_axis,
            physical_to_logical_map=physical_to_logical_map,
        )

    return mappings
