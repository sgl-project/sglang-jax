import logging
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.model_config import ModelConfig

from .specs import WeightSpec

logger = logging.getLogger(__name__)


class TensorLayout:
    """Pure layout conversions: accept target schemas, return arrays, never assign NNX state."""

    def __init__(
        self,
        model_config: ModelConfig,
        mesh: Mesh,
    ):
        self.model_config = model_config
        self.mesh = mesh
        if hasattr(model_config, "num_attention_heads"):
            self.num_heads = model_config.num_attention_heads
            # Use original count for replication logic
            self.num_kv_heads = model_config.get_total_num_kv_heads()
            self.hidden_size = model_config.hidden_size
            # Read head_dim / v_head_dim from hf_text_config rather than model_config:
            # patch_model_config writes mc.head_dim for KV-cache / MemoryPools sizing,
            # but the loader needs the per-layer truth for split-QKV weight slicing.
            # hf_text_config stays unpatched so split-QKV slicing is correct for
            # hybrid-attention models.
            hf_cfg = getattr(model_config, "hf_text_config", model_config)
            self.head_dim_original = getattr(hf_cfg, "head_dim", self.hidden_size // self.num_heads)

            self.head_dim_pad = (self.head_dim_original + 127) // 128 * 128 - self.head_dim_original
            self.head_dim = self.head_dim_original
            self.v_head_dim = getattr(hf_cfg, "v_head_dim", self.head_dim_original)
        if hasattr(self.mesh, "shape") and "tensor" in self.mesh.shape:
            self.sharding_size = self.mesh.shape["tensor"]
        else:
            self.sharding_size = 1

    def _maybe_convert_epmoe_scale_for_kernel(
        self,
        weight: jax.Array,
        target: jax.ShapeDtypeStruct,
        target_path: str,
    ) -> jax.Array:
        """Convert offline EPMoE/FusedEPMoE scales into kernel-ready 4D layout.

        Offline checkpoints may store MoE scales in one of several compact
        layouts, for example:

        - per-channel: ``[E, out_dim]``
        - block-channel: ``[E, out_dim, k_blocks]`` or ``[E, k_blocks, out_dim]``
        - 2D block quant: ``[E, out_blocks, k_blocks]``

        The runtime GMM kernel consumes ``[E, k_blocks, 1, out_dim]``.
        The FusedEPMoE kernel consumes ``[E, k_blocks, 1, out_groups_padded]``.

        This helper performs the cheap layout conversion during weight loading
        so the forward path does not need to reinterpret checkpoint tensors.
        """
        # Match both EPMoE (wi_0_scale etc.) and FusedEPMoE (w1_scale etc.)
        if not target_path.endswith(
            ("wi_0_scale", "wi_1_scale", "wo_scale", "w1_scale", "w2_scale", "w3_scale")
        ):
            return weight

        if weight.ndim == 4 or target.ndim != 4:
            return weight

        param_shape = target.shape
        num_experts = param_shape[0]

        # Compressed-tensors per-channel checkpoints (e.g. Ling-2.6-1T) emit
        # each expert's scale as ``[out_dim, 1]``; after stacking they show up
        # here as ``[E, out_dim, 1]``. The kernel still wants
        # ``[E, 1, 1, out_dim]`` (k_blocks=1), so squeeze the trailing 1 first
        # and let the per-channel path below handle the 4D promotion.
        if weight.ndim == 3 and weight.shape[0] == num_experts and weight.shape[-1] == 1:
            weight = jnp.squeeze(weight, axis=-1)

        # --- FusedEPMoE legacy 2D block-wise placeholder ---
        # Older placeholders may use (E, K_groups, N_groups, 1).
        if param_shape[3] == 1 and param_shape[2] > 1 and weight.ndim == 3:
            return weight[..., None]

        # --- EPMoE / GMM path (also FusedEPMoE 1D sub-channel) ---
        if param_shape[2] != 1:
            return weight
        num_experts, k_blocks, _, out_dim = param_shape
        if weight.ndim == 2 and weight.shape == (num_experts, out_dim):
            return weight[:, None, None, :]

        if weight.ndim != 3:
            return weight

        quant_cfg = getattr(self.model_config, "quantization_config", None)
        weight_block_size = getattr(quant_cfg, "weight_block_size", None)
        block_size_out = None
        if isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2:
            block_size_out = int(weight_block_size[0])

        is_fused_scale = target_path.endswith(("w1_scale", "w2_scale", "w3_scale"))
        if is_fused_scale and block_size_out is not None and block_size_out > 0:
            expected_out_blocks = (out_dim + block_size_out - 1) // block_size_out
            if weight.shape == (num_experts, k_blocks, expected_out_blocks):
                logger.info(
                    "Expanding fused MoE 2D scale %s from %s to fast kernel layout %s",
                    target_path,
                    weight.shape,
                    target.shape,
                )
                idx = jnp.arange(out_dim) // block_size_out
                return jnp.take(weight, idx, axis=2)[:, :, None, :]
            if weight.shape == (num_experts, expected_out_blocks, k_blocks):
                logger.info(
                    "Transposing+expanding fused MoE 2D scale %s from %s to fast kernel layout %s",
                    target_path,
                    weight.shape,
                    target.shape,
                )
                weight = jnp.transpose(weight, (0, 2, 1))
                idx = jnp.arange(out_dim) // block_size_out
                return jnp.take(weight, idx, axis=2)[:, :, None, :]

        if weight.shape == (num_experts, out_dim, k_blocks):
            return jnp.expand_dims(jnp.transpose(weight, (0, 2, 1)), axis=2)

        if weight.shape == (num_experts, k_blocks, out_dim):
            return weight[:, :, None, :]

        # FusedEPMoE 2D block-quant checkpoints (e.g., MiMo) often store scales
        # compactly as [E, K_blocks, N_blocks] or [E, N_blocks, K_blocks], while
        # the fused kernel expects [E, K_blocks, 1, out_groups_padded].
        if is_fused_scale and weight.ndim == 3:
            if weight.shape[0] == num_experts and weight.shape[1] == k_blocks:
                n_groups = weight.shape[2]
                if n_groups <= out_dim:
                    if n_groups < out_dim:
                        logger.info(
                            "Padding fused MoE scale %s from %s to kernel layout %s",
                            target_path,
                            weight.shape,
                            target.shape,
                        )
                        weight = jnp.pad(weight, ((0, 0), (0, 0), (0, out_dim - n_groups)))
                    return weight[:, :, None, :]
            if weight.shape[0] == num_experts and weight.shape[2] == k_blocks:
                n_groups = weight.shape[1]
                if n_groups <= out_dim:
                    logger.info(
                        "Transposing fused MoE scale %s from %s to kernel layout %s",
                        target_path,
                        weight.shape,
                        target.shape,
                    )
                    weight = jnp.transpose(weight, (0, 2, 1))
                    if n_groups < out_dim:
                        weight = jnp.pad(weight, ((0, 0), (0, 0), (0, out_dim - n_groups)))
                    return weight[:, :, None, :]

        if not (isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2):
            return weight

        block_size_out = int(weight_block_size[0])
        if block_size_out <= 0:
            return weight

        expected_out_blocks = (out_dim + block_size_out - 1) // block_size_out
        if weight.shape != (num_experts, expected_out_blocks, k_blocks):
            return weight

        logger.info(
            "Converting offline EPMoE scale %s from shape %s to GMM layout %s",
            target_path,
            weight.shape,
            target.shape,
        )
        out_block_ids = np.arange(out_dim, dtype=np.int32) // block_size_out
        scale_per_out = jnp.take(weight, jnp.asarray(out_block_ids), axis=1)
        return jnp.expand_dims(jnp.transpose(scale_per_out, (0, 2, 1)), axis=2)

    def _maybe_expand_linear_block_scale(
        self,
        weight: jax.Array,
        target: jax.ShapeDtypeStruct,
        target_path: str,
    ) -> jax.Array:
        """Expand 2D block-quant scale [out_blocks, in_blocks] to 3D [in_blocks, 1, n_out] at load time."""
        if not target_path.endswith("weight_scale"):
            return weight

        # Per-channel compressed-tensors checkpoints (e.g. Ling-2.6-1T) ship the
        # weight scale as a 2D ``[out_dim, 1]`` tensor while the model placeholder
        # is a 1D ``[out_dim]``. Squeeze the trailing singleton so the shapes
        # line up — this is the per-channel sibling of the block-quant expand
        # path below.
        if (
            weight.ndim == 2
            and weight.shape[-1] == 1
            and target.ndim == 1
            and target.shape[0] == weight.shape[0]
        ):
            return jnp.squeeze(weight, axis=-1)

        # Only convert when checkpoint has 2D scale and model expects 3D.
        if weight.ndim != 2 or target.ndim != 3:
            return weight

        # Model param shape: [in_blocks, 1, n_out]
        if target.shape[1] != 1:
            return weight

        quant_cfg = getattr(self.model_config, "quantization_config", None)
        weight_block_size = getattr(quant_cfg, "weight_block_size", None)
        if not (isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2):
            return weight

        block_size_out = int(weight_block_size[0])
        if block_size_out <= 0:
            return weight

        from sgl_jax.srt.kernels.quantized_matmul.blockwise_utils import (
            expand_block_scale,
        )

        n_out = int(target.shape[2])
        logger.info(
            "Expanding linear block-quant scale %s from %s to kernel-ready layout [%d, 1, %d]",
            target_path,
            weight.shape,
            weight.shape[1],
            n_out,
        )
        expanded = expand_block_scale(weight, n_out, block_size_out)
        # The expansion inherits the compact 2D scale's (effectively replicated)
        # layout, but the model placeholder declares the kernel-boundary sharding
        # (e.g. P(None, None, "tensor")). jax 0.8.x shard_map silently reshards on
        # this textual mismatch; jax 0.10.x checks strictly and raises. Same class
        # of fix as the MoE/MLA boundaries in #1493.
        target_sharding = getattr(target, "sharding", None)
        if target_sharding is not None and expanded.ndim == len(
            getattr(target_sharding, "spec", ())
        ):
            expanded = jax.sharding.reshard(expanded, target_sharding)
        return expanded

    def transform(
        self,
        hf_key: str,
        weight: jax.Array,
        mapping: WeightSpec,
        targets: dict[str, jax.ShapeDtypeStruct],
    ) -> tuple[jax.Array, ...]:
        """Return converted values in target_path order, preserving FP8 storage."""
        if mapping.transpose_axes is not None and not hf_key.endswith(".bias"):
            weight = jnp.transpose(weight, mapping.transpose_axes)
        elif mapping.transpose and not hf_key.endswith(".bias"):
            weight = jnp.transpose(weight, (1, 0))
        if isinstance(mapping.target_path, list):
            return self._split_weight(targets, hf_key, weight, mapping)
        return (self._single_weight(targets, hf_key, weight, mapping),)

    def transform_experts(self, weight, mapping, target):
        """Finish an already-stacked expert tensor using the same path during tracing."""
        if mapping.reshape is not None:
            weight = jnp.reshape(weight, mapping.reshape)
        if mapping.repeat is not None:
            axis, count = mapping.repeat
            weight = jnp.repeat(weight, count, axis=axis)
        weight = self._maybe_convert_epmoe_scale_for_kernel(weight, target, mapping.target_path)
        return self._cast(weight, target)

    @staticmethod
    def _cast(weight, target):
        return (
            weight
            if weight.dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2)
            else weight.astype(target.dtype)
        )

    def _single_weight(
        self,
        targets: dict[str, jax.ShapeDtypeStruct],
        hf_key: str,
        weight: jax.Array,
        mapping: WeightSpec,
    ):
        assert isinstance(mapping.target_path, str)
        jax_path: str = mapping.target_path
        processed_weight = weight

        # Apply output_multiplier_scale to lm_head weights (matching PyTorch implementation)
        if "lm_head" in hf_key and hasattr(
            getattr(self.model_config, "hf_config", None), "output_multiplier_scale"
        ):
            logger.info(
                "Applying output_multiplier_scale (%.2f) to %s",
                self.model_config.hf_config.output_multiplier_scale,
                hf_key,
            )
            processed_weight = processed_weight.astype(jnp.float32)
            processed_weight = (
                processed_weight * self.model_config.hf_config.output_multiplier_scale
            )

        if mapping.reshape is not None:
            processed_weight = jnp.reshape(processed_weight, mapping.reshape)
        if mapping.repeat is not None:
            axis, times = mapping.repeat
            processed_weight = jnp.repeat(processed_weight, times, axis=axis)
        if mapping.kv_head_padding:
            processed_weight = self._apply_kv_head_padding(processed_weight, hf_key)

        if mapping.pad_width is not None:
            processed_weight = jnp.pad(processed_weight, mapping.pad_width)

        assert mapping.sharding is not None
        sharded_weight = self._shard_weight(processed_weight, mapping.sharding)

        try:
            target = targets[jax_path]

            # Expand 2D block-quant scale to 3D kernel-ready layout.
            sharded_weight = self._maybe_expand_linear_block_scale(sharded_weight, target, jax_path)

            logger.debug(
                "Loading %s -> %s, shape: %s, transpose: %s",
                hf_key,
                jax_path,
                processed_weight.shape,
                mapping.transpose,
            )
            return self._cast(sharded_weight, target)
        except Exception as e:
            logger.error("Failed to load %s -> %s: %s", hf_key, jax_path, str(e))
            raise

    def _split_weight(
        self,
        targets: dict[str, jax.ShapeDtypeStruct],
        hf_key: str,
        weight: jax.Array,
        mapping: WeightSpec,
    ):
        if mapping.split_sizes is not None:
            if (
                len(mapping.split_sizes) != len(mapping.target_path)
                or sum(mapping.split_sizes) != weight.shape[mapping.split_axis]
            ):
                raise ValueError(
                    f"Split shape mismatch for {hf_key}: {weight.shape}, expected {mapping.split_sizes}"
                )
            parts = jnp.split(
                weight,
                np.cumsum(mapping.split_sizes)[:-1].tolist(),
                axis=mapping.split_axis,
            )
            return tuple(
                self._single_weight(targets, hf_key, part, replace(mapping, target_path=path))
                for part, path in zip(parts, mapping.target_path)
            )
        return self._split_qkv_weight(targets, hf_key, weight, mapping)

    def _split_qkv_weight(self, targets, hf_key, weight, mapping):
        import math

        v_dim = self.v_head_dim
        heads = (self.num_heads, self.num_kv_heads, self.num_kv_heads)
        dims = (self.head_dim_original, self.head_dim_original, v_dim)
        sizes = [n * d for n, d in zip(heads, dims)]
        is_scale = "scale" in hf_key
        block = getattr(
            getattr(getattr(self, "model_config", None), "quantization_config", None),
            "weight_block_size",
            None,
        )
        block_scale = is_scale and weight.ndim == 2 and block is not None
        axis = 0 if is_scale or hf_key.endswith(".bias") or not mapping.transpose else 1
        if block_scale:
            sizes[:2] = [math.ceil(size / block[0]) for size in sizes[:2]]
            sizes[2] = weight.shape[0] - sum(sizes[:2])
        if sum(sizes) != weight.shape[axis]:
            raise ValueError(f"QKV shape mismatch for {hf_key}: {weight.shape}, expected {sizes}")
        splits = jnp.split(weight, np.cumsum(sizes)[:-1].tolist(), axis=axis)
        outputs = []
        for part, path, nheads, dim in zip(splits, mapping.target_path, heads, dims):
            pad = (-dim) % 128
            if mapping.head_dim_padding and pad and not block_scale:
                shape = list(part.shape)
                shape[axis : axis + 1] = [nheads, dim]
                part = part.reshape(shape)
                pads = [(0, 0)] * part.ndim
                pads[axis + 1] = (0, pad)
                part = jnp.pad(part, pads)
                shape[axis : axis + 2] = [nheads * (dim + pad)]
                part = part.reshape(shape)
            if mapping.kv_head_padding and ("k_proj" in path or "v_proj" in path):
                part = self._apply_kv_head_padding(part, path)
            part = self._shard_weight(part, mapping.sharding)
            target = targets[path]
            part = self._maybe_expand_linear_block_scale(part, target, path)
            outputs.append(self._cast(part, target))
        return tuple(outputs)

    def _shard_weight(
        self,
        weight: jax.Array,
        sharding_spec: tuple,
        mesh: jax.sharding.Mesh | None = None,
    ) -> jax.Array:
        if mesh is None:
            mesh = self.mesh
        target_sharding = jax.sharding.NamedSharding(mesh, P(*sharding_spec))
        # Reads may use checkpoint axes; enforce the final parameter layout.
        return jax.device_put(weight, target_sharding)

    def _apply_kv_head_padding(self, weight: jax.Array, hf_key: str) -> jax.Array:
        """Apply KV head padding/replication when tp_size > total_kv_heads.

        Handles:
        1. Bias/Scale (1D or 2D with shape[0]=heads) -> Pad Axis 0
        2. Standard Weight (2D with shape[1]=heads*dim) -> Pad Axis 1
        3. Static Quant Weight (2D with shape[0]=heads*dim) -> Pad Axis 0
        """
        if not (
            any(proj in hf_key for proj in ["k_proj", "v_proj"])
            and self.model_config.needs_kv_head_replication(self.sharding_size)
        ):
            return weight

        total_kv_heads = self.model_config.get_total_num_kv_heads()
        num_replicas = self.model_config.get_num_kv_head_replicas(self.sharding_size)
        padding_strategy = self.model_config.get_kv_padding_strategy()

        target_axis = -1
        step_size = -1

        dim0 = weight.shape[0]
        if dim0 == total_kv_heads:
            target_axis = 0
            step_size = 1
        elif dim0 == total_kv_heads * self.head_dim:
            target_axis = 0
            step_size = self.head_dim

        if target_axis == -1 and weight.ndim > 1:
            dim1 = weight.shape[1]
            if dim1 == total_kv_heads * self.head_dim:
                target_axis = 1
                step_size = self.head_dim
            elif hasattr(self.model_config.hf_text_config, "swa_head_dim"):
                full_dim, swa_dim, orig_swa_heads, orig_full_heads, target_heads = (
                    self.model_config.get_swa_weight_params()
                )
                if dim1 == full_dim and self.sharding_size > 1:
                    total_kv_heads = 1
                    num_replicas = self.sharding_size
                    target_axis = 1
                    step_size = full_dim
                    logger.info(
                        "MQA Full KV replication: matching dim1=%d to 1 head, replicating %d times",
                        dim1,
                        num_replicas,
                    )
                elif dim1 == orig_full_heads * full_dim and target_heads > orig_full_heads:
                    total_kv_heads = orig_full_heads
                    num_replicas = target_heads // orig_full_heads
                    target_axis = 1
                    step_size = full_dim
                    logger.info(
                        "GQA Full KV replication: matching dim1=%d to %d heads, replicating %d times to %d heads",
                        dim1,
                        total_kv_heads,
                        num_replicas,
                        target_heads,
                    )
                elif dim1 == orig_swa_heads * swa_dim and self.sharding_size > orig_swa_heads:
                    total_kv_heads = orig_swa_heads
                    num_replicas = self.sharding_size // orig_swa_heads
                    target_axis = 1
                    step_size = swa_dim
                    logger.info(
                        "GQA SWA KV replication: matching dim1=%d to %d heads, replicating %d times",
                        dim1,
                        total_kv_heads,
                        num_replicas,
                    )

        if target_axis == -1:
            return weight

        if padding_strategy == "replicate":
            replicated_parts = []

            for original_head_id in range(total_kv_heads):
                start = original_head_id * step_size
                end = (original_head_id + 1) * step_size

                part = weight[start:end] if target_axis == 0 else weight[:, start:end]

                for _ in range(num_replicas):
                    replicated_parts.append(part)

            weight = jnp.concatenate(replicated_parts, axis=target_axis)

        elif padding_strategy == "zero":
            target_heads_total = total_kv_heads * num_replicas

            target_len = (
                target_heads_total if step_size == 1 else target_heads_total * self.head_dim
            )

            current_len = weight.shape[target_axis]
            padding_len = target_len - current_len

            if padding_len > 0:
                pad_shape = list(weight.shape)
                pad_shape[target_axis] = padding_len

                padding = jnp.zeros(tuple(pad_shape), dtype=weight.dtype)
                weight = jnp.concatenate([weight, padding], axis=target_axis)

        return weight
