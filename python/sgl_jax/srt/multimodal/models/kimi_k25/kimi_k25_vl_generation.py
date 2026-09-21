from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3ForCausalLM
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import KimiK25ModelVitConfig
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vit import (
    Kimi_K25_VisionModel,
    create_kimi_vision_weight_mappings,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader

logger = logging.getLogger(__name__)


class KimiK25ForConditionalGeneration(DeepseekV3ForCausalLM, InModelMultimodalContract):
    """Kimi-K2.5 on the in-model multimodal path.

    The vision tower lives inside this class as ``self.visual``. The engine calls
    ``get_input_embeddings()`` to embed text and ``get_multimodal_encode_funcs()``
    to encode media, then merges the two itself
    (``multimodal/in_model/host_orchestration.py``).

    Why Kimi does not reuse Qwen's ``run_mrope_vision_model``/``lane_packing``
    helpers: ``lane_packing`` asserts
    ``feature_patches == grid_patches == placeholder_patches``, where
    ``placeholder_patches`` is the placeholder-token count times the merge unit.
    Kimi's ``sd2_tpool`` merger averages the temporal axis away, so a grid
    ``(t, h, w)`` produces ``t*h*w`` patches but only ``h*w/merge_area``
    placeholder tokens. For any video chunk with ``t > 1`` that assertion fails.
    Those helpers are Qwen conveniences, not contract obligations - the contract
    only requires an item-ordered ``[capacity, hidden]`` array - so Kimi supplies
    its own encode function below.
    """

    def __init__(
        self,
        config=None,
        dtype=None,
        mesh=None,
        rngs: nnx.Rngs | None = None,
    ):
        # super().__init__ sets self.config to text_config; the full VL config is
        # only needed here to pull the text half out of it.
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16
        self.mesh = mesh

        # text_config.quantization_config is a raw dict from the JSON.
        # ModelConfig already handles quantization at the top-level hf_config, but
        # due to quantization config being nested in text_config, it still in JSON
        # Clear it here so FusedMoE doesn't receive a raw dict as the config contains 'pack-quantized'
        # format which isn't yet supported.
        if isinstance(getattr(self.text_config, "quantization_config", None), dict):
            self.text_config.quantization_config = None

        super().__init__(
            config=self.text_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        self.hf_weight_prefix = "language_model."

        # The model loader constructs models as ``cls(config, dtype=, mesh=)`` and
        # never passes rngs, so a default is required here.
        #
        # The dataclass defaults are the shipped Kimi-K2.5 architecture: every
        # field matches the checkpoint's own ``vision_config`` exactly, so there
        # is nothing to overlay from ``config``.
        self.vision_config = KimiK25ModelVitConfig()
        self.visual = Kimi_K25_VisionModel(
            self.vision_config,
            dtype=self.dtype,
            rngs=rngs or nnx.Rngs(0),
            mesh=self.mesh,
        )
        self._encode_vision_fn: Callable | None = None

    # ------------------------------------------------------------------
    # InModelMultimodalContract
    # ------------------------------------------------------------------

    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        return self.model.embed_tokens

    def get_multimodal_encode_funcs(self):
        # Kimi's processor reports images and video chunks in one grid_thws
        # tensor and routes both through the same tower, so every visual
        # modality maps to the same encoder. IMAGE is what the processor tags
        # today; the other two are registered so a future tagging change cannot
        # silently drop embeddings.
        return {
            Modality.IMAGE: self.encode_vision_items,
            Modality.MULTI_IMAGES: self.encode_vision_items,
            Modality.VIDEO: self.encode_vision_items,
        }

    def encode_vision_items(self, items: list[MultimodalDataItem]) -> jax.Array:
        """Encode media items into one item-ordered ``[tokens, hidden]`` array.

        Row ``i`` of the result is the ``i``-th visual token in item order, which
        is exactly what ``host_orchestration._gather_merge`` expects. The tower's
        temporal pooling means each item contributes ``h*w/merge_area`` rows
        regardless of its frame count.
        """
        if not items:
            return jnp.zeros((0, self.vision_config.text_hidden_size), dtype=self.dtype)

        grids, features = self._collect_grids_and_features(items)
        pixel_values = np.concatenate(features, axis=0)

        encode = self._ensure_encoder()
        (
            rope_freqs_cis,
            cu_seqlens,
            abs_pos_embs,
            merge_indices,
            merge_weights,
        ) = self.visual.vision_tower.compute_aux_arrays(grids)

        # TODO(kimi-buckets): pixel_values / merge_indices are passed at their
        # natural size, so a new grid combination triggers a recompile. Padding
        # them to fixed capacities (and reporting those from
        # get_multimodal_embedding_packed_capacities) is tracked separately
        # because the bucket ladder depends on the deployed frame budget.
        return encode(
            pixel_values.astype(self.dtype),
            abs_pos_embs,
            rope_freqs_cis,
            cu_seqlens,
            merge_indices,
            merge_weights,
        )

    @staticmethod
    def _collect_grids_and_features(
        items: list[MultimodalDataItem],
    ) -> tuple[tuple[tuple[int, int, int], ...], list[np.ndarray]]:
        grids: list[tuple[int, int, int]] = []
        features: list[np.ndarray] = []
        for item in items:
            grid = item.get("image_grid_thw")
            if grid is None:
                raise ValueError("Kimi-K2.5 multimodal item is missing image_grid_thw.")
            grid_rows = np.asarray(grid, dtype=np.int32).reshape(-1, 3)
            if grid_rows.shape[0] != 1:
                raise ValueError(
                    "Kimi-K2.5 expects exactly one grid per multimodal item, got "
                    f"{grid_rows.shape[0]}. The processor must split media per grid."
                )
            feature = np.asarray(item.feature)
            expected = int(grid_rows[0].prod())
            if feature.shape[0] != expected:
                raise ValueError(
                    "Kimi-K2.5 item feature rows do not match its grid: "
                    f"{feature.shape[0]} != {expected}."
                )
            grids.append(tuple(int(value) for value in grid_rows[0]))
            features.append(feature)
        return tuple(grids), features

    def _ensure_encoder(self) -> Callable:
        """Build the jitted tower + projector, once.

        ``nnx.split`` captures parameter values, so this cannot run before
        ``load_weights`` -- which is why ``load_weights`` calls it at the end.
        The guard makes the per-request call a no-op.
        """
        if self._encode_vision_fn is not None:
            return self._encode_vision_fn

        model_def, model_state = nnx.split(self.visual)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        def _encode_vision_impl(
            model_def,
            model_state_def,
            model_state_leaves,
            pixel_values,
            abs_pos_embs,
            rope_freqs_cis,
            cu_seqlens,
            merge_indices,
            merge_weights,
        ):
            state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            visual = nnx.merge(model_def, state)
            hidden_states = visual.vision_tower.compute_hidden_states(
                pixel_values,
                abs_pos_embs,
                rope_freqs_cis,
                cu_seqlens,
                merge_indices,
                merge_weights,
            )
            return visual.mm_projector(hidden_states)

        jitted = jax.jit(_encode_vision_impl, static_argnames=["model_state_def"])

        # The graphdef and the parameter leaves are fixed for the life of the
        # model, so bind them once; callers pass only the per-request arrays.
        encode = functools.partial(jitted, model_def, model_state_def, model_state_leaves)

        self._encode_vision_fn = encode
        return encode

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings(model_config)
        # Language-model and vision weights live in the same checkpoint under
        # different prefixes, so one pass covers both.
        weight_mappings.update(
            create_kimi_vision_weight_mappings(
                self.vision_config.vt_num_hidden_layers,
                target_prefix="visual.",
            )
        )
        loader.load_weights_from_safetensors(weight_mappings)

        for layer in self.model.layers:
            layer.self_attn.post_load_weights()
        # Build the encoder now that parameters are real. nnx.split captures
        # values, so an encoder built earlier would hold initialisation noise,
        # and building it later would mutate a module attribute after
        # ModelRunner.initialize_jit has already snapshotted the graph.
        self._ensure_encoder()
        logger.info("Kimi K2.5 language model and vision tower weights loaded successfully!")


EntryClass = KimiK25ForConditionalGeneration
