"""Qwen3.8-Flash-Next (``qwen4_exp``).

Qwen3.5's backbone with three substitutions. The layer schedule is the same --
``full_attention_interval`` picks 12 full-attention layers out of 48 and the
rest run Gated DeltaNet -- so the GDN and MoE blocks are reused as they are.
What differs:

* **Hyper Connections replace the residual stream.** Layers carry ``hc_count``
  parallel streams, ``[T, hc_count * hidden_size]``, and every block is wrapped
  in a ``mix`` / ``combine`` pair that owns the normalization. There is no
  ``input_layernorm``, no ``post_attention_layernorm`` and no final ``norm``.
* **Full attention is QSA.** The indexer's weights live here, in the attention
  module, and its per-step outputs are handed to the backend, the way
  ``glm5_moe`` hands DSA's.
* **One GDN layer also carries the N-gram embedding**, added into the hyper
  streams before the attention block reads them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.layers.attention.qsa_indexer import QSAIndexer
from sgl_jax.srt.layers.embeddings import (
    Embed,
    MRotaryEmbedding,
    ParallelLMHead,
    RotaryEmbedding,
)
from sgl_jax.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.model_loader.weights import WeightSpec
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2_moe import Qwen2MoeMLP
from sgl_jax.srt.models.qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5MoeBlock,
    Qwen3_5MoeForConditionalGeneration,
    _create_qwen3_5_weight_mappings,
)


class Qwen4ExpAttention(nnx.Module):
    """Full attention plus the QSA indexer whose weights it owns.

    The indexer produces what the sparse backend needs to select blocks, and
    that travels down as keyword arguments, the way ``glm5_moe`` passes DSA's.
    Selection itself needs batch metadata the model does not carry, so the
    backend does it; this module only projects and compresses.

    The indexer rotates its queries with the attention's own rotary embedding,
    so ``rotary_dim`` (head_dim * partial_rotary_factor) has to fit inside the
    indexer's narrower head.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        text_cfg = config.text_config
        self.mesh = mesh
        self.layer_id = layer_id
        self.hidden_size = text_cfg.hidden_size
        self.num_heads = text_cfg.num_attention_heads
        self.num_kv_heads = text_cfg.num_key_value_heads
        self.head_dim = text_cfg.head_dim
        self.scaling = self.head_dim**-0.5

        def _linear(in_size, out_size, axes, name):
            return LinearBase(
                input_size=in_size,
                output_size=out_size,
                mesh=mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=axes,
                scope_name=name,
            )

        # The output gate doubles the query projection: [q | gate] per head.
        self.q_proj = _linear(
            self.hidden_size, self.num_heads * 2 * self.head_dim, (None, "tensor"), "q_proj"
        )
        self.k_proj = _linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, (None, "tensor"), "k_proj"
        )
        self.v_proj = _linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, (None, "tensor"), "v_proj"
        )
        self.o_proj = _linear(
            self.num_heads * self.head_dim, self.hidden_size, ("tensor", None), "o_proj"
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, epsilon=text_cfg.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, epsilon=text_cfg.rms_norm_eps)

        rotary_dim = int(self.head_dim * float(text_cfg.partial_rotary_factor))
        rope_scaling = text_cfg.rope_scaling or {}
        common = dict(
            rotary_dim=rotary_dim,
            max_position_embeddings=text_cfg.max_position_embeddings,
            base=int(text_cfg.rope_theta),
            is_neox_style=True,
            dtype=dtype,
        )
        if "mrope_section" in rope_scaling:
            self.rotary_emb = MRotaryEmbedding(
                head_size=self.head_dim,
                mrope_section=rope_scaling["mrope_section"],
                mrope_interleaved=bool(rope_scaling.get("mrope_interleaved", False)),
                **common,
            )
        else:
            self.rotary_emb = RotaryEmbedding(head_size=self.head_dim, **common)

        # The indexer rotates only the leading rotary_dim of its own narrower
        # head, so it takes a rotary sized to that slice. Its positions are
        # derived from the group index and are scalar by construction, which is
        # why this one is never the multimodal variant.
        self.indexer_rotary_emb = RotaryEmbedding(head_size=rotary_dim, **common)
        self.indexer = QSAIndexer(
            hidden_size=self.hidden_size,
            indexer_n_heads=text_cfg.indexer_n_heads,
            indexer_kv_heads=text_cfg.indexer_kv_heads,
            indexer_head_dim=text_cfg.indexer_head_dim,
            indexer_budget=text_cfg.indexer_budget,
            indexer_compress_ratio=text_cfg.indexer_compress_ratio,
            rotary_dim=rotary_dim,
            mesh=mesh,
            rms_norm_eps=text_cfg.rms_norm_eps,
            params_dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            layer_id=layer_id,
        )

    def _heads(self, x: jax.Array, n_heads: int, width: int) -> jax.Array:
        return x.reshape(
            x.shape[0],
            n_heads,
            width,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )

    def _indexer_step(self, hidden_states, forward_batch):
        """The indexer inputs the backend needs, or ``None`` when it keeps no
        compressed cache for this layer.

        Only the projection happens here. The backend compresses and writes the
        keys itself, inside its shard_map, where each data-parallel rank sees
        its own requests, so it is handed the indexer and its rotary too.

        Reads ``forward_batch.positions``, the scalar sequence, rather than the
        positions the attention was handed, which are multimodal under mrope:
        the indexer's rotary is the plain one, and a compressed key is rotated
        at its group's first position, an index only the scalar sequence has.
        """
        # Flash-Next is hybrid, so its backend is the wrapper that routes the
        # full-attention layers to full_attn_backend.
        backend = forward_batch.attn_backend
        backend = getattr(backend, "full_attn_backend", backend)
        slot_map = getattr(backend, "full_slot", None)
        if slot_map is None or self.layer_id not in slot_map:
            return None
        indexer_q, raw_key = self.indexer.project(
            hidden_states, forward_batch.positions, self.indexer_rotary_emb
        )
        return {
            "indexer_q": indexer_q,
            "indexer_k": raw_key,
            "indexer": self.indexer,
            "indexer_rotary_emb": self.indexer_rotary_emb,
        }

    def __call__(self, positions, hidden_states, forward_batch, token_to_kv_pool):
        T = hidden_states.shape[0]
        qsa_kwargs = self._indexer_step(hidden_states, forward_batch) or {}

        q_raw, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q_gate = self._heads(q_raw, self.num_heads, 2 * self.head_dim)
        q, gate = q_gate[..., : self.head_dim], q_gate[..., self.head_dim :]
        k = self._heads(k, self.num_kv_heads, self.head_dim)
        v = self._heads(v, self.num_kv_heads, self.head_dim)

        # The gate skips both the norm and the rotation.
        q, k = self.rotary_emb(positions, self.q_norm(q), self.k_norm(k))

        attn_out, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool, **qsa_kwargs)
        attn_out = self._heads(attn_out, self.num_heads, self.head_dim) * jax.nn.sigmoid(gate)
        out, _ = self.o_proj(attn_out.reshape(T, self.num_heads * self.head_dim))
        return out, kv_fused


def _hc_config(text_cfg: PretrainedConfig, mesh, dtype) -> HyperConnectionConfig:
    return HyperConnectionConfig(
        hc_count=text_cfg.hc_count,
        hidden_size=text_cfg.hidden_size,
        hc_lowrank=text_cfg.hc_lowrank,
        rms_norm_eps=text_cfg.rms_norm_eps,
        hc_per_branch_norm=True,
        params_dtype=dtype,
        mesh=mesh,
    )


class Qwen4ExpDecoderLayer(nnx.Module):
    """One backbone layer: an attention-family block and an MLP block, each
    wrapped in its own hyper connection.

    The two blocks are why there are two ``GatedResidual`` instances per layer
    rather than one per layer type: a GDN layer and a full-attention layer both
    have an attention slot and an MLP slot.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        text_cfg = config.text_config
        self.layer_id = layer_id
        self.hidden_size = text_cfg.hidden_size
        self.hc_count = text_cfg.hc_count
        self.is_full_attn = layer_id in text_cfg.full_attention_layer_ids
        self.is_moe = text_cfg.is_moe

        if self.is_full_attn:
            self.self_attn = Qwen4ExpAttention(config, mesh, layer_id, dtype=dtype)
        else:
            self.self_attn = Qwen3_5GatedDeltaNet(config, mesh, layer_id, dtype=dtype)

        # ``ple_layer_ids`` is 1-based, matching the checkpoint's own numbering.
        self.ple = None
        if (layer_id + 1) in text_cfg.ple_layer_ids:
            from sgl_jax.srt.layers.ngram_embedding import NGramEmbedding

            self.ple = NGramEmbedding(text_cfg, mesh, params_dtype=dtype)

        if self.is_moe:
            self.mlp = Qwen3_5MoeBlock(config, mesh, layer_id, dtype=dtype)
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=text_cfg.hidden_size,
                intermediate_size=text_cfg.intermediate_size,
                mesh=mesh,
                layer_id=layer_id,
                dtype=dtype,
                gate_up_down_bias=False,
            )

        hc = _hc_config(text_cfg, mesh, dtype)
        self.attn_hyper_connection = GatedResidual(hc, scope_name="attn_hyper_connection")
        self.mlp_hyper_connection = GatedResidual(hc, scope_name="mlp_hyper_connection")

    def _to_streams(self, hidden_states: jax.Array) -> jax.Array:
        """Widen ``[T, HS]`` to ``[T, HC*HS]`` by repeating the embedding.

        Only the first layer sees a narrow input; the rest receive what the
        previous layer's ``combine`` produced.
        """
        width = self.hc_count * self.hidden_size
        if hidden_states.shape[-1] == width:
            return hidden_states
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected {self.hidden_size} or {width} on the last axis, "
                f"got {hidden_states.shape[-1]}"
            )
        return jnp.concatenate([hidden_states] * self.hc_count, axis=-1)

    def __call__(
        self,
        positions,
        hidden_states,
        forward_batch,
        memory_pools,
        dispatch_info=None,
    ):
        hidden_states = self._to_streams(hidden_states)
        ple_state = None

        if self.ple is not None:
            hidden_states = hidden_states + self.ple(
                hidden_states, forward_batch, memory_pools.recurrent_state_pool
            )

        mixed, carry = self.attn_hyper_connection.mix(hidden_states)
        pool = (
            memory_pools.token_to_kv_pool
            if self.is_full_attn
            else memory_pools.recurrent_state_pool
        )
        block_out, attn_state = self.self_attn(positions, mixed, forward_batch, pool)
        hidden_states = self.attn_hyper_connection.combine(block_out, carry)

        mixed, carry = self.mlp_hyper_connection.mix(hidden_states)
        if self.is_moe:
            block_out, topk_ids = self.mlp(mixed, forward_batch, dispatch_info)
        else:
            block_out, topk_ids = self.mlp(mixed), None
        hidden_states = self.mlp_hyper_connection.combine(block_out, carry)

        return hidden_states, attn_state, topk_ids, ple_state


class Qwen4ExpModel(nnx.Module):
    """Embedding, the 48 backbone layers, and the mixer that reads the streams
    back down to one.

    There is no final norm: the mixer's own ``hc_norm`` is the last
    normalization, and it is built read-only because nothing downstream injects
    back into the streams.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        text_cfg = config.text_config
        self.config = config
        self.embed_tokens = Embed(
            num_embeddings=text_cfg.vocab_size,
            features=text_cfg.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                Qwen4ExpDecoderLayer(config, mesh, layer_id=i, dtype=dtype)
                for i in range(text_cfg.num_hidden_layers)
            ]
        )
        self.hyper_connection_mixer = GatedResidual(
            _hc_config(text_cfg, mesh, dtype),
            use_combine=False,
            scope_name="hyper_connection_mixer",
        )

    def __call__(self, forward_batch: ForwardBatch, memory_pools):
        input_embeds = (
            forward_batch.input_embedding
            if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            else None
        )
        hidden_states = (
            self.embed_tokens(forward_batch.input_ids) if input_embeds is None else input_embeds
        )
        positions = forward_batch.mrope_positions
        if positions is None:
            positions = forward_batch.positions

        from sgl_jax.srt.layers.attention.qsa_sparse_backend import QSAFusedCache

        layers_kv_fused = []
        layers_compressed = []
        layers_rings = []
        layers_rec_buffers = []
        layers_conv_buffers = []
        layers_topk_ids = []
        for layer in self.layers:
            hidden_states, attn_state, topk_ids, ple_state = layer(
                positions,
                hidden_states,
                forward_batch,
                memory_pools,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            if isinstance(attn_state, QSAFusedCache):
                layers_kv_fused.append(attn_state.kv)
                layers_compressed.append(attn_state.compressed)
                layers_rings.append(attn_state.ring)
            elif layer.is_full_attn:
                layers_kv_fused.append(attn_state)
            else:
                rec_buf, conv_buf_list = attn_state
                layers_rec_buffers.append(rec_buf)
                layers_conv_buffers.append(conv_buf_list)
            if ple_state is not None:
                layers_conv_buffers.append(ple_state)
            if topk_ids is not None:
                layers_topk_ids.append(topk_ids)

        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        # Under QSA the KV pool also takes back each layer's compressed indexer
        # cache and ring, as one tuple led by the KV list.
        kv_update = (
            (layers_kv_fused, layers_compressed, layers_rings)
            if layers_compressed
            else layers_kv_fused
        )
        return (
            hidden_states,
            kv_update,
            (layers_rec_buffers, layers_conv_buffers),
            layers_topk_ids,
        )


class Qwen4ExpForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.model = Qwen4ExpModel(config, mesh, dtype=dtype)

    def __call__(self, forward_batch: ForwardBatch, memory_pools):
        return self.model(forward_batch, memory_pools)


class Qwen4ExpForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    """Flash-Next's entry point.

    The checkpoint layout it loads from is Qwen3.5's -- same GDN striping, same
    pre-fused experts, same ``model.language_model.*`` rooting -- so everything
    below the backbone is inherited and only the backbone and the mapping table
    are replaced.

    It is text-only: the checkpoint's vision sub-config is carried along, but no
    tower is built from it, so the vision weights are skipped.
    """

    causal_lm_class = Qwen4ExpForCausalLM
    vision_model_class = None

    def _weight_mappings(self, hf_config):
        return _create_qwen4_exp_weight_mappings(hf_config, getattr(self, "lm_head", None))


EntryClass = [Qwen4ExpForConditionalGeneration]


_HC_PROJECTIONS = ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")


def _hyper_connection_mappings(src: str, dst: str, *, use_combine: bool) -> dict:
    """The four tensors of one ``GatedResidual``, or three without the inject.

    They ship as ``[out, in]`` and stay replicated: ``input_mix_weight_up``
    emits ``HC*HS``, so splitting it would need an all-reduce to rebuild a
    stream the very next op reads whole.
    """
    out = {
        f"{src}.hc_norm.weight": WeightSpec(
            target_path=f"{dst}.hc_norm.weight", sharding=(None,), transpose=False
        )
    }
    for name in _HC_PROJECTIONS:
        if name == "block_inject_weight" and not use_combine:
            continue
        out[f"{src}.{name}.weight"] = WeightSpec(
            target_path=f"{dst}.{name}.weight", sharding=(None, None), transpose=True
        )
    return out


def _create_qwen4_exp_weight_mappings(hf_config, lm_head: ParallelLMHead | None = None):
    """Qwen3.5's table, minus the norms Hyper Connections absorbed, plus the
    hyper connections themselves and the QSA indexer.

    Returns ``(mappings, visual_skip_patterns, mtp_skip_patterns)``.
    """
    mappings, visual_skip, mtp_skip = _create_qwen3_5_weight_mappings(hf_config, lm_head)

    tc = hf_config.text_config
    num_layers = int(tc.num_hidden_layers)
    full_attn_ids = set(tc.full_attention_layer_ids)

    # The hc_norm inside every mix is the only normalization left, so the three
    # RMSNorms Qwen3.5 keeps have no weights to load.
    absorbed = {"model.language_model.norm.weight"}
    for i in range(num_layers):
        absorbed.add(f"model.language_model.layers.{i}.input_layernorm.weight")
        absorbed.add(f"model.language_model.layers.{i}.post_attention_layernorm.weight")
    missing = absorbed - set(mappings)
    if missing:
        raise RuntimeError(f"expected these in the Qwen3.5 table but found none: {sorted(missing)}")
    for key in absorbed:
        del mappings[key]

    for i in range(num_layers):
        src = f"model.language_model.layers.{i}"
        dst = f"language_model.model.layers.{i}"
        for block in ("attn_hyper_connection", "mlp_hyper_connection"):
            mappings.update(
                _hyper_connection_mappings(f"{src}.{block}", f"{dst}.{block}", use_combine=True)
            )

        if i not in full_attn_ids:
            continue
        idx_src = f"{src}.self_attn.indexer"
        idx_dst = f"{dst}.self_attn.indexer"
        mappings[f"{idx_src}.index_qk_proj.weight"] = WeightSpec(
            target_path=f"{idx_dst}.index_qk_proj.weight",
            sharding=(None, None),
            transpose=True,
        )
        for norm in ("q_layernorm", "k_layernorm"):
            mappings[f"{idx_src}.{norm}.weight"] = WeightSpec(
                target_path=f"{idx_dst}.{norm}.weight", sharding=(None,), transpose=False
            )

    # Read-only: nothing downstream injects back into the streams.
    mappings.update(
        _hyper_connection_mappings(
            "model.language_model.hyper_connection_mixer",
            "language_model.model.hyper_connection_mixer",
            use_combine=False,
        )
    )

    if not tc.ple_layer_ids:
        # No layer builds the N-gram module, but the checkpoint still ships its
        # tensors. The loader treats both skip lists alike.
        mtp_skip = [*mtp_skip, r"^model\.language_model\.layers\.\d+\.ple\."]
    return mappings, visual_skip, mtp_skip
