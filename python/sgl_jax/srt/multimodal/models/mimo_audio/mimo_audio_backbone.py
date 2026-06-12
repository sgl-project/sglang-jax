import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.mimo_audio_transformer import MiMoAudioTransformer
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.multimodal.configs.mimo_audio.mimo_audio_backbone_config import (
    MiMoAudioArguments,
    MiMoAudioBackboneConfig,
    MiMoSamplerConfig,
)
from sgl_jax.srt.multimodal.models.mimo_audio.mimo_audio_backbone_weights_mapping import (
    to_mappings,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader


class MiMoAudioForCausalLM(nnx.Module):
    def __init__(
        self,
        config: MiMoAudioBackboneConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        args: MiMoAudioArguments | None = None,
    ):
        self.config = config
        self.args = args or MiMoAudioArguments()
        self.mesh = mesh
        self.dtype = dtype

        self.model = MiMoAudioTransformer(
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            mesh=mesh,
            use_bias=config.attention_bias,
            use_causal_mask=True,
            has_embedder=True,
            use_qwen2_layers=True,
            dtype=dtype,
        )

        self.patch_decoder = MiMoAudioTransformer(
            hidden_size=config.local_dim,
            num_layers=config.local_layers,
            num_heads=config.local_attn_heads,
            num_kv_heads=config.local_attn_heads,
            head_dim=config.local_head_dim,
            intermediate_size=config.local_ffn_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            mesh=mesh,
            use_bias=config.attention_bias,
            use_causal_mask=True,
            has_embedder=False,
            dtype=dtype,
        )

        self.patch_encoder = MiMoAudioTransformer(
            hidden_size=config.input_local_dim,
            num_layers=config.input_local_layers,
            num_heads=config.local_attn_heads,
            num_kv_heads=config.local_attn_heads,
            head_dim=config.local_head_dim,
            intermediate_size=config.local_ffn_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            mesh=mesh,
            use_bias=config.attention_bias,
            use_causal_mask=False,
            has_embedder=False,
            dtype=dtype,
        )

        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        self.patch_decoder_lm_heads = nnx.List(
            [
                LinearBase(
                    input_size=config.local_dim,
                    output_size=config.speech_vocab_sizes[i],
                    use_bias=False,
                    kernel_axes=(None, None),
                    params_dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.audio_channels)
            ]
        )

        self.speech_embeddings = nnx.List(
            [
                Embed(
                    num_embeddings=config.speech_vocab_sizes[i],
                    features=config.input_local_dim,
                    dtype=dtype,
                    kernel_axes=(None, None),
                    param_dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.audio_channels)
            ]
        )

        self.speech_group_downcast = LinearBase(
            input_size=config.input_local_dim * config.group_size,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.hidden_states_downcast = LinearBase(
            input_size=config.hidden_size,
            output_size=config.local_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(to_mappings(self.config))

    def apply_patch_encoder(self, speech_embeddings: jax.Array) -> jax.Array:
        B, T_groups, group_size, hidden_size = speech_embeddings.shape
        input_embeddings = speech_embeddings.reshape(B * T_groups, group_size, hidden_size)
        positions = jnp.arange(group_size)
        output, _, _ = self.patch_encoder(input_embeddings, positions)
        return output.reshape(B, T_groups, group_size, hidden_size)

    def _prepare_input_embeds(self, input_ids: jax.Array) -> jax.Array:
        B = input_ids.shape[0]
        text_input_ids = input_ids[:, 0, :: self.config.group_size]
        speech_input_ids = input_ids[:, 1:, :]
        speech_input_ids = speech_input_ids.reshape(
            B, self.config.audio_channels, -1, self.config.group_size
        ).transpose(0, 2, 1, 3)
        is_speech = text_input_ids == self.args.empty_idx
        T_groups = is_speech.shape[1]
        speech_embeds = jnp.zeros(
            (B, T_groups, self.config.group_size, self.config.input_local_dim),
            dtype=self.dtype,
        )

        for idx in range(self.config.audio_channels):
            cur_empty = self.config.speech_empty_ids[idx]
            cur_embed = self.speech_embeddings[idx]
            cur_speech_ids = speech_input_ids[:, :, idx, :]
            cur_speech_embeds = cur_embed(cur_speech_ids)
            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds = cur_speech_embeds * ~cur_mask[..., None]
            speech_embeds = speech_embeds + cur_speech_embeds

        speech_embeds = speech_embeds * is_speech[:, :, None, None]
        speech_embeds = self.apply_patch_encoder(speech_embeds)
        speech_embeds = speech_embeds * is_speech[:, :, None, None]
        speech_grouped_embeds, _ = self.speech_group_downcast(
            speech_embeds.reshape(B, T_groups, -1)
        )
        text_input_ids_safe = jnp.where(text_input_ids == -100, 0, text_input_ids)
        text_embeds = self.model.embed_tokens(text_input_ids_safe)
        text_zero_mask = (text_input_ids == self.args.empty_idx) | (text_input_ids == -100)
        text_embeds = text_embeds * ~text_zero_mask[..., None]
        result = text_embeds + speech_grouped_embeds
        return result

    def forward_simple(
        self, input_ids: jax.Array, positions: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        inputs_embeds = self._prepare_input_embeds(input_ids)
        B, T_groups, H = inputs_embeds.shape
        hidden_states, _, _ = self.model(inputs_embeds, positions, None, None)
        hidden_states_promoted, embedding = self.lm_head.promote_dtype(hidden_states)
        text_logits = jnp.matmul(hidden_states_promoted, embedding.T)
        local_hidden_states, _ = self.hidden_states_downcast(hidden_states[:, -1:, :])
        return text_logits, local_hidden_states

    def forward(
        self,
        input_ids: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ) -> tuple[jax.Array, jax.Array, None, list, list]:
        inputs_embeds = self._prepare_input_embeds(input_ids)
        B, T_groups, H = inputs_embeds.shape
        inputs_embeds_flat = inputs_embeds.reshape(-1, H)
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            inputs_embeds_flat, forward_batch.positions, forward_batch, token_to_kv_pool
        )
        text_logits = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        hidden_states = hidden_states.reshape(B, T_groups, H)
        local_hidden_states, _ = self.hidden_states_downcast(hidden_states[:, -1:, :])
        return text_logits, local_hidden_states, None, layers_kv_fused, layers_callback_flag

    def patch_decode(
        self,
        local_embeds: jax.Array,
        key: jax.Array,
        sampler_config: MiMoSamplerConfig | None = None,
    ) -> jax.Array:
        if sampler_config is None:
            sampler_config = MiMoSamplerConfig()

        B = local_embeds.shape[0]
        delay_iters = self.config.group_size + max(self.config.delay_pattern)
        local_tokens = jnp.zeros(
            (B, self.config.group_size, self.config.audio_channels), dtype=jnp.int32
        )

        for t in range(delay_iters):
            positions = jnp.array([t])
            hidden_state, _, _ = self.patch_decoder(local_embeds, positions)
            next_local_embeds = jnp.zeros_like(local_embeds)

            for idx in range(self.config.audio_channels):
                cur_start = self.config.delay_pattern[idx]
                cur_end = cur_start + self.config.group_size
                cur_empty = self.config.speech_empty_ids[idx]

                if cur_start <= t < cur_end:
                    cur_lm_head = self.patch_decoder_lm_heads[idx]
                    cur_logits, _ = cur_lm_head(hidden_state[:, -1, :])
                    cur_logits = cur_logits.at[:, cur_empty].set(-jnp.inf)
                    key, subkey = jax.random.split(key)
                    if sampler_config.do_sample:
                        cur_logits = cur_logits / sampler_config.temperature
                        cur_token = jax.random.categorical(subkey, cur_logits)
                    else:
                        cur_token = jnp.argmax(cur_logits, axis=-1)
                    local_tokens = local_tokens.at[:, t - cur_start, idx].set(cur_token)
                    cur_input_embed = self.speech_embeddings[idx](cur_token[:, None])
                    next_local_embeds = next_local_embeds + cur_input_embed

            local_embeds = next_local_embeds

        return local_tokens


EntryClass = MiMoAudioForCausalLM
