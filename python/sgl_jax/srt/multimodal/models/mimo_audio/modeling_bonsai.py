from typing import Optional, Tuple, List
import jax
import jax.numpy as jnp
from flax import nnx
from bonsai.models.qwen2.modeling import Qwen2, Cache
from bonsai.models.qwen3.modeling import shard
from bonsai.utils.samplers import Sampler, GreedySampler
from bonsai.models.mimo_audio.mimo_audio_configuration import MiMoAudioConfig, MiMoAudioArguments, MiMoSamplerConfig


class MiMoSampler:
    """Sampling utilities for generation"""

    def __init__(self, config: MiMoSamplerConfig):
        self.config = config
        if config.do_sample:
            self._sampler = Sampler(temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)
        else:
            self._sampler = GreedySampler()

    def sample(
        self, logits: jnp.ndarray, key: jax.random.PRNGKey, removed_tokens: Optional[List[int]] = None
    ) -> jnp.ndarray:
        if removed_tokens:
            for t in removed_tokens:
                logits = logits.at[:, t].set(-jnp.inf)

        result = self._sampler(logits, key=key)  # [B, 1]
        return result[:, 0]  # [B]


class FlaxMiMoAudioForCausalLM(nnx.Module):
    def __init__(
        self,
        config: MiMoAudioConfig,
        args: MiMoAudioArguments,
        rngs: Optional[nnx.Rngs] = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.config = config
        self.args = args
        self.dtype = dtype
        self.shd_cfg = config.shd_cfg

        # Fixed model-specific configurations
        self.speech_vocab_sizes = [1025, 1025, 129, 129, 129, 129, 129, 129]
        self.speech_empty_ids = [1024, 1024, 128, 128, 128, 128, 128, 128]
        self.delay_pattern = [0, 1, 2, 3, 4, 5, 6, 7]

        self.group_size = config.group_size
        self.audio_channels = config.audio_channels

        self.qwen2_config = config.create_qwen2_config()
        self.local_qwen2_config = config.create_local_qwen2_config()
        self.input_local_qwen2_config = config.create_input_local_qwen2_config()

        self.model = Qwen2(self.qwen2_config, rngs=rngs)
        self.local_transformer = Qwen2(self.local_qwen2_config, rngs=rngs)
        self.input_local_transformer = Qwen2(self.input_local_qwen2_config, rngs=rngs)

        self.local_transformer.embedder = None
        self.input_local_transformer.embedder = None

        self.lm_head = shard(
            nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, dtype=self.dtype, rngs=rngs),
            self.shd_cfg.emb_dv,
        )

        self.local_transformer_lm_heads = nnx.List(
            [
                shard(
                    nnx.Linear(
                        config.local_dim, self.speech_vocab_sizes[i], use_bias=False, dtype=self.dtype, rngs=rngs
                    ),
                    self.shd_cfg.emb_dv,
                )
                for i in range(self.audio_channels)
            ]
        )

        self.speech_embeddings = nnx.List(
            [
                shard(
                    nnx.Embed(self.speech_vocab_sizes[i], config.input_local_dim, dtype=self.dtype, rngs=rngs),
                    self.shd_cfg.emb_vd,
                )
                for i in range(self.audio_channels)
            ]
        )

        self.speech_group_downcast = shard(
            nnx.Linear(
                config.input_local_dim * config.group_size,
                config.hidden_size,
                use_bias=False,
                dtype=self.dtype,
                rngs=rngs,
            ),
            self.shd_cfg.ffw_weight_df,
        )

        self.hidden_states_downcast = shard(
            nnx.Linear(config.hidden_size, config.local_dim, use_bias=False, dtype=self.dtype, rngs=rngs),
            self.shd_cfg.ffw_weight_df,
        )

    def apply_input_local_transformer(
        self, speech_embeddings: jnp.ndarray, cache: Optional[Cache] = None
    ) -> jnp.ndarray:
        """Apply input local transformer to speech embeddings"""
        B, T_groups, group_size, hidden_size = speech_embeddings.shape

        input_embeddings = speech_embeddings.reshape(B * T_groups, group_size, hidden_size)
        segment_ids = jnp.ones((B * T_groups, group_size), dtype=jnp.int32)

        if cache is None:
            cache = self.input_local_transformer.init_cache(
                self.input_local_qwen2_config, B * T_groups, group_size, generate_steps=0, dtype=self.dtype
            )

        x = input_embeddings
        for i, layer in enumerate(self.input_local_transformer.layers):
            x = layer(x, cache[i], segment_ids)
        x = self.input_local_transformer.final_norm(x)

        return x.reshape(B, T_groups, group_size, hidden_size)

    def _prepare_input_embeds(self, input_ids: jnp.ndarray, text_embed_fn) -> jnp.ndarray:
        """Prepare input embeddings from interleaved text and speech tokens"""
        B = input_ids.shape[0]

        text_input_ids = input_ids[:, 0, :: self.group_size]
        speech_input_ids = (
            input_ids[:, 1:, :].reshape(B, self.audio_channels, -1, self.group_size).transpose(0, 2, 1, 3)
        )

        is_speech = text_input_ids == self.args.empty_idx

        speech_embeds = jnp.zeros(
            (B, is_speech.shape[1], self.group_size, self.config.input_local_dim), dtype=self.dtype
        )

        for idx in range(self.audio_channels):
            cur_empty = self.speech_empty_ids[idx]
            cur_embed = self.speech_embeddings[idx]
            cur_speech_ids = speech_input_ids[:, :, idx, :]
            cur_speech_embeds = cur_embed(cur_speech_ids)

            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds = cur_speech_embeds * ~cur_mask[..., None]
            speech_embeds = speech_embeds + cur_speech_embeds

        speech_embeds = speech_embeds * is_speech[:, :, None, None]
        speech_embeds = self.apply_input_local_transformer(speech_embeds, cache=None)

        speech_embeds = speech_embeds * is_speech[:, :, None, None]

        T_groups = speech_embeds.shape[1]
        speech_grouped_embeds = self.speech_group_downcast(speech_embeds.reshape(B, T_groups, -1))

        text_input_ids_safe = jnp.where(text_input_ids == -100, 0, text_input_ids)
        text_embeds = text_embed_fn(text_input_ids_safe)

        text_zero_mask = (text_input_ids == self.args.empty_idx) | (text_input_ids == -100)
        text_embeds = text_embeds * ~text_zero_mask[..., None]

        output = text_embeds + speech_grouped_embeds
        return shard(output, self.shd_cfg.act_btd)

    def forward(
        self,
        input_ids: jnp.ndarray,
        cache: Cache,
        pad_id: int = 0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Cache]:
        text_input_ids = input_ids[:, 0, :: self.group_size]

        def text_embed_fn(x):
            return self.model.embedder.embedding.value[x]

        inputs_embeds = self._prepare_input_embeds(input_ids, text_embed_fn)

        B, T_groups, _ = inputs_embeds.shape
        segment_ids = 1 * (text_input_ids != -100)

        x = inputs_embeds
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cache[i], segment_ids)
        hidden_states = self.model.final_norm(x)

        text_logits = self.lm_head(hidden_states[:, -1:, :])  # [B, 1, vocab_size]

        # Downcast hidden states for local transformer
        local_hidden_states = self.hidden_states_downcast(hidden_states[:, -1:, :])  # [B, 1, local_dim]

        return text_logits, local_hidden_states, cache

    def local_forward(
        self,
        local_embeds: jnp.ndarray,
        key: jax.random.PRNGKey,
        local_sampler: Optional[MiMoSampler] = None,
    ) -> jnp.ndarray:
        """
        Generate audio tokens for one group using local transformer.

        Args:
            local_embeds: [B, 1, local_dim]
            key: Random key for sampling
            local_sampler: Sampler configuration

        Returns:
            local_tokens: [B, group_size, audio_channels]
        """
        B = local_embeds.shape[0]
        delay_iters = self.group_size + max(self.delay_pattern)

        local_tokens = jnp.zeros((B, self.group_size, self.audio_channels), dtype=jnp.int32)

        if local_sampler is None:
            local_sampler = MiMoSampler(MiMoSamplerConfig())

        cache = self.local_transformer.init_cache(
            self.local_qwen2_config,
            B,
            token_len=1,
            generate_steps=delay_iters - 1,
            dtype=self.dtype,
        )

        segment_ids = jnp.ones((B, 1), dtype=jnp.int32)

        for t in range(delay_iters):
            hidden_state, cache = _local_transformer_step_jit(self.local_transformer, local_embeds, cache, segment_ids)

            next_local_embeds = jnp.zeros_like(local_embeds)

            for idx in range(self.audio_channels):
                cur_start = self.delay_pattern[idx]
                cur_end = cur_start + self.group_size
                cur_empty = self.speech_empty_ids[idx]

                if cur_start <= t < cur_end:
                    cur_lm_head = self.local_transformer_lm_heads[idx]
                    cur_logits = cur_lm_head(hidden_state[:, -1, :])

                    key, subkey = jax.random.split(key)
                    cur_token = local_sampler.sample(cur_logits, subkey, removed_tokens=[cur_empty])

                    local_tokens = local_tokens.at[:, t - cur_start, idx].set(cur_token)

                    cur_input_embed = self.speech_embeddings[idx](cur_token[:, None])

                    next_local_embeds = next_local_embeds + cur_input_embed

            local_embeds = next_local_embeds

        return local_tokens


@jax.jit
def _local_transformer_step_jit(
    local_transformer: nnx.Module,
    local_embeds: jnp.ndarray,
    cache: Cache,
    segment_ids: jnp.ndarray,
) -> Tuple[jnp.ndarray, Cache]:
    x = local_embeds
    for i, layer in enumerate(local_transformer.layers):
        x = layer(x, cache[i], segment_ids)
    hidden_state = local_transformer.final_norm(x)
    return hidden_state, cache


@jax.jit
def forward_jit(
    model: FlaxMiMoAudioForCausalLM,
    input_ids: jnp.ndarray,
    cache: Cache,
    pad_id: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Cache]:
    text_logits, local_hidden_states, cache = model.forward(input_ids, cache, pad_id)
    return text_logits, local_hidden_states, cache


@jax.jit
def local_forward_jit(
    model: FlaxMiMoAudioForCausalLM,
    local_embeds: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    return model.local_forward(local_embeds, key, local_sampler=None)
