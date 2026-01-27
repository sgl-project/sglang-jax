import math
from typing import Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import get_abstract_mesh, reshard

from bonsai.models.mimo_audio.mimo_audio_tokenizer_configuration import (
    MiMoShardingCfg,
    MiMoAudioTokenizerConfig,
    EncoderOutput,
    VocoderOutput,
)

Array = jnp.ndarray


def shard(x: Array, s) -> Array:
    """Apply sharding to array if mesh is available."""
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        return reshard(x, s)
    return x


def make_sequence_mask(lengths: Array, max_length: Optional[int] = None) -> Array:
    max_len = max_length or int(jnp.max(lengths))
    base = jnp.arange(max_len)[None, :]
    return base < lengths[:, None]


def get_position_ids(lengths: Array, max_length: Optional[int] = None) -> Array:
    max_len = max_length or int(jnp.max(lengths))
    base = jnp.arange(max_len)[None, :]
    return jnp.broadcast_to(base, (lengths.shape[0], max_len))


def rotate_half(x: Array) -> Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary(x: Array, cos: Array, sin: Array) -> Array:
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


class MelSpectrogram:
    def __init__(
            self,
            sample_rate: int,
            n_fft: int,
            hop_length: int,
            win_length: int,
            f_min: float,
            f_max: float,
            n_mels: int,
            power: float = 1.0,
            center: bool = True,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.n_mels = int(n_mels)
        self.power = float(power)
        self.center = center

        self._window = self._build_window()
        self._mel_filterbank = self._build_mel_filterbank()

    def _build_window(self) -> jnp.ndarray:
        if self.win_length <= 1:
            return jnp.ones((self.win_length,), dtype=jnp.float32)
        n = jnp.arange(self.win_length, dtype=jnp.float32)
        return 0.5 - 0.5 * jnp.cos(2 * jnp.pi * n / (self.win_length - 1))

    def _hz_to_mel(self, freq: jnp.ndarray) -> jnp.ndarray:
        return 2595.0 * jnp.log10(1.0 + freq / 700.0)

    def _mel_to_hz(self, mel: jnp.ndarray) -> jnp.ndarray:
        return 700.0 * (jnp.power(10.0, mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> jnp.ndarray:
        freq_bins = jnp.linspace(
            0.0,
            self.sample_rate / 2,
            self.n_fft // 2 + 1,
            dtype=jnp.float32,
            )
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = jnp.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        filterbanks = []
        for i in range(self.n_mels):
            lower = hz_points[i]
            center = hz_points[i + 1]
            upper = hz_points[i + 2]
            denom_left = jnp.maximum(center - lower, 1e-10)
            denom_right = jnp.maximum(upper - center, 1e-10)
            left_slope = (freq_bins - lower) / denom_left
            right_slope = (upper - freq_bins) / denom_right
            filterbanks.append(jnp.maximum(0.0, jnp.minimum(left_slope, right_slope)))

        return jnp.stack(filterbanks, axis=0)

    def _frame_signal(self, waveform: jnp.ndarray) -> jnp.ndarray:
        frame_length = self.n_fft

        if self.center:
            pad = self.n_fft // 2
            if waveform.shape[0] > 1:
                waveform = jnp.pad(waveform, (pad, pad), mode="reflect")
            else:
                waveform = jnp.pad(waveform, (pad, pad))

        total_length = int(waveform.shape[0])
        if total_length < frame_length:
            pad_amount = frame_length - total_length
            waveform = jnp.pad(waveform, (0, pad_amount))
            total_length = int(waveform.shape[0])

        num_frames = 1 + max(0, (total_length - frame_length) // self.hop_length)
        if num_frames <= 0:
            num_frames = 1

        starts = [idx * self.hop_length for idx in range(num_frames)]
        frames = jnp.stack([waveform[start : start + frame_length] for start in starts], axis=0)
        return frames

    def _mel_spectrogram(self, waveform: jnp.ndarray) -> jnp.ndarray:
        waveform = jnp.asarray(waveform, dtype=jnp.float32)
        frames = self._frame_signal(waveform)
        if self.win_length < self.n_fft:
            total_pad = self.n_fft - self.win_length
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            window = jnp.pad(self._window, (pad_left, pad_right))
        else:
            window = self._window[: self.n_fft]
        windowed = frames * window
        stft = jnp.fft.rfft(windowed, n=self.n_fft, axis=1)
        magnitude = jnp.abs(stft) ** self.power
        mel_spec = magnitude @ self._mel_filterbank.T
        return mel_spec.T

    def __call__(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """Compute mel spectrogram from waveform.

        Args:
            waveform: JAX array of shape (samples,) or (batch, samples)

        Returns:
            Mel spectrogram of shape (n_mels, time) or (batch, n_mels, time)
        """
        waveform = jnp.asarray(waveform, dtype=jnp.float32)

        squeeze_dim = False
        if waveform.ndim == 1:
            waveform = waveform[jnp.newaxis, :]
            squeeze_dim = True

        mel_outputs = []
        for sample in waveform:
            mel_outputs.append(self._mel_spectrogram(sample))

        mel_stack = jnp.stack(mel_outputs, axis=0)
        if squeeze_dim:
            mel_stack = jnp.squeeze(mel_stack, axis=0)
        return mel_stack


class RotaryEmbedding(nnx.Module):
    def __init__(self, base: float, dim: int, max_seq_len: int, rope_type: str = "default", dtype=jnp.float32):
        self.base = base
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_type = rope_type
        self.dtype = dtype
        half_dim = dim // 2
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / float(half_dim)))
        self.inv_freq = nnx.Param(inv_freq)
        self.attention_scaling = 1.0

    def __call__(self, hidden_states: Array, position_ids: Array) -> Tuple[Array, Array]:
        freq = position_ids[..., None] * self.inv_freq[None, None, :]
        emb = jnp.concatenate([freq, freq], axis=-1)
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling
        return cos.astype(hidden_states.dtype), sin.astype(hidden_states.dtype)


class ConvTranspose1d(nnx.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            shd_cfg: MiMoShardingCfg | None = None,
            dtype=jnp.float32,
            rngs: Optional[nnx.Rngs] = None,
    ):
        self.stride = stride
        self.shd_cfg = shd_cfg or MiMoShardingCfg.no_sharding()

        kshape = (in_channels, out_channels, kernel_size)
        kernel = jnp.zeros(kshape, dtype=dtype)
        self.kernel = shard(nnx.Param(kernel), self.shd_cfg.conv_transpose_weight)

        bias = jnp.zeros((out_channels,), dtype=dtype)
        self.bias = shard(nnx.Param(bias), self.shd_cfg.conv_transpose_bias)

    def __call__(self, x: Array) -> Array:
        batch, length, channels = x.shape
        kernel = self.kernel.value
        kernel_size = kernel.shape[-1]
        up_len = (length - 1) * self.stride + 1
        idx = jnp.arange(length) * self.stride
        upsampled = jnp.zeros((batch, up_len, channels), dtype=x.dtype)
        upsampled = upsampled.at[:, idx, :].set(x)
        upsampled = jnp.pad(upsampled, ((0, 0), (kernel_size - 1, kernel_size - 1), (0, 0)))
        lhs = jnp.swapaxes(upsampled, 1, 2)
        rhs = jnp.flip(kernel, axis=-1).transpose(1, 0, 2)
        y = jax.lax.conv_general_dilated(
            lhs=lhs,
            rhs=rhs,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        y = y + self.bias.value[None, :, None]
        y = jnp.swapaxes(y, 1, 2)
        return y


class ISTFT(nnx.Module):
    def __init__(
            self,
            n_fft: int,
            hop_length: int,
            win_length: int,
            padding: str = "same",
            shd_cfg: MiMoShardingCfg | None = None,
            dtype=jnp.float32,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.padding = padding
        self.shd_cfg = shd_cfg or MiMoShardingCfg.no_sharding()

        self.window = shard(nnx.Param(jnp.hanning(win_length).astype(dtype)), self.shd_cfg.istft_window)

        self.pad = (self.win_length - self.hop_length) // 2 if padding == "same" else 0

    def __call__(self, spec: Array) -> Array:
        frames = jnp.fft.irfft(spec, n=self.n_fft, axis=1, norm="backward")
        frames = frames * self.window[None, :, None]
        frames = jnp.swapaxes(frames, 1, 2)
        batch, num_frames, _ = frames.shape
        output_size = (num_frames - 1) * self.hop_length + self.win_length
        audio = jnp.zeros((batch, output_size), dtype=frames.dtype)
        env = jnp.zeros_like(audio)
        window_sq = jnp.square(self.window)

        def body(i, carry):
            audio_acc, env_acc = carry
            start = i * self.hop_length
            frame = frames[:, i, :]
            current_audio = jax.lax.dynamic_slice(
                audio_acc,
                (0, start),
                (batch, self.win_length),
            )
            current_env = jax.lax.dynamic_slice(
                env_acc,
                (0, start),
                (batch, self.win_length),
            )
            updated_audio = current_audio + frame
            updated_env = current_env + window_sq
            audio_acc = jax.lax.dynamic_update_slice(audio_acc, updated_audio, (0, start))
            env_acc = jax.lax.dynamic_update_slice(env_acc, updated_env, (0, start))
            return audio_acc, env_acc

        audio, env = jax.lax.fori_loop(0, num_frames, body, (audio, env))
        if self.pad > 0:
            audio = audio[:, self.pad : -self.pad]
            env = env[:, self.pad : -self.pad]
        env = jnp.maximum(env, 1e-11)
        audio = audio / env
        return audio


class ISTFTHead(nnx.Module):
    def __init__(
            self,
            dim: int,
            n_fft: int,
            hop_length: int,
            padding: str = "same",
            shd_cfg: MiMoShardingCfg | None = None,
            dtype=jnp.float32,
            rngs: Optional[nnx.Rngs] = None,
    ):
        self.shd_cfg = shd_cfg or MiMoShardingCfg.no_sharding()

        self.linear = shard(nnx.Linear(dim, n_fft + 2, dtype=dtype, rngs=rngs), self.shd_cfg.istft_linear_weight)

        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding, shd_cfg=self.shd_cfg, dtype=dtype
        )

    def __call__(self, hidden_states: Array) -> Array:
        x = self.linear(hidden_states)
        x = jnp.swapaxes(x, 1, 2)
        mag, phase = jnp.split(x, 2, axis=1)

        original_dtype = hidden_states.dtype
        mag = mag.astype(jnp.float32)
        phase = phase.astype(jnp.float32)

        mag = jnp.clip(jnp.exp(mag), a_max=1e2)
        real = jnp.cos(phase)
        imag = jnp.sin(phase)
        spec = mag * (real + 1j * imag)

        audio = self.istft(spec)
        audio = audio.astype(original_dtype)
        return audio


class Attention(nnx.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            window_size: Tuple[int, int],
            causal: bool,
            shd_cfg: MiMoShardingCfg,
            dtype=jnp.float32,
            rngs: Optional[nnx.Rngs] = None,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.window_size = window_size
        self.causal = causal
        self.shd_cfg = shd_cfg

        self.q_proj = shard(
            nnx.Linear(embed_dim, embed_dim, use_bias=True, dtype=dtype, rngs=rngs), shd_cfg.attn_qkvo_weight
        )
        self.k_proj = shard(
            nnx.Linear(embed_dim, embed_dim, use_bias=False, dtype=dtype, rngs=rngs), shd_cfg.attn_qkvo_weight
        )
        self.v_proj = shard(
            nnx.Linear(embed_dim, embed_dim, use_bias=True, dtype=dtype, rngs=rngs), shd_cfg.attn_qkvo_weight
        )
        self.out_proj = shard(nnx.Linear(embed_dim, embed_dim, dtype=dtype, rngs=rngs), shd_cfg.attn_qkvo_weight)

    def _window_mask(self, seq_len: int) -> Optional[Array]:
        left, right = self.window_size
        if left < 0 and right < 0:
            return None
        pos = jnp.arange(seq_len)
        rel = pos[None, :] - pos[:, None]
        mask = jnp.ones((seq_len, seq_len), dtype=bool)
        if left >= 0:
            mask &= rel >= -left
        if right >= 0:
            mask &= rel <= right
        return mask

    def __call__(self, x: Array, mask: Optional[Array], rope: Optional[Tuple[Array, Array]]) -> Array:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def reshape(t):
            t = t.reshape(batch, seq_len, self.num_heads, self.head_dim)
            return jnp.swapaxes(t, 1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        q = shard(q, self.shd_cfg.act_btnh)
        k = shard(k, self.shd_cfg.act_btnh)
        v = shard(v, self.shd_cfg.act_btnh)

        if rope is not None:
            cos, sin = rope
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        if mask is not None:
            scores = jnp.where(mask[:, None, None, :], scores, -1e9)
        if self.causal:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
            scores = jnp.where(causal_mask, scores, -1e9)
        wmask = self._window_mask(seq_len)
        if wmask is not None:
            scores = jnp.where(wmask, scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        context = jnp.swapaxes(context, 1, 2).reshape(batch, seq_len, self.embed_dim)
        out = self.out_proj(context)
        if mask is not None:
            out = out * mask[..., None]

        out = shard(out, self.shd_cfg.act_btd)
        return out


class TransformerLayer(nnx.Module):
    def __init__(
            self,
            d_model: int,
            attention_heads: int,
            ffn_dim: int,
            causal: bool,
            attn_window_size: Tuple[int, int],
            shd_cfg: MiMoShardingCfg,
            dtype=jnp.float32,
            rngs: Optional[nnx.Rngs] = None,
    ):
        self.act = jax.nn.gelu
        self.shd_cfg = shd_cfg

        self.self_attn = Attention(d_model, attention_heads, attn_window_size, causal, shd_cfg, dtype=dtype, rngs=rngs)

        self.self_attn_layer_norm = shard(
            nnx.LayerNorm(d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs), shd_cfg.norm_scale
        )
        self.final_layer_norm = shard(
            nnx.LayerNorm(d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs), shd_cfg.norm_scale
        )

        self.fc1 = shard(nnx.Linear(d_model, ffn_dim, dtype=dtype, rngs=rngs), shd_cfg.ffn_weight_in)
        self.fc2 = shard(nnx.Linear(ffn_dim, d_model, dtype=dtype, rngs=rngs), shd_cfg.ffn_weight_out)

    def __call__(self, hidden_states: Array, mask: Optional[Array], rope: Optional[Tuple[Array, Array]]) -> Array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask, rope)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = shard(hidden_states, self.shd_cfg.act_btd)
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class ResidualVectorQuantizer(nnx.Module):
    def __init__(
            self,
            dimension: int,
            n_q: int,
            bins: Sequence[int],
            shd_cfg: MiMoShardingCfg,
            dtype=jnp.float32,
            rngs: Optional[nnx.Rngs] = None,
    ):
        self.dimension = dimension
        self.n_q = n_q
        self.shd_cfg = shd_cfg

        codebooks_list = []
        for i in range(n_q):
            size = bins[min(i, len(bins) - 1)]
            embed = jnp.zeros((size, dimension), dtype=dtype)
            codebooks_list.append(shard(nnx.Param(embed), shd_cfg.codebook))
        self.codebooks = nnx.List(codebooks_list)

    def encode(
            self, hidden_states: Array, mask: Optional[Array] = None, n_q: Optional[int] = None
    ) -> Tuple[Array, Array]:
        num_levels = n_q or self.n_q
        residual = hidden_states
        quantized = jnp.zeros_like(hidden_states)
        codes = []
        mask = None if mask is None else mask[..., None]
        for i in range(num_levels):
            codebook = self.codebooks[i].value
            dist = jnp.sum((residual[:, None, :] - codebook[None, :, :]) ** 2, axis=-1)
            idx = jnp.argmin(dist, axis=-1)
            chosen = codebook[idx]
            if mask is not None:
                chosen = chosen * mask
            quantized = quantized + chosen
            residual = residual - chosen
            codes.append(idx)
        return jnp.stack(codes, axis=0), quantized

    def decode(self, codes: Array) -> Array:
        num_levels = codes.shape[0]
        flat = codes.reshape(num_levels, -1)
        decoded = jnp.zeros((flat.shape[1], self.dimension), dtype=jnp.float32)
        for i in range(num_levels):
            codebook = self.codebooks[i].value
            decoded = decoded + codebook[flat[i]]
        return decoded.reshape(*codes.shape[1:], self.dimension)


class AudioEncoder(nnx.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig, dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.config = config
        self.shd_cfg = config.shd_cfg

        self.conv1 = shard(
            nnx.Conv(
                in_features=config.n_mels,
                out_features=config.d_model,
                kernel_size=config.kernel_size,
                padding=1,
                param_dtype=dtype,
                rngs=rngs,
            ),
            self.shd_cfg.conv_weight,
        )
        self.conv2 = shard(
            nnx.Conv(
                in_features=config.d_model,
                out_features=config.d_model,
                kernel_size=config.kernel_size,
                strides=config.stride_size,
                padding=1,
                param_dtype=dtype,
                rngs=rngs,
            ),
            self.shd_cfg.conv_weight,
        )

        self.position_embedding = RotaryEmbedding(
            config.rope_theta,
            config.d_model // config.encoder_attention_heads,
            config.max_audio_seconds * config.sampling_rate // config.hop_length,
            config.rope_type,
            dtype=dtype,
            )

        self.layers = nnx.List(
            [
                TransformerLayer(
                    config.d_model,
                    config.encoder_attention_heads,
                    config.encoder_ffn_dim,
                    config.encoder_causal,
                    tuple(config.encoder_attn_window_size),
                    self.shd_cfg,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(config.encoder_layers)
            ]
        )

        self.layer_norm = shard(
            nnx.LayerNorm(config.d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs), self.shd_cfg.norm_scale
        )

        if config.avg_pooler != 1:
            self.down_sample_layer = shard(
                nnx.Conv(
                    in_features=config.d_model,
                    out_features=config.d_model,
                    kernel_size=config.avg_pooler,
                    strides=config.avg_pooler,
                    padding="SAME",
                    use_bias=False,
                    param_dtype=dtype,
                    rngs=rngs,
                ),
                self.shd_cfg.conv_weight,
            )
            self.down_norm = shard(
                nnx.LayerNorm(config.d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs), self.shd_cfg.norm_scale
            )
        else:
            self.down_sample_layer = None
            self.down_norm = None

        if config.num_quantizers:
            bins = config.codebook_size or [1024]
            self.quantizer = ResidualVectorQuantizer(
                config.d_model, config.num_quantizers, bins, self.shd_cfg, dtype=dtype, rngs=rngs
            )
        else:
            self.quantizer = None

    def get_output_length(self, mel_len: Array) -> Array:
        tgt = mel_len + 3 - self.config.kernel_size
        return (tgt + 2 - self.config.kernel_size) // self.config.stride_size + 1

    def __call__(
            self, input_features: Array, input_lens: Array, use_quantizer: bool = True, n_q: Optional[int] = None
    ) -> EncoderOutput:
        x = input_features
        x = jax.nn.gelu(self.conv1(x))
        x = shard(x, self.shd_cfg.act_btd)

        x = jax.nn.gelu(self.conv2(x))
        x = shard(x, self.shd_cfg.act_btd)

        lengths = self.get_output_length(input_lens)
        max_len = x.shape[1]
        mask = make_sequence_mask(lengths, max_len)
        pos = get_position_ids(lengths, max_len)
        rope = self.position_embedding(x, pos)
        skip = 0.0
        for idx, layer in enumerate(self.layers):
            x = layer(x, mask, rope)
            if self.config.encoder_skip_layer_id and idx == self.config.encoder_skip_layer_id - 1:
                skip = x
        x = x + skip
        x = self.layer_norm(x)
        if self.down_sample_layer is not None:
            x = jax.nn.gelu(self.down_sample_layer(x))
            x = shard(x, self.shd_cfg.act_btd)

            lengths = (lengths // self.config.avg_pooler) + ((lengths % self.config.avg_pooler) != 0).astype(
                lengths.dtype
            )
            max_len = x.shape[1]
            mask = make_sequence_mask(lengths, max_len)
            x = self.down_norm(x)
        x = x * mask[..., None]
        packed = x.reshape(-1, self.config.d_model)
        mask_flat = mask.reshape(-1)
        codes = None
        if self.quantizer is not None and use_quantizer:
            codes, quantized = self.quantizer.encode(packed, mask=mask_flat, n_q=n_q)
            packed = quantized
        packed = packed.reshape(x.shape)
        return EncoderOutput(hidden_states=packed, packed_states=packed, output_lengths=lengths, codes=codes)

    def decode_vq(self, codes: Array) -> Array:
        if self.quantizer is None:
            raise ValueError("Quantizer disabled")
        return self.quantizer.decode(codes)


class CausalConvTranspose1d(nnx.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            shd_cfg: MiMoShardingCfg | None = None,
            dtype=jnp.float32,
            rngs: Optional[nnx.Rngs] = None,
    ):
        self.shd_cfg = shd_cfg or MiMoShardingCfg.no_sharding()

        self.conv = ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, shd_cfg=self.shd_cfg, dtype=dtype, rngs=rngs
        )

        self.norm = shard(
            nnx.GroupNorm(num_features=out_channels, num_groups=1, epsilon=1e-5, param_dtype=dtype, rngs=rngs),
            self.shd_cfg.norm_scale,
        )

        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x: Array, input_length: Array) -> Tuple[Array, Array]:
        y = self.conv(x)
        y = self.norm(y)
        trim = max(0, self.kernel_size - self.stride)
        if trim > 0:
            y = y[:, :-trim, :]
        output_len = (input_length - 1) * self.stride + self.kernel_size - trim
        return y, output_len


class TransformerVocos(nnx.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig, dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.config = config
        self.shd_cfg = config.shd_cfg

        self.embeddings = shard(
            nnx.Linear(config.n_mels, config.vocoder_dim, use_bias=False, dtype=dtype, rngs=rngs),
            self.shd_cfg.attn_qkvo_weight,
        )

        self.position_embedding = RotaryEmbedding(
            config.rope_theta,
            config.vocoder_dim // config.vocoder_attention_heads,
            config.max_audio_seconds * config.sampling_rate // config.hop_length,
            config.rope_type,
            dtype=dtype,
            )

        self.layers = nnx.List(
            [
                TransformerLayer(
                    config.vocoder_dim,
                    config.vocoder_attention_heads,
                    config.vocoder_intermediate_dim,
                    False,
                    tuple(config.vocoder_attn_window_size),
                    self.shd_cfg,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(config.vocoder_num_layers)
            ]
        )

        self.layer_norm = shard(
            nnx.LayerNorm(config.vocoder_dim, epsilon=1e-6, param_dtype=dtype, rngs=rngs), self.shd_cfg.norm_scale
        )

        self.head = ISTFTHead(
            config.vocoder_dim,
            config.nfft,
            config.hop_length,
            config.vocoder_padding,
            shd_cfg=self.shd_cfg,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, mels: Array, input_length: Array) -> VocoderOutput:
        x = self.embeddings(mels)
        mask = make_sequence_mask(input_length, x.shape[1])
        pos = get_position_ids(input_length, x.shape[1])
        rope = self.position_embedding(x, pos)
        for layer in self.layers:
            x = layer(x, mask, rope)
        x = self.layer_norm(x)
        x = x * mask[..., None]
        wav = self.head(x)
        wav_len = input_length * self.config.hop_length
        wav = wav[:, None, :]
        return VocoderOutput(wav=wav, wav_lengths=wav_len)


class AudioDecoder(nnx.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig, dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.config = config
        self.shd_cfg = config.shd_cfg

        if config.avg_pooler != 1:
            self.dconv1 = CausalConvTranspose1d(
                config.d_model,
                config.d_model,
                config.avg_pooler,
                config.avg_pooler,
                shd_cfg=self.shd_cfg,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.dconv1 = None

        self.position_embedding = RotaryEmbedding(
            config.rope_theta,
            config.d_model // config.decoder_attention_heads,
            config.max_audio_seconds * config.sampling_rate // config.hop_length,
            config.rope_type,
            dtype=dtype,
            )

        self.layers = nnx.List(
            [
                TransformerLayer(
                    config.d_model,
                    config.decoder_attention_heads,
                    config.decoder_ffn_dim,
                    config.decoder_causal,
                    tuple(config.decoder_attn_window_size),
                    self.shd_cfg,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(config.decoder_layers)
            ]
        )

        self.layer_norm = shard(
            nnx.LayerNorm(config.d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs), self.shd_cfg.norm_scale
        )

        self.dconv2 = CausalConvTranspose1d(
            config.d_model,
            config.n_mels,
            config.decoder_kernel_size,
            config.decoder_stride_size,
            shd_cfg=self.shd_cfg,
            dtype=dtype,
            rngs=rngs,
        )

        self.vocoder = TransformerVocos(config, dtype=dtype, rngs=rngs)

    def __call__(self, audio_embed: Array, input_length: Array) -> Array:
        x = audio_embed
        lengths = input_length
        if self.dconv1 is not None:
            x, lengths = self.dconv1(x, lengths)
        mask = make_sequence_mask(lengths, x.shape[1])
        pos = get_position_ids(lengths, x.shape[1])
        rope = self.position_embedding(x, pos)
        for layer in self.layers:
            x = layer(x, mask, rope)
        x = self.layer_norm(x)
        coarse, mel_lengths = self.dconv2(x, lengths)
        vocoder_out = self.vocoder(coarse, mel_lengths)
        return vocoder_out.wav


class FlaxMiMoAudioTokenizer(nnx.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig, dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.config = config
        self.encoder = AudioEncoder(config, dtype=dtype, rngs=rngs)
        self.decoder = AudioDecoder(config, dtype=dtype, rngs=rngs)
        self.downsample_rate = int(config.hop_length * 2 * config.avg_pooler)

    def __call__(self, mels: Array, input_lens: Array, use_quantizer: bool = True) -> Array:
        enc = self.encoder(mels, input_lens, use_quantizer=use_quantizer)
        return self.decoder(enc.hidden_states, enc.output_lengths)

    def encode(
            self, mels: Array, input_lens: Array, use_quantizer: bool = True, n_q: Optional[int] = None
    ) -> EncoderOutput:
        return self.encoder(mels, input_lens, use_quantizer=use_quantizer, n_q=n_q)

    def decode(self, codes: Array) -> Array:
        hidden = self.encoder.decode_vq(codes)
        hidden = hidden[None, ...]
        lengths = jnp.array([hidden.shape[1]])
        return self.decoder(hidden, lengths)
