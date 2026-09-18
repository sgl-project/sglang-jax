"""TT recurrent attention: explicit kernels with scheduler-owned state slots."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from sgl_jax.srt.hardware_backend.tt.attention import ops
from sgl_jax.srt.hardware_backend.tt.attention.tt_backend import TTAttention
from sgl_jax.srt.kernels.gdn.gated_delta import _l2norm
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend


@jax.tree_util.register_pytree_node_class
@dataclass
class TTGDNMetadata(LinearRecurrentAttnBackendMetadata):
    max_prefill_len: int = 0

    def tree_flatten(self):
        children, _ = super().tree_flatten()
        return children, self.max_prefill_len

    @classmethod
    def tree_unflatten(cls, max_prefill_len, children):
        meta = super().tree_unflatten({}, children)
        meta.max_prefill_len = max_prefill_len
        return meta


class TTGDNAttnBackend(GDNAttnBackend):
    # Quantize only annotated matrix weights; recurrent arithmetic stays FP32.
    compiler_options = {
        key: value
        for key, value in TTAttention.compiler_options.items()
        if key != "experimental_weight_dtype"
    }
    # Materialize weight transposes once instead of strided DRAM reads on decode.
    compiler_options["experimental_enable_permute_matmul_fusion"] = "false"

    def __init__(self, **kwargs):
        kwargs["prefill_impl"] = "chunked_jax"
        super().__init__(**kwargs)
        if self.mesh.size != 1:
            raise NotImplementedError("TT GDN currently supports one device")
        if (self.head_k_dim, self.head_v_dim, self.conv_kernel_size) != (128, 128, 4):
            raise NotImplementedError("TT GDN requires 128-wide heads and a four-tap convolution")

    def get_forward_metadata(self, batch):
        meta = super().get_forward_metadata(batch)
        if batch.forward_mode.is_extend():
            # Prefill packs each live sequence; exclude the scheduler's dummy rows.
            size = batch.real_bs
            meta.cu_q_lens = meta.cu_q_lens[: size + 1]
            # The native kernel needs dense sequences. Bucket the longest one
            # instead of padding every sequence to the entire batch length.
            length = int(batch.extend_seq_lens[:size].max())
            meta = TTGDNMetadata(
                **vars(meta), max_prefill_len=1 << (max(length, 32) - 1).bit_length()
            )
        return meta

    def _metadata(self):
        meta = self.forward_metadata
        if meta.recurrent_track_indices is not None:
            raise NotImplementedError("TT GDN recurrent snapshots are not yet supported")
        return meta, meta.recurrent_indices, meta.has_initial_state

    def _qkv(self, mixed):
        count = mixed.shape[0]
        # Q and K use the same normalization and head expansion. Process them
        # together to avoid launching the identical operation chain twice.
        qk = mixed[:, : 2 * self.key_dim].reshape(count, 2 * self.num_k_heads, self.head_k_dim)
        v = mixed[:, 2 * self.key_dim :].reshape(count, self.num_v_heads, self.head_v_dim)
        repeats = self.num_v_heads // self.num_k_heads
        sharding = jax.sharding.NamedSharding(self.mesh, jax.typeof(qk).sharding.spec)
        qk = jnp.repeat(_l2norm(qk.astype(jnp.float32)), repeats, axis=1, out_sharding=sharding)
        q, k = qk[:, : self.num_v_heads], qk[:, self.num_v_heads :]
        return q * self.head_k_dim**-0.5, k, v.astype(jnp.float32)

    def forward_decode(
        self, mixed_qkv, conv_state_in, recurrent_state_in, b, a, conv1d_weight, A_log, dt_bias
    ):
        _, indices, initial = self._metadata()
        new_conv, conv_out = ops.causal_conv1d_update(
            conv_state_in, mixed_qkv, conv1d_weight, indices, initial
        )
        q, k, v = self._qkv(conv_out)
        new_rec, out = ops.gated_delta_decode(
            recurrent_state_in, q, k, v, b, a, A_log, dt_bias, indices, initial
        )
        return out.astype(mixed_qkv.dtype), new_conv, new_rec

    def forward_extend(
        self,
        mixed_qkv,
        conv_state_in,
        recurrent_state_in,
        b,
        a,
        conv1d_weight,
        A_log,
        dt_bias,
        seq_lens,
    ):
        del seq_lens
        meta, indices, initial = self._metadata()
        count = mixed_qkv.shape[0]
        batch = meta.cu_q_lens.shape[0] - 1
        indices, initial = indices[:batch], initial[:batch]
        replicated = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
        cu_q_lens = jax.sharding.reshard(meta.cu_q_lens, replicated)
        starts = cu_q_lens[:-1]
        lengths = cu_q_lens[1] if batch == 1 else jnp.diff(cu_q_lens)
        width = getattr(meta, "max_prefill_len", 0) or count
        width = count if batch == 1 else min(width, count)
        valid = jnp.arange(width) < lengths[..., None]

        def gather(pool):
            # Slot 0 is the pool's immutable zero state. Select it for a fresh
            # request instead of materializing a masked copy of the state.
            slots = jnp.where(initial, indices, 0)
            return pool.at[slots].get(mode="clip", out_sharding=replicated)

        # Pack ragged requests into independent sequences for the native kernel.
        def pack(value):
            if batch == 1:
                return value
            positions = starts[:, None] + jnp.arange(width)
            return value.at[positions].get(mode="clip", out_sharding=replicated)

        def batched(value):
            return value[None] if batch == 1 else value

        packed = pack(mixed_qkv)
        sharding = jax.sharding.NamedSharding(self.mesh, jax.typeof(packed).sharding.spec)
        saved = gather(conv_state_in)
        if batch == 1:
            saved = saved[0]
        saved = jax.sharding.reshard(saved.swapaxes(-1, -2), sharding)
        history = jnp.concatenate((saved, packed), axis=-2)
        convolved = sum(
            history[..., tap : tap + width, :].astype(jnp.float32)
            * conv1d_weight[:, tap].astype(jnp.float32)
            for tap in range(self.conv_kernel_size)
        ).astype(mixed_qkv.dtype)
        convolved = jax.nn.silu(convolved)
        tail_indices = lengths[..., None] + jnp.arange(self.conv_kernel_size - 1)
        if batch > 1:
            tail_indices += jnp.arange(batch)[:, None] * history.shape[-2]
        tail = (
            history.reshape((-1, history.shape[-1]))
            .at[tail_indices]
            .get(mode="clip", out_sharding=replicated)
        )
        new_conv = ops.state_pool_update(conv_state_in, indices, batched(tail.swapaxes(-1, -2)))

        q, k, v = (
            x.reshape((*convolved.shape[:-1], self.num_v_heads, self.head_v_dim))
            for x in self._qkv(convolved.reshape((-1, convolved.shape[-1])))
        )
        beta = jax.nn.sigmoid(pack(b).astype(jnp.float32))
        gate = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
            pack(a).astype(jnp.float32) + dt_bias.astype(jnp.float32)
        )
        # Padding is an identity recurrence: zero update and zero log-decay.
        q, k, v = (jnp.where(valid[..., None, None], x, 0) for x in (q, k, v))
        beta, gate = (jnp.where(valid[..., None], x, 0) for x in (beta, gate))
        state, out = ops.gated_delta_rule(
            batched(q),
            batched(k),
            batched(v),
            batched(gate),
            batched(beta),
            gather(recurrent_state_in),
        )
        new_rec = ops.state_pool_update(recurrent_state_in, indices, state)
        if batch == 1:
            out = out[0]
        else:
            token = jnp.arange(count)
            sequence = jnp.searchsorted(cu_q_lens[1:], token, side="right")
            sequence = jnp.minimum(sequence, batch - 1)
            row = sequence * width + token - starts[sequence]
            out = (
                out.reshape((batch * width, -1))
                .at[row]
                .get(mode="clip", out_sharding=replicated)
                .reshape((count, self.num_v_heads, self.head_v_dim))
            )
            out = jnp.where((token < cu_q_lens[-1])[:, None, None], out, 0)
        return out.astype(mixed_qkv.dtype), new_conv, new_rec
