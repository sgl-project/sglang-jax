"""TT recurrent attention: explicit kernels with scheduler-owned state slots."""

import jax
import jax.numpy as jnp

from sgl_jax.srt.hardware_backend.tt.attention import ops
from sgl_jax.srt.hardware_backend.tt.attention.tt_backend import TTAttention
from sgl_jax.srt.kernels.gdn.gated_delta import _l2norm
from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend


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

    def prepare_model_state(self, leaves):
        return tuple(
            (
                ops.annotate_weight_dtype(leaf, "bfp_bf8" if leaf.ndim >= 2 else "bf16")
                if getattr(leaf, "ndim", 0) > 0
                and getattr(leaf, "dtype", None) in (jnp.bfloat16, jnp.float32)
                else leaf
            )
            for leaf in leaves
        )

    def get_forward_metadata(self, batch):
        if batch.forward_mode.is_extend() and batch.real_bs > 1:
            raise NotImplementedError("TT GDN prefill supports one request at a time")
        return super().get_forward_metadata(batch)

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
        # Prefill admits one request; metadata is padded to max_running_requests.
        indices, initial = indices[:1], initial[:1]
        count = mixed_qkv.shape[0]
        length = meta.cu_q_lens[1]
        valid = jnp.arange(count) < length
        replicated = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

        def gather(pool):
            # Slot 0 is the pool's immutable zero state. Select it for a fresh
            # request instead of materializing a masked copy of the state.
            slots = jnp.where(initial, indices, 0)
            return pool.at[slots].get(mode="clip", out_sharding=replicated)

        # Short FIR prefill; the saved history precedes this chunk's live tokens.
        sharding = jax.sharding.NamedSharding(self.mesh, jax.typeof(mixed_qkv).sharding.spec)
        saved = jax.sharding.reshard(gather(conv_state_in)[0].T, sharding)
        history = jnp.concatenate((saved, mixed_qkv), axis=0)
        convolved = sum(
            history[tap : tap + count].astype(jnp.float32)
            * conv1d_weight[:, tap].astype(jnp.float32)
            for tap in range(self.conv_kernel_size)
        ).astype(mixed_qkv.dtype)
        convolved = jax.nn.silu(convolved)
        tail = history.at[length + jnp.arange(self.conv_kernel_size - 1)].get(
            mode="clip", out_sharding=replicated
        )
        new_conv = ops.state_pool_update(conv_state_in, indices, tail.T[None])

        q, k, v = self._qkv(convolved)
        beta = jax.nn.sigmoid(b.astype(jnp.float32))
        gate = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
            a.astype(jnp.float32) + dt_bias.astype(jnp.float32)
        )
        # Padding is an identity recurrence: zero update and zero log-decay.
        q, k, v = (jnp.where(valid[:, None, None], x, 0) for x in (q, k, v))
        beta, gate = (jnp.where(valid[:, None], x, 0) for x in (beta, gate))
        state, out = ops.gated_delta_rule(
            q[None], k[None], v[None], gate[None], beta[None], gather(recurrent_state_in)
        )
        new_rec = ops.state_pool_update(recurrent_state_in, indices, state)
        return out[0].astype(mixed_qkv.dtype), new_conv, new_rec
