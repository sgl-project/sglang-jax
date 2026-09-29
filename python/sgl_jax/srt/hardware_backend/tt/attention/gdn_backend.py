"""TT recurrent attention: explicit kernels with scheduler-owned state slots."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

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
        if self.mesh.shape.get("data", 1) != 1:
            raise NotImplementedError("TT GDN currently supports tensor parallelism only")
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
        # Runs on one device's heads inside shard_map.
        tp = self.mesh.shape["tensor"]
        key_dim, num_k_heads, num_v_heads = (
            self.key_dim // tp,
            self.num_k_heads // tp,
            self.num_v_heads // tp,
        )
        shape = mixed.shape[:-1]
        # Q and K use the same normalization and head expansion. Process them
        # together to avoid launching the identical operation chain twice.
        qk = mixed[..., : 2 * key_dim].reshape(*shape, 2 * num_k_heads, self.head_k_dim)
        v = mixed[..., 2 * key_dim :].reshape(*shape, num_v_heads, self.head_v_dim)
        qk = jnp.repeat(_l2norm(qk.astype(jnp.float32)), num_v_heads // num_k_heads, axis=-2)
        q, k = qk[..., :num_v_heads, :], qk[..., num_v_heads:, :]
        return q * self.head_k_dim**-0.5, k, v.astype(jnp.float32)

    def _per_device(self, local, extra_specs):
        # Like GDNAttnBackend: each device runs the kernels on its heads.
        in_specs = (
            P("data", "tensor"),  # mixed_qkv
            P("data", "tensor", None),  # conv_state
            P("data", "tensor", None, None),  # recurrent_state
            P("data", "tensor"),  # b
            P("data", "tensor"),  # a
            P("tensor", None),  # conv1d weight
            P("tensor"),  # A_log
            P("tensor"),  # dt_bias
        ) + extra_specs
        out_specs = (
            P("data", "tensor", None),  # out
            P("data", "tensor", None),  # new_conv_state
            P("data", "tensor", None, None),  # new_rec_state
        )
        return jax.shard_map(
            local, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        )

    def forward_decode(
        self, mixed_qkv, conv_state_in, recurrent_state_in, b, a, conv1d_weight, A_log, dt_bias
    ):
        _, indices, initial = self._metadata()

        def local(mixed_qkv, conv_state, recurrent_state, b, a, weight, A_log, dt_bias, indices, initial):
            new_conv, conv_out = ops.causal_conv1d_update(
                conv_state, mixed_qkv, weight, indices, initial
            )
            # The kernel reads q, k and v as heads of the flat convolution
            # output, normalizes and scales q and k like _qkv does, and lets
            # each key head serve its group of value heads.
            num_k_heads = self.num_k_heads // self.mesh.shape["tensor"]
            mixed = conv_out.astype(jnp.float32)
            new_rec, out = ops.gated_delta_decode(
                recurrent_state, mixed, mixed, mixed, b, a, A_log, dt_bias, indices, initial,
                key_head_offset=num_k_heads,
                value_head_offset=2 * num_k_heads,
                num_key_heads=num_k_heads,
                normalize_eps=1e-6,
                query_scale=self.head_k_dim**-0.5,
            )
            return out.astype(mixed_qkv.dtype), new_conv, new_rec

        return self._per_device(local, (P("data"), P("data")))(
            mixed_qkv, conv_state_in, recurrent_state_in, b, a, conv1d_weight, A_log, dt_bias,
            indices, initial,
        )

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
        batch = meta.cu_q_lens.shape[0] - 1
        max_prefill_len = getattr(meta, "max_prefill_len", 0)
        return self._per_device(
            lambda *args: self._extend_local(batch, max_prefill_len, *args),
            (P("data"), P("data"), P("data")),
        )(
            mixed_qkv, conv_state_in, recurrent_state_in, b, a, conv1d_weight, A_log, dt_bias,
            meta.cu_q_lens, indices[:batch], initial[:batch],
        )

    def _extend_local(
        self, batch, max_prefill_len, mixed_qkv, conv_state_in, recurrent_state_in, b, a,
        conv1d_weight, A_log, dt_bias, cu_q_lens, indices, initial,
    ):
        count = mixed_qkv.shape[0]
        starts = cu_q_lens[:-1]
        lengths = jnp.diff(cu_q_lens)
        width = max_prefill_len or count
        width = count if batch == 1 else min(width, count)
        valid = jnp.arange(width) < lengths[..., None]

        def gather(pool):
            # Slot 0 is the pool's immutable zero state. Select it for a fresh
            # request instead of materializing a masked copy of the state.
            slots = jnp.where(initial, indices, 0)
            return pool.at[slots].get(mode="clip")

        # Pack ragged requests into independent sequences for the native kernel.
        def pack(value):
            if batch == 1:
                return value[None]
            positions = starts[:, None] + jnp.arange(width)
            return value.at[positions].get(mode="clip")

        packed = pack(mixed_qkv)
        saved = gather(conv_state_in).swapaxes(-1, -2)
        history = jnp.concatenate((saved, packed), axis=-2)
        convolved = None
        for tap in range(self.conv_kernel_size):
            window = history[..., tap : tap + width, :].astype(jnp.float32)
            weight = conv1d_weight[:, tap].astype(jnp.float32)
            term = window * weight
            convolved = term if convolved is None else convolved + term
        convolved = convolved.astype(mixed_qkv.dtype)
        convolved = jax.nn.silu(convolved)
        tail_indices = lengths[..., None] + jnp.arange(self.conv_kernel_size - 1)
        tail_indices += jnp.arange(batch)[:, None] * history.shape[-2]
        tail = (
            history.reshape((-1, history.shape[-1]))
            .at[tail_indices]
            .get(mode="clip")
        )
        new_conv = ops.state_pool_update(conv_state_in, indices, tail.swapaxes(-1, -2))

        q, k, v = self._qkv(convolved)
        beta = jax.nn.sigmoid(pack(b).astype(jnp.float32))
        gate = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
            pack(a).astype(jnp.float32) + dt_bias.astype(jnp.float32)
        )
        # Zero beta and log-decay preserve state; zero queries mask padded output.
        q = jnp.where(valid[..., None, None], q, 0)
        beta, gate = (jnp.where(valid[..., None], x, 0) for x in (beta, gate))
        state, out = ops.gated_delta_rule(q, k, v, gate, beta, gather(recurrent_state_in))
        new_rec = ops.state_pool_update(recurrent_state_in, indices, state)
        out = out.reshape(-1, self.num_v_heads // self.mesh.shape["tensor"], self.head_v_dim)
        if batch > 1:
            token = jnp.arange(count)
            sequence = jnp.searchsorted(cu_q_lens[1:], token, side="right")
            sequence = jnp.minimum(sequence, batch - 1)
            row = sequence * width + token - starts[sequence]
            out = out.at[row].get(mode="clip")
            out = jnp.where((token < cu_q_lens[-1])[:, None, None], out, 0)
        return out.astype(mixed_qkv.dtype), new_conv, new_rec
