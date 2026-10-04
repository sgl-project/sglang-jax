"""TT recurrent attention: explicit kernels with scheduler-owned state slots."""

import functools
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


def _per_device(forward):
    """Runs forward on each device's heads, like GDNAttnBackend, passing it
    the forward metadata after self."""

    heads = P("data", "tensor", None)
    state = P("data", "tensor", None, None)
    in_specs = (
        P("data"),  # forward metadata
        P("data", "tensor"),  # mixed_qkv
        heads,  # conv_state
        state,  # recurrent_state
        P("data", "tensor"),  # b
        P("data", "tensor"),  # a
        P("tensor", None),  # conv1d weight
        P("tensor"),  # A_log
        P("tensor"),  # dt_bias
    )

    @functools.wraps(forward)
    def run(self, *args, **kwargs):
        return jax.shard_map(
            lambda *args: forward(self, *args, **kwargs),
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=(heads, heads, state),
        )(self.forward_metadata, *args)

    return run


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
            # A static slice runs on the mesh; an indexed get would build its
            # index as a single-device array, a separate program on one device.
            meta.cu_q_lens = jax.lax.slice_in_dim(meta.cu_q_lens, 0, size + 1)
            # The native kernel needs dense sequences. Bucket the longest one
            # instead of padding every sequence to the entire batch length.
            length = int(batch.extend_seq_lens[:size].max())
            meta = TTGDNMetadata(
                **vars(meta), max_prefill_len=1 << (max(length, 32) - 1).bit_length()
            )
        return meta

    def _metadata(self, meta):
        if meta.recurrent_track_indices is not None:
            raise NotImplementedError("TT GDN recurrent snapshots are not yet supported")
        return meta, meta.recurrent_indices, meta.has_initial_state

    def _qkv(self, mixed):
        shape = mixed.shape[:-1]
        # One device's heads. The kernels let each q/k head serve its group of
        # value heads, so q and k are not repeated.
        key_dim = self.key_dim // self.mesh.shape["tensor"]
        qk = mixed[..., : 2 * key_dim].reshape(*shape, -1, self.head_k_dim)
        v = mixed[..., 2 * key_dim :].reshape(*shape, -1, self.head_v_dim)
        q, k = jnp.split(_l2norm(qk.astype(jnp.float32)), 2, axis=-2)
        return q * self.head_k_dim**-0.5, k, v.astype(jnp.float32)

    @_per_device
    def forward_decode(
        self,
        meta,
        mixed_qkv,
        conv_state_in,
        recurrent_state_in,
        b,
        a,
        conv1d_weight,
        A_log,
        dt_bias,
    ):
        _, indices, initial = self._metadata(meta)
        new_conv, conv_out = ops.causal_conv1d_update(
            conv_state_in, mixed_qkv, conv1d_weight, indices, initial
        )
        # The kernel reads q, k and v as heads of the convolution output, and
        # normalizes and scales q and k like _qkv.
        new_rec, out = ops.gated_delta_decode(
            recurrent_state_in,
            conv_out,
            b,
            a,
            A_log,
            dt_bias,
            indices,
            initial,
        )
        return out.astype(mixed_qkv.dtype), new_conv, new_rec

    @_per_device
    def forward_extend(
        self,
        meta,
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
        meta, indices, initial = self._metadata(meta)
        count = mixed_qkv.shape[0]
        batch = meta.cu_q_lens.shape[0] - 1
        indices, initial = indices[:batch], initial[:batch]
        cu_q_lens = meta.cu_q_lens
        starts = cu_q_lens[:-1]
        lengths = jnp.diff(cu_q_lens)
        width = getattr(meta, "max_prefill_len", 0) or count
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
        tail = history.reshape((-1, history.shape[-1])).at[tail_indices].get(mode="clip")
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
        out = out.reshape(-1, *out.shape[-2:])
        if batch > 1:
            token = jnp.arange(count)
            sequence = jnp.searchsorted(cu_q_lens[1:], token, side="right")
            sequence = jnp.minimum(sequence, batch - 1)
            row = sequence * width + token - starts[sequence]
            out = out.at[row].get(mode="clip")
            out = jnp.where((token < cu_q_lens[-1])[:, None, None], out, 0)
        return out.astype(mixed_qkv.dtype), new_conv, new_rec
