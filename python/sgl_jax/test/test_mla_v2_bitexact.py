"""Bit-exact parity tests for the absorbed-MLA v2 kernel against pre-#1733 semantics.

Provides two complementary layers of bit-exact parity verification against the
pre-#1733 (``c136c733``) MLA v2 kernel:

1. **Fast CPU source-level guard (``test_v2_kernel_bitexact_parity_output_and_l``)**:
   ``run_v2_kernel_with_capture`` replaces ``pl.pallas_call`` with a NumPy/JAX
   emulation of ``_mla_ragged_paged_attention_kernel``, replaces the
   ``pl.reciprocal(l, approx=True)`` call with exact ``lax.reciprocal``, and
   compares ``output``, ``l`` (``l_ref`` normalizer), ``acc`` (``acc_ref``
   accumulator), and ``updated_cache_kv`` against a Python reimplementation of
   the ``c136c733`` parent kernel (``pre1733_reference_mla_v2``). This runs on
   CPU in ~10s and catches source-level numeric or KV-packing edits (such as the
   ``jnp.exp2`` / ``log2(e)`` rewrite, folded ``float(sm_scale * q_scale)``
   scaling, or unaligned ``merge_kv`` bounds), without executing the
   Mosaic-compiled kernel or TPU MXU.
2. **TPU compiled Mosaic/MXU guard (``test_v2_kernel_tpu_compiled_bitexact_parity``)**:
   When running on TPU (``jax.default_backend() == "tpu"``), executes
   ``mla_ragged_paged_attention`` completely unpatched through ``pl.pallas_call``
   (full Mosaic lowering, MXU ``Q @ K^T`` and ``P @ V`` accumulation, VMEM bitcast
   reshapes, and hardware ``pl.reciprocal(l, approx=True)``) and asserts
   ``max_abs_diff == 0.0`` against the frozen ``c136c733`` Pallas block reference
   compiled on the same TPU device. Both calls use a scoped single-device
   explicit mesh and the same ``shard_map`` boundary as the production backend.
"""

from __future__ import annotations

import unittest
import warnings
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.mla.v2 import kernel as kmod


def _cpu_bitcast(arr, target_dtype):
    np_arr = np.asarray(arr)
    src_dtype = np_arr.dtype
    tgt_dtype = np.dtype(target_dtype)
    if src_dtype.itemsize == tgt_dtype.itemsize:
        return jnp.asarray(np_arr.view(tgt_dtype))
    if src_dtype.itemsize < tgt_dtype.itemsize:
        packing = tgt_dtype.itemsize // src_dtype.itemsize
        sh = np_arr.shape
        v = np.swapaxes(np_arr.reshape(*sh[:-2], sh[-2] // packing, packing, sh[-1]), -1, -2)
        return jnp.asarray(np.ascontiguousarray(v).view(tgt_dtype).squeeze(-1))
    packing = src_dtype.itemsize // tgt_dtype.itemsize
    expanded = np.ascontiguousarray(np_arr)[..., None].view(tgt_dtype)
    swapped = np.swapaxes(expanded, -1, -2)
    return jnp.asarray(
        swapped.reshape(*np_arr.shape[:-2], np_arr.shape[-2] * packing, np_arr.shape[-1])
    )


def _cpu_reciprocal(x, *, approx=False):
    del approx
    return lax.reciprocal(x)


class _CpuRef:
    """Lightweight Pallas Ref emulator that executes ``_mla_ragged_paged_attention_kernel`` on CPU."""

    def __init__(self, buf, idx=(), reshape_shape=None, bitcast_dtype=None, on_root_read=None):
        self._buf = buf
        self._idx = idx
        self._reshape_shape = reshape_shape
        self._bitcast_dtype = bitcast_dtype
        self._on_root_read = on_root_read

    def _get_view(self):
        v = self._buf
        for s in self._idx:
            v = v[s]
        if self._bitcast_dtype is not None and self._bitcast_dtype.itemsize != v.dtype.itemsize:
            packing = self._bitcast_dtype.itemsize // v.dtype.itemsize
            sh = v.shape
            v = np.swapaxes(v.reshape(*sh[:-2], sh[-2] // packing, packing, sh[-1]), -1, -2)
            v = np.ascontiguousarray(v).view(self._bitcast_dtype).squeeze(-1)
        elif self._bitcast_dtype is not None and self._bitcast_dtype != v.dtype:
            v = np.ascontiguousarray(v).view(self._bitcast_dtype)
        if self._reshape_shape is not None:
            v = v.reshape(self._reshape_shape)
        return v

    def _set_view(self, new_v):
        new_np = np.asarray(new_v)
        base_v = self._buf
        for s in self._idx:
            base_v = base_v[s]
        if (
            self._bitcast_dtype is not None
            and self._bitcast_dtype.itemsize != base_v.dtype.itemsize
        ):
            packing = self._bitcast_dtype.itemsize // base_v.dtype.itemsize
            sh = base_v.shape
            b_sh = (*sh[:-2], sh[-2] // packing, sh[-1])
            new_np = new_np.reshape(b_sh)
            expanded = np.ascontiguousarray(new_np)[..., None].view(base_v.dtype)
            swapped = np.swapaxes(expanded, -1, -2)
            new_np = swapped.reshape(base_v.shape)
        elif self._bitcast_dtype is not None and self._bitcast_dtype != base_v.dtype:
            new_np = np.ascontiguousarray(new_np).view(base_v.dtype).reshape(base_v.shape)
        else:
            new_np = new_np.reshape(base_v.shape)
        target = self._buf
        for s in self._idx[:-1]:
            target = target[s]
        if self._idx:
            target[self._idx[-1]] = new_np
        else:
            self._buf[...] = new_np

    @property
    def shape(self):
        return self._get_view().shape

    @property
    def dtype(self):
        return jnp.dtype(self._get_view().dtype)

    def _norm_idx(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        out = []
        for item in idx:
            if isinstance(item, pl.Slice):
                out.append(slice(int(item.start), int(item.start) + int(item.size)))
            elif isinstance(item, (jax.Array, np.ndarray)) and item.ndim == 0:
                out.append(int(item))
            else:
                out.append(item)
        return tuple(out)

    @property
    def at(self):
        parent = self

        class _At:
            def __getitem__(self, idx):
                norm = parent._norm_idx(idx)
                if parent._reshape_shape is not None or parent._bitcast_dtype is not None:
                    return _SubViewRef(parent, norm)
                return _CpuRef(parent._buf, parent._idx + norm)

        return _At()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _CpuRef(
            self._buf,
            self._idx,
            reshape_shape=shape,
            bitcast_dtype=self._bitcast_dtype,
        )

    def bitcast(self, dtype):
        return _CpuRef(
            self._buf,
            self._idx,
            reshape_shape=self._reshape_shape,
            bitcast_dtype=np.dtype(dtype),
        )

    def __getitem__(self, idx):
        if idx is Ellipsis:
            if (
                not self._idx
                and self._reshape_shape is None
                and self._bitcast_dtype is None
                and self._on_root_read is not None
            ):
                self._on_root_read(self._buf.copy())
            return jnp.asarray(self._get_view())
        view = self._get_view()
        norm = list(self._norm_idx(idx))
        for d, item in enumerate(norm):
            if isinstance(item, int) and d < view.ndim:
                norm[d] = min(max(item, 0), view.shape[d] - 1)
        return jnp.asarray(view[tuple(norm)])

    def __setitem__(self, idx, val):
        cur = self._get_view().copy()
        if idx is Ellipsis:
            cur[...] = np.asarray(val)
        else:
            norm = self._norm_idx(idx)
            cur[norm] = np.asarray(val)
        self._set_view(cur)


class _SubViewRef(_CpuRef):
    def __init__(self, parent, sub_idx, reshape_shape=None, bitcast_dtype=None):
        self._parent = parent
        self._sub_idx = sub_idx
        self._idx = sub_idx
        self._reshape_shape = reshape_shape
        self._bitcast_dtype = bitcast_dtype
        self._on_root_read = None

    def _get_view(self):
        v = self._parent._get_view()[self._sub_idx]
        if self._bitcast_dtype is not None and self._bitcast_dtype.itemsize != v.dtype.itemsize:
            packing = self._bitcast_dtype.itemsize // v.dtype.itemsize
            sh = v.shape
            v = np.swapaxes(v.reshape(*sh[:-2], sh[-2] // packing, packing, sh[-1]), -1, -2)
            v = np.ascontiguousarray(v).view(self._bitcast_dtype).squeeze(-1)
        elif self._bitcast_dtype is not None and self._bitcast_dtype != v.dtype:
            v = np.ascontiguousarray(v).view(self._bitcast_dtype)
        if self._reshape_shape is not None:
            v = v.reshape(self._reshape_shape)
        return v

    def _set_view(self, new_v):
        cur = self._parent._get_view().copy()
        base_v = cur[self._sub_idx]
        new_np = np.asarray(new_v)
        if (
            self._bitcast_dtype is not None
            and self._bitcast_dtype.itemsize != base_v.dtype.itemsize
        ):
            packing = self._bitcast_dtype.itemsize // base_v.dtype.itemsize
            sh = base_v.shape
            b_sh = (*sh[:-2], sh[-2] // packing, sh[-1])
            new_np = new_np.reshape(b_sh)
            expanded = np.ascontiguousarray(new_np)[..., None].view(base_v.dtype)
            swapped = np.swapaxes(expanded, -1, -2)
            new_np = swapped.reshape(base_v.shape)
        elif self._bitcast_dtype is not None and self._bitcast_dtype != base_v.dtype:
            new_np = np.ascontiguousarray(new_np).view(base_v.dtype).reshape(base_v.shape)
        else:
            new_np = new_np.reshape(base_v.shape)
        cur[self._sub_idx] = new_np
        self._parent._set_view(cur)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _SubViewRef(
            self._parent,
            self._sub_idx,
            reshape_shape=shape,
            bitcast_dtype=self._bitcast_dtype,
        )

    def bitcast(self, dtype):
        return _SubViewRef(
            self._parent,
            self._sub_idx,
            reshape_shape=self._reshape_shape,
            bitcast_dtype=np.dtype(dtype),
        )


class _FakeAsyncCopy:
    def __init__(self, src, dst, sem):
        del sem
        self.src = src
        self.dst = dst

    def start(self):
        if self.src.shape != self.dst.shape or self.src._get_view().size == 0:
            return
        self.dst[...] = self.src[...]

    def wait(self):
        pass


def _py_when(cond):
    def dec(fn):
        if bool(cond):
            fn()
        return fn

    return dec


def _py_fori_loop(lower, upper, body_fun, init_val, *, unroll=None):
    del unroll
    val = init_val
    for i in range(int(lower), int(upper)):
        val = body_fun(i, val)
    return val


def run_v2_kernel_with_capture(*args, **kwargs):
    """Execute ``sgl_jax.srt.kernels.mla.v2.kernel.mla_ragged_paged_attention``
    and capture ``(output, updated_kv, l_snapshots, acc_snapshots)``."""
    l_snapshots = []
    acc_snapshots = []

    def fake_pallas_call(kernel_fn, *, grid_spec, out_shape, input_output_aliases=None, **kw):
        del kw

        def runner(*call_args):
            num_sp = grid_spec.num_scalar_prefetch
            sp_args = [_CpuRef(np.array(a, copy=True)) for a in call_args[:num_sp]]
            in_bufs = [np.array(a, copy=True) for a in call_args[num_sp:]]
            out_bufs = [np.zeros(s.shape, dtype=s.dtype) for s in out_shape]
            if input_output_aliases:
                for in_idx, out_idx in input_output_aliases.items():
                    out_bufs[out_idx] = in_bufs[in_idx - num_sp].copy()
            in_refs = [_CpuRef(b) for b in in_bufs]
            out_refs = [_CpuRef(b) for b in out_bufs]
            scratch_refs = []
            num_ss = len(grid_spec.scratch_shapes)
            for idx_s, ss in enumerate(grid_spec.scratch_shapes):
                if hasattr(ss, "dtype") and str(ss.dtype) != "dma_sem":
                    cb = None
                    if idx_s == num_ss - 3:
                        cb = l_snapshots.append
                    elif idx_s == num_ss - 1:
                        cb = acc_snapshots.append
                    scratch_refs.append(
                        _CpuRef(np.zeros(ss.shape, dtype=ss.dtype), on_root_read=cb)
                    )
                else:
                    scratch_refs.append(_CpuRef(np.zeros(ss.shape, dtype=np.int32)))
            g0 = int(grid_spec.grid[0])
            for pid in range(g0):
                with mock.patch.object(pl, "program_id", side_effect=lambda axis, p=pid: p):
                    kernel_fn(*sp_args, *in_refs, *out_refs, *scratch_refs)
            return [jnp.asarray(b) for b in out_bufs]

        return runner

    cpu_devices = jax.devices("cpu")
    ctx = jax.default_device(cpu_devices[0]) if cpu_devices else mock.MagicMock()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with (
            ctx,
            mock.patch.object(pl, "pallas_call", side_effect=fake_pallas_call),
            mock.patch.object(pl, "when", side_effect=_py_when),
            mock.patch.object(pl, "reciprocal", side_effect=_cpu_reciprocal),
            mock.patch.object(pltpu, "reciprocal", side_effect=_cpu_reciprocal, create=True),
            mock.patch.object(pltpu, "make_async_copy", side_effect=_FakeAsyncCopy),
            mock.patch.object(pltpu, "bitcast", side_effect=_cpu_bitcast),
            mock.patch.object(lax, "fori_loop", side_effect=_py_fori_loop),
        ):
            out, updated_kv = kmod.mla_ragged_paged_attention.__wrapped__(*args, **kwargs)
    return np.asarray(out), np.asarray(updated_kv), l_snapshots, acc_snapshots


def _make_tpu_pallas_flash_attention_block_ref(
    *,
    bsz: int,
    bq_sz: int,
    num_q_heads: int,
    bkv_sz: int,
    lkv_dim: int,
    r_dim: int,
    num_bkv: int,
    sm_scale: float,
    mask_value: float,
    q_dtype,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
):
    """Frozen pre-#1733 (``c136c733``) Pallas TPU block kernel for Mosaic/MXU parity."""
    q_rows = bq_sz * num_q_heads

    def _broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        target_minor = kmod.align_to(shape[-1], src.shape[-1])
        return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[
            ..., : shape[-1]
        ]

    def _kernel(
        meta_ref,
        ql_ref,
        qpe_ref,
        kvc_all_ref,
        kpe_all_ref,
        out_ref,
        m_ref,
        l_ref,
        acc_ref,
    ):
        bq_idx = meta_ref[0]
        ql_vec = ql_ref[...]
        qpe_vec = qpe_ref[...]

        def load_with_init(ref, init_val, bkv_idx):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref[...], init_val), ref[...])

        def body(bkv_idx, _):
            kv_c = kvc_all_ref[bkv_idx]
            k_pe = kpe_all_ref[bkv_idx]
            q = jnp.concatenate([ql_vec, qpe_vec], axis=-1)
            k = jnp.concatenate([kv_c, k_pe], axis=-1)
            s = jnp.einsum("bnd,bmd->bnm", q, k, preferred_element_type=jnp.float32)
            s *= sm_scale
            if k_scale is not None:
                s *= k_scale
            if q_scale is not None:
                s *= q_scale

            k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, s.shape[1:], 1)
            mask_list = []
            for b in range(bsz):
                q_len = meta_ref[1 + 2 * b]
                kv_len = meta_ref[2 + 2 * b]
                q_span = (
                    kv_len
                    - q_len
                    + bq_idx * bq_sz
                    + lax.div(lax.broadcasted_iota(jnp.int32, s.shape[1:], 0), num_q_heads)
                )
                mask = q_span < k_span
                if sliding_window is not None:
                    mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)
                mask_list.append(mask)
            mask = jnp.stack(mask_list, axis=0)

            if soft_cap is not None:
                s = soft_cap * jnp.tanh(s / soft_cap)
            s = jnp.where(mask, mask_value, s)
            s_rowmax = jnp.max(s, axis=2, keepdims=True)

            head_m_ref = m_ref.at[:, : s.shape[1]]
            head_l_ref = l_ref.at[:, : s.shape[1]]
            head_acc_ref = acc_ref.at[:, : s.shape[1]]

            m_prev = load_with_init(head_m_ref, -jnp.inf, bkv_idx)
            m_curr = jnp.maximum(m_prev, s_rowmax)
            head_m_ref[...] = m_curr
            p = jnp.exp(s - _broadcast_minor(m_curr, s.shape))

            pv = jnp.einsum("bnm,bmd->bnd", p, kv_c, preferred_element_type=jnp.float32)
            if v_scale is not None:
                pv *= v_scale

            p_rowsum = jnp.sum(p, axis=2, keepdims=True)
            exp_m_diff = jnp.exp(m_prev - m_curr)
            l_prev = load_with_init(head_l_ref, 0.0, bkv_idx)
            l_curr = exp_m_diff * l_prev + p_rowsum
            head_l_ref[...] = l_curr
            o_prev = load_with_init(head_acc_ref, 0.0, bkv_idx)
            o_curr = _broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
            head_acc_ref[...] = o_curr

        lax.fori_loop(0, num_bkv, body, None, unroll=False)
        acc = acc_ref[...]
        l_val = _broadcast_minor(l_ref[...], acc.shape)
        out_ref[...] = (
            lax.div(acc, l_val)
            if q_dtype == jnp.float32
            else (acc * pl.reciprocal(l_val, approx=True)).astype(q_dtype)
        )

    pallas_fn = pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((bsz, q_rows, 128), jnp.float32),
                pltpu.VMEM((bsz, q_rows, 128), jnp.float32),
                pltpu.VMEM((bsz, q_rows, lkv_dim), jnp.float32),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((bsz, q_rows, lkv_dim), q_dtype),
    )
    return jax.shard_map(
        pallas_fn,
        mesh=jax.sharding.get_mesh(),
        in_specs=(P(),) * 5,
        out_specs=P(),
        check_vma=False,
    )


def pre1733_reference_mla_v2(
    ql_nope,
    q_pe,
    new_kv_c,
    new_k_pe,
    cache_kv,
    kv_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    *,
    sm_scale=1.0,
    sliding_window=None,
    soft_cap=None,
    mask_value=kmod.DEFAULT_MASK_VALUE,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    num_kv_pages_per_block=(1, 1, 1),
    num_queries_per_block=(1, 1, 8),
    decode_batch_size=1,
    use_tpu_pallas=False,
):
    """Self-contained pre-#1733 (``c136c733``) reference for the v2 MLA kernel.

    Executes the exact pre-#1733 tiled FlashAttention-2 recurrence:
      - ``q = jnp.concatenate([ql_nope, q_pe], axis=-1)``
      - ``k = jnp.concatenate([kv_c, k_pe], axis=-1)``
      - ``s = jnp.einsum("bnd,bmd->bnm", q, k, preferred_element_type=jnp.float32) * sm_scale``
      - natural-base ``jnp.exp`` online softmax (``m_curr``, ``p``, ``exp_m_diff``, ``l_curr``)
      - ``float32`` ``p`` in ``pv = jnp.einsum("bnm,bmd->bnd", p, kv_c, preferred_element_type=jnp.float32)``
    """
    cpu_devices = jax.devices("cpu")
    ctx = (
        jax.default_device(cpu_devices[0])
        if (cpu_devices and not use_tpu_pallas)
        else mock.MagicMock()
    )
    with ctx:
        q_dtype = ql_nope.dtype
        kv_dtype = cache_kv.dtype
        _, actual_num_q_heads, actual_lkv_dim = ql_nope.shape
        actual_r_dim = q_pe.shape[-1]
        q_packing = kmod.get_dtype_packing(q_dtype)
        kv_packing = kmod.get_dtype_packing(kv_dtype)
        num_q_heads = kmod.align_to(actual_num_q_heads, q_packing)
        lkv_dim = kmod.align_to(actual_lkv_dim, 128)
        r_dim = kmod.align_to(actual_r_dim, 128)
        _, page_size_per_kv_packing, _, _ = cache_kv.shape
        page_size = page_size_per_kv_packing * kv_packing

        ql_pad = jnp.pad(
            ql_nope,
            ((0, 0), (0, num_q_heads - actual_num_q_heads), (0, lkv_dim - actual_lkv_dim)),
        )
        qpe_pad = jnp.pad(
            q_pe,
            ((0, 0), (0, num_q_heads - actual_num_q_heads), (0, r_dim - actual_r_dim)),
        )
        kvc_pad = jnp.pad(new_kv_c, ((0, 0), (0, lkv_dim - actual_lkv_dim)))
        kpe_pad = jnp.pad(new_k_pe, ((0, 0), (0, r_dim - actual_r_dim)))

        updated_cache = np.array(cache_kv, copy=True)
        cu_q = np.asarray(cu_q_lens)
        cu_kv = np.asarray(cu_kv_lens)
        kvl = np.asarray(kv_lens)
        pidx = np.asarray(page_indices)
        dist = np.asarray(distribution)

        for seq_idx in range(int(dist[2])):
            q_start, q_end = int(cu_q[seq_idx]), int(cu_q[seq_idx + 1])
            q_len = q_end - q_start
            kv_len = int(kvl[seq_idx])
            prefix_len = kv_len - q_len
            start_page = int(cu_kv[seq_idx]) // page_size
            for t in range(q_len):
                pos = prefix_len + t
                p_i = pidx[start_page + pos // page_size]
                in_page = pos % page_size
                w_row, w_col = in_page // kv_packing, in_page % kv_packing
                updated_cache[p_i, w_row, w_col, :lkv_dim] = np.asarray(kvc_pad[q_start + t])
                updated_cache[p_i, w_row, w_col, lkv_dim : lkv_dim + r_dim] = np.asarray(
                    kpe_pad[q_start + t]
                )

        def broadcast_minor(src, shape):
            if src.shape == shape:
                return src
            target_minor = kmod.align_to(shape[-1], src.shape[-1])
            return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[
                ..., : shape[-1]
            ]

        out_full = np.asarray(ql_pad, copy=True)
        l_blocks = []
        acc_blocks = []

        batch_dist = (int(dist[0]) // decode_batch_size) * decode_batch_size
        stages = [
            (
                0,
                batch_dist,
                1,
                num_kv_pages_per_block[0],
                num_queries_per_block[0],
                decode_batch_size,
            ),
            (batch_dist, int(dist[0]), 1, num_kv_pages_per_block[0], num_queries_per_block[0], 1),
            (
                int(dist[1]),
                int(dist[2]),
                None,
                num_kv_pages_per_block[2],
                num_queries_per_block[2],
                1,
            ),
        ]

        for start_seq, end_seq, static_q_len, bkv_p, num_q_per_blk, bsz in stages:
            bq_sz = min(num_q_per_blk, static_q_len) if static_q_len is not None else num_q_per_blk
            bkv_sz = bkv_p * page_size
            grid_len = (end_seq - start_seq) // bsz
            for pid in range(grid_len):
                batch_start = start_seq + pid * bsz
                kv_len_max = max(int(kvl[batch_start + b]) for b in range(bsz))
                q_len_max = max(
                    int(cu_q[batch_start + b + 1] - cu_q[batch_start + b]) for b in range(bsz)
                )
                num_bkv = (kv_len_max + bkv_sz - 1) // bkv_sz
                actual_bq_sz = bq_sz if static_q_len is None else min(bq_sz, static_q_len)
                num_bq = (
                    (q_len_max + actual_bq_sz - 1) // actual_bq_sz
                    if static_q_len is None
                    else (static_q_len + actual_bq_sz - 1) // actual_bq_sz
                )

                bq_nope_buf = np.zeros((bsz, bq_sz, num_q_heads, lkv_dim), dtype=q_dtype)
                bq_pe_buf = np.zeros((bsz, bq_sz, num_q_heads, r_dim), dtype=q_dtype)

                for bq_idx in range(num_bq):
                    for b in range(bsz):
                        s_i = batch_start + b
                        q_s = int(cu_q[s_i]) + bq_idx * bq_sz
                        q_e = int(cu_q[s_i + 1])
                        sz = min(bq_sz, q_e - q_s)
                        if sz > 0:
                            bq_nope_buf[b, :sz] = np.asarray(ql_pad[q_s : q_s + sz])
                            bq_pe_buf[b, :sz] = np.asarray(qpe_pad[q_s : q_s + sz])

                    ql_vec = jnp.asarray(bq_nope_buf[:, :actual_bq_sz]).reshape(
                        bsz, actual_bq_sz * num_q_heads, lkv_dim
                    )
                    qpe_vec = jnp.asarray(bq_pe_buf[:, :actual_bq_sz]).reshape(
                        bsz, actual_bq_sz * num_q_heads, r_dim
                    )

                    if use_tpu_pallas:
                        kvc_list = []
                        kpe_list = []
                        for bkv_idx in range(num_bkv):
                            kvc_blk = np.zeros((bsz, bkv_sz, lkv_dim), dtype=kv_dtype)
                            kpe_blk = np.zeros((bsz, bkv_sz, r_dim), dtype=kv_dtype)
                            for b in range(bsz):
                                s_i = batch_start + b
                                kv_len = int(kvl[s_i])
                                start_page = int(cu_kv[s_i]) // page_size
                                kv_start = bkv_idx * bkv_sz
                                valid_n = max(min(kv_len - kv_start, bkv_sz), 0)
                                for t in range(valid_n):
                                    pos = kv_start + t
                                    p_i = pidx[start_page + pos // page_size]
                                    in_page = pos % page_size
                                    w_row, w_col = in_page // kv_packing, in_page % kv_packing
                                    kvc_blk[b, t] = updated_cache[p_i, w_row, w_col, :lkv_dim]
                                    kpe_blk[b, t] = updated_cache[
                                        p_i, w_row, w_col, lkv_dim : lkv_dim + r_dim
                                    ]
                            kvc_list.append(jnp.asarray(kvc_blk))
                            kpe_list.append(jnp.asarray(kpe_blk))
                        meta_vals = [bq_idx]
                        for b in range(bsz):
                            s_i = batch_start + b
                            meta_vals.append(int(cu_q[s_i + 1] - cu_q[s_i]))
                            meta_vals.append(int(kvl[s_i]))
                        while len(meta_vals) % 8 != 0:
                            meta_vals.append(0)
                        meta_arr = jnp.asarray(meta_vals, dtype=jnp.int32)
                        pallas_fn = _make_tpu_pallas_flash_attention_block_ref(
                            bsz=bsz,
                            bq_sz=actual_bq_sz,
                            num_q_heads=num_q_heads,
                            bkv_sz=bkv_sz,
                            lkv_dim=lkv_dim,
                            r_dim=r_dim,
                            num_bkv=num_bkv,
                            sm_scale=sm_scale,
                            mask_value=mask_value,
                            q_dtype=q_dtype,
                            q_scale=q_scale,
                            k_scale=k_scale,
                            v_scale=v_scale,
                            sliding_window=sliding_window,
                            soft_cap=soft_cap,
                        )
                        out_blk = pallas_fn(
                            meta_arr,
                            ql_vec,
                            qpe_vec,
                            jnp.stack(kvc_list, axis=0),
                            jnp.stack(kpe_list, axis=0),
                        )
                    else:
                        m_curr = jnp.full(
                            (bsz, actual_bq_sz * num_q_heads, 128), -jnp.inf, dtype=jnp.float32
                        )
                        l_curr = jnp.zeros(
                            (bsz, actual_bq_sz * num_q_heads, 128), dtype=jnp.float32
                        )
                        o_curr = jnp.zeros(
                            (bsz, actual_bq_sz * num_q_heads, lkv_dim), dtype=jnp.float32
                        )

                        for bkv_idx in range(num_bkv):
                            kvc_blk = np.zeros((bsz, bkv_sz, lkv_dim), dtype=kv_dtype)
                            kpe_blk = np.zeros((bsz, bkv_sz, r_dim), dtype=kv_dtype)
                            for b in range(bsz):
                                s_i = batch_start + b
                                kv_len = int(kvl[s_i])
                                start_page = int(cu_kv[s_i]) // page_size
                                kv_start = bkv_idx * bkv_sz
                                valid_n = max(min(kv_len - kv_start, bkv_sz), 0)
                                for t in range(valid_n):
                                    pos = kv_start + t
                                    p_i = pidx[start_page + pos // page_size]
                                    in_page = pos % page_size
                                    w_row, w_col = in_page // kv_packing, in_page % kv_packing
                                    kvc_blk[b, t] = updated_cache[p_i, w_row, w_col, :lkv_dim]
                                    kpe_blk[b, t] = updated_cache[
                                        p_i, w_row, w_col, lkv_dim : lkv_dim + r_dim
                                    ]

                            kv_c = jnp.asarray(kvc_blk)
                            k_pe = jnp.asarray(kpe_blk)

                            q = jnp.concatenate([ql_vec, qpe_vec], axis=-1)
                            k = jnp.concatenate([kv_c, k_pe], axis=-1)
                            s = jnp.einsum("bnd,bmd->bnm", q, k, preferred_element_type=jnp.float32)
                            s *= sm_scale
                            if k_scale is not None:
                                s *= k_scale
                            if q_scale is not None:
                                s *= q_scale

                            k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(
                                jnp.int32, s.shape[1:], 1
                            )
                            mask_list = []
                            for b in range(bsz):
                                s_i = batch_start + b
                                q_len = int(cu_q[s_i + 1] - cu_q[s_i])
                                kv_len = int(kvl[s_i])
                                q_span = (
                                    kv_len
                                    - q_len
                                    + bq_idx * bq_sz
                                    + (
                                        lax.broadcasted_iota(jnp.int32, s.shape[1:], 0)
                                        // num_q_heads
                                    )
                                )
                                mask = q_span < k_span
                                if sliding_window is not None:
                                    mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)
                                mask_list.append(mask)
                            mask = jnp.stack(mask_list, axis=0)

                            if soft_cap is not None:
                                s = soft_cap * jnp.tanh(s / soft_cap)
                            s = jnp.where(mask, mask_value, s)
                            s_rowmax = jnp.max(s, axis=2, keepdims=True)
                            m_prev = m_curr
                            m_curr = jnp.maximum(m_prev, s_rowmax)
                            p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

                            pv = jnp.einsum(
                                "bnm,bmd->bnd", p, kv_c, preferred_element_type=jnp.float32
                            )
                            if v_scale is not None:
                                pv *= v_scale

                            p_rowsum = jnp.sum(p, axis=2, keepdims=True)
                            exp_m_diff = jnp.exp(m_prev - m_curr)
                            l_curr = exp_m_diff * l_curr + p_rowsum
                            o_curr = broadcast_minor(exp_m_diff, o_curr.shape) * o_curr + pv

                        max_sz = max(
                            min(
                                bq_sz,
                                max(
                                    int(
                                        cu_q[batch_start + b + 1]
                                        - (cu_q[batch_start + b] + bq_idx * bq_sz)
                                    ),
                                    0,
                                ),
                            )
                            for b in range(bsz)
                        )
                        valid_q_rows = max_sz * num_q_heads
                        l_blocks.append((valid_q_rows, np.asarray(l_curr[:, :valid_q_rows, :])))
                        acc_blocks.append((valid_q_rows, np.asarray(o_curr[:, :valid_q_rows, :])))

                        l_bcast = broadcast_minor(l_curr, o_curr.shape)
                        out_blk = (
                            lax.div(o_curr, l_bcast)
                            if q_dtype == jnp.float32
                            else (o_curr * lax.reciprocal(l_bcast)).astype(q_dtype)
                        )

                    out_blk_np = np.asarray(out_blk).reshape(
                        bsz, actual_bq_sz, num_q_heads, lkv_dim
                    )
                    for b in range(bsz):
                        s_i = batch_start + b
                        q_s = int(cu_q[s_i]) + bq_idx * bq_sz
                        q_e = int(cu_q[s_i + 1])
                        sz = min(bq_sz, q_e - q_s)
                        if sz > 0:
                            out_full[q_s : q_s + sz] = out_blk_np[b, :sz]

        return (
            out_full[:, :actual_num_q_heads, :actual_lkv_dim],
            updated_cache,
            l_blocks,
            acc_blocks,
        )


def make_mla_v2_parity_inputs(
    lens,
    *,
    page_size=16,
    num_heads=8,
    kv_lora_rank=512,
    qk_rope_dim=64,
    dtype=jnp.bfloat16,
    seed=42,
):
    rng = np.random.default_rng(seed)
    kv_packing = kmod.get_dtype_packing(dtype)
    nope_dim = kmod.align_to(kv_lora_rank, 128)
    rope_dim = kmod.align_to(qk_rope_dim, 128)
    kv_dim = nope_dim + rope_dim
    page_size_per_kv_packing = page_size // kv_packing

    q_lens = [q for q, _ in lens]
    kv_lens = [kv for _, kv in lens]
    total_q = sum(q_lens)
    aligned_kv_lens = [((kv + page_size - 1) // page_size) * page_size for kv in kv_lens]
    page_counts = [akv // page_size for akv in aligned_kv_lens]
    total_pages = sum(page_counts) + 2

    ql_nope = jnp.asarray(rng.standard_normal((total_q, num_heads, kv_lora_rank)) * 0.25, dtype)
    q_pe = jnp.asarray(rng.standard_normal((total_q, num_heads, qk_rope_dim)) * 0.25, dtype)
    new_kv_c = jnp.asarray(rng.standard_normal((total_q, kv_lora_rank)) * 0.25, dtype)
    new_k_pe = jnp.asarray(rng.standard_normal((total_q, qk_rope_dim)) * 0.25, dtype)
    cache_kv = jnp.asarray(
        rng.standard_normal((total_pages, page_size_per_kv_packing, kv_packing, kv_dim)) * 0.25,
        dtype,
    )
    cache_kv = cache_kv.at[:, :, :, nope_dim + qk_rope_dim :].set(0)

    cu_q_lens = jnp.asarray(np.r_[0, np.cumsum(q_lens)], jnp.int32)
    cu_kv_lens = jnp.asarray(np.r_[0, np.cumsum(aligned_kv_lens)], jnp.int32)
    page_indices = jnp.arange(sum(page_counts), dtype=jnp.int32)
    kv_lens_arr = jnp.asarray(kv_lens, jnp.int32)

    is_decode = all(q == 1 for q in q_lens)
    n_seqs = len(lens)
    distribution = (
        jnp.asarray([n_seqs, n_seqs, n_seqs], jnp.int32)
        if is_decode
        else jnp.asarray([0, 0, n_seqs], jnp.int32)
    )
    return (
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens_arr,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
    )


def extract_valid_kv_tokens(cache_arr, lens, page_size, cu_kv_lens, page_indices, dtype):
    kv_packing = kmod.get_dtype_packing(dtype)
    rows = []
    cu_kv = np.asarray(cu_kv_lens)
    pidx = np.asarray(page_indices)
    for s_i, (_, kv_len) in enumerate(lens):
        start_page = int(cu_kv[s_i]) // page_size
        for pos in range(kv_len):
            p_i = pidx[start_page + pos // page_size]
            in_page = pos % page_size
            w_row, w_col = in_page // kv_packing, in_page % kv_packing
            rows.append(cache_arr[p_i, w_row, w_col])
    return np.stack(rows, axis=0)


def _single_device_mesh():
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_sharded_v2_kernel(num_inputs, **kwargs):
    # Match MLAAttentionBackend: Pallas refs must see per-shard avals, not the
    # enclosing explicit mesh's rank-dependent sharding specs.
    return jax.shard_map(
        lambda *args: kmod.mla_ragged_paged_attention(*args, **kwargs),
        mesh=jax.sharding.get_mesh(),
        in_specs=(P(),) * num_inputs,
        out_specs=(P(), P()),
        check_vma=False,
    )


class TestMLAV2BitExactParity(unittest.TestCase):
    """Bit-exact parity checks (``max_abs_diff == 0``) for ``output``, ``l``,
    ``acc``, and ``updated_cache_kv`` against the pre-#1733 v2 kernel."""

    PARITY_CASES = [
        ("decode_aligned_and_unaligned", [(1, 15), (1, 16), (1, 17), (1, 31)], 16, 8, 2, {}),
        ("decode_single_tail", [(1, 12), (1, 27), (1, 33)], 16, 8, 2, {}),
        ("prefill_pure_and_extend", [(8, 8), (5, 21), (1, 19)], 16, 8, 1, {}),
        ("prefill_multi_block", [(16, 32), (12, 28)], 16, 8, 1, {}),
        (
            "prefill_qkv_scales",
            [(8, 24), (7, 19)],
            16,
            8,
            1,
            {"q_scale": 0.125, "k_scale": 0.25, "v_scale": 0.5},
        ),
        ("prefill_sliding_window", [(12, 32), (8, 25)], 16, 8, 1, {"sliding_window": 16}),
        ("prefill_soft_cap", [(8, 24), (6, 18)], 16, 8, 1, {"soft_cap": 30.0}),
        ("prefill_custom_mask", [(8, 24), (6, 18)], 16, 8, 1, {"mask_value": -1e20}),
        ("decode_2heads_page8_3d_refs", [(1, 7), (1, 8), (1, 9), (1, 15)], 8, 2, 2, {}),
    ]

    def _assert_bitexact_case(self, name, lens, page_size, num_heads, dbs, extra_kw):
        inputs = make_mla_v2_parity_inputs(
            lens, page_size=page_size, num_heads=num_heads, dtype=jnp.bfloat16, seed=42
        )
        common_kw = dict(
            sm_scale=(512 + 64) ** -0.5,
            num_kv_pages_per_block=(1, 1, 1),
            num_queries_per_block=(1, 1, 8),
            decode_batch_size=dbs,
            **extra_kw,
        )
        out_cur, kv_cur, l_cur, acc_cur = run_v2_kernel_with_capture(*inputs, **common_kw)
        out_ref, kv_ref, l_ref, acc_ref = pre1733_reference_mla_v2(*inputs, **common_kw)

        valid_kv_cur = extract_valid_kv_tokens(
            kv_cur, lens, page_size, inputs[8], inputs[6], jnp.bfloat16
        )
        valid_kv_ref = extract_valid_kv_tokens(
            kv_ref, lens, page_size, inputs[8], inputs[6], jnp.bfloat16
        )

        max_diff_out = float(
            np.max(np.abs(out_cur.astype(np.float32) - out_ref.astype(np.float32)))
        )
        max_diff_kv = float(
            np.max(np.abs(valid_kv_cur.astype(np.float32) - valid_kv_ref.astype(np.float32)))
        )
        max_diff_l = max(
            float(np.max(np.abs(a[:, :vr, :] - b))) for a, (vr, b) in zip(l_cur, l_ref)
        )
        max_diff_acc = max(
            float(np.max(np.abs(a[:, :vr, :] - b))) for a, (vr, b) in zip(acc_cur, acc_ref)
        )

        self.assertEqual(max_diff_out, 0.0, f"{name}: output max_abs_diff={max_diff_out}")
        self.assertEqual(max_diff_l, 0.0, f"{name}: l max_abs_diff={max_diff_l}")
        self.assertEqual(max_diff_acc, 0.0, f"{name}: acc max_abs_diff={max_diff_acc}")
        self.assertEqual(max_diff_kv, 0.0, f"{name}: updated_cache_kv max_abs_diff={max_diff_kv}")
        np.testing.assert_array_equal(out_cur, out_ref)
        np.testing.assert_array_equal(valid_kv_cur, valid_kv_ref)
        for a, (vr, b) in zip(l_cur, l_ref):
            np.testing.assert_array_equal(a[:, :vr, :], b)
        for a, (vr, b) in zip(acc_cur, acc_ref):
            np.testing.assert_array_equal(a[:, :vr, :], b)

    def test_v2_kernel_bitexact_parity_output_and_l(self):
        for name, lens, page_size, num_heads, dbs, extra_kw in self.PARITY_CASES:
            with self.subTest(case=name):
                self._assert_bitexact_case(name, lens, page_size, num_heads, dbs, extra_kw)

    TPU_PARITY_CASES = [
        ("decode_tpu_compiled", [(1, 127), (1, 128), (1, 95), (1, 128)], 128, 16, 2, {}),
        (
            "decode_multi_bkv_tail_tpu_compiled",
            [(1, 512), (1, 495), (1, 380), (1, 512), (1, 410)],
            128,
            16,
            2,
            {},
        ),
        ("decode_2heads_tpu_compiled", [(1, 127), (1, 256)], 128, 2, 2, {}),
        ("prefill_tpu_compiled", [(8, 128), (5, 119)], 128, 16, 1, {}),
        ("prefill_multi_bkv_tpu_compiled", [(8, 240)], 128, 16, 1, {}),
        ("prefill_multi_bq_bkv_tpu_compiled", [(24, 512), (16, 490)], 128, 16, 1, {}),
        (
            "prefill_qkv_scales_tpu_compiled",
            [(16, 256), (12, 235)],
            128,
            16,
            1,
            {"q_scale": 0.125, "k_scale": 0.25, "v_scale": 0.5},
        ),
        (
            "prefill_sliding_window_tpu_compiled",
            [(16, 360), (12, 300)],
            128,
            16,
            1,
            {"sliding_window": 192},
        ),
        (
            "prefill_soft_cap_tpu_compiled",
            [(16, 256), (12, 220)],
            128,
            16,
            1,
            {"soft_cap": 30.0},
        ),
    ]

    def test_v2_kernel_traces_under_explicit_mesh(self):
        """Exercise the compiled test's call boundary without requiring a TPU."""
        for name, lens, page_size, num_heads, dbs, extra_kw in self.TPU_PARITY_CASES:
            with self.subTest(case=name), jax.sharding.set_mesh(_single_device_mesh()):
                inputs = make_mla_v2_parity_inputs(
                    lens, page_size=page_size, num_heads=num_heads, dtype=jnp.bfloat16, seed=42
                )
                common_kw = dict(
                    sm_scale=(512 + 64) ** -0.5,
                    num_kv_pages_per_block=(1, 1, 1),
                    num_queries_per_block=(1, 1, 8),
                    decode_batch_size=dbs,
                    **extra_kw,
                )
                run = _make_sharded_v2_kernel(len(inputs), **common_kw)
                jax.jit(run).trace(*inputs)

    @unittest.skipIf(
        jax.default_backend() != "tpu",
        "Requires TPU backend for unpatched Mosaic/MXU compilation and execution.",
    )
    def test_v2_kernel_tpu_compiled_bitexact_parity(self):
        """Runs ``kmod.mla_ragged_paged_attention`` completely unpatched on TPU
        (executing full Mosaic lowering, MXU ``Q @ K^T`` / ``P @ V``, and
        ``pl.reciprocal(l, approx=True)``) and verifies ``max_abs_diff == 0.0``
        against the frozen ``c136c733`` Pallas block reference."""
        for name, lens, page_size, num_heads, dbs, extra_kw in self.TPU_PARITY_CASES:
            with self.subTest(case=name), jax.sharding.set_mesh(_single_device_mesh()):
                inputs = make_mla_v2_parity_inputs(
                    lens, page_size=page_size, num_heads=num_heads, dtype=jnp.bfloat16, seed=42
                )
                common_kw = dict(
                    sm_scale=(512 + 64) ** -0.5,
                    num_kv_pages_per_block=(1, 1, 1),
                    num_queries_per_block=(1, 1, 8),
                    decode_batch_size=dbs,
                    **extra_kw,
                )
                # Read the original cache before invoking the donating kernel.
                out_ref, kv_ref, _, _ = pre1733_reference_mla_v2(
                    *inputs, use_tpu_pallas=True, **common_kw
                )

                run = _make_sharded_v2_kernel(len(inputs), **common_kw)
                out_tpu, kv_tpu = run(*inputs)
                out_tpu_np = np.asarray(out_tpu)
                kv_tpu_np = np.asarray(kv_tpu)

                valid_kv_tpu = extract_valid_kv_tokens(
                    kv_tpu_np, lens, page_size, inputs[8], inputs[6], jnp.bfloat16
                )
                valid_kv_ref = extract_valid_kv_tokens(
                    kv_ref, lens, page_size, inputs[8], inputs[6], jnp.bfloat16
                )

                max_diff_out = float(
                    np.max(np.abs(out_tpu_np.astype(np.float32) - out_ref.astype(np.float32)))
                )
                max_diff_kv = float(
                    np.max(
                        np.abs(valid_kv_tpu.astype(np.float32) - valid_kv_ref.astype(np.float32))
                    )
                )
                self.assertEqual(
                    max_diff_out, 0.0, f"{name}: TPU output max_abs_diff={max_diff_out}"
                )
                self.assertEqual(
                    max_diff_kv,
                    0.0,
                    f"{name}: TPU updated_cache_kv max_abs_diff={max_diff_kv}",
                )
                np.testing.assert_array_equal(out_tpu_np, out_ref)
                np.testing.assert_array_equal(valid_kv_tpu, valid_kv_ref)

    def test_prepare_q_inputs_optimization_barrier_for_all_head_counts(self):
        """Whenever ``head_dim != actual_head_dim`` (e.g. ``q_pe`` with dim 64 -> 128),
        ``prepare_q_inputs`` must emit ``optimization_barrier`` regardless of
        ``num_q_heads``, whereas already-aligned ``ql_nope`` (dim 512) skips both
        ``pad`` and ``optimization_barrier``."""
        for num_q_heads in (2, 4, 8, 16, 64):
            q_pe = jnp.ones((8, num_q_heads, 64), dtype=jnp.bfloat16)
            jaxpr_pe = jax.make_jaxpr(kmod.prepare_q_inputs)(q_pe)
            primitives_pe = [eqn.primitive.name for eqn in jaxpr_pe.jaxpr.eqns]
            self.assertIn(
                "optimization_barrier",
                primitives_pe,
                f"q_pe (num_q_heads={num_q_heads}) must emit optimization_barrier",
            )

            ql_nope = jnp.ones((8, num_q_heads, 512), dtype=jnp.bfloat16)
            jaxpr_nope = jax.make_jaxpr(kmod.prepare_q_inputs)(ql_nope)
            primitives_nope = [eqn.primitive.name for eqn in jaxpr_nope.jaxpr.eqns]
            self.assertNotIn("pad", primitives_nope)
            self.assertNotIn("optimization_barrier", primitives_nope)


if __name__ == "__main__":
    unittest.main()
