"""单 slice ICI P/D 分离（路线 B）。

架构：单 process group（满足 libtpu 全 slice init），device 对半切 P/D sub-mesh，
各 load 一份 TP=half model + KV pool。KV 传输用 full_mesh ppermute(pd,[(0,1)])。

关键 workaround（Stage 2.0 v17c）：multi-host sub-mesh 上 model forward 必须禁
async collective，否则 libtpu lowering bug → BoundsCheck dma.hbm_to_smem：
  LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=disabled \\
    --xla_enable_async_all_reduce=disabled \\
    --xla_enable_async_collective_permute=disabled \\
    --xla_tpu_enable_async_collective_fusion=false"
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)

LIBTPU_DISABLE_ASYNC_COLLECTIVE = (
    "--xla_enable_async_all_gather=disabled "
    "--xla_enable_async_all_reduce=disabled "
    "--xla_enable_async_collective_permute=disabled "
    "--xla_tpu_enable_async_collective_fusion=false"
)

# b/391624260 推荐的 SC Offload 替代。v23 实测：sub-mesh 上能绕过 BoundsCheck，
# 但 decode bs<32 时比全禁略慢（+5~14%，TC↔SC 搬运开销），bs≥64 时持平。
# 声称的 +10~15% 是 full-mesh training 场景，不适用于 sub-mesh inference 小 bs。
LIBTPU_SC_OFFLOAD = (
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false "
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true "
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true "
    "--xla_tpu_use_tc_device_shape_on_sc=true "
    "--xla_sc_disable_megacore_partitioning=true"
)


def install_submesh_patches():
    """multi-host sub-mesh 必需的 monkeypatch（无 local shard 的 process 上的 dtype 推断）。"""
    import sys

    _orig_mafc = jax.make_array_from_callback
    _local_ids = {d.id for d in jax.local_devices()}

    def _patched_mafc(shape, sharding, cb, **kw):
        if "dtype" not in kw and not ({d.id for d in sharding.device_set} & _local_ids):
            try:
                probe = cb(tuple(slice(0, 0) for _ in shape))
                kw["dtype"] = np.asarray(probe).dtype
            except Exception:
                kw["dtype"] = jnp.bfloat16
        return _orig_mafc(shape, sharding, cb, **kw)

    jax.make_array_from_callback = _patched_mafc

    import sgl_jax.srt.utils.jax_utils as _ju

    def _patched_device_array(*data, sharding=None, **kwargs):
        if sharding is None:
            return jax.device_put(*data, device=sharding, **kwargs)

        def _to_dev(arr):
            arr = np.asarray(arr)
            return _orig_mafc(arr.shape, sharding, lambda idx, a=arr: a[idx], dtype=arr.dtype)

        return jax.tree.map(_to_dev, *data)

    _ju.device_array = _patched_device_array
    for _m in list(sys.modules):
        if _m.startswith("sgl_jax") and hasattr(sys.modules[_m], "device_array"):
            sys.modules[_m].device_array = _patched_device_array

    # ModelRunner.get_available_device_memory：原版 distributed=True 取全 64 device
    # min，导致 P/D sub-mesh 互相污染（P load 后 P 侧 6GB 拉低 D 的预算）。改为只
    # 在 self.mesh 覆盖的 process 内取 min（broadcast from sub-mesh 内最小 pid）。
    from jax.experimental.multihost_utils import broadcast_one_to_all

    import sgl_jax.srt.model_executor.model_runner as _mr

    def _submesh_avail(self):
        local = _ju.get_available_device_memory(self.device, distributed=False)
        mesh_pids = {int(d.process_index) for d in self.mesh.devices.flatten()}
        src = min(mesh_pids)
        is_src = jax.process_index() == src
        v = float(broadcast_one_to_all(np.float64(local), is_source=is_src))
        logger.info(
            "[ici_pd] sub-mesh avail: local=%.2fGB src_pid=%d → %.2fGB", local / 1e9, src, v / 1e9
        )
        return v

    _mr.ModelRunner.get_available_device_memory = _submesh_avail


_NPG_BUCKETS = (8, 16, 32, 64)
# 单次 stacked transfer 的 npg 上限：64pg×78L×80KB ≈ 0.39GB 临时 HBM。
# v6e 32GB - weight 25.4GB - D KV 2.5GB ≈ 4GB，但实测 jit executable 等
# 再占 ~2.6GB，实际剩 ~1.3GB。zeros+recv 2×0.39=0.78GB 是上限。
_NPG_BATCH_MAX = 64


def _bucket_npg(n: int) -> int:
    for b in _NPG_BUCKETS:
        if n <= b:
            return b
    return _NPG_BATCH_MAX


def make_pd_meshes(dp_size: int, tp_per_side: int) -> tuple[Mesh, Mesh, Mesh]:
    devs = jax.devices()
    n = len(devs)
    assert n == 2 * tp_per_side, f"expect {2*tp_per_side} devices, got {n}"
    half = n // 2
    tp_axis = tp_per_side // dp_size
    et = (jax.sharding.AxisType.Explicit,)
    full = Mesh(
        np.asarray(devs).reshape(2, dp_size, tp_axis),
        axis_names=("pd", "data", "tensor"),
        axis_types=et * 3,
    )
    p = Mesh(
        np.asarray(devs[:half]).reshape(dp_size, tp_axis),
        axis_names=("data", "tensor"),
        axis_types=et * 2,
    )
    d = Mesh(
        np.asarray(devs[half:]).reshape(dp_size, tp_axis),
        axis_names=("data", "tensor"),
        axis_types=et * 2,
    )
    logger.info("[ici_pd] full=%s p=%s d=%s", full.shape, p.shape, d.shape)
    return full, p, d


class ICIPDKVTransfer:
    """MLA KV pool P→D ppermute 传输（v19c 验证 + stack-78L 优化）。

    v19c 实测：78L per-layer 循环 = 604ms/16pg、1.8s/128pg（host overhead 占主）。
    stacked 模式：78 层 stack 成单 array 一次 ppermute，理论 <50ms。
    """

    def __init__(self, full_mesh: Mesh, p_mesh: Mesh, d_mesh: Mesh, p_pool, d_pool):
        self.full_mesh = full_mesh
        self.p_mesh = p_mesh
        self.d_mesh = d_mesh
        self.p_pool = p_pool
        self.d_pool = d_pool
        self.L = p_pool.layer_num
        self.page_shape = p_pool.kv_buffer[0].shape[1:]
        self.dtype = p_pool.kv_buffer[0].dtype
        self.p_devs = set(full_mesh.devices[0].flatten().tolist())
        self._full_spec = P("pd", "data", None, None, None, None)
        self._full_shard = NamedSharding(full_mesh, self._full_spec)
        self._d_shard = NamedSharding(d_mesh, P("data", None, None, None, None))
        # 预创建 D 侧 zeros（每 local D device 一份，按 npg bucket 缓存）
        self._zeros_cache: dict[int, dict] = {}
        self._pp_cache: dict[int, callable] = {}

    def _zeros_for(self, npg: int) -> dict:
        # 只 cache 单个 bucket（最近用的），避免多 bucket 累积占 HBM
        if npg not in self._zeros_cache:
            for old in list(self._zeros_cache):
                for a in self._zeros_cache.pop(old).values():
                    a.delete()
            z = np.zeros((1, self.L, npg) + self.page_shape, self.dtype)
            self._zeros_cache[npg] = {
                dv: jax.device_put(z, dv)
                for dv in self._full_shard.addressable_devices
                if dv not in self.p_devs
            }
        return self._zeros_cache[npg]

    def _pp_for(self, npg: int):
        if npg not in self._pp_cache:
            spec = self._full_spec

            def _body(a):
                return jax.lax.ppermute(a, "pd", [(0, 1)])

            try:
                fn = jax.jit(
                    shard_map(
                        _body, mesh=self.full_mesh, in_specs=spec, out_specs=spec, check_rep=False
                    )
                )
            except TypeError:
                fn = jax.jit(shard_map(_body, mesh=self.full_mesh, in_specs=spec, out_specs=spec))
            self._pp_cache[npg] = fn
        return self._pp_cache[npg]

    @staticmethod
    @jax.jit
    def _gather_stack(bufs: tuple, idx):
        return jnp.stack([b.at[idx].get(out_sharding=P("data", None, None, None)) for b in bufs])

    _scatter_jit = staticmethod(
        jax.jit(
            lambda bufs, idx, stacked: tuple(
                b.at[idx].set(stacked[i], out_sharding=P("data", None, None, None))
                for i, b in enumerate(bufs)
            ),
            donate_argnums=(0,),
        )
    )

    def transfer(self, p_page_indices, d_page_indices) -> None:
        """P pool[p_pages] 全 78 层 → D pool[d_pages]（stacked ppermute，npg>128 分批）。"""
        p_page_indices = np.asarray(p_page_indices, np.int32)
        d_page_indices = np.asarray(d_page_indices, np.int32)
        n = len(p_page_indices)
        assert n == len(d_page_indices)
        for i in range(0, n, _NPG_BATCH_MAX):
            self._transfer_batch(
                p_page_indices[i : i + _NPG_BATCH_MAX],
                d_page_indices[i : i + _NPG_BATCH_MAX],
            )

    def _transfer_batch(self, p_page_indices, d_page_indices) -> None:
        npg = len(p_page_indices)
        if npg == 0:
            return
        bucket = _bucket_npg(npg)
        if bucket > npg:
            pad = np.zeros(bucket - npg, np.int32)
            p_page_indices = np.concatenate([np.asarray(p_page_indices, np.int32), pad])
            d_page_indices = np.concatenate([np.asarray(d_page_indices, np.int32), pad])
            npg = bucket
        p_idx = jax.device_put(
            np.asarray(p_page_indices, np.int32), NamedSharding(self.p_mesh, P(None))
        )
        d_idx = jax.device_put(
            np.asarray(d_page_indices, np.int32), NamedSharding(self.d_mesh, P(None))
        )
        # 1) gather + stack 78L on p_mesh → (L, npg, *page_shape)
        with jax.set_mesh(self.p_mesh):
            p_stacked = self._gather_stack(tuple(self.p_pool.kv_buffer), p_idx)
        # 2) embed 到 full_mesh (2, L, npg, *page_shape)
        zeros = self._zeros_for(npg)
        p_map = {s.device: s.data for s in p_stacked.addressable_shards}
        shards = [
            (p_map[dv][None] if dv in self.p_devs else zeros[dv])
            for dv in self._full_shard.addressable_devices
        ]
        staged = jax.make_array_from_single_device_arrays(
            (2, self.L, npg) + self.page_shape, self._full_shard, shards, dtype=self.dtype
        )
        # 3) ppermute pd 0→1
        recv = self._pp_for(npg)(staged)
        # 4) extract pd=1 → d_mesh (L, npg, *page_shape)
        d_loc = [s.data[0] for s in recv.addressable_shards if s.device not in self.p_devs]
        d_stacked = jax.make_array_from_single_device_arrays(
            (self.L, npg) + self.page_shape, self._d_shard, d_loc, dtype=self.dtype
        )
        # 5) unstack + scatter 进 D pool（donate 旧 buffer 避免 2× HBM）
        with jax.set_mesh(self.d_mesh):
            new_bufs = self._scatter_jit(tuple(self.d_pool.kv_buffer), d_idx, d_stacked)
        self.d_pool.kv_buffer = list(new_bufs)


def slots_to_ordered_pages(slots: np.ndarray, page_size: int) -> np.ndarray:
    """token-level slot 序列 → page 序列（保持首次出现顺序，去重）。

    PD 禁 radix，单 req 的 slot 总是按 page 整块连续，故等价于 slots[::page_size]//page_size，
    但 chunked req 多次 alloc 可能不从 page 边界续（part1 填尾），用通用去重保险。
    """
    pages = np.asarray(slots, np.int64) // page_size
    _, first_idx = np.unique(pages, return_index=True)
    return pages[np.sort(first_idx)].astype(np.int32)


def migrate_reqs_p_to_d(
    reqs: list,
    page_size: int,
    p_r2t,
    p_alloc,
    d_r2t,
    d_alloc,
    kv_transfer: ICIPDKVTransfer,
) -> None:
    """把已完成 prefill 的 reqs 从 P pool 迁移到 D pool（含 KV ppermute）。

    每个 req：① 读 P slots → ② D alloc 等量 page-aligned slot + req_pool_idx →
    ③ 改写 req.req_pool_idx/prefix_indices/kv_committed_len → ④ 收集 page 对 →
    ⑤ free P。最后批量 transfer（一次 ppermute）。
    """
    if not reqs:
        return
    p_pages_all, d_pages_all = [], []
    for r in reqs:
        seq_len = len(r.fill_ids)
        p_idx_old = r.req_pool_idx
        p_slots = p_r2t.req_to_token[p_idx_old, :seq_len].copy()
        p_pages = slots_to_ordered_pages(p_slots, page_size)
        n_pages = len(p_pages)

        d_slots = d_alloc.alloc(n_pages * page_size, dp_rank=r.dp_rank or 0)
        if d_slots is None:
            raise RuntimeError(
                f"[ici_pd] D pool OOM during migrate: need {n_pages} pages, "
                f"avail {d_alloc.available_size()}"
            )
        d_pages = slots_to_ordered_pages(d_slots, page_size)
        # 按 page 内 offset 重映射：token i 在 P page k 的 offset o → D page k 的 offset o
        p_slots64 = np.asarray(p_slots, np.int64)
        offsets = p_slots64 % page_size
        page_pos = {int(pg): k for k, pg in enumerate(p_pages)}
        tok_page_k = np.array([page_pos[int(s)] for s in p_slots64 // page_size], np.int64)
        d_slot_per_tok = (d_pages[tok_page_k].astype(np.int64) * page_size + offsets).astype(
            np.int32
        )

        p_alloc.free(p_slots, dp_rank=r.dp_rank or 0)
        p_r2t.free_slots.append(p_idx_old)
        r.req_pool_idx = None
        d_r2t.alloc([r])  # 赋新 req_pool_idx
        d_r2t.req_to_token[r.req_pool_idx, :seq_len] = d_slot_per_tok
        r.prefix_indices = d_slot_per_tok
        r.last_node = None
        r.cache_protected_len = 0
        r.kv_committed_len = seq_len
        r.kv_allocated_len = seq_len

        p_pages_all.append(p_pages)
        d_pages_all.append(d_pages)

    kv_transfer.transfer(np.concatenate(p_pages_all), np.concatenate(d_pages_all))
