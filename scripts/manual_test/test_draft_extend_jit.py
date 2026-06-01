#!/usr/bin/env python3
"""Pod test: load real model + captured data, compare original vs fused JIT.

Usage (on perf-16 pod):
  cd /tmp/sglang-jax/python && uv run python ../scripts/manual_test/test_draft_extend_jit.py \
    --capture-path /models/capature_niu/draft_extend_0.npz \
    --model-path /models/model_scope/XiaomiMiMo/MiMo-V2-Flash
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_server_args(model_path: str):
    """Construct ServerArgs matching the launch config."""
    from sgl_jax.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        trust_remote_code=True,
        speculative_algorithm="NEXTN",
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        speculative_draft_model_path=model_path,
        tp_size=16,
        dp_size=4,
        ep_size=16,
        moe_backend="epmoe",
        page_size=64,
        context_length=4096,
        max_prefill_tokens=4096,
        dtype="bfloat16",
        mem_fraction_static=0.7,
        max_total_tokens=8192,
        swa_full_tokens_ratio=0.15,
        max_running_requests=32,
        attention_backend="fa",
        disable_overlap_schedule=True,
        host="0.0.0.0",
        port=30271,
        nnodes=4,
        dist_init_addr="perf-16-0.niu-mimo-v16-headless-svc:5000",
    )
    return args


def load_capture(path: str) -> dict:
    """Load captured .npz."""
    data = dict(np.load(path, allow_pickle=True))
    # Fix bfloat16 stored as void bytes
    for k, v in data.items():
        if v.dtype.kind == "V" and v.dtype.itemsize == 2:
            data[k] = v.view(jnp.bfloat16)
    return data


def build_mwb_from_capture(cap: dict, draft_worker):
    """Reconstruct ModelWorkerBatch from captured data."""
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    dp_size = int(cap["mwb_dp_size"])
    per_dp_bs = int(cap["mwb_per_dp_bs_size"])
    real_bs = int(cap["mwb_real_bs"])
    bs = len(cap["mwb_seq_lens"])

    mwb = ModelWorkerBatch(
        bid=int(cap["mwb_bid"]),
        forward_mode=ForwardMode(int(cap["mwb_forward_mode"])),
        input_ids=cap["mwb_input_ids"],
        real_input_ids_len=len(cap["mwb_input_ids"]),
        seq_lens=cap["mwb_seq_lens"],
        out_cache_loc=cap["mwb_out_cache_loc"],
        req_pool_indices=cap["mwb_req_pool_indices"],
        sampling_info=None,
        positions=cap["mwb_positions"],
        cache_loc=cap["mwb_cache_loc"],
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_seq_lens=cap["mwb_extend_seq_lens"],
        extend_prefix_lens=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=cap["mwb_logits_indices"],
        real_bs=real_bs,
        real_bs_per_dp=[real_bs // dp_size] * dp_size,
        logits_indices_selector=cap["mwb_logits_indices_selector"],
        dp_size=dp_size,
        per_dp_bs_size=per_dp_bs,
    )

    si = EagleDraftInput(
        hidden_states=cap["si_hidden_states"],
        accept_length=cap.get("si_accept_length"),
        allocate_lens=cap.get("si_allocate_lens"),
    )
    si.capture_hidden_mode = CaptureHiddenMode.FULL
    mwb.spec_info_padded = si
    mwb.capture_hidden_mode = CaptureHiddenMode.FULL
    mwb.speculative_eagle_topk = draft_worker.topk
    mwb.speculative_num_steps = draft_worker.speculative_num_steps
    mwb.speculative_num_draft_tokens = draft_worker.speculative_num_draft_tokens

    return mwb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-path", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    # JAX distributed init MUST come before any JAX calls
    import os

    node_rank = int(os.environ.get("TPU_WORKER_ID", "0"))
    logger.info("Initializing JAX distributed (node_rank=%d)...", node_rank)
    jax.distributed.initialize(
        "perf-16-0.niu-mimo-v16-headless-svc:5000",
        num_processes=4,
        process_id=node_rank,
    )
    logger.info("JAX distributed initialized. devices=%d", jax.device_count())

    logger.info("Loading capture from %s", args.capture_path)
    cap = load_capture(args.capture_path)
    for k, v in sorted(cap.items()):
        if hasattr(v, "shape"):
            logger.info("  %s: shape=%s dtype=%s", k, v.shape, v.dtype)

    logger.info("Initializing draft models only (no target model)...")
    server_args = make_server_args(args.model_path)
    server_args.max_num_reqs = 32
    server_args.max_recurrent_state_size = None
    cache_loc_max = int(cap.get("mwb_cache_loc", np.array([0])).max()) + 1
    page_size = server_args.page_size
    min_tokens = ((cache_loc_max + page_size - 1) // page_size) * page_size
    server_args.draft_runner_cache_size = max(min_tokens, 2048)

    from sgl_jax.srt.managers.tp_worker import ModelWorker
    from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
    from sgl_jax.srt.speculative.multi_layer_draft_worker import (
        MultiLayerDraftWorker,
        _server_args_with_mtp_layer,
    )
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    mesh = create_device_mesh(
        ici_parallelism=[server_args.dp_size, server_args.tp_size // server_args.dp_size],
        dcn_parallelism=[1, 1],
    )
    logger.info("Mesh created: %s", mesh)

    # Create shared req_to_token_pool (lightweight, no target model needed)
    req_to_token_pool = ReqToTokenPool(
        size=64,
        max_context_len=server_args.context_length + 4,
        dtype=np.int32,
    )

    # Create draft workers directly (each is a ModelWorker with is_draft_worker=True)
    num_mtp_layers = server_args.speculative_num_steps
    workers = []
    for i in range(num_mtp_layers):
        sa = _server_args_with_mtp_layer(server_args, i)
        w = ModelWorker(sa, mesh, req_to_token_pool=req_to_token_pool, is_draft_worker=True)
        workers.append(w)
        logger.info("Draft layer %d loaded", i)

    # Load embed + lm_head from target model safetensors (not the full model)
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from safetensors import safe_open

    model_dir = args.model_path
    logger.info("Loading embed + lm_head from target safetensors...")
    with safe_open(f"{model_dir}/model_embedding.safetensors", framework="numpy") as sf:
        embed_np = sf.get_tensor("model.embed_tokens.weight")
    with safe_open(f"{model_dir}/model_final.safetensors", framework="numpy") as sf:
        lm_head_np = sf.get_tensor("lm_head.weight")
    embed_device = jax.device_put(
        embed_np.astype(jnp.bfloat16),
        NamedSharding(mesh, P("tensor", None)),
    )
    lm_head_device = jax.device_put(
        lm_head_np.astype(jnp.bfloat16),
        NamedSharding(mesh, P("tensor", None)),
    )
    for w in workers:
        m = w.model_runner.model
        m.model.embed_tokens.embedding.value = embed_device
        m.lm_head.embedding.value = lm_head_device
    logger.info(
        "embed (%s) + lm_head (%s) loaded and shared to all draft layers",
        embed_np.shape,
        lm_head_np.shape,
    )

    # Initialize JIT for each layer
    for w in workers:
        w.model_runner.initialize_jit()

    # Build a minimal draft_worker-like object for testing
    class DraftWorkerShim:
        @property
        def draft_model_runner(self):
            return self._workers[0].model_runner

    draft_worker = DraftWorkerShim()
    draft_worker._workers = workers
    draft_worker._worker = workers[0]
    draft_worker.topk = server_args.speculative_eagle_topk
    draft_worker.speculative_num_steps = server_args.speculative_num_steps
    draft_worker.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
    draft_worker.page_size = server_args.page_size
    draft_worker.mesh = mesh
    draft_worker.hot_token_ids = None

    # Bind methods from MultiLayerDraftWorker
    import types

    draft_worker.draft_extend_for_decode = types.MethodType(
        MultiLayerDraftWorker.draft_extend_for_decode, draft_worker
    )
    draft_worker.runner = types.MethodType(MultiLayerDraftWorker.runner, draft_worker)
    draft_worker._rotate_ids = MultiLayerDraftWorker._rotate_ids

    # Set precompile paddings (needed by some code paths)
    draft_worker.precompile_bs_paddings = [4, 8, 16, 32]
    draft_worker.precompile_token_paddings = [256, 512, 1024, 2048, 4096]
    draft_worker.precompile_cache_loc_paddings = [
        p * server_args.speculative_num_draft_tokens for p in draft_worker.precompile_bs_paddings
    ]

    EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
        server_args.speculative_num_steps * server_args.speculative_eagle_topk,
        server_args.speculative_num_draft_tokens,
    )

    logger.info("Draft workers initialized (%d layers). No target model loaded.", num_mtp_layers)

    logger.info("Model initialized. Building test batch from capture...")

    # Build batch_output from captured data
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    mwb = build_mwb_from_capture(cap, draft_worker)

    # Reconstruct batch_output — use numpy arrays (not device arrays)
    # draft_extend_for_decode does device_get on accept_lens internally
    # target_hidden needs to be a device array (it's batch_output.logits_output.hidden_states)
    target_hidden = jax.device_put(cap["si_hidden_states"], NamedSharding(mesh, P()))
    accept_lens_np = cap["accept_host"].astype(np.int32)
    verified_id_np = cap["bo_verified_id"].astype(np.int32)

    logits_output = LogitsProcessorOutput(
        next_token_logits=None,
        hidden_states=target_hidden,
    )
    next_draft_input = EagleDraftInput(
        hidden_states=None,
        verified_id=verified_id_np,
        allocate_lens=cap["bo_allocate_lens"],
    )
    batch_output = GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=None,
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
        bid=int(cap["mwb_bid"]),
        cache_miss_count=0,
        next_draft_input=next_draft_input,
        allocate_lens=cap["bo_allocate_lens"],
        accept_lens=accept_lens_np,
    )

    # --- Single-process test: run golden, then eager fused, then jit fused ---
    import types

    from sgl_jax.srt.speculative.draft_extend_fused import (
        _build_fused_draft_extend_eager,
        _build_fused_draft_extend_jit,
        draft_extend_for_decode_fused,
    )
    from sgl_jax.srt.speculative.multi_layer_draft_worker import MultiLayerDraftWorker

    draft_worker._draft_extend_for_decode_original = types.MethodType(
        MultiLayerDraftWorker._draft_extend_for_decode_original, draft_worker
    )

    def snapshot_kv(wkrs):
        """Snapshot addressable KV buffer shards from each draft layer."""
        snaps = []
        for w in wkrs:
            pool = w.model_runner.memory_pools.token_to_kv_pool
            layer_snaps = []
            for buf in pool.kv_buffer:
                local_shards = [np.asarray(s.data) for s in buf.addressable_shards]
                layer_snaps.append(local_shards)
            snaps.append(layer_snaps)
        return snaps

    def restore_kv(wkrs, snaps):
        """Restore KV buffers from a snapshot (re-put on device with original sharding)."""
        for w, layer_snaps in zip(wkrs, snaps):
            pool = w.model_runner.memory_pools.token_to_kv_pool
            for i, shard_list in enumerate(layer_snaps):
                sharding = pool.kv_buffer[i].sharding
                full = np.asarray(
                    jax.experimental.multihost_utils.process_allgather(
                        pool.kv_buffer[i], tiled=True
                    )
                )
                # Reconstruct from local shards — just re-put the original buffer
                pool.kv_buffer[i] = jax.device_put(
                    jax.make_array_from_single_device_arrays(
                        pool.kv_buffer[i].shape,
                        sharding,
                        [
                            jax.device_put(s, d)
                            for s, d in zip(
                                shard_list,
                                [sh.device for sh in pool.kv_buffer[i].addressable_shards],
                            )
                        ],
                    ),
                    sharding,
                )

    # Snapshot KV state before any forward (baseline)
    logger.info("Snapshotting KV pools before forward...")
    kv_before = snapshot_kv(workers)

    # --- Run GOLDEN (original per-layer path) ---
    logger.info("=== Running GOLDEN (original) ===")
    mwb_g = build_mwb_from_capture(cap, draft_worker)
    mwb_g.input_ids = (
        mwb_g.input_ids.copy()
        if isinstance(mwb_g.input_ids, np.ndarray)
        else np.asarray(mwb_g.input_ids).copy()
    )
    bo_g = copy.copy(batch_output)
    bo_g.next_draft_input = EagleDraftInput(
        hidden_states=None,
        verified_id=verified_id_np.copy(),
        allocate_lens=cap["bo_allocate_lens"].copy(),
    )
    bo_g.allocate_lens = cap["bo_allocate_lens"].copy()
    bo_g.accept_lens = accept_lens_np.copy()

    t0 = time.time()
    draft_worker._draft_extend_for_decode_original(mwb_g, bo_g)
    t1 = time.time()
    logger.info("Golden took %.3f s", t1 - t0)

    golden_outputs = {
        k: np.asarray(getattr(bo_g.next_draft_input, k))
        for k in ["hidden_states", "topk_p", "topk_index", "verified_id"]
    }
    for k, v in golden_outputs.items():
        logger.info("  golden %s: %s %s", k, v.shape, v.dtype)

    kv_after_golden = snapshot_kv(workers)

    def compare_outputs(label, bo_test, kv_test):
        nonlocal ok
        logger.info("=== Comparing %s outputs ===", label)
        for k in ["hidden_states", "topk_p", "topk_index", "verified_id"]:
            a = golden_outputs[k].astype(np.float32)
            b = np.asarray(getattr(bo_test.next_draft_input, k)).astype(np.float32)
            if a.shape != b.shape:
                logger.error("[%s] FAIL %s: shape %s vs %s", label, k, a.shape, b.shape)
                ok = False
                continue
            d = np.abs(a - b)
            if np.issubdtype(golden_outputs[k].dtype, np.integer):
                m = np.array_equal(a, b)
            else:
                m = np.allclose(a, b, atol=1e-3, rtol=1e-2)
            if m:
                logger.info("[%s] PASS %s: max_diff=%.6f", label, k, d.max())
            else:
                logger.error("[%s] FAIL %s: max_diff=%.6f mean=%.6f", label, k, d.max(), d.mean())
                ok = False

        logger.info("=== Comparing %s KV pools ===", label)
        for layer_i in range(len(workers)):
            for buf_j in range(len(kv_after_golden[layer_i])):
                g_shards = kv_after_golden[layer_i][buf_j]
                f_shards = kv_test[layer_i][buf_j]
                for shard_k, (gs, fs) in enumerate(zip(g_shards, f_shards)):
                    d = np.abs(gs.astype(np.float32) - fs.astype(np.float32))
                    maxd = d.max()
                    if maxd >= 1e-3:
                        logger.error(
                            "[%s] FAIL KV layer%d buf%d shard%d: max_diff=%.6f n_mismatch=%d/%d",
                            label,
                            layer_i,
                            buf_j,
                            shard_k,
                            maxd,
                            (d > 1e-3).sum(),
                            d.size,
                        )
                        ok = False
                    elif shard_k == 0:
                        logger.info(
                            "[%s] PASS KV layer%d buf%d (all shards): max_diff=%.6f",
                            label,
                            layer_i,
                            buf_j,
                            maxd,
                        )

    # === EAGER FUSED (no jit) ===
    logger.info("Restoring KV pools to pre-forward state...")
    restore_kv(workers, kv_before)

    logger.info("=== Running EAGER FUSED (no jit) ===")
    mwb_e = build_mwb_from_capture(cap, draft_worker)
    bo_e = copy.copy(batch_output)
    bo_e.next_draft_input = EagleDraftInput(
        hidden_states=None,
        verified_id=verified_id_np.copy(),
        allocate_lens=cap["bo_allocate_lens"].copy(),
    )
    bo_e.allocate_lens = cap["bo_allocate_lens"].copy()
    bo_e.accept_lens = accept_lens_np.copy()

    # Call fused logic directly without jit wrapping
    eager_fn = _build_fused_draft_extend_eager(
        num_layers=draft_worker.speculative_num_steps,
        topk=draft_worker.topk,
    )
    mr0 = draft_worker._workers[0].model_runner

    # Prepare same way as draft_extend_for_decode_fused host prep
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch as FB

    sel_e = np.asarray(mwb_e.logits_indices_selector)
    accept_e = bo_e.accept_lens.copy()
    sel_pos_e = np.clip(accept_e - 1, 0, None).astype(np.int64)
    mwb_e.input_ids = (
        mwb_e.input_ids.copy()
        if isinstance(mwb_e.input_ids, np.ndarray)
        else np.asarray(mwb_e.input_ids).copy()
    )

    all_fbs_e, all_pools_e, all_leaves_e = [], [], []
    for w in draft_worker._workers:
        mr = w.model_runner
        mwb_e.spec_info_padded.hidden_states = target_hidden
        mr.attn_backend.forward_metadata = mr.attn_backend.get_eagle_forward_metadata(mwb_e)
        fb_e = FB.init_new(mwb_e, mr)
        fb_e.bid = int(cap["mwb_bid"])
        all_fbs_e.append(fb_e)
        all_pools_e.append(mr.memory_pools)
        all_leaves_e.append(tuple(mr.model_state_leaves))

    sel_pos_dev_e = jax.device_put(sel_pos_e, NamedSharding(draft_worker.mesh, P("data")))
    lm_e = LogitsMetadata.from_model_worker_batch(mwb_e, draft_worker.mesh)

    t0 = time.time()
    with jax.set_mesh(draft_worker.mesh):
        l0h_e, tp_e, ti_e, pup_e = eager_fn(
            mr0._model_def,
            mr0._model_state_def,
            tuple(all_leaves_e),
            tuple(all_fbs_e),
            tuple(all_pools_e),
            lm_e,
            target_hidden,
            sel_pos_dev_e,
            num_layers=draft_worker.speculative_num_steps,
        )
    t1 = time.time()
    logger.info("Eager fused took %.3f s", t1 - t0)

    # Post-process same as draft_extend_for_decode_fused
    for i, w in enumerate(draft_worker._workers):
        w.model_runner.memory_pools.replace_all(pup_e[i])

    select_index_e = sel_e * (draft_worker.speculative_num_steps + 1) + accept_e[sel_e] - 1
    bo_e.next_draft_input.hidden_states = np.asarray(l0h_e)[select_index_e]
    bo_e.next_draft_input.topk_p = np.asarray(tp_e)[sel_e]
    bo_e.next_draft_input.topk_index = np.asarray(ti_e)[sel_e]
    bo_e.next_draft_input.verified_id = np.asarray(batch_output.next_draft_input.verified_id)[
        select_index_e
    ]
    kv_after_eager = snapshot_kv(workers)

    ok = True
    compare_outputs("EAGER", bo_e, kv_after_eager)

    if ok:
        logger.info("EAGER FUSED ALL PASS — logic is correct without jit")
    else:
        logger.error("EAGER FUSED FAIL — logic bug exists even without jit")

    # === JIT FUSED ===
    logger.info("Restoring KV pools to pre-forward state...")
    restore_kv(workers, kv_before)

    logger.info("=== Running JIT FUSED ===")
    mwb_f = build_mwb_from_capture(cap, draft_worker)
    bo_f = copy.copy(batch_output)
    bo_f.next_draft_input = EagleDraftInput(
        hidden_states=None,
        verified_id=verified_id_np.copy(),
        allocate_lens=cap["bo_allocate_lens"].copy(),
    )
    bo_f.allocate_lens = cap["bo_allocate_lens"].copy()
    bo_f.accept_lens = accept_lens_np.copy()

    t0 = time.time()
    draft_extend_for_decode_fused(draft_worker, mwb_f, bo_f)
    t1 = time.time()
    logger.info("Jit fused took %.3f s", t1 - t0)

    kv_after_jit = snapshot_kv(workers)

    ok_jit = True
    ok = True
    compare_outputs("JIT", bo_f, kv_after_jit)
    ok_jit = ok

    if ok_jit:
        logger.info("JIT FUSED ALL PASS!")
    else:
        logger.error("JIT FUSED FAIL — jit introduces errors")

    logger.info("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        try:
            jax.distributed.shutdown()
        except Exception:
            pass
