"""Real C/M/A/B runner binding and scheduler lifecycle, without TPU assumptions."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.mem_cache.chunk_cache import DeepseekV4ChunkCache
from sgl_jax.srt.mem_cache.common import release_kv_cache
from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.srt.model_executor.deepseek_v4_runtime import (
    precompile_capacity_variants,
    reclaim_batch_swa,
    validate_pool_updates,
    validate_runtime_config,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.jax_utils import packed_device_array


@pytest.fixture(scope="module", params=[("jit", 128), ("aot", 128), ("jit", 256), ("aot", 256)])
def runner(tmp_path_factory, request):
    execution, page_size = request.param
    path = tmp_path_factory.mktemp("v4-dummy")
    cfg = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        vocab_size=32,
        hidden_size=128,
        num_hidden_layers=3,
        compress_ratios=[0, 4, 128],
        num_attention_heads=2,
        head_dim=128,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=1,
        moe_intermediate_size=128,
        index_n_heads=2,
        index_head_dim=128,
        max_position_embeddings=256,
    )
    cfg.save_pretrained(path)
    args = ServerArgs(
        model_path=str(path),
        device="cpu",
        load_format="dummy",
        dtype="bfloat16",
        disable_radix_cache=True,
        max_running_requests=2,
        max_total_tokens=2048,
        page_size=page_size,
        max_prefill_tokens=256,
        chunked_prefill_size=page_size,
        precompile_bs_paddings=[2],
        precompile_token_paddings=[128, 256],
        precompile_num_threads=2 if execution == "jit" else 1,
    )
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    config = ModelConfig(model_path=str(path), hf_config=cfg, dtype="bfloat16")
    with (
        patch.dict(os.environ, {"DSV4_FUSED_WO_A": "0"}),
        patch("sgl_jax.srt.model_executor.aot_dispatch._ENV", "1" if execution == "aot" else "0"),
        patch.object(ModelRunner, "get_available_device_memory", return_value=64 << 20),
    ):
        with jax.set_mesh(mesh):
            model_runner = ModelRunner(config, 0.8, 1, 1, args, mesh)
        yield model_runner


def manager(runner):
    return CompilationManager(
        runner.server_args,
        2,
        256,
        1,
        1,
        runner.page_size,
        256,
        32,
        max_total_num_tokens=runner.max_total_num_tokens,
        attn_backend=runner.attn_backend,
    )


def new_request(runner):
    req = SimpleNamespace(
        req_pool_idx=None,
        is_chunked=0,
        kv_committed_len=0,
        kv_allocated_len=0,
        dp_rank=0,
        swa_evicted_seqlen=0,
        is_retracted=False,
        finished=lambda: False,
    )
    assert runner.req_to_token_pool.alloc([req]) is not None
    return req


def step_batch(runner, req, start, end, *, decode=False):
    pool, allocator = runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    last = int(pool.req_to_token[req.req_pool_idx, start - 1]) if start else -1
    locations = allocator.alloc_extend([start], [end], [last], end - start)
    assert locations is not None
    pool.write((req.req_pool_idx, slice(start, end)), locations)
    req.kv_committed_len = req.kv_allocated_len = end
    n = end - start
    tokens = 2 if decode else 128 if n <= 128 else 256
    batch = manager(runner)._make_dummy_batch(
        2,
        tokens,
        ForwardMode.DECODE if decode else ForwardMode.EXTEND,
        512,
        dp_size=1,
        per_dp_bs_size=2,
    )
    batch.input_ids = np.r_[np.arange(start, end) % 31 + 1, np.zeros(tokens - n)].astype(np.int32)
    batch.seq_lens = np.array([end, 0], np.int32)
    batch.req_pool_indices = np.array([req.req_pool_idx, runner.req_to_token_pool.size], np.int32)
    batch.positions = np.r_[np.arange(start, end), np.zeros(tokens - n)].astype(np.int32)
    batch.out_cache_loc = np.r_[locations, np.full(tokens - n, -1)].astype(np.int32)
    batch.cache_loc = np.zeros(512, np.int32)
    if not decode:
        batch.extend_seq_lens = np.array([n, 0], np.int32)
        batch.extend_prefix_lens = np.array([start, 0], np.int32)
        batch.logits_indices = np.array([n - 1, 0], np.int32)
    return batch


def forward(runner, batch):
    fb = batch.forward_batch or ForwardBatch.init_new(batch, runner)
    lm = LogitsMetadata.from_model_worker_batch(batch, runner.mesh)
    result = runner.forward(fb, lm)
    from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

    sm = SamplingMetadata.from_model_worker_batch(
        batch, 0, runner.mesh, runner.model_config.vocab_size
    )
    jax.block_until_ready(runner.sample(result[0], sm))
    return result


def test_real_runner_warmup_chunked_decode_and_recycled_state(runner):
    """Warmed dummies preserve pools; real chunks reuse signatures and both owners."""
    import jax._src.test_util as jtu

    assert runner.is_hybrid and not runner.use_mla_backend
    assert runner.attn_backend.resources_bound
    before = [np.asarray(x).copy() for x in jax.tree.leaves(runner.memory_pools)]
    cm = manager(runner)
    cm.precompile_all(lambda batch, **_: forward(runner, batch), runner, runner.mesh)
    for actual, expected in zip(jax.tree.leaves(runner.memory_pools), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    cache = DeepseekV4ChunkCache(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator, runner.page_size, 128
    )
    req = new_request(runner)
    for start, end, decode in ((0, 128, False), (128, 130, False), (130, 131, True)):
        batch = step_batch(runner, req, start, end, decode=decode)
        old_kv = runner.memory_pools.token_to_kv_pool.get_swa_buffer(0)
        with (
            jtu.count_jit_and_pmap_lowerings() as compiles,
            patch(
                "jax._src.compiler.backend_compile_and_load",
                side_effect=AssertionError("backend compilation after warmup"),
            ),
        ):
            output, _, _ = forward(runner, batch)
            jax.block_until_ready(output)
        assert compiles() == 0
        assert bool(jnp.all(jnp.isfinite(output.next_token_logits)))
        assert old_kv.is_deleted()
        assert runner.memory_pools.token_to_kv_pool.get_swa_buffer(0) is not old_kv
    used_slot = req.req_pool_idx
    release_kv_cache(req, cache)
    assert req.req_pool_idx is None and req.kv_allocated_len == 0
    assert runner.token_to_kv_pool_allocator.full_available_size() == runner.max_total_num_tokens
    assert runner.token_to_kv_pool_allocator.swa_available_size() == runner.swa_max_total_num_tokens
    other = new_request(runner)
    req = new_request(runner)
    assert req.req_pool_idx == used_slot
    batch = step_batch(runner, req, 0, 4)
    md = runner.get_attention_metadata(batch)
    assert bool(np.asarray(md.resolve()[0].state_init_mask)[0])
    jax.block_until_ready(forward(runner, batch))
    release_kv_cache(req, cache)
    release_kv_cache(other, cache)


def test_update_rejected_before_donation(runner):
    pools = runner.memory_pools
    with pytest.raises(ValueError, match="owner keys"):
        validate_pool_updates(pools, {"token_to_kv_pool": {}})
    assert not pools.token_to_kv_pool.get_swa_buffer(0).is_deleted()


@pytest.mark.parametrize("return_hidden_states", [False, True])
def test_overlap_reclaims_completed_snapshot_not_next_length(runner, return_hidden_states):
    from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo

    cache = DeepseekV4ChunkCache(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator, runner.page_size, 128
    )
    req = new_request(runner)
    completed_len = 254
    step_batch(runner, req, 0, completed_len)
    batch = ScheduleBatch(
        reqs_info=[ScheduleReqsInfo(reqs=[req], seq_lens=np.array([completed_len]))],
        return_hidden_states=return_hidden_states,
    )
    submitted = batch.copy()
    # With page size 128, the next decode crosses the SWA reclamation boundary.
    step_batch(runner, req, completed_len, completed_len + 1, decode=True)
    batch.reqs_info[0].seq_lens[:] += 1
    assert submitted.reqs_info[0].seq_lens.tolist() == [completed_len]
    scheduler_batch = SimpleNamespace(tree_cache=cache, is_hybrid=True)
    ScheduleBatch.maybe_evict_swa(scheduler_batch)
    reclaim_batch_swa(submitted, cache)
    assert req.swa_evicted_seqlen == 0
    assert req.kv_committed_len == completed_len + 1
    # The completed forward still needs the last token of the first page.
    last_kept = min(completed_len - 1, runner.page_size - 1)
    loc = runner.req_to_token_pool.req_to_token[req.req_pool_idx, last_kept]
    assert runner.token_to_kv_pool_allocator.full_to_swa_index_mapping[loc] != 0
    release_kv_cache(req, cache)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize(
    "phase",
    ["prefill", "chunk", "mixed", "decode", "finish-prefill", "finish-decode", "abort-chunk"],
)
def test_completed_result_reclaims_and_releases_real_batch(runner, overlap, phase):
    """Exercise real batch copies, output processors and C's lifecycle helpers."""
    from unittest.mock import Mock

    from sgl_jax.srt.managers.schedule_batch import (
        FINISH_ABORT,
        Req,
        ScheduleBatch,
        ScheduleReqsInfo,
    )
    from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    cache = DeepseekV4ChunkCache(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator, runner.page_size, 128
    )
    completed_len = 128 if phase == "prefill" else 254
    finishing = phase.startswith("finish-")
    req = Req(
        phase,
        "",
        [1] * completed_len,
        SamplingParams(max_new_tokens=1 if finishing else 4, ignore_eos=True),
        dp_rank=0,
    )
    assert runner.req_to_token_pool.alloc([req]) is not None
    step_batch(runner, req, 0, completed_len)
    chunked = phase in ("chunk", "abort-chunk")
    req.is_chunked = int(chunked)
    mode = ForwardMode.DECODE if "decode" in phase else ForwardMode.EXTEND
    original = ScheduleBatch(
        reqs_info=[
            ScheduleReqsInfo(
                reqs=[req],
                seq_lens=np.array([completed_len], np.int32),
                extend_lens=[completed_len],
                extend_logprob_start_lens=[0],
                decoding_reqs=[req] if phase == "mixed" else None,
            )
        ],
        forward_mode=mode,
        per_dp_bs_size=1,
    )
    submitted = original.copy() if overlap else original
    if overlap:
        # Both the array and shared Req may advance before previous-result processing.
        step_batch(runner, req, completed_len, completed_len + 1, decode=True)
        original.reqs_info[0].seq_lens[:] += 1

    scheduler = SchedulerOutputProcessorMixin()
    scheduler.is_generation, scheduler.enable_overlap, scheduler.pd = True, overlap, None
    scheduler.spec_algorithm, scheduler.tree_cache = None, cache
    scheduler.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator
    scheduler.num_generated_tokens, scheduler.forward_ct_decode = 0, 0
    scheduler.server_args = SimpleNamespace(decode_log_interval=1000)
    scheduler._pending_chunked_abort_reqs = [req if phase == "abort-chunk" else None]
    scheduler.chunked_reqs = [req if chunked else None]
    scheduler._release_prefill_host_buffer = Mock()
    scheduler.set_next_batch_sampling_info_done = Mock()
    scheduler.stream_output = Mock()
    logits = SimpleNamespace(hidden_states=None, next_token_logprobs=None)

    def resolve(_):
        # Reclamation must not run until the worker reports completed execution.
        assert req.swa_evicted_seqlen == 0
        return logits, [2], 0

    scheduler.tp_worker = SimpleNamespace(resolve_last_batch_result=Mock(side_effect=resolve))
    result = SimpleNamespace(
        logits_output=logits,
        next_token_ids=[2],
        cache_miss_count=0,
        bid=1,
        extend_input_len_per_req=[completed_len],
        extend_logprob_start_len_per_req=[0],
        next_draft_input=None,
    )
    if phase == "abort-chunk":
        req.to_finish = FINISH_ABORT("cancelled")
    try:
        if mode.is_decode():
            scheduler.process_batch_result_decode(submitted, result)
        else:
            scheduler.process_batch_result_prefill(submitted, result)
        scheduler.stream_output.assert_called_once()
        assert scheduler.tp_worker.resolve_last_batch_result.call_count == int(overlap)
        if finishing or phase == "abort-chunk":
            assert req.finished() and req.req_pool_idx is None
            assert req.kv_allocated_len == 0
        else:
            assert req.swa_evicted_seqlen == 0
            assert req.kv_committed_len == completed_len + int(overlap)
            last_kept = min(completed_len - 1, runner.page_size - 1)
            loc = runner.req_to_token_pool.req_to_token[req.req_pool_idx, last_kept]
            assert runner.token_to_kv_pool_allocator.full_to_swa_index_mapping[loc] != 0
            assert req.output_ids == ([] if chunked else [2])
        if phase == "abort-chunk":
            assert scheduler.chunked_reqs == scheduler._pending_chunked_abort_reqs == [None]
    finally:
        release_kv_cache(req, cache)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dp_size", 2),
        ("page_size", 64),
        ("kv_cache_dtype", "fp8"),
        ("disable_radix_cache", False),
        ("speculative_algorithm", "EAGLE3"),
        ("disaggregation_mode", "prefill"),
    ],
)
def test_unsupported_configs_fail_early(runner, field, value):
    args = SimpleNamespace(**(vars(runner.server_args) | {field: value}))
    with pytest.raises(ValueError):
        validate_runtime_config(args, dp_size=value if field == "dp_size" else 1)


def test_capacity_variants_cover_independent_csa_and_hca_buckets():
    from sgl_jax.srt.layers.attention.deepseek_v4_backend import capacity_bucket

    for tuned in (False, True):
        backend = SimpleNamespace(max_context_len=65536, use_pallas_hca=tuned, page_size=128)
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
            variants = set(precompile_capacity_variants(backend, mode, 4, 262144))
            for lengths in ((8192,) * 4, (32769, 1, 1, 1), (65536,) * 4, (16384, 32768, 65536, 2)):
                c4 = capacity_bucket(
                    max(n // 4 for n in lengths)
                    if mode.is_decode()
                    else sum(n // 4 for n in lengths)
                )
                c128 = capacity_bucket(
                    max(n // 128 for n in lengths) if tuned else sum(n // 128 for n in lengths)
                )
                assert (c4, 128 if tuned else c128, c128) in variants


def test_shared_packed_transport_snapshots_dp_fields():
    devices = jax.devices()[:2]
    mesh = Mesh(
        np.asarray(devices).reshape(2, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    sharding = NamedSharding(mesh, P("data"))
    left = np.arange(8, dtype=np.int32)
    right = np.arange(12, dtype=np.int64).reshape(4, 3)
    expected = left.copy(), right.copy()
    a, b = packed_device_array((left, right), sharding)
    left.fill(-1)
    right.fill(-1)
    np.testing.assert_array_equal(a, expected[0])
    np.testing.assert_array_equal(b, expected[1])
    assert a.sharding == b.sharding == sharding


def test_offline_v4_inputs_match_serving_and_executable_store(runner, tmp_path):
    from sgl_jax.srt.model_executor.aot_executable import (
        ExecutableStore,
        save_executable,
    )
    from sgl_jax.srt.model_executor.aot_inputs import AbstractModel

    options = SimpleNamespace(
        target="cpu",
        tp_size=1,
        dp_size=1,
        ep_size=1,
        attention_backend="fa",
        moe_backend="epmoe",
        page_size=runner.page_size,
        kv_capacity=2048,
        recurrent_capacity=None,
        batch_size=2,
        context_length=256,
        mtp_layer_idx=0,
        draft_token_num=None,
        workload="decode",
        cache_loc_size=512,
        decode_page_count=None,
        num_tokens=None,
        chunked_prefill_size=None,
        v4_capacities=(128, 128, 128),
    )
    cfg = DeepseekV4Config.from_dict(runner.model_config.hf_text_config.to_dict())
    mc = ModelConfig(model_path=runner.server_args.model_path, hf_config=cfg, dtype="bfloat16")
    mc.configure_for_serving(runner.server_args)
    with patch.dict(os.environ, {"DSV4_FUSED_WO_A": "0"}), jax.set_mesh(runner.mesh):
        abstract = AbstractModel(mc, options, runner.mesh, server_args=runner.server_args)
        assert all(
            isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree.leaves(abstract.memory_pools)
        )
        fn, args, _, _ = abstract.build_inputs(options)
        lowered = fn.lower(*args)
        compiled = lowered.compile()
        save_executable(compiled, lowered, runner.mesh, None, tmp_path)
        dummy = next(manager(runner).iter_precompile_batches(runner, ForwardMode.DECODE))
        serving = runner.lower_model(dummy)
        store = ExecutableStore(tmp_path, runner.mesh)
        with patch(
            "jax._src.compiler.backend_compile_and_load",
            side_effect=AssertionError("store loading must not compile"),
        ):
            store.load(serving, None)
            from sgl_jax.srt.model_executor.aot_dispatch import AotDispatcher
            from sgl_jax.srt.model_executor.model_forward import make_jitted_run_model

            dispatcher = AotDispatcher(
                make_jitted_run_model(runner.attn_backend),
                (runner._model_def, runner._model_state_def, runner.model_state_leaves),
                (runner._model_def, runner.model_state_leaves),
                "v4-store-test",
                executable_store=store,
                allow_fast_dispatch=False,
            )
            lm = LogitsMetadata.from_model_worker_batch(dummy, runner.mesh)
            with patch.object(store, "load", wraps=store.load) as loads:
                for _ in range(2):
                    old = runner.memory_pools.token_to_kv_pool.get_swa_buffer(0)
                    result = dispatcher(dummy.forward_batch, runner.memory_pools, lm)
                    jax.block_until_ready(result)
                    runner.memory_pools.replace_all(result[1])
                    assert old.is_deleted()
                assert loads.call_count == 1


def test_mixed_chunk_and_decode_use_warmed_extend(runner):
    import jax._src.test_util as jtu

    first, second = new_request(runner), new_request(runner)
    cache = DeepseekV4ChunkCache(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator, runner.page_size, 128
    )
    try:
        jax.block_until_ready(forward(runner, step_batch(runner, first, 0, 4)))
        # ScheduleBatch.mix_with_running represents a decode row as EXTEND q_len=1.
        batch = step_batch(runner, second, 0, 4)
        continuation = step_batch(runner, first, 4, 5)
        batch.input_ids[4] = continuation.input_ids[0]
        batch.positions[4] = 4
        batch.out_cache_loc[4] = continuation.out_cache_loc[0]
        batch.seq_lens[:] = (4, 5)
        batch.req_pool_indices[:] = (second.req_pool_idx, first.req_pool_idx)
        batch.extend_seq_lens[:] = (4, 1)
        batch.extend_prefix_lens[:] = (0, 4)
        batch.logits_indices[:] = (3, 4)
        with (
            jtu.count_jit_and_pmap_lowerings() as compiles,
            patch(
                "jax._src.compiler.backend_compile_and_load",
                side_effect=AssertionError("mixed batch compiled after warmup"),
            ),
        ):
            result = forward(runner, batch)
            jax.block_until_ready(result)
        assert compiles() == 0
        assert np.isfinite(np.asarray(result[0].next_token_logits)).all()
    finally:
        release_kv_cache(first, cache)
        release_kv_cache(second, cache)


def test_jit_rejects_incomplete_payload_without_donating(runner):
    from flax import nnx

    from sgl_jax.srt.model_executor.model_forward import make_jitted_run_model

    class BadModel(nnx.Module):
        def __call__(self, batch, pools, logits):
            return jnp.zeros(1), {"token_to_kv_pool": {}}, True, None

    graph, state = nnx.split(BadModel())
    leaves, tree = jax.tree.flatten(state)
    batch = next(manager(runner).iter_precompile_batches(runner, ForwardMode.DECODE))
    old = runner.memory_pools.token_to_kv_pool.get_swa_buffer(0)
    with pytest.raises(ValueError, match="owner keys"):
        make_jitted_run_model(runner.attn_backend)(
            graph,
            tree,
            leaves,
            batch.forward_batch,
            runner.memory_pools,
            LogitsMetadata.from_model_worker_batch(batch, runner.mesh),
        )
    assert not old.is_deleted()


def test_cache_route_budget_and_retraction_abort(runner):
    from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
    from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
    from sgl_jax.srt.mem_cache.registry import (
        TreeCacheBuildContext,
        default_radix_cache_factory,
    )
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    budget = runner.deepseek_v4_pool_budget
    assert (
        sum(x.nbytes for x in jax.tree.leaves(runner.memory_pools))
        == budget.allocated_bytes_per_device
    )
    assert budget.allocated_bytes_per_device <= budget.available_bytes_per_device
    assert budget.history_tokens <= runner.server_args.max_total_tokens
    cache = default_radix_cache_factory(
        TreeCacheBuildContext(
            server_args=runner.server_args,
            params=CacheInitParams(
                runner.req_to_token_pool,
                runner.token_to_kv_pool_allocator,
                runner.page_size,
                sliding_window_size=128,
            ),
            is_hybrid_swa=True,
            disable_radix_cache=True,
            effective_chunked_prefill_size=128,
            model_config=runner.model_config,
            tp_size=1,
        )
    )
    assert isinstance(cache, DeepseekV4ChunkCache)
    req = Req("oom", "", [1], SamplingParams(max_new_tokens=16), dp_rank=0)
    runner.req_to_token_pool.alloc([req])
    step_batch(runner, req, 0, 128)
    allocator = runner.token_to_kv_pool_allocator
    # Reserve all remaining SWA pages for other work without a second request slot.
    pressure = allocator.alloc_extend(
        [0], [allocator.swa_available_size()], [-1], allocator.swa_available_size()
    )
    assert pressure is not None
    batch = object.__new__(ScheduleBatch)
    batch.dp_size, batch.is_hybrid = 1, True
    batch.reqs_info = [SimpleNamespace(reqs=[req])]
    batch.req_to_token_pool = runner.req_to_token_pool
    batch.token_to_kv_pool_allocator, batch.tree_cache = allocator, cache
    from unittest.mock import Mock

    batch.filter_batch = Mock()
    try:
        if runner.page_size == 128:
            assert not batch.check_decode_mem()
        else:
            # No new page is required for token 128 inside a 256-token page.
            assert batch.check_decode_mem()
        # Exhaust the append mapping to exercise C's SWA demand without a history page.
        allocator.free_swa(runner.req_to_token_pool.read(req.req_pool_idx, 128))
        extra = allocator.alloc_extend([0], [runner.page_size], [-1], runner.page_size)
        assert extra is not None
        pressure = np.r_[pressure, extra]
        assert not batch.check_decode_mem()
        retracted, _, aborted = batch.retract_decode(runner.server_args)
        assert not retracted and aborted == [req]
        assert req.req_pool_idx is None and req.kv_allocated_len == 0
        assert req.is_retracted and req.to_finish.is_error
        batch.filter_batch.assert_called_once_with(keep_indices={0: []})
    finally:
        release_kv_cache(req, cache)
        allocator.free(pressure)
    assert allocator.full_available_size() == budget.history_tokens
    assert allocator.swa_available_size() == budget.swa_tokens


def test_full_serving_bundle_uses_runtime_bucket_plan(runner, tmp_path):
    import json
    from dataclasses import replace

    from sgl_jax.srt.model_executor.aot_executable import ExecutableStore
    from sgl_jax.srt.model_executor.aot_server import export_server

    args = replace(
        runner.server_args,
        device="tpu",
        save_aot=str(tmp_path / "bundle"),
        aot_topology="cpu-fixture",
        precompile_num_threads=2,
        enable_topk_kernel=False,
    )
    # Exercise the full exporter on CPU; topology generation itself needs libtpu.
    with (
        patch("sgl_jax.srt.model_executor.aot_server.build_mesh", return_value=runner.mesh),
        patch("sgl_jax.srt.eplb.expert_location._GLOBAL_SERVER_ARGS", None),
    ):
        export_server(args)
    manifest = json.loads((tmp_path / "bundle" / "serving.json").read_text())
    assert manifest["status"] == "complete"
    assert {row["workload"] for row in manifest["buckets"]} == {"prefill", "decode"}
    assert all("v4_capacities" in row for row in manifest["buckets"])
    assert manifest["max_total_num_tokens"] == runner.max_total_num_tokens
    assert len({row["directory"] for row in manifest["sampling"]}) == len(manifest["sampling"])
    store = ExecutableStore(tmp_path / "bundle", runner.mesh)
    cm = manager(runner)
    with patch(
        "jax._src.compiler.backend_compile_and_load",
        side_effect=AssertionError("bundle did not cover serving bucket"),
    ):
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
            for batch in cm.iter_precompile_batches(runner, mode):
                if len(batch.input_ids) > args.chunked_prefill_size and mode.is_extend():
                    continue
                with jax.set_mesh(runner.mesh):
                    store.load(runner.lower_model(batch))


def test_cpu_export_selects_tpu_kernels_from_explicit_target():
    from sgl_jax.srt.kernels.dsv4.topk_threshold import _default_interpret
    from sgl_jax.srt.layers.attention.dsv4.decode import resolve_decode_indexer_backend
    from sgl_jax.srt.layers.attention.dsv4.dispatch import resolve_csa_attention_backend
    from sgl_jax.srt.layers.attention.dsv4.indexer import resolve_indexer_backend
    from sgl_jax.srt.layers.deepseek_v4_mhc import resolve_backend as mhc_backend
    from sgl_jax.srt.utils import jax_utils

    assert jax.default_backend() == "cpu"
    token = jax_utils._COMPILATION_TARGET.set(SimpleNamespace(platform="tpu", device_kind="TPU7x"))
    try:
        with patch.dict(
            os.environ,
            {
                "PALLAS_INTERPRET": "0",
                "DSV4_INDEXER_BACKEND": "auto",
                "DSV4_DECODE_INDEXER_BACKEND": "auto",
            },
        ):
            os.environ.pop("DSV4_CSA_ATTENTION", None)
            assert mhc_backend() == "pallas"
            assert resolve_indexer_backend() == "kernel"
            assert resolve_decode_indexer_backend() == "kernel"
            assert resolve_csa_attention_backend() == "fused"
            assert not _default_interpret()
    finally:
        jax_utils._COMPILATION_TARGET.reset(token)
    assert not jax_utils.is_tpu_runtime()
    assert mhc_backend() == "reference"
