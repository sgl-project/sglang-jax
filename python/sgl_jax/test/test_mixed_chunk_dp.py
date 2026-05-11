import unittest
from types import SimpleNamespace

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.managers.schedule_policy import PrefillAdder
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput


class _DummyAllocator:
    def available_size(self, dp_rank: int = 0):
        return 1_000_000


class _DummyTreeCache:
    def evictable_size(self, dp_rank: int = 0):
        return 0


def _make_req(rid: str, dp_rank: int, input_len: int = 4, output_len: int = 2) -> Req:
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=list(range(input_len)),
        sampling_params=SamplingParams(max_new_tokens=8),
        dp_rank=dp_rank,
        eos_token_ids={2},
        vocab_size=32000,
    )
    req.output_ids = list(range(output_len))
    return req


def _make_extend_batch(
    reqs_per_dp: list[list[Req]],
    input_ids_per_dp: list[np.ndarray],
    out_cache_loc_per_dp: list[np.ndarray],
    req_pool_indices_per_dp: list[np.ndarray],
    seq_lens_per_dp: list[np.ndarray],
) -> ScheduleBatch:
    batch = ScheduleBatch.init_new(
        reqs=reqs_per_dp,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        model_config=SimpleNamespace(vocab_size=32000),
        enable_overlap=False,
        dp_size=len(reqs_per_dp),
        spec_algorithm=None,
        mesh=None,
    )

    for dp_rank, info in enumerate(batch.reqs_info):
        info.input_ids = input_ids_per_dp[dp_rank]
        info.out_cache_loc = out_cache_loc_per_dp[dp_rank]
        info.req_pool_indices = req_pool_indices_per_dp[dp_rank]
        info.seq_lens = seq_lens_per_dp[dp_rank]
        info.prefix_lens = [len(r.origin_input_ids) - 1 for r in info.reqs]
        info.extend_lens = [1 for _ in info.reqs]
        info.extend_num_tokens = len(info.reqs)
        info.extend_logprob_start_lens = [0 for _ in info.reqs]
        info.seq_lens_sum = int(np.sum(info.seq_lens)) if info.seq_lens is not None else 0

    return batch


class TestMixedChunkDP(unittest.TestCase):
    def test_mix_with_running_merges_per_dp_fields(self):
        # DP0: one prefill req + one running decode req
        prefill_req_dp0 = _make_req("prefill-dp0", dp_rank=0)
        running_req_dp0 = _make_req("running-dp0", dp_rank=0)
        # DP1: one prefill req + one running decode req
        prefill_req_dp1 = _make_req("prefill-dp1", dp_rank=1)
        running_req_dp1 = _make_req("running-dp1", dp_rank=1)

        new_batch = _make_extend_batch(
            reqs_per_dp=[[prefill_req_dp0], [prefill_req_dp1]],
            input_ids_per_dp=[
                np.array([10], dtype=np.int32),
                np.array([20], dtype=np.int32),
            ],
            out_cache_loc_per_dp=[
                np.array([100], dtype=np.int32),
                np.array([200], dtype=np.int32),
            ],
            req_pool_indices_per_dp=[
                np.array([1], dtype=np.int32),
                np.array([2], dtype=np.int32),
            ],
            seq_lens_per_dp=[
                np.array([len(prefill_req_dp0.origin_input_ids) + len(prefill_req_dp0.output_ids)]),
                np.array([len(prefill_req_dp1.origin_input_ids) + len(prefill_req_dp1.output_ids)]),
            ],
        )

        running_batch = _make_extend_batch(
            reqs_per_dp=[[running_req_dp0], [running_req_dp1]],
            input_ids_per_dp=[
                np.array([11], dtype=np.int32),
                np.array([21], dtype=np.int32),
            ],
            out_cache_loc_per_dp=[
                np.array([101], dtype=np.int32),
                np.array([201], dtype=np.int32),
            ],
            req_pool_indices_per_dp=[
                np.array([3], dtype=np.int32),
                np.array([4], dtype=np.int32),
            ],
            seq_lens_per_dp=[
                np.array([len(running_req_dp0.origin_input_ids) + len(running_req_dp0.output_ids)]),
                np.array([len(running_req_dp1.origin_input_ids) + len(running_req_dp1.output_ids)]),
            ],
        )

        new_batch.mix_with_running(running_batch)

        # DP0 checks
        info0 = new_batch.reqs_info[0]
        self.assertEqual([r.rid for r in info0.reqs], ["prefill-dp0", "running-dp0"])
        np.testing.assert_array_equal(info0.input_ids, np.array([10, 11], dtype=np.int32))
        np.testing.assert_array_equal(info0.out_cache_loc, np.array([100, 101], dtype=np.int32))
        self.assertEqual(info0.extend_lens, [1, 1])
        self.assertEqual(info0.extend_num_tokens, 2)
        self.assertEqual(info0.extend_logprob_start_lens, [0, 0])

        # DP1 checks
        info1 = new_batch.reqs_info[1]
        self.assertEqual([r.rid for r in info1.reqs], ["prefill-dp1", "running-dp1"])
        np.testing.assert_array_equal(info1.input_ids, np.array([20, 21], dtype=np.int32))
        np.testing.assert_array_equal(info1.out_cache_loc, np.array([200, 201], dtype=np.int32))
        self.assertEqual(info1.extend_lens, [1, 1])
        self.assertEqual(info1.extend_num_tokens, 2)
        self.assertEqual(info1.extend_logprob_start_lens, [0, 0])

        # Running requests should be converted to 1-token extend for mixed chunk.
        self.assertEqual(running_req_dp0.extend_input_len, 1)
        self.assertEqual(running_req_dp1.extend_input_len, 1)
        self.assertEqual(
            running_req_dp0.fill_ids,
            running_req_dp0.origin_input_ids + running_req_dp0.output_ids,
        )
        self.assertEqual(
            running_req_dp1.fill_ids,
            running_req_dp1.origin_input_ids + running_req_dp1.output_ids,
        )

    def test_prefill_adder_accepts_per_dp_mixed_decode_tokens(self):
        adder = PrefillAdder(
            page_size=1,
            tree_cache=_DummyTreeCache(),
            token_to_kv_pool_allocator=_DummyAllocator(),
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=100,
            rem_chunk_tokens=20,
            mixed_with_decode_tokens=[3, 1],
            dp_size=2,
        )

        self.assertEqual(adder.rem_total_token_offset, [3, 1])
        self.assertEqual(adder.cur_rem_token_offset, [3, 1])
        self.assertEqual(adder.rem_input_tokens, 96)
        self.assertEqual(adder.rem_chunk_tokens_list, [17, 19])

    def test_filter_batch_uses_array_indices_for_jax_state(self):
        mesh = Mesh(np.array(jax.devices()).reshape((1,)), ("data",))
        vector_sharding = NamedSharding(mesh, P("data"))
        matrix_sharding = NamedSharding(mesh, P("data", None))
        reqs = [_make_req(f"req-{i}", dp_rank=0) for i in range(3)]
        batch = ScheduleBatch.init_new(
            reqs=[reqs],
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            model_config=SimpleNamespace(vocab_size=32000),
            enable_overlap=False,
            dp_size=1,
            spec_algorithm=None,
            mesh=None,
        )
        info = batch.reqs_info[0]
        info.req_pool_indices = jax.device_put(
            jnp.array([10, 11, 12], dtype=jnp.int32), vector_sharding
        )
        info.seq_lens = jax.device_put(jnp.array([20, 21, 22], dtype=jnp.int32), vector_sharding)
        info.output_ids = jax.device_put(jnp.array([30, 31, 32], dtype=jnp.int32), vector_sharding)
        info.spec_info = EagleDraftInput(
            topk_p=jax.device_put(jnp.arange(6, dtype=jnp.float32).reshape(3, 2), matrix_sharding),
            topk_index=jax.device_put(
                jnp.arange(6, dtype=jnp.int32).reshape(3, 2), matrix_sharding
            ),
            hidden_states=jax.device_put(
                jnp.arange(12, dtype=jnp.float32).reshape(3, 4), matrix_sharding
            ),
            verified_id=jax.device_put(jnp.array([40, 41, 42], dtype=jnp.int32), vector_sharding),
            allocate_lens=jax.device_put(jnp.array([50, 51, 52], dtype=jnp.int32), vector_sharding),
            new_seq_lens=jax.device_put(jnp.array([60, 61, 62], dtype=jnp.int32), vector_sharding),
        )

        batch.filter_batch(
            keep_indices={0: [0, 2]},
            chunked_req_to_exclude={0: None},
        )

        self.assertEqual([r.rid for r in info.reqs], ["req-0", "req-2"])
        np.testing.assert_array_equal(np.asarray(info.req_pool_indices), [10, 12])
        np.testing.assert_array_equal(np.asarray(info.seq_lens), [20, 22])
        np.testing.assert_array_equal(np.asarray(info.output_ids), [30, 32])
        np.testing.assert_array_equal(np.asarray(info.spec_info.verified_id), [40, 42])
        np.testing.assert_array_equal(np.asarray(info.spec_info.allocate_lens), [50, 52])
        np.testing.assert_array_equal(np.asarray(info.spec_info.new_seq_lens), [60, 62])

    def test_filter_batch_slices_already_filtered_jax_spec_state(self):
        mesh = Mesh(np.array(jax.devices()).reshape((1,)), ("data",))
        vector_sharding = NamedSharding(mesh, P("data"))
        matrix_sharding = NamedSharding(mesh, P("data", None))
        reqs = [_make_req(f"req-{i}", dp_rank=0) for i in range(3)]
        batch = ScheduleBatch.init_new(
            reqs=[reqs],
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            model_config=SimpleNamespace(vocab_size=32000),
            enable_overlap=False,
            dp_size=1,
            spec_algorithm=None,
            mesh=None,
        )
        info = batch.reqs_info[0]
        info.req_pool_indices = jax.device_put(
            jnp.array([10, 11, 12], dtype=jnp.int32), vector_sharding
        )
        info.seq_lens = jax.device_put(jnp.array([20, 21, 22], dtype=jnp.int32), vector_sharding)
        info.output_ids = jax.device_put(jnp.array([30, 31, 32], dtype=jnp.int32), vector_sharding)
        info.spec_info = EagleDraftInput(
            topk_p=jax.device_put(jnp.arange(6, dtype=jnp.float32).reshape(3, 2), matrix_sharding),
            topk_index=jax.device_put(
                jnp.arange(6, dtype=jnp.int32).reshape(3, 2), matrix_sharding
            ),
            hidden_states=jax.device_put(
                jnp.arange(12, dtype=jnp.float32).reshape(3, 4), matrix_sharding
            ),
            verified_id=jax.device_put(jnp.array([40, 41, 42], dtype=jnp.int32), vector_sharding),
            allocate_lens=jax.device_put(jnp.array([50, 51, 52], dtype=jnp.int32), vector_sharding),
            new_seq_lens=jax.device_put(jnp.array([60, 61, 62], dtype=jnp.int32), vector_sharding),
        )

        batch.filter_batch(keep_indices={0: [0, 2]})

        self.assertEqual([r.rid for r in info.reqs], ["req-0", "req-2"])
        np.testing.assert_array_equal(np.asarray(info.spec_info.verified_id), [40, 42])
        np.testing.assert_array_equal(np.asarray(info.spec_info.allocate_lens), [50, 52])


if __name__ == "__main__":
    unittest.main()
