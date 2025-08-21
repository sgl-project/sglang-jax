import logging
import unittest

import jax

from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import (
    JAX_PRECOMPILE_DEFAULT_DECODE_BS_PADDINGS,
    JAX_PRECOMPILE_DEFAULT_PREFILL_TOKEN_PADDINGS,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import (
    CustomTestCase,
    generate_schedule_batch,
    generate_server_args,
)


def get_forward_batch(
    server_args: ServerArgs,
    bs: int,
    num_tokens_per_req: int,
    max_running_requests: int,
    max_total_num_tokens: int,
    mode: ForwardMode,
) -> ForwardBatch:
    schedule_batch = generate_schedule_batch(
        bs, num_tokens_per_req, mode, TestPrecompile.tp_worker.model_runner
    )
    model_worker_batch = schedule_batch.get_model_worker_batch(
        max_running_requests,
        max_total_num_tokens,
        (
            server_args.jax_precompile_decode_bs_paddings
            if server_args.jax_precompile_decode_bs_paddings is not None
            else JAX_PRECOMPILE_DEFAULT_DECODE_BS_PADDINGS
        ),
        (
            server_args.jax_precompile_prefill_token_paddings
            if server_args.jax_precompile_prefill_token_paddings is not None
            else JAX_PRECOMPILE_DEFAULT_PREFILL_TOKEN_PADDINGS
        ),
    )
    return ForwardBatch.init_new(
        model_worker_batch, TestPrecompile.tp_worker.model_runner
    )


class TestPrecompile(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # setup ServerArgs
        server_args = generate_server_args()
        # update server_args
        server_args.jax_precompile_prefill_token_paddings = [1, 4096]
        server_args.jax_precompile_decode_bs_paddings = [1, 2]
        server_args.disable_jax_precompile = False
        print(f"Complete to generate server_args: {server_args}")

        # setup Mesh
        cls.mesh = create_device_mesh(
            ici_parallelism=[-1, server_args.tp_size, 1, 1],
            dcn_parallelism=[1, 1, 1, 1],
        )

        cls.tp_worker = ModelWorker(
            server_args=server_args,
            mesh=cls.mesh,
        )
        print(f"Complete to initialize ModelWorker!")

        (
            max_total_num_tokens,
            _,
            max_running_requests,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = cls.tp_worker.get_worker_info()

        cls.forward_batch_extend_bs_1 = get_forward_batch(
            server_args,
            1,
            4,
            max_running_requests,
            max_total_num_tokens,
            ForwardMode.EXTEND,
        )
        cls.forward_batch_extend_bs_gt_1 = get_forward_batch(
            server_args,
            2,
            4,
            max_running_requests,
            max_total_num_tokens,
            ForwardMode.EXTEND,
        )
        cls.forward_batch_decode_bs_1 = get_forward_batch(
            server_args,
            1,
            1,
            max_running_requests,
            max_total_num_tokens,
            ForwardMode.DECODE,
        )
        cls.forward_batch_decode_bs_gt_1 = get_forward_batch(
            server_args,
            2,
            1,
            max_running_requests,
            max_total_num_tokens,
            ForwardMode.DECODE,
        )
        print(f"Complete to get ForwardBatch!")

    @classmethod
    def tearDownClass(cls):
        pass

    def test_precompile(self):
        # precompile
        TestPrecompile.tp_worker.run_precompile()
        print(f"Complete to run precompile!")

        # run forward_extend
        import jax._src.test_util as jtu

        with jtu.count_pjit_cpp_cache_miss() as count, TestPrecompile.mesh, jax.sharding.use_mesh(
            TestPrecompile.mesh
        ):
            # extend: bs = 1
            TestPrecompile.tp_worker.model_runner.attn_backend.init_forward_metadata(
                TestPrecompile.forward_batch_extend_bs_1
            )
            _, _, _ = TestPrecompile.tp_worker.model_runner.model_fn(
                TestPrecompile.tp_worker.model_runner.state,
                TestPrecompile.forward_batch_extend_bs_1.input_ids,
                TestPrecompile.forward_batch_extend_bs_1.positions,
                TestPrecompile.forward_batch_extend_bs_1,
            )

            cache_miss_count = count()
            assert (
                cache_miss_count == 0
            ), f"[mode={TestPrecompile.forward_batch_extend_bs_1.forward_mode}, bs=1] CACHE_MISS count, real: {cache_miss_count}, expect: 0"
            print(
                f"Pass [mode={TestPrecompile.forward_batch_extend_bs_1.forward_mode}, bs=1] case!"
            )

            TestPrecompile.tp_worker.model_runner.attn_backend.init_forward_metadata(
                TestPrecompile.forward_batch_extend_bs_gt_1
            )
            _, _, _ = TestPrecompile.tp_worker.model_runner.model_fn(
                TestPrecompile.tp_worker.model_runner.state,
                TestPrecompile.forward_batch_extend_bs_gt_1.input_ids,
                TestPrecompile.forward_batch_extend_bs_gt_1.positions,
                TestPrecompile.forward_batch_extend_bs_gt_1,
            )

            cache_miss_count = count()
            assert (
                cache_miss_count == 0
            ), f"[mode={TestPrecompile.forward_batch_extend_bs_gt_1.forward_mode}, bs=1] CACHE_MISS count, real: {cache_miss_count}, expect: 0"
            print(
                f"Pass [mode={TestPrecompile.forward_batch_extend_bs_gt_1.forward_mode}, bs>1] case!"
            )

            TestPrecompile.tp_worker.model_runner.attn_backend.init_forward_metadata(
                TestPrecompile.forward_batch_decode_bs_1
            )
            _, _, _ = TestPrecompile.tp_worker.model_runner.model_fn(
                TestPrecompile.tp_worker.model_runner.state,
                TestPrecompile.forward_batch_decode_bs_1.input_ids,
                TestPrecompile.forward_batch_decode_bs_1.positions,
                TestPrecompile.forward_batch_decode_bs_1,
            )

            cache_miss_count = count()
            assert (
                cache_miss_count == 0
            ), f"[mode={TestPrecompile.forward_batch_decode_bs_1.forward_mode}, bs=1] CACHE_MISS count, real: {cache_miss_count}, expect: 0"
            print(
                f"Pass [mode={TestPrecompile.forward_batch_decode_bs_1.forward_mode}, bs=1] case!"
            )

            TestPrecompile.tp_worker.model_runner.attn_backend.init_forward_metadata(
                TestPrecompile.forward_batch_decode_bs_gt_1
            )
            _, _, _ = TestPrecompile.tp_worker.model_runner.model_fn(
                TestPrecompile.tp_worker.model_runner.state,
                TestPrecompile.forward_batch_decode_bs_gt_1.input_ids,
                TestPrecompile.forward_batch_decode_bs_gt_1.positions,
                TestPrecompile.forward_batch_decode_bs_gt_1,
            )

            cache_miss_count = count()
            assert (
                cache_miss_count == 0
            ), f"[mode={TestPrecompile.forward_batch_decode_bs_gt_1.forward_mode}, bs=1] CACHE_MISS count, real: {cache_miss_count}, expect: 0"
            print(
                f"Pass [mode={TestPrecompile.forward_batch_decode_bs_gt_1.forward_mode}, bs>1] case!"
            )

        print(f"Pass test_precompile case!")


if __name__ == "__main__":
    unittest.main()
