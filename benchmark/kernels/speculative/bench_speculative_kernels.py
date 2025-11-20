import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from sgl_jax.srt.kernels.speculative.build_eagle_tree_structure_kernel import (
    build_eagle_tree_structure_pallas_call,
)
from sgl_jax.srt.kernels.speculative.tree_speculative_sampling_target_only_kernel import (
    tree_speculative_sampling_target_only_pallas_call,
)
from sgl_jax.srt.kernels.speculative.verify_tree_greedy_kernel import (
    verify_tree_greedy_pallas_call,
)
from sgl_jax.srt.speculative.eagle_util import build_tree_kernel_efficient_preprocess
from sgl_jax.test.test_utils import CustomTestCase, is_in_ci


def _block_until_ready(outputs):
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), outputs)


def _create_build_tree_inputs():
    verified_id = jnp.array([29974, 13], dtype=jnp.int32)

    score_list = [
        jnp.array(
            [
                [[7.1127e-01, 2.8292e-01, 2.2995e-03, 1.7357e-03]],
                [[9.7476e-01, 2.2219e-02, 6.5031e-04, 1.3212e-04]],
            ],
            dtype=jnp.float32,
        ),
        jnp.array(
            [
                [
                    [6.9142e-01, 1.2863e-02, 1.6873e-03, 1.1871e-03],
                    [2.4787e-01, 1.8818e-02, 1.4204e-02, 9.2235e-04],
                    [2.2971e-03, 1.6700e-06, 1.8737e-07, 8.3146e-08],
                    [1.2771e-03, 2.4374e-04, 1.7832e-04, 1.1947e-05],
                ],
                [
                    [8.4832e-02, 6.6068e-02, 5.8304e-02, 5.7851e-02],
                    [2.3616e-03, 1.1243e-03, 5.4368e-04, 2.7768e-04],
                    [2.5286e-04, 1.5578e-04, 2.8817e-05, 1.2888e-05],
                    [1.2834e-04, 2.5417e-06, 1.1279e-06, 1.6088e-08],
                ],
            ],
            dtype=jnp.float32,
        ),
        jnp.array(
            [
                [
                    [6.6438e-01, 2.6997e-02, 2.4236e-05, 4.0821e-06],
                    [2.4402e-01, 2.8409e-03, 5.0935e-04, 2.9022e-04],
                    [1.6178e-02, 2.0567e-03, 4.5892e-04, 3.0034e-05],
                    [1.3023e-02, 5.0497e-04, 3.6371e-04, 8.7750e-05],
                ],
                [
                    [2.3263e-02, 2.0054e-02, 9.3990e-03, 2.7783e-03],
                    [6.4156e-02, 5.5506e-04, 1.0429e-04, 9.7211e-05],
                    [4.9950e-02, 5.0630e-03, 9.0068e-04, 3.3656e-04],
                    [7.5817e-03, 8.5731e-04, 6.9972e-04, 6.0793e-04],
                ],
            ],
            dtype=jnp.float32,
        ),
        jnp.array(
            [
                [
                    [6.6420e-01, 1.0525e-04, 6.5864e-05, 1.2253e-06],
                    [1.3019e-01, 1.0461e-01, 5.2083e-03, 1.6777e-03],
                    [2.0103e-02, 6.7335e-03, 1.2625e-04, 1.0364e-05],
                    [1.5142e-02, 7.0819e-04, 9.6595e-05, 8.7951e-05],
                ],
                [
                    [5.8608e-02, 1.8840e-03, 7.8535e-04, 4.4400e-04],
                    [1.2185e-02, 2.0684e-03, 1.7418e-03, 1.4327e-03],
                    [6.2455e-03, 6.1487e-03, 2.6862e-03, 1.8034e-03],
                    [1.8590e-03, 1.6151e-03, 1.2481e-03, 3.6038e-04],
                ],
            ],
            dtype=jnp.float32,
        ),
    ]

    token_list = [
        jnp.array(
            [[29896, 29906, 29900, 29945], [13, 2, 29871, 28956]],
            dtype=jnp.int32,
        ),
        jnp.array(
            [
                [
                    29889,
                    29974,
                    29945,
                    29900,
                    29974,
                    29922,
                    29930,
                    29958,
                    29889,
                    29974,
                    29930,
                    29945,
                    29974,
                    29922,
                    29930,
                    29958,
                ],
                [
                    22550,
                    4136,
                    16492,
                    8439,
                    29871,
                    2,
                    3001,
                    13,
                    2,
                    13,
                    29906,
                    29946,
                    2,
                    13,
                    29871,
                    259,
                ],
            ],
        ),
        jnp.array(
            [
                [
                    29946,
                    29945,
                    29953,
                    29906,
                    29896,
                    29945,
                    29900,
                    29906,
                    29896,
                    29945,
                    29906,
                    29953,
                    29896,
                    29945,
                    29906,
                    29946,
                ],
                [
                    29871,
                    2,
                    29901,
                    29889,
                    29871,
                    2,
                    395,
                    259,
                    29901,
                    29871,
                    2,
                    29889,
                    3001,
                    1234,
                    7146,
                    2186,
                ],
            ],
        ),
        jnp.array(
            [
                [
                    29946,
                    29974,
                    29945,
                    29930,
                    29889,
                    29922,
                    29974,
                    29930,
                    29974,
                    29946,
                    29930,
                    29922,
                    29889,
                    29974,
                    29945,
                    29922,
                ],
                [
                    29941,
                    29906,
                    2,
                    29946,
                    29871,
                    450,
                    319,
                    14990,
                    29946,
                    29941,
                    2,
                    29906,
                    29871,
                    2,
                    3001,
                    13,
                ],
            ],
        ),
    ]

    parents_list = [
        jnp.array([[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]], dtype=jnp.int32),
        jnp.array([[4, 8, 9, 10], [4, 5, 6, 7]], dtype=jnp.int32),
        jnp.array([[20, 24, 21, 28], [24, 28, 20, 21]], dtype=jnp.int32),
        jnp.array([[36, 40, 41, 44], [36, 40, 44, 45]], dtype=jnp.int32),
    ]

    seq_lens = jnp.array([5, 10], dtype=jnp.int32)
    topk = 4
    num_draft_token = 8
    parent_list, selected_index, _ = build_tree_kernel_efficient_preprocess(
        verified_id=verified_id,
        score_list=score_list,
        token_list=token_list,
        parents_list=parents_list,
        num_verify_tokens=num_draft_token,
    )
    seq_lens_sum = jnp.asarray(jnp.sum(seq_lens), dtype=jnp.int32)

    return {
        "parent_list": parent_list,
        "selected_index": selected_index,
        "verified_seq_len": seq_lens,
        "seq_lens_sum": seq_lens_sum,
        "draft_token_num": num_draft_token,
        "topk": topk,
        "max_context_len": int(seq_lens.max()),
    }


def _base_speculative_tree_arrays():
    candidates = jnp.array(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=jnp.int32,
    )
    retrive_index = jnp.array(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=jnp.int32,
    )
    retrive_next_token = jnp.array(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=jnp.int32,
    )
    retrive_next_sibling = jnp.array(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=jnp.int32,
    )
    return {
        "candidates": candidates,
        "retrive_index": retrive_index,
        "retrive_next_token": retrive_next_token,
        "retrive_next_sibling": retrive_next_sibling,
    }


def _build_target_logits():
    target_logits = jnp.full((2, 6, 20), 1, dtype=jnp.float32)
    target_logits = target_logits.at[0, 0, 3].set(10)
    target_logits = target_logits.at[0, 3, 4].set(10)
    target_logits = target_logits.at[0, 4, 5].set(10)
    target_logits = target_logits.at[1, 0, 11].set(10)
    target_logits = target_logits.at[1, 4, 12].set(10)

    for bid in range(target_logits.shape[0]):
        for tid in range(target_logits.shape[1]):
            mask = jnp.max(target_logits[bid, tid]) < 10
            target_logits = target_logits.at[bid, tid, 18].set(
                jnp.where(mask, 10, target_logits[bid, tid, 18])
            )
    return target_logits


def _create_verify_tree_inputs():
    arrays = _base_speculative_tree_arrays()
    target_logits = _build_target_logits()
    target_predict = jnp.argmax(target_logits, axis=-1).flatten()
    bs, num_spec_tokens = 2, 4
    predict_shape = (arrays["candidates"].shape[0] * arrays["candidates"].shape[1],)
    return {
        "predicts_template": np.empty(predict_shape, dtype=np.int32),
        "accept_index_template": np.full((bs, num_spec_tokens), -1, dtype=np.int32),
        "accept_token_num_template": np.zeros((bs,), dtype=np.int32),
        "target_predict": target_predict,
        "num_spec_tokens": num_spec_tokens,
        **arrays,
    }


def _create_tree_sampling_inputs():
    arrays = _base_speculative_tree_arrays()
    target_logits = _build_target_logits()
    bs, num_draft_tokens = arrays["candidates"].shape
    num_spec_tokens = 4
    temperatures = jnp.array(
        [0.01, 0.01],
        dtype=jnp.float32,
    ).reshape(bs, 1, 1)
    target_probs = jax.nn.softmax(target_logits / temperatures, axis=-1).reshape(
        bs * num_draft_tokens, -1
    )
    draft_probs = jnp.zeros_like(target_probs)
    return {
        "predicts_template": np.full((bs * num_draft_tokens,), -1, dtype=np.int32),
        "accept_index_template": np.full((bs, num_spec_tokens), -1, dtype=np.int32),
        "accept_token_num_template": np.zeros((bs,), dtype=np.int32),
        "target_probs": target_probs,
        "draft_probs": draft_probs,
        "uniform_samples": random.uniform(
            random.PRNGKey(42), (bs, num_draft_tokens), dtype=jnp.float32
        ),
        "uniform_samples_for_final_sampling": random.uniform(
            random.PRNGKey(42), (bs,), dtype=jnp.float32
        ),
        "num_spec_tokens": num_spec_tokens,
        **arrays,
    }


def benchmark_build_eagle_tree_structure():
    inputs = _create_build_tree_inputs()

    def run_kernel():
        return build_eagle_tree_structure_pallas_call(
            inputs["parent_list"],
            inputs["selected_index"],
            inputs["verified_seq_len"],
            inputs["seq_lens_sum"],
            draft_token_num=inputs["draft_token_num"],
            topk=inputs["topk"],
            max_context_len=inputs["max_context_len"],
            tree_mask_mode=0,
        )

    _block_until_ready(run_kernel())
    times = []
    for _ in range(3):
        start = time.perf_counter()
        outputs = run_kernel()
        _block_until_ready(outputs)
        times.append(time.perf_counter() - start)
    return np.mean(times)


def benchmark_verify_tree_greedy():
    inputs = _create_verify_tree_inputs()
    candidates = inputs["candidates"]
    retrive_index = inputs["retrive_index"]
    retrive_next_token = inputs["retrive_next_token"]
    retrive_next_sibling = inputs["retrive_next_sibling"]
    target_predict = inputs["target_predict"]
    num_draft_tokens = candidates.shape[1]
    num_spec_tokens = inputs["num_spec_tokens"]

    def run_kernel():
        predicts = jax.device_put(inputs["predicts_template"])
        accept_index = jax.device_put(inputs["accept_index_template"])
        accept_token_num = jax.device_put(inputs["accept_token_num_template"])
        return verify_tree_greedy_pallas_call(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
            draft_token_num=num_draft_tokens,
            num_spec_tokens=num_spec_tokens,
        )

    _block_until_ready(run_kernel())
    times = []
    for _ in range(3):
        start = time.perf_counter()
        outputs = run_kernel()
        _block_until_ready(outputs)
        times.append(time.perf_counter() - start)
    return np.mean(times)


def benchmark_tree_speculative_sampling(
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
):
    inputs = _create_tree_sampling_inputs()
    candidates = inputs["candidates"]
    retrive_index = inputs["retrive_index"]
    retrive_next_token = inputs["retrive_next_token"]
    retrive_next_sibling = inputs["retrive_next_sibling"]

    def run_kernel():
        predicts = jax.device_put(inputs["predicts_template"])
        accept_index = jax.device_put(inputs["accept_index_template"])
        accept_token_num = jax.device_put(inputs["accept_token_num_template"])
        return tree_speculative_sampling_target_only_pallas_call(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            uniform_samples=inputs["uniform_samples"],
            uniform_samples_for_final_sampling=inputs["uniform_samples_for_final_sampling"],
            target_probs=inputs["target_probs"],
            draft_probs=inputs["draft_probs"],
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            deterministic=True,
        )

    _block_until_ready(run_kernel())
    times = []
    for _ in range(3):
        start = time.perf_counter()
        outputs = run_kernel()
        _block_until_ready(outputs)
        times.append(time.perf_counter() - start)
    return np.mean(times)


def full_benchmark():
    build_time = benchmark_build_eagle_tree_structure()
    print(f"[build_eagle_tree_structure] {build_time * 1000:.3f} ms")
    verify_time = benchmark_verify_tree_greedy()
    print(f"[verify_tree_greedy] {verify_time * 1000:.3f} ms")
    sampling_time = benchmark_tree_speculative_sampling()
    print(f"[tree_speculative_sampling_target_only] {sampling_time * 1000:.3f} ms")


class TestPerformance(CustomTestCase):
    def test_build_eagle_tree_structure_performance(self, floating_threshold: float = 0.25):
        baseline_ms = 120.0
        cost = benchmark_build_eagle_tree_structure()
        allowed = baseline_ms * (1 + floating_threshold)
        print(f"[build_eagle_tree_structure] res={cost * 1000:.3f}ms, expected<{allowed}ms")
        self.assertLess(
            cost * 1000,
            allowed,
            "Run build_eagle_tree_structure performance test failed",
        )

    def test_verify_tree_greedy_performance(self, floating_threshold: float = 0.25):
        baseline_ms = 60.0
        cost = benchmark_verify_tree_greedy()
        allowed = baseline_ms * (1 + floating_threshold)
        print(f"[verify_tree_greedy] res={cost * 1000:.3f}ms, expected<{allowed}ms")
        self.assertLess(
            cost * 1000,
            allowed,
            "Run verify_tree_greedy performance test failed",
        )

    def test_tree_speculative_sampling_performance(
        self,
        floating_threshold: float = 0.25,
    ):
        baseline_ms = 80.0
        cost = benchmark_tree_speculative_sampling()
        allowed = baseline_ms * (1 + floating_threshold)
        print(
            "[tree_speculative_sampling_target_only] "
            f"res={cost * 1000:.3f}ms, expected<{allowed}ms"
        )
        self.assertLess(
            cost * 1000,
            allowed,
            "Run tree_speculative_sampling_target_only performance test failed",
        )


if __name__ == "__main__":
    if is_in_ci():
        print("Run Speculative Kernel Performance Tests...")
        suite = TestPerformance()
        suite.test_build_eagle_tree_structure_performance()
        suite.test_verify_tree_greedy_performance()
        suite.test_tree_speculative_sampling_performance()
    else:
        print("Run Speculative Kernel Full Benchmark...")
        full_benchmark()
