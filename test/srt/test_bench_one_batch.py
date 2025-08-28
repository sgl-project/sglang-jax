import os
import unittest

from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    run_bench_offline_throughput,
    write_github_step_summary,
)

# We use `run_bench_offline_throughput`` instead of `run_bench_one_batch`


class TestBenchOneBatch(CustomTestCase):

    def test_bs1_default(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST,
            ["--tp-size", "1", "--mem-fraction-static", "0.1"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs1_default (qwen-7b-chat)\n"
                f"output_throughput: {output_throughput:.2f} token/s\n"
            )
            self.assertGreater(output_throughput, 80)


if __name__ == "__main__":
    unittest.main()
