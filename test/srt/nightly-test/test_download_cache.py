import argparse
import asyncio
import itertools
import os
import time
import unittest
from random import random, uniform

import requests

from sgl_jax.bench_serving import SHAREGPT_URL, download_and_cache_file, run_benchmark
from sgl_jax.test.test_utils import (
    BAILING_MOE,
    DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    GEMMA2_2B_IT,
    QWEN2_5_7B_INSTRUCT,
    QWEN3_8B,
    QWEN3_CODER_30B_A3B_INSTRUCT,
    QWEN3_MOE_30B,
    QWEN_7B,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_serving,
    write_github_step_summary,
)


sharegpt_dataset_path = download_and_cache_file(SHAREGPT_URL)
print(f"Dataset is ready at location: {sharegpt_dataset_path}")