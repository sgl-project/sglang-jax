# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

DEFAULT_PROMPTS = [
    "Apple is red. Banana is Yellow. " * 800 + "Apple is",
    "The capital of the United Kingdom is",
    "Today is a sunny day and I like",
    "AI is a field of computer science focused on",
    # the output of gemma-2-2b from SRT is unstable on the commented prompt
    # "The capital of France is",
]
TEST_RERANK_QUERY_DOCS = [
    {
        "query": "How many people live in Berlin?",
        "documents": [
            "Berlin is well known for its museums.",
        ],
    },
    {
        "query": "How many people live in Berlin?",
        "documents": [
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "Berlin is well known for its museums.",
        ],
    },
]

dirpath = os.path.dirname(__file__)
with open(os.path.join(dirpath, "long_prompt.txt")) as f:
    long_prompt = f.read()
DEFAULT_PROMPTS.append(long_prompt)

NUM_TOP_LOGPROBS = 5
