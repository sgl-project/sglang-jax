# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

## Math-500

### Introduction

Environment: TPU v6e-1.
Version: main-5fc4fa54a12ea0cbf05c4e304f0f69595e556aa7

### Instructions

```bash
# sky-31d4-pseudonym
# launch server, precision = bfloat16
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
--model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--trust-remote-code  \
--tp-size=1 \
--device=tpu \
--mem-fraction-static=0.8 \
--chunked-prefill-size=2048 \
--download-dir=/tmp \
--dtype=bfloat16 \
--max-running-requests 256 \
--skip-server-warmup \
--page-size=128  \
--disable-radix-cache \
--use-sort-for-toppk-minp

# sky-495d-pseudonym
# launch server, precision = float32
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
--model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--trust-remote-code  \
--tp-size=1 \
--device=tpu \
--mem-fraction-static=0.8 \
--chunked-prefill-size=2048 \
--download-dir=/tmp \
--dtype=float32 \
--max-running-requests 256 \
--skip-server-warmup \
--page-size=128  \
--disable-radix-cache \
--use-sort-for-toppk-minp

# eval: evalscope = 0.17.1
## Sampling parameters refer to https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B.
## Eval generation config refers to https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html#id5.
## Note: n in generation-config does not take effect due to https://github.com/sgl-project/sglang-jax/issues/296. So please get mean grade manually.
evalscope eval  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--api-url http://127.0.0.1:30000/v1/chat/completions \
--api-key EMPTY \
--eval-type service \
--datasets math_500  \
--eval-batch-size 64 \
--dataset-args '{"math_500":{"metric_list":["Pass@1"]}}' --generation-config '{"max_tokens": 32768, "temperature": 0.6, "top_p": 0.95}' \
--timeout 120000
```

### Evaluation Results

- bloat16: 0.818, 0.82, 0.826.
- float32: 0.81, 0.796, 0.808.

#### Details

Note:
- Every test under bfloat16 costs about 35 minutes.
- Every test under float32 costs about 40 minutes.

```bash
###############################################################################################
#################################### Precision: bfloat16 ######################################
###############################################################################################
#################################### First time result   ######################################
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| Model                         | Dataset   | Metric   | Subset   |   Num |   Score | Cat.0   |
+===============================+===========+==========+==========+=======+=========+=========+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 1  |    43 |  0.907  | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 2  |    90 |  0.9889 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 3  |   105 |  0.8857 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 4  |   128 |  0.8125 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 5  |   134 |  0.6269 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | OVERALL  |   500 |  0.818  | -       |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
#################################### Second time result   #####################################
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| Model                         | Dataset   | Metric   | Subset   |   Num |   Score | Cat.0   |
+===============================+===========+==========+==========+=======+=========+=========+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 1  |    43 |  0.9302 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 2  |    90 |  0.9444 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 3  |   105 |  0.9048 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 4  |   128 |  0.8125 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 5  |   134 |  0.6418 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | OVERALL  |   500 |  0.82   | -       |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
#################################### Third time result   ######################################
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| Model                         | Dataset   | Metric   | Subset   |   Num |   Score | Cat.0   |
+===============================+===========+==========+==========+=======+=========+=========+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 1  |    43 |  0.9535 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 2  |    90 |  0.9444 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 3  |   105 |  0.9048 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 4  |   128 |  0.7969 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 5  |   134 |  0.6716 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | OVERALL  |   500 |  0.826  | -       |
+-------------------------------+-----------+----------+----------+-------+---------+---------+



###############################################################################################
#################################### Precision: float32 #######################################
###############################################################################################
#################################### First time result   ######################################
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| Model                         | Dataset   | Metric   | Subset   |   Num |   Score | Cat.0   |
+===============================+===========+==========+==========+=======+=========+=========+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 1  |    43 |  0.907  | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 2  |    90 |  0.9444 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 3  |   105 |  0.9143 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 4  |   128 |  0.8047 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 5  |   134 |  0.6119 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | OVERALL  |   500 |  0.81   | -       |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
#################################### Second time result   #####################################
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| Model                         | Dataset   | Metric   | Subset   |   Num |   Score | Cat.0   |
+===============================+===========+==========+==========+=======+=========+=========+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 1  |    43 |  0.907  | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 2  |    90 |  0.9111 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 3  |   105 |  0.9238 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 4  |   128 |  0.7422 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 5  |   134 |  0.6343 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | OVERALL  |   500 |  0.796  | -       |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
#################################### Third time result   ######################################
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| Model                         | Dataset   | Metric   | Subset   |   Num |   Score | Cat.0   |
+===============================+===========+==========+==========+=======+=========+=========+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 1  |    43 |  0.9535  | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 2  |    90 |  0.9444 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 3  |   105 |  0.8857 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 4  |   128 |  0.7812 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | Level 5  |   134 |  0.6343 | default |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-1.5B | math_500  | Pass@1   | OVERALL  |   500 |  0.808  | -       |
+-------------------------------+-----------+----------+----------+-------+---------+---------+
```

#### Complete Evaluation Configuration
```yaml
analysis_report: false
api_key: EMPTY
api_url: http://127.0.0.1:30000/v1/chat/completions
chat_template: null
dataset_args:
  math_500:
    dataset_id: AI-ModelScope/MATH-500
    description: MATH-500 is a benchmark for evaluating mathematical reasoning capabilities
      of AI models. It consists of 500 diverse math problems across five levels of
      difficulty, designed to test a model's ability to solve complex mathematical
      problems by generating step-by-step solutions and providing the correct final
      answer.
    eval_split: test
    extra_params: {}
    few_shot_num: 0
    few_shot_random: false
    filters: null
    metric_list:
    - Pass@1
    model_adapter: generation
    name: math_500
    output_types:
    - generation
    pretty_name: MATH-500
    prompt_template: '{query}

      Please reason step by step, and put your final answer within \boxed{{}}.'
    query_template: null
    subset_list:
    - Level 1
    - Level 2
    - Level 3
    - Level 4
    - Level 5
    system_prompt: null
    tags:
    - Mathematics
    train_split: null
dataset_dir: /home/gcpuser/.cache/modelscope/hub/datasets
dataset_hub: modelscope
datasets:
- math_500
debug: false
dry_run: false
eval_backend: Native
eval_batch_size: 64
eval_config: null
eval_type: service
generation_config:
  max_tokens: 32768
  temperature: 0.6
  top_p: 0.95
ignore_errors: false
judge_model_args: {}
judge_strategy: auto
judge_worker_num: 1
limit: null
mem_cache: false
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
model_args: {}
model_id: DeepSeek-R1-Distill-Qwen-1.5B
model_task: text_generation
outputs: null
seed: 42
stage: all
stream: false
template_type: null
timeout: 120000.0
use_cache: null
work_dir: ./outputs/20251121_043226
```
