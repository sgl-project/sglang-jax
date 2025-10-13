## Motivation

Integrate with tunix to make contribution to post-training.

## Solution

Sgl-jax will implement a SglJaxRollout in tunix to generate completions.

### Work in `tunix`

Implement a complete sgl-jax rollout adapter for Tunix integration. This enables using sgl-jax's high-performance inference capabilities within Tunix's GRPO/PPO training pipeline.

#### Key Components

1. **SglJaxSampler** (`tunix/generate/sgljax_sampler.py`)
   - Wraps sgl-jax ModelRunner with Tunix-compatible sampling interface
   - Supports parameter updates, batch generation, and logprobs computation

2. **SglJaxRollout** (`tunix/rl/rollout/sgljax_rollout.py`)
   - Implements BaseRollout interface for seamless GRPO/PPO integration
   - Provides parameter synchronization and per-token logprobs calculation
   - Includes factory function `create_sgljax_rollout` for easy setup


##### `update_params` API
For sgl-jax, we directly update the model runner's model state since both use the same NNX format.
```python
def update_params(
    self,
    updated_weights: jaxtyping.PyTree,
    filter_types: Optional[Tuple[Any, ...]] = None,
):
    """Update model parameters.

    For sgl-jax, we directly update the model runner's model state
    since both use the same NNX format.
    """
    from flax import nnx

    if filter_types is not None:
        # Filter and update only specific parameter types (e.g., LoRA)
        current_state = nnx.state(self._model, filter_types)
        filtered_updates = nnx.state(updated_weights, filter_types) if hasattr(updated_weights, '__call__') else {}

        # Merge filtered updates with current state
        for key, value in filtered_updates.items():
            if key in current_state:
                current_state[key] = value

        nnx.update(self._model, current_state)
    else:
        nnx.update(self._model, updated_weights)
```
Note: Add key_mappings and transpose_keys which follows the example of `utils.transfer_state_with_mappings` to handle the mapping and transformation between states.

#### Usage Example

```python
from tunix.rl.rollout.sgljax_rollout import create_sgljax_rollout
from tunix.rl.grpo.grpo_learner import GrpoLearner

# Create sgl-jax rollout worker
rollout = create_sgljax_rollout(
    model=model,
    tokenizer=tokenizer,
    model_config=model_config,
    mesh=mesh,
    max_model_len=2048,
)

# Integrate with GRPO training
rl_cluster = rl_cluster_lib.RLCluster(
    actor=actor_model,
    reference=reference_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
    rollout_worker=rollout,  # Use sgl-jax rollout
)

grpo_trainer = GrpoLearner(rl_cluster, reward_fns, grpo_config)
grpo_trainer.train(dataset)
```


### Work in Sgl-jax

1. Support `generate()` and `get_default_sampling_params()` in `Engine`.
2. Support abilities listed in the following table.
3. Support single process in sgl-jax Engine for Pathways.

#### `generate()` API

Note: ensure the output contains all information tunix need.

```python
def generate(
    self,
    ## The input prompt. It can be a single prompt or a batch of prompts.
    prompt: Optional[Union[List[str], str]] = None,
    # The token ids for text; one can specify either text or input_ids
    sampling_params: Optional[Union[List[Dict], Dict]] = None,
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None,
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = False,
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    logprob_start_len: Optional[Union[List[int], int]] = None,
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: Optional[Union[List[int], int]] = None,
    # If return logprobs, the token ids to return logprob for.
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
     # Whether to stream output
    stream: bool = False,
) -> Union[Dict, Iterator[Dict]]:
    pass
```

- generate output format

  ```json
  {
      "text": " Paris. It is located in the", ## output seq
      "output_ids": [12095, 13, 1084, 374, 7407, 304],
       "meta_info": {
          "id": "d8e57938583e45eb979db5eb5e8959a0",
          "finish_reason": {
              "type": "length",
              "length": 6
          },
          "prompt_tokens": 5,
          "completion_tokens": 6,
          "cached_tokens": 0,
          "cache_miss_count": 0,
          "e2e_latency": 0.8139922618865967
      }
  }
  ```

- use example

```python
from sgl_jax.srt.entrypoints.engine import Engine
if __name__ == '__main__':
    engine = Engine(model_path = 'Qwen/Qwen-7B-Chat', trust_remote_code = True, dist_init_addr = '0.0.0.0:10011', nnodes = 1 , tp_size = 4, device = 'tpu' ,random_seed = 3, node_rank = 0, mem_fraction_static = 0.4, chunked_prefill_size = 8192, download_dir = '/tmp', dtype = 'bfloat16', precompile_bs_paddings = [64], max_running_requests = 64, skip_server_warmup = True, attention_backend = 'fa',precompile_token_paddings = [8192], page_size = 64 ,log_requests = True, log_requests_level = 3)
    output = engine.generate(prompt = ['您好', "hello"], sampling_params = {"n",2, "temperature": 0.7})
    print(len(list(output)), output)
```

#### `vllm Sample` vs `sgl_jax Sample`

##### Fields Discussion

- `seed`: It is set by tunix sampler but is not used when sampling. Will it be used in the future?
   - Answer: It will be used in the future. But now it is used to test. It is not sure that whether the deterministic inference will be required or not by algorithm engineers. So here we decide to support deterministic sampling for every request.
- `presence_penalty` & `frequency_penalty`: Will they be used in the future?
   - Answer: Support them.
- `logprobs`: Does it mean logprobs of `top_number+1` for every output position? 1 means it include output_id's logprob.
   - Answer: No, only output_ids' logprobs are required. But the output_ids' logprobs will be recalculated due to accuracy problem between inference framework and training framework. So these logprobs are not required.

  > From vLLM: Number of log probabilities to return per output token. When set to
    `None`, no probability is returned. If set to a non-`None` value, the
    result includes the log probabilities of the specified number of most
    likely tokens, as well as the chosen tokens. Note that the implementation
    follows the OpenAI API: The API will always return the log probability of
    the sampled token, so there may be up to `logprobs+1` elements in the
    response. When set to -1, return all `vocab_size` log probabilities.

- `prompt_logprobs`: Does it mean logprobs of `top_number` for every prompt position?
   - Answer: No, they are not required.
  > From vLLM: Number of log probabilities to return per prompt token.
    When set to -1, return all `vocab_size` log probabilities.

Note:

- `repetition_penalty`, `temperature`, `top_p`, `top_k`, `min_p` and `max_tokens` will be set by `get_default_sampling_params()`.
- `get_default_sampling_params()` will be supported in `Engine`.

| Fields                        | vllm                          | tunix 设置 vllm               | sgl_jax                    |
|------------------------------|-------------------------------|-------------------------------|-------------------------------|
| n                            | 1                             | multi_sampling=1             | n:int = 1, to check                        |
| best_of                      | 1                             |                              |         lack, not to support because tunix does not use it|
| _real_n                      | None                          |                              |lack, not to support because tunix does not use it|
| presence_penalty             | 0.0                           |                              |lack, not to support because tunix does not use it|
| frequency_penalty            | 0.0                           |                              |lack, not to support because tunix does not use it|
| repetition_penalty           | get_default_sampling_params() |                              |to support|
| temperature                  | get_default_sampling_params() | temperature                   | temperature:float=1.0         |
| top_p                        | get_default_sampling_params() | top_p                         | top_p:float=1.0               |
| top_k                        | get_default_sampling_params() | top_k                         | top_k:int=1.0                 |
| min_p                        | get_default_sampling_params() |                               | min_p:float=0.0               |
| seed                         | tunix sets but not used when sampling  |                      |lack, not to support because tunix does not use it|
| stop                         | None                          |                               |None, to check                |                               |
| stop_token_ids               | None                          | [self.tokenizer.eos_id()]     | None: to check                |
| ignore_eos                   | False                         |                               |False, to check               |                               |
| max_tokens                   | get_default_sampling_params() | max_generation_steps          | max_new_tokens                |
| min_tokens                   | 0                             |                               | min_new_tokens:int=0, to check|
| logprobs                     | None                          | 1                             | tunix does not need it, it will be recalculated.|
| prompt_logprobs              | None                          | 1                             | tunix does not need them|
| detokenize                   | True                          | False                         | to support                    |
| skip_special_tokens          | True                          | True                          | skip_special_tokens:bool=True, to check |
| spaces_between_special_tokens| True                          |                               | spaces_between_special_tokens:bool=True, to check |
| logits_processors            | None                          |                               |lack, not to support because tunix does not use it |                               |
| include_stop_str_in_output   | False                         |                               |lack, not to support because tunix uses token_ids |                               |
| truncate_prompt_tokens       | None                          |                               |lack, not to support because tunix disables it with None |                               |
| output_kind                  | RequestOutputKind.CUMULATIVE  |                               |lack, but output is cumulative|                               |
| output_text_buffer_length    | 0                             |                               |lack, not to support because tunix does not use it |                               |
| guided_decoding              | None                          |                               |lack, not to support because tunix does not use it |                               |
| logit_bias                   | None                          |                                |None, not to support because tunix does not use it |                               |
| allowed_token_ids            | None                          |                               | tunix does not need it|
| extra_args                   | None                          |                               |lack, not to support because tunix does not use it |                               |
| bad_words                    | None                          |                               |lack, not to support because tunix does not use it |                               |
| _bad_words_token_ids         | None                          |                               |lack, not to support because tunix does not use it |                               |

#### Single Process for Pathways

Pathways mode is prior to multi controller jax mode, and Pathways requires the sgl-jax Engine runs in a single process. But now the sgl-jax Engine runs in several processes. So we need to support single process. In order to minimum the code modifications, we choose use multi threads to replace multi processes.

```python3
def pathways_available() -> bool:
    try:
        import pathwaysutils

        return True
    except ImportError:
        return False

def _launch_subprocesses_or_threads(
    if pathways_available():
        return _launch_threads(server_args, port_args)
    else:
        return _launch_subprocesses(server_args, port_args)
```


## Discussion

D1. Will beam search be required in the future?
Sglang has a MR[https://github.com/sgl-project/sglang/pull/3066] to support it, but it has no progress.
  - Answer: Currently it is not required.

## Test

1. Add unittest for `generate()` and `update_params()` API
2. Replace VanillaRollout with SglJaxRollout to run GRPO
3. Keep the result with RL baseline
