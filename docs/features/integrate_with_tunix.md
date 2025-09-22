## Motivation

Integrate with tunix to make contribution to post-training.

## Solution

Sglang-jax will implement a SglangJaxRollout in tunix to generate completions.

### SglangJaxRollout

```python3
class BaseRollout(abc.ABC):
  """Base RolloutWorker."""

  @abc.abstractmethod
  def generate(
      self,
      prompts: list[str],
      rollout_config: RolloutConfig,
      **kwargs,
  ) -> RolloutOutput:
    """Generates samples from the model."""

  @abc.abstractmethod
  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the model."""

  @abc.abstractmethod
  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Updates the rollout model parameters."""

  @abc.abstractmethod
  def pad_id(self) -> int:
    """Returns the pad id."""

  @abc.abstractmethod
  def eos_id(self) -> int:
    """Returns the eos id."""

  @abc.abstractmethod
  def model(self) -> Any:
    """Returns the rollout model."""
```

### Work in Sglang-jax

1. Support `generate()` and `get_default_sampling_params()` in `Engine`.
2. Support abilities listed in the following table, like jitted `return_logprob` logic, `repetition_penalty`, etc.

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

- use example

```python
from sgl_jax.srt.entrypoints.engine import Engine
if __name__ == '__main__':
    engine=Engine(model_path= 'Qwen/Qwen-7B-Chat', trust_remote_code=True, dist_init_addr='0.0.0.0:10011', nnodes=1 , tp_size=4, device='tpu' ,random_seed=3, node_rank=0, mem_fraction_static=0.4, chunked_prefill_size=8192, download_dir='/tmp', dtype='bfloat16', precompile_bs_paddings = [64], max_running_requests = 64, skip_server_warmup=True, attention_backend='fa',precompile_token_paddings=[8192], page_size=64 ,log_requests=True, log_requests_level=3)
    output=engine.generate(prompt=['您好', "hello"], sampling_params={"n",2, "temperature": 0.7}, return_logprob=True)
    print(len(list(output)), output)
```


#### `vllm Sample` vs `sglang_jax Sample`

##### Fields Discussion

- `seed`: It is set by tunix sampler but is not used when sampling. Will it be used in the future?
- `presence_penalty` & `frequency_penalty`: Will they be used in the future?
- `logit_bias`: Will it be used in the future?
- `logprobs`: Does it mean logprobs of `top_number+1` for every output position? 1 means it include output_id's logprob.

  > From vLLM: Number of log probabilities to return per output token. When set to
    `None`, no probability is returned. If set to a non-`None` value, the
    result includes the log probabilities of the specified number of most
    likely tokens, as well as the chosen tokens. Note that the implementation
    follows the OpenAI API: The API will always return the log probability of
    the sampled token, so there may be up to `logprobs+1` elements in the
    response. When set to -1, return all `vocab_size` log probabilities.
   - Confusion1: According to the usage of logprobs in the PPO codes, it looks like we need to get the logprob of every token_id in prompt position rather than the top_num logprob.
- `prompt_logprobs`: Does it mean logprobs of `top_number` for every prompt position?
  > From vLLM: Number of log probabilities to return per prompt token.
    When set to -1, return all `vocab_size` log probabilities.
   - Confusion2: According to `get_old_per_token_logps()` implemented in `VanillaRollout` and `VllmRollout`, the logprobs of prompt_ids do not return.

Note:

- `repetition_penalty`, `temperature`, `top_p`, `top_k`, `min_p` and `max_tokens` will be set by `get_default_sampling_params()`.
- `get_default_sampling_params()` will be supported in `Engine`.

| Fields                        | vllm                          | tunix 设置 vllm               | sglang_jax                    |
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
| logprobs                     | None                          | 1                             | support, set in payload body:<br>- return_logprob = true<br>- top_logprobs_num = 1 |
| prompt_logprobs              | None                          | 1                             | support, set in payload body:<br>- return_logprob = true<br>- top_logprobs_num = 1<br>- logprob_start_len = 0 |
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
| allowed_token_ids            | None                          |                               | support, set in payload body:<br>- return_logprob = true<br>- token_ids_logprob = None |
| extra_args                   | None                          |                               |lack, not to support because tunix does not use it |                               |
| bad_words                    | None                          |                               |lack, not to support because tunix does not use it |                               |
| _bad_words_token_ids         | None                          |                               |lack, not to support because tunix does not use it |                               |


## Discussion

D1. Is beam search is required in the future?
Sglang has a MR[https://github.com/sgl-project/sglang/pull/3066] to support it, but it has no progress.

D3: For `generate`, is it enough to only have the `logprob` corresponding to each output_id? Are prompt_ids logprobs necessary?

## Test

1. Test whether sampling result is expected
2. Test `generate()` interface
3. Add unit test for SglangJaxRollout
3. Test e2e result compared with VllmRollout: PPO, GRPO