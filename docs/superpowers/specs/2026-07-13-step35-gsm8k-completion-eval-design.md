# Step-3.5 GSM8K Completion Evaluation Design

## Problem

`test_step3p5_mtp_e2e.py` describes its GSM8K check as a port of upstream
SGLang, but the two tests currently measure different tasks. Upstream runs a
five-shot completion evaluation with a 0.83 score threshold. The local test
uses the generic zero-shot chat-completion evaluator while expecting a score
from the five-shot completion benchmark.

The mismatch is observable independently of speculative decoding: the local
chat evaluation scores about 0.55 for both speculative and non-speculative
servers, while the existing five-shot `/generate` benchmark scores about 0.9.
Consequently, the local accuracy failure does not identify a serving,
speculative decoding, chunked prefill, or concurrency regression.

## Goals

- Make the Step-3.5 MTP E2E GSM8K test measure the same five-shot completion
  task as upstream SGLang.
- Keep the generic zero-shot chat GSM8K evaluation unchanged for existing
  callers.
- Preserve the optional strict comparison against a non-speculative baseline.
- Test prompt construction, request routing, and answer extraction on CPU.

## Non-Goals

- Changing model chat templates.
- Changing speculative decoding or chunked prefill serving code.
- Replacing the benchmark CLI or changing its output format.
- Making all `run_eval` evaluations use completion requests.

## Design

Add an explicit completion mode to the local evaluation harness. `run_eval`
will continue to use `ChatCompletionSampler` by default. When a caller passes
`api="completion"`, it will construct a completion sampler and configure the
GSM8K evaluator with the requested number of few-shot examples.

The completion GSM8K prompt will use the canonical structure already used by
the repository benchmark:

```text
Question: <example question>
Answer: <example worked answer>

...

Question: <test question>
Answer:
```

The sampler will send that prompt through the OpenAI-compatible completion
endpoint. The evaluator will extract the last numeric answer from the model
response, matching the established benchmark behavior. The existing chat
mode will retain its current instruction, `Answer:` parser, and conversation
reporting.

`test_step3p5_mtp_e2e.py` will explicitly set `api="completion"` and
`num_shots=5`, with `max_tokens=512`. Its fallback score floor will be 0.83,
matching the upstream Step-3.5 test. When `SGLANG_GSM8K_BASELINE` is supplied,
the stricter spec-versus-greedy comparison remains authoritative.

## Data Flow

1. The Step-3.5 test builds `run_eval` arguments with completion mode and five
   shots.
2. `run_eval` selects the completion sampler and configures `GSM8KEval`.
3. `GSM8KEval` builds one five-shot prompt per held-out test example.
4. The completion sampler sends the prompt to the server's completion API.
5. The evaluator extracts the final number, compares it with the GSM8K target,
   and aggregates the same metrics/report shape returned today.

## Error Handling

Network retry behavior remains in the sampler layer. Evaluation exceptions
continue to produce an empty prediction for that example, so transport
failures reduce the score rather than being mistaken for correct answers.
Unsupported `api` values fail immediately with a clear `ValueError`.

## Tests

CPU tests will prove:

- completion prompt construction contains exactly the requested demonstrations
  followed by the held-out question;
- completion mode selects the completion endpoint and forwards deterministic
  generation parameters;
- completion answer extraction uses the final numeric answer;
- default GSM8K chat mode remains unchanged;
- the Step-3.5 E2E test requests five-shot completion mode and uses the upstream
  fallback threshold.

After focused tests pass, run the registered CPU unit suite and pre-commit.
The TPU acceptance is one bundled run: Step-3.5 speculative and greedy servers
on the same commit, 200 GSM8K questions, five-shot completion, concurrency 32,
plus the existing pool-idle and Weng checks for Problem 2.

## Acceptance Criteria

- Existing callers that do not set `api` continue using chat completion.
- Step-3.5 E2E uses five-shot completion and no longer reports the known
  zero-shot chat score as a serving regression.
- The speculative score remains within the configured tolerance of a supplied
  greedy baseline, or exceeds 0.83 when no baseline is supplied.
- Problem 2's slot ownership behavior and tests remain unchanged.
