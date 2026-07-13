"""
Usage:
python3 test/srt/run_eval.py --port 30000 --eval-name mmlu --num-examples 10
"""

import argparse
import json
import os
import time

from eval.simple_eval_common import (
    ChatCompletionSampler,
    CompletionSampler,
    make_report,
    set_ulimit,
)

# Sampling params the SGLang server accepts but the OpenAI chat-completion API
# does not expose as native kwargs. They ride in `extra_body`, which the OpenAI
# client merges into the request body so the server reads them at the JSON top
# level. Keeping the OpenAI-vs-SGLang split here (rather than inside the sampler)
# means callers forward a flat generation config and routing lives in one place.
SGLANG_EXTRA_SAMPLING_PARAMS = (
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "frequency_penalty",
    "seed",
)


def build_extra_body(args) -> dict | None:
    """Assemble the OpenAI ``extra_body`` from a flat generation config.

    Collects the SGLang-only sampling params plus ``chat_template_kwargs``
    (e.g. thinking-mode toggles). Returns ``None`` when nothing is set so the
    sampler omits ``extra_body`` entirely.
    """
    extra_body: dict = {}
    chat_template_kwargs = getattr(args, "chat_template_kwargs", None)
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    for name in SGLANG_EXTRA_SAMPLING_PARAMS:
        value = getattr(args, name, None)
        if value is not None:
            extra_body[name] = value
    return extra_body or None


def build_sampler(args, base_url: str):
    api = getattr(args, "api", "chat")
    max_tokens = getattr(args, "max_tokens", None)
    top_p = getattr(args, "top_p", None)
    common_kwargs = dict(
        model=getattr(args, "model", None),
        max_tokens=2048 if max_tokens is None else max_tokens,
        base_url=base_url,
        temperature=getattr(args, "temperature", 0.0),
    )
    if api == "completion":
        return CompletionSampler(
            **common_kwargs,
            top_p=1.0 if top_p is None else top_p,
            stop=getattr(args, "stop", ["Question", "Assistant:", "<|separator|>"]),
        )
    if api == "chat":
        return ChatCompletionSampler(
            **common_kwargs,
            top_p=top_p,
            extra_body=build_extra_body(args),
        )
    raise ValueError(f"Unsupported eval API: {api!r}")


def get_gsm8k_num_shots(args) -> int:
    if getattr(args, "api", "chat") != "completion":
        return 0
    return getattr(args, "num_shots", 0)


def run_eval(args):
    set_ulimit()

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    base_url = f"{args.base_url}/v1" if args.base_url else f"http://{args.host}:{args.port}/v1"

    if args.eval_name == "mmlu":
        from eval.simple_eval_mmlu import MMLUEval

        filename = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        eval_obj = MMLUEval(filename, args.num_examples, args.num_threads)
    elif args.eval_name == "sglang_mmlu":
        from eval.sglang_mmlu import SglangMMLUEval

        filename = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        eval_obj = SglangMMLUEval(filename, args.num_examples, args.num_threads)
    elif args.eval_name == "math":
        from eval.simple_eval_math import MathEval

        equality_checker = ChatCompletionSampler(model="gpt-4-turbo")

        filename = "https://openaipublic.blob.core.windows.net/simple-evals/math_test.csv"
        eval_obj = MathEval(filename, equality_checker, args.num_examples, args.num_threads)
    elif args.eval_name == "mgsm":
        from eval.simple_eval_mgsm import MGSMEval

        eval_obj = MGSMEval(args.num_examples, args.num_threads)
    elif args.eval_name == "mgsm_en":
        from eval.simple_eval_mgsm import MGSMEval

        eval_obj = MGSMEval(args.num_examples, args.num_threads, languages=["en"])
    elif args.eval_name == "gpqa":
        from eval.simple_eval_gpqa import GPQAEval

        filename = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
        eval_obj = GPQAEval(filename, args.num_examples, args.num_threads)
    elif args.eval_name == "humaneval":
        from eval.simple_eval_humaneval import HumanEval

        eval_obj = HumanEval(args.num_examples, args.num_threads)
    elif args.eval_name == "gsm8k":
        from eval.simple_eval_gsm8k import GSM8KEval

        eval_obj = GSM8KEval(
            args.num_examples,
            args.num_threads,
            num_shots=get_gsm8k_num_shots(args),
        )
    elif args.eval_name == "aime25":
        from eval.simple_eval_aime25 import AIME25Eval

        eval_obj = AIME25Eval(args.num_examples, args.num_threads)
    elif args.eval_name == "aime26":
        from eval.simple_eval_aime26 import AIME26Eval

        eval_obj = AIME26Eval(args.num_examples, args.num_threads)
    elif args.eval_name == "csimpleqa":
        from eval.simple_eval_csimpleqa import ChineseSimpleQAEval

        # Self-grading: same served endpoint, deterministic, only one of A/B/C
        # is needed so an 8-token cap is more than enough.
        grader = ChatCompletionSampler(
            base_url=base_url,
            model=args.model,
            temperature=0.0,
            max_tokens=8,
        )
        eval_obj = ChineseSimpleQAEval(grader, args.num_examples, args.num_threads)
    else:
        raise ValueError(f"Invalid eval name: {args.eval_name}")

    chat_template_kwargs = dict(getattr(args, "chat_template_kwargs", {}) or {})
    if getattr(args, "enable_thinking", False):
        chat_template_kwargs["enable_thinking"] = True
    args.chat_template_kwargs = chat_template_kwargs or None
    sampler = build_sampler(args, base_url)

    # Run eval
    tic = time.perf_counter()
    result = eval_obj(sampler)
    latency = time.perf_counter() - tic

    # Dump reports
    metrics = result.metrics | {"score": result.score}
    file_stem = f"{args.eval_name}_{sampler.model.replace('/', '_')}"
    report_filename = f"/tmp/{file_stem}.html"
    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(make_report(result))
    metrics = result.metrics | {"score": result.score}
    print(metrics)
    result_filename = f"/tmp/{file_stem}.json"
    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print(f"Writing results to {result_filename}")

    # Print results
    print(f"Total latency: {latency:.3f} s")
    print(f"Score: {metrics['score']:.3f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0.")
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    )
    parser.add_argument("--eval-name", type=str, default="mmlu")
    parser.add_argument("--api", choices=("chat", "completion"), default="chat")
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--num-threads", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--topp", dest="top_p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--enable-thinking",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="Enable thinking mode for evals (true/false).",
    )
    args = parser.parse_args()

    run_eval(args)
