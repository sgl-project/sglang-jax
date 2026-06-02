"""MMLU-Pro PPL eval (suffix-only) — sgl_jax in-process Engine, Ling3-Flash.

Scores only the option-letter token after "Answer: " (lm-eval-harness
multiple_choice convention). Robust to DP mesh-shape numerical perturbation.

Engine flags are tuned for Ling-3-flash (237 GiB bf16) on TPU v6e 4x4,
tp=16 nnodes=4. Defaults to the accuracy-baseline config (chunked_prefill_size
=4096, mem_fraction_static=0.8, max_running_requests=256); set EVAL_EFFICIENT=1
for the faster, lower-memory variant. Reproduces the MMLU-Pro number reported in
https://github.com/primatrix/sglang-jax-private/pull/8#issuecomment-4560593346

Usage:
    DIST_INIT_ADDR=<host:port> NODE_RANK=<rank> python mmlu_pro_ppl_eval_flash.py \\
        <model_path> <data_path> '<cat1,cat2,...|empty=all>' <dp> [<pred_dump.jsonl>]
"""

import json
import os
import random
import sys

from sgl_jax.srt.entrypoints.engine import Engine

CATEGORIES = [
    "math", "physics", "chemistry", "law", "engineering", "other",
    "economics", "health", "psychology", "business", "biology",
    "philosophy", "computer_science", "history",
]
OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def load_data(data_path, category, split):
    file_path = os.path.join(data_path, split, f"{split}_{category}.jsonl")
    data = []
    with open(file_path, "r") as f:
        for line in f:
            row = json.loads(line)
            options_str = ""
            for i, opt in enumerate(row["options"]):
                if opt == "N/A":
                    continue
                options_str += f"{OPTIONS[i]}. {opt}\n"
            data.append({
                "question": row["question"],
                "answer": row["answer"],
                "options_str": options_str.strip(),
            })
    return data


def build_few_shot(val_data, indices=(0, 1, 2, 3, 4)):
    parts = []
    for idx in indices:
        item = val_data[idx]
        parts.append(
            f"Question:\n{item['question']}\nOptions:\n{item['options_str']}\n"
            f"Answer: {item['answer']}\n"
        )
    return "\n".join(parts) + "\n"


def compute_ppl(input_token_logprobs):
    log_values = [lp[0] for lp in input_token_logprobs if lp[0] is not None]
    if not log_values:
        return float("inf")
    return -sum(log_values) / len(log_values)


def longest_common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_per_prompt_starts(prefix, full_prompts, tokenizer):
    # start = common - 1: Engine returns len(full_ids) - start entries; first
    # is None placeholder, real logprobs cover full_ids[common:].
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    starts, scored_counts = [], []
    for p in full_prompts:
        full_ids = tokenizer(p, add_special_tokens=False).input_ids
        common = longest_common_prefix(prefix_ids, full_ids)
        start = max(common - 1, 0)
        starts.append(start)
        scored_counts.append(max(len(full_ids) - start - 1, 0))
    return starts, scored_counts


if __name__ == "__main__":
    model_path = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "./data/mmlu_pro"
    categories = (
        sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else CATEGORIES
    )
    dp = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    dump_path = sys.argv[5] if len(sys.argv) > 5 else ""

    node_rank = int(os.environ.get("NODE_RANK", "0"))
    dist_init_addr = os.environ.get("DIST_INIT_ADDR")
    smoke_n = int(os.environ.get("SMOKE_N", "0"))
    if dist_init_addr is None:
        raise SystemExit("DIST_INIT_ADDR env var is required (e.g. 10.31.153.25:29500)")

    # Default: accuracy-baseline config. EVAL_EFFICIENT=1 -> faster/lower-mem.
    efficient = os.environ.get("EVAL_EFFICIENT", "0") == "1"

    llm = Engine(
        model_path=model_path,
        trust_remote_code=True,
        tp_size=16,
        nnodes=4,
        node_rank=node_rank,
        dist_init_addr=dist_init_addr,
        dp_size=dp,
        device="tpu",
        dtype="bfloat16",
        mem_fraction_static=0.6 if efficient else 0.8,
        chunked_prefill_size=512 if efficient else 4096,
        page_size=256,
        max_running_requests=64 if efficient else 256,
        disable_radix_cache=True,
        precompile_bs_paddings=[1, 16, 64],
        precompile_token_paddings=(
            [16, 128, 512] if efficient else [16, 128, 512, 1024, 2048, 4096]
        ),
        context_length=4096,
        skip_server_warmup=True,
        log_level="info",
    )

    tokenizer = llm.tokenizer_manager.tokenizer

    total_correct = 0
    total_count = 0
    per_category = []
    dump_fp = open(dump_path, "w") if (node_rank == 0 and dump_path) else None

    for category in categories:
        val_data = load_data(data_path, category, "val")
        test_data = load_data(data_path, category, "test")

        if smoke_n > 0 and smoke_n < len(test_data):
            rng = random.Random(42)
            test_data = rng.sample(test_data, smoke_n)

        few_shot_str = build_few_shot(val_data)
        hint = (
            f"The following are multiple choice questions (with answers) about "
            f'{category.replace("_", " ")}. Think step by step and then finish '
            f'your answer with "The answer is (X)" where X is the correct letter '
            f"choice. If none or more than one of the options match, choose the "
            f"one that is the closest.\n\n"
        )

        all_prompts = []
        all_starts = []
        all_scored_counts = []
        golds = []
        for item in test_data:
            prefix = (
                f"{hint}{few_shot_str}"
                f"Question:\n{item['question']}\nOptions:\n{item['options_str']}\n"
                f"Answer: "
            )
            full_prompts_this_q = [prefix + opt for opt in OPTIONS]
            all_prompts.extend(full_prompts_this_q)
            starts, scored_counts = build_per_prompt_starts(
                prefix, full_prompts_this_q, tokenizer
            )
            all_starts.extend(starts)
            all_scored_counts.extend(scored_counts)
            golds.append(item["answer"])

        outputs = llm.generate(
            all_prompts,
            {"max_new_tokens": 1, "temperature": 1, "top_p": 1},
            return_logprob=True,
            logprob_start_len=all_starts,
        )

        correct = 0
        num_options = len(OPTIONS)
        for i in range(len(golds)):
            ppls = []
            for j in range(num_options):
                idx = i * num_options + j
                ppl = compute_ppl(outputs[idx]["meta_info"]["input_token_logprobs"])
                ppls.append(ppl)
            pred = OPTIONS[ppls.index(min(ppls))]
            if pred == golds[i]:
                correct += 1
            if dump_fp is not None:
                base = i * num_options
                dump_fp.write(json.dumps({
                    "category": category,
                    "idx": i,
                    "gold": golds[i],
                    "pred": pred,
                    "ppls": ppls,
                    "score_mode": "suffix_only",
                    "logprob_start_lens": all_starts[base : base + num_options],
                    "suffix_token_counts": all_scored_counts[base : base + num_options],
                }) + "\n")

        total = len(golds)
        total_correct += correct
        total_count += total
        per_category.append((category, correct, total))
        if node_rank == 0:
            print(f"[{category}] acc: {correct}/{total} = {correct / total:.4f}", flush=True)
            if dump_fp is not None:
                dump_fp.flush()

    if node_rank == 0:
        print()
        for cat, c, t in per_category:
            print(f"  {cat:<18} {c}/{t} = {c / t:.4f}")
        print(f"\n[total] acc: {total_correct}/{total_count} = {total_correct / total_count:.4f}")
        if dump_fp is not None:
            dump_fp.close()
            print(f"dumped per-question PPLs to {dump_path}")
