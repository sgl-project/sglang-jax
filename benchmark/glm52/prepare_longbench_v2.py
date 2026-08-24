"""Build a tokenizer-exact LongBench v2 workload for GLM-5.2 serving.

The workload is intentionally data preparation only.  It selects the LongBench
v2 ``Code repo QA`` and ``Financial`` sub-domains, tokenizes each source with
the same tokenizer directory used by the serving job, and emits 64 exact-size
requests.  Each request contains a 128K-token cacheable prefix followed by a
1K-token measured extension; the 1K output length is recorded for the serving
client rather than fabricated in the input data.

Long sources may contribute more than one non-overlapping window.  Selection
prefers the first window from distinct sources before later windows and is then
stable-hashed, so rebuilding from the pinned dataset revision is deterministic.
"""

from __future__ import annotations

import argparse
import array
import dataclasses
import gzip
import hashlib
import heapq
import json
import shutil
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_ID = "zai-org/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATASET_FILENAME = "data.json"
CODE_REPO_QA = "Code repo QA"
FINANCIAL = "Financial"
DEFAULT_QUOTAS = {CODE_REPO_QA: 32, FINANCIAL: 32}


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    prefix_len: int = 131_072
    extend_len: int = 1_024
    output_len: int = 1_024
    code_quota: int = 32
    financial_quota: int = 32
    selection_seed: str = "glm52-longbench-v2-code-financial-v1"

    @property
    def total_input_len(self) -> int:
        return self.prefix_len + self.extend_len

    @property
    def quotas(self) -> dict[str, int]:
        return {CODE_REPO_QA: self.code_quota, FINANCIAL: self.financial_quota}


@dataclasses.dataclass(frozen=True)
class Candidate:
    priority: int
    source_id: str
    domain: str
    sub_domain: str
    difficulty: str
    length_bucket: str
    question: str
    choices: dict[str, str]
    answer: str
    window_index: int
    context_token_start: int
    context_token_end: int
    source_context_tokens: int
    suffix_tokens: int
    source_context_sha256: str
    input_ids: tuple[int, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_sha256(token_ids: Sequence[int]) -> str:
    payload = array.array("I", token_ids)
    if payload.itemsize != 4:
        raise RuntimeError(f"unexpected unsigned-int width: {payload.itemsize}")
    if sys.byteorder != "little":
        payload.byteswap()
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _encode(tokenizer: Any, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("tokenizer returned more than one encoded sequence")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _format_suffix(row: Mapping[str, Any]) -> str:
    return (
        "\n\nQuestion: "
        + str(row["question"])
        + "\nA. "
        + str(row["choice_A"])
        + "\nB. "
        + str(row["choice_B"])
        + "\nC. "
        + str(row["choice_C"])
        + "\nD. "
        + str(row["choice_D"])
        + "\nAnswer:"
    )


def _priority(seed: str, source_id: str, window_index: int) -> int:
    # Window index is the high-order key.  This maximizes source diversity by
    # selecting a first window before a second, then a stable hash removes any
    # dependence on source-file order.
    digest = hashlib.sha256(
        f"{seed}\0{source_id}\0{window_index}".encode("utf-8")
    ).digest()
    return (window_index << 256) | int.from_bytes(digest, "big")


def _candidate_windows(
    row: Mapping[str, Any], tokenizer: Any, config: BuildConfig
) -> tuple[list[Candidate], str | None]:
    sub_domain = str(row.get("sub_domain", ""))
    if sub_domain not in config.quotas:
        return [], "unselected_sub_domain"

    suffix_ids = _encode(tokenizer, _format_suffix(row))
    if len(suffix_ids) > config.extend_len:
        return [], "suffix_exceeds_extend"

    context = str(row["context"])
    context_ids = _encode(tokenizer, context)
    context_budget = config.total_input_len - len(suffix_ids)
    if len(context_ids) < context_budget:
        return [], "context_too_short"

    source_id = str(row["_id"])
    source_context_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()
    choices = {
        label: str(row[f"choice_{label}"]) for label in ("A", "B", "C", "D")
    }
    candidates = []
    for window_index, start in enumerate(
        range(0, len(context_ids) - context_budget + 1, context_budget)
    ):
        end = start + context_budget
        input_ids = tuple(context_ids[start:end] + suffix_ids)
        if len(input_ids) != config.total_input_len:
            raise AssertionError((len(input_ids), config.total_input_len))
        candidates.append(
            Candidate(
                priority=_priority(config.selection_seed, source_id, window_index),
                source_id=source_id,
                domain=str(row.get("domain", "")),
                sub_domain=sub_domain,
                difficulty=str(row.get("difficulty", "")),
                length_bucket=str(row.get("length", "")),
                question=str(row["question"]),
                choices=choices,
                answer=str(row["answer"]),
                window_index=window_index,
                context_token_start=start,
                context_token_end=end,
                source_context_tokens=len(context_ids),
                suffix_tokens=len(suffix_ids),
                source_context_sha256=source_context_sha256,
                input_ids=input_ids,
            )
        )
    return candidates, None


def _select_candidates(
    candidates: Iterable[Candidate], quotas: Mapping[str, int]
) -> list[Candidate]:
    # A max-heap encoded through negative priorities retains only the globally
    # smallest stable priorities for each sub-domain.
    heaps: dict[str, list[tuple[int, int, Candidate]]] = {
        sub_domain: [] for sub_domain in quotas
    }
    sequence = 0
    for candidate in candidates:
        quota = quotas.get(candidate.sub_domain)
        if quota is None or quota <= 0:
            continue
        heap = heaps[candidate.sub_domain]
        entry = (-candidate.priority, sequence, candidate)
        sequence += 1
        if len(heap) < quota:
            heapq.heappush(heap, entry)
        elif candidate.priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    missing = {
        sub_domain: quota - len(heaps[sub_domain])
        for sub_domain, quota in quotas.items()
        if len(heaps[sub_domain]) < quota
    }
    if missing:
        raise ValueError(f"not enough tokenizer-eligible windows: {missing}")

    selected = [entry[2] for heap in heaps.values() for entry in heap]
    return sorted(selected, key=lambda candidate: (candidate.sub_domain, candidate.priority))


def _iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import ijson
    except ImportError as error:
        raise RuntimeError(
            "ijson is required to stream LongBench v2 without loading its "
            "~465 MB JSON file into memory"
        ) from error

    with path.open("rb") as stream:
        yield from ijson.items(stream, "item")


def _download_source(dataset_id: str, revision: str, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface-hub is required to download LongBench v2") from error

    return Path(
        hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=DATASET_FILENAME,
            revision=revision,
            cache_dir=cache_dir,
        )
    )


def _load_tokenizer(tokenizer_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required to load the GLM-5.2 tokenizer") from error

    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )


def _tokenizer_identity(tokenizer: Any, tokenizer_path: str) -> dict[str, Any]:
    return {
        "path": tokenizer_path,
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", tokenizer_path)),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "model_max_length": int(getattr(tokenizer, "model_max_length", -1)),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_requests(
    path: Path, selected: Sequence[Candidate], config: BuildConfig
) -> None:
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_stream, compresslevel=6, mtime=0
        ) as compressed:
            for request_index, candidate in enumerate(selected):
                input_ids = list(candidate.input_ids)
                row = {
                    "request_id": f"longbench-v2-{request_index:03d}",
                    "source_id": candidate.source_id,
                    "domain": candidate.domain,
                    "sub_domain": candidate.sub_domain,
                    "difficulty": candidate.difficulty,
                    "length_bucket": candidate.length_bucket,
                    "question": candidate.question,
                    "choices": candidate.choices,
                    "answer": candidate.answer,
                    "window_index": candidate.window_index,
                    "context_token_start": candidate.context_token_start,
                    "context_token_end": candidate.context_token_end,
                    "source_context_tokens": candidate.source_context_tokens,
                    "suffix_tokens": candidate.suffix_tokens,
                    "prefix_len": config.prefix_len,
                    "extend_len": config.extend_len,
                    "output_len": config.output_len,
                    "source_context_sha256": candidate.source_context_sha256,
                    "input_ids_sha256_u32le": _token_sha256(input_ids),
                    "prefix_sha256_u32le": _token_sha256(
                        input_ids[: config.prefix_len]
                    ),
                    "input_ids": input_ids,
                }
                payload = json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
                compressed.write(payload + b"\n")


def _build(
    *,
    source_json: Path,
    tokenizer: Any,
    tokenizer_path: str,
    output_dir: Path,
    dataset_id: str,
    dataset_revision: str,
    builder_source_commit: str,
    config: BuildConfig,
    keep_source_json: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_rows = Counter()
    eligibility = Counter()
    eligible_sources = Counter()
    candidate_windows = Counter()

    def iter_candidates() -> Iterator[Candidate]:
        for row in _iter_json_array(source_json):
            sub_domain = str(row.get("sub_domain", ""))
            if sub_domain not in config.quotas:
                continue
            filtered_rows[sub_domain] += 1
            candidates, reason = _candidate_windows(row, tokenizer, config)
            if reason is not None:
                eligibility[f"{sub_domain}:{reason}"] += 1
                continue
            eligible_sources[sub_domain] += 1
            candidate_windows[sub_domain] += len(candidates)
            yield from candidates

    # _select_candidates keeps only quota-sized heaps.  We therefore never
    # retain token windows for the whole 465 MB source dataset in memory.
    selected = _select_candidates(iter_candidates(), config.quotas)
    selected_counts = Counter(candidate.sub_domain for candidate in selected)
    if selected_counts != Counter(config.quotas):
        raise AssertionError((selected_counts, config.quotas))
    if len({_token_sha256(candidate.input_ids) for candidate in selected}) != len(
        selected
    ):
        raise ValueError("selected workload contains duplicate token sequences")

    requests_path = output_dir / "requests.jsonl.gz"
    _write_requests(requests_path, selected, config)

    source_sha256 = _sha256_file(source_json)
    source_copy = None
    if keep_source_json:
        source_dir = output_dir / "source"
        source_dir.mkdir()
        source_copy = source_dir / DATASET_FILENAME
        shutil.copyfile(source_json, source_copy)
        if _sha256_file(source_copy) != source_sha256:
            raise IOError("source JSON checksum changed while copying")

    selected_source_counts = Counter(candidate.source_id for candidate in selected)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder": {"source_commit": builder_source_commit},
        "dataset": {
            "id": dataset_id,
            "revision": dataset_revision,
            "filename": DATASET_FILENAME,
            "sha256": source_sha256,
            "saved_source_json": source_copy is not None,
        },
        "tokenizer": _tokenizer_identity(tokenizer, tokenizer_path),
        "workload": {
            "prefix_len": config.prefix_len,
            "extend_len": config.extend_len,
            "total_input_len": config.total_input_len,
            "output_len": config.output_len,
            "concurrency": sum(config.quotas.values()),
            "quotas": config.quotas,
            "selection_seed": config.selection_seed,
            "selection_order": "window_index_then_sha256(seed,source_id,window_index)",
            "window_stride": "non-overlapping context budget",
            "suffix_contract": "question, choices, and Answer marker fit in final 1024 tokens",
        },
        "audit": {
            "filtered_source_rows": dict(filtered_rows),
            "eligible_sources": dict(eligible_sources),
            "eligible_windows": dict(candidate_windows),
            "ineligible_rows": dict(eligibility),
            "selected_windows": dict(selected_counts),
            "selected_unique_sources": len(selected_source_counts),
            "selected_windows_per_source_max": max(selected_source_counts.values()),
        },
        "artifacts": {
            "requests": {
                "path": requests_path.name,
                "bytes": requests_path.stat().st_size,
                "sha256": _sha256_file(requests_path),
                "format": "gzip JSON Lines; each row contains exact input_ids",
            },
        },
    }
    if source_copy is not None:
        manifest["artifacts"]["source"] = {
            "path": str(source_copy.relative_to(output_dir)),
            "bytes": source_copy.stat().st_size,
            "sha256": source_sha256,
        }

    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    success = {
        "schema_version": 1,
        "manifest_sha256": _sha256_file(manifest_path),
        "requests_sha256": manifest["artifacts"]["requests"]["sha256"],
        "request_count": len(selected),
        "total_input_len": config.total_input_len,
        "output_len": config.output_len,
    }
    _write_json(output_dir / "_SUCCESS.json", success)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--builder-source-commit", default="unknown")
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--download-cache", type=Path, default=Path("/tmp/hf-cache"))
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-len", type=int, default=131_072)
    parser.add_argument("--extend-len", type=int, default=1_024)
    parser.add_argument("--output-len", type=int, default=1_024)
    parser.add_argument("--code-quota", type=int, default=32)
    parser.add_argument("--financial-quota", type=int, default=32)
    parser.add_argument(
        "--selection-seed", default="glm52-longbench-v2-code-financial-v1"
    )
    parser.add_argument(
        "--omit-source-json",
        action="store_true",
        help="do not retain the pinned raw LongBench v2 data.json in the output",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = BuildConfig(
        prefix_len=args.prefix_len,
        extend_len=args.extend_len,
        output_len=args.output_len,
        code_quota=args.code_quota,
        financial_quota=args.financial_quota,
        selection_seed=args.selection_seed,
    )
    if min(
        config.prefix_len,
        config.extend_len,
        config.output_len,
        config.code_quota,
        config.financial_quota,
    ) <= 0:
        raise ValueError("token lengths and quotas must all be positive")

    source_json = args.source_json or _download_source(
        args.dataset_id, args.dataset_revision, args.download_cache
    )
    tokenizer = _load_tokenizer(args.tokenizer_path)
    manifest = _build(
        source_json=source_json,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        builder_source_commit=args.builder_source_commit,
        config=config,
        keep_source_json=not args.omit_source_json,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
