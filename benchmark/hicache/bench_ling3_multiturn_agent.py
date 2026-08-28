"""Validate Ling-3 multi-turn quality and cache-first DP affinity.

This probe exercises the user-facing OpenAI chat API with independent agent
sessions. Every session carries a long, unique case file, grows over four turns,
and must recall exact facts at the end. The server reports cached-token counts
and its actual scheduler-assigned DP rank, allowing the probe to verify that
later turns both hit Radix cache and stay on the cache-holding rank.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests


@dataclass
class AgentSession:
    index: int
    codename: str
    owner: str
    region: str
    budget: str
    checkpoint: str
    messages: list[dict[str, str]] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ling-3 multi-turn Agent quality + Radix/DP-affinity probe."
    )
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model", default="ling3-tiny")
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--expected-dp-size", type=int, default=8)
    parser.add_argument("--min-quality-rate", type=float, default=0.95)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _build_sessions(count: int) -> list[AgentSession]:
    owners = ("Ada", "Lin", "Mira", "Noor", "Owen", "Pia", "Ravi", "Zoe")
    regions = ("Osaka", "Lima", "Accra", "Riga", "Perth", "Quito", "Seoul", "Tunis")
    sessions = []
    for index in range(count):
        session = AgentSession(
            index=index,
            codename=f"ORBIT-{index:03d}",
            owner=f"{owners[index % len(owners)]}-{index:03d}",
            region=f"{regions[index % len(regions)]}-{index:03d}",
            budget=f"{7300 + index * 17}",
            checkpoint=f"GATE-{(index * 13 + 7) % 997:03d}",
        )
        background = " ".join(
            f"Archive note {note:03d} for {session.codename} is background evidence only."
            for note in range(96)
        )
        system = (
            f"You are the case-memory agent for unique session {session.codename}. "
            "Never mix facts between sessions. Retain updates across all turns. "
            f"Initial owner={session.owner}; region={session.region}; "
            f"approved budget={session.budget}. {background}"
        )
        session.messages.append({"role": "system", "content": system})
        sessions.append(session)
    return sessions


def _turn_prompt(session: AgentSession, turn: int) -> tuple[str, tuple[str, ...]]:
    if turn == 0:
        return (
            f"Open case {session.codename}. Reply with CASE_ACCEPTED and the codename.",
            ("case_accepted", session.codename.lower()),
        )
    if turn == 1:
        return (
            "Update the case status to ACTIVE. In one sentence, state the owner and status.",
            (session.owner.lower(), "active"),
        )
    if turn == 2:
        return (
            f"Record checkpoint {session.checkpoint}. Reply with the codename and checkpoint.",
            (session.codename.lower(), session.checkpoint.lower()),
        )
    if turn == 3:
        expected = (
            session.codename.lower(),
            session.owner.lower(),
            "active",
            session.checkpoint.lower(),
            session.budget.lower(),
            session.region.lower(),
        )
        return (
            "Audit the full case. Return one compact line containing codename, owner, "
            "status, checkpoint, approved budget, and region.",
            expected,
        )
    raise ValueError(f"unsupported turn {turn}")


def _request_turn(
    session: AgentSession,
    turn: int,
    *,
    url: str,
    model: str,
    max_tokens: int,
) -> dict:
    prompt, expected = _turn_prompt(session, turn)
    session.messages.append({"role": "user", "content": prompt})
    start = time.perf_counter()
    response = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": session.messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
            "separate_reasoning": True,
        },
        timeout=900,
    )
    latency = time.perf_counter() - start
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    normalized = content.lower()
    quality_ok = all(value in normalized for value in expected)
    cached_tokens = (
        (body.get("usage", {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        )
        or 0
    )
    dp_rank = choice.get("dp_rank")
    if dp_rank is None:
        raise AssertionError("chat response did not expose scheduler-assigned dp_rank")

    session.messages.append({"role": "assistant", "content": content})
    result = {
        "turn": turn + 1,
        "dp_rank": int(dp_rank),
        "prompt_tokens": int(body["usage"]["prompt_tokens"]),
        "cached_tokens": int(cached_tokens),
        "completion_tokens": int(body["usage"]["completion_tokens"]),
        "finish_reason": choice.get("finish_reason"),
        "quality_ok": quality_ok,
        "expected": list(expected),
        "content": content,
        "reasoning_chars": len(reasoning),
        "latency_seconds": latency,
    }
    session.turns.append(result)
    return result


def _server_contract(url: str, expected_dp_size: int) -> dict:
    response = requests.get(f"{url}/get_server_info", timeout=30)
    response.raise_for_status()
    info = response.json()
    assert info["dp_size"] == expected_dp_size
    assert info["disable_radix_cache"] is False
    assert info["enable_unified_radix_tree"] is True
    assert info["enable_recurrent_extra_buffer"] is True
    assert info["disable_overlap_schedule"] is False
    assert info["dp_schedule_policy"] == "cache_aware"
    assert info["enable_cache_report"] is True
    return info


def main():
    args = parse_args()
    info = _server_contract(args.server_url, args.expected_dp_size)
    flush = requests.post(f"{args.server_url}/flush_cache", timeout=120)
    flush.raise_for_status()

    sessions = _build_sessions(args.sessions)
    round_summaries = []
    for turn in range(4):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [
                pool.submit(
                    _request_turn,
                    session,
                    turn,
                    url=args.server_url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                )
                for session in sessions
            ]
            results = [future.result() for future in futures]
        round_summaries.append(
            {
                "turn": turn + 1,
                "wall_seconds": time.perf_counter() - start,
                "quality_rate": sum(result["quality_ok"] for result in results)
                / len(results),
                "cached_tokens_total": sum(result["cached_tokens"] for result in results),
                "cached_tokens_p50": statistics.median(
                    result["cached_tokens"] for result in results
                ),
                "dp_rank_counts": {
                    str(rank): sum(result["dp_rank"] == rank for result in results)
                    for rank in range(args.expected_dp_size)
                },
            }
        )

    later_turns = [turn for session in sessions for turn in session.turns[1:]]
    sticky = [
        turn["dp_rank"] == session.turns[0]["dp_rank"]
        for session in sessions
        for turn in session.turns[1:]
    ]
    cache_hits = [turn["cached_tokens"] > 0 for turn in later_turns]
    quality = [turn["quality_ok"] for session in sessions for turn in session.turns]
    summary = {
        "sessions": args.sessions,
        "turns_per_session": 4,
        "parallel": args.parallel,
        "page_size": info["page_size"],
        "recurrent_track_interval": info["recurrent_track_interval"],
        "dp_schedule_policy": info["dp_schedule_policy"],
        "quality_rate": sum(quality) / len(quality),
        "later_turn_cache_hit_rate": sum(cache_hits) / len(cache_hits),
        "later_turn_rank_sticky_rate": sum(sticky) / len(sticky),
        "rounds": round_summaries,
        "session_results": [
            {"codename": session.codename, "turns": session.turns}
            for session in sessions
        ],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    compact_summary = {
        key: value for key, value in summary.items() if key != "session_results"
    }
    print(json.dumps(compact_summary, indent=2, sort_keys=True))

    assert summary["quality_rate"] >= args.min_quality_rate, summary["quality_rate"]
    assert summary["later_turn_cache_hit_rate"] == 1.0, summary[
        "later_turn_cache_hit_rate"
    ]
    assert summary["later_turn_rank_sticky_rate"] == 1.0, summary[
        "later_turn_rank_sticky_rate"
    ]
    print("[Ling-3 multi-turn agent] PASS")


if __name__ == "__main__":
    main()
