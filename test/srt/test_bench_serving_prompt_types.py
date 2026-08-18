from sgl_jax.bench_serving import _is_multi_turn_prompt


def test_multi_turn_prompt_distinguishes_text_turns_from_token_ids():
    assert _is_multi_turn_prompt(["first question", "follow-up question"])
    assert not _is_multi_turn_prompt([101, 102, 103])
    assert not _is_multi_turn_prompt("single-turn prompt")
