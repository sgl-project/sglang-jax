"""Opt-in TPU smoke test: set MODEL_PATH, TP_SIZE, and optionally EP_SIZE."""

import os

import pytest


@pytest.mark.skipif(not os.environ.get("MODEL_PATH"), reason="Set MODEL_PATH to an INT4 checkpoint")
def test_kimi_int4_e2e_prompt():
    # Import only when opted in: engine initialization changes process state.
    from sgl_jax.srt.entrypoints.engine import Engine

    with Engine(
        model_path=os.environ["MODEL_PATH"],
        trust_remote_code=True,
        tp_size=int(os.environ.get("TP_SIZE", "8")),
        ep_size=int(os.environ.get("EP_SIZE", "1")),
        moe_backend="epmoe",
        device="tpu",
        dtype="bfloat16",
        attention_backend="fa",
        mem_fraction_static=0.8,
        page_size=64,
        max_running_requests=1,
        skip_server_warmup=True,
    ) as engine:
        for prompt, expected in [
            ("The capital of France is", "paris"),
            ("Calculate: 25 * 4 = ", "100"),
        ]:
            result = engine.generate(
                prompt, sampling_params={"temperature": 0.0, "max_new_tokens": 64}
            )
            assert expected in result["text"].lower(), result
