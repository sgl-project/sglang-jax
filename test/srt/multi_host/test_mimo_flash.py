from multi_host_suite import AccuracyCase, ModelRun, MultiHostSuite, PerfCase


def get_suites() -> list[MultiHostSuite]:
    return [
        MultiHostSuite(
            name="mimo-flash-pref-test",
            runs=[
                ModelRun(
                    launch_profile="launch_profiles/mimo-flash-v6e-4x4.yaml",
                    cases=[
                        PerfCase(
                            name="mimo-flash-benchmark",
                            input_len=16384,
                            output_len=1024,
                            num_prompts=256,
                            max_concurrency=64,
                            request_rate=100,
                            seed=12345,
                            flush_cache=True,
                        ),
                        AccuracyCase(
                            name="mimo-flash-gsm8k",
                            dataset="gsm8k",
                            model_id="XiaomiMiMo/MiMo-V2-Flash",
                            eval_batch_size=32,
                            generation_config={"temperature": 0.8, "top_p": 0.95},
                        ),
                    ],
                )
            ],
        )
    ]
