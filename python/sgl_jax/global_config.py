"""Global configurations"""

import os


class GlobalConfig:
    """
    Store some global constants.

    See also python/sglang/srt/managers/schedule_batch.py::global_server_args_dict, which stores
    many global runtime arguments as well.
    """

    def __init__(self):
        # Verbosity level
        # 0: do not output anything
        # 2: output final text after every run
        self.verbosity = 0

        # Default backend of the language
        self.default_backend = None

        # Runtime constants: New generation token ratio estimation
        self.default_init_new_token_ratio = float(
            os.environ.get("SGLANG_INIT_NEW_TOKEN_RATIO", 0.7)
        )
        self.default_min_new_token_ratio_factor = float(
            os.environ.get("SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR", 0.14)
        )
        self.default_new_token_ratio_decay_steps = float(
            os.environ.get("SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS", 600)
        )

        # Runtime constants: others
        self.retract_decode_steps = 20
        self.flashinfer_workspace_size = os.environ.get(
            "FLASHINFER_WORKSPACE_SIZE", 384 * 1024 * 1024
        )

        # Output tokenization configs
        self.skip_special_tokens_in_output = True
        self.spaces_between_special_tokens_in_out = True

        # Language frontend interpreter optimization configs
        self.enable_precache_with_tracing = True
        self.enable_parallel_encoding = True

        # Control whether row parallel linear/MoE activation needs to be reduce-scattered.
        # 128 is a conservative number. 8 is the minimum number in order to fully utilize
        # VPU sub-lanes
        self.tpu_scatter_min_local_size = int(os.environ.get("TPU_SCATTER_MIN_LOCAL_SIZE", 128))


global_config = GlobalConfig()
