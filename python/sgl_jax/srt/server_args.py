"""The arguments of the server."""

import argparse
import dataclasses
import json
import logging
import os
import tempfile

import jax

from sgl_jax.srt.function_call.function_call_parser import FunctionCallParser
from sgl_jax.srt.hf_transformers_utils import (
    check_gguf_file,
    download_from_hf,
    get_config,
)
from sgl_jax.srt.reasoning_parser import ReasoningParser
from sgl_jax.srt.utils.common_utils import (
    LORA_TARGET_ALL_MODULES,
    SUPPORTED_LORA_TARGET_MODULES,
    is_remote_url,
    is_valid_ipv6_address,
    nullable_str,
)

logger = logging.getLogger(__name__)

GRAMMAR_BACKEND_CHOICES = ["llguidance", "none"]


@dataclasses.dataclass
class ServerArgs:
    # Model and tokenizer
    model_path: str
    tokenizer_path: str | None = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    model_loader_extra_config: str = "{}"
    trust_remote_code: bool = False
    context_length: int | None = None
    is_embedding: bool = False
    revision: str | None = None
    model_impl: str = "auto"
    model_layer_nums: int | None = None

    # HTTP server
    host: str = "127.0.0.1"
    port: int = 30000
    skip_server_warmup: bool = False
    warmups: str | None = None

    # Quantization and data type
    dtype: str = "auto"
    quantization: str | None = None
    quantization_param_path: str | None = None
    quantization_config_path: str | None = None
    kv_cache_dtype: str = "auto"

    # Memory and scheduling
    mem_fraction_static: float | None = None
    max_running_requests: int | None = None
    max_total_tokens: int | None = None
    max_prefill_tokens: int = 16384
    chunked_prefill_size: int | None = None
    enable_mixed_chunk: bool = False
    schedule_policy: str = "fcfs"
    schedule_conservativeness: float = 1.0
    page_size: int = 1
    swa_full_tokens_ratio: float = 0.8
    disable_hybrid_swa_memory: bool = False

    # Runtime options
    device: str | None = None
    device_indexes: list[int] | None = None
    tp_size: int = 1
    ep_size: int = 1
    stream_interval: int = 1
    stream_output: bool = False
    random_seed: int | None = None
    constrained_json_whitespace_pattern: str | None = None
    constrained_json_disable_any_whitespace: bool = False
    watchdog_timeout: float = 300
    dist_timeout: int | None = None  # timeout for distributed initialization
    download_dir: str | None = None
    sleep_on_idle: bool = False

    # Data parallel
    dp_size: int = 1

    # Logging
    log_level: str = "info"
    log_level_http: str | None = None
    log_requests: bool = False
    log_requests_level: int = 0
    crash_dump_folder: str | None = None
    show_time_cost: bool = False
    bucket_time_to_first_token: list[float] | None = None
    bucket_inter_token_latency: list[float] | None = None
    bucket_e2e_request_latency: list[float] | None = None
    decode_log_interval: int = 40
    enable_request_time_stats_logging: bool = False
    kv_events_config: str | None = None

    # API related
    api_key: str | None = None
    served_model_name: str | None = None
    chat_template: Optional[str] = None
    file_storage_path: str = "sglang_storage"
    enable_cache_report: bool = False
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None

    # Multi-node distributed serving
    dist_init_addr: str | None = None
    nnodes: int = 1
    node_rank: int = 0

    # Model override args in JSON
    json_model_override_args: str = "{}"
    preferred_sampling_params: str | None = None

    # Optimization/debug options
    disable_radix_cache: bool = False
    allow_auto_truncate: bool = False
    enable_tokenizer_batch_encode: bool = False
    disable_overlap_schedule: bool = False
    enable_precision_tracer: bool = False

    # Kernel backend
    attention_backend: str | None = "fa"
    moe_backend: str = "epmoe"

    grammar_backend: str | None = None

    max_seq_len: int = 4096

    precompile_token_paddings: list[int] | None = None
    precompile_bs_paddings: list[int] | None = None

    disable_precompile: bool = False

    # Speculative decoding
    speculative_algorithm: str | None = None
    speculative_draft_model_path: str | None = None
    speculative_draft_model_revision: str | None = None
    speculative_num_steps: int = 4
    speculative_eagle_topk: int = 5
    speculative_num_draft_tokens: int = 4
    speculative_accept_threshold_single: float = 1.0
    speculative_accept_threshold_acc: float = 1.0

    # For deterministic sampling
    enable_deterministic_sampling: bool = False
    enable_single_process: bool = False
    enable_nan_detection: bool = False

    # For sampling
    use_sort_for_toppk_minp: bool = False

    # LoRA
    enable_lora: bool | None = None
    max_lora_rank: int | None = None
    lora_target_modules: set[str] | list[str] | None = None
    lora_paths: dict[str, str] | list[dict[str, str]] | list[str] | list | None = None
    max_loaded_loras: int | None = None
    max_loras_per_batch: int = 8
    lora_eviction_policy: str = "lru"
    enable_static_lora: bool | None = None
    lora_scaling: float | None = None

    # For engine
    enable_engine_loop_run_forever_daemon: bool | None = None

    # Multimodal
    multimodal: bool = False

    enable_return_routed_experts: bool = False

    def __post_init__(self):
        # Set missing default values
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        # update device
        if self.device:
            platform_env = os.environ.get("JAX_PLATFORMS", self.device)
            assert (
                self.device == platform_env
            ), f"device {self.device} is not consistent with 'JAX_PLATFORMS' {platform_env}"
        else:
            platform_env = os.environ.get("JAX_PLATFORMS", "")
            if platform_env != "":
                self.device = platform_env
            else:
                self.device = "tpu"

        if self.served_model_name is None:
            self.served_model_name = self.model_path

        if self.random_seed is None:
            self.random_seed = 42

        # Set mem fraction static
        if self.mem_fraction_static is None:
            if self.device == "cpu":
                self.mem_fraction_static = 0.5 / jax.process_count()
            else:
                self.mem_fraction_static = 0.88

        # Set chunked prefill size
        if self.chunked_prefill_size is None:
            self.chunked_prefill_size = 4096

        # GGUF
        if (self.load_format == "auto" or self.load_format == "gguf") and check_gguf_file(
            self.model_path
        ):
            self.quantization = self.load_format = "gguf"

        if is_remote_url(self.model_path):
            self.load_format = "remote"

        if (
            self.enable_precision_tracer
            and self.chunked_prefill_size is not None
            and self.chunked_prefill_size > 0
        ):
            logger.warning(
                "Chunked prefill is enabled, but precision tracer is also enabled. "
                "This may cause incorrect precision tracer results."
                "Disabling chunked prefill."
            )
            self.chunked_prefill_size = -1

        # Disable radix cache for multimodal mode (e.g., UMT5 Encoder without KV cache)
        if self.multimodal and not self.disable_radix_cache:
            logger.info("Multimodal mode enabled, disabling radix cache")
            self.disable_radix_cache = True

        if self.grammar_backend is None:
            self.grammar_backend = "llguidance"

        # Normalize speculative_algorithm: treat empty string as None
        if isinstance(self.speculative_algorithm, str) and self.speculative_algorithm.strip() == "":
            self.speculative_algorithm = None

        os.environ["SGLANG_ENABLE_DETERMINISTIC_SAMPLING"] = (
            "1" if self.enable_deterministic_sampling else "0"
        )

        if self.nnodes > 1 and self.device_indexes is not None:
            logger.warning("In a multi-machine scenario, device_indexes will be set to None.")
            self.device_indexes = None
        if self.multimodal:
            self.model_path = download_from_hf(self.model_path, allow_patterns=None)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # Model and tokenizer
        parser.add_argument(
            "--model-path",
            "--model",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=ServerArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=ServerArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action="store_true",
            help="If set, skip init tokenizer and pass input_ids in generate request.",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=ServerArgs.load_format,
            choices=[
                "auto",
                "pt",
                "safetensors",
                "npcache",
                "dummy",
                "sharded_state",
                "gguf",
                "bitsandbytes",
                "layered",
                "remote",
            ],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the jax format if safetensors format "
            "is not available. "
            '"jax" will load the weights in the jax format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in jax format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling."
            '"gguf" will load the weights in the gguf format. '
            '"bitsandbytes" will load the weights using bitsandbytes '
            "quantization."
            '"layered" loads weights layer by layer so that one can quantize a '
            "layer before loading another to make the peak memory envelope "
            "smaller.",
        )
        parser.add_argument(
            "--model-loader-extra-config",
            type=str,
            help="Extra config for model loader. "
            "This will be passed to the model loader corresponding to the chosen load_format.",
            default=ServerArgs.model_loader_extra_config,
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        )
        parser.add_argument(
            "--context-length",
            type=int,
            default=ServerArgs.context_length,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
        )
        parser.add_argument(
            "--is-embedding",
            action="store_true",
            help="Whether to use a CausalLM as an embedding model.",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="The specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--model-impl",
            type=str,
            default=ServerArgs.model_impl,
            help="Which implementation of the model to use.\n\n"
            '* "auto" will try to use the SGLang implementation if it exists '
            "and fall back to the Transformers implementation if no SGLang "
            "implementation is available.\n"
            '* "sglang" will use the SGLang model implementation.\n'
            '* "transformers" will use the Transformers model '
            "implementation.\n",
        )
        parser.add_argument(
            "--model-layer-nums",
            type=int,
            default=ServerArgs.model_layer_nums,
            help="Number of model layers to load and use for inference. If not specified, uses the value from model config.",
        )
        parser.add_argument(
            "--grammar-backend",
            type=str,
            choices=GRAMMAR_BACKEND_CHOICES,
            default=ServerArgs.grammar_backend,
            help="Choose the backend for grammar-guided decoding.",
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=ServerArgs.chat_template,
            help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
        )
        
        # HTTP server
        parser.add_argument(
            "--host",
            type=str,
            default=ServerArgs.host,
            help="The host of the HTTP server.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=ServerArgs.port,
            help="The port of the HTTP server.",
        )
        parser.add_argument(
            "--skip-server-warmup",
            action="store_true",
            help="If set, skip warmup.",
        )
        parser.add_argument(
            "--warmups",
            type=str,
            required=False,
            help="Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 "
            "will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
        )

        # Quantization and data type
        parser.add_argument(
            "--dtype",
            type=str,
            default=ServerArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="Data type for model weights and activations.\n\n"
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            "BF16 precision for BF16 models.\n"
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.',
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=ServerArgs.quantization,
            choices=[
                "awq",
                "fp8",
                "gptq",
                "marlin",
                "gptq_marlin",
                "awq_marlin",
                "bitsandbytes",
                "gguf",
                "modelopt",
                "modelopt_fp4",
                "petit_nvfp4",
                "w8a8_int8",
                "w8a8_fp8",
                "moe_wna16",
                "qoq",
                "w4afp8",
            ],
            help="The quantization method.",
        )
        parser.add_argument(
            "--quantization-param-path",
            type=nullable_str,
            default=None,
            help="Path to the JSON file containing the KV cache "
            "scaling factors. This should generally be supplied, when "
            "KV cache dtype is FP8. Otherwise, KV cache scaling factors "
            "default to 1.0, which may cause accuracy issues. ",
        )
        parser.add_argument(
            "--quantization-config-path",
            type=str,
            default=ServerArgs.quantization_config_path,
            help="Path to quantization config YAML file. Can be an absolute path, "
            "relative path, or just a filename (will look up in built-in configs). "
            "Built-in configs: int8.yaml, fp8.yaml, fp8_w8a8.yaml",
        )
        parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            default=ServerArgs.kv_cache_dtype,
            choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16"],
            help='Data type for kv cache storage. "auto" will use model data type. "fp8_e5m2" and "fp8_e4m3" is supported for CUDA 11.8+.',
        )

        # Memory and scheduling
        parser.add_argument(
            "--mem-fraction-static",
            type=float,
            default=ServerArgs.mem_fraction_static,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-running-requests",
            type=int,
            default=ServerArgs.max_running_requests,
            help="The maximum number of running requests.",
        )
        parser.add_argument(
            "--max-total-tokens",
            type=int,
            default=ServerArgs.max_total_tokens,
            help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. "
            "This option is typically used for development and debugging purposes.",
        )
        parser.add_argument(
            "--chunked-prefill-size",
            type=int,
            default=ServerArgs.chunked_prefill_size,
            help="The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.",
        )
        parser.add_argument(
            "--enable-mixed-chunk",
            action="store_true",
            help="Enabling mixing prefill and decode in a batch when using chunked prefill.",
        )
        parser.add_argument(
            "--max-prefill-tokens",
            type=int,
            default=ServerArgs.max_prefill_tokens,
            help="The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length.",
        )
        parser.add_argument(
            "--disable-overlap-schedule",
            action="store_true",
            help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
        )
        parser.add_argument(
            "--schedule-policy",
            type=str,
            default=ServerArgs.schedule_policy,
            choices=["lpm", "random", "fcfs", "dfs-weight"],
            help="The scheduling policy of the requests.",
        )
        parser.add_argument(
            "--schedule-conservativeness",
            type=float,
            default=ServerArgs.schedule_conservativeness,
            help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            default=ServerArgs.page_size,
            help="The number of tokens in a page.",
        )
        parser.add_argument(
            "--swa-full-tokens-ratio",
            type=float,
            default=ServerArgs.swa_full_tokens_ratio,
            help="The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the number of swa:full layers. It should be between 0 and 1. "
            "E.g. 0.5 means if each swa layer has 50 tokens, then each full layer has 100 tokens.",
        )
        parser.add_argument(
            "--disable-hybrid-swa-memory",
            action="store_true",
            help="Disable the hybrid SWA memory.",
        )

        # Runtime options
        parser.add_argument(
            "--device",
            type=str,
            default=ServerArgs.device,
            help="The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). Defaults to auto-detection if not specified.",
        )

        parser.add_argument(
            "--device-indexes",
            type=int,
            nargs="+",
            help="The device indexes to use build mesh. Defaults is all if not specified.",
        )

        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=ServerArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--ep-size",
            type=int,
            default=ServerArgs.ep_size,
            help="The expert parallelism size",
        )
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=ServerArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--stream-output",
            action="store_true",
            help="Whether to output as a sequence of disjoint segments.",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=ServerArgs.random_seed,
            help="The random seed.",
        )
        parser.add_argument(
            "--constrained-json-whitespace-pattern",
            type=str,
            default=ServerArgs.constrained_json_whitespace_pattern,
            help="(llguidance backends only) Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [\n\t ]*",
        )
        parser.add_argument(
            "--constrained-json-disable-any-whitespace",
            action="store_true",
            help="(llguidance backends only) Enforce compact representation in JSON constrained output.",
        )
        parser.add_argument(
            "--watchdog-timeout",
            type=float,
            default=ServerArgs.watchdog_timeout,
            help="Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=ServerArgs.dist_timeout,
            help="Set timeout for jax.distributed initialization.",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=ServerArgs.download_dir,
            help="Model download directory for huggingface.",
        )
        parser.add_argument(
            "--sleep-on-idle",
            action="store_true",
            help="Reduce CPU usage when sglang is idle.",
        )

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=ServerArgs.log_level_http,
            help="The logging level of HTTP server. If not set, reuse --log-level by default.",
        )
        parser.add_argument(
            "--log-requests",
            action="store_true",
            help="Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level",
        )
        parser.add_argument(
            "--log-requests-level",
            type=int,
            default=0,
            help="0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output.",
            choices=[0, 1, 2, 3],
        )
        parser.add_argument(
            "--crash-dump-folder",
            type=str,
            default=ServerArgs.crash_dump_folder,
            help="Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled.",
        )
        parser.add_argument(
            "--show-time-cost",
            action="store_true",
            help="Show time cost of custom marks.",
        )
        parser.add_argument(
            "--enable-metrics",
            action="store_true",
            help="Enable log prometheus metrics.",
        )
        parser.add_argument(
            "--enable-metrics-for-all-schedulers",
            action="store_true",
            help="Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) "
            "to record request metrics separately. This is especially useful when dp_attention is enabled, as "
            "otherwise all metrics appear to come from TP 0.",
        )
        parser.add_argument(
            "--bucket-time-to-first-token",
            type=float,
            nargs="+",
            default=ServerArgs.bucket_time_to_first_token,
            help="The buckets of time to first token, specified as a list of floats.",
        )
        parser.add_argument(
            "--bucket-inter-token-latency",
            type=float,
            nargs="+",
            default=ServerArgs.bucket_inter_token_latency,
            help="The buckets of inter-token latency, specified as a list of floats.",
        )
        parser.add_argument(
            "--bucket-e2e-request-latency",
            type=float,
            nargs="+",
            default=ServerArgs.bucket_e2e_request_latency,
            help="The buckets of end-to-end request latency, specified as a list of floats.",
        )
        parser.add_argument(
            "--decode-log-interval",
            type=int,
            default=ServerArgs.decode_log_interval,
            help="The log interval of decode batch.",
        )
        parser.add_argument(
            "--enable-request-time-stats-logging",
            action="store_true",
            default=ServerArgs.enable_request_time_stats_logging,
            help="Enable per request time stats logging",
        )
        parser.add_argument(
            "--kv-events-config",
            type=str,
            default=None,
            help="Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used.",
        )

        # API related
        parser.add_argument(
            "--api-key",
            type=str,
            default=ServerArgs.api_key,
            help="Set API key of the server. It is also used in the OpenAI API compatible server.",
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=ServerArgs.served_model_name,
            help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
        )
        parser.add_argument(
            "--file-storage-path",
            type=str,
            default=ServerArgs.file_storage_path,
            help="The path of the file storage in backend.",
        )
        parser.add_argument(
            "--enable-cache-report",
            action="store_true",
            help="Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
        )
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            choices=list(ReasoningParser.DetectorMap.keys()),
            default=ServerArgs.reasoning_parser,
            help=f"Specify the parser for reasoning models, supported parsers are: {list(ReasoningParser.DetectorMap.keys())}.",
        )
        tool_call_parser_choices = list(FunctionCallParser.ToolCallParserEnum.keys())
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            choices=tool_call_parser_choices,
            default=ServerArgs.tool_call_parser,
            help=f"Specify the parser for handling tool-call interactions. Options include: {tool_call_parser_choices}.",
        )

        # Data parallelism
        parser.add_argument(
            "--data-parallel-size",
            "--dp-size",
            type=int,
            default=ServerArgs.dp_size,
            help="The data parallelism size.",
        )

        # Multi-node distributed serving
        parser.add_argument(
            "--dist-init-addr",
            type=str,
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
        )
        parser.add_argument(
            "--nnodes", type=int, default=ServerArgs.nnodes, help="The number of nodes."
        )
        parser.add_argument(
            "--node-rank", type=int, default=ServerArgs.node_rank, help="The node rank."
        )

        # Model override args
        parser.add_argument(
            "--json-model-override-args",
            type=str,
            help="A dictionary in JSON string format used to override default model configurations.",
            default=ServerArgs.json_model_override_args,
        )
        parser.add_argument(
            "--preferred-sampling-params",
            type=str,
            help="json-formatted sampling settings that will be returned in /get_model_info",
        )

        # Optimization/debug options
        parser.add_argument(
            "--disable-radix-cache",
            action="store_true",
            help="Disable RadixAttention for prefix caching.",
        )
        parser.add_argument(
            "--allow-auto-truncate",
            action="store_true",
            help="Allow automatically truncating requests that exceed the maximum input length instead of returning an error.",
        )
        parser.add_argument(
            "--enable-tokenizer-batch-encode",
            action="store_true",
            help="Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds.",
        )
        parser.add_argument(
            "--enable-precision-tracer",
            action="store_true",
            help="Enable precision tracer for debugging tensor values. May have performance impact.",
        )

        parser.add_argument(
            "--max-seq-len",
            type=int,
            default=ServerArgs.max_seq_len,
            help="maximum sequence length",
        )
        parser.add_argument(
            "--precompile-token-paddings",
            type=int,
            nargs="+",
            help="Set the list of token buckets for jax jit",
        )
        parser.add_argument(
            "--precompile-bs-paddings",
            type=int,
            nargs="+",
            help="Set the list of batch sizes buckets for jax jit",
        )
        parser.add_argument(
            "--disable-precompile",
            action="store_true",
            help="whether disable precompile",
        )
        # Kernel backend
        parser.add_argument(
            "--attention-backend",
            type=str,
            choices=[
                "native",
                "fa",
            ],
            default=ServerArgs.attention_backend,
            help="Choose the kernels for attention layers.",
        )
        parser.add_argument(
            "--moe-backend",
            type=str,
            choices=["epmoe", "fused", "auto"],
            default=ServerArgs.moe_backend,
            help="The backend to use for MoE models.",
        )

        parser.add_argument(
            "--enable-nan-detection",
            action="store_true",
            help="Enable the NaN detection for debugging purposes.",
        )

        # Speculative decoding
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE"],
            help="Speculative algorithm.",
            default=ServerArgs.speculative_algorithm,
        )
        parser.add_argument(
            "--speculative-draft-model-path",
            "--speculative-draft-model",
            type=str,
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
            default=ServerArgs.speculative_draft_model_path,
        )
        parser.add_argument(
            "--speculative-draft-model-revision",
            type=str,
            default=None,
            help="The specific draft model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--speculative-num-steps",
            type=int,
            help="The number of steps sampled from draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_steps,
        )
        parser.add_argument(
            "--speculative-eagle-topk",
            type=int,
            help="The number of tokens sampled from the draft model in eagle2 each step.",
            default=ServerArgs.speculative_eagle_topk,
        )
        parser.add_argument(
            "--speculative-num-draft-tokens",
            type=int,
            help="The number of tokens sampled from the draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_draft_tokens,
        )
        parser.add_argument(
            "--speculative-accept-threshold-single",
            type=float,
            help="Accept a draft token if its probability in the target model is greater than this threshold.",
            default=ServerArgs.speculative_accept_threshold_single,
        )
        parser.add_argument(
            "--speculative-accept-threshold-acc",
            type=float,
            help="The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc).",
            default=ServerArgs.speculative_accept_threshold_acc,
        )

        # For deterministic sampling
        parser.add_argument(
            "--enable-deterministic-sampling",
            action="store_true",
            help="Enable deterministic sampling",
        )

        parser.add_argument(
            "--enable-single-process",
            action="store_true",
            help="Enable run the engine with single process.",
        )

        # For sampling
        parser.add_argument(
            "--use-sort-for-toppk-minp",
            action="store_true",
            help="Use jnp.sort to deal with top_k, top_p and min_p, which improves the grades for math-500 but increase precompile time a lot",
        )

        parser.add_argument(
            "--multimodal",
            action="store_true",
            help="Enable multimodal HTTP server.",
        )

        # LoRA
        parser.add_argument(
            "--enable-lora",
            action="store_true",
            help="Enable LoRA support. LoRA (Low-Rank Adaptation) allows serving multiple fine-tuned models with minimal overhead.",
        )
        parser.add_argument(
            "--lora-paths",
            type=str,
            nargs="*",
            default=None,
            help="List of LoRA adapters to preload. Can be local paths or HuggingFace repo IDs. "
            "Format: 'adapter_name=path' or just 'path' (will use basename as name).",
        )
        parser.add_argument(
            "--max-loras-per-batch",
            type=int,
            default=8,
            help="Maximum number of different LoRA adapters that can be used in a single batch.",
        )
        parser.add_argument(
            "--max-lora-rank",
            type=int,
            default=None,
            help="Maximum LoRA rank to support. If not specified, will be determined from loaded adapters.",
        )
        parser.add_argument(
            "--max-loaded-loras",
            type=int,
            default=None,
            help="Maximum number of LoRA adapters to keep loaded in memory.",
        )
        parser.add_argument(
            "--lora-target-modules",
            type=str,
            choices=SUPPORTED_LORA_TARGET_MODULES + [LORA_TARGET_ALL_MODULES],
            nargs="*",
            default=None,
            help="List of module names to apply LoRA to. If not specified, will be determined from adapters. If not specified, "
            "it will be automatically inferred from the adapters provided in --lora-paths. If 'all' is specified, "
            "all supported modules will be targeted.",
        )
        parser.add_argument(
            "--lora-eviction-policy",
            type=str,
            default="lru",
            choices=["lru"],
            help="Policy for evicting LoRA adapters when max_loaded_loras is reached.",
        )
        parser.add_argument(
            "--enable-static-lora",
            action="store_true",
            help="Enable static LoRA support for RL, and it is different from the combination of enable-lora and max-loras-per-batch = 1",
        )
        parser.add_argument(
            "--lora-scaling",
            type=float,
            default=ServerArgs.lora_scaling,
            help="Lora scaling is required for static LoRA, scaling = alpha/rank",
        )
        parser.add_argument(
            "--enable-engine-loop-run-forever-daemon",
            action="store_true",
            help="Run engine loop forever when engine.async_generate is called in other threads, this is used in Tunix",
        )
        parser.add_argument(
            "--enable-return-routed-experts",
            action="store_true",
            help="Enable returning routed experts of each layer with responses.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.tp_size = args.tensor_parallel_size
        args.dp_size = args.data_parallel_size
        if cls is ServerArgs and getattr(args, "multimodal", False):
            from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs

            return MultimodalServerArgs.from_cli_args(args)

        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "ServerArgs":
        """
        Create ServerArgs from command line arguments.

        Args:
            argv: Command line arguments. If None or empty, uses sys.argv[1:].

        Returns:
            The server arguments.
        """
        import sys

        parser = argparse.ArgumentParser()
        cls.add_cli_args(parser)
        from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs

        MultimodalServerArgs.add_cli_args(parser)
        return cls.from_cli_args(parser.parse_args(argv or sys.argv[1:]))

    def url(self):
        if is_valid_ipv6_address(self.host):
            return f"http://[{self.host}]:{self.port}"
        else:
            return f"http://{self.host}:{self.port}"

    def get_hf_config(self):
        kwargs = {}
        hf_config = get_config(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            model_override_args=json.loads(self.json_model_override_args),
            **kwargs,
        )
        return hf_config

    def check_server_args(self):
        assert (self.tp_size) % self.nnodes == 0, "tp_size must be divisible by number of nodes"

        # Check chunked prefill
        # Skip validation if chunked prefill is disabled (i.e., size <= 0).
        if self.chunked_prefill_size > 0:
            assert (
                self.chunked_prefill_size % self.page_size == 0
            ), "chunked_prefill_size must be divisible by page_size"

        # Check LoRA configuration
        self.check_lora_server_args()

        # Disallow overlap scheduler when speculative decoding is enabled
        if self.speculative_algorithm is not None and not self.disable_overlap_schedule:
            raise ValueError(
                "Speculative decoding does not support overlap scheduler. "
                "Please pass --disable-overlap-schedule when using --speculative-algorithm."
            )

    def check_lora_server_args(self):
        """Validate and normalize LoRA-related server arguments."""
        # Import LoRARef here to avoid circular imports
        from sgl_jax.srt.lora.lora_registry import LoRARef

        if self.lora_paths:
            self.enable_lora = True
            logger.info("Auto-enabling LoRA because lora_paths are provided")

        if not self.enable_lora and not self.enable_static_lora:
            return

        assert not (
            self.enable_lora and self.enable_static_lora
        ), f"{self.enable_lora} and {self.enable_static_lora} can not be enable at the same time"

        self.enable_lora = True

        # Validate max_loras_per_batch
        assert self.max_loras_per_batch > 0, "max_loras_per_batch must be positive"

        # Expand target modules
        if self.lora_target_modules:
            self.lora_target_modules = set(self.lora_target_modules)
            if "all" in self.lora_target_modules:
                assert (
                    len(self.lora_target_modules) == 1
                ), "If 'all' is specified in --lora-target-modules, it should be the only module specified."
                self.lora_target_modules = set(SUPPORTED_LORA_TARGET_MODULES)

        # Ensure sufficient information is provided for LoRA initialization.
        assert self.lora_paths or (
            self.max_lora_rank and self.lora_target_modules
        ), "When no initial --lora-paths is provided, you need to specify both --max-lora-rank and --lora-target-modules for LoRA initialization."

        def check_static_lora_args():
            assert (
                self.lora_scaling is not None
            ), "lora_scaling is required when enable-static-lora is enabled"

            assert (
                self.lora_paths is None
            ), "lora-paths is not required when enable-static-lora is enabled"
            assert (
                self.max_loras_per_batch == 1
            ), "max-loras-per-batch is required to be 1 when enable-static-lora is enabled"

        def check_dynamic_lora_args():
            # Normalize lora_paths to List[LoRARef]
            if self.lora_paths is not None:
                normalized_lora_refs = []

                # Normalize lora_paths to List[LoRARef]
                if self.lora_paths is not None:
                    normalized_lora_refs = []

                    if isinstance(self.lora_paths, dict):
                        # Dict format: {"name": "path", ...}
                        for name, path in self.lora_paths.items():
                            if name == "0":
                                raise ValueError(
                                    "This key(0) is a server-reserved symbol, used for requests that do not go through LoRA."
                                )
                            normalized_lora_refs.append(
                                LoRARef(lora_name=name, lora_path=path, pinned=True)
                            )
                    elif isinstance(self.lora_paths, list):
                        for item in self.lora_paths:
                            if isinstance(item, str):
                                # String format: "name=path" or just "path"
                                if "=" in item:
                                    name, path = item.split("=", 1)
                                    normalized_lora_refs.append(
                                        LoRARef(
                                            lora_name=name.strip(),
                                            lora_path=path.strip(),
                                            pinned=True,
                                        )
                                    )
                                else:
                                    # Use basename as name
                                    import os

                                    name = os.path.basename(item.rstrip("/"))
                                    normalized_lora_refs.append(
                                        LoRARef(lora_name=name, lora_path=item, pinned=True)
                                    )
                            elif isinstance(item, dict):
                                # Dict format in list: {"name": "adapter1", "path": "/path/to/adapter"}
                                name = item.get("name") or item.get("lora_name")
                                path = item.get("path") or item.get("lora_path")
                                pinned = item.get("pinned", True)
                                normalized_lora_refs.append(
                                    LoRARef(lora_name=name, lora_path=path, pinned=pinned)
                                )
                            elif hasattr(item, "lora_name"):
                                # Already a LoRARef object
                                normalized_lora_refs.append(item)
                            else:
                                raise ValueError(f"Unsupported lora_paths item format: {item}")

                    self.lora_paths = normalized_lora_refs

                    # Validate max_loaded_loras
                    if self.max_loaded_loras is not None:
                        assert (
                            self.max_loaded_loras >= self.max_loras_per_batch
                        ), "max_loaded_loras must be >= max_loras_per_batch"

                    logger.info(
                        "Loaded %d LoRA adapters: %s",
                        len(self.lora_paths),
                        [ref.lora_name for ref in self.lora_paths],
                    )

        if self.enable_static_lora:
            check_static_lora_args()
        else:
            check_dynamic_lora_args()


ZMQ_TCP_PORT_DELTA = 233


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for tokenizer to receive inputs from detokenizer (zmq)
    tokenizer_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str
    # The ipc filename for detokenizer to receive inputs from scheduler (zmq)
    detokenizer_ipc_name: str

    # The addr is used to broadcast recv_reqs from scheduler_0 to others
    pub_sub_addr: str
    # The addr is used to ensure pubilisher and subscribers are ready
    pub_sub_sync_addr: str

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str

    @staticmethod
    def init_new(server_args, dp_rank: int | None = None) -> "PortArgs":
        if server_args.nnodes > 1:
            dist_init_addr = server_args.dist_init_addr.split(":")
            dist_init_host, dist_init_port = dist_init_addr
            port_base = int(dist_init_port) + 1

        return PortArgs(
            tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            pub_sub_addr=(
                f"tcp://{dist_init_host}:{port_base + 4}" if server_args.nnodes > 1 else None
            ),
            pub_sub_sync_addr=(
                f"tcp://{dist_init_host}:{port_base + 5}" if server_args.nnodes > 1 else None
            ),
        )
