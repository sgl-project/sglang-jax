"""Benchmark script comparing FusedEPMoE vs EPMoE implementations.

This script performs layer-level benchmarking with synthetic weights and controlled
token distribution scenarios (random, balanced, imbalanced).

Example usage:
    # Quick test
    python benchmark/fused_moe/bench_fused_vs_epmoe.py \
        --num-experts 8 --num-experts-per-tok 2 \
        --hidden-size 1024 --intermediate-size 4096 \
        --num-tokens 512 --scenarios random

    # Using HF model config
    python benchmark/fused_moe/bench_fused_vs_epmoe.py \
        --model-path Qwen/Qwen2.5-MoE-A2.7B \
        --ep-size 8 --num-tokens 1024 2048 4096 \
        --scenarios random balanced imbalanced \
        --profile --profile-dir ./profiles/qwen

    # 4 GPUs with expert parallelism (tp=4, ep=4, tp_actual=1)
    python benchmark/fused_moe/bench_fused_vs_epmoe.py \
        --model-path Qwen/Qwen2.5-MoE-A2.7B \
        --tp-size 4 --ep-size 4

    # 8 GPUs with expert and tensor parallelism (tp=8, ep=4, tp_actual=2)
    python benchmark/fused_moe/bench_fused_vs_epmoe.py \
        --model-path Qwen/Qwen2.5-MoE-A2.7B \
        --tp-size 8 --ep-size 4
"""

import argparse
import os
import sys

import jax

# Add python directory to path for imports
benchmark_dir = os.path.dirname(os.path.abspath(__file__))  # benchmark/fused_moe
benchmark_root = os.path.dirname(benchmark_dir)  # benchmark
project_root = os.path.dirname(benchmark_root)  # sgl-jax
python_dir = os.path.join(project_root, "python")  # sgl-jax/python
sys.path.insert(0, python_dir)
sys.path.insert(0, project_root)  # For benchmark imports

from benchmark.fused_moe.benchmark_runner import MoEBenchmarkRunner  # noqa: E402
from benchmark.fused_moe.config_utils import MoEBenchmarkConfig  # noqa: E402
from benchmark.fused_moe.output_formatter import save_results  # noqa: E402
from benchmark.fused_moe.synthetic_data import create_synthetic_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark FusedEPMoE vs EPMoE implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--model-path",
        type=str,
        help="Path or name of HuggingFace model to load config from",
    )
    config_group.add_argument(
        "--manual-config",
        action="store_true",
        help="Use manual configuration (requires --num-experts, etc.)",
    )

    # Manual configuration options
    parser.add_argument("--num-experts", type=int, help="Number of experts")
    parser.add_argument("--num-experts-per-tok", type=int, help="Top-k value")
    parser.add_argument("--hidden-size", type=int, help="Hidden dimension")
    parser.add_argument("--intermediate-size", type=int, help="Intermediate dimension")
    parser.add_argument(
        "--activation",
        type=str,
        default="silu",
        choices=["silu", "gelu", "swigluoai"],
        help="Activation function",
    )

    # Distributed configuration
    parser.add_argument(
        "--ep-size",
        type=int,
        default=1,
        help="Expert parallel size (default: 1)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Total number of devices to use (default: 1)",
    )
    parser.add_argument(
        "--dist-init-addr",
        type=str,
        help="Distributed initialization address (e.g., 10.0.0.1:12345)",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Current node rank (default: 0)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="List of token counts to test (default: 512 1024 2048)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["random", "balanced", "imbalanced"],
        choices=["random", "balanced", "imbalanced"],
        help="Scenarios to test (default: all)",
    )
    parser.add_argument(
        "--imbalance-factor",
        type=float,
        default=3.0,
        help="Target imbalance factor for 'imbalanced' scenario (default: 3.0)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Warmup iterations (default: 1, only need one for JAX JIT)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=10,
        help="Benchmark iterations (default: 10)",
    )

    # Profiling
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable JAX profiler",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./profiles",
        help="Profile output directory (default: ./profiles)",
    )

    # Output
    parser.add_argument(
        "--output-format",
        type=str,
        default="both",
        choices=["csv", "markdown", "both"],
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./benchmark_results",
        help="Output file base path (default: ./benchmark_results)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate manual config
    if args.manual_config:
        required_manual = ["num_experts", "num_experts_per_tok", "hidden_size", "intermediate_size"]
        missing = [arg for arg in required_manual if getattr(args, arg) is None]
        if missing:
            parser.error(
                f"--manual-config requires: {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )

    return args


def setup_distributed(args: argparse.Namespace) -> None:
    """Initialize JAX distributed environment if needed."""
    if args.nnodes > 1:
        if not args.dist_init_addr:
            raise ValueError("--dist-init-addr is required for multi-node setup")

        print(f"Initializing distributed: nnodes={args.nnodes}, rank={args.node_rank}")
        jax.distributed.initialize(
            coordinator_address=args.dist_init_addr,
            num_processes=args.nnodes,
            process_id=args.node_rank,
        )
        print(f"Distributed initialized. Process rank: {jax.process_index()}")


def create_mesh(tp_size: int) -> jax.sharding.Mesh:
    """
    Create JAX mesh for MoE execution using create_device_mesh.

    This follows the same logic as scheduler.py. The MoE layers (FusedEPMoE and EPMoE)
    will internally compute world_size from the mesh and calculate the actual tensor
    parallel size as: tp_actual = world_size // ep_size

    Args:
        tp_size: Total number of devices to use

    Returns:
        JAX mesh with (data, tensor) axes
    """
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    mesh = create_device_mesh(
        ici_parallelism=[-1, tp_size],
        dcn_parallelism=[1, 1],
    )

    return mesh


def load_or_create_config(args: argparse.Namespace) -> MoEBenchmarkConfig:
    """Load configuration from model path or create from manual args."""
    if args.model_path:
        print(f"Loading config from model: {args.model_path}")
        config = MoEBenchmarkConfig.from_model_path(
            args.model_path,
            ep_size=args.ep_size,
            tp_size=args.tp_size,
        )
    else:
        print("Using manual configuration")
        config = MoEBenchmarkConfig(
            num_experts=args.num_experts,
            num_experts_per_tok=args.num_experts_per_tok,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            activation=args.activation,
            ep_size=args.ep_size,
            tp_size=args.tp_size,
        )

    # Validate config
    config.validate()

    if args.verbose:
        print("\n" + str(config))

    return config


def main():
    """Main execution flow."""
    args = parse_args()

    print("=" * 80)
    print("MoE Benchmark: FusedEPMoE vs EPMoE")
    print("=" * 80)

    # Setup distributed
    setup_distributed(args)

    # Create mesh
    print(f"\nCreating JAX mesh: tp_size={args.tp_size}, ep_size={args.ep_size}")
    mesh = create_mesh(args.tp_size)
    print(f"Mesh created with {len(mesh.devices.flatten())} devices")
    print(f"Mesh shape: {mesh.shape}")

    # Load configuration
    config = load_or_create_config(args)

    # Generate synthetic weights
    print("\nGenerating synthetic weights...")
    fused_weights, epmoe_weights = create_synthetic_weights(config, mesh)
    print(f"Weights generated: w1={fused_weights['w1'].shape}, w2={fused_weights['w2'].shape}")

    # Initialize benchmark runner
    print("\nInitializing benchmark runner...")
    runner = MoEBenchmarkRunner(
        config=config,
        mesh=mesh,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        verbose=args.verbose,
    )

    runner.initialize_layers(fused_weights, epmoe_weights)

    # Run benchmarks
    print("\n" + "=" * 80)
    print("Running Benchmarks")
    print("=" * 80)

    all_results = []

    for scenario in args.scenarios:
        for num_tokens in args.num_tokens:
            print(f"\n{'=' * 80}")
            print(f"Scenario: {scenario}, Tokens: {num_tokens}")
            print(f"{'=' * 80}")

            if args.profile:
                # Profile each implementation separately
                profile_dir_fused = os.path.join(
                    args.profile_dir, f"{scenario}_tokens{num_tokens}_fused"
                )
                profile_dir_epmoe = os.path.join(
                    args.profile_dir, f"{scenario}_tokens{num_tokens}_epmoe"
                )

                os.makedirs(profile_dir_fused, exist_ok=True)
                os.makedirs(profile_dir_epmoe, exist_ok=True)

                print(f"Profiling enabled: {profile_dir_fused}, {profile_dir_epmoe}")

                # Run with profiling
                jax.profiler.start_trace(profile_dir_fused)
                fused_result, _ = runner.benchmark_scenario(
                    scenario, num_tokens, args.imbalance_factor
                )
                jax.profiler.stop_trace()

                jax.profiler.start_trace(profile_dir_epmoe)
                _, epmoe_result = runner.benchmark_scenario(
                    scenario, num_tokens, args.imbalance_factor
                )
                jax.profiler.stop_trace()

                all_results.extend([fused_result, epmoe_result])

            else:
                # Run without profiling
                fused_result, epmoe_result = runner.benchmark_scenario(
                    scenario, num_tokens, args.imbalance_factor
                )
                all_results.extend([fused_result, epmoe_result])

            # Print summary
            speedup = epmoe_result.latency_mean / fused_result.latency_mean
            print("\nResults:")
            print(f"  FusedEPMoE: {fused_result.latency_mean:.4f} ms (mean)")
            print(f"  EPMoE:      {epmoe_result.latency_mean:.4f} ms (mean)")
            print(f"  Speedup:    {speedup:.2f}x")
            print(f"  Imbalance:  {fused_result.max_imbalance:.2f}x")

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    save_results(all_results, args.output_file, args.output_format)

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
