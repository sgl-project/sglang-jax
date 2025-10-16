"""
Qwen3 MoE JAXModelLoader Test Runner

This script provides a convenient way to run Qwen3 MoE JAXModelLoader tests with
different configurations and model paths.

Usage:
    # Run all tests
    python run_qwen3_moe_test.py

    # Run with specific model path
    python run_qwen3_moe_test.py --model-path /path/to/jax/qwen3_moe/model

    # Run specific test
    python run_qwen3_moe_test.py --test test_load_model_with_jax_loader

    # Run with verbose output
    python run_qwen3_moe_test.py --verbose

    # Run with JAX dependencies check
    python run_qwen3_moe_test.py --check-jax

    # Create sample MoE model for testing
    python run_qwen3_moe_test.py --create-sample /tmp/sample_qwen3_moe
"""

import argparse
import os
import subprocess
import sys
import importlib
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def check_jax_dependencies():
    """Check if JAX dependencies are available"""
    try:
        import jax
        import jax.numpy as jnp
        importlib.util.find_spec("flax.nnx")

        print(f"✓ JAX version: {jax.__version__}")
        print(f"✓ JAX backend: {jax.default_backend()}")
        print(f"✓ Available devices: {len(jax.devices())} devices")
        print(f"✓ Device types: {[d.platform for d in jax.devices()]}")
        print("✓ Flax NNX available")

        # Test basic JAX operations
        x = jnp.array([1, 2, 3])
        y = jnp.sum(x)
        print(f"✓ JAX basic test: sum([1,2,3]) = {y}")

        return True
    except ImportError as e:
        print(f"✗ JAX/Flax not available: {e}")
        print("  Install with: pip install jax jaxlib flax")
        return False
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        return False


def check_sglang_dependencies():
    """Check if SGLang dependencies are available"""
    try:
        importlib.util.find_spec("sgl_jax.srt.configs.load_config.LoadFormat")
        importlib.util.find_spec("sgl_jax.srt.model_loader.loader.JAXModelLoader")
        importlib.util.find_spec("sgl_jax.srt.models.qwen3_moe.Qwen3MoeForCausalLMJaxModel")

        print("✓ SGLang JAXModelLoader available")
        print("✓ Qwen3MoeForCausalLMJaxModel available")
        return True
    except ImportError as e:
        print(f"✗ SGLang dependencies not available: {e}")
        print("  Make sure SGLang is properly installed")
        return False


def check_transformers_dependencies():
    """Check if Transformers dependencies are available"""
    try:
        import transformers

        print(f"✓ Transformers version: {transformers.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Transformers not available: {e}")
        print("  Install with: pip install transformers")
        return False


def run_tests(test_name=None, model_path=None, verbose=False):
    """Run the Qwen3 MoE JAXModelLoader tests"""
    env = os.environ.copy()
    if model_path:
        env["MODEL_PATH"] = model_path
        print(f"Using model path: {model_path}")

    if test_name:
        test_target = f"test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights.{test_name}"
    else:
        test_target = "test_qwen3_moe_load_weights.TestQwen3MoeLoadWeights"

    cmd = [sys.executable, "-m", "unittest"]
    if verbose:
        cmd.append("-v")
    cmd.append(test_target)

    print(f"Running command: {' '.join(cmd)}")
    print("=" * 50)

    try:
        del env["JAX_PLATFORMS"]
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def create_sample_qwen3_moe_model(output_dir):
    """Create a sample Qwen3 MoE JAX model directory for testing"""
    import json

    import msgpack
    import numpy as np

    model_dir = Path(output_dir) / "sample_qwen3_moe_jax_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create config.json for Qwen3 MoE model
    config = {
        "model_type": "qwen3",
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "vocab_size": 1000,
        "intermediate_size": 512,
        "max_position_embeddings": 512,
        "layer_norm_epsilon": 1e-6,
        "rope_theta": 10000,
        "architectures": ["Qwen3MoeForCausalLMJaxModel"],
        # MoE specific config
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "mlp_only_layers": [],
        "shared_expert_intermediate_size": 256,
        "moe_layer_freq": 1,
        "norm_topk_prob": False,
        "output_router_logits": False,
        "router_aux_loss_coef": 0.001,
    }

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create a mock msgpack file with Qwen3 MoE structure
    try:
        mock_weights = {
            "model": {
                "embed_tokens": {
                    "kernel": np.random.randn(
                        config["vocab_size"], config["hidden_size"]
                    ).astype(np.float32)
                },
                "layers": {},
                "norm": {"scale": np.ones(config["hidden_size"], dtype=np.float32)},
            },
            "lm_head": {
                "kernel": np.random.randn(
                    config["hidden_size"], config["vocab_size"]
                ).astype(np.float32)
            },
        }

        # Add layer weights with MoE structure
        for i in range(config["num_hidden_layers"]):
            layer_weights = {
                "input_layernorm": {
                    "scale": np.ones(config["hidden_size"], dtype=np.float32)
                },
                "self_attn": {
                    "q_proj": {
                        "kernel": np.random.randn(
                            config["hidden_size"], config["hidden_size"]
                        ).astype(np.float32)
                    },
                    "k_proj": {
                        "kernel": np.random.randn(
                            config["hidden_size"], config["hidden_size"]
                        ).astype(np.float32)
                    },
                    "v_proj": {
                        "kernel": np.random.randn(
                            config["hidden_size"], config["hidden_size"]
                        ).astype(np.float32)
                    },
                    "o_proj": {
                        "kernel": np.random.randn(
                            config["hidden_size"], config["hidden_size"]
                        ).astype(np.float32)
                    },
                },
                "post_attention_layernorm": {
                    "scale": np.ones(config["hidden_size"], dtype=np.float32)
                },
            }

            # Add MoE structure for this layer
            if i not in config["mlp_only_layers"]:
                # MoE layer
                moe_weights = {
                    "gate": {
                        "kernel": np.random.randn(
                            config["num_experts"], config["hidden_size"]
                        ).astype(np.float32)
                    },
                    "experts": {},
                }

                # Add expert weights
                for expert_idx in range(config["num_experts"]):
                    expert_weights = {
                        "gate_proj": {
                            "kernel": np.random.randn(
                                config["moe_intermediate_size"], config["hidden_size"]
                            ).astype(np.float32)
                        },
                        "up_proj": {
                            "kernel": np.random.randn(
                                config["moe_intermediate_size"], config["hidden_size"]
                            ).astype(np.float32)
                        },
                        "down_proj": {
                            "kernel": np.random.randn(
                                config["hidden_size"], config["moe_intermediate_size"]
                            ).astype(np.float32)
                        },
                    }
                    moe_weights["experts"][str(expert_idx)] = expert_weights

                layer_weights["mlp"] = moe_weights
            else:
                # Regular MLP layer
                layer_weights["mlp"] = {
                    "gate_proj": {
                        "kernel": np.random.randn(
                            config["intermediate_size"], config["hidden_size"]
                        ).astype(np.float32)
                    },
                    "up_proj": {
                        "kernel": np.random.randn(
                            config["intermediate_size"], config["hidden_size"]
                        ).astype(np.float32)
                    },
                    "down_proj": {
                        "kernel": np.random.randn(
                            config["hidden_size"], config["intermediate_size"]
                        ).astype(np.float32)
                    },
                }

            mock_weights["model"]["layers"][str(i)] = layer_weights

        msgpack_file = model_dir / "model.msgpack"
        with open(msgpack_file, "wb") as f:
            msgpack.pack(mock_weights, f)

        print(f"Created sample Qwen3 MoE JAX model at: {model_dir}")
        print("  - config.json: Model configuration with MoE settings")
        print(f"  - model.msgpack: Mock weights ({msgpack_file.stat().st_size} bytes)")
        print(
            f"  - MoE config: {config['num_experts']} experts, {config['num_experts_per_tok']} experts per token"
        )

    except ImportError:
        # Fallback: create a simple binary file
        msgpack_file = model_dir / "model.msgpack"
        with open(msgpack_file, "wb") as f:
            f.write(b"mock_qwen3_moe_jax_model_data")
        print(f"Created sample Qwen3 MoE JAX model at: {model_dir} (simple mock)")

    return str(model_dir)


def list_available_tests():
    """List all available test methods"""
    tests = [
        "test_jax_loader_initialization",
        "test_jax_loader_invalid_format",
        "test_load_model_with_jax_loader",
        "test_prepare_jax_weights_no_msgpack_files",
    ]

    print("Available test methods:")
    for test in tests:
        print(f"  - {test}")
    return tests


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3 MoE JAXModelLoader tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-path",
        help="Path to Qwen3 MoE JAX model directory (must contain .msgpack files)",
    )

    parser.add_argument(
        "--test",
        help="Specific test method to run (use --list-tests to see available tests)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose test output"
    )

    parser.add_argument(
        "--check-jax", action="store_true", help="Check JAX dependencies and exit"
    )

    parser.add_argument(
        "--check-deps", action="store_true", help="Check all dependencies and exit"
    )

    parser.add_argument(
        "--create-sample",
        help="Create a sample Qwen3 MoE JAX model directory at the specified path",
    )

    parser.add_argument(
        "--list-tests", action="store_true", help="List all available test methods"
    )

    args = parser.parse_args()

    if args.list_tests:
        list_available_tests()
        return 0

    if args.check_jax or args.check_deps:
        print("Checking JAX dependencies...")
        jax_ok = check_jax_dependencies()

        if args.check_deps:
            print("\nChecking Transformers dependencies...")
            transformers_ok = check_transformers_dependencies()

            print("\nChecking SGLang dependencies...")
            sglang_ok = check_sglang_dependencies()

            if jax_ok and transformers_ok and sglang_ok:
                print("\n✓ All dependencies are available")
                return 0
            else:
                print("\n✗ Some dependencies are missing")
                return 1

        return 0 if jax_ok else 1

    if args.create_sample:
        try:
            model_path = create_sample_qwen3_moe_model(args.create_sample)
            print("\nYou can now run tests with:")
            print(f"python {__file__} --model-path {model_path}")
            return 0
        except Exception as e:
            print(f"Error creating sample model: {e}")
            return 1

    print("Checking dependencies...")

    # Check transformers first
    if not check_transformers_dependencies():
        print("\nCannot run tests without Transformers")
        return 1

    # Check SGLang
    if not check_sglang_dependencies():
        print("\nCannot run tests without SGLang dependencies")
        return 1

    # Check JAX (optional but recommended)
    jax_available = check_jax_dependencies()
    if not jax_available:
        print("\nWarning: JAX not available, some tests may fail")

    print("\nRunning Qwen3 MoE JAXModelLoader tests...")

    success = run_tests(
        test_name=args.test, model_path=args.model_path, verbose=args.verbose
    )

    if success:
        print("\n✓ All Qwen3 MoE tests passed!")
        return 0
    else:
        print("\n✗ Some Qwen3 MoE tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
