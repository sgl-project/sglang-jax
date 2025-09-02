import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path


def check_jax_dependencies():
    """Check if JAX dependencies are available"""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx

        print(f"✓ JAX version: {jax.__version__}")
        print(f"✓ JAX backend: {jax.default_backend()}")
        print(f"✓ Flax NNX available")

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
        from sgl_jax.srt.configs.load_config import LoadFormat
        from sgl_jax.srt.model_loader.loader import JAXModelLoader
        from sgl_jax.srt.models.qwen import QWenLMHeadJaxModel

        print("✓ SGLang JAXModelLoader available")
        print("✓ QWenLMHeadJaxModel available")
        return True
    except ImportError as e:
        print(f"✗ SGLang dependencies not available: {e}")
        print("  Make sure SGLang is properly installed")
        return False


def check_transformers_dependencies():
    """Check if Transformers dependencies are available"""
    try:
        import transformers
        from transformers import PretrainedConfig

        print(f"✓ Transformers version: {transformers.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Transformers not available: {e}")
        print("  Install with: pip install transformers")
        return False


def run_tests(
    test_name=None, model_path=None, verbose=False, enable_precision_tracer=False
):
    """Run the QWen JAXModelLoader tests"""
    env = os.environ.copy()
    if model_path:
        env["MODEL_PATH"] = model_path
        print(f"Using model path: {model_path}")
    if enable_precision_tracer:
        print("✓ Enable precision tracer", enable_precision_tracer)
        env["ENABLE_PRECISION_TRACER"] = "true"
    if test_name:
        test_target = f"test_qwen_load_weights.TestQWenLoadWeights.{test_name}"
    else:
        test_target = "test_qwen_load_weights.TestQWenLoadWeights"

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


def create_sample_qwen_model(output_dir):
    """Create a sample QWen JAX model directory for testing"""
    import json

    import msgpack
    import numpy as np

    model_dir = Path(output_dir) / "sample_qwen_jax_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create config.json
    config = {
        "model_type": "qwen",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "vocab_size": 1000,
        "intermediate_size": 256,
        "max_position_embeddings": 512,
        "layer_norm_epsilon": 1e-6,
        "rope_theta": 10000,
        "architectures": ["QWenLMHeadJaxModel"],
    }

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create a mock msgpack file with QWen-like structure
    try:
        mock_weights = {
            "transformer": {
                "embed_tokens": {
                    "kernel": np.random.randn(
                        config["vocab_size"], config["hidden_size"]
                    ).astype(np.float32)
                },
                "h": {},
                "ln_f": {"scale": np.ones(config["hidden_size"], dtype=np.float32)},
            },
            "lm_head": {
                "kernel": np.random.randn(
                    config["hidden_size"], config["vocab_size"]
                ).astype(np.float32)
            },
            "logits_processor": {
                "kernel": np.random.randn(
                    config["hidden_size"], config["vocab_size"]
                ).astype(np.float32)
            },
        }

        # Add layer weights
        for i in range(config["num_hidden_layers"]):
            layer_weights = {
                "ln_1": {"scale": np.ones(config["hidden_size"], dtype=np.float32)},
                "attn": {
                    "c_attn": {
                        "kernel": np.random.randn(
                            config["hidden_size"], config["hidden_size"] * 3
                        ).astype(np.float32)
                    },
                    "c_proj": {
                        "kernel": np.random.randn(
                            config["hidden_size"], config["hidden_size"]
                        ).astype(np.float32)
                    },
                },
                "ln_2": {"scale": np.ones(config["hidden_size"], dtype=np.float32)},
                "mlp": {
                    "w1": {
                        "weight": np.random.randn(
                            config["hidden_size"], config["intermediate_size"] // 2
                        ).astype(np.float32)
                    },
                    "w2": {
                        "weight": np.random.randn(
                            config["hidden_size"], config["intermediate_size"] // 2
                        ).astype(np.float32)
                    },
                    "c_proj": {
                        "weight": np.random.randn(
                            config["intermediate_size"] // 2, config["hidden_size"]
                        ).astype(np.float32)
                    },
                },
            }
            mock_weights["transformer"]["h"][str(i)] = layer_weights

        msgpack_file = model_dir / "model.msgpack"
        with open(msgpack_file, "wb") as f:
            msgpack.pack(mock_weights, f)

        print(f"Created sample QWen JAX model at: {model_dir}")
        print(f"  - config.json: Model configuration")
        print(f"  - model.msgpack: Mock weights ({msgpack_file.stat().st_size} bytes)")

    except ImportError:
        # Fallback: create a simple binary file
        msgpack_file = model_dir / "model.msgpack"
        with open(msgpack_file, "wb") as f:
            f.write(b"mock_qwen_jax_model_data")
        print(f"Created sample QWen JAX model at: {model_dir} (simple mock)")

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
        description="Run QWen JAXModelLoader tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-path",
        help="Path to QWen JAX model directory (must contain .msgpack files)",
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
        help="Create a sample QWen JAX model directory at the specified path",
    )

    parser.add_argument(
        "--list-tests", action="store_true", help="List all available test methods"
    )
    parser.add_argument(
        "--enable-precision-tracer",
        action="store_true",
        help="Enable precision tracer for debugging purposes.",
    )

    args = parser.parse_args()

    os.environ["JAX_PLATFORMS"] = "cpu"

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
            model_path = create_sample_qwen_model(args.create_sample)
            print(f"\nYou can now run tests with:")
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

    print("\nRunning QWen JAXModelLoader tests...")
    success = run_tests(
        test_name=args.test,
        model_path=args.model_path,
        verbose=args.verbose,
        enable_precision_tracer=args.enable_precision_tracer,
    )

    if success:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
