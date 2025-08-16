"""
JAXModelLoader Test Runner

This script provides a convenient way to run JAXModelLoader tests with
different configurations and model paths.

Usage:
    # Run all tests
    python run_jax_loader_test.py

    # Run with specific model path
    python run_jax_loader_test.py --model-path /path/to/jax/model

    # Run specific test
    python run_jax_loader_test.py --test test_load_model_with_custom_path

    # Run with verbose output
    python run_jax_loader_test.py --verbose

    # Run with JAX dependencies check
    python run_jax_loader_test.py --check-jax
"""

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

        print(f"✓ JAX version: {jax.__version__}")
        print(f"✓ JAX backend: {jax.default_backend()}")

        x = jnp.array([1, 2, 3])
        y = jnp.sum(x)
        print(f"✓ JAX basic test: sum([1,2,3]) = {y}")

        return True
    except ImportError as e:
        print(f"✗ JAX not available: {e}")
        print("  Install JAX with: pip install jax jaxlib")
        return False
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        return False


def check_sglang_dependencies():
    """Check if SGLang dependencies are available"""
    try:
        from sgl_jax.srt.configs.load_config import LoadFormat
        from sgl_jax.srt.model_loader.loader import JAXModelLoader

        print("✓ SGLang JAXModelLoader available")
        return True
    except ImportError as e:
        print(f"✗ SGLang dependencies not available: {e}")
        return False


def run_tests(test_name=None, model_path=None, verbose=False):
    """Run the JAXModelLoader tests"""
    env = os.environ.copy()
    if model_path:
        env["MODEL_PATH"] = model_path
        print(f"Using model path: {model_path}")

    if test_name:
        test_target = f"test_jax_model_loader.TestJAXModelLoader.{test_name}"
    else:
        test_target = "test_jax_model_loader.TestJAXModelLoader"

    cmd = [sys.executable, "-m", "unittest"]
    if verbose:
        cmd.append("-v")
    cmd.append(test_target)

    print(f"Running command: {' '.join(cmd)}")
    print("=" * 50)

    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def create_sample_jax_model(output_dir):
    """Create a sample JAX model directory for testing"""
    import json
    import tempfile

    model_dir = Path(output_dir) / "sample_jax_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "architectures": ["LlamaForCausalLM"],
    }

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    msgpack_file = model_dir / "model.msgpack"
    with open(msgpack_file, "wb") as f:
        f.write(b"mock_jax_model_data")

    print(f"Created sample JAX model at: {model_dir}")
    return str(model_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run JAXModelLoader tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-path", help="Path to JAX model directory (must contain .msgpack files)"
    )

    parser.add_argument(
        "--test",
        help="Specific test method to run (e.g., test_load_model_with_custom_path)",
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
        help="Create a sample JAX model directory at the specified path",
    )

    args = parser.parse_args()

    if args.check_jax or args.check_deps:
        print("Checking JAX dependencies...")
        jax_ok = check_jax_dependencies()

        if args.check_deps:
            print("\nChecking SGLang dependencies...")
            sglang_ok = check_sglang_dependencies()

            if jax_ok and sglang_ok:
                print("\n✓ All dependencies are available")
                return 0
            else:
                print("\n✗ Some dependencies are missing")
                return 1

        return 0 if jax_ok else 1

    if args.create_sample:
        try:
            model_path = create_sample_jax_model(args.create_sample)
            print(f"\nYou can now run tests with:")
            print(f"python {__file__} --model-path {model_path}")
            return 0
        except Exception as e:
            print(f"Error creating sample model: {e}")
            return 1

    print("Checking dependencies...")
    if not check_sglang_dependencies():
        print("\nCannot run tests without SGLang dependencies")
        return 1

    jax_available = check_jax_dependencies()
    if not jax_available:
        print("\nWarning: JAX not available, some tests will use mocks")

    print("\nRunning JAXModelLoader tests...")

    success = run_tests(
        test_name=args.test, model_path=args.model_path, verbose=args.verbose
    )

    if success:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
