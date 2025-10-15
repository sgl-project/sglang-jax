"""
JAXModelLoader Unit Tests

Usage:
    python -m unittest test_jax_model_loader.TestJAXModelLoader

    # Test with specific model path:
    MODEL_PATH=/path/to/jax/model python -m unittest test_jax_model_loader.TestJAXModelLoader.test_load_model_with_custom_path
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from flax import nnx
from transformers.configuration_utils import PretrainedConfig

from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_loader.loader import JAXModelLoader
from sgl_jax.test.test_utils import CustomTestCase, create_device_mesh


class MockJAXModel:
    """Mock JAX model class for testing"""

    def __init__(self, config: PretrainedConfig, rngs: nnx.Rngs):
        self.config = config
        self.weights_loaded = False
        self.pytree_data = None

    def load_pytree_weights(self, pytree):
        """Mock implementation of load_pytree_weights"""
        self.weights_loaded = True
        self.pytree_data = pytree
        print("\n=== Loading PyTree Weights ===")
        self._print_pytree_structure(self.pytree_data)
        print("==============================\n")
        return True

    def _print_pytree_structure(self, pytree, prefix="", max_depth=10, current_depth=0):
        """Recursively print PyTree structure with detailed information including size, shape, type, and statistics"""
        if current_depth > max_depth:
            print(f"{prefix}... (max depth reached)")
            return

        if isinstance(pytree, dict):
            print(f"{prefix}dict ({len(pytree)} keys):")
            for key, value in pytree.items():
                print(f"{prefix}  {key}:")
                self._print_pytree_structure(
                    value, prefix + "    ", max_depth, current_depth + 1
                )
        elif isinstance(pytree, (list, tuple)):
            type_name = "list" if isinstance(pytree, list) else "tuple"
            print(f"{prefix}{type_name} ({len(pytree)} items):")
            for i, item in enumerate(pytree):
                print(f"{prefix}  [{i}]:")
                self._print_pytree_structure(
                    item, prefix + "    ", max_depth, current_depth + 1
                )
        elif hasattr(pytree, "shape") and hasattr(pytree, "dtype"):
            shape = pytree.shape
            dtype = pytree.dtype
            size = pytree.size if hasattr(pytree, "size") else 1
            for dim in shape:
                size *= dim if not hasattr(pytree, "size") else 1

            try:
                if hasattr(pytree, "nbytes"):
                    memory_bytes = pytree.nbytes
                else:
                    # Estimate memory usage
                    dtype_size = (
                        pytree.dtype.itemsize
                        if hasattr(pytree.dtype, "itemsize")
                        else 4
                    )
                    memory_bytes = size * dtype_size

                if memory_bytes >= 1024**3:  # GB
                    memory_str = f"{memory_bytes / (1024**3):.2f} GB"
                elif memory_bytes >= 1024**2:  # MB
                    memory_str = f"{memory_bytes / (1024**2):.2f} MB"
                elif memory_bytes >= 1024:  # KB
                    memory_str = f"{memory_bytes / 1024:.2f} KB"
                else:
                    memory_str = f"{memory_bytes} bytes"
            except:
                memory_str = "unknown"

            stats_str = ""
            try:
                if hasattr(pytree, "min") and hasattr(pytree, "max") and size > 0:
                    min_val = float(pytree.min())
                    max_val = float(pytree.max())
                    if hasattr(pytree, "mean"):
                        mean_val = float(pytree.mean())
                        stats_str = f", range=[{min_val:.6f}, {max_val:.6f}], mean={mean_val:.6f}"
                    else:
                        stats_str = f", range=[{min_val:.6f}, {max_val:.6f}]"
            except:
                pass

            # Print tensor data preview
            data_preview = ""
            try:
                if len(shape) >= 2 and shape[0] > 0 and shape[1] > 0:
                    # For 2D+ tensors, show first 2 rows with head and tail 3 columns
                    if shape[1] <= 6:
                        # If total columns <= 6, show all
                        preview_data = pytree[:2, :]
                        data_preview = f"\n{prefix}  First 2 rows (all {shape[1]} cols):\n{prefix}    {preview_data}"
                    else:
                        # Show first 3 and last 3 columns
                        head_cols = pytree[:2, :3]
                        tail_cols = pytree[:2, -3:]
                        data_preview = f"\n{prefix}  First 2 rows (head 3 + tail 3 cols):\n{prefix}    Head: {head_cols}\n{prefix}    Tail: {tail_cols}"
                elif len(shape) == 1 and shape[0] > 0:
                    # For 1D tensors, show head and tail 3 elements
                    if shape[0] <= 6:
                        preview_data = pytree[:]
                        data_preview = f"\n{prefix}  All {shape[0]} elements:\n{prefix}    {preview_data}"
                    else:
                        head_elements = pytree[:3]
                        tail_elements = pytree[-3:]
                        data_preview = f"\n{prefix}  Head 3 + tail 3 elements:\n{prefix}    Head: {head_elements}\n{prefix}    Tail: {tail_elements}"
                elif len(shape) == 0:
                    # For scalar tensors
                    data_preview = f"\n{prefix}  Value: {pytree}"
            except Exception as e:
                data_preview = f"\n{prefix}  Data preview error: {str(e)}"

            print(
                f"{prefix}{type(pytree).__name__}: shape={shape}, dtype={dtype}, size={size:,}, memory={memory_str}{stats_str}{data_preview}"
            )
        elif hasattr(pytree, "__len__") and not isinstance(pytree, str):
            # Other sequence types
            print(f"{prefix}{type(pytree).__name__} (length={len(pytree)})")
        else:
            # Scalar or other types
            if isinstance(pytree, str) and len(pytree) > 50:
                print(
                    f"{prefix}{type(pytree).__name__} (length={len(pytree)}): {repr(pytree[:50])}..."
                )
            elif isinstance(pytree, (int, float, complex)):
                print(f"{prefix}{type(pytree).__name__}: {pytree}")
            else:
                print(f"{prefix}{type(pytree).__name__}: {repr(pytree)}")


class TestJAXModelLoader(CustomTestCase):
    """Test cases for JAXModelLoader"""

    def setUp(self):
        self.mesh = create_device_mesh(
            ici_parallelism=[-1, 1, 1], dcn_parallelism=[1, 1, 1]
        )

        self.test_model_path = os.environ.get("MODEL_PATH", "/tmp/test_jax_model")
        self.load_config = LoadConfig(load_format=LoadFormat.JAX)

        self.temp_dir = tempfile.mkdtemp()
        self.mock_model_path = os.path.join(self.temp_dir, "mock_jax_model")
        os.makedirs(self.mock_model_path, exist_ok=True)

        self.mock_msgpack_file = os.path.join(self.mock_model_path, "model.msgpack")
        with open(self.mock_msgpack_file, "wb") as f:
            f.write(b"mock_msgpack_data")

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_jax_loader_initialization(self):
        """Test JAXModelLoader initialization"""
        loader = JAXModelLoader(self.load_config)
        self.assertEqual(loader.load_config.load_format, LoadFormat.JAX)

    def test_prepare_jax_weights_local_path(self):
        """Test preparing JAX weights from local path"""
        loader = JAXModelLoader(self.load_config)

        hf_folder, hf_weights_files = loader._prepare_jax_weights(
            self.mock_model_path, None
        )

        self.assertEqual(hf_folder, self.mock_model_path)
        self.assertEqual(len(hf_weights_files), 1)
        self.assertTrue(hf_weights_files[0].endswith(".msgpack"))

    def test_prepare_jax_weights_no_msgpack_files(self):
        """Test preparing JAX weights when no .msgpack files exist"""
        empty_dir = os.path.join(self.temp_dir, "empty_model")
        os.makedirs(empty_dir, exist_ok=True)

        loader = JAXModelLoader(self.load_config)

        with self.assertRaises(RuntimeError) as context:
            loader._prepare_jax_weights(empty_dir, None)

        self.assertIn("Cannot find any JAX model weights", str(context.exception))

    def test_load_model_with_real_path(self):
        """Test loading model with real model path (integration test)"""
        if not os.path.exists(self.test_model_path):
            self.skipTest(f"Real model path {self.test_model_path} not found")

        msgpack_files = [
            f for f in os.listdir(self.test_model_path) if f.endswith(".msgpack")
        ]
        if not msgpack_files:
            self.skipTest(f"No .msgpack files found in {self.test_model_path}")

        model_config = ModelConfig(
            model_path=self.test_model_path, model_override_args="{}"
        )

        loader = JAXModelLoader(self.load_config)

        with patch(
            "sgl_jax.srt.model_loader.loader.get_model_architecture"
        ) as mock_arch:
            mock_arch.return_value = (MockJAXModel, None)
            model = loader.load_model(
                model_config=model_config,
                mesh=self.mesh,
            )

            self.assertIsInstance(model, MockJAXModel)
            self.assertTrue(model.weights_loaded)


if __name__ == "__main__":
    unittest.main()
