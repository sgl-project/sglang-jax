import tempfile
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from sgl_jax.srt.utils.weight_utils import SequentialSafetensorManager, WeightLoader


def _make_single_device_mesh() -> jax.sharding.Mesh:
    devices = np.array(jax.devices()[:1], dtype=object).reshape(1, 1)
    return jax.sharding.Mesh(devices, axis_names=("data", "tensor"))


def _build_loader(mesh: jax.sharding.Mesh) -> WeightLoader:
    model_config = SimpleNamespace(
        _dummy_mode=False,
        ep_size=1,
        quantization_config=None,
    )
    return WeightLoader(
        model=None,
        model_config=model_config,
        mesh=mesh,
        dtype=jnp.bfloat16,
    )


def _write_expert_safetensors(
    tmp_dir: str, expert_tensors: dict[str, np.ndarray]
) -> dict[str, list[dict]]:
    weight_info = {}
    for i, (hf_key, tensor) in enumerate(expert_tensors.items()):
        st_path = f"{tmp_dir}/expert_{i}.safetensors"
        save_file({hf_key: tensor}, st_path)
        with safe_open(st_path, framework="np", device="cpu") as f:
            sl = f.get_slice(hf_key)
            weight_info[hf_key] = [
                {
                    "file": st_path,
                    "shape": tuple(sl.get_shape()),
                    "dtype": sl.get_dtype(),
                }
            ]
    return weight_info


def test_create_stacked_moe_lazy_tensor_weight_without_physical_to_logical_map():
    mesh = _make_single_device_mesh()
    loader = _build_loader(mesh)

    expert_tensors = {
        "exp0": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "exp1": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "exp2": np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
    }
    expected_hf_keys = list(expert_tensors.keys())

    with tempfile.TemporaryDirectory() as tmp_dir:
        weight_info = _write_expert_safetensors(tmp_dir, expert_tensors)
        with SequentialSafetensorManager() as file_manager:
            stacked = loader._create_stacked_moe_lazy_tensor(
                expected_hf_keys=expected_hf_keys,
                weight_info=weight_info,
                file_manager=file_manager,
                do_transpose=True,  # Weight path in fused MoE
                target_sharding=None,
                physical_to_logical_map=None,
            )
            stacked_np = np.array(stacked)

    expected = np.stack([expert_tensors[k].T for k in expected_hf_keys], axis=0)
    np.testing.assert_array_equal(stacked_np, expected)
    assert stacked_np.shape == (3, 2, 2)


def test_create_stacked_moe_lazy_tensor_weight_with_physical_to_logical_map():
    mesh = _make_single_device_mesh()
    loader = _build_loader(mesh)

    expert_tensors = {
        "exp0": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "exp1": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "exp2": np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
    }
    expected_hf_keys = list(expert_tensors.keys())
    physical_to_logical_map = np.array([2, 0, 2, 1, 0], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp_dir:
        weight_info = _write_expert_safetensors(tmp_dir, expert_tensors)
        with SequentialSafetensorManager() as file_manager:
            stacked = loader._create_stacked_moe_lazy_tensor(
                expected_hf_keys=expected_hf_keys,
                weight_info=weight_info,
                file_manager=file_manager,
                do_transpose=True,
                target_sharding=None,
                physical_to_logical_map=physical_to_logical_map,
            )
            stacked_np = np.array(stacked)

    logical = [expert_tensors[k].T for k in expected_hf_keys]
    expected = np.stack([logical[idx] for idx in physical_to_logical_map], axis=0)
    np.testing.assert_array_equal(stacked_np, expected)
    assert stacked_np.shape == (5, 2, 2)


def test_create_stacked_moe_lazy_tensor_scale_without_physical_to_logical_map():
    mesh = _make_single_device_mesh()
    loader = _build_loader(mesh)

    expert_scales = {
        "exp0_scale": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "exp1_scale": np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float32),
        "exp2_scale": np.array([2.1, 2.2, 2.3, 2.4], dtype=np.float32),
    }
    expected_hf_keys = list(expert_scales.keys())

    with tempfile.TemporaryDirectory() as tmp_dir:
        weight_info = _write_expert_safetensors(tmp_dir, expert_scales)
        with SequentialSafetensorManager() as file_manager:
            stacked = loader._create_stacked_moe_lazy_tensor(
                expected_hf_keys=expected_hf_keys,
                weight_info=weight_info,
                file_manager=file_manager,
                do_transpose=False,  # Scale path
                target_sharding=None,
                physical_to_logical_map=None,
            )
            stacked_np = np.array(stacked)

    expected = np.stack([expert_scales[k] for k in expected_hf_keys], axis=0)
    np.testing.assert_array_equal(stacked_np, expected)
    assert stacked_np.shape == (3, 4)

    # Match static fused MoE post-processing: reshape + repeat
    num_blocks = 3
    post = np.repeat(stacked_np.reshape(3, 1, 1, 4), repeats=num_blocks, axis=1)
    assert post.shape == (3, 3, 1, 4)
    np.testing.assert_array_equal(post[:, 0, 0, :], expected)


def test_create_stacked_moe_lazy_tensor_scale_with_physical_to_logical_map():
    mesh = _make_single_device_mesh()
    loader = _build_loader(mesh)

    expert_scales = {
        "exp0_scale": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "exp1_scale": np.array([1.1, 1.2, 1.3, 1.4], dtype=np.float32),
        "exp2_scale": np.array([2.1, 2.2, 2.3, 2.4], dtype=np.float32),
    }
    expected_hf_keys = list(expert_scales.keys())
    physical_to_logical_map = np.array([1, 2, 1, 0, 2], dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp_dir:
        weight_info = _write_expert_safetensors(tmp_dir, expert_scales)
        with SequentialSafetensorManager() as file_manager:
            stacked = loader._create_stacked_moe_lazy_tensor(
                expected_hf_keys=expected_hf_keys,
                weight_info=weight_info,
                file_manager=file_manager,
                do_transpose=False,
                target_sharding=None,
                physical_to_logical_map=physical_to_logical_map,
            )
            stacked_np = np.array(stacked)

    logical = [expert_scales[k] for k in expected_hf_keys]
    expected = np.stack([logical[idx] for idx in physical_to_logical_map], axis=0)
    np.testing.assert_array_equal(stacked_np, expected)
    assert stacked_np.shape == (5, 4)

    # Match static fused MoE post-processing: reshape + repeat
    num_blocks = 2
    post = np.repeat(stacked_np.reshape(5, 1, 1, 4), repeats=num_blocks, axis=1)
    assert post.shape == (5, 2, 1, 4)
    np.testing.assert_array_equal(post[:, 0, 0, :], expected)
