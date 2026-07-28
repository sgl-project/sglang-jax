import jax
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils.weight_utils import WeightLoader


class _ArraySlice:
    def __init__(self, value: np.ndarray):
        self._value = value

    def __getitem__(self, index):
        return self._value[index]


class _ArrayHandle:
    def __init__(self, values: dict[str, np.ndarray]):
        self._values = values

    def get_slice(self, key: str):
        return _ArraySlice(self._values[key])


class _ArrayFileManager:
    def __init__(self, values: dict[str, np.ndarray]):
        self._handle = _ArrayHandle(values)
        self.handles = {"model.safetensors": self._handle}

    def get_handle(self, _filename: str):
        return self._handle


def test_deferred_moe_transpose_preserves_target_sharding():
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1),
        axis_names=("expert", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    target_sharding = NamedSharding(mesh, P("expert", None, "tensor"))
    hf_weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    key = "model.layers.0.mlp.experts.0.gate_proj.weight"

    loader = object.__new__(WeightLoader)
    loader.mesh = mesh
    result = loader._create_stacked_moe_lazy_tensor(
        expected_hf_keys=[key],
        weight_info={
            key: [
                {
                    "file": "model.safetensors",
                    "shape": hf_weight.shape,
                    "dtype": "F32",
                }
            ]
        },
        file_manager=_ArrayFileManager({key: hf_weight}),
        do_transpose=True,
        target_sharding=target_sharding,
    )

    assert result.shape == (1, 3, 2)
    np.testing.assert_array_equal(np.asarray(result[0]), hf_weight.T)
    assert result.sharding == target_sharding
