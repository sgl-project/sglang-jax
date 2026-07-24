import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.utils import jax_utils


class FakeDeviceArray:
    def __init__(
        self,
        data,
        fully_addressable,
        fully_replicated=False,
        local_data=None,
        name=None,
    ):
        self.data = np.asarray(data)
        self.is_fully_addressable = fully_addressable
        self.is_fully_replicated = fully_replicated
        self.local_data = self.data if local_data is None else np.asarray(local_data)
        self.name = name
        self.copy_count = 0

    @property
    def shape(self):
        return self.data.shape

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def copy_to_host_async(self):
        self.copy_count += 1

    def addressable_data(self, index):
        assert index == 0
        return FakeDeviceArray(
            self.local_data,
            True,
            True,
            name=self.name,
        )


class TestMultihostEagleHostTransfer(unittest.TestCase):
    def test_materialize_to_host_respects_array_sharding(self):
        calls = []

        def fake_allgather(value, *, tiled):
            calls.append((value, tiled))
            return FakeDeviceArray(value.data + 100, True)

        local = FakeDeviceArray([1, 2], True)
        replicated = FakeDeviceArray([9, 9], False, True, [3, 4])
        sharded = FakeDeviceArray([5, 6], False)

        def materialize(value):
            return jax_utils.materialize_to_host(value, jax_utils.prefetch_to_host(value))

        with mock.patch.object(jax_utils, "process_allgather", fake_allgather):
            np.testing.assert_array_equal(materialize(local), np.array([1, 2]))
            np.testing.assert_array_equal(
                materialize(replicated),
                np.array([3, 4]),
            )
            np.testing.assert_array_equal(
                materialize(sharded),
                np.array([105, 106]),
            )

        self.assertEqual(local.copy_count, 1)
        self.assertEqual(calls, [(sharded, True)])

    def test_eagle_ensure_host_gathers_cross_rank_fields(self):
        gathered = {
            "topk_p": np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
            "topk_index": np.array([[10], [20], [30]], dtype=np.int32),
            "hidden_states": np.array(
                [[100, 101], [200, 201], [300, 301]],
                dtype=np.float32,
            ),
            "verified_id": np.array([1, 2, 3], dtype=np.int32),
            "accept_length": np.array([4, 5, 6], dtype=np.int32),
        }
        calls = []

        def fake_allgather(value, *, tiled):
            calls.append((value.name, tiled))
            return FakeDeviceArray(gathered[value.name], True)

        fields = ("topk_p", "topk_index", "hidden_states", "verified_id", "accept_length")
        flat = EagleDraftInput(
            **{name: FakeDeviceArray([], False, name=name) for name in fields},
        )

        with mock.patch.object(jax_utils, "process_allgather", fake_allgather):
            flat._ensure_host()

        self.assertEqual([name for name, tiled in calls if tiled], list(fields))
        for name in fields:
            np.testing.assert_array_equal(getattr(flat, name), gathered[name])


if __name__ == "__main__":
    unittest.main()
