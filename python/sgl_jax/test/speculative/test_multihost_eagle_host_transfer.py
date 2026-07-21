import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.speculative import eagle_util
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput


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
    def test_host_spec_array_gathers_only_nonlocal_sharded_arrays(self):
        calls = []

        def fake_allgather(value, *, tiled):
            calls.append((value, tiled))
            return FakeDeviceArray(value.data + 100, True)

        local = FakeDeviceArray([1, 2], True)
        replicated = FakeDeviceArray([9, 9], False, True, [3, 4])
        sharded = FakeDeviceArray([5, 6], False)

        with mock.patch.object(eagle_util, "process_allgather", fake_allgather):
            np.testing.assert_array_equal(eagle_util._host_spec_array(local), np.array([1, 2]))
            np.testing.assert_array_equal(
                eagle_util._host_spec_array(replicated),
                np.array([3, 4]),
            )
            np.testing.assert_array_equal(
                eagle_util._host_spec_array(sharded),
                np.array([105, 106]),
            )

        self.assertEqual(local.copy_count, 1)
        self.assertEqual(calls, [(sharded, True)])

    def test_split_spec_info_ensure_host_gathers_cross_rank_flat_layout(self):
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
            allocate_lens=np.array([8, 9, 10], dtype=np.int32),
        )

        with mock.patch.object(eagle_util, "process_allgather", fake_allgather):
            parts = ScheduleBatch._split_spec_info_per_rank(flat, [2, 1])

        self.assertEqual([name for name, tiled in calls if tiled], list(fields))
        self.assertIsNotNone(parts[0])
        self.assertIsNotNone(parts[1])
        np.testing.assert_array_equal(parts[0].verified_id, np.array([1, 2], dtype=np.int32))
        np.testing.assert_array_equal(parts[1].verified_id, np.array([3], dtype=np.int32))
        np.testing.assert_array_equal(parts[0].topk_index, np.array([[10], [20]], dtype=np.int32))
        np.testing.assert_array_equal(
            parts[1].hidden_states, np.array([[300, 301]], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
