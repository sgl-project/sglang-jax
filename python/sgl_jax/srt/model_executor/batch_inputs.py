"""Batch-owned host inputs, packed within each DP rank before upload."""

from dataclasses import dataclass
from functools import lru_cache
from math import prod

import numpy as np

FORWARD_INPUT_NAMES = (
    "input_ids",
    "seq_lens",
    "out_cache_loc",
    "positions",
    "req_pool_indices",
    "extend_prefix_lens",
    "extend_seq_lens",
    "lora_scalings",
    "lora_token_indices",
    "lora_ranks",
    "recurrent_indices",
    "recurrent_cow_src_indices",
    "recurrent_track_indices",
    "recurrent_track_mask",
)


@dataclass(frozen=True)
class _PackedField:
    buffer: np.ndarray
    start: int
    size: int
    shape: tuple[int, ...]

    def rank_view(self, rank):
        view = self.buffer[rank, self.start : self.start + self.size]
        if len(self.shape) == 1:
            return view
        return view.reshape(self.shape[0] // len(self.buffer), *self.shape[1:])

    def host_array(self):
        # Flattening across DP segments may copy. Only host consumers pay for
        # this representation; device upload uses the original packed buffer.
        array = np.ascontiguousarray(self.buffer[:, self.start : self.start + self.size])
        array = array.reshape(self.shape)
        array.flags.writeable = False
        return array


class BatchInputs:
    """An immutable packed snapshot with optional replacements for a new forward.

    Reading a host field materializes its flat request/token order on demand.
    Replacing a field returns a new container, so shallow-copied worker batches
    can safely specialize their inputs. Already submitted buffers are never reused.
    """

    def __init__(self, fields, groups=()):
        self._fields = fields
        self._groups = groups
        self._host_cache = {}

    @classmethod
    def from_arrays(cls, **arrays):
        return cls(arrays)

    def host(self, name):
        value = self._fields.get(name)
        if not isinstance(value, _PackedField):
            return value
        if name not in self._host_cache:
            self._host_cache[name] = value.host_array()
        return self._host_cache[name]

    def with_array(self, name, value):
        if name not in FORWARD_INPUT_NAMES:
            raise KeyError(name)
        return BatchInputs({**self._fields, name: value})

    def to_device(self, sharding):
        from sgl_jax.srt.utils.jax_utils import (
            _metadata_unpacker,
            canonicalize_sharding,
            device_array,
            packed_device_array,
        )

        if not self._groups:
            # Speculative/LoRA steps may replace fields with different shapes
            # or device arrays. Keep the existing immutable packing path for
            # those inputs rather than uploading a stale initial snapshot.
            return packed_device_array(tuple(self.host(n) for n in FORWARD_INPUT_NAMES), sharding)

        sharding = canonicalize_sharding(sharding)
        result = {}
        for names, buffer in self._groups:
            shapes = tuple(self._fields[name].shape for name in names)
            num_shards, _, buffer_sharding, unpack = _metadata_unpacker(shapes, sharding)
            if num_shards != len(buffer):
                raise ValueError("Batch input DP layout does not match the upload sharding")
            outputs = unpack(device_array(buffer, sharding=buffer_sharding))
            result.update(zip(names, outputs))
        return tuple(result.get(name) for name in FORWARD_INPUT_NAMES)


@lru_cache(maxsize=128)
def _buffer_layout(dp_size, fields):
    """Cache shape arithmetic only; every batch still owns fresh storage."""
    groups = {}
    for name, shape, dtype, fill in fields:
        if name not in FORWARD_INPUT_NAMES:
            raise KeyError(name)
        if dp_size < 1 or not shape or shape[0] % dp_size:
            raise ValueError(f"Invalid DP input shape for {name}: {shape}")
        groups.setdefault(np.dtype(dtype), []).append((name, shape, fill))
    layouts = []
    for dtype, entries in groups.items():
        offset = 0
        layout = []
        for name, shape, fill in entries:
            size = prod(shape) // dp_size
            layout.append((name, shape, fill, offset, size))
            offset += size
        layouts.append((dtype, offset, tuple(layout)))
    return tuple(layouts)


class BatchInputBuffer:
    """Reserve fields once, fill rank-local views, then publish one snapshot."""

    def __init__(self, dp_size, fields):
        # fields maps names to (global shape, dtype, padding value).
        layout = _buffer_layout(dp_size, tuple((name, *spec) for name, spec in fields.items()))
        self._fields = {}
        self._groups = []
        self._finished = False
        self._views = []
        for dtype, size, entries in layout:
            buffer = np.empty((dp_size, size), dtype=dtype)
            for name, shape, fill, offset, size in entries:
                buffer[:, offset : offset + size].fill(fill)
                self._fields[name] = _PackedField(buffer, offset, size, shape)
            self._groups.append((tuple(entry[0] for entry in entries), buffer))

    def view(self, name, rank):
        if self._finished:
            raise RuntimeError("Batch input buffer has already been published")
        view = self._fields[name].rank_view(rank)
        self._views.append(view)
        return view

    def finish(self):
        if self._finished:
            raise RuntimeError("Batch input buffer has already been published")
        self._finished = True
        for view in self._views:
            view.flags.writeable = False
        for _, buffer in self._groups:
            buffer.flags.writeable = False
        return BatchInputs(self._fields, tuple(self._groups))


def input_field(name):
    """Keep worker field access explicit while storage lives in BatchInputs."""

    def get(batch):
        return batch.inputs.host(name)

    def set_(batch, value):
        batch.inputs = batch.inputs.with_array(name, value)
        batch.layout = None

    return property(get, set_)
