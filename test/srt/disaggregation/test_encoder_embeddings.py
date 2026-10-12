"""CPU regressions for EPD lane layouts, reconstruction, and receive ownership."""

import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
    ReceivedEmbeddings,
    build_received_embedding_batch,
    embed_received_inputs,
    release_received_embeddings,
)
from sgl_jax.srt.disaggregation.encoder.raiden_receiver import (
    RaidenReceiverBackend,
    RaidenReceiveSession,
)
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    pack_vision_inputs,
    plan_encoder_lanes,
)
from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sgl_jax.srt.multimodal.processors.encoder import DisaggregatedInputMixin


def _mesh():
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


@pytest.mark.parametrize("num_lanes", [1, 2, 4])
@pytest.mark.parametrize("lengths", [(4,), (4, 12, 8), (16, 4, 4, 8, 12)])
def test_transfer_indices_restore_item_order_from_packed_encoder_output(num_lanes, lengths):
    items = [
        MultimodalDataItem(
            Modality.IMAGE,
            feature=np.arange(length, dtype=np.float32).reshape(-1, 1) + index * 100,
            model_specific_data={"image_grid_thw": [[1, 2, length // 2]]},
        )
        for index, length in enumerate(lengths)
    ]
    lanes, output_indices = plan_encoder_lanes(lengths, num_lanes, merge_unit=4)
    packed, capacity, _ = pack_vision_inputs(
        [[items[index] for index in lane] for lane in lanes],
        merge_unit=4,
        input_sharding=NamedSharding(_mesh(), PartitionSpec()),
    )
    # An independent encoder oracle reduces each group of four input patches.
    encoded = np.asarray(packed).reshape(num_lanes, capacity // 4, 4).mean(axis=-1).ravel()
    expected = np.concatenate([item.feature.reshape(-1, 4).mean(axis=-1) for item in items])
    logical_rows = output_indices[output_indices >= 0]
    assert len(output_indices) == len(encoded)
    assert len(np.unique(logical_rows)) == len(expected)
    np.testing.assert_array_equal(encoded[logical_rows], expected)


def test_out_of_order_parts_reconstruct_media_rows_and_cache_identity():
    buffer = jnp.zeros((16, 2))
    session = Mock()
    parts = MultiModalEmbeddingData(3)
    # Part order and request media order differ; two image parts share a session.
    specs = [
        (Modality.IMAGE, [7, 2], [1, 2, 4], 101),
        (Modality.VIDEO, [9, 4], [2, 2, 2], 202),
        (Modality.IMAGE, [5], [1, 2, 2], 303),
    ]
    for index in [2, 0, 1]:
        modality, rows, grid, media_hash = specs[index]
        parts.add(
            EmbeddingData(
                req_id="request",
                num_parts=3,
                part_idx=index,
                modality=modality,
                grid_dim=[grid],
                item_hashes=[media_hash],
                second_per_grid_ts=[0.5] if modality == Modality.VIDEO else None,
            ),
            ReceivedEmbeddings(buffer, np.asarray(rows, np.int32), 2, [session]),
        )
        assert parts.ready == (index == 1)

    embeddings = parts.get_embedding()
    metadata = parts.get_mm_extra_meta()
    processor = NS(
        hf_config=NS(image_token_id=100, video_token_id=200),
        spatial_merge_size=2,
        _to_grid_list=BaseMultimodalProcessor._to_grid_list,
        _to_numpy=BaseMultimodalProcessor._to_numpy,
    )
    processor.build_input_ids = lambda *args, **kwargs: DisaggregatedInputMixin.build_input_ids(
        processor, *args, **kwargs
    )
    inputs = DisaggregatedInputMixin.get_mm_data(
        processor, [1, 100, 2, 200, 3, 100, 4], embeddings, **metadata
    )

    assert inputs.input_ids == [1, 100, 100, 2, 200, 200, 3, 100, 4]
    assert [item.hash for item in inputs.mm_items] == [101, 202, 303]
    assert [item.placeholder_ranges for item in inputs.mm_items] == [[(1, 3)], [(4, 6)], [(7, 8)]]
    for item, (_, rows, _, _) in zip(inputs.mm_items, specs, strict=True):
        source = item.precomputed_embeddings
        assert source.buffer is buffer
        assert source.sessions == [session]
        np.testing.assert_array_equal(source.row_indices, rows)
        start, end = item.placeholder_ranges[0]
        assert inputs.radix_input_ids[start:end] == [item.pad_value] * (end - start)
    assert [inputs.radix_input_ids[index] for index in [0, 3, 6, 8]] == [1, 2, 3, 4]
    assert metadata["second_per_grid_ts"] == [0.5]


class _LanguageModel(InModelMultimodalContract):
    deepstack_visual_layers = 1

    def get_input_embeddings(self):
        return lambda ids: jnp.full((len(ids), 1), -1.0)


@pytest.mark.parametrize(
    "prefix,length,expected",
    [
        (0, 2, [-1, -1]),
        (2, 2, [10, 11]),
        (3, 4, [11, -1, -1, 12]),
        (6, 2, [12, 13]),
    ],
)
def test_chunked_prefill_gathers_paged_rows_and_tracks_device_reads(prefix, length, expected):
    rows = np.asarray([5, 1, 6, 2], np.int32)
    physical = np.full((2, 4, 2, 8, 128), np.nan, np.float32)
    physical.reshape(8, -1)[rows, :2] = [[10, 110], [11, 111], [12, 112], [13, 113]]
    session = Mock()
    source = ReceivedEmbeddings(jnp.asarray(physical), rows, 2, [session])
    item = MultimodalDataItem(
        Modality.IMAGE,
        placeholder_ranges=[(2, 4), (6, 8)],
        precomputed_embeddings=source,
    )
    info = NS(
        reqs=[NS(mm_inputs=MultimodalInputs(mm_items=[item]))],
        prefix_lens=[prefix],
        extend_lens=[length],
    )
    batch = build_received_embedding_batch([info], 1, length)
    result, deepstack, enabled = embed_received_inputs(
        batch, jnp.zeros(length, jnp.int32), _LanguageModel()
    )
    np.testing.assert_array_equal(result[:, 0], expected)
    np.testing.assert_array_equal(
        deepstack[0, :, 0], [value + 100 if value != -1 else 0 for value in expected]
    )
    assert enabled == bool(batch.items)
    assert session.record_read.call_count == int(bool(batch.items))
    # The scheduled batch retains its source even when the request is released.
    release_received_embeddings(info.reqs[0].mm_inputs)
    assert item.precomputed_embeddings is None
    assert all(received is source for _, received in batch.items)


def test_request_release_waits_for_device_reads_and_reclaims_shared_pages_once():
    receiver = object.__new__(RaidenReceiverBackend)
    receiver._condition = threading.Condition()
    receiver.pool = NS(release=Mock())
    session = RaidenReceiveSession(receiver, "request:0", (2, 3), 4, received=True, delivered=True)
    receiver._receives = {session.transfer_id: session}
    reader = Mock()
    reader.is_ready.return_value = False
    session.record_read(reader)
    source = ReceivedEmbeddings(jnp.zeros((8, 2)), np.arange(4, dtype=np.int32), 2, [session])
    inputs = MultimodalInputs(
        mm_items=[
            MultimodalDataItem(Modality.IMAGE, precomputed_embeddings=source.slice_rows(0, 2)),
            MultimodalDataItem(Modality.IMAGE, precomputed_embeddings=source.slice_rows(2, 4)),
        ]
    )
    release_received_embeddings(inputs)
    receiver.pool.release.assert_not_called()
    assert session.transfer_id in receiver._receives
    with pytest.raises(RuntimeError, match="released encoder pages"):
        session.record_read(reader)
    reader.is_ready.return_value = True
    with receiver._condition:
        receiver._reclaim(session)
    release_received_embeddings(inputs)
    session.release()
    receiver.pool.release.assert_called_once_with((2, 3))
    assert not receiver._receives
