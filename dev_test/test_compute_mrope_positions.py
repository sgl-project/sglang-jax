import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, List


class ForwardMode:
    """Mock ForwardMode class"""
    DECODE = 1
    EXTEND = 2
    
    def __init__(self, mode):
        self.mode = mode
    
    def is_decode(self):
        return self.mode == self.DECODE
    
    def is_extend(self):
        return self.mode == self.EXTEND


@dataclass
class MockMultimodalInput:
    """Mock multimodal input"""
    mrope_positions: jnp.ndarray


@dataclass
class MockModelRunner:
    """Mock ModelRunner"""
    device: str = "cpu"


@dataclass
class MockBatch:
    """Mock ModelWorkerBatch"""
    multimodal_inputs: List[Optional[MockMultimodalInput]]
    extend_seq_lens: Optional[jnp.ndarray] = None
    extend_prefix_lens: Optional[jnp.ndarray] = None


class TestForwardBatch:
    """Test class with JAX implementation"""
    
    def __init__(self, forward_mode, seq_lens):
        self.forward_mode = forward_mode
        self.seq_lens = seq_lens
        self.mrope_positions = None
    
    def _expand_mrope_from_input(self, mm_input, seq_len, device):
        """Mock implementation - returns last position repeated 3 times"""
        return jnp.full((3, 1), seq_len - 1, dtype=jnp.int64)
    
    def _compute_mrope_positions(
        self, model_runner: MockModelRunner, batch: MockBatch
    ):
        # batch_size * [3 * seq_len]
        batch_size = self.seq_lens.shape[0]
        mrope_positions_list = [[]] * batch_size
        for batch_idx in range(batch_size):
            mm_input = batch.multimodal_inputs[batch_idx]
            if self.forward_mode.is_decode():
                # 3 * N
                if mm_input is None:
                    mrope_positions_list[batch_idx] = jnp.full(
                        (3, 1),
                        self.seq_lens[batch_idx] - 1,
                        dtype=jnp.int64,
                    )
                else:
                    mrope_positions = self._expand_mrope_from_input(
                        mm_input, self.seq_lens[batch_idx], model_runner.device
                    )
                    mrope_positions_list[batch_idx] = mrope_positions
            elif self.forward_mode.is_extend():
                extend_seq_len, extend_prefix_len = (
                    batch.extend_seq_lens[batch_idx],
                    batch.extend_prefix_lens[batch_idx],
                )
                if mm_input is None:
                    # text only
                    mrope_positions = jnp.array(
                        [
                            [
                                pos
                                for pos in range(
                                    extend_prefix_len,
                                    extend_prefix_len + extend_seq_len,
                                )
                            ]
                        ]
                        * 3
                    )
                else:
                    mrope_positions = mm_input.mrope_positions[
                        :,
                        extend_prefix_len : extend_prefix_len + extend_seq_len,
                    ]
                    if mrope_positions.size == 0:
                        mrope_positions = self._expand_mrope_from_input(
                            mm_input, self.seq_lens[batch_idx], model_runner.device
                        )
                mrope_positions_list[batch_idx] = mrope_positions

        self.mrope_positions = jnp.concatenate(
            mrope_positions_list,
            axis=1,
        ).astype(jnp.int64)


def test_decode_mode_no_mm():
    """Test decode mode without multimodal input"""
    print("=" * 80)
    print("TEST 1: Decode mode without multimodal input (JAX)")
    print("=" * 80)
    
    seq_lens = jnp.array([10, 15, 20])
    forward_mode = ForwardMode(ForwardMode.DECODE)
    batch = MockBatch(multimodal_inputs=[None, None, None])
    model_runner = MockModelRunner()
    
    fb = TestForwardBatch(forward_mode, seq_lens)
    fb._compute_mrope_positions(model_runner, batch)
    
    print(f"Input seq_lens: {seq_lens}")
    print(f"Output shape: {fb.mrope_positions.shape}")
    print(f"Output dtype: {fb.mrope_positions.dtype}")
    print(f"Output:\n{fb.mrope_positions}")
    print()


def test_extend_mode_no_mm():
    """Test extend mode without multimodal input (text only)"""
    print("=" * 80)
    print("TEST 2: Extend mode without multimodal input - text only (JAX)")
    print("=" * 80)
    
    seq_lens = jnp.array([15, 20])
    forward_mode = ForwardMode(ForwardMode.EXTEND)
    
    # Batch 0: extend from position 5 to 10 (5 tokens)
    # Batch 1: extend from position 10 to 15 (5 tokens)
    batch = MockBatch(
        multimodal_inputs=[None, None],
        extend_seq_lens=jnp.array([5, 5]),
        extend_prefix_lens=jnp.array([5, 10])
    )
    model_runner = MockModelRunner()
    
    fb = TestForwardBatch(forward_mode, seq_lens)
    fb._compute_mrope_positions(model_runner, batch)
    
    print(f"Input seq_lens: {seq_lens}")
    print(f"Extend seq_lens: {batch.extend_seq_lens}")
    print(f"Extend prefix_lens: {batch.extend_prefix_lens}")
    print(f"Output shape: {fb.mrope_positions.shape}")
    print(f"Output dtype: {fb.mrope_positions.dtype}")
    print(f"Output:\n{fb.mrope_positions}")
    print()


def test_extend_mode_with_mm():
    """Test extend mode with multimodal input"""
    print("=" * 80)
    print("TEST 3: Extend mode with multimodal input (JAX)")
    print("=" * 80)
    
    seq_lens = jnp.array([20])
    forward_mode = ForwardMode(ForwardMode.EXTEND)
    
    # Create mock mrope_positions: 3 x 20
    mock_mrope = jnp.arange(60).reshape(3, 20)
    mm_input = MockMultimodalInput(mrope_positions=mock_mrope)
    
    # Extract positions from 5 to 10
    batch = MockBatch(
        multimodal_inputs=[mm_input],
        extend_seq_lens=jnp.array([5]),
        extend_prefix_lens=jnp.array([5])
    )
    model_runner = MockModelRunner()
    
    fb = TestForwardBatch(forward_mode, seq_lens)
    fb._compute_mrope_positions(model_runner, batch)
    
    print(f"Input seq_lens: {seq_lens}")
    print(f"Mock mrope_positions shape: {mock_mrope.shape}")
    print(f"Mock mrope_positions:\n{mock_mrope}")
    print(f"Extend seq_lens: {batch.extend_seq_lens}")
    print(f"Extend prefix_lens: {batch.extend_prefix_lens}")
    print(f"Output shape: {fb.mrope_positions.shape}")
    print(f"Output dtype: {fb.mrope_positions.dtype}")
    print(f"Output:\n{fb.mrope_positions}")
    print()


def test_extend_mode_with_empty_mm():
    """Test extend mode with empty multimodal positions"""
    print("=" * 80)
    print("TEST 4: Extend mode with empty multimodal positions (JAX)")
    print("=" * 80)
    
    seq_lens = jnp.array([10])
    forward_mode = ForwardMode(ForwardMode.EXTEND)
    
    # Create empty mrope_positions
    mock_mrope = jnp.array([]).reshape(3, 0)
    mm_input = MockMultimodalInput(mrope_positions=mock_mrope)
    
    batch = MockBatch(
        multimodal_inputs=[mm_input],
        extend_seq_lens=jnp.array([5]),
        extend_prefix_lens=jnp.array([5])
    )
    model_runner = MockModelRunner()
    
    fb = TestForwardBatch(forward_mode, seq_lens)
    fb._compute_mrope_positions(model_runner, batch)
    
    print(f"Input seq_lens: {seq_lens}")
    print(f"Mock mrope_positions shape: {mock_mrope.shape}")
    print(f"Mock mrope_positions size: {mock_mrope.size}")
    print(f"Output shape: {fb.mrope_positions.shape}")
    print(f"Output dtype: {fb.mrope_positions.dtype}")
    print(f"Output:\n{fb.mrope_positions}")
    print()


def test_mixed_batch():
    """Test mixed batch with multiple sequences"""
    print("=" * 80)
    print("TEST 5: Mixed batch - decode mode (JAX)")
    print("=" * 80)
    
    seq_lens = jnp.array([10, 15, 20])
    forward_mode = ForwardMode(ForwardMode.DECODE)
    
    # Mix of None and multimodal inputs
    batch = MockBatch(multimodal_inputs=[None, None, None])
    model_runner = MockModelRunner()
    
    fb = TestForwardBatch(forward_mode, seq_lens)
    fb._compute_mrope_positions(model_runner, batch)
    
    print(f"Input seq_lens: {seq_lens}")
    print(f"Output shape: {fb.mrope_positions.shape}")
    print(f"Output dtype: {fb.mrope_positions.dtype}")
    print(f"Output:\n{fb.mrope_positions}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("JAX VERSION TESTS")
    print("=" * 80 + "\n")
    
    test_decode_mode_no_mm()
    test_extend_mode_no_mm()
    test_extend_mode_with_mm()
    test_extend_mode_with_empty_mm()
    test_mixed_batch()
    
    print("=" * 80)
    print("ALL JAX TESTS COMPLETED")
    print("=" * 80)
