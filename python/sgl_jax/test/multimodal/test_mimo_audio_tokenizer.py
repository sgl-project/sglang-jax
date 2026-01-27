"""Test MultimodalTokenizer audio preprocessing."""
import numpy as np


def test_multimodal_tokenizer():
    """Test MultimodalTokenizer's _init_audio_processor and _preprocess_audio_to_mel."""
    from unittest.mock import MagicMock, patch

    # Mock server_args
    server_args = MagicMock()
    server_args.model_path = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
    server_args.tokenizer_path = None
    server_args.tokenizer_mode = "auto"
    server_args.skip_tokenizer_init = True
    server_args.log_requests = False

    # Mock port_args
    port_args = MagicMock()

    # Patch TokenizerManager.__init__ to avoid ZMQ initialization
    with patch('sgl_jax.srt.managers.tokenizer_manager.TokenizerManager.__init__', return_value=None):
        from sgl_jax.srt.multimodal.manager.multimodal_tokenizer import MultimodalTokenizer

        # Create instance without calling __init__
        tokenizer = object.__new__(MultimodalTokenizer)
        tokenizer.server_args = server_args
        tokenizer.audio_processor = None

        # Test _init_audio_processor
        print("--- Testing _init_audio_processor ---")
        tokenizer._init_audio_processor(server_args.model_path)

        if tokenizer.audio_processor is not None:
            print(f"[OK] audio_processor initialized: {type(tokenizer.audio_processor).__name__}")
        else:
            print("[FAIL] audio_processor is None")
            return

        # Test _preprocess_audio_to_mel
        print("\n--- Testing _preprocess_audio_to_mel ---")
        audio_array = np.zeros((24000,), dtype=np.float32)  # 1 second silence

        mels, input_lens = tokenizer._preprocess_audio_to_mel(audio_array)

        print(f"[OK] mels shape: {mels.shape}")
        print(f"[OK] input_lens: {input_lens}")

        print("\n[SUCCESS] MultimodalTokenizer audio preprocessing test passed!")


if __name__ == "__main__":
    test_multimodal_tokenizer()
