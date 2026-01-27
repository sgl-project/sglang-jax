from transformers import PretrainedConfig

from sgl_jax.srt.utils.weight_utils import WeightMapping

# Conv1D transpose: PyTorch (O,I,K) -> JAX (K,I,O)
TRANSPOSE_1D_CONV = (2, 1, 0)


def to_mappings(config: PretrainedConfig) -> dict[str, WeightMapping]:
    """Generate weight mappings from HuggingFace to JAX format for MiMo Audio Tokenizer."""
    mappings = {
        # ===== Encoder Convolutions =====
        "encoder.conv1.weight": WeightMapping(target_path="encoder.conv1.kernel", transpose_axes=TRANSPOSE_1D_CONV),
        "encoder.conv1.bias": WeightMapping(target_path="encoder.conv1.bias", sharding=()),
        "encoder.conv2.weight": WeightMapping(target_path="encoder.conv2.kernel", transpose_axes=TRANSPOSE_1D_CONV),
        "encoder.conv2.bias": WeightMapping(target_path="encoder.conv2.bias", sharding=()),
        # ===== Encoder LayerNorm =====
        "encoder.layer_norm.weight": WeightMapping(target_path="encoder.layer_norm.scale", sharding=()),
        "encoder.layer_norm.bias": WeightMapping(target_path="encoder.layer_norm.bias", sharding=()),
        # ===== Encoder Layers =====
        "encoder.layers.*.self_attn.q_proj.weight": WeightMapping(target_path="encoder.layers.*.self_attn.q_proj.weight", transpose=True),
        "encoder.layers.*.self_attn.q_proj.bias": WeightMapping(target_path="encoder.layers.*.self_attn.q_proj.bias", sharding=()),
        "encoder.layers.*.self_attn.k_proj.weight": WeightMapping(target_path="encoder.layers.*.self_attn.k_proj.weight", transpose=True),
        "encoder.layers.*.self_attn.k_proj.bias": WeightMapping(target_path="encoder.layers.*.self_attn.k_proj.bias", sharding=()),
        "encoder.layers.*.self_attn.v_proj.weight": WeightMapping(target_path="encoder.layers.*.self_attn.v_proj.weight", transpose=True),
        "encoder.layers.*.self_attn.v_proj.bias": WeightMapping(target_path="encoder.layers.*.self_attn.v_proj.bias", sharding=()),
        "encoder.layers.*.self_attn.out_proj.weight": WeightMapping(target_path="encoder.layers.*.self_attn.out_proj.weight", transpose=True),
        "encoder.layers.*.self_attn.out_proj.bias": WeightMapping(target_path="encoder.layers.*.self_attn.out_proj.bias", sharding=()),
        "encoder.layers.*.self_attn_layer_norm.weight": WeightMapping(target_path="encoder.layers.*.self_attn_layer_norm.scale", sharding=()),
        "encoder.layers.*.self_attn_layer_norm.bias": WeightMapping(target_path="encoder.layers.*.self_attn_layer_norm.bias", sharding=()),
        "encoder.layers.*.final_layer_norm.weight": WeightMapping(target_path="encoder.layers.*.final_layer_norm.scale", sharding=()),
        "encoder.layers.*.final_layer_norm.bias": WeightMapping(target_path="encoder.layers.*.final_layer_norm.bias", sharding=()),
        "encoder.layers.*.fc1.weight": WeightMapping(target_path="encoder.layers.*.fc1.weight", transpose=True),
        "encoder.layers.*.fc1.bias": WeightMapping(target_path="encoder.layers.*.fc1.bias", sharding=()),
        "encoder.layers.*.fc2.weight": WeightMapping(target_path="encoder.layers.*.fc2.weight", transpose=True),
        "encoder.layers.*.fc2.bias": WeightMapping(target_path="encoder.layers.*.fc2.bias", sharding=()),
        # ===== Encoder Down Sample =====
        "encoder.down_sample_layer.0.weight": WeightMapping(target_path="encoder.down_sample_layer.kernel", transpose_axes=TRANSPOSE_1D_CONV),
        "encoder.down_sample_norm.weight": WeightMapping(target_path="encoder.down_norm.scale", sharding=()),
        "encoder.down_sample_norm.bias": WeightMapping(target_path="encoder.down_norm.bias", sharding=()),
        # NOTE: codebook weights are loaded manually in load_weights() to avoid WeightLoader._get_param issues with nnx.List + Embed
        # ===== Decoder Deconv =====
        "decoder.dconv1.conv.weight": WeightMapping(target_path="decoder.dconv1.conv.kernel"),
        "decoder.dconv1.conv.bias": WeightMapping(target_path="decoder.dconv1.conv.bias", sharding=()),
        "decoder.dconv1.norm.weight": WeightMapping(target_path="decoder.dconv1.norm.scale", sharding=()),
        "decoder.dconv1.norm.bias": WeightMapping(target_path="decoder.dconv1.norm.bias", sharding=()),
        "decoder.layer_norm.weight": WeightMapping(target_path="decoder.layer_norm.scale", sharding=()),
        "decoder.layer_norm.bias": WeightMapping(target_path="decoder.layer_norm.bias", sharding=()),
        "decoder.dconv2.conv.weight": WeightMapping(target_path="decoder.dconv2.conv.kernel"),
        "decoder.dconv2.conv.bias": WeightMapping(target_path="decoder.dconv2.conv.bias", sharding=()),
        "decoder.dconv2.norm.weight": WeightMapping(target_path="decoder.dconv2.norm.scale", sharding=()),
        "decoder.dconv2.norm.bias": WeightMapping(target_path="decoder.dconv2.norm.bias", sharding=()),
        # ===== Decoder Layers =====
        "decoder.layers.*.self_attn.q_proj.weight": WeightMapping(target_path="decoder.layers.*.self_attn.q_proj.weight", transpose=True),
        "decoder.layers.*.self_attn.q_proj.bias": WeightMapping(target_path="decoder.layers.*.self_attn.q_proj.bias", sharding=()),
        "decoder.layers.*.self_attn.k_proj.weight": WeightMapping(target_path="decoder.layers.*.self_attn.k_proj.weight", transpose=True),
        "decoder.layers.*.self_attn.k_proj.bias": WeightMapping(target_path="decoder.layers.*.self_attn.k_proj.bias", sharding=()),
        "decoder.layers.*.self_attn.v_proj.weight": WeightMapping(target_path="decoder.layers.*.self_attn.v_proj.weight", transpose=True),
        "decoder.layers.*.self_attn.v_proj.bias": WeightMapping(target_path="decoder.layers.*.self_attn.v_proj.bias", sharding=()),
        "decoder.layers.*.self_attn.out_proj.weight": WeightMapping(target_path="decoder.layers.*.self_attn.out_proj.weight", transpose=True),
        "decoder.layers.*.self_attn.out_proj.bias": WeightMapping(target_path="decoder.layers.*.self_attn.out_proj.bias", sharding=()),
        "decoder.layers.*.self_attn_layer_norm.weight": WeightMapping(target_path="decoder.layers.*.self_attn_layer_norm.scale", sharding=()),
        "decoder.layers.*.self_attn_layer_norm.bias": WeightMapping(target_path="decoder.layers.*.self_attn_layer_norm.bias", sharding=()),
        "decoder.layers.*.final_layer_norm.weight": WeightMapping(target_path="decoder.layers.*.final_layer_norm.scale", sharding=()),
        "decoder.layers.*.final_layer_norm.bias": WeightMapping(target_path="decoder.layers.*.final_layer_norm.bias", sharding=()),
        "decoder.layers.*.fc1.weight": WeightMapping(target_path="decoder.layers.*.fc1.weight", transpose=True),
        "decoder.layers.*.fc1.bias": WeightMapping(target_path="decoder.layers.*.fc1.bias", sharding=()),
        "decoder.layers.*.fc2.weight": WeightMapping(target_path="decoder.layers.*.fc2.weight", transpose=True),
        "decoder.layers.*.fc2.bias": WeightMapping(target_path="decoder.layers.*.fc2.bias", sharding=()),
        # ===== Vocoder =====
        "decoder.vocoder.embeddings.weight": WeightMapping(target_path="decoder.vocoder.embeddings.weight", transpose=True),
        "decoder.vocoder.layer_norm.weight": WeightMapping(target_path="decoder.vocoder.layer_norm.scale", sharding=()),
        "decoder.vocoder.layer_norm.bias": WeightMapping(target_path="decoder.vocoder.layer_norm.bias", sharding=()),
        "decoder.vocoder.head.out.weight": WeightMapping(target_path="decoder.vocoder.head.linear.weight", transpose=True),
        "decoder.vocoder.head.out.bias": WeightMapping(target_path="decoder.vocoder.head.linear.bias", sharding=()),
        "decoder.vocoder.head.istft.window": WeightMapping(target_path="decoder.vocoder.head.istft.window", sharding=()),
        # ===== Vocoder Layers =====
        "decoder.vocoder.layers.*.self_attn.q_proj.weight": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.q_proj.weight", transpose=True),
        "decoder.vocoder.layers.*.self_attn.q_proj.bias": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.q_proj.bias", sharding=()),
        "decoder.vocoder.layers.*.self_attn.k_proj.weight": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.k_proj.weight", transpose=True),
        "decoder.vocoder.layers.*.self_attn.k_proj.bias": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.k_proj.bias", sharding=()),
        "decoder.vocoder.layers.*.self_attn.v_proj.weight": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.v_proj.weight", transpose=True),
        "decoder.vocoder.layers.*.self_attn.v_proj.bias": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.v_proj.bias", sharding=()),
        "decoder.vocoder.layers.*.self_attn.out_proj.weight": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.out_proj.weight", transpose=True),
        "decoder.vocoder.layers.*.self_attn.out_proj.bias": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn.out_proj.bias", sharding=()),
        "decoder.vocoder.layers.*.self_attn_layer_norm.weight": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn_layer_norm.scale", sharding=()),
        "decoder.vocoder.layers.*.self_attn_layer_norm.bias": WeightMapping(target_path="decoder.vocoder.layers.*.self_attn_layer_norm.bias", sharding=()),
        "decoder.vocoder.layers.*.final_layer_norm.weight": WeightMapping(target_path="decoder.vocoder.layers.*.final_layer_norm.scale", sharding=()),
        "decoder.vocoder.layers.*.final_layer_norm.bias": WeightMapping(target_path="decoder.vocoder.layers.*.final_layer_norm.bias", sharding=()),
        "decoder.vocoder.layers.*.fc1.weight": WeightMapping(target_path="decoder.vocoder.layers.*.fc1.weight", transpose=True),
        "decoder.vocoder.layers.*.fc1.bias": WeightMapping(target_path="decoder.vocoder.layers.*.fc1.bias", sharding=()),
        "decoder.vocoder.layers.*.fc2.weight": WeightMapping(target_path="decoder.vocoder.layers.*.fc2.weight", transpose=True),
        "decoder.vocoder.layers.*.fc2.bias": WeightMapping(target_path="decoder.vocoder.layers.*.fc2.bias", sharding=()),
    }
    return mappings
