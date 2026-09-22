"""Abstract MTP draft and target-verify inputs for the serving model forward."""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def configure_mtp(config, options):
    # Match ModelConfig(is_draft_model=True): each runner has one SWA block.
    count = getattr(config, "num_nextn_predict_layers", None)
    if count is not None and options.mtp_layer_idx >= count:
        raise ValueError("mtp_layer_idx exceeds num_nextn_predict_layers")
    config.architectures = ["MiMoV2MTPForCausalLM"]
    config.num_hidden_layers = 1
    config.num_attention_heads = config.swa_num_attention_heads
    config.num_key_value_heads = config.swa_num_key_value_heads
    config.head_dim = config.swa_head_dim
    config.hybrid_layer_pattern = [1]
    config.moe_layer_freq = [0]
    config.mtp_layer_idx = options.mtp_layer_idx


def bind_shared_parameter_specs(specs):
    # MultiLayerDraftWorker copies these two arrays from the target. Preserve
    # MiMoV2FlashForCausalLM's checkpoint mapping, rather than initializer sharding.
    specs["model.embed_tokens.embedding"] = P("tensor", None)
    specs["lm_head.embedding"] = P("tensor", None)


def configure_batch(config, options, mesh, batch, logits):
    def shaped(shape, dtype=jnp.int32):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, P("data")))

    bs = options.batch_size
    width = options.draft_token_num or 1
    nt = bs * width
    batch.input_ids = shaped((nt,))
    batch.positions = shaped((nt,))
    batch.out_cache_loc = shaped((nt,))
    batch.spec_algorithm = SpeculativeAlgorithm.NEXTN
    batch.capture_hidden_mode = CaptureHiddenMode.FULL
    logits.capture_hidden_mode = CaptureHiddenMode.FULL
    metadata = batch.attn_backend.forward_metadata
    pages_per_request = -(-options.context_length // options.page_size)
    # Same buckets as FlashAttention's speculative metadata builders. Page
    # IDs, cumulative lengths, and distribution remain dynamic model inputs.
    if options.workload == "mtp-draft":
        page_count = 16384  # get_eagle_multi_step_metadata's per-step page buffer
        if bs * pages_per_request > page_count:
            raise ValueError("MTP draft pages exceed the serving 16384-entry page buffer")
    else:
        page_count = bs * max(16, 1 << (pages_per_request - 1).bit_length())
    metadata.page_indices = shaped((page_count,))
    metadata.swa_page_indices = None if options.workload == "mtp-draft" else shaped((page_count,))

    if options.workload == "target-verify":
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        # NEXTN topk=1 is a causal chain; candidate selection/acceptance runs
        # outside the model forward, so its sampling-only arrays stay absent.
        batch.spec_info = EagleVerifyInput(
            draft_token=None,
            custom_mask=None,
            positions=None,
            retrive_index=None,
            retrive_next_token=None,
            retrive_next_sibling=None,
            spec_steps=width - 1,
            draft_token_num=width,
        )
        logits.logits_indices = shaped((nt,))
    else:
        batch.spec_info = EagleDraftInput(
            hidden_states=shaped((nt, config.hidden_size), jnp.bfloat16)
        )
        if options.workload == "mtp-draft-extend":
            batch.forward_mode = ForwardMode.DRAFT_EXTEND
            batch.extend_prefix_lens = shaped((bs,))
            batch.extend_seq_lens = shaped((bs,))
            batch.spec_info.accept_length = shaped((bs,))
            logits.extend_seq_lens = batch.extend_seq_lens
            logits.accept_lens = batch.spec_info.accept_length
            # Serving emits all hidden rows for the next MTP runner but only
            # the last accepted token's logits for each request.
        else:
            batch.capture_hidden_mode = CaptureHiddenMode.LAST
            logits.capture_hidden_mode = CaptureHiddenMode.LAST
            batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
    logits.forward_mode = batch.forward_mode
