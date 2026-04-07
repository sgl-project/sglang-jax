# weight_utils.py Refactoring TODO

Remaining items identified during MiMo-V2-Flash weight loading extraction (PR #149 review).

## 1. MoE Scale Reshape Dedup

`_maybe_convert_epmoe_scale_for_kernel` contains ~130 lines of duplicated reshape logic
for EPMoE scales. The fused-MoE and EPMoE paths both compute `(E, k_blocks, 1, out_size)`
layouts with nearly identical code. Should be deduplicated into a shared helper.

## 2. `_maybe_convert_epmoe_scale_for_kernel` Ownership

This method lives in the shared `WeightLoader` but is EPMoE-specific. Consider whether it
should be a model-side hook (like `_expand_linear_block_scale`) or remain shared since
multiple models may use EPMoE.

## 3. Missing Weight Zero-Fill Safety Review

Some weight loading paths silently skip weights that don't match expected shapes.
Review whether a warning/error should be logged when a weight is skipped due to
shape mismatch, to catch silent correctness bugs.

## 4. model_config.py SWA Notes

`configure_for_tensor_parallel` contains a 12-line block for `swa_num_key_value_heads`
adjustment. This is attribute-driven and runs before model instantiation, so it cannot
use the model hook pattern. It's generic enough (triggered by HF config attribute presence)
to remain in `ModelConfig`, but should be documented as a known coupling point.
