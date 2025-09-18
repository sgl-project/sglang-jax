# RFC: Multi-Token Prediction (MTP) for EAGLE Speculative Decoding

**Created:** 2025-09-16
**Status:** Draft

## Summary

This RFC proposes the implementation of Multi-Token Prediction (MTP) as an enhancement to the existing EAGLE speculative decoding algorithm in SGLang. MTP enables models to predict multiple tokens simultaneously during inference, significantly improving throughput while maintaining generation quality. The feature leverages specially trained model architectures that can natively generate multiple tokens per forward pass.

## Motivation

Current autoregressive language models generate tokens sequentially, which creates inherent bottlenecks in inference throughput. While speculative decoding techniques like EAGLE improve performance through draft-verify mechanisms, they still rely on single-token predictions from the base model. Multi-Token Prediction addresses this limitation by enabling the model to directly predict multiple tokens, reducing the number of forward passes required for sequence generation.

### Key Problems Addressed

1. **Sequential Token Generation Bottleneck:** Traditional autoregressive generation requires one forward pass per token
2. **Inference Latency:** High time-to-first-token and overall generation latency
3. **Resource Utilization:** Suboptimal GPU utilization due to sequential dependencies
4. **Scalability Limitations:** Poor scaling characteristics for long sequence generation

## Goals

### Primary Goals

- Implement MTP capability for compatible model architectures (Qwen, etc.)
- Integrate MTP seamlessly with existing EAGLE speculative decoding framework
- Achieve significant throughput improvements (target: 1.5-1.8x speedup)
- Maintain generation quality and model accuracy
- Support multiple attention backends (FlashAttention3, FlashMLA, Triton)

### Non-Goals

- Retrofitting MTP to models not architecturally designed for it
- Breaking compatibility with existing EAGLE implementations
- Implementing MTP for non-transformer architectures

## Proposal

### Design Overview

MTP extends the EAGLE speculative decoding framework by leveraging models with built-in multi-token prediction capabilities. Instead of generating single draft tokens, the system generates multiple tokens simultaneously from both draft and target models.

### Architecture Components

#### 1. MTP-Enabled Model Interface

```python
class MTPCapableModel:
    def forward_mtp(self,
                    input_ids: torch.Tensor,
                    num_predict_tokens: int,
                    **kwargs) -> MTPOutput:
        """Forward pass with multi-token prediction capability"""
        pass

    @property
    def max_predict_tokens(self) -> int:
        """Maximum number of tokens this model can predict simultaneously"""
        pass
```

#### 2. MTP Configuration

```python
@dataclass
class MTPConfig:
    enabled: bool = False
    max_predict_tokens: int = 4
    draft_tokens_per_step: int = 2
    verify_tokens_per_step: int = 2
    fallback_to_single_token: bool = True
```

#### 3. Integration with EAGLE Worker

```python
class MTPEagleWorker(EAGLEWorker):
    def __init__(self, server_args: ServerArgs, mtp_config: MTPConfig, ...):
        super().__init__(server_args, ...)
        self.mtp_config = mtp_config
        self.mtp_enabled = self._check_mtp_compatibility()

    def draft_forward_mtp(self, forward_batch: ForwardBatch) -> MTPDraftOutput:
        """Multi-token draft generation"""
        pass

    def verify_mtp(self, batch: ScheduleBatch, mtp_draft: MTPDraftOutput) -> MTPVerifyOutput:
        """Multi-token verification"""
        pass
```

## Implementation Details

### 1. Model Architecture Detection

```python
def detect_mtp_capability(model_config: ModelConfig) -> bool:
    """Detect if model supports multi-token prediction"""
    supported_archs = [
        "DeepseekV3ForCausalLM",
        "Qwen3ForCausalLM",  # hypothetical
        "LlamaForCausalLM"   # with MTP extensions
    ]
    return (
        model_config.hf_config.architectures[0] in supported_archs and
        hasattr(model_config.hf_config, 'mtp_config') and
        model_config.hf_config.mtp_config.get('enabled', False)
    )
```

### 2. Multi-Token Draft Generation

```python
def forward_mtp_draft(self, forward_batch: ForwardBatch) -> List[torch.Tensor]:
    """Generate multiple draft tokens per step"""
    batch_size = forward_batch.batch_size
    token_sequences = []

    for step in range(self.speculative_num_steps):
        # Generate multiple tokens simultaneously
        mtp_output = self.draft_model_runner.model.forward_mtp(
            input_ids=forward_batch.input_ids,
            num_predict_tokens=self.mtp_config.draft_tokens_per_step,
            positions=forward_batch.positions,
            **forward_batch.model_kwargs
        )

        # Process multi-token output
        next_tokens = self._process_mtp_output(mtp_output)
        token_sequences.append(next_tokens)

        # Update input for next step
        forward_batch.input_ids = next_tokens[:, -1:]  # Use last token
        forward_batch.positions.add_(self.mtp_config.draft_tokens_per_step)

    return token_sequences
```

### 3. Tree Construction for MTP

```python
def build_mtp_tree(self,
                   verified_tokens: torch.Tensor,
                   mtp_sequences: List[torch.Tensor],
                   scores: List[torch.Tensor]) -> MTPTree:
    """Build verification tree for multi-token sequences"""
    # Construct tree with multi-token branches
    # Each node can have multiple children representing token sequences
    tree_structure = self._build_sequence_tree(mtp_sequences)

    # Generate attention masks for parallel verification
    attention_mask = self._generate_mtp_attention_mask(tree_structure)

    return MTPTree(
        sequences=mtp_sequences,
        tree_structure=tree_structure,
        attention_mask=attention_mask,
        position_ids=self._compute_mtp_positions(tree_structure)
    )
```

### 4. Parallel Verification

```python
def verify_mtp_sequences(self,
                        batch: ScheduleBatch,
                        mtp_tree: MTPTree) -> MTPVerifyResult:
    """Verify multiple token sequences in parallel"""
    # Prepare batch for multi-token verification
    verify_batch = self._prepare_mtp_verify_batch(batch, mtp_tree)

    # Run target model verification
    logits_output = self.target_worker.forward_batch_generation(
        verify_batch, skip_sample=True
    )

    # Accept/reject sequences based on target model predictions
    accepted_sequences = self._evaluate_mtp_sequences(
        logits_output, mtp_tree.sequences
    )

    return MTPVerifyResult(
        accepted_sequences=accepted_sequences,
        acceptance_rate=len(accepted_sequences) / len(mtp_tree.sequences),
        next_tokens=self._extract_accepted_tokens(accepted_sequences)
    )
```

## Configuration Integration

### Server Arguments

```python
# New server arguments for MTP
parser.add_argument(
    "--enable-mtp",
    action="store_true",
    help="Enable Multi-Token Prediction for compatible models"
)
parser.add_argument(
    "--mtp-max-predict-tokens",
    type=int,
    default=4,
    help="Maximum number of tokens to predict simultaneously"
)
parser.add_argument(
    "--mtp-draft-tokens-per-step",
    type=int,
    default=2,
    help="Number of tokens to generate per draft step"
)
```

### Model Configuration

```python
def configure_mtp(self, server_args: ServerArgs) -> MTPConfig:
    """Configure MTP based on model and server settings"""
    if not server_args.enable_mtp:
        return MTPConfig(enabled=False)

    model_max_tokens = self.model_config.get_mtp_max_tokens()
    return MTPConfig(
        enabled=True,
        max_predict_tokens=min(
            server_args.mtp_max_predict_tokens,
            model_max_tokens
        ),
        draft_tokens_per_step=server_args.mtp_draft_tokens_per_step,
        verify_tokens_per_step=min(
            server_args.mtp_draft_tokens_per_step,
            model_max_tokens
        )
    )
```

## Implementation Plan

### Phase 1: Foundation (4 weeks)

- Implement MTP model interface and detection
- Create MTPConfig and integration with ServerArgs
- Develop basic MTP-enabled EAGLEWorker
- Add unit tests for core MTP functionality

### Phase 2: Core Implementation (6 weeks)

- Implement multi-token draft generation
- Develop MTP tree construction algorithms
- Create parallel verification mechanisms
- Integrate with existing attention backends

### Phase 3: Optimization (4 weeks)

- Implement precompile support for MTP
- Add memory optimization for multi-token sequences
- Performance tuning and profiling
- Benchmark against baseline implementations

### Phase 4: Validation & Documentation (2 weeks)

- Comprehensive testing with supported models
- Performance validation and regression testing
- Documentation and user guides
- Integration testing with existing SGLang features

## Alternatives Considered

### 1. Independent MTP Implementation

- **Approach:** Implement MTP as a separate speculative decoding algorithm
- **Pros:** Clean separation, no impact on existing EAGLE code
- **Cons:** Code duplication, maintenance overhead
- **Decision:** Rejected in favor of EAGLE integration

### 2. Model-Agnostic MTP

- **Approach:** Attempt to retrofit MTP to any model architecture
- **Pros:** Universal applicability
- **Cons:** Significant complexity, potential quality degradation
- **Decision:** Rejected; focus on architecturally-supported models

### 3. Token-Level Parallelism Only

- **Approach:** Implement only the parallel verification aspect
- **Pros:** Simpler implementation, lower risk
- **Cons:** Limited performance gains
- **Decision:** Rejected; full MTP provides better benefits

## Risks and Mitigations

### Technical Risks

#### 1. Memory Consumption

- **Risk:** Multi-token sequences require significantly more memory
- **Mitigation:**
  - Implement adaptive batch sizing based on available memory
  - Add memory monitoring and graceful degradation
  - Provide configuration options for memory-constrained environments

#### 2. Model Compatibility

- **Risk:** Limited number of models support native MTP
- **Mitigation:**
  - Clear documentation of supported models
  - Graceful fallback to standard EAGLE for unsupported models
  - Provide model compatibility checking utilities

#### 3. Quality Degradation

- **Risk:** Multi-token prediction might reduce generation quality
- **Mitigation:**
  - Comprehensive quality benchmarking against baselines
  - Tunable acceptance thresholds for quality vs. speed trade-offs
  - A/B testing framework for quality validation

### Operational Risks

#### 1. Configuration Complexity

- **Risk:** Many new parameters might confuse users
- **Mitigation:**
  - Provide sensible defaults for all MTP parameters
  - Auto-configuration based on model architecture
  - Clear documentation with usage examples

#### 2. Backward Compatibility

- **Risk:** Changes might break existing EAGLE implementations
- **Mitigation:**
  - Extensive regression testing
  - Feature flag for MTP enablement
  - Maintain separate code paths where necessary

## Success Metrics

### Performance Targets

- **Throughput Improvement:** 1.5x-1.8x speedup for supported models
- **Latency Reduction:** 20-30% reduction in time-to-first-token
- **Memory Efficiency:** <50% increase in memory usage
- **Quality Preservation:** <2% degradation in standard benchmarks

### Adoption Metrics

- Integration with at least 2 popular MTP-capable model architectures
- Successful deployment in production environments
- Positive community feedback and adoption

## Graduation Criteria

### Alpha Release Criteria

- Basic MTP functionality working with DeepSeek V3
- Core API stability achieved
- Initial performance benchmarks available
- Basic documentation complete

### Beta Release Criteria

- Support for multiple model architectures
- Performance targets achieved
- Comprehensive test coverage
- Production-ready stability

### Stable Release Criteria

- All success metrics achieved
- Community validation and feedback incorporated
- Full feature parity with EAGLE where applicable
- Production deployments successful

## References

1. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
2. [Multi-Token Prediction Paper](https://arxiv.org/abs/2412.19437)
3. [Speculative Decoding Overview](https://arxiv.org/abs/2312.07104)
4. [SGLang EAGLE Documentation](https://docs.sglang.ai/advanced_features/speculative_decoding.html)
5. [Parallel Decoding Paper](https://arxiv.org/abs/2404.05109)

---

**Note:** This RFC is a living document and will be updated as the implementation progresses and community feedback is incorporated.
