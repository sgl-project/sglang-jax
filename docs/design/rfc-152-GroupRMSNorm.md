# GroupRMSNorm Layer Implementation

## Background

GroupRMSNorm is an internal component of Linear Attention, used to perform grouped normalization on attention outputs before output gating.

Linear Attention lacks the softmax constraint on numerical range, so output magnitudes can be unbounded. GroupRMSNorm normalizes before the gate to stabilize values. Within the same model, [MLA](https://huggingface.co/inclusionAI/Ling-2.5-1T/blob/main/modeling_bailing_moe_v2_5.py#L492) (which has softmax) does not need GroupRMSNorm, while [Linear Attention](https://huggingface.co/inclusionAI/Ling-2.5-1T/blob/main/modeling_bailing_moe_v2_5.py#L878-L883) (which has no softmax) does.

### Position in Linear Attention

Reference: [Ling-2.5-1T official implementation](https://huggingface.co/inclusionAI/Ling-2.5-1T/blob/main/modeling_bailing_moe_v2_5.py#L878-L883)

```
Q, K, V (projections)
  ↓
Linear Attention Kernel(Q, K, V)  →  produces o (shape: batch, seq, num_heads, head_v_dim)
  ↓
GroupRMSNorm(o)                   →  per-head group normalization (group_norm_size=8)
  ↓
× sigmoid(gate)                   →  gating
  ↓
Output Projection                 →  project back to hidden_size
```

![Architecture](https://mdn.alipayobjects.com/huamei_d2byvp/afts/img/b5bYQJCGUzwAAAAAWJAAAAgADod9AQFr/original)

## Goal

Implement the GroupRMSNorm layer for [Ling-2.5-1T](https://huggingface.co/inclusionAI/Ling-2.5-1T) Linear Attention output gating.

## Implementation

JAX port of the following [PyTorch implementation](https://huggingface.co/inclusionAI/Ling-2.5-1T/blob/main/modeling_bailing_moe_v2_5.py#L185-L204):

```python
class BailingMoeV2_5GroupRMSNorm(nn.Module):
    def __init__(self, hidden_size, group_norm_size, eps=1e-6):
        """
        BailingMoeV2_5RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_norm_size = group_norm_size
        assert hidden_size % group_norm_size == 0, "hidden_size must be divisible by group_norm_size"
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.size()
        group_input_shape = input_shape[:-1] + (self.group_norm_size, input_shape[-1] // self.group_norm_size)
        hidden_states = hidden_states.view(group_input_shape)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype).view(input_shape)
```

## Testing

Test designed with reference to:
```python
@pytest.mark.skip(reason="group_rms_norm function removed from attention_gla")
class TestGroupRMSNorm:
  """GroupRMSNorm compared to numpy reference."""

  @pytest.mark.parametrize("group_size", [4, 8])
  def test_random(self, group_size: int) -> None:
    rng = np.random.default_rng(99)
    D = 128
    shape = (2, 16, D)
    x_np = rng.standard_normal(shape).astype(np.float32)
    w_np = rng.standard_normal(D).astype(np.float32)

    jax_out = group_rms_norm(jnp.array(x_np), jnp.array(w_np), group_size, eps=1e-6)  # pylint: disable=undefined-variable

    # Manual reference
    gs = group_size
    x_g = x_np.reshape(*shape[:-1], gs, D // gs)
    var = np.mean(x_g**2, axis=-1, keepdims=True)
    x_normed = x_g / np.sqrt(var + 1e-6)
    ref = w_np * x_normed.reshape(shape)
    _assert_close(jax_out, ref, msg=f"group_rms_norm gs={group_size}")
```

### Shape Preservation

Construct random input `(batch=2, seq=16, hidden=128)` with `num_groups=8`, assert output shape matches input.

### Group Independence

Construct two inputs that differ only in group 0 (first 16 dimensions), assert that the remaining 7 groups' outputs are unchanged and group 0's output has changed.

### Learnable Scale

Set random weights, compare JAX output against a NumPy reference implementation (`weight * normed_x`), assert agreement within `rtol=1e-6`.

### Cross-Framework Consistency

Compare JAX `GroupRMSNorm` against the official PyTorch `BailingMoeV2_5GroupRMSNorm` with identical inputs and weights:

- Output shapes match
- Numerical match with default weights (ones) at `rtol=1e-6`
- Numerical match with random weights
- 2D input `(batch, hidden)` compatibility
- Precision report: `max abs diff = 9.5e-7`, `mean abs diff = 1.8e-8`

Precision differences stem from different floating-point operation ordering between XLA and PyTorch backends — normal float32 rounding errors.

### Implementation Decisions

1. **JAX implementation aligned with official PyTorch**: reshape → per-group variance → rsqrt → reshape back → scale
2. **float32 upcast**: Consistent with the PyTorch implementation — upcast to float32 before computation, cast back to original dtype after normalization
