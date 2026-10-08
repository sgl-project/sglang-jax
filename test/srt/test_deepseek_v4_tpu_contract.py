"""Trace production attention dispatch with Flash geometry on eight CPU devices.

The real backend, metadata, C pools and execution adapters run. Pallas calls
return their declared output shapes and aliases without tracing kernel bodies.
This detects host/trace contract failures before a costly model load; it does
not validate TPU lowering, memory use, kernel math or model accuracy.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.experimental import pallas as pl
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from test_deepseek_v4_backend import append, batch

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttentionBackend
from sgl_jax.srt.layers.attention.dsv4 import dispatch
from sgl_jax.srt.layers.attention.dsv4.execution import CompressorWeights, IndexerInputs
from sgl_jax.srt.layers.deepseek_v4_mhc import DeepseekV4MHC
from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.models.deepseek_v4 import DeepseekV4Attention
from sgl_jax.srt.utils import jax_utils


@pytest.fixture
def tpu_trace_target(monkeypatch):
    jax.clear_caches()
    token = jax_utils._COMPILATION_TARGET.set(SimpleNamespace(platform="tpu", device_kind="TPU7x"))
    calls = []
    original_compress = dispatch._row_sharded_compress

    def row_sharded_compress(*args, **kwargs):
        calls.append("compressor-local" if kwargs.get("local") else "compressor-full")
        return original_compress(*args, **kwargs)

    monkeypatch.setattr(dispatch, "_row_sharded_compress", row_sharded_compress)

    def declared_call(kernel, *, out_shape, input_output_aliases=None, name=None, **kwargs):
        calls.append(name)
        aliases = {
            out_index: in_index for in_index, out_index in (input_output_aliases or {}).items()
        }

        def call(*inputs):
            inputs = jax.tree.leaves(inputs)
            leaves, tree = jax.tree.flatten(out_shape)
            for output_index, input_index in aliases.items():
                assert 0 <= output_index < len(leaves) and 0 <= input_index < len(inputs)
                assert inputs[input_index].shape == leaves[output_index].shape
                assert inputs[input_index].dtype == leaves[output_index].dtype
            result = []
            for i, shape in enumerate(leaves):
                if i in aliases:
                    result.append(inputs[aliases[i]])
                else:
                    result.append(jnp.zeros(shape.shape, shape.dtype))
            return jax.tree.unflatten(tree, result)

        return call

    monkeypatch.setattr(pl, "pallas_call", declared_call)
    monkeypatch.setattr(
        "sgl_jax.srt.layers.attention.dsv4.hca.get_hca_kernel_schedule",
        lambda _, **kw: get_hca_kernel_schedule("TPU7x", **kw),
    )
    yield calls
    jax_utils._COMPILATION_TARGET.reset(token)
    jax.clear_caches()


@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("project_model", [False, True], ids=["backend", "model-projections"])
@pytest.mark.parametrize("layer_id", [0, 1, 2], ids=["swa", "csa", "hca"])
@pytest.mark.parametrize(
    "start,end,mode",
    [
        (0, 128, ForwardMode.EXTEND),
        (128, 130, ForwardMode.EXTEND),
        (129, 130, ForwardMode.DECODE),
        (255, 256, ForwardMode.DECODE),
        (4095, 4096, ForwardMode.DECODE),
        (0, 8192, ForwardMode.EXTEND),
    ],
    ids=[
        "first-prefill",
        "continued-prefill",
        "decode",
        "boundary-decode",
        "long-decode",
        "8k-prefill",
    ],
)
def test_flash_tpu_dispatch_contract(
    tpu_trace_target, page_size, project_model, layer_id, start, end, mode
):
    if len(jax.devices()) < 8:
        pytest.skip("requires eight CPU devices configured before JAX initialization")
    mesh = Mesh(
        np.asarray(jax.devices()[:8]).reshape(1, 8),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    config = SimpleNamespace(hidden_size=4096, num_attention_heads=64, head_dim=512)
    context_len = max(512, end)
    backend = DeepseekV4AttentionBackend(
        mesh=mesh, page_size=page_size, max_context_len=context_len, config=config
    )
    # Physical CPU devices require overriding the constructor's mesh probe;
    # subsequent dispatch uses the explicit TPU tracing target.
    backend.use_pallas_hca = True
    req = ReqToTokenPool(2, context_len)
    spec = DeepseekV4CacheSpec((0, 4, 128), head_dim=512, index_head_dim=128)
    with jax.set_mesh(mesh):
        capacity = max(4 * page_size, (end + page_size - 1) // page_size * page_size)
        kv = DeepseekV4TokenToKVPool(capacity, capacity, page_size, spec, mesh)
        state = DeepseekV4CompressStatePool(2, spec, mesh)
    allocator = DeepseekV4TokenToKVPoolAllocator(kv)
    backend.bind_resources(req, allocator)
    append(req, allocator, 0, 0, end)
    b = batch(req, 0, start, end, mode=mode, padding=0, page_size=page_size)
    if end == 8192:
        # One request exercises CSA's row-sharded compressor and indexer.
        for field in ("seq_lens", "req_pool_indices", "extend_seq_lens", "extend_prefix_lens"):
            setattr(b, field, getattr(b, field)[:1])
    md = backend.get_forward_metadata(b, request_pool=req, allocator=allocator)
    md.packed = jax.device_put(md.packed, NamedSharding(mesh, P("data")))
    backend.forward_metadata = md
    b.positions = jax.device_put(b.positions, NamedSharding(mesh, P("data")))
    tokens = end - start

    def abstract(shape, dtype, spec):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))

    def compressor(dim, ratio, overlap=1):
        return CompressorWeights(
            abstract((overlap * dim, 4096), jnp.bfloat16, P()),
            abstract((overlap * dim, 4096), jnp.bfloat16, P()),
            abstract((ratio, overlap * dim), jnp.float32, P()),
            abstract((dim,), jnp.float32, P()),
            abstract((context_len, 64), jnp.float32, P()),
        )

    q = abstract((tokens, 64, 512), jnp.bfloat16, P("data", "tensor", None))
    new_kv = abstract((tokens, 512), jnp.bfloat16, P("data", None))
    hidden = abstract((tokens, 4096), jnp.bfloat16, P("data", None))
    sink = abstract((64,), jnp.float32, P("tensor"))
    ratio = spec.compress_ratios[layer_id]
    weights = compressor(512, ratio, 2 if ratio == 4 else 1) if ratio else None
    indexer = (
        IndexerInputs(
            abstract((tokens, 64, 128), jnp.bfloat16, P("data", "tensor", None)),
            abstract((tokens, 64), jnp.float32, P("data", "tensor")),
            compressor(128, 4, 2),
        )
        if ratio == 4
        else None
    )

    def forward(q, new_kv, hidden, sink, weights, indexer):
        return backend(
            q,
            new_kv,
            new_kv,
            SimpleNamespace(layer_id=layer_id, scaling=512**-0.5),
            b,
            kv,
            compressor_state_pool=state,
            compressor_input=hidden,
            compressor=weights,
            indexer=indexer,
            attention_sink=sink,
            norm_eps=3e-5,
            index_topk=512,
        )

    with jax.set_mesh(mesh):
        if project_model:
            config = DeepseekV4Config(
                num_hidden_layers=3,
                compress_ratios=[0, 4, 128],
                max_position_embeddings=context_len,
                rms_norm_eps=3e-5,
            )
            config.quantization_config = QuantizationConfig(is_static_checkpoint=True)

            def initialize():
                attention = DeepseekV4Attention(config, mesh, layer_id, jnp.bfloat16)
                attention.prepare_grouped_wo_a(abstract=True)
                if ratio == 128:
                    attention.compressor.prepare_fused_projection(mesh, abstract=True)
                return attention

            attention = nnx.eval_shape(initialize)
            b.attn_backend = backend
            pools = SimpleNamespace(token_to_kv_pool=kv, compressor_state_pool=state)
            rope_cache = abstract((context_len, 128), jnp.float32, P())
            output, updates = nnx.eval_shape(
                lambda attention, x, rope: attention(
                    x,
                    b,
                    pools,
                    rope,
                    (rope[:, :32], rope[:, 32:64]) if mode.is_decode() else None,
                    sp_local=ratio == 4 and end == 8192,
                    wo_out_sharding=(
                        NamedSharding(mesh, P("tensor", None))
                        if ratio == 4 and end == 8192
                        else None
                    ),
                ),
                attention,
                (
                    abstract(hidden.shape, hidden.dtype, P("tensor", None))
                    if ratio == 4 and end == 8192
                    else hidden
                ),
                rope_cache,
            )
            assert output.shape == hidden.shape and output.dtype == hidden.dtype
        else:
            output, updates = jax.eval_shape(forward, q, new_kv, hidden, sink, weights, indexer)
            assert output.shape == q.shape
            # SWA/CSA retain FP32 accumulators; tuned HCA returns activation dtype.
            assert output.dtype == (q.dtype if ratio == 128 else jnp.float32)
    expected = {"swa"} if ratio == 0 else {"swa", "state", "compressed"}
    if ratio == 4:
        expected |= {"indexer", "indexer_state"}
    assert set(updates) == expected
    owners = {"swa": kv.get_swa_buffer(layer_id)}
    if ratio:
        owners.update(
            compressed=kv.get_compressed_buffer(layer_id),
            state=state.get_buffer(f"c{ratio}", layer_id),
        )
    if ratio == 4:
        owners.update(
            indexer=kv.get_indexer_buffer(layer_id),
            indexer_state=state.get_buffer("indexer", layer_id),
        )
    for family, updated in updates.items():
        assert updated.shape == owners[family].shape
        assert updated.dtype == owners[family].dtype
    if ratio:
        assert tpu_trace_target, "production compressed-attention kernels were bypassed"
    if ratio == 4 and end == 8192:
        assert (
            tpu_trace_target.count("compressor-local" if project_model else "compressor-full") >= 2
        )


@pytest.mark.parametrize("tokens", [1, 128, 768, 8192])
def test_flash_mhc_production_contract(tpu_trace_target, tokens):
    config = DeepseekV4Config()
    mhc = DeepseekV4MHC(config, backend="pallas")

    def abstract(shape, dtype=jnp.float32):
        return jax.ShapeDtypeStruct(shape, dtype)

    streams = abstract((tokens, 4, 4096), jnp.bfloat16)
    fn, base, scale = abstract((24, 16384)), abstract((24,)), abstract((3,))
    hidden, post, comb = jax.eval_shape(mhc.pre, streams, fn, base, scale)
    assert hidden.shape == (tokens, 4096)
    assert post.shape == (tokens, 4) and comb.shape == (tokens, 4, 4)
    updated = jax.eval_shape(mhc.post, hidden, streams, post, comb)
    assert updated.shape == streams.shape
    collapsed = jax.eval_shape(
        mhc.collapse_head, streams, abstract((4, 16384)), abstract((4,)), abstract((1,))
    )
    assert collapsed.shape == hidden.shape
    assert tpu_trace_target, "production mHC kernels were bypassed"
