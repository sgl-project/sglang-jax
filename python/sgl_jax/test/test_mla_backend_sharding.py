from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend


class _FakeKVPool:
    def __init__(self, cache):
        self.cache = cache

    def get_fused_kv_buffer(self, _layer_id):
        return self.cache


def test_mla_backend_reshards_new_k_pe_for_shard_map():
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    captured_inputs = {}

    def fake_shard_map(_fn, *, in_specs, out_specs, check_vma):
        del in_specs, out_specs, check_vma

        def invoke(*args):
            captured_inputs["new_k_pe"] = args[3]
            return args[0], args[4]

        return invoke

    with jax.set_mesh(mesh):
        backend = MLAAttentionBackend(
            num_attn_heads=1,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            page_size=16,
            mesh=mesh,
            attention_data_partition_axis="data",
            vmem_limit_bytes=1,
        )
        metadata_sharding = NamedSharding(mesh, P("data"))
        for field in (
            "seq_lens",
            "page_indices",
            "cu_q_lens",
            "cu_kv_lens",
            "distribution",
        ):
            setattr(
                backend.forward_metadata,
                field,
                jax.device_put(jnp.zeros((1,), dtype=jnp.int32), metadata_sharding),
            )

        q_sharding = NamedSharding(mesh, P("data", "tensor", None))
        kv_sharding = NamedSharding(mesh, P("data", None, None))
        replicated_k_pe_sharding = NamedSharding(mesh, P(None, None, None))
        cache_sharding = NamedSharding(mesh, P("data", None, None, None))
        q = jax.device_put(jnp.ones((1, 1, 4), dtype=jnp.bfloat16), q_sharding)
        k = jax.device_put(jnp.ones((1, 1, 4), dtype=jnp.bfloat16), kv_sharding)
        k_rope = jax.device_put(
            jnp.ones((1, 1, 4), dtype=jnp.bfloat16),
            replicated_k_pe_sharding,
        )
        cache = jax.device_put(
            jnp.zeros((1, 1, 1, 8), dtype=jnp.bfloat16),
            cache_sharding,
        )
        layer = SimpleNamespace(
            layer_id=0,
            scaling=1.0,
            sliding_window_size=None,
            logit_cap=None,
        )

        with mock.patch(
            "sgl_jax.srt.layers.attention.mla_backend.jax.shard_map",
            side_effect=fake_shard_map,
        ):
            backend(
                q,
                k,
                k,
                layer,
                forward_batch=None,
                token_to_kv_pool=_FakeKVPool(cache),
                q_rope=q,
                k_rope=k_rope,
            )

    assert captured_inputs["new_k_pe"].sharding.spec == P("data", None)
