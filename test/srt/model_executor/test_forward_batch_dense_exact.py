"""ForwardBatch.dsa_dense_exact is static aux data: survives the pytree round trip and defaults to False."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


def test_dense_exact_round_trip():
    obj = ForwardBatch.__new__(ForwardBatch)
    for name in ForwardBatch.__dataclass_fields__:
        setattr(obj, name, None)
    obj.bid = 0
    obj.deterministic = True
    obj.dsa_dense_exact = True
    children, aux = obj.tree_flatten()
    assert aux["dsa_dense_exact"] is True
    assert set(aux) == {
        "forward_mode",
        "batch_size",
        "spec_algorithm",
        "capture_hidden_mode",
        "deterministic",
        "dsa_dense_exact",
    }
    back = ForwardBatch.tree_unflatten(aux, children)
    assert back.dsa_dense_exact is True
    aux2 = dict(aux)
    del aux2["dsa_dense_exact"]
    assert ForwardBatch.tree_unflatten(aux2, children).dsa_dense_exact is False
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    assert jax.tree_util.tree_unflatten(treedef, leaves).dsa_dense_exact is True
