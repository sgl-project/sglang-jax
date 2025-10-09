import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput


class SingleModule(nnx.Module):

    def __init__(self, rngs: nnx.Rngs, mesh: Mesh):
        self.w = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), ("tensor", None), mesh=mesh)(
                rngs.params(), (12, 4), jnp.float32
            )
        )
        print(self.w.value.sharding)

    def __call__(self, x):
        return LogitsProcessorOutput(next_token_logits=x @ self.w)


@functools.partial(
    nnx.jit,
    donate_argnames=["x"],
    static_argnames=["model_state_def"],
)
def f(model_def, model_state_def, model_state_leaves, x):
    model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
    model = nnx.merge(model_def, model_state)
    output = model(x)
    jax.debug.inspect_array_sharding(model.w, callback=lambda x: print("OOO ", x))
    return output, [], True


def main():
    devs = jax.devices()
    mesh = Mesh(devs, axis_names=("tensor",))

    x = jnp.asarray(np.random.randn(1, 12), dtype=jnp.float32)
    print(mesh)

    model = SingleModule(rngs=nnx.Rngs(default=0), mesh=mesh)
    with jax.sharding.use_mesh(mesh):
        model_def, model_state = nnx.split(model)
        model_state_leaves, model_state_def = jax.tree.flatten(model_state)

        print(model_def)
        print(model_state)
        print(model_state_def)
        print(model_state_leaves)

        print(x.sharding)
        y, _, _ = f(model_def, model_state_def, model_state_leaves, x)
        print("OK:", y.next_token_logits.shape)
        print(y.next_token_logits.sharding)


if __name__ == "__main__":
    main()
