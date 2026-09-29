import jax
import jax.numpy as jnp

from sgl_jax.srt import precision_tracer as tracer_module


def test_disabled_tracing_adds_no_executable_outputs():
    tracer = tracer_module.PrecisionTracer()

    @jax.jit
    def forward(x):
        flags = [tracer.jit_pure_callback_record(x, "x", "TEST", i) for i in range(109)]
        return x * 2, flags

    x = jnp.arange(4, dtype=jnp.float32)
    compiled = forward.lower(x).compile()
    assert len(jax.tree.leaves(compiled.out_info)) == 1
