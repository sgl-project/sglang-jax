import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.multimodal.models.mimo_audio.mimo_audio_tokenizer import ISTFTHead


def test_istft_head_forward():
    mesh = jax.make_mesh((1, 1), ("data", "tensor"))
    hidden_states = jnp.full((1, 3, 4), 1e4, dtype=jnp.float32)

    with jax.set_mesh(mesh):
        head = ISTFTHead(
            dim=4,
            n_fft=8,
            hop_length=2,
            mesh=mesh,
            dtype=jnp.float32,
        )
        audio = head(hidden_states)

    assert audio.shape == (1, 6)
    assert np.isfinite(np.asarray(audio)).all()
