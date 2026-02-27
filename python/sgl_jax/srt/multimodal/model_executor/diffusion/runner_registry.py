"""A registry for mapping diffusion models to their specific runners."""
from sgl_jax.srt.multimodal.model_executor.diffusion.diffusion_model_runner import (
    DiffusionModelRunner,
)
from sgl_jax.srt.multimodal.models.ltx2.diffusion.ltx2_dit import (
    LTX2Transformer3DModel,
)

# Mapping from model class to the specific runner class
_RUNNER_REGISTRY = {
    # LTX2Transformer3DModel will fall back to DiffusionModelRunner natively
}

def get_diffusion_runner_class(model_class):
    """
    Gets the appropriate diffusion runner class for a given model class.

    Args:
        model_class: The model's class type.

    Returns:
        The specialized runner class, or the default DiffusionModelRunner.
    """
    return _RUNNER_REGISTRY.get(model_class, DiffusionModelRunner)
