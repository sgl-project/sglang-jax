import pytest


def _detect_accelerator() -> bool:
    """Check if any non-CPU accelerator (TPU or GPU) is available."""
    try:
        import jax

        return jax.default_backend() != "cpu"
    except Exception:
        return False


HAS_ACCELERATOR = _detect_accelerator()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tpu_only" in item.keywords and not HAS_ACCELERATOR:
            item.add_marker(pytest.mark.skip(reason="tpu_only: no TPU available"))
        if "cpu_only" in item.keywords and HAS_ACCELERATOR:
            item.add_marker(pytest.mark.skip(reason="cpu_only: running on accelerator"))
