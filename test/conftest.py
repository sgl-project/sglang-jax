import pytest


def _detect_tpu() -> bool:
    """Check if TPU hardware is available."""
    try:
        import jax

        return jax.default_backend() == "tpu"
    except Exception:
        return False


HAS_TPU = _detect_tpu()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tpu_only" in item.keywords and not HAS_TPU:
            item.add_marker(pytest.mark.skip(reason="tpu_only: no TPU available"))
        if "cpu_only" in item.keywords and HAS_TPU:
            item.add_marker(pytest.mark.skip(reason="cpu_only: running on accelerator"))
