"""CPU-only tests of namespace selection and native-before-JAX ordering."""

import sys
import types

import pytest
from sgl_jax import raiden


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    for name in list(sys.modules):
        if name.split(".")[0] in ("jax", "jaxlib", "tpu_sync", "tpu_raiden"):
            monkeypatch.delitem(sys.modules, name)


def extension(namespace):
    return namespace + ".frameworks.jax._tpu_raiden_jax"


@pytest.mark.parametrize("namespace", ["tpu_sync", "tpu_raiden"])
def test_preload_selects_namespace_and_keeps_manager_consistent(monkeypatch, namespace):
    calls = []
    manager = object()

    def import_module(name):
        calls.append(name)
        assert "jax" not in sys.modules
        if name == extension("tpu_sync") and namespace == "tpu_raiden":
            raise ModuleNotFoundError(name="tpu_sync")
        module = types.ModuleType(name)
        if name.endswith("kv_cache_manager"):
            module.KVCacheManager = manager
        monkeypatch.setitem(sys.modules, name, module)
        return module

    monkeypatch.setattr(raiden.importlib, "import_module", import_module)
    raiden.preload_raiden()
    raiden.require_raiden_preloaded()
    assert raiden.get_raiden_kv_cache_manager() is manager
    expected = [extension("tpu_sync")]
    if namespace == "tpu_raiden":
        expected.append(extension("tpu_raiden"))
    assert calls == expected + [namespace + ".api.jax.kv_cache_manager"]
    # Repeated preload after JAX initialization is safe and never switches wheels.
    monkeypatch.setitem(sys.modules, "jax", types.ModuleType("jax"))
    raiden.preload_raiden()
    assert calls == expected + [namespace + ".api.jax.kv_cache_manager"]


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError(name="tpu_sync.frameworks"),
        ModuleNotFoundError(name="native_dependency"),
        ImportError("undefined symbol: PJRT_Api"),
        OSError("incompatible architecture"),
    ],
)
def test_broken_new_wheel_never_falls_back(monkeypatch, error):
    calls = []

    def import_module(name):
        calls.append(name)
        raise error

    monkeypatch.setattr(raiden.importlib, "import_module", import_module)
    with pytest.raises(type(error)) as caught:
        raiden.preload_raiden()
    assert caught.value is error
    assert calls == [extension("tpu_sync")]


def test_absent_wheels_report_installation_error(monkeypatch):
    def import_module(name):
        raise ModuleNotFoundError(name=name.split(".")[0])

    monkeypatch.setattr(raiden.importlib, "import_module", import_module)
    with pytest.raises(ModuleNotFoundError, match="Neither tpu_sync nor tpu_raiden"):
        raiden.preload_raiden()


@pytest.mark.parametrize("module", ["jax", "jaxlib"])
def test_preload_rejects_late_native_loading(monkeypatch, module):
    monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
    with pytest.raises(RuntimeError, match="before jax/jaxlib"):
        raiden.preload_raiden()


def test_manager_requires_preload():
    with pytest.raises(RuntimeError, match="was not preloaded"):
        raiden.get_raiden_kv_cache_manager()


@pytest.mark.parametrize(
    "operation",
    [
        raiden.preload_raiden,
        raiden.require_raiden_preloaded,
        raiden.get_raiden_kv_cache_manager,
    ],
)
def test_mixed_native_extensions_rejected(monkeypatch, operation):
    for namespace in ("tpu_sync", "tpu_raiden"):
        name = extension(namespace)
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    with pytest.raises(RuntimeError, match="Both tpu_sync and tpu_raiden"):
        operation()


def test_preloaded_legacy_is_retained_when_new_api_available(monkeypatch):
    name = extension("tpu_raiden")
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    calls = []

    def import_module(name):
        calls.append(name)
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(raiden.importlib, "import_module", import_module)
    raiden.preload_raiden()
    with pytest.raises(ModuleNotFoundError):
        raiden.get_raiden_kv_cache_manager()
    assert calls == ["tpu_raiden.api.jax.kv_cache_manager"]
