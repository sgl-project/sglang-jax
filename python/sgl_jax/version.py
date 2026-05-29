try:
    from sgl_jax._version import __version__
except ModuleNotFoundError:
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("sglang-jax")
    except PackageNotFoundError:
        __version__ = "0.0.0.dev0"
