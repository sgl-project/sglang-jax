# refer to tunix
def pathways_available() -> bool:
    try:
        # ruff: noqa: F401
        import pathwaysutils

        return True
    except ImportError:
        return False
