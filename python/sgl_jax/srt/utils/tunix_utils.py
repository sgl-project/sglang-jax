# refer to tunix
def pathways_available() -> bool:
    try:
        import pathwaysutils  # pylint: disable=g-import-not-at-top, unused-import

        return True
    except ImportError:
        return False
