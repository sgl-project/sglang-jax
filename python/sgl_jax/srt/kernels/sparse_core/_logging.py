"""Minimal ``init_logger`` shim so the vendored tpu-inference kernels keep their logging calls."""

import logging


class _OnceLogger(logging.LoggerAdapter):
    def __init__(self, logger):
        super().__init__(logger, {})
        self._seen: set[str] = set()

    def warning_once(self, msg, *args, **kwargs):
        key = msg % args if args else str(msg)
        if key in self._seen:
            return
        self._seen.add(key)
        self.warning(msg, *args, **kwargs)


def init_logger(name: str) -> _OnceLogger:
    return _OnceLogger(logging.getLogger(name))
