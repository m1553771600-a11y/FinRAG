"""finrag._logging —— 标准 logging 的轻量封装。

与旧 logger.py 的区别：无 import 副作用、无 sys.path hack、不读全局配置。
"""

from __future__ import annotations

import logging
from typing import Any


class _Logger:
    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def _fmt(self, event: str, **kw: Any) -> str:
        if kw:
            return event + " | " + " ".join(f"{k}={v!r}" for k, v in kw.items())
        return event

    def debug(self, event: str, **kw: Any) -> None:
        self._log.debug(self._fmt(event, **kw))

    def info(self, event: str, *args: Any, **kw: Any) -> None:
        if args:
            self._log.info(event, *args)
        else:
            self._log.info(self._fmt(event, **kw))

    def warning(self, event: str, *args: Any, **kw: Any) -> None:
        if args:
            self._log.warning(event, *args)
        else:
            self._log.warning(self._fmt(event, **kw))

    def error(self, event: str, **kw: Any) -> None:
        exc_info = kw.pop("exc_info", False)
        self._log.error(self._fmt(event, **kw), exc_info=exc_info)


def get_logger(name: str) -> _Logger:
    return _Logger(name)
