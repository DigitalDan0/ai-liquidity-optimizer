from __future__ import annotations

import sys
from dataclasses import dataclass as _stdlib_dataclass
from typing import Any, Callable


def dataclass(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    """Compatibility wrapper for dataclasses.dataclass.

    Python 3.10+ supports ``slots=...``; Python 3.9 does not.
    This wrapper drops the ``slots`` argument on 3.9 so the same code runs.
    """

    if sys.version_info < (3, 10):
        kwargs.pop("slots", None)
    return _stdlib_dataclass(*args, **kwargs)

