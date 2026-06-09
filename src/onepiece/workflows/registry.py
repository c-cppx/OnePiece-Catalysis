"""Operation registry helpers for OnePiece workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

OperationHandler = Callable[..., Any]


def operation_handlers() -> dict[str, OperationHandler]:
    """Return the registered workflow operation handlers."""
    from onepiece.workflows.engine import _OPERATION_HANDLERS

    return dict(_OPERATION_HANDLERS)
