"""Deterministic diagnostics shared by Python AIR passes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class AIRLocation:
    module_id: object
    function_id: Optional[int] = None
    block_id: Optional[int] = None
    statement_id: Optional[int] = None
    node_id: Optional[int] = None
    file_id: Optional[int] = None
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass(frozen=True)
class Diagnostic:
    severity: DiagnosticSeverity
    code: str
    message: str
    location: Optional[AIRLocation] = None
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("diagnostic code must not be empty")
        if not self.message:
            raise ValueError("diagnostic message must not be empty")
        object.__setattr__(self, "notes", tuple(self.notes))


def error(code: str, message: str, *, notes=(), location=None) -> Diagnostic:
    return Diagnostic(
        DiagnosticSeverity.ERROR, code, message, location, tuple(notes)
    )
