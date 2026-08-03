"""Temporary native adapters retained only for staged migration."""

from .tentative_native_inliner import (
    TentativeNativeInlinerAdapter,
    tentative_native_inliner_barriers,
)

__all__ = [
    "TentativeNativeInlinerAdapter",
    "tentative_native_inliner_barriers",
]
