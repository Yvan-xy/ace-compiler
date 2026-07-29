"""Temporary pass facade for binding-side generated-helper inlining.

This baseline-kernel E2E bridge is not the independent Python AIR algorithm
planned for M13. M13 replaces it, and M15 removes the tentative native path.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class PassResult:
    success: bool
    changed: bool
    calls_inlined: int
    helpers_removed: int
    diagnostics: Tuple[str, ...]


class FunctionInlinerPass:
    """Temporarily delegate the generated leaf-helper subset to the binding."""

    _SELECTION = {
        "attribute": "ace.vector_kernel.generated_call",
        "kind": "same-module-generated-leaf",
    }

    @classmethod
    def run(
        cls,
        glob_scope,
        policy: str = "always",
        predicate: Optional[Callable[[Mapping[str, str]], bool]] = None,
    ) -> PassResult:
        if policy not in ("always", "never"):
            return PassResult(
                False,
                False,
                0,
                0,
                (f"unsupported function-inliner policy: {policy}",),
            )
        if policy == "never":
            return PassResult(True, False, 0, 0, ())
        if glob_scope is None:
            return PassResult(False, False, 0, 0, ("glob_scope is None",))
        if predicate is not None:
            try:
                selected = bool(predicate(dict(cls._SELECTION)))
            except Exception as error:
                return PassResult(
                    False,
                    False,
                    0,
                    0,
                    (f"function-inliner predicate failed: {error}",),
                )
            if not selected:
                return PassResult(True, False, 0, 0, ())
        if not hasattr(glob_scope, "inline_generated_vector_kernel_helpers"):
            return PassResult(
                False,
                False,
                0,
                0,
                ("glob_scope has no generated-helper inlining binding",),
            )

        try:
            raw = glob_scope.inline_generated_vector_kernel_helpers()
        except Exception as error:
            return PassResult(False, False, 0, 0, (str(error),))

        diagnostic = str(raw.get("diagnostic", ""))
        return PassResult(
            bool(raw.get("success", False)),
            bool(raw.get("changed", False)),
            int(raw.get("calls_inlined", 0)),
            int(raw.get("helpers_removed", 0)),
            (diagnostic,) if diagnostic else (),
        )
