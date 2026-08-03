"""Explicitly temporary adapter for the binding-side native inliner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from ..framework.api import freeze_metrics
from ..framework.diagnostics import error
from ..framework.pipeline_hooks import NativeBarrierOutcome, NativeBarrierStep


_SUPPORTED_PLAN_KINDS = frozenset(
    ("auto", "baseline-gemm", "baseline-conv", "fast-gemm", "fast-conv")
)


def requires_tentative_native_inliner(config) -> bool:
    return (
        config is not None
        and config.kernel_impl == "dsl"
        and config.plan_kind in _SUPPORTED_PLAN_KINDS
    )


def tentative_native_inliner_barriers(config):
    return (TentativeNativeInlinerAdapter(),) if requires_tentative_native_inliner(config) else ()


@dataclass(frozen=True)
class TentativeNativeInlinerAdapter(NativeBarrierStep):
    policy: str = "always"
    predicate: Optional[Callable[[Mapping[str, str]], bool]] = None
    pass_id = "air.transition.tentative-native-inliner"

    _SELECTION = {
        "attribute": "ace.vector_kernel.generated_call",
        "kind": "same-module-generated-leaf",
    }

    def run(self, glob_scope) -> NativeBarrierOutcome:
        if self.policy not in ("always", "never"):
            return NativeBarrierOutcome(
                False,
                False,
                (
                    error(
                        "air.native-inliner.unsupported-policy",
                        f"unsupported tentative native-inliner policy: {self.policy}",
                    ),
                ),
            )
        if self.policy == "never":
            return NativeBarrierOutcome(True, False)
        if glob_scope is None:
            return NativeBarrierOutcome(
                False,
                False,
                (error("air.native-inliner.missing-scope", "glob_scope is None"),),
            )
        if self.predicate is not None:
            try:
                selected = bool(self.predicate(dict(self._SELECTION)))
            except Exception as exc:
                return NativeBarrierOutcome(
                    False,
                    False,
                    (
                        error(
                            "air.native-inliner.predicate-exception",
                            f"tentative native-inliner predicate raised {type(exc).__name__}: {exc}",
                        ),
                    ),
                )
            if not selected:
                return NativeBarrierOutcome(True, False)
        if not hasattr(glob_scope, "inline_generated_vector_kernel_helpers"):
            return NativeBarrierOutcome(
                False,
                False,
                (
                    error(
                        "air.native-inliner.missing-binding",
                        "glob_scope has no generated-helper inlining binding",
                    ),
                ),
            )
        try:
            raw = glob_scope.inline_generated_vector_kernel_helpers()
        except Exception as exc:
            return NativeBarrierOutcome(
                False,
                False,
                (
                    error(
                        "air.native-inliner.exception",
                        f"native inliner raised {type(exc).__name__}: {exc}",
                    ),
                ),
            )
        diagnostic = str(raw.get("diagnostic", ""))
        success = bool(raw.get("success", False))
        return NativeBarrierOutcome(
            success,
            bool(raw.get("changed", False)) if success else False,
            ()
            if success or not diagnostic
            else (error("air.native-inliner.failed", diagnostic),),
            freeze_metrics(
                {
                    "calls_inlined": int(raw.get("calls_inlined", 0)),
                    "helpers_removed": int(raw.get("helpers_removed", 0)),
                }
            ),
        )
