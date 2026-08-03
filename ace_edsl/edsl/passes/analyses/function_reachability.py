"""Transitive function reachability over the generic AIR call graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..framework.api import (
    CURRENT_MODULE,
    AnalysisPass,
    AnalysisRequest,
    AnalysisResult,
    IRUnitRef,
    analysis_failure,
    analysis_success,
)
from ..framework.diagnostics import error
from .call_graph import CallGraph, CallGraphAnalysis


@dataclass(frozen=True)
class FunctionReachability:
    roots: Tuple[IRUnitRef, ...]
    reachable: Tuple[IRUnitRef, ...]
    unreachable: Tuple[IRUnitRef, ...]

    def contains(self, function: IRUnitRef) -> bool:
        return function in self.reachable


@dataclass(frozen=True)
class FunctionReachabilityAnalysis(AnalysisPass[FunctionReachability]):
    explicit_roots: Tuple[int, ...] = ()
    exported_roots: Tuple[int, ...] = ()
    address_taken_roots: Tuple[int, ...] = ()
    entry_descriptor_roots: Tuple[int, ...] = ()
    include_program_entries: bool = True
    include_exported: bool = True
    include_address_taken: bool = True
    include_entry_descriptors: bool = True
    conservative_indirect_calls: bool = True
    pass_id = "air.analysis.function-reachability"

    @property
    def _call_graph(self):
        return CallGraphAnalysis(self.conservative_indirect_calls)

    def dependencies(self, unit: IRUnitRef):
        return (AnalysisRequest(self._call_graph, CURRENT_MODULE),)

    def run(self, read_only_unit, context) -> AnalysisResult[FunctionReachability]:
        if not hasattr(read_only_unit, "functions"):
            return analysis_failure(
                error(
                    "air.reachability.module-required",
                    "function reachability analysis requires a module unit",
                )
            )
        request = AnalysisRequest(self._call_graph, CURRENT_MODULE)
        graph_result = context.result(request, read_only_unit.unit_ref)
        if not graph_result.success or not isinstance(graph_result.value, CallGraph):
            return analysis_failure(
                error(
                    "air.reachability.call-graph-missing",
                    "function reachability requires a successful call graph",
                )
            )
        graph = graph_result.value
        by_id = {function.native_id: function for function in graph.functions}
        requested_ids = (
            self.explicit_roots,
            self.exported_roots,
            self.address_taken_roots,
            self.entry_descriptor_roots,
        )
        unknown = sorted(
            {native_id for group in requested_ids for native_id in group}
            - by_id.keys()
        )
        if unknown:
            return analysis_failure(
                error(
                    "air.reachability.unknown-root",
                    "unknown function root IDs: " + ", ".join(map(str, unknown)),
                )
            )
        roots = [by_id[native_id] for native_id in self.explicit_roots]
        if self.include_program_entries:
            roots.extend(graph.program_roots)
        if self.include_exported:
            roots.extend(graph.exported_roots)
            roots.extend(by_id[native_id] for native_id in self.exported_roots)
        if self.include_address_taken:
            roots.extend(graph.address_taken_roots)
            roots.extend(by_id[native_id] for native_id in self.address_taken_roots)
        if self.include_entry_descriptors:
            roots.extend(graph.entry_descriptor_roots)
            roots.extend(
                by_id[native_id] for native_id in self.entry_descriptor_roots
            )
        roots_tuple = tuple(dict.fromkeys(roots))

        adjacency = {function: [] for function in graph.functions}
        for edge in graph.edges:
            if edge.callee not in adjacency[edge.caller]:
                adjacency[edge.caller].append(edge.callee)
        visited = set()

        def visit(function):
            if function in visited:
                return
            visited.add(function)
            for callee in adjacency[function]:
                visit(callee)

        for root in roots_tuple:
            visit(root)
        reachable = tuple(function for function in graph.functions if function in visited)
        unreachable = tuple(
            function for function in graph.functions if function not in visited
        )
        return analysis_success(
            FunctionReachability(roots_tuple, reachable, unreachable),
            metrics={
                "roots": len(roots_tuple),
                "reachable": len(reachable),
                "unreachable": len(unreachable),
            },
        )
