"""Domain-neutral, deterministic AIR call-graph analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..framework.api import (
    AIRObjectId,
    AnalysisPass,
    AnalysisResult,
    IRUnitRef,
    analysis_failure,
    analysis_success,
)
from ..framework.diagnostics import error


@dataclass(frozen=True)
class CallEdge:
    caller: IRUnitRef
    callee: IRUnitRef
    call: AIRObjectId
    indirect: bool = False


@dataclass(frozen=True)
class CallGraph:
    functions: Tuple[IRUnitRef, ...]
    edges: Tuple[CallEdge, ...]
    program_roots: Tuple[IRUnitRef, ...] = ()
    exported_roots: Tuple[IRUnitRef, ...] = ()
    address_taken_roots: Tuple[IRUnitRef, ...] = ()
    entry_descriptor_roots: Tuple[IRUnitRef, ...] = ()
    recursive_components: Tuple[Tuple[IRUnitRef, ...], ...] = ()

    def callees(self, function: IRUnitRef) -> Tuple[IRUnitRef, ...]:
        return tuple(edge.callee for edge in self.edges if edge.caller == function)


def _walk_node(node):
    yield node
    for child in node.children:
        if child.id.kind == "block":
            yield from _walk_block(child)
        else:
            yield from _walk_node(child)


def _walk_block(block):
    for statement in block.statements:
        yield from _walk_node(statement.node)


def _recursive_components(functions, edges):
    adjacency = {function: [] for function in functions}
    for edge in edges:
        if edge.callee not in adjacency[edge.caller]:
            adjacency[edge.caller].append(edge.callee)
    indices = {}
    lowlinks = {}
    stack = []
    on_stack = set()
    components = []
    next_index = 0

    def visit(function):
        nonlocal next_index
        indices[function] = lowlinks[function] = next_index
        next_index += 1
        stack.append(function)
        on_stack.add(function)
        for callee in adjacency[function]:
            if callee not in indices:
                visit(callee)
                lowlinks[function] = min(lowlinks[function], lowlinks[callee])
            elif callee in on_stack:
                lowlinks[function] = min(lowlinks[function], indices[callee])
        if lowlinks[function] == indices[function]:
            component = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == function:
                    break
            component.sort(key=lambda item: item.native_id)
            if len(component) > 1 or function in adjacency[function]:
                components.append(tuple(component))

    for function in functions:
        if function not in indices:
            visit(function)
    components.sort(key=lambda component: component[0].native_id)
    return tuple(components)


@dataclass(frozen=True)
class CallGraphAnalysis(AnalysisPass[CallGraph]):
    conservative_indirect_calls: bool = True
    pass_id = "air.analysis.call-graph"

    def run(self, read_only_unit, context) -> AnalysisResult[CallGraph]:
        if not hasattr(read_only_unit, "functions"):
            return analysis_failure(
                error(
                    "air.call-graph.module-required",
                    "call-graph analysis requires a module unit",
                )
            )
        functions = tuple(
            IRUnitRef("function", read_only_unit.unit_ref.module_id, function.native_id)
            for function in read_only_unit.functions
        )
        by_native_id = {function.native_id: function for function in functions}
        edges = []
        for function_view, caller in zip(read_only_unit.functions, functions):
            if function_view.entry_block is None:
                continue
            for node in _walk_block(function_view.entry_block):
                if node.call_target is not None:
                    callee_id = node.call_target.owning_function.native_id
                    callee = by_native_id.get(callee_id)
                    if callee is None:
                        return analysis_failure(
                            error(
                                "air.call-graph.unresolved-direct-call",
                                f"direct call in function {caller.native_id} targets undefined function {callee_id}",
                            )
                        )
                    edges.append(CallEdge(caller, callee, node.id))
                elif node.indirect_call:
                    if not self.conservative_indirect_calls:
                        return analysis_failure(
                            error(
                                "air.call-graph.unsafe-indirect-policy",
                                "indirect calls require conservative target retention",
                            )
                        )
                    edges.extend(
                        CallEdge(caller, callee, node.id, True)
                        for callee in functions
                    )

        program_roots = tuple(
            by_native_id[entry.owning_function.native_id]
            for entry in read_only_unit.entries
            if entry.program_entry
            and entry.owning_function.native_id in by_native_id
        )
        exported_roots = tuple(
            by_native_id[entry.owning_function.native_id]
            for entry in read_only_unit.entries
            if getattr(entry, "exported", False)
            and entry.owning_function.native_id in by_native_id
        )
        address_taken = []
        entry_descriptors = []
        for constant in read_only_unit.constants:
            entry = constant.referenced_entry
            if entry is None or entry.owning_function.native_id not in by_native_id:
                continue
            target = by_native_id[entry.owning_function.native_id]
            kind = constant.kind.upper()
            if kind == "ENTRY_PTR":
                address_taken.append(target)
            elif kind == "ENTRY_FUNC_DESC":
                entry_descriptors.append(target)

        def unique(values):
            return tuple(dict.fromkeys(values))

        edges_tuple = tuple(edges)
        recursive_components = _recursive_components(functions, edges_tuple)
        return analysis_success(
            CallGraph(
                functions,
                edges_tuple,
                unique(program_roots),
                unique(exported_roots),
                unique(address_taken),
                unique(entry_descriptors),
                recursive_components,
            ),
            metrics={
                "functions": len(functions),
                "edges": len(edges_tuple),
                "recursive_components": len(recursive_components),
            },
        )
