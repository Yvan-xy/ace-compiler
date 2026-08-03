"""Side-effect-free generic AIR analyses."""

from .call_graph import CallEdge, CallGraph, CallGraphAnalysis
from .function_reachability import (
    FunctionReachability,
    FunctionReachabilityAnalysis,
)

__all__ = [
    "CallEdge",
    "CallGraph",
    "CallGraphAnalysis",
    "FunctionReachability",
    "FunctionReachabilityAnalysis",
]
