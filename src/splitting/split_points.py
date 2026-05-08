"""
Identify candidate split points from an FX graph node list.

A candidate split point is a call_module node where:
  1. It has exactly one tensor input (lies on a single sequential path)
  2. It has a single tensor output
  3. It is on the top level of the model's forward() (target contains no '.')

These are the safest preemption points for the FPP scheduler: each chunk boundary
corresponds to a complete sub-module finishing, giving a clean memory footprint.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.splitting.fx_graph import NodeInfo


@dataclass
class SplitCandidate:
    """One candidate split point in the model's computation graph."""
    node_name:     str
    module_target: str
    input_shape:   Optional[Tuple[int, ...]]
    output_shape:  Optional[Tuple[int, ...]]
    is_top_level:  bool   # target has no '.' → direct attribute of the model


def find_candidates(nodes: List[NodeInfo]) -> List[SplitCandidate]:
    """
    Return call_module nodes with single tensor I/O (candidate split points).
    Top-level candidates (no '.' in target) are the recommended split points.
    """
    candidates: List[SplitCandidate] = []
    for node in nodes:
        if node.op != "call_module":
            continue
        in_shape = node.input_shapes[0] if len(node.input_shapes) == 1 else None
        candidates.append(SplitCandidate(
            node_name=node.name,
            module_target=node.target,
            input_shape=in_shape,
            output_shape=node.output_shape,
            is_top_level="." not in node.target,
        ))
    return candidates


def top_level_sequence(nodes: List[NodeInfo]) -> List[NodeInfo]:
    """
    Return all nodes in forward-execution order for the top-level forward().
    Includes call_module and call_function nodes; excludes placeholder/output.
    """
    return list(nodes)  # already in topological order from trace_and_profile()
