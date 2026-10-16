# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fully unroll short constant-trip loops, so small recurrences become straight-line code instead of maps."""

from typing import Any, cast

from dace import SDFG, symbolic
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.passes.analysis import loop_analysis

DEFAULT_UNROLL_LIMIT = 8


def trip_count(loop: LoopRegion, sdfg: SDFG) -> int | None:
    """Iteration count of an ascending constant-trip loop, else None."""
    # SymbolicType's stubs omit sympy arithmetic.
    start = cast(Any, loop_analysis.get_init_assignment(loop))
    end = cast(Any, loop_analysis.get_loop_end(loop))
    stride = cast(Any, loop_analysis.get_loop_stride(loop))
    if start is None or end is None or stride is None:
        return None
    if symbolic.issymbolic(stride, sdfg.constants) or symbolic.issymbolic(end - start, sdfg.constants):
        return None
    step = int(symbolic.evaluate(stride, sdfg.constants))
    extent = int(symbolic.evaluate(end - start + 1, sdfg.constants))
    return len(range(0, extent, step)) if step > 0 and extent > 0 else None


def depth(loop: LoopRegion) -> int:
    level, graph = 0, loop.parent_graph
    while graph is not None and not isinstance(graph, SDFG):
        level, graph = level + 1, graph.parent_graph
    return level


def fuse_states_in(sdfg: SDFG, region: ControlFlowRegion) -> None:
    """StateFusionExtended on ``region``'s subtree only; ``apply_transformations`` cannot be scoped to it."""
    fused = True
    while fused:
        fused = False
        for cfr in [region, *region.all_control_flow_regions(recursive=True)]:
            for edge in cfr.edges():
                u, v = edge.src, edge.dst
                if (
                    isinstance(u, SDFGState)
                    and isinstance(v, SDFGState)
                    and StateFusionExtended.can_be_applied_to(sdfg, first_state=u, second_state=v)
                ):
                    StateFusionExtended.apply_to(sdfg, first_state=u, second_state=v, verify=False, annotate=False)
                    fused = True
                    break
            if fused:
                break


class ShortLoopUnroll(ppl.Pass):
    """Unroll every constant-trip loop of at most ``unroll_limit`` iterations, deepest first.

    Each unrolled region is fused right away, so an enclosing loop copies a compact body.
    """

    def __init__(self, unroll_limit: int = DEFAULT_UNROLL_LIMIT):
        super().__init__()
        self.unroll_limit = unroll_limit

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> int | None:
        unrolled = 0
        changed = self.unroll_limit > 0
        while changed:
            changed = False
            loops = [r for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable]
            for loop in sorted(loops, key=depth, reverse=True):
                trip = trip_count(loop, sdfg)
                if (
                    trip is None
                    or trip > self.unroll_limit
                    or not LoopUnroll.can_be_applied_to(loop.sdfg, loop=cast(Any, loop))
                ):
                    continue
                parent = loop.parent_graph
                LoopUnroll().apply_to(loop.sdfg, loop=cast(Any, loop), annotate=False, verify=False)
                fuse_states_in(sdfg, parent)
                unrolled += 1
                changed = True
                break
        if not unrolled:
            return None
        # Once at the end: per-apply propagation dominates the cost of unrolling deep nests.
        propagate_memlets_sdfg(sdfg)
        return unrolled
