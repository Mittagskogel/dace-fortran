# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Move View reads off interstate edges into dataflow.

A View exists in generated code only where an access node and its source link define it, in that node's scope.
An interstate assignment that reads one (``iind = ind(l)`` where ``ind`` is a dummy aliasing an actual of another
rank) therefore reads an undeclared or out-of-scope pointer. Each such read is staged by a tasklet in a state of its
own, from an access node linked to the View's source, into a scalar the assignment then reads.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import dace
from dace import InterstateEdge
from dace.memlet import Memlet

from dace_fortran.builder.emit_tasklet import ensure_view_read_link

if TYPE_CHECKING:
    from dace.sdfg import SDFG

    from dace_fortran.builder import SDFGBuilder


def _subscript_end(text: str, open_bracket: int) -> int:
    """Index just past the ``]`` matching the ``[`` at ``open_bracket``."""
    depth = 0
    for i in range(open_bracket, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError(f"unbalanced subscript in {text!r}")


def stage_view_reads_off_interstate_edges(builder: SDFGBuilder, sdfg: SDFG) -> None:
    views = sorted((name for name, desc in sdfg.arrays.items() if isinstance(desc, dace.data.View)), key=len)
    if not views:
        return
    reference = re.compile(r"\b(" + "|".join(re.escape(v) for v in reversed(views)) + r")\[")
    for region in list(sdfg.all_control_flow_regions()):
        for edge in list(region.edges()):
            if not any(reference.search(rhs) for rhs in edge.data.assignments.values()):
                continue
            state = region.add_state(f"view_reads_{edge.src.label}")
            staged: dict[str, str] = {}

            def stage(match: re.Match[str], text: str) -> tuple[str, int]:
                view = match.group(1)
                end = _subscript_end(text, match.end() - 1)
                element = text[match.start() : end]
                if element not in staged:
                    name, _ = sdfg.add_scalar(
                        f"__{view}_read", sdfg.arrays[view].dtype, transient=True, find_new_name=True
                    )
                    view_node = state.add_access(view)
                    ensure_view_read_link(builder, state, view_node, view)
                    tasklet = state.add_tasklet(f"read_{name}", {"_in"}, {"_out"}, "_out = _in")
                    state.add_edge(view_node, None, tasklet, "_in", Memlet(element))
                    state.add_edge(tasklet, "_out", state.add_write(name), None, Memlet(name))
                    staged[element] = name
                return staged[element], end

            assignments = {}
            for target, rhs in edge.data.assignments.items():
                pieces, pos = [], 0
                while match := reference.search(rhs, pos):
                    pieces.append(rhs[pos : match.start()])
                    name, pos = stage(match, rhs)
                    pieces.append(name)
                assignments[target] = "".join(pieces) + rhs[pos:]
            region.remove_edge(edge)
            region.add_edge(edge.src, state, InterstateEdge(condition=edge.data.condition))
            region.add_edge(state, edge.dst, InterstateEdge(assignments=assignments))
