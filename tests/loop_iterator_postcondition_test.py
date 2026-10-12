"""The builder's post-condition: no interstate edge inside a ``LoopRegion`` assigns that region's iterator."""

import pytest
from dace import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion

from dace_fortran.builder import reject_loop_iterator_assignments
from tests._util import build_sdfg


def counted_loop(sdfg: SDFG) -> LoopRegion:
    """``for i in range(10)`` holding two states joined by an assignment-free edge."""
    loop = LoopRegion("loop", "i < 10", "i", "i = 0", "i = i + 1")
    sdfg.add_node(loop, is_start_block=True)
    first = loop.add_state("first", is_start_block=True)
    loop.add_edge(first, loop.add_state("second"), InterstateEdge())
    return loop


def test_a_clean_loop_passes():
    sdfg = SDFG("clean")
    sdfg.add_symbol("i", int)
    counted_loop(sdfg)
    reject_loop_iterator_assignments(sdfg)


def test_a_body_edge_assigning_the_iterator_is_refused():
    sdfg = SDFG("body_edge")
    sdfg.add_symbol("i", int)
    loop = counted_loop(sdfg)
    loop.edges()[0].data.assignments["i"] = "i // 2"
    with pytest.raises(RuntimeError, match="body_edge/loop: first -> second"):
        reject_loop_iterator_assignments(sdfg)


def test_an_assignment_in_a_branch_inside_the_loop_is_refused():
    """The edge sits in a region nested in the loop, not in the loop itself."""
    sdfg = SDFG("in_branch")
    sdfg.add_symbol("i", int)
    loop = counted_loop(sdfg)
    branch = ConditionalBlock("branch")
    loop.add_edge(loop.nodes()[1], branch, InterstateEdge())
    body = ControlFlowRegion("then", sdfg=sdfg)
    branch.add_branch(None, body)
    body.add_edge(body.add_state("a", is_start_block=True), body.add_state("b"), InterstateEdge(assignments={"i": "0"}))
    with pytest.raises(RuntimeError, match="in_branch/loop: a -> b"):
        reject_loop_iterator_assignments(sdfg)


def test_an_assignment_after_the_loop_passes():
    """Outside its loop the iterator is an ordinary symbol."""
    sdfg = SDFG("after")
    sdfg.add_symbol("i", int)
    loop = counted_loop(sdfg)
    sdfg.add_edge(loop, sdfg.add_state("after"), InterstateEdge(assignments={"i": "3"}))
    reject_loop_iterator_assignments(sdfg)


@pytest.mark.parametrize("body", ["b(i) = a(i)", "if (a(i) > 100.0d0) exit\n      b(i) = a(i)"])
def test_built_do_loops_satisfy_it(tmp_path, body):
    """``build`` runs the check, so a counted DO and one with an EXIT build through it."""
    src = f"""
subroutine copy(a, b, n)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: b(n)
  integer :: i
  do i = 1, n
      {body}
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="copy").build()
    assert any(isinstance(r, LoopRegion) for r in sdfg.all_control_flow_regions())
