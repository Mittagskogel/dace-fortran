"""MATMUL over array sections indexed through data, into section destinations, and inside an expression.

The OMEN self-energy kernel (``Sigma(k, e, a, :, :) = Sigma(k, e, a, :, :) + matmul(dHG, dHD)`` with
``dHG = matmul(G(k, e - w, neigh_idx(a, b) + 1, :, :), dH(a, b, i, :, :))``) needs all of these at once.
"""

import numpy as np
from dace.libraries.blas.nodes.matmul import MatMul

from tests._util import build_sdfg

GATHER = """
module m
contains
  subroutine gather(idx, g, h, out, n, m)
    integer, intent(in) :: n, m
    integer, intent(in) :: idx(m)
    real(8), intent(in) :: g(m, n, n), h(n, n)
    real(8), intent(inout) :: out(m, n, n)
    integer :: a
    do a = 1, m
      out(a, :, :) = matmul(g(idx(a) + 1, :, :), h)
    end do
  end subroutine
end module
"""


def matmul_edges(sdfg):
    """The ``(input edges, output edges)`` of every MatMul node in ``sdfg``."""
    ins, outs = [], []
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, MatMul):
            ins += state.in_edges(node)
            outs += state.out_edges(node)
    return ins, outs


def run_gather(sdfg):
    n, m = 3, 4
    rng = np.random.default_rng(0)
    idx = np.asfortranarray(rng.permutation(m).astype(np.int32))
    g, h = np.asfortranarray(rng.random((m, n, n))), np.asfortranarray(rng.random((n, n)))
    out = np.zeros((m, n, n), order="F")
    sdfg(idx=idx, g=g, h=h, out=out, n=n, m=m)
    np.testing.assert_allclose(out, np.stack([g[idx[a]] @ h for a in range(m)]))


def test_matmul_of_a_section_indexed_through_an_array(tmp_path):
    """``g(idx(a) + 1, :, :)``: the index is read into a symbol before the call, since a memlet subset names symbols
    only; the call's input subset then reads ``idx`` nowhere."""
    sdfg = build_sdfg(GATHER, tmp_path / "sdfg", name="gather", entry="m::gather").build()
    ins, outs = matmul_edges(sdfg)
    assert ins and outs
    for e in ins:
        assert not {str(s) for s in e.data.subset.free_symbols} & set(sdfg.arrays), e.data.subset
    run_gather(sdfg)


def test_matmul_of_a_section_indexed_through_a_scalar(tmp_path):
    """``k = idx(a) + 1; ... g(k, :, :)``: the index is a symbol, never the data scalar ``k``."""
    src = GATHER.replace("integer :: a\n", "integer :: a, k\n").replace(
        "      out(a, :, :) = matmul(g(idx(a) + 1, :, :), h)\n",
        "      k = idx(a) + 1\n      out(a, :, :) = matmul(g(k, :, :), h)\n",
    )
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="gather_scalar", entry="m::gather").build()
    ins, _ = matmul_edges(sdfg)
    for e in ins:
        assert not {str(s) for s in e.data.subset.free_symbols} & set(sdfg.arrays), e.data.subset
    run_gather(sdfg)


def test_matmul_into_a_section_destination(tmp_path):
    """``out(a, :, :) = matmul(...)``: the destination is a rank-3 slice, not an element whose (lb, ub, stride) triples
    read as seven dimensions."""
    src = GATHER.replace("matmul(g(idx(a) + 1, :, :), h)", "matmul(g(a, :, :), h)")
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="section_dest", entry="m::gather").build()
    _, outs = matmul_edges(sdfg)
    assert outs
    for e in outs:
        assert e.data.subset.dims() == 3, e.data.subset
    n, m = 3, 4
    rng = np.random.default_rng(1)
    g, h = np.asfortranarray(rng.random((m, n, n))), np.asfortranarray(rng.random((n, n)))
    out = np.zeros((m, n, n), order="F")
    sdfg(idx=np.zeros(m, np.int32), g=g, h=h, out=out, n=n, m=m)
    np.testing.assert_allclose(out, np.stack([g[a] @ h for a in range(m)]))


def test_complex_matmul_of_dynamic_extent_inside_an_expression(tmp_path):
    """``s = s + matmul(x, y)`` with runtime extents: the result temporary takes its extents from the operands
    (rows of ``x``, columns of ``y``) and keeps the complex element type."""
    src = """
module m
contains
  subroutine acc(x, y, s, n, p)
    integer, intent(in) :: n, p
    complex(8), intent(in) :: x(n, p), y(p, n)
    complex(8), intent(inout) :: s(n, n)
    s = s + matmul(x, y)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="acc", entry="m::acc").build()
    temps = {name: desc for name, desc in sdfg.arrays.items() if name.startswith("_libtmp_")}
    assert temps, sorted(sdfg.arrays)
    for desc in temps.values():
        assert desc.dtype.type == np.complex128
        assert [str(e) for e in desc.shape] == [str(e) for e in sdfg.arrays["s"].shape]
    n, p = 3, 5
    rng = np.random.default_rng(2)
    x = np.asfortranarray(rng.random((n, p)) + 1j * rng.random((n, p)))
    y = np.asfortranarray(rng.random((p, n)) + 1j * rng.random((p, n)))
    s = np.asfortranarray(rng.random((n, n)) + 1j * rng.random((n, n)))
    expected = s + x @ y
    sdfg(x=x, y=y, s=s, n=n, p=p)
    np.testing.assert_allclose(s, expected)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
