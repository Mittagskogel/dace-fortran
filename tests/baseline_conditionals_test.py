"""Baseline HLFIR coverage: IF/ELSE, ``IF (cond) CYCLE`` inside a DO."""

from pathlib import Path

import numpy as np

from tests._util import build_sdfg, f2py_compile


def _build(src: str, tmp: Path, name: str):
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


def test_empty_if_else(tmp_path):
    src = """
subroutine pick(d, flag)
  implicit none
  logical, intent(in)    :: flag
  real(8), intent(inout) :: d(2)
  if (flag) then
    d(1) = 7.0d0
  else
    d(1) = -7.0d0
  end if
end subroutine pick
"""
    mod = f2py_compile(src, tmp_path / "ref", "pick")
    sdfg = _build(src, tmp_path / "sdfg", name="pick")

    d_ref = np.zeros(2, order="F")
    mod.pick(d_ref, True)
    d_sdfg = np.zeros(2, dtype=np.float64)
    # ``flag`` feeds a branch condition, so the classifier promotes it to an SDFG symbol.
    sdfg(d=d_sdfg, flag=True)
    np.testing.assert_allclose(d_sdfg, d_ref)


def test_cond_array(tmp_path):
    """IF guard on a derived boolean expression (``s + 5.5 > 5.0``) drives an array write."""
    src = """
subroutine cond_arr(d)
  implicit none
  real(4), intent(inout) :: d(5, 5)
  real(4) :: s
  s = d(2, 1) + 1.0
  if (s + 5.5 > 5.0) then
    d(2, 1) = 11.0
  else
    d(2, 1) = 12.0
  end if
end subroutine cond_arr
"""
    mod = f2py_compile(src, tmp_path / "ref", "cond_arr")
    sdfg = _build(src, tmp_path / "sdfg", name="cond_arr")

    d_ref = np.full((5, 5), 42.0, order="F", dtype=np.float32)
    mod.cond_arr(d_ref)
    d_sdfg = np.full((5, 5), 42.0, dtype=np.float32, order="F")
    sdfg(d=d_sdfg)
    np.testing.assert_allclose(d_sdfg, d_ref)


def test_if_cycle(tmp_path):
    """``IF (i == 2) CYCLE`` inside a DO  --  branches over a write."""
    src = """
subroutine ifcyc(d)
  implicit none
  real(8), intent(inout) :: d(4)
  integer :: i
  do i = 1, 4
    if (i == 2) cycle
    d(i) = 5.5d0
  end do
end subroutine ifcyc
"""
    mod = f2py_compile(src, tmp_path / "ref", "ifcyc")
    sdfg = _build(src, tmp_path / "sdfg", name="ifcyc")

    d_ref = np.full(4, 42.0, order="F")
    mod.ifcyc(d_ref)
    d_sdfg = np.full(4, 42.0, dtype=np.float64)
    sdfg(d=d_sdfg)
    np.testing.assert_allclose(d_sdfg, d_ref)


def test_cycle_before_a_loop_bounded_by_an_array_element(tmp_path):
    """``IF (...) CYCLE`` ahead of ``DO ih = 1, nh(nt)`` (QE h_psi's ``add_vuspsir_gamma``): the CYCLE turns the
    enclosing loop into an scf.while, whose DO bounds must keep their subscript instead of the bare ``nh``."""
    src = """
SUBROUTINE cycbound(nh, ityp, mb, nsp, nat, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nsp, nat
  INTEGER, INTENT(IN) :: nh(nsp), ityp(nat), mb(nat)
  INTEGER, INTENT(OUT) :: out
  INTEGER :: ih, nt, ia
  out = 0
  DO nt = 1, nsp
    DO ia = 1, nat
      IF (ityp(ia) == nt) THEN
        IF (mb(ia) == 0) CYCLE
        DO ih = 1, nh(nt)
          out = out + ih * ia
        END DO
      END IF
    END DO
  END DO
END SUBROUTINE
"""
    mod = f2py_compile(src, tmp_path / "ref", "cycbound")
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cycbound").build()  # the default pipeline, as h_psi builds

    nh = np.array([3, 5], dtype=np.int32)
    ityp = np.array([1, 2, 2, 1], dtype=np.int32)
    mb = np.array([1, 0, 1, 1], dtype=np.int32)
    want = mod.cycbound(nh, ityp, mb)
    out = np.zeros(1, dtype=np.int32)
    sdfg(nh=nh, ityp=ityp, mb=mb, nsp=np.int32(2), nat=np.int32(4), out=out)
    assert out[0] == want


def test_cycle_before_an_indirect_scalar_read(tmp_path):
    """``IF (idx(mc) < 1) CYCLE`` then ``i1 = in1(idx(mc))`` (QE h_psi's ``sticks_dist_new``): the nested index read
    must keep its subscript instead of assigning the whole ``in1`` to the scalar."""
    src = """
SUBROUTINE cycind(idx, in1, yc, nct, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nct, n
  INTEGER, INTENT(IN) :: idx(nct), in1(n)
  INTEGER, INTENT(INOUT) :: yc(n)
  INTEGER :: mc, i1
  DO mc = 1, nct
    IF (idx(mc) < 1) CYCLE
    i1 = in1(idx(mc))
    yc(i1) = yc(i1) + 1
  END DO
END SUBROUTINE
"""
    mod = f2py_compile(src, tmp_path / "ref", "cycind")
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cycind").build()
    idx = np.array([2, 0, 3, 2], dtype=np.int32)
    in1 = np.array([3, 1, 2], dtype=np.int32)
    want = np.zeros(3, dtype=np.int32)
    mod.cycind(idx, in1, want)
    got = np.zeros(3, dtype=np.int32)
    sdfg(idx=idx, in1=in1, yc=got, nct=np.int32(4), n=np.int32(3))
    np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
