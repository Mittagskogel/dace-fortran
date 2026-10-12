# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``SUM`` of a complex elemental over a section whose extent is read from data (QE h_psi's ``add_vuspsir_gamma``:
``SUM(betasave(ir, 1 : nh(nt)) * CMPLX(w1(1 : nh(nt)), w2(1 : nh(nt))))``). The temporary holding the elemental is
complex, and its extent ``nh(nt)`` becomes a symbol."""

import numpy as np

from tests._util import build_sdfg, f2py_compile

_SRC = """
SUBROUTINE box_sum(betasave, w1, w2, nh, out, nbox, nhm, nsp)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nbox, nhm, nsp
  INTEGER, INTENT(IN) :: nh(nsp)
  DOUBLE PRECISION, INTENT(IN) :: betasave(nbox, nhm), w1(nhm), w2(nhm)
  COMPLEX(8), INTENT(OUT) :: out(nbox, nsp)
  INTEGER :: ir, nt
  DO nt = 1, nsp
    DO ir = 1, nbox
      out(ir, nt) = SUM(betasave(ir, 1 : nh(nt)) * CMPLX(w1(1 : nh(nt)), w2(1 : nh(nt)), kind = 8))
    END DO
  END DO
END SUBROUTINE
"""


def test_sum_of_a_complex_elemental_over_a_data_extent(tmp_path):
    mod = f2py_compile(_SRC, tmp_path / "ref", "box_sum")
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="box_sum").build()
    rng = np.random.default_rng(0)
    betasave = np.asfortranarray(rng.random((3, 4)))
    w1, w2 = rng.random(4), rng.random(4)
    nh = np.array([4, 2], dtype=np.int32)
    want = mod.box_sum(betasave, w1, w2, nh, 3, 4, 2)
    got = np.zeros((3, 2), dtype=np.complex128, order="F")
    sdfg(betasave=betasave, w1=w1, w2=w2, nh=nh, out=got, nbox=np.int32(3), nhm=np.int32(4), nsp=np.int32(2))
    np.testing.assert_allclose(got, want, rtol=1e-14)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        test_sum_of_a_complex_elemental_over_a_data_extent(Path(tmp))
