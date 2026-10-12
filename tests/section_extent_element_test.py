# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An actual section bounded by a constant-index element (QE h_psi's ``wave_g2r(orbital(1 : ngk(1), ...))``): the
extent symbol ``__sym_ngk_1`` the callee's ``SIZE`` reads is seeded from ``ngk(1)`` at program entry."""

import numpy as np

from tests._util import build_sdfg

_SRC = """
MODULE secsize_mod
CONTAINS
SUBROUTINE inner(f_in, out)
  IMPLICIT NONE
  COMPLEX(8), INTENT(IN) :: f_in(:, :)
  INTEGER, INTENT(OUT) :: out
  out = SIZE(f_in(:, 1)) * 10 + SIZE(f_in(1, :))
END SUBROUTINE
SUBROUTINE outer(orbital, ngk, out, n, m)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  COMPLEX(8), INTENT(IN) :: orbital(n, m)
  INTEGER, INTENT(IN) :: ngk(2)
  INTEGER, INTENT(OUT) :: out
  CALL inner(orbital(1 : ngk(1), 1 : 2), out)
END SUBROUTINE
END MODULE
"""


def test_section_bounded_by_a_constant_index_element(tmp_path):
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="section_extent_element", entry="secsize_mod::outer").build()
    out = np.zeros(1, np.int32)
    orbital = np.zeros((5, 3), np.complex128, order="F")
    sdfg(orbital=orbital, ngk=np.array([4, 9], np.int32), out=out, n=np.int32(5), m=np.int32(3))
    assert out[0] == 42


_SRC_WRITTEN = """
MODULE secsize_written_mod
CONTAINS
SUBROUTINE inner(f_in, out)
  IMPLICIT NONE
  COMPLEX(8), INTENT(IN) :: f_in(:, :)
  INTEGER, INTENT(OUT) :: out
  out = SIZE(f_in(:, 1))
END SUBROUTINE
SUBROUTINE outer(orbital, ngk, out, n, m)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  COMPLEX(8), INTENT(IN) :: orbital(n, m)
  INTEGER, INTENT(INOUT) :: ngk(2)
  INTEGER, INTENT(OUT) :: out
  ngk(1) = ngk(1) - 1
  CALL inner(orbital(1 : ngk(1), 1 : m), out)
END SUBROUTINE
END MODULE
"""


def test_section_bounded_by_a_written_element(tmp_path):
    """``ngk`` is written before the call (QE h_psi's ``gk_sort`` sets it), so ``ngk(1)`` cannot be a symbol seeded at
    entry: the extent reads the element where the section is formed."""
    sdfg = build_sdfg(
        _SRC_WRITTEN, tmp_path / "sdfg", name="section_extent_written", entry="secsize_written_mod::outer"
    )
    sdfg = sdfg.build()
    out = np.zeros(1, np.int32)
    ngk = np.array([4, 9], np.int32)
    sdfg(orbital=np.zeros((5, 3), np.complex128, order="F"), ngk=ngk, out=out, n=np.int32(5), m=np.int32(3))
    assert out[0] == 3
    assert ngk[0] == 3


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        test_section_bounded_by_a_constant_index_element(Path(tmp) / "const")
        test_section_bounded_by_a_written_element(Path(tmp) / "written")
