# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fortran locals named like C++ keywords compile: the generated code would otherwise declare ``int new;``."""

import numpy as np

from tests._util import build_sdfg


def test_locals_named_like_cpp_keywords(tmp_path):
    """``new`` and ``class`` are ordinary Fortran names (QE h_psi declares ``INTEGER :: new``)."""
    src = """
MODULE cpp_keyword_mod
CONTAINS
SUBROUTINE cpp_keyword_locals(a, out, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: out
  INTEGER :: new, i
  REAL(8) :: class
  new = 0
  class = 0.0_8
  DO i = 1, n
    IF (a(i) > 0.0_8) new = new + 1
    class = class + a(i)
  END DO
  out = class * new
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cpp_keyword_locals", entry="cpp_keyword_mod::cpp_keyword_locals")
    sdfg = sdfg.build()
    assert not {"new", "class"} & (set(sdfg.arrays) | set(sdfg.symbols))
    a = np.array([1.0, -2.0, 3.0, 4.0], order="F")
    out = np.zeros(1)
    sdfg(a=a, out=out, n=np.int32(4))
    assert out[0] == (1.0 - 2.0 + 3.0 + 4.0) * 3


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        test_locals_named_like_cpp_keywords(Path(tmp))
