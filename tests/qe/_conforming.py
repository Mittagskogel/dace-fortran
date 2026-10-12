# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Source rewrites that make QE checkpoints compile without ``-fallow-argument-mismatch``.

QE passes COMPLEX actuals into REAL dummies of twice the length. Each rewrite turns one such site into the same
arithmetic on the declared types.
"""

#: ``(old, new, occurrences)``. ``qvan2`` fills ``qg(1 | 2, ig)``, the real and imaginary part of a COMPLEX ``qg(ig)``.
QVAN2_REWRITES: tuple[tuple[str, str, int], ...] = (
    ("  REAL(KIND = dp), INTENT(OUT) :: qg(2, ngy)", "  COMPLEX(KIND = dp), INTENT(OUT) :: qg(ngy)", 1),
    (
        "qg(ind, ig) = qg(ind, ig) + sig * ylmk0(ig, lp) * work",
        "IF (ind == 1) THEN\n qg(ig) = qg(ig) + sig * ylmk0(ig, lp) * work\n ELSE\n"
        " qg(ig) = qg(ig) + (0.0_dp, 1.0_dp) * (sig * ylmk0(ig, lp) * work)\n END IF",
        1,
    ),
)


def apply_rewrites(source: str, rewrites: tuple[tuple[str, str, int], ...]) -> str:
    """Apply ``rewrites``; each must match exactly ``occurrences`` times, or the checkpoint changed under them."""
    for old, new, occurrences in rewrites:
        if source.count(old) != occurrences:
            raise ValueError(f"rewrite expects {occurrences} of {old!r}, found {source.count(old)}")
        source = source.replace(old, new)
    return source
