# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON ``velocity_tendencies`` through the parallelization pipeline: ``samples/velocity_tendencies.py``."""

import pytest

from samples import velocity_tendencies
from tests.icon.atmosphere._atmo_harness import VELOCITY_TU_VARIANTS

pytestmark = pytest.mark.e2e


@pytest.mark.xdist_group("atmo_velocity_fparser")
@pytest.mark.parametrize(
    "tu_name,loop_exchange",
    VELOCITY_TU_VARIANTS,
    ids=[("loopexch" if le else "noloopexch") for _, le in VELOCITY_TU_VARIANTS],
)
def test_velocity_tendencies_pipeline_numerical_e2e(monkeypatch, tu_name, loop_exchange):
    """Unoptimized and optimized bindings both match gfortran bit for bit; every loop maps."""
    monkeypatch.delenv("OCEAN_E2E_SDFG_HOOK", raising=False)  # restored after the test, whatever the legs set
    velocity_tendencies.run_layout(tu_name, loop_exchange, optimized=False)
    velocity_tendencies.run_layout(tu_name, loop_exchange, optimized=True)


if __name__ == "__main__":
    velocity_tendencies.run()
