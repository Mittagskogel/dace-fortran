# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Monolithic CLOUDSC through the parallelization pipeline: ``samples/cloudsc_monolith.py``."""

import pytest

from samples import cloudsc_monolith

pytestmark = pytest.mark.e2e


def test_cloudsc_pipeline_numerical_e2e(tmp_path, e2e_cpu_args):
    """The column loops map, and the optimized SDFG matches gfortran and the unoptimized SDFG."""
    cloudsc_monolith.run(tmp_path)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        cloudsc_monolith.run(Path(tmp))
