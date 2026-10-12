# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The fused ICON AES graupel through the parallelization pipeline: ``samples/graupel.py``."""

import pytest

from samples import graupel

pytestmark = pytest.mark.e2e


def test_graupel_pipeline_numerical_e2e(tmp_path, e2e_cpu_args):
    """The column loop maps, the pipeline is bit-exact, and the result matches gfortran."""
    graupel.run(tmp_path)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        graupel.run(Path(tmp))
