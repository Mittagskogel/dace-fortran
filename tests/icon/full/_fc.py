"""Shared Fortran-compiler discovery + pytest parametrization helper.

Locates gfortran / flang-new / nvfortran (PATH + standard NVHPC prefix) and exposes ``FORTRAN_COMPILERS`` for ``@pytest.mark.parametrize``; the iso_c wrappers + solve_nh patch must parse cleanly under all three ICON-shipped compilers.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from dace_fortran.llvm_toolchain import find_flang, flang_names, llvm_prefix

# LLVM-flang binary names probed in order, for every supported LLVM major; distributions differ on which spelling is canonical.
_FLANG_NAMES = tuple(flang_names())


def _looks_like_llvm_flang(path: str) -> bool:
    """True when ``path --version`` self-identifies as LLVM flang; gates $FC override so a non-flang $FC doesn't hijack the flang slot."""
    try:
        out = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT, timeout=5).decode(errors="replace")
    except (OSError, subprocess.SubprocessError):
        return False
    return "flang version" in out


def _find_flang() -> Path | None:
    """Locate an LLVM-flang binary: ``$FC`` if set and self-identifying as flang, else the first ``_FLANG_NAMES`` hit on PATH."""
    fc = os.environ.get("FC")
    if fc:
        fc_path = shutil.which(fc) or (fc if os.path.isfile(fc) else None)
        if fc_path and _looks_like_llvm_flang(fc_path):
            return Path(fc_path)
    for name in _FLANG_NAMES:
        p = shutil.which(name)
        if p is not None:
            return Path(p)
    return None


def _find_gfortran() -> Path | None:
    p = shutil.which("gfortran")
    return Path(p) if p else None


def discover_fortran_compilers() -> dict:
    """{display_name: path} for gfortran and flang (both on CI); each test compiles against an ICON .mod tree built by the SAME compiler since .mod files aren't cross-compatible.  nvfortran excluded -- multi-GB CI dependency, GPU builds out of scope.

    The flang key is the version-neutral ``"flang"`` whatever binary spelling or LLVM major resolved, so parametrized tests branch on a stable name."""
    out: dict = {}
    gfortran = _find_gfortran()
    if gfortran:
        out["gfortran"] = gfortran
    flang = _find_flang()
    if flang:
        out["flang"] = flang
    return out


def _make_params() -> list:
    """Build the ``pytest.param(...)`` list for the discovered set."""
    found = discover_fortran_compilers()
    if not found:
        # No compiler on the host: keep the gfortran param so its test fails loudly.
        return [pytest.param(("gfortran", "gfortran"), id="gfortran")]
    return [pytest.param((name, str(path)), id=name) for name, path in found.items()]


#: (name, executable_path) tuples for @pytest.mark.parametrize("fc", FORTRAN_COMPILERS); with no compiler found the gfortran param stays and its test fails.
FORTRAN_COMPILERS = _make_params()


def fortran_compiler_flags(fc_name: str) -> list[str]:
    """Per-compiler base flags for wrapper-syntax checks: gfortran needs -ffree-line-length-none (default 132-col limit) for long generated lines; flang-new/nvfortran need nothing (no limit, and flang rejects the gfortran flag)."""
    if fc_name == "gfortran":
        return ["-ffree-line-length-none"]
    return []


def syntax_check_argv(fc_name: str, scratch_dir: Path) -> list[str]:
    """Per-compiler argv tail for a parse+semantic-check pass: gfortran/flang-new use -fsyntax-only; nvfortran has no equivalent, falls back to bare -c (no -o -- it refuses -c -o single.o with multiple sources; caller's cwd=scratch_dir keeps per-source .o files xdist-safe)."""
    del scratch_dir  # reserved for future single-output compilers
    if fc_name == "nvfortran":
        return ["-c"]
    return ["-fsyntax-only"]


def cpp_flag(fc_name: str) -> str:
    """Per-compiler -cpp-equivalent flag: gfortran/flang-new use -cpp, nvfortran uses -Mpreprocess."""
    return "-Mpreprocess" if fc_name == "nvfortran" else "-cpp"


def find_flang_runtime_dir() -> str | None:
    """Directory holding ``libflang_rt.runtime.a`` (flang-21's ``-lflang_rt.runtime`` archive) in the found flang's
    prefix, or None.  Tests linking with flang-new-21 use this to skip cleanly or inject LIBRARY_PATH."""
    flang = find_flang()
    if flang is None:
        return None
    # The resource dir's major and target-triple spelling vary across installs, so glob for them.
    for d in sorted(llvm_prefix(flang).glob("lib/clang/*/lib/*")):
        if (d / "libflang_rt.runtime.a").is_file():
            return str(d)
    return None


def env_with_flang_runtime(fc_name: str) -> dict:
    """Copy of os.environ with LIBRARY_PATH prepended for flang's runtime, when reachable and ``fc_name`` is a flang variant.  No-op for gfortran/nvfortran (they ship their own runtimes)."""
    env = dict(os.environ)
    if "flang" in fc_name:
        rt = find_flang_runtime_dir()
        if rt is not None:
            existing = env.get("LIBRARY_PATH", "")
            env["LIBRARY_PATH"] = f"{rt}:{existing}" if existing else rt
    return env


#: pytest.skip reason for tests needing a full flang link but no runtime found.
FLANG_RT_HINT = (
    "flang-new-21 needs ``libflang_rt.runtime.a`` for a full link; "
    "build it locally with the recipe at the top of the README's "
    "Fortran-compiler matrix section into the flang install's resource directory."
)
