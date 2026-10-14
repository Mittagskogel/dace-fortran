"""Unit coverage for module-global provenance merge (``effective_module_sources``)
and ``FrozenSignature.module_symbol_origins`` JSON round-trip.

E2e ``_QM<mod>E<entity>`` recovery is exercised by ``test_velocity_full_auto_module_e2e``;
this pins explicit-override precedence, the auto-only path, and serialisation stability.
"""

from dace_fortran.bindings.block_builders import _build_symbol_assigns, effective_module_sources
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import OriginalInterface
from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenArgKind, FrozenSignature, ModuleOrigin


def _frozen(origins):
    return FrozenSignature(entry="k", mangled="_QPk", args=(), free_symbols=("nproma",), module_symbol_origins=origins)


def _iface(sources):
    return OriginalInterface(entry="k", args=(), module_symbol_sources=sources)


def test_auto_only_no_handauthored_list():
    """Bridge-auto-detected origin is used when the iface map is empty."""
    fs = _frozen({"nproma": ModuleOrigin("mo_parallel_config", "nproma")})
    merged = effective_module_sources(fs, _iface({}))
    assert merged == {"nproma": ("mo_parallel_config", "nproma")}


def test_explicit_override_wins_on_conflict():
    """A hand-authored entry overrides a (mis-)auto-detected origin."""
    fs = _frozen({"nproma": ModuleOrigin("wrong_mod", "nproma")})
    merged = effective_module_sources(fs, _iface({"nproma": ModuleOrigin("mo_parallel_config", "nproma")}))
    assert merged["nproma"] == ("mo_parallel_config", "nproma")


def test_explicit_supplements_auto():
    """Explicit entries the bridge could not recover are still honoured;
    auto entries the iface omits survive the merge."""
    fs = _frozen({"nproma": ModuleOrigin("mo_parallel_config", "nproma")})
    merged = effective_module_sources(fs, _iface({"nrdmax": ModuleOrigin("mo_vertical_grid", "nrdmax")}))
    assert merged == {
        "nproma": ("mo_parallel_config", "nproma"),
        "nrdmax": ("mo_vertical_grid", "nrdmax"),
    }


def test_missing_origins_attr_degrades_to_explicit():
    """A frozen signature without the new field (old snapshot) falls
    back to the explicit map without raising."""
    fs = FrozenSignature(entry="k", mangled="_QPk", args=())
    merged = effective_module_sources(fs, _iface({"nproma": ModuleOrigin("mo_parallel_config", "nproma")}))
    assert merged == {"nproma": ("mo_parallel_config", "nproma")}


def test_frozen_signature_json_roundtrip(tmp_path):
    """``module_symbol_origins`` round-trips through JSON as tuples, not a separate per-arg representation."""
    fs = FrozenSignature(
        entry="k",
        mangled="_QPk",
        args=(
            FrozenArg(
                fortran_name="nrdmax",
                sdfg_name="nrdmax",
                kind=FrozenArgKind.ARRAY,
                dtype="int32",
                rank=1,
                shape=("10",),
                intent="inout",
            ),
        ),
        free_symbols=("nproma",),
        module_symbol_origins={"nproma": ModuleOrigin("mo_parallel_config", "nproma")},
    )
    p = tmp_path / "fs.json"
    fs.to_json(str(p))
    rt = FrozenSignature.from_json(str(p))
    assert rt.module_symbol_origins == {"nproma": ("mo_parallel_config", "nproma")}
    assert rt.args[0].shape == ("10",)


def test_record_member_symbol_reads_the_host_member(tmp_path):
    """A free symbol read off a module record's scalar member (QE ``v(dffts % nnr)``) is assigned from that member
    of the imported record, and the member survives a JSON round trip."""
    fs = FrozenSignature(
        entry="k",
        mangled="_QPk",
        args=(),
        free_symbols=("dffts_nnr",),
        module_symbol_origins={"dffts_nnr": ModuleOrigin("fft_base", "dffts")},
        module_symbol_members={"dffts_nnr": "nnr"},
    )
    p = tmp_path / "fs.json"
    fs.to_json(str(p))
    rt = FrozenSignature.from_json(str(p))
    assert rt.module_symbol_members == {"dffts_nnr": "nnr"}
    assert "    dffts_nnr = int(dffts_nnr__mod%nnr, c_int)" in _build_symbol_assigns(
        rt, FlattenPlan(), set(), _iface({})
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
