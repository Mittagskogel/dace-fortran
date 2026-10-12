"""A binding allocate sized by a position symbol (``__sym_smap_lb_1``, the SDFG's name for ``smap_lb(1)``) spells the
element instead: a leading ``_`` is no Fortran identifier (QE h_psi's ``smap_exx % lb`` / ``% ub`` extents)."""

from dace_fortran.bindings.block_builders import _position_symbols_as_elements
from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenArgKind, FrozenSignature


def _sig() -> FrozenSignature:
    """``smap_lb(3)`` and ``smap_ub(3)``, the flat members, plus a rank-2 ``smap_lb_tab`` sharing the prefix."""

    def array(name: str, rank: int) -> FrozenArg:
        return FrozenArg(
            fortran_name=name, sdfg_name=name, kind=FrozenArgKind.ARRAY, dtype="int32", rank=rank, shape=("3",) * rank
        )

    return FrozenSignature(
        entry="k", mangled="_QPk", args=(array("smap_lb", 1), array("smap_ub", 1), array("smap_lb_tab", 2))
    )


def test_position_symbols_become_elements():
    expr = "-__sym_smap_lb_1 + __sym_smap_ub_1 + 1, -__sym_smap_lb_2 + __sym_smap_ub_2 + 1"
    assert _position_symbols_as_elements(expr, _sig()) == "-smap_lb(1) + smap_ub(1) + 1, -smap_lb(2) + smap_ub(2) + 1"


def test_longest_array_of_matching_rank_wins():
    """``smap_lb_tab_1_2`` is ``smap_lb_tab(1, 2)``; ``smap_lb`` is rank 1 and cannot take ``tab_1_2``."""
    assert _position_symbols_as_elements("__sym_smap_lb_tab_1_2", _sig()) == "smap_lb_tab(1, 2)"


def test_unknown_position_symbol_is_left_as_is():
    assert _position_symbols_as_elements("__sym_other_1 + n", _sig()) == "__sym_other_1 + n"


if __name__ == "__main__":
    test_position_symbols_become_elements()
    test_longest_array_of_matching_rank_wins()
    test_unknown_position_symbol_is_left_as_is()
