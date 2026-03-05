"""Unit tests for script-level search_materials CLI helpers."""

from __future__ import annotations

from argparse import Namespace

import pandas as pd
import pytest

from scripts.phase5_active_learning.search_materials import (
    _build_criteria,
    _parse_custom_filter,
    _resolve_display_columns,
    _validate_args,
    _validate_query_columns,
    main,
)


def _base_args() -> Namespace:
    return Namespace(
        metal=False,
        semiconductor=False,
        insulator=False,
        bandgap_min=None,
        bandgap_max=None,
        melting_min=None,
        melting_max=None,
        hardness_min=None,
        hardness_max=None,
        youngs_min=None,
        youngs_max=None,
        seebeck_min=None,
        em_min=None,
        em_max=None,
        kappa_min=None,
        kappa_max=None,
        density_min=None,
        density_max=None,
        ehull_max=0.1,
        ductile=False,
        filter=None,
        sort=None,
        desc=False,
        max=30,
        summary=False,
        columns=None,
        save=None,
    )


def test_validate_args_rejects_conflicting_presets():
    args = _base_args()
    args.metal = True
    args.semiconductor = True
    ok, message = _validate_args(args)
    assert ok is False
    assert "Only one of --metal/--semiconductor/--insulator" in message


def test_validate_args_rejects_non_finite_and_out_of_range_values():
    args = _base_args()
    args.bandgap_min = float("nan")
    ok, message = _validate_args(args)
    assert ok is False
    assert "--bandgap-min must be finite" in message

    args = _base_args()
    args.em_min = 1.5
    ok, message = _validate_args(args)
    assert ok is False
    assert "--em-min must be within [0, 1]" in message

    args = _base_args()
    args.max = True
    ok, message = _validate_args(args)
    assert ok is False
    assert "--max must be a positive integer" in message

    args = _base_args()
    args.ehull_max = True
    ok, message = _validate_args(args)
    assert ok is False
    assert "--ehull-max must be finite" in message

    args = _base_args()
    args.bandgap_min = False
    ok, message = _validate_args(args)
    assert ok is False
    assert "--bandgap-min must be finite" in message


def test_parse_custom_filter_supports_comparison_operators():
    assert _parse_custom_filter("ehull<0.2") == ("ehull", (None, 0.2))
    assert _parse_custom_filter("density>=5.0") == ("density", (5.0, None))


def test_build_criteria_merges_custom_bounds():
    args = _base_args()
    args.bandgap_min = 0.1
    args.filter = ["bandgap_best<1.5", "bandgap_best>0.5"]
    criteria = _build_criteria(args)
    assert criteria["bandgap_best"] == (0.5, 1.5)


def test_build_criteria_rejects_inconsistent_bounds():
    args = _base_args()
    args.bandgap_min = 1.5
    args.filter = ["bandgap_best<1.0"]
    with pytest.raises(ValueError, match="Inconsistent bounds for bandgap_best"):
        _build_criteria(args)


def test_resolve_display_columns_deduplicates_and_filters_existing_columns():
    results = pd.DataFrame(
        {
            "jid": ["a"],
            "formula": ["SrTiO3"],
            "density": [5.1],
            "kappa_slack": [2.3],
        }
    )
    columns = _resolve_display_columns(results, extra_columns=["density", "missing", "jid"])
    assert columns == ["jid", "formula", "kappa_slack", "density"]


def test_validate_query_columns_rejects_missing_or_non_numeric_filters():
    df = pd.DataFrame({"jid": ["a"], "formula": ["SrTiO3"], "density": ["heavy"]})
    ok, message = _validate_query_columns(
        df,
        criteria={"unknown": (0.0, 1.0)},
        sort_by=None,
    )
    assert ok is False
    assert "Unknown criteria columns" in message

    ok, message = _validate_query_columns(
        df,
        criteria={"density": (0.0, 10.0)},
        sort_by=None,
    )
    assert ok is False
    assert "numeric column" in message

    ok, message = _validate_query_columns(
        pd.DataFrame({"density": [5.0]}),
        criteria={"density": (0.0, 10.0)},
        sort_by="missing",
    )
    assert ok is False
    assert "Unknown sort column" in message


def test_main_prints_validation_errors_to_stderr(capsys):
    code = main(["--metal", "--semiconductor"])
    captured = capsys.readouterr()
    assert code == 2
    assert "[ERROR]" in captured.err
