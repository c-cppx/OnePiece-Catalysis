from __future__ import annotations

import numpy as np
import pandas as pd
import sympy as sp

from onepiece.phase_diagrams import (
    build_corrected_phase_expressions,
    build_grouped_surface_phase_diagrams,
    build_phase_field_grid,
    build_surface_phase_diagram,
    default_phase_variables,
    estimate_phase_scan_slopes,
    solve_phase_boundaries,
    stable_phase_scan,
)


def test_stable_phase_scan_matches_notebook_style_hull_selection() -> None:
    frame = pd.DataFrame(
        {
            "legend": ["A", "B", "C"],
            "expr": [sp.Float(0), sp.Symbol("x") - 1, 1 - sp.Symbol("x")],
        },
        index=["phase_a", "phase_b", "phase_c"],
    )

    result = stable_phase_scan(
        frame,
        expression_column="expr",
        variable_values=np.array([0.5, 1.0, 1.5, 2.5, 3.5]),
    )

    assert set(result.stable_frame.index) == {"phase_a", "phase_b", "phase_c"}
    assert list(result.scan_table["phase_minimum"]) == [
        "phase_b",
        "phase_a",
        "phase_c",
        "phase_c",
        "phase_c",
    ]


def test_solve_phase_boundaries_finds_crossing_between_neighbor_phases() -> None:
    x = sp.Symbol("x", positive=True)
    stable = pd.DataFrame(
        {
            "Name": ["left", "right"],
            "coverage": [0.0, 1.0],
            "expr": [0, x - 2],
        },
        index=["left_idx", "right_idx"],
    )

    boundaries = solve_phase_boundaries(
        stable,
        expression_column="expr",
        solve_symbol="x",
        sort_by="coverage",
    )

    assert len(boundaries) == 1
    assert boundaries.iloc[0]["left_name"] == "left"
    assert boundaries.iloc[0]["right_name"] == "right"
    assert boundaries.iloc[0]["solutions"] == [2]


def test_build_corrected_phase_expressions_supports_generic_reference_map() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["clean", "covered"],
            "form_G": [0.0, 4.0],
            "Area": [2.0, 2.0],
            "delta_M": [0.0, 2.0],
            "mu_M": [1.0, 1.0],
        }
    )

    corrected = build_corrected_phase_expressions(
        frame,
        correction_map={"delta_M": ("mu_M", "x")},
        output_column="expr",
        normalized_energy_column=None,
    )

    expr = corrected.loc[1, "expr"]
    assert str(sp.simplify(expr)) == "3.0 - 1.0*x"
    assert corrected.loc[0, "expr"] == 0


def test_build_phase_field_grid_handles_symbolic_temperature_and_pressure_ratio() -> None:
    frame = pd.DataFrame(
        {
            "expr": [
                sp.Float(0),
                sp.Symbol("x") - 2,
                sp.Symbol("T") / 1000 - sp.Symbol("x"),
            ]
        },
        index=["phase_a", "phase_b", "phase_c"],
    )

    field = build_phase_field_grid(
        frame,
        expression_column="expr",
        x_symbol="x",
        x_values=np.array([0.5, 1.5, 2.5]),
        t_symbol="T",
        t_values=np.array([300.0, 800.0]),
    )

    assert field.energy_grid.shape == (3, 2, 3)
    assert field.stable_index.shape == (2, 3)
    assert field.stable_index[0, 0] == 1
    assert field.stable_index[1, 2] == 2


def test_estimate_phase_scan_slopes_returns_symbolic_derivatives() -> None:
    variables = default_phase_variables(T=500.0)
    x = sp.Symbol("x", positive=True)
    frame = pd.DataFrame(
        {
            "expr": [
                0,
                sp.log(x),
                sp.Symbol("T") * sp.log(x),
            ]
        },
        index=["flat", "log", "scaled_log"],
    )

    slopes = estimate_phase_scan_slopes(
        frame,
        expression_column="expr",
        temperature=variables["T"],
    )

    assert slopes["flat"] == 0
    assert sp.simplify(slopes["log"] - 1) == 0
    assert sp.simplify(slopes["scaled_log"] - variables["T"]) == 0


def test_build_surface_phase_diagram_returns_summary_and_field() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["clean", "covered"],
            "Formula": ["M4", "M3X"],
            "form_G": [0.0, 4.0],
            "Area": [2.0, 2.0],
            "delta_M": [0.0, 2.0],
            "mu_M": [1.0, 1.0],
        }
    )
    result = build_surface_phase_diagram(
        frame,
        correction_map={"delta_M": ("mu_M", "x")},
        normalized_energy_column=None,
        x_values=np.array([0.5, 1.0, 5.0]),
        t_values=np.array([300.0, 500.0]),
    )

    assert result.field.energy_grid.shape == (2, 2, 3)
    assert not result.stable_summary.empty
    assert {"phase_id", "Name", "stable_percent", "x_min", "T_min"}.issubset(result.stable_summary.columns)


def test_build_grouped_surface_phase_diagrams_splits_by_group_column() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["a_clean", "a_cov", "b_clean", "b_cov"],
            "hkl": ["100", "100", "111", "111"],
            "form_G": [0.0, 4.0, 0.0, 6.0],
            "Area": [2.0, 2.0, 2.0, 2.0],
            "delta_M": [0.0, 2.0, 0.0, 2.0],
            "mu_M": [1.0, 1.0, 1.0, 1.0],
        }
    )
    grouped = build_grouped_surface_phase_diagrams(
        frame,
        group_column="hkl",
        correction_map={"delta_M": ("mu_M", "x")},
        normalized_energy_column=None,
        x_values=np.array([0.5, 5.0]),
        t_values=np.array([400.0]),
    )

    assert set(grouped.groups) == {"100", "111"}
    assert all(not result.stable_summary.empty for result in grouped.groups.values())
