from __future__ import annotations

import numpy as np
import pandas as pd
import sympy as sp

from onepiece.phase_diagrams import (
    PhaseCandidateValidationRules,
    build_corrected_phase_expressions,
    build_grouped_surface_phase_diagrams,
    build_phase_field_grid,
    build_surface_phase_diagram,
    clean_phase_candidates,
    default_phase_variables,
    estimate_phase_scan_slopes,
    solve_phase_boundaries,
    stable_phase_scan,
    validate_phase_candidates,
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


def test_validate_phase_candidates_supports_generic_cleaning_rules() -> None:
    rules = PhaseCandidateValidationRules(
        adsorbate_tokens=("NO3",),
        adsorbate_name_search_after_pattern=r"\d+x\d+",
        excluded_name_substrings={"backup_candidate": ("backup",)},
        element_count_columns=("M", "X", "O"),
    )
    frame = pd.DataFrame(
        {
            "Name": [
                "M-111-clean-2x2",
                "M-111-clean-2x2-backup",
                "M-111-clean-2x2-NO3-1",
                "M-111-clean-2x2-O-1",
                "M-111-clean-2x2-column",
            ],
            "adsorbate": ["", "", "", "", "CO"],
            "M": [4, 4, 4, 4, 4],
            "X": [1, 1, 1, 1, 1],
            "O": [0, 0, 0, 1, 0],
        }
    )

    audit = validate_phase_candidates(
        frame,
        system="MX",
        phase_set="111",
        allowed_elements=("M", "X"),
        rules=rules,
    )
    cleaned = clean_phase_candidates(
        frame,
        system="MX",
        phase_set="111",
        allowed_elements=("M", "X"),
        rules=rules,
    )

    assert cleaned["Name"].tolist() == ["M-111-clean-2x2"]
    assert cleaned.attrs["phase_input_cleaning_audit"].equals(audit)
    dropped_reasons = ";".join(audit.loc[audit["status"].eq("dropped"), "drop_reasons"])
    assert "backup_candidate" in dropped_reasons
    assert "adsorbate_name_marker" in dropped_reasons
    assert "extra_element_count" in dropped_reasons
    assert "adsorbate_column" in dropped_reasons


def test_phase_candidate_validation_rules_round_trip_from_mapping() -> None:
    rules = PhaseCandidateValidationRules.from_mapping(
        {
            "adsorbate_tokens": ["NO3"],
            "adsorbate_name_search_after_pattern": r"\d+x\d+",
            "excluded_name_patterns": {"temporary_candidate": [r"tmp$"]},
            "allowed_elements": ["M"],
        }
    )
    payload = rules.to_mapping()
    frame = pd.DataFrame(
        {
            "Name": ["M-clean-2x2", "M-clean-2x2-NO3-1", "M-clean-2x2-tmp"],
            "M": [4, 4, 4],
            "N": [0, 1, 0],
        }
    )

    audit = validate_phase_candidates(frame, rules=payload)

    assert rules.adsorbate_tokens == ("NO3",)
    assert payload["adsorbate_tokens"] == ["NO3"]
    assert tuple(payload["excluded_name_patterns"]["temporary_candidate"]) == (r"tmp$",)
    assert audit["status"].tolist() == ["kept", "dropped", "dropped"]
    assert "adsorbate_name_marker" in audit.loc[1, "drop_reasons"]
    assert "temporary_candidate" in audit.loc[2, "drop_reasons"]


def test_validate_phase_candidates_applies_configurable_bulk_exclusions() -> None:
    rules = PhaseCandidateValidationRules(
        bulk_excluded_name_substrings={"not_true_bulk_candidate": ("XANES",)},
        bulk_excluded_path_patterns={"not_true_bulk_candidate": (r"[/\\]slabs[/\\]",)},
    )
    frame = pd.DataFrame(
        {
            "Name": ["M-bulk", "M-bulk-XANES", "M-slab-row"],
            "Path": ["/calc/M/bulk", "/calc/M/bulk/XANES", "/calc/M/slabs/111/ads"],
            "M": [1, 1, 1],
        }
    )

    audit = validate_phase_candidates(
        frame,
        phase_set="Bulk",
        allowed_elements=("M",),
        rules=rules,
    )

    assert audit["status"].tolist() == ["kept", "dropped", "dropped"]
    assert audit.loc[1:, "drop_reasons"].tolist() == ["not_true_bulk_candidate", "not_true_bulk_candidate"]
