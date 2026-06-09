from __future__ import annotations

import numpy as np
import pandas as pd

from onepiece.ir import (
    DEFAULT_CO_FREQUENCY_SCALING,
    DEFAULT_IR_TOLERANCE_CM1,
    add_ir_peak_matches,
    adsorption_frequency_plot_table,
    match_ir_frequencies,
    plot_adsorption_energy_vs_frequency,
)
from onepiece.workflows.engine import apply_operation


def test_match_ir_frequencies_uses_30_cm1_tolerance() -> None:
    matches = match_ir_frequencies([1910.0, 2065.0, -50.0], [1913.0, 2054.0], tolerance_cm1=30.0)

    assert len(matches) == 2
    assert matches[0]["experimental_peak_cm1"] == 1913.0
    assert matches[0]["within_tolerance"] is True
    assert matches[1]["experimental_peak_cm1"] == 2054.0
    assert matches[1]["within_tolerance"] is True


def test_add_ir_peak_matches_for_co_and_ch3o_rows() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Ni-211-CO-top", "Ni-211-CH3O-bridge", "clean-slab"],
            "adsorbate": ["CO", "CH3O", ""],
            "frequencies_cm1": [[2058.0, 350.0], [1040.0, 1200.0], [150.0, 200.0]],
        },
        index=["co", "ch3o", "clean"],
    )

    enriched = add_ir_peak_matches(frame)

    co = enriched.loc["co"]
    ch3o = enriched.loc["ch3o"]
    clean = enriched.loc["clean"]

    assert co["ir_reference_species"] == "CO"
    assert np.isclose(co["ir_best_experimental_peak_cm1"], 2054.0)
    assert np.isclose(co["ir_best_delta_cm1"], 4.0)
    assert bool(co["ir_best_within_tolerance"]) is True
    assert np.isclose(co["ir_peak_tolerance_cm1"], DEFAULT_IR_TOLERANCE_CM1)

    assert ch3o["ir_reference_species"] == "CH3O"
    assert np.isclose(ch3o["ir_best_experimental_peak_cm1"], 1043.0)
    assert np.isclose(ch3o["ir_best_abs_delta_cm1"], 3.0)
    assert bool(ch3o["ir_best_within_tolerance"]) is True

    assert pd.isna(clean["ir_best_experimental_peak_cm1"])
    assert clean["ir_reference_species"] is None


def test_add_ir_peak_matches_can_infer_species_from_name() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["candidate-CO-site-a"],
            "frequencies_cm1": [[1932.0]],
        },
        index=["row-a"],
    )

    enriched = add_ir_peak_matches(frame, species_column="missing_column")

    row = enriched.loc["row-a"]
    assert row["ir_reference_species"] == "CO"
    assert np.isclose(row["ir_best_experimental_peak_cm1"], 1934.0)
    assert bool(row["ir_best_within_tolerance"]) is True


def test_workflow_operation_adds_ir_matching_columns() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["row-co"],
            "adsorbate": ["CO"],
            "frequencies_cm1": [[2180.0]],
        },
        index=["row-co"],
    )

    result = apply_operation(
        frame,
        {
            "kind": "derive_ir_peak_matches",
            "tolerance_cm1": 30.0,
        },
    )

    row = result.loc["row-co"]
    assert row["ir_reference_species"] == "CO"
    assert np.isclose(row["ir_best_experimental_peak_cm1"], 2187.0)
    assert bool(row["ir_best_within_tolerance"]) is True


def test_adsorption_frequency_plot_table_scales_co_only() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["co-row", "ch3o-row", "clean-row"],
            "adsorbate": ["CO", "CH3O", ""],
            "surface_ref": ["Cu(211)", "Ga/Cu(211)", None],
            "frequencies_cm1": [[2058.0], [1040.0], [400.0]],
            "E_ads_CO_eV": [-0.42, np.nan, np.nan],
            "E_ads_CH3OH_to_CH3O_eV": [np.nan, -1.15, np.nan],
        },
        index=["co", "ch3o", "clean"],
    )

    plot_table = adsorption_frequency_plot_table(frame)

    assert list(plot_table["adsorbate"]) == ["CO", "CH3O"]

    co = plot_table.loc[plot_table["adsorbate"] == "CO"].iloc[0]
    ch3o = plot_table.loc[plot_table["adsorbate"] == "CH3O"].iloc[0]

    assert np.isclose(co["adsorption_energy_eV"], -0.42)
    assert co["surface_ref"] == "Cu(211)"
    assert np.isclose(co["frequency_scaling_factor"], DEFAULT_CO_FREQUENCY_SCALING)
    assert np.isclose(co["scaled_frequency_cm1"], 2058.0 * DEFAULT_CO_FREQUENCY_SCALING)
    assert np.isclose(co["experimental_peak_cm1"], 2054.0)

    assert np.isclose(ch3o["adsorption_energy_eV"], -1.15)
    assert ch3o["surface_ref"] == "Ga/Cu(211)"
    assert np.isclose(ch3o["frequency_scaling_factor"], 1.0)
    assert np.isclose(ch3o["scaled_frequency_cm1"], 1040.0)
    assert np.isclose(ch3o["experimental_peak_cm1"], 1043.0)


def test_plot_adsorption_energy_vs_frequency_returns_faceted_axes_with_limited_y_window() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["co-row-a", "co-row-b", "ch3o-row-a", "ch3o-row-b"],
            "adsorbate": ["CO", "CO", "CH3O", "CH3O"],
            "surface_ref": ["Cu(211)", "Ga/Cu(211)", "Cu(211)", "Ga/Cu(211)"],
            "frequencies_cm1": [[2058.0], [2064.0], [1040.0], [1052.0]],
            "E_ads_CO_eV": [-0.42, -0.30, np.nan, np.nan],
            "E_ads_CH3OH_to_CH3O_eV": [np.nan, np.nan, -1.15, -1.05],
        },
        index=["co-a", "co-b", "ch3o-a", "ch3o-b"],
    )

    fig, axes = plot_adsorption_energy_vs_frequency(frame, frequency_window_cm1=50.0)
    assert fig is not None
    assert len(axes) == 2
    for ax in axes:
        assert ax.get_xlabel() == "Adsorption energy / eV"
        assert ax.get_ylabel() == "Frequency / cm$^{-1}$"
        ymin, ymax = ax.get_ylim()
        assert np.isclose(ymax - ymin, 50.0)
    fig.clf()
