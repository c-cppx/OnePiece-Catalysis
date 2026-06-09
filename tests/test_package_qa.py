from __future__ import annotations

from pathlib import Path

from onepiece import bundled_catalysis_hub_dataset, run_catalysis_hub_self_test


def test_bundled_catalysis_hub_dataset_exists() -> None:
    path = bundled_catalysis_hub_dataset()

    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == "catalysis_hub_co2_subset.hdf"


def test_catalysis_hub_self_test_passes_on_bundled_dataset() -> None:
    result = run_catalysis_hub_self_test()

    assert result.passed is True
    assert result.details["rows"] > 0
    assert result.details["adsorbate_rows"] > 0
    assert result.details["computed_adsorption_rows"] > 0
    assert result.details["max_abs_delta_vs_reaction_energy_ev"] is not None
    assert result.details["max_abs_delta_vs_reaction_energy_ev"] < 1e-8
