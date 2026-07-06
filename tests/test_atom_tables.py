from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write

from onepiece.atom_tables import (
    atom_reference_delta_table,
    atom_reference_map,
    atomic_properties_multiindex_table,
    ensure_structure_columns,
    gas_reference_element_statistics,
    summarize_atom_reference_deltas,
)


def test_element_specific_gas_mapping_ignores_gas_atom_order() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Cu2-clean", "gasphases-CO", "Cu2-clean-CO-1"],
            "record_kind": ["surface", "gas", "adsorbed_surface"],
            "adsorbate": ["", "CO", "CO"],
            "surface_ref_name": ["Cu2-clean", "", "Cu2-clean"],
            "CONTCAR": [
                _atoms(["Cu", "Cu"]),
                _atoms(["O", "C"]),
                _atoms(["Cu", "Cu", "C", "O"]),
            ],
            "integrated_electron_populations": [
                [10.9, 10.8],
                [8.0, 4.0],
                [10.8, 10.9, 3.5, 8.5],
            ],
            "atomic_charges": [
                [0.1, 0.2],
                [-2.0, 2.0],
                [0.2, 0.1, 2.5, -1.5],
            ],
            "atomic_magnetic_moments": [
                [0.0, 0.1],
                [-0.2, 0.2],
                [0.0, 0.2, 0.25, -0.1],
            ],
        }
    )

    properties = atomic_properties_multiindex_table(frame)
    gas_stats = gas_reference_element_statistics(frame)
    reference_map = atom_reference_map(frame)
    deltas = atom_reference_delta_table(properties, reference_map, gas_stats)

    assert properties.index.names == ["calculation_name", "atom_index"]
    assert reference_map.loc[("Cu2-clean-CO-1", 2), "atom_role"] == "adsorbate"
    assert reference_map.loc[("Cu2-clean-CO-1", 3), "atom_role"] == "adsorbate"
    assert np.isclose(gas_stats.loc[("CO", "C"), "gas_atomic_charge_mean_e"], 2.0)
    assert np.isclose(gas_stats.loc[("CO", "O"), "gas_atomic_charge_mean_e"], -2.0)
    assert np.isclose(deltas.loc[("Cu2-clean-CO-1", 2), "delta_atomic_charge_vs_ref_e"], 0.5)
    assert np.isclose(deltas.loc[("Cu2-clean-CO-1", 3), "delta_atomic_charge_vs_ref_e"], 0.5)
    assert np.isclose(deltas.loc[("Cu2-clean-CO-1", 1), "delta_atomic_charge_vs_ref_e"], -0.1)


def test_duplicate_gas_elements_use_element_mean_reference() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Cu2-clean", "gasphases-HCOO", "Cu2-clean-HCOO-1"],
            "record_kind": ["surface", "gas", "adsorbed_surface"],
            "adsorbate": ["", "HCOO", "HCOO"],
            "surface_ref_name": ["Cu2-clean", "", "Cu2-clean"],
            "CONTCAR": [
                _atoms(["Cu", "Cu"]),
                _atoms(["C", "O", "O", "H"]),
                _atoms(["Cu", "Cu", "O", "O", "C", "H"]),
            ],
            "integrated_electron_populations": [
                [11.0, 11.0],
                [3.0, 7.0, 9.0, 0.5],
                [10.8, 11.1, 8.5, 9.5, 2.8, 0.6],
            ],
            "atomic_charges": [
                [0.0, 0.0],
                [1.0, -1.0, -3.0, 0.5],
                [0.2, -0.1, -1.5, -2.5, 1.2, 0.4],
            ],
            "atomic_magnetic_moments": [
                [0.0, 0.0],
                [0.1, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.1, 0.1, 0.2, 0.0],
            ],
        }
    )

    properties = atomic_properties_multiindex_table(frame)
    gas_stats = gas_reference_element_statistics(frame)
    deltas = atom_reference_delta_table(properties, atom_reference_map(frame), gas_stats)
    summary = summarize_atom_reference_deltas(deltas)

    assert np.isclose(gas_stats.loc[("HCOO", "O"), "gas_atomic_charge_mean_e"], -2.0)
    assert np.isclose(gas_stats.loc[("HCOO", "O"), "gas_atomic_charge_std_e"], 1.0)
    assert np.isclose(deltas.loc[("Cu2-clean-HCOO-1", 2), "reference_atomic_charge_e"], -2.0)
    assert np.isclose(deltas.loc[("Cu2-clean-HCOO-1", 3), "reference_atomic_charge_e"], -2.0)
    assert np.isclose(deltas.loc[("Cu2-clean-HCOO-1", 2), "delta_atomic_charge_vs_ref_e"], 0.5)
    assert np.isclose(deltas.loc[("Cu2-clean-HCOO-1", 3), "delta_atomic_charge_vs_ref_e"], -0.5)
    assert summary.loc[summary["Name"].eq("Cu2-clean-HCOO-1"), "n_adsorbate_atoms"].iloc[0] == 4


def test_gas_reference_statistics_empty_table_keeps_full_schema() -> None:
    stats = gas_reference_element_statistics(
        pd.DataFrame(
            {
                "Name": ["Cu2-clean"],
                "record_kind": ["surface"],
                "adsorbate": [""],
                "CONTCAR": [_atoms(["Cu", "Cu"])],
            }
        )
    )

    assert stats.empty
    assert stats.index.names == ["adsorbate", "element"]
    assert "gas_atomic_charge_mean_e" in stats.columns
    assert "gas_integrated_electron_population_std" in stats.columns
    assert "gas_atomic_magnetic_moment_mean_muB" in stats.columns


def test_gas_reference_override_selects_requested_reference() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["gasphases-CO", "gasphases-CO-alt"],
            "record_kind": ["gas", "gas"],
            "adsorbate": ["CO", "CO"],
            "CONTCAR": [_atoms(["C", "O"]), _atoms(["C", "O"])],
            "integrated_electron_populations": [[4.0, 8.0], [5.0, 9.0]],
            "atomic_charges": [[1.0, -1.0], [2.0, -2.0]],
            "atomic_magnetic_moments": [[0.1, -0.1], [0.2, -0.2]],
        }
    )

    stats = gas_reference_element_statistics(
        frame,
        gas_reference_by_adsorbate={"CO": "gasphases-CO-alt"},
    )
    reference_map = atom_reference_map(
        frame,
        gas_reference_by_adsorbate={"CO": "gasphases-CO-alt"},
    )

    assert stats.loc[("CO", "C"), "gas_reference_name"] == "gasphases-CO-alt"
    assert np.isclose(stats.loc[("CO", "C"), "gas_atomic_charge_mean_e"], 2.0)
    assert reference_map.loc[("gasphases-CO-alt", 0), "reference_name"] == "gasphases-CO-alt"


def test_ensure_structure_columns_recovers_contcar_and_falls_back(tmp_path: Path) -> None:
    atoms = _atoms(["Cu", "O"])
    contcar = tmp_path / "CONTCAR"
    write(contcar, atoms, format="vasp")

    recovered = ensure_structure_columns(pd.DataFrame({"Name": ["calc"], "contcar_path": [str(contcar)]}))
    assert recovered.loc[0, "CONTCAR"].__class__.__name__ == "Atoms"
    assert recovered.loc[0, "structure_source_for_atom_table"] == "contcar_path"

    fallback = ensure_structure_columns(pd.DataFrame({"Name": ["calc"], "struc": [atoms]}))
    assert fallback.loc[0, "CONTCAR"].__class__.__name__ == "Atoms"
    assert fallback.loc[0, "structure_source_for_atom_table"] == "fallback:struc"


def test_surface_same_index_element_mismatch_is_flagged() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Cu2-clean", "gasphases-CO", "Cu2-clean-CO-bad"],
            "record_kind": ["surface", "gas", "adsorbed_surface"],
            "adsorbate": ["", "CO", "CO"],
            "surface_ref_name": ["Cu2-clean", "", "Cu2-clean"],
            "CONTCAR": [
                _atoms(["Cu", "Cu"]),
                _atoms(["C", "O"]),
                _atoms(["Cu", "Ni", "C", "O"]),
            ],
            "integrated_electron_populations": [
                [11.0, 11.0],
                [4.0, 8.0],
                [10.9, 9.0, 4.0, 8.0],
            ],
            "atomic_charges": [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.1, 1.0, 0.0, 0.0],
            ],
            "atomic_magnetic_moments": [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        }
    )

    properties = atomic_properties_multiindex_table(frame)
    reference_map = atom_reference_map(frame)
    deltas = atom_reference_delta_table(properties, reference_map, gas_reference_element_statistics(frame))

    assert reference_map.loc[("Cu2-clean-CO-bad", 1), "mapping_status"] == "element_mismatch"
    assert np.isnan(deltas.loc[("Cu2-clean-CO-bad", 1), "delta_atomic_charge_vs_ref_e"])


def test_missing_gas_reference_uses_adsorbate_formula_tail_count() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Cu2-clean", "Cu2-clean-CH3-1"],
            "record_kind": ["surface", "adsorbed_surface"],
            "adsorbate": ["", "CH3"],
            "surface_ref_name": ["Cu2-clean", "Cu2-clean"],
            "CONTCAR": [
                _atoms(["Cu", "Cu"]),
                _atoms(["Cu", "Cu", "C", "H", "H", "H"]),
            ],
            "integrated_electron_populations": [
                [11.0, 11.0],
                [10.9, 11.0, 4.0, 1.0, 1.0, 1.0],
            ],
            "atomic_charges": [
                [0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "atomic_magnetic_moments": [
                [0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        }
    )

    reference_map = atom_reference_map(frame)
    deltas = atom_reference_delta_table(
        atomic_properties_multiindex_table(frame),
        reference_map,
        gas_reference_element_statistics(frame),
    )

    adsorbate_rows = reference_map.loc["Cu2-clean-CH3-1"]
    adsorbate_rows = adsorbate_rows[adsorbate_rows["atom_role"].eq("adsorbate")]
    assert adsorbate_rows["atom_index"].to_list() == [2, 3, 4, 5]
    assert adsorbate_rows["mapping_status"].eq("missing_reference").all()
    assert np.isnan(deltas.loc[("Cu2-clean-CH3-1", 2), "delta_atomic_charge_vs_ref_e"])


def test_atom_delta_pickle_round_trip_preserves_multiindex(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Name": ["gasphases-CO"],
            "record_kind": ["gas"],
            "adsorbate": ["CO"],
            "surface_ref_name": [""],
            "CONTCAR": [_atoms(["C", "O"])],
            "integrated_electron_populations": [[4.0, 8.0]],
            "atomic_charges": [[0.0, 0.0]],
            "atomic_magnetic_moments": [[0.0, 0.0]],
        }
    )
    properties = atomic_properties_multiindex_table(frame)
    deltas = atom_reference_delta_table(properties, atom_reference_map(frame), gas_reference_element_statistics(frame))

    path = tmp_path / "atom_reference_deltas.pkl"
    deltas.to_pickle(path)
    loaded = pd.read_pickle(path)

    assert loaded.index.names == ["calculation_name", "atom_index"]
    csv = loaded.reset_index(drop=True)
    assert {"calculation_name", "atom_index"} <= set(csv.columns)


def _atoms(symbols: list[str]) -> Atoms:
    positions = [[float(i), 0.0, 0.0] for i in range(len(symbols))]
    return Atoms(symbols=symbols, positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)
