from __future__ import annotations

import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixBondLength

from onepiece.adsorption import (
    DEFAULT_ADSORBATE_TOKENS,
    add_adsorption_energies,
    add_catalysis_hub_adsorption_energies,
    add_elemental_adsorption_energy,
    add_elemental_adsorption_free_energy,
    add_recipe_adsorption_energies,
    adsorption_view,
    annotate_copt_paths,
    assign_surface_references,
    assign_surface_reference_descendants,
    compile_adsorbate_pattern,
    compile_reference_descendant_pattern,
    copt_barrier_summary,
    copt_profile_points,
    default_room_temperature_phase,
    guess_adsorbate,
    infer_adsorbate_recipe,
    infer_adsorption_recipes,
    infer_reference_equation_from_formula,
    parse_reference_equation,
    surface_key_from_name,
)
from onepiece.thermo import (
    add_gibbs_free_energy,
    apply_gas_reference_corrections,
    oxygen_reference_from_h2o_h2,
)
from onepiece.workflows import apply_operation


def test_assign_references_and_adsorption_energies() -> None:
    frame = pd.DataFrame(
        {
            "Name": [
                "Ni-211-clean-1x1",
                "Ni-211-clean-1x1-CO-1",
                "Ni-211-clean-1x1-CH3O-1",
            ],
            "Formula": ["Ni4", "CNi4O", "CH3Ni4O"],
            "E": [-10.0, -25.0, -31.0],
            "Ni": [4, 4, 4],
            "C": [0, 1, 1],
            "H": [0, 0, 3],
            "O": [0, 1, 1],
        }
    )

    referenced = assign_surface_references(frame)
    result = add_adsorption_energies(
        referenced,
        {"CO": -14.0, "CH3OH": -20.0, "H2": -6.0},
    )
    view = adsorption_view(result)

    co = view.loc[view["adsorbate"].eq("CO")].iloc[0]
    ch3o = view.loc[view["adsorbate"].eq("CH3O")].iloc[0]

    assert co["surface_ref_name"] == "Ni-211-clean-1x1"
    assert np.isclose(co["E_ads_CO_total_eV"], -1.0)
    assert np.isclose(co["E_ads_CO_eV"], -1.0)
    assert np.isclose(ch3o["E_ads_CH3OH_to_CH3O_total_eV"], -4.0)
    assert np.isclose(ch3o["E_ads_CH3OH_to_CH3O_eV"], -4.0)


def test_co_adsorption_energy_per_co_divides_total_by_n_co() -> None:
    frame = pd.DataFrame(
        {
            "Name": [
                "Ni-211-clean-1x1",
                "Ni-211-clean-1x1-CO-2",
            ],
            "Formula": ["Ni4", "C2Ni4O2"],
            "E": [-10.0, -38.0],
            "Ni": [4, 4],
            "C": [0, 2],
            "H": [0, 0],
            "O": [0, 2],
        }
    )

    referenced = assign_surface_references(frame)
    result = add_adsorption_energies(referenced, {"CO": -13.0})
    co = adsorption_view(result).iloc[0]

    assert co["n_CO_adsorbates"] == 2
    assert np.isclose(co["E_ads_CO_total_eV"], -2.0)
    assert np.isclose(co["E_ads_CO_eV"], -1.0)


def test_copt_barrier_summary_from_path_scan() -> None:
    frame = pd.DataFrame(
        {
            "dataset_label": ["Ni-slabs"] * 4,
            "Name": [
                "Ni-211-clean-3x3x4-copt-CO_H%HCO-1-00",
                "Ni-211-clean-3x3x4-copt-CO_H%HCO-1-01",
                "Ni-211-clean-3x3x4-copt-CO_H%HCO-1-02",
                "Ni-211-clean-3x3x4-copt-CO_H%HCO-1-03",
            ],
            "Path": [
                "/tmp/Ni/slabs/211/clean/3x3x4/copt/CO_H%HCO/1/00",
                "/tmp/Ni/slabs/211/clean/3x3x4/copt/CO_H%HCO/1/01",
                "/tmp/Ni/slabs/211/clean/3x3x4/copt/CO_H%HCO/1/02",
                "/tmp/Ni/slabs/211/clean/3x3x4/copt/CO_H%HCO/1/03",
            ],
            "Formula": ["CHNi36O"] * 4,
            "E": [-10.0, -9.0, -8.5, -9.2],
        }
    )

    points = copt_profile_points(frame)
    summary = copt_barrier_summary(frame)

    assert len(points) == 4
    assert points["relative_E_from_initial_eV"].max() == 1.5
    assert len(summary) == 1
    assert summary["copt_reaction"].iloc[0] == "CO_H%HCO"
    assert np.isclose(summary["forward_barrier_eV"].iloc[0], 1.5)
    assert np.isclose(summary["reaction_energy_eV"].iloc[0], 0.8)


def test_annotate_copt_paths_uses_name_index_and_fixbond_constraints() -> None:
    atoms = Atoms("CO", positions=[(0, 0, 0), (1.15, 0, 0)])
    atoms.set_constraint(FixBondLength(0, 1))
    frame = pd.DataFrame({"Path": ["/tmp/copt/CO2%COOH/pathA/00"], "struc": [atoms], "E": [-1.0]})
    frame.index = pd.Index(["Cu-211-Ga-copt-CO2%COOH-pathA-00"], name="Name")

    annotated = annotate_copt_paths(frame)
    row = annotated.iloc[0]

    assert annotated.index.tolist() == ["Cu-211-Ga-copt-CO2%COOH-pathA-00"]
    assert row["copt_initial_state"] == "CO2"
    assert row["copt_final_state"] == "COOH"
    assert row["copt_constraint_kind"] == "FixBondLengths"
    assert row["copt_fixed_bond_pairs"] == [(0, 1)]
    assert len(row["copt_fixed_bond_lengths_A"]) == 1


def test_annotate_copt_paths_builds_series_id_from_path_when_name_is_only_step() -> None:
    frame = pd.DataFrame(
        {
            "Path": ["/tmp/Cu++slabs++211++Ga-surface++3x3x4++1++a/copt/H2COOH_AB_2%H2CO_OH_1/1/03"],
            "E": [-1.0],
        },
        index=pd.Index(["03"], name="Name"),
    )

    annotated = annotate_copt_paths(frame)
    row = annotated.iloc[0]

    assert row["copt_surface_base"] == "Cu++slabs++211++Ga-surface++3x3x4++1++a"
    assert row["copt_reaction"] == "H2COOH_AB_2%H2CO_OH_1"
    assert row["copt_path_id"] == "1"
    assert row["copt_step"] == 3
    assert str(row["copt_series_id"]).endswith("|Cu++slabs++211++Ga-surface++3x3x4++1++a|H2COOH_AB_2%H2CO_OH_1|1")


def test_recipe_adsorption_energies_match_expected_values() -> None:
    frame = pd.DataFrame(
        {
            "Name": [
                "Ni-211-clean-1x1",
                "Ni-211-clean-1x1-CO-1",
                "Ni-211-clean-1x1-CH3O-1",
            ],
            "Formula": ["Ni4", "CNi4O", "CH3Ni4O"],
            "E": [-10.0, -25.0, -31.0],
            "Ni": [4, 4, 4],
            "C": [0, 1, 1],
            "H": [0, 0, 3],
            "O": [0, 1, 1],
        }
    )
    referenced = assign_surface_references(frame)
    result = add_recipe_adsorption_energies(
        referenced,
        {"CO": -14.0, "H2": -6.0},
        {
            "CO": {"basis": "C", "gas_refs": {"CO": 1.0}},
            "CH3O": {"basis": "C", "gas_refs": {"CO": 1.0, "H2": 1.5}},
        },
    )

    co = result.loc[result["Name"] == "Ni-211-clean-1x1-CO-1"].iloc[0]
    ch3o = result.loc[result["Name"] == "Ni-211-clean-1x1-CH3O-1"].iloc[0]

    assert np.isclose(co["E_ads_CO_total_eV"], -1.0)
    assert np.isclose(co["E_ads_CO_eV"], -1.0)
    assert np.isclose(ch3o["E_ads_CH3O_total_eV"], 2.0)
    assert np.isclose(ch3o["E_ads_CH3O_eV"], 2.0)


def test_recipe_adsorption_uses_basis_multiplier_for_multiple_adsorbates() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Ni-211-clean-1x1", "Ni-211-clean-1x1-CO-2"],
            "Formula": ["Ni4", "C2Ni4O2"],
            "E": [-10.0, -38.0],
            "Ni": [4, 4],
            "C": [0, 2],
            "H": [0, 0],
            "O": [0, 2],
        }
    )
    referenced = assign_surface_references(frame)
    result = add_recipe_adsorption_energies(
        referenced,
        {"CO": -13.0},
        {"CO": {"basis": "C", "gas_refs": {"CO": 1.0}}},
    )
    row = result.loc[result["Name"] == "Ni-211-clean-1x1-CO-2"].iloc[0]

    assert row["n_CO_adsorbates"] == 2
    assert np.isclose(row["E_ads_CO_total_eV"], -2.0)
    assert np.isclose(row["E_ads_CO_eV"], -1.0)


def test_parse_reference_equation_supports_add_and_subtract_terms() -> None:
    refs = parse_reference_equation("CO2+H2-H2O")

    assert refs == {"CO2": 1.0, "H2": 1.0, "H2O": -1.0}


def test_infer_reference_equation_from_formula_uses_default_chon_basis() -> None:
    refs = infer_reference_equation_from_formula("CH3O")

    assert refs == {"CO2": 1.0, "H2": 2.5, "H2O": -1.0}


def test_infer_adsorbate_recipe_tracks_bulk_refs_for_metal_containing_formula() -> None:
    recipe = infer_adsorbate_recipe("CuCO", formula="CuCO")

    assert recipe["basis"] == "C"
    assert recipe["gas_refs"] == {"CO2": 1.0, "H2": 1.0, "H2O": -1.0}
    assert recipe["bulk_refs"] == {"Cu": 1.0}
    assert recipe["bulk_phases"] == {"Cu": "fcc"}
    assert default_room_temperature_phase("Cu") == "fcc"


def test_infer_adsorption_recipes_from_dataset_and_auto_apply_recipes() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Ni-211-clean-1x1", "Ni-211-clean-1x1-CH3O-1"],
            "Formula": ["Ni4", "CH3Ni4O"],
            "E": [-10.0, -31.0],
            "Ni": [4, 4],
            "C": [0, 1],
            "H": [0, 3],
            "O": [0, 1],
            "adsorbate": ["", "CH3O"],
        }
    )
    referenced = assign_surface_references(frame)
    recipes = infer_adsorption_recipes(referenced)
    result = add_recipe_adsorption_energies(
        referenced,
        {"CO2": -20.0, "H2": -6.0, "H2O": -12.0, "NH3": -8.0},
        None,
    )

    row = result.loc[result["Name"] == "Ni-211-clean-1x1-CH3O-1"].iloc[0]
    assert recipes["CH3O"]["gas_refs"] == {"CO2": 1.0, "H2": 2.5, "H2O": -1.0}
    assert np.isclose(row["E_ads_CH3O_total_eV"], 2.0)
    assert np.isclose(row["E_ads_CH3O_eV"], 2.0)


def test_guess_adsorbate_supports_broader_adsorbate_set() -> None:
    assert "HCOO" in DEFAULT_ADSORBATE_TOKENS
    assert "NH2" in DEFAULT_ADSORBATE_TOKENS
    assert guess_adsorbate("MgOCu-HCOO-1") == "HCOO"
    assert guess_adsorbate("CuMgO-NH2-bridge") == "NH2"
    assert guess_adsorbate("CuMgO-H2NCHO-4") == "H2NCHO"
    assert guess_adsorbate("CuMgO-CO_NH2-1") == "CO_NH2"


def test_adsorbate_detection_accepts_custom_token_api() -> None:
    pattern = compile_adsorbate_pattern(("NO3",))

    assert guess_adsorbate("Cu-111-clean-NO3-1", tokens=("NO3",), pattern=pattern) == "NO3"
    assert surface_key_from_name("Cu-111-clean-NO3-1", tokens=("NO3",), adsorbate_pattern=pattern) == "Cu-111-clean"
    assert guess_adsorbate("Cu-111-clean-NO3-1", tokens=()) == ""


def test_surface_key_strips_extended_adsorbate_tokens() -> None:
    assert surface_key_from_name("CuMgO-H2NCHO-4") == "CuMgO"
    assert surface_key_from_name("MgOCu-HCOO_H-1") == "MgOCu"


def test_lowercase_o_slab_size_prefix_is_not_adsorbate() -> None:
    name = "Cu-slabs-111-clean-o_3x3x4-Ga10O12"
    assert guess_adsorbate(name) == ""
    assert surface_key_from_name(name) == name


def test_ci_and_copt_names_resolve_to_parent_surface_key() -> None:
    surface = "Cu-slabs-111-Ga-surface-3x3x4-1"

    assert surface_key_from_name(f"{surface}-copt") == surface
    assert surface_key_from_name(f"{surface}-copt-HCOO_H%H2COO-AAB-00") == surface
    assert surface_key_from_name(f"{surface}-CI-HCOO_AB%HCOOH_AB-1-00") == surface


def test_assign_surface_references_uses_parent_folder_for_ci_and_copt_descendants() -> None:
    surface = "Cu-slabs-111-Ga-surface-3x3x4-1"
    frame = pd.DataFrame(
        {
            "Name": [
                surface,
                f"{surface}-copt",
                f"{surface}-CI-HCOO_AB%HCOOH_AB-1-00",
                "00",
            ],
            "relative_path": [
                "Cu/slabs/111/Ga-surface/3x3x4/1",
                "Cu/slabs/111/Ga-surface/3x3x4/1/copt",
                "Cu/slabs/111/Ga-surface/3x3x4/1/CI/HCOO_AB%HCOOH_AB/1/00",
                "Cu/slabs/111/Ga-surface/3x3x4/1/custom_scan/00",
            ],
            "Formula": ["Cu35Ga", "CHCu35GaO", "CH2Cu35GaO2", "H2Cu35Ga"],
            "E": [-30.0, -31.0, -32.0, -30.5],
            "Cu": [35, 35, 35, 35],
            "Ga": [1, 1, 1, 1],
            "C": [0, 1, 1, 0],
            "H": [0, 1, 2, 2],
            "O": [0, 1, 2, 0],
        }
    )

    referenced = assign_surface_references(frame)

    descendants = referenced.loc[referenced["Name"].ne(surface)]
    assert descendants["surface_ref_name"].tolist() == [surface, surface, surface]
    assert descendants["surface_key"].tolist() == [surface, surface, surface]
    assert descendants["is_adsorbate"].tolist() == [True, True, True]
    assert descendants["surface_key_from_reference_path"].tolist() == [True, True, True]


def test_surface_reference_descendant_logic_accepts_custom_path_columns_and_markers() -> None:
    surface = "M-clean"
    assert compile_reference_descendant_pattern(("neb",)).search(f"{surface}-neb-00")
    frame = pd.DataFrame(
        {
            "Name": [surface, f"{surface}-neb-00", "custom-child"],
            "calc_path": ["M/clean", "M/clean/neb/00", "M/clean/scan/01"],
            "Formula": ["M4", "M4H", "M4H2"],
            "E": [-10.0, -10.5, -11.0],
            "M": [4, 4, 4],
            "H": [0, 1, 2],
        }
    )

    referenced = assign_surface_references(
        frame,
        reference_descendant_markers=("neb",),
        reference_path_columns=("calc_path",),
    )
    remapped = assign_surface_reference_descendants(
        referenced,
        referenced.loc[referenced["Name"].eq(surface)],
        path_columns=("calc_path",),
    )

    descendants = remapped.loc[remapped["Name"].ne(surface)]
    assert descendants["surface_ref_name"].tolist() == [surface, surface]
    assert descendants["surface_key"].tolist() == [surface, surface]
    assert descendants["surface_key_from_reference_path"].tolist() == [True, True]


def test_nested_oxygen_adsorption_sequence_uses_topmost_surface_reference() -> None:
    surface = "Cu-slabs-111-Ga-surface-3x3x4-1"
    oxygenated = f"{surface}-Ox4-1"
    co_on_oxygenated = f"{oxygenated}-CO-1"
    frame = pd.DataFrame(
        {
            "Name": [surface, oxygenated, co_on_oxygenated],
            "relative_path": [
                "Cu/slabs/111/Ga-surface/3x3x4/1",
                "Cu/slabs/111/Ga-surface/3x3x4/1/Ox4/1",
                "Cu/slabs/111/Ga-surface/3x3x4/1/Ox4/1/CO/1",
            ],
            "Formula": ["Cu35Ga", "Cu35GaO4", "CCu35GaO5"],
            "E": [-30.0, -44.0, -58.0],
            "Cu": [35, 35, 35],
            "Ga": [1, 1, 1],
            "C": [0, 0, 1],
            "H": [0, 0, 0],
            "O": [0, 4, 5],
        }
    )

    referenced = assign_surface_references(frame)
    descendants = referenced.loc[referenced["Name"].ne(surface)]

    assert descendants["surface_ref_name"].tolist() == [surface, surface]
    assert descendants["surface_key"].tolist() == [surface, surface]
    assert descendants["is_adsorbate"].tolist() == [True, True]
    assert descendants["surface_key_from_reference_path"].tolist() == [True, True]
    assert descendants["delta_O"].tolist() == [4.0, 5.0]
    assert referenced.loc[referenced["Name"].eq(co_on_oxygenated), "adsorbate"].iloc[0] == "CO"


def test_elemental_adsorption_energy_uses_structure_difference_and_mu_references() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Cu-211-Ga", "Cu-211-Ga-CH3O-1"],
            "Formula": ["Cu4Ga", "Cu4GaCH3O"],
            "E": [-10.0, -18.5],
            "struc": [
                Atoms("Cu4Ga"),
                Atoms("Cu4GaCH3O"),
            ],
        }
    )

    referenced = assign_surface_references(frame)
    result = add_elemental_adsorption_energy(
        referenced,
        {"CO2": -20.0, "H2": -6.0, "H2O": -12.0},
    )
    row = result.loc[result["Name"] == "Cu-211-Ga-CH3O-1"].iloc[0]

    assert row["C_ads"] == 1
    assert row["H_ads"] == 3
    assert row["O_ads"] == 1
    assert np.isclose(row["mu_H_eV"], -3.0)
    assert np.isclose(row["mu_O_eV"], -6.0)
    assert np.isclose(row["mu_C_eV"], -11.0)
    assert np.isclose(row["adsorption_energy"], 17.5)


def test_add_gibbs_free_energy_distinguishes_gas_and_adsorbate_rows() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["gasphases-H2", "Cu-211-Ga-CH3O-1"],
            "record_class": ["gas_reference", "adsorbate"],
            "E": [-6.0, -18.5],
            "E_ZPE": [0.5, 0.2],
            "Cv_trans": [0.1, np.nan],
            "Cv_rot": [0.2, np.nan],
            "Cv_vib": [0.3, 0.4],
            "S_trans": [0.01, np.nan],
            "S_rot": [0.02, np.nan],
            "S_vib": [0.03, 0.01],
        }
    )

    result = add_gibbs_free_energy(frame, temperature=100.0)

    assert np.isclose(result.loc[0, "G"], -10.9)
    assert np.isclose(result.loc[1, "G"], -18.9)


def test_apply_gas_reference_corrections_returns_corrected_copy() -> None:
    refs = {"CO2": -19.2, "H2": -7.5}

    corrected = apply_gas_reference_corrections(refs, {"CO2": 0.35, "missing": -1.0})

    assert np.isclose(corrected["CO2"], -18.85)
    assert np.isclose(corrected["H2"], -7.5)
    assert refs == {"CO2": -19.2, "H2": -7.5}


def test_oxygen_reference_from_h2o_h2_uses_water_hydrogen_basis() -> None:
    assert np.isclose(oxygen_reference_from_h2o_h2({"H2O": -13.2, "H2": -7.5}), -5.7)


def test_elemental_adsorption_free_energy_uses_gibbs_column_and_gas_gibbs_references() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Cu-211-Ga", "Cu-211-Ga-CH3O-1"],
            "Formula": ["Cu4Ga", "Cu4GaCH3O"],
            "E": [-10.0, -18.5],
            "G": [-9.5, -18.0],
            "struc": [
                Atoms("Cu4Ga"),
                Atoms("Cu4GaCH3O"),
            ],
        }
    )

    referenced = assign_surface_references(frame)
    result = add_elemental_adsorption_free_energy(
        referenced,
        {"CO2": -19.0, "H2": -5.0, "H2O": -11.0},
    )
    row = result.loc[result["Name"] == "Cu-211-Ga-CH3O-1"].iloc[0]

    assert np.isclose(row["surface_ref_G"], -9.5)
    assert np.isclose(row["mu_H_G_eV"], -2.5)
    assert np.isclose(row["mu_O_G_eV"], -6.0)
    assert np.isclose(row["mu_C_G_eV"], -10.5)
    assert np.isclose(row["adsorption_free_energy"], 15.5)


def test_workflow_operations_can_derive_gibbs_and_adsorption_free_energy() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["gasphases-CO2", "gasphases-H2", "gasphases-H2O", "Cu-211-Ga", "Cu-211-Ga-CH3O-1"],
            "Formula": ["CO2", "H2", "H2O", "Cu4Ga", "Cu4GaCH3O"],
            "record_class": ["gas_reference", "gas_reference", "gas_reference", "surface", "adsorbate"],
            "E": [-20.0, -6.0, -8.0, -10.0, -18.5],
            "E_ZPE": [0.1, 0.1, 0.1, 0.2, 0.2],
            "Cv_trans": [0.1, 0.1, 0.1, np.nan, np.nan],
            "Cv_rot": [0.1, 0.1, 0.1, np.nan, np.nan],
            "Cv_vib": [0.1, 0.1, 0.1, 0.4, 0.4],
            "S_trans": [0.001, 0.001, 0.001, np.nan, np.nan],
            "S_rot": [0.001, 0.001, 0.001, np.nan, np.nan],
            "S_vib": [0.001, 0.001, 0.001, 0.01, 0.01],
            "struc": [None, None, None, Atoms("Cu4Ga"), Atoms("Cu4GaCH3O")],
        }
    )

    result = apply_operation(frame, {"kind": "derive_gibbs_free_energy", "temperature": 100.0})
    result = apply_operation(
        result,
        {
            "kind": "derive_gibbs_adsorption",
            "temperature": 100.0,
            "gas_references": {"CO2": -20.0, "H2": -6.0, "H2O": -8.0},
        },
    )
    row = result.loc[result["Name"] == "Cu-211-Ga-CH3O-1"].iloc[0]

    assert pd.notna(row["G"])
    assert pd.notna(row["adsorption_free_energy"])


def test_catalysis_hub_adsorption_energy_reproduces_published_reaction_energy() -> None:
    frame = pd.DataFrame(
        {
            "reaction_id": ["rxn-1", "rxn-1", "rxn-1"],
            "reaction_system_name": ["CO2gas", "star", "CO2star"],
            "surfaceComposition": ["Ni-fcc", "Ni-fcc", "Ni-fcc"],
            "E": [-22.965938, -122.794795, -145.921782],
            "reactionEnergy": [-0.161049, -0.161049, -0.161049],
        }
    )

    result = add_catalysis_hub_adsorption_energies(frame)
    row = result.loc[result["reaction_system_name"] == "CO2star"].iloc[0]

    assert row["cathub_system_kind"] == "adsorbate"
    assert row["cathub_adsorbate"] == "CO2"
    assert np.isclose(row["surface_ref_E"], -122.794795)
    assert np.isclose(row["gas_ref_E"], -22.965938)
    assert np.isclose(row["adsorption_energy"], -0.161049)
    assert np.isclose(row["adsorption_energy_delta_vs_reactionEnergy"], 0.0)
