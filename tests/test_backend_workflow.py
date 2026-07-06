from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ase import Atoms
from ase.io import write

from onepiece.calculation_frame import CalculationFrame
from onepiece.classification import add_file_status_columns, add_workflow_classification
from onepiece.manifests import legacy_manifest_payload, write_workflow_stage_manifest
from onepiece.phase_diagrams import PhaseCandidateValidationRules
from onepiece.workflow_config import ProjectWorkflowConfig, coerce_project_workflow_config
from onepiece.workflows import apply_operation


def test_project_workflow_config_round_trip_mapping() -> None:
    config = ProjectWorkflowConfig.from_mapping(
        {
            "adsorbates": ["CO", "HCOO"],
            "gas_reference_by_adsorbate": {"CO": "gasphases-CO"},
            "systems": ["CuGa"],
            "facets": ["111"],
            "structure_column": "CONTCAR",
            "charge_coordinate_max_delta_A": 0.1,
            "adsorbate_tokens": ["NO3"],
            "reference_descendant_markers": ["neb"],
            "exclude_reference_name_substrings": ["backup"],
            "phase_candidate_validation_rules": {
                "adsorbate_tokens": ["NO3"],
                "excluded_name_substrings": {"backup_candidate": ["backup"]},
            },
        }
    )

    payload = config.to_mapping()

    assert payload["adsorbates"] == ["CO", "HCOO"]
    assert payload["gas_reference_by_adsorbate"] == {"CO": "gasphases-CO"}
    assert payload["systems"] == ["CuGa"]
    assert payload["facets"] == ["111"]
    assert payload["charge_coordinate_max_delta_A"] == 0.1
    assert payload["adsorbate_tokens"] == ["NO3"]
    assert payload["reference_descendant_markers"] == ["neb"]
    assert payload["exclude_reference_name_substrings"] == ["backup"]
    assert payload["phase_candidate_validation_rules"]["adsorbate_tokens"] == ["NO3"]


def test_project_workflow_config_allows_optional_mapping_fields() -> None:
    config = ProjectWorkflowConfig.from_mapping(
        {
            "adsorbates": "CO",
            "gas_reference_by_adsorbate": None,
            "output_names": None,
        }
    )

    assert config.adsorbates == frozenset({"CO"})
    assert config.gas_reference_by_adsorbate == {}
    assert config.output_names == {}
    assert "adsorbate_tokens" not in ProjectWorkflowConfig().reference_assignment_kwargs()
    assert coerce_project_workflow_config(config) is config


def test_project_workflow_config_accepts_phase_rule_objects() -> None:
    rules = PhaseCandidateValidationRules(excluded_name_substrings={"backup_candidate": ("backup",)})
    config = ProjectWorkflowConfig(phase_candidate_validation_rules=rules)

    assert config.phase_candidate_rules() is rules
    assert config.to_mapping()["phase_candidate_validation_rules"]["excluded_name_substrings"] == {
        "backup_candidate": ["backup"]
    }


def test_classification_prioritizes_gas_before_adsorbate_detection() -> None:
    config = ProjectWorkflowConfig(adsorbates=frozenset({"CO", "HCOO"}))
    frame = pd.DataFrame(
        {
            "Name": ["gasphases-CO", "Cu-slabs-111-clean-CO-1", "Cu-bulk-Cu"],
            "Path": ["/calc/gasphases/CO", "/calc/Cu/slabs/111/clean/CO/1", "/calc/Cu/bulk/Cu"],
            "Cu": [0, 4, 1],
            "C": [1, 1, 0],
            "O": [1, 1, 0],
        }
    )

    result = add_workflow_classification(frame, config=config)

    assert result["record_kind"].tolist() == ["gas", "adsorbed_surface", "bulk"]
    assert result["adsorbate"].tolist()[:2] == ["CO", "CO"]
    assert result["workflow_facet"].tolist()[1] == "111"


def test_calculation_frame_chains_structure_recovery_and_file_status(tmp_path: Path) -> None:
    calc_dir = tmp_path / "Cu" / "slabs" / "111" / "clean"
    calc_dir.mkdir(parents=True)
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    contcar = calc_dir / "CONTCAR"
    write(contcar, atoms, format="vasp")
    (calc_dir / "OUTCAR").write_text("ZVAL = 11.00\n")
    frame = pd.DataFrame(
        {
            "Name": ["Cu-slabs-111-clean"],
            "Path": [str(calc_dir)],
            "contcar_path": [str(contcar)],
            "Cu": [2],
        }
    )
    path = tmp_path / "frame.pkl"
    frame.to_pickle(path)

    wrapped = (
        CalculationFrame.from_pickle(str(path), config=ProjectWorkflowConfig(adsorbates=frozenset({"CO"})))
        .ensure_structures()
        .classify_records()
        .add_file_status()
    )

    assert wrapped.dataframe.loc[0, "CONTCAR"].__class__.__name__ == "Atoms"
    assert wrapped.dataframe.loc[0, "record_kind"] == "surface"
    assert wrapped.dataframe.loc[0, "workflow_facet"] == "111"
    assert bool(wrapped.dataframe.loc[0, "has_outcar"]) is True
    assert len(wrapped.audit_log) == 3


def test_calculation_frame_uses_configured_surface_reference_rules() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Ni-211-clean-1x1", "Ni-211-clean-1x1-NO3-1"],
            "Formula": ["Ni4", "Ni4NO3"],
            "E": [-10.0, -20.0],
            "Ni": [4, 4],
            "N": [0, 1],
            "O": [0, 3],
        }
    )
    config = ProjectWorkflowConfig(adsorbate_tokens=("NO3",), adsorbate_elements=("N", "O"))

    wrapped = CalculationFrame(frame, config=config).assign_surface_references()
    row = wrapped.dataframe.loc[wrapped.dataframe["Name"].eq("Ni-211-clean-1x1-NO3-1")].iloc[0]

    assert row["adsorbate"] == "NO3"
    assert row["surface_ref_name"] == "Ni-211-clean-1x1"
    assert wrapped.audit_log[-1]["step"] == "assign_surface_references"


def test_calculation_frame_cleans_phase_candidates_with_configured_rules() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["M-clean", "M-clean-backup"],
            "Path": ["/calc/M-clean", "/calc/M-clean-backup"],
            "M": [4, 4],
        }
    )
    config = ProjectWorkflowConfig(
        phase_candidate_validation_rules={
            "excluded_name_substrings": {"backup_candidate": ["backup"]},
        }
    )

    wrapped = CalculationFrame(frame, config=config).clean_phase_candidates(allowed_elements=("M",))

    assert wrapped.dataframe["Name"].tolist() == ["M-clean"]
    assert "phase_input_cleaning_audit" in wrapped.dataframe.attrs


def test_workflow_operation_accepts_project_config_for_reference_assignment() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Ni-211-clean-1x1", "Ni-211-clean-1x1-NO3-1"],
            "Formula": ["Ni4", "Ni4NO3"],
            "E": [-10.0, -20.0],
            "Ni": [4, 4],
            "N": [0, 1],
            "O": [0, 3],
        }
    )

    result = apply_operation(
        frame,
        {
            "kind": "derive_recipe_adsorption",
            "workflow_config": {
                "adsorbate_tokens": ["NO3"],
                "adsorbate_elements": ["N", "O"],
            },
            "gas_reference_values": {"NO3gas": -5.0},
            "recipes": {"NO3": {"basis": "N", "gas_refs": {"NO3gas": 1.0}}},
        },
    )
    row = result.loc[result["Name"].eq("Ni-211-clean-1x1-NO3-1")].iloc[0]

    assert row["adsorbate"] == "NO3"
    assert row["surface_ref_name"] == "Ni-211-clean-1x1"
    assert row["n_NO3_adsorbates"] == 1
    assert row["E_ads_NO3_total_eV"] == -5.0


def test_manifest_builder_preserves_flat_keys_and_adds_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "table.csv"
    payload = legacy_manifest_payload(
        {"table": output},
        {"table": 3},
        config={"systems": ["CuGa"]},
        validation={"status": "ready"},
        extra={"legacy": "kept"},
    )

    assert payload["table"] == str(output)
    assert payload["rows"] == {"table": 3}
    assert payload["artifacts"]["table"]["rows"] == 3
    assert payload["config"] == {"systems": ["CuGa"]}
    assert payload["validation"] == {"status": "ready"}
    assert payload["legacy"] == "kept"


def test_workflow_stage_manifest_records_files_tables_and_audit(tmp_path: Path) -> None:
    table_path = tmp_path / "cleaned.csv"
    source_path = tmp_path / "source.csv"
    frame = pd.DataFrame({"Name": ["kept", "dropped"]})
    audit = pd.DataFrame(
        {
            "status": ["kept", "dropped"],
            "reason_code": ["", "backup_candidate"],
        }
    )

    manifest = write_workflow_stage_manifest(
        tmp_path / "stage_manifest.json",
        stage_id="demo_stage",
        files={"cleaned": table_path},
        tables={"cleaned": frame},
        audit_ledger=audit,
        config={"tokens": ("CO", "HCOO")},
        source_inputs={"source": source_path},
        validation={"status": "ready"},
    )
    loaded = json.loads((tmp_path / "stage_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "onepiece.workflow_stage_manifest.v1"
    assert loaded["stage_id"] == "demo_stage"
    assert loaded["artifacts"]["cleaned"]["rows"] == 2
    assert loaded["table_rows"] == {"cleaned": 2}
    assert loaded["audit"]["status_counts"] == {"dropped": 1, "kept": 1}
    assert loaded["audit"]["drop_reason_counts"]["backup_candidate"] == 1
    assert loaded["config"]["tokens"] == ["CO", "HCOO"]
    assert loaded["source_inputs"] == {"source": str(source_path)}


def test_file_status_columns_report_missing_files(tmp_path: Path) -> None:
    calc_dir = tmp_path / "calc"
    calc_dir.mkdir()
    (calc_dir / "CONTCAR").write_text("placeholder")
    result = add_file_status_columns(pd.DataFrame({"Name": ["calc"], "Path": [str(calc_dir)]}))

    assert bool(result.loc[0, "has_contcar"]) is True
    assert bool(result.loc[0, "has_outcar"]) is False
    assert result.loc[0, "file_status"] == "partial"
    assert "OUTCAR" in result.loc[0, "missing_files"]
