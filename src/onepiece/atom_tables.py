from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

from onepiece.frame_utils import ensure_name_index, row_name

DEFAULT_MAPPING_MODE = "surface_same_index_adsorbate_last_element_reference"
GAS_REFERENCE_STAT_COLUMNS = [
    "adsorbate",
    "element",
    "gas_reference_name",
    "n_gas_atoms_element",
    "gas_reference_status",
    "gas_atomic_charge_mean_e",
    "gas_atomic_charge_min_e",
    "gas_atomic_charge_max_e",
    "gas_atomic_charge_std_e",
    "gas_integrated_electron_population_mean",
    "gas_integrated_electron_population_min",
    "gas_integrated_electron_population_max",
    "gas_integrated_electron_population_std",
    "gas_atomic_magnetic_moment_mean_muB",
    "gas_atomic_magnetic_moment_min_muB",
    "gas_atomic_magnetic_moment_max_muB",
    "gas_atomic_magnetic_moment_std_muB",
]


def ensure_structure_columns(
    frame: pd.DataFrame,
    *,
    preferred_structure_column: str = "CONTCAR",
    fallback_columns: Sequence[str] = ("struc", "structure", "atoms"),
    path_column: str = "contcar_path",
) -> pd.DataFrame:
    """Ensure a preferred atom-order structure column exists for atom tables.

    The returned frame always contains ``preferred_structure_column`` and a
    ``structure_source_for_atom_table`` status column. The preferred structure is
    kept when present, read from ``path_column`` when possible, and otherwise
    copied from the first available fallback structure column.
    """
    df = frame.copy()
    if preferred_structure_column not in df.columns:
        df[preferred_structure_column] = None
    df["structure_source_for_atom_table"] = "missing"

    for index, row in df.iterrows():
        preferred = row.get(preferred_structure_column)
        if _is_atoms(preferred):
            df.at[index, "structure_source_for_atom_table"] = preferred_structure_column
            continue

        contcar_path = _path_or_none(row.get(path_column))
        if contcar_path is not None and contcar_path.exists():
            try:
                atoms = read(contcar_path, index=-1)
            except Exception:
                atoms = None
            if _is_atoms(atoms):
                df.at[index, preferred_structure_column] = atoms
                df.at[index, "structure_source_for_atom_table"] = path_column
                continue

        for column in fallback_columns:
            fallback = row.get(column)
            if _is_atoms(fallback):
                df.at[index, preferred_structure_column] = fallback.copy()
                df.at[index, "structure_source_for_atom_table"] = f"fallback:{column}"
                break

    return df


def atomic_properties_multiindex_table(
    frame: pd.DataFrame,
    *,
    structure_column: str = "CONTCAR",
) -> pd.DataFrame:
    """Expand calculation-level atom arrays into a MultiIndex atom table."""
    df = ensure_name_index(frame)
    rows: list[dict[str, object]] = []
    metadata_columns = [
        "record_kind",
        "adsorbate",
        "surface_ref_name",
        "system",
        "facet",
        "charge_source_used",
        "charge_coordinate_match",
        "charge_coordinate_max_delta_A",
        "charge_coordinate_max_delta_raw_A",
        "charge_coordinate_validation_mode",
        "charge_quality_status",
        "charge_quality_issue",
        "charge_quality_system",
        "structure_source_for_atom_table",
    ]

    for _, row in df.iterrows():
        name = row_name(row)
        atoms = _row_atoms(row, structure_column)
        populations = _as_array(row.get("integrated_electron_populations"))
        charges = _as_array(row.get("atomic_charges"))
        magnetic_moments = _as_array(row.get("atomic_magnetic_moments"))
        if atoms is None:
            continue
        positions = np.asarray(atoms.get_positions(), dtype=float)
        symbols = atoms.get_chemical_symbols()
        for atom_index, symbol in enumerate(symbols):
            record: dict[str, object] = {
                "calculation_name": name,
                "atom_index": int(atom_index),
                "element": symbol,
                "x": float(positions[atom_index, 0]),
                "y": float(positions[atom_index, 1]),
                "z": float(positions[atom_index, 2]),
                "integrated_electron_population": _array_value(populations, atom_index),
                "atomic_charge_e": _array_value(charges, atom_index),
                "atomic_magnetic_moment_muB": _array_value(magnetic_moments, atom_index),
            }
            for column in metadata_columns:
                if column in row.index:
                    record[column] = row.get(column)
            rows.append(record)

    return _multiindex_table(
        rows,
        columns=[
            "calculation_name",
            "atom_index",
            "element",
            "x",
            "y",
            "z",
            "integrated_electron_population",
            "atomic_charge_e",
            "atomic_magnetic_moment_muB",
            *metadata_columns,
        ],
    )


def gas_reference_element_statistics(
    frame: pd.DataFrame,
    *,
    adsorbate_column: str = "adsorbate",
    structure_column: str = "CONTCAR",
    gas_reference_by_adsorbate: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize gas references by adsorbate and element.

    Duplicate elements are reduced to element-specific means while retaining
    min/max/std/count diagnostics.
    """
    df = ensure_name_index(frame)
    gas_rows = _canonical_gas_reference_rows(
        df,
        adsorbate_column=adsorbate_column,
        structure_column=structure_column,
        gas_reference_by_adsorbate=gas_reference_by_adsorbate,
    )
    rows: list[dict[str, object]] = []
    for adsorbate, row in gas_rows.items():
        atoms = _row_atoms(row, structure_column)
        if atoms is None:
            continue
        charges = _as_array(row.get("atomic_charges"))
        populations = _as_array(row.get("integrated_electron_populations"))
        magnetic = _as_array(row.get("atomic_magnetic_moments"))
        symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
        for element in sorted(set(symbols)):
            mask = symbols == element
            charge_values = _masked_values(charges, mask)
            population_values = _masked_values(populations, mask)
            magnetic_values = _masked_values(magnetic, mask)
            record: dict[str, object] = {
                "adsorbate": adsorbate,
                "element": element,
                "gas_reference_name": row_name(row),
                "n_gas_atoms_element": int(mask.sum()),
                "gas_reference_status": _gas_reference_status(charge_values, population_values, magnetic_values),
            }
            record.update(_stat_columns("gas_atomic_charge", charge_values, unit_suffix="_e"))
            record.update(_stat_columns("gas_integrated_electron_population", population_values))
            record.update(_stat_columns("gas_atomic_magnetic_moment", magnetic_values, unit_suffix="_muB"))
            rows.append(record)

    if not rows:
        return _gas_reference_statistics_table(rows)
    return _gas_reference_statistics_table(rows).sort_index()


def atom_reference_map(
    frame: pd.DataFrame,
    *,
    mapping_mode: str = DEFAULT_MAPPING_MODE,
    structure_column: str = "CONTCAR",
    adsorbate_column: str = "adsorbate",
    gas_reference_by_adsorbate: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build a per-atom reference map without using gas atom ordering."""
    df = ensure_name_index(frame)
    gas_rows = _canonical_gas_reference_rows(
        df,
        adsorbate_column=adsorbate_column,
        structure_column=structure_column,
        gas_reference_by_adsorbate=gas_reference_by_adsorbate,
    )
    structure_by_name = {
        row_name(row): _row_atoms(row, structure_column)
        for _, row in df.iterrows()
        if _row_atoms(row, structure_column) is not None
    }
    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        name = row_name(row)
        atoms = _row_atoms(row, structure_column)
        if atoms is None:
            continue
        symbols = atoms.get_chemical_symbols()
        adsorbate = _adsorbate_label(row, adsorbate_column=adsorbate_column)
        record_kind = str(row.get("record_kind", "") or "")
        is_gas_reference = _is_gas_reference_row(row)
        is_adsorbed = _is_adsorbed_row(row, adsorbate=adsorbate, is_gas_reference=is_gas_reference)
        surface_ref_name = _clean_text(row.get("surface_ref_name"))

        gas_atoms = _row_atoms(gas_rows.get(adsorbate, pd.Series(dtype=object)), structure_column)
        gas_reference_name = row_name(gas_rows[adsorbate]) if adsorbate in gas_rows else None
        n_adsorbate_atoms = len(gas_atoms) if gas_atoms is not None else _fallback_adsorbate_count(row, len(symbols))
        surface_cutoff = max(0, len(symbols) - int(n_adsorbate_atoms)) if is_adsorbed else len(symbols)

        for atom_index, element in enumerate(symbols):
            common = {
                "calculation_name": name,
                "atom_index": int(atom_index),
                "element": element,
                "record_kind": record_kind,
                "adsorbate": adsorbate,
                "surface_ref_name": surface_ref_name,
                "mapping_mode": mapping_mode,
            }
            if is_gas_reference:
                rows.append(
                    {
                        **common,
                        "atom_role": "gas_reference",
                        "reference_kind": "self",
                        "reference_name": name,
                        "reference_atom_index": int(atom_index),
                        "reference_element": element,
                        "mapping_status": "ok",
                        "mapping_issue": "",
                    }
                )
                continue
            if not is_adsorbed:
                role = "surface_reference" if surface_ref_name == name else _non_adsorbed_role(record_kind)
                rows.append(
                    {
                        **common,
                        "atom_role": role,
                        "reference_kind": "self",
                        "reference_name": name,
                        "reference_atom_index": int(atom_index),
                        "reference_element": element,
                        "mapping_status": "ok",
                        "mapping_issue": "",
                    }
                )
                continue
            if atom_index >= surface_cutoff:
                status = "ok"
                issue = ""
                if gas_atoms is None:
                    status = "missing_reference"
                    issue = "missing_gas_reference"
                elif element not in set(gas_atoms.get_chemical_symbols()):
                    status = "missing_gas_element_reference"
                    issue = f"gas reference for {adsorbate} has no {element}"
                rows.append(
                    {
                        **common,
                        "atom_role": "adsorbate",
                        "reference_kind": "gas_element_mean" if status != "missing_reference" else "missing",
                        "reference_name": gas_reference_name,
                        "reference_atom_index": np.nan,
                        "reference_element": element,
                        "mapping_status": status,
                        "mapping_issue": issue,
                    }
                )
                continue

            reference_atoms = structure_by_name.get(surface_ref_name)
            status = "ok"
            issue = ""
            reference_element = None
            if reference_atoms is None:
                status = "missing_reference"
                issue = "missing_surface_reference"
            elif atom_index >= len(reference_atoms):
                status = "missing_reference_atom"
                issue = "surface reference has fewer atoms"
            else:
                reference_element = reference_atoms.get_chemical_symbols()[atom_index]
                if reference_element != element:
                    status = "element_mismatch"
                    issue = f"{element} != {reference_element}"
            rows.append(
                {
                    **common,
                    "atom_role": "surface",
                    "reference_kind": "surface_same_index" if status == "ok" else "missing",
                    "reference_name": surface_ref_name,
                    "reference_atom_index": int(atom_index) if status == "ok" else np.nan,
                    "reference_element": reference_element if reference_element is not None else element,
                    "mapping_status": status,
                    "mapping_issue": issue,
                }
            )

    return _multiindex_table(
        rows,
        columns=[
            "calculation_name",
            "atom_index",
            "element",
            "record_kind",
            "adsorbate",
            "surface_ref_name",
            "atom_role",
            "reference_kind",
            "reference_name",
            "reference_atom_index",
            "reference_element",
            "mapping_mode",
            "mapping_status",
            "mapping_issue",
        ],
    )


def atom_reference_delta_table(
    atom_properties: pd.DataFrame,
    reference_map: pd.DataFrame,
    gas_element_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Join atom properties to mapped references and compute per-atom deltas."""
    props = _ensure_atom_index(atom_properties)
    refs = _ensure_atom_index(reference_map)
    gas = _ensure_gas_stats_index(gas_element_stats)

    rows: list[dict[str, object]] = []
    for index, mapping in refs.iterrows():
        if index not in props.index:
            continue
        prop = props.loc[index]
        record = {column: mapping.get(column) for column in refs.columns if column not in {"calculation_name", "atom_index"}}
        record.update(
            {
                "calculation_name": index[0],
                "atom_index": int(index[1]),
                "atomic_charge_e": prop.get("atomic_charge_e", np.nan),
                "integrated_electron_population": prop.get("integrated_electron_population", np.nan),
                "atomic_magnetic_moment_muB": prop.get("atomic_magnetic_moment_muB", np.nan),
            }
        )
        for column in (
            "charge_quality_status",
            "charge_quality_issue",
            "charge_quality_system",
            "charge_coordinate_max_delta_A",
            "charge_coordinate_max_delta_raw_A",
            "charge_coordinate_validation_mode",
            "charge_coordinate_match",
        ):
            if column in props.columns and column not in record:
                record[column] = prop.get(column)
        reference_values = _reference_values(mapping, props, gas)
        record.update(reference_values)
        record["delta_atomic_charge_vs_ref_e"] = _delta(
            record["atomic_charge_e"],
            record["reference_atomic_charge_e"],
        )
        record["delta_integrated_electron_population_vs_ref"] = _delta(
            record["integrated_electron_population"],
            record["reference_integrated_electron_population"],
        )
        record["delta_atomic_magnetic_moment_vs_ref_muB"] = _delta(
            record["atomic_magnetic_moment_muB"],
            record["reference_atomic_magnetic_moment_muB"],
        )
        record["delta_reference_valid"] = (
            str(record.get("mapping_status")) == "ok"
            and _charge_quality_ok(record.get("charge_quality_status"))
            and any(
                pd.notna(record.get(column))
                for column in (
                    "reference_atomic_charge_e",
                    "reference_integrated_electron_population",
                    "reference_atomic_magnetic_moment_muB",
                )
            )
        )
        rows.append(record)

    return _multiindex_table(
        rows,
        columns=[
            "calculation_name",
            "atom_index",
            *[c for c in refs.columns if c not in {"calculation_name", "atom_index"}],
            "atomic_charge_e",
            "reference_atomic_charge_e",
            "delta_atomic_charge_vs_ref_e",
            "integrated_electron_population",
            "reference_integrated_electron_population",
            "delta_integrated_electron_population_vs_ref",
            "atomic_magnetic_moment_muB",
            "reference_atomic_magnetic_moment_muB",
            "delta_atomic_magnetic_moment_vs_ref_muB",
            "charge_quality_status",
            "charge_quality_issue",
            "charge_quality_system",
            "charge_coordinate_max_delta_A",
            "charge_coordinate_max_delta_raw_A",
            "charge_coordinate_validation_mode",
            "charge_coordinate_match",
            "delta_reference_valid",
        ],
    )


def summarize_atom_reference_deltas(atom_delta_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate atom-reference deltas to one row per calculation."""
    table = _ensure_atom_index(atom_delta_table)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "Name",
                "record_kind",
                "adsorbate",
                "surface_ref_name",
                "charge_quality_status",
                "charge_quality_issue",
                "charge_coordinate_max_delta_A",
                "charge_coordinate_max_delta_raw_A",
                "n_atoms",
                "n_surface_atoms",
                "n_adsorbate_atoms",
                "n_mapped_atoms",
                "n_mapping_issues",
                "n_charge_quality_ok_atoms",
                "n_charge_quality_issue_atoms",
                "adsorbate_charge_delta_sum_e",
                "surface_charge_delta_sum_e",
                "charge_delta_balance_residual_e",
                "adsorbate_magnetic_moment_delta_sum_muB",
                "surface_magnetic_moment_delta_sum_muB",
                "max_abs_atom_charge_delta_e",
                "max_abs_atom_magnetic_delta_muB",
            ]
        )
    rows: list[dict[str, object]] = []
    for name, group in table.groupby(level="calculation_name", sort=False):
        adsorbate_mask = group["atom_role"].astype(str).eq("adsorbate")
        surface_mask = group["atom_role"].astype(str).eq("surface")
        mapping_ok = group["mapping_status"].astype(str).eq("ok")
        if "charge_quality_status" in group:
            quality_ok = group["charge_quality_status"].map(_charge_quality_ok)
        else:
            quality_ok = pd.Series(True, index=group.index)
        valid_for_charge_summary = mapping_ok & quality_ok
        ads_charge = pd.to_numeric(group.loc[adsorbate_mask & valid_for_charge_summary, "delta_atomic_charge_vs_ref_e"], errors="coerce")
        surf_charge = pd.to_numeric(group.loc[surface_mask & valid_for_charge_summary, "delta_atomic_charge_vs_ref_e"], errors="coerce")
        ads_mag = pd.to_numeric(group.loc[adsorbate_mask & valid_for_charge_summary, "delta_atomic_magnetic_moment_vs_ref_muB"], errors="coerce")
        surf_mag = pd.to_numeric(group.loc[surface_mask & valid_for_charge_summary, "delta_atomic_magnetic_moment_vs_ref_muB"], errors="coerce")
        all_charge = pd.to_numeric(group.loc[valid_for_charge_summary, "delta_atomic_charge_vs_ref_e"], errors="coerce")
        all_mag = pd.to_numeric(group.loc[valid_for_charge_summary, "delta_atomic_magnetic_moment_vs_ref_muB"], errors="coerce")
        first = group.iloc[0]
        ads_charge_sum = float(ads_charge.sum(skipna=True)) if not ads_charge.empty else 0.0
        surf_charge_sum = float(surf_charge.sum(skipna=True)) if not surf_charge.empty else 0.0
        charge_quality_status = first.get("charge_quality_status", "ok")
        rows.append(
            {
                "Name": name,
                "record_kind": first.get("record_kind"),
                "adsorbate": first.get("adsorbate"),
                "surface_ref_name": first.get("surface_ref_name"),
                "charge_quality_status": charge_quality_status,
                "charge_quality_issue": first.get("charge_quality_issue", ""),
                "charge_coordinate_max_delta_A": first.get("charge_coordinate_max_delta_A", np.nan),
                "charge_coordinate_max_delta_raw_A": first.get("charge_coordinate_max_delta_raw_A", np.nan),
                "n_atoms": int(len(group)),
                "n_surface_atoms": int(surface_mask.sum()),
                "n_adsorbate_atoms": int(adsorbate_mask.sum()),
                "n_mapped_atoms": int(mapping_ok.sum()),
                "n_mapping_issues": int((~mapping_ok).sum()),
                "n_charge_quality_ok_atoms": int(quality_ok.sum()),
                "n_charge_quality_issue_atoms": int((~quality_ok).sum()),
                "adsorbate_charge_delta_sum_e": ads_charge_sum,
                "surface_charge_delta_sum_e": surf_charge_sum,
                "charge_delta_balance_residual_e": ads_charge_sum + surf_charge_sum,
                "adsorbate_magnetic_moment_delta_sum_muB": float(ads_mag.sum(skipna=True)) if not ads_mag.empty else 0.0,
                "surface_magnetic_moment_delta_sum_muB": float(surf_mag.sum(skipna=True)) if not surf_mag.empty else 0.0,
                "max_abs_atom_charge_delta_e": float(all_charge.abs().max(skipna=True)) if all_charge.notna().any() else np.nan,
                "max_abs_atom_magnetic_delta_muB": float(all_mag.abs().max(skipna=True)) if all_mag.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _multiindex_table(rows: list[dict[str, object]], *, columns: Sequence[str]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=list(dict.fromkeys(columns)))
    if table.empty:
        table = pd.DataFrame(columns=list(dict.fromkeys(columns)))
    return table.set_index(["calculation_name", "atom_index"], drop=False)


def _gas_reference_statistics_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=GAS_REFERENCE_STAT_COLUMNS)
    if table.empty:
        table = pd.DataFrame(columns=GAS_REFERENCE_STAT_COLUMNS)
    return table.set_index(["adsorbate", "element"], drop=False)


def _ensure_atom_index(table: pd.DataFrame) -> pd.DataFrame:
    if list(table.index.names) == ["calculation_name", "atom_index"]:
        return table
    return table.set_index(["calculation_name", "atom_index"], drop=False)


def _ensure_gas_stats_index(table: pd.DataFrame) -> pd.DataFrame:
    if list(table.index.names) == ["adsorbate", "element"]:
        return table
    return table.set_index(["adsorbate", "element"], drop=False)


def _row_atoms(row: pd.Series | None, structure_column: str) -> Atoms | None:
    if row is None:
        return None
    value = row.get(structure_column)
    return value if _is_atoms(value) else None


def _is_atoms(value: object) -> bool:
    return value.__class__.__name__ == "Atoms"


def _path_or_none(value: object) -> Path | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    return Path(text) if text else None


def _as_array(value: object) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    if isinstance(value, list | tuple):
        return np.asarray(value, dtype=float)
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, list | tuple):
            return np.asarray(parsed, dtype=float)
    return None


def _array_value(values: np.ndarray | None, index: int) -> float:
    if values is None or index >= values.shape[0]:
        return np.nan
    return float(values[index])


def _masked_values(values: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    if values is None or values.shape[0] != mask.shape[0]:
        return np.asarray([], dtype=float)
    selected = values[mask]
    return selected[np.isfinite(selected)]


def _stat_columns(prefix: str, values: np.ndarray, *, unit_suffix: str = "") -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean{unit_suffix}": np.nan,
            f"{prefix}_min{unit_suffix}": np.nan,
            f"{prefix}_max{unit_suffix}": np.nan,
            f"{prefix}_std{unit_suffix}": np.nan,
        }
    return {
        f"{prefix}_mean{unit_suffix}": float(values.mean()),
        f"{prefix}_min{unit_suffix}": float(values.min()),
        f"{prefix}_max{unit_suffix}": float(values.max()),
        f"{prefix}_std{unit_suffix}": float(values.std(ddof=0)),
    }


def _gas_reference_status(
    charge_values: np.ndarray,
    population_values: np.ndarray,
    magnetic_values: np.ndarray,
) -> str:
    if charge_values.size and population_values.size:
        return "ok"
    if population_values.size or charge_values.size or magnetic_values.size:
        return "partial"
    return "missing_values"


def _canonical_gas_reference_rows(
    frame: pd.DataFrame,
    *,
    adsorbate_column: str,
    structure_column: str,
    gas_reference_by_adsorbate: Mapping[str, str] | None,
) -> dict[str, pd.Series]:
    df = ensure_name_index(frame)
    candidates: dict[str, list[pd.Series]] = {}
    for _, row in df.iterrows():
        if not _is_gas_reference_row(row):
            continue
        adsorbate = _adsorbate_label(row, adsorbate_column=adsorbate_column)
        if not adsorbate:
            continue
        if _row_atoms(row, structure_column) is None:
            continue
        candidates.setdefault(adsorbate, []).append(row)

    selected: dict[str, pd.Series] = {}
    overrides = dict(gas_reference_by_adsorbate or {})
    for adsorbate, rows in candidates.items():
        override = overrides.get(adsorbate)
        if override:
            for row in rows:
                if row_name(row) == override:
                    selected[adsorbate] = row
                    break
        if adsorbate not in selected:
            selected[adsorbate] = sorted(rows, key=lambda row: _gas_reference_priority(row, adsorbate))[0]
    return selected


def _gas_reference_priority(row: pd.Series, adsorbate: str) -> tuple[int, int, str]:
    name = row_name(row)
    lowered = name.lower()
    exact = f"gasphases-{adsorbate}".lower()
    priority = 0 if lowered == exact else 1
    if any(token in lowered for token in ("dftu", "b3lyp", "opt", "trans")):
        priority += 5
    return priority, len(name), name


def _is_gas_reference_row(row: pd.Series) -> bool:
    record_kind = str(row.get("record_kind", "") or "").lower()
    record_class = str(row.get("record_class", "") or "").lower()
    name = row_name(row).lower()
    path = str(row.get("Path", "") or "").lower()
    if record_kind in {"gas", "gas_reference"} or record_class in {"gas", "gas_reference", "gas-phase", "gas_phase"}:
        return True
    return "gasphases" in name or "gasphase" in name or "/gas/" in path or "/gasphases/" in path


def _adsorbate_label(row: pd.Series, *, adsorbate_column: str) -> str:
    value = _clean_text(row.get(adsorbate_column))
    if value:
        return value
    name = row_name(row)
    match = re.search(r"gasphases[-_/]([A-Za-z0-9]+)", name)
    return match.group(1) if match else ""


def _is_adsorbed_row(row: pd.Series, *, adsorbate: str, is_gas_reference: bool) -> bool:
    if is_gas_reference or not adsorbate:
        return False
    record_kind = str(row.get("record_kind", "") or "").lower()
    if record_kind in {"adsorbed_surface", "adsorbate"}:
        return True
    path = str(row.get("Path", "") or "").lower()
    return "/slabs/" in path and adsorbate not in {"nan", "none"}


def _non_adsorbed_role(record_kind: str) -> str:
    lowered = str(record_kind).lower()
    if lowered == "bulk":
        return "bulk"
    if lowered == "gas":
        return "gas_reference"
    if lowered == "surface":
        return "surface_reference"
    return "unknown"


def _fallback_adsorbate_count(row: pd.Series, natoms: int) -> int:
    indices = row.get("adsorbate_atom_indices")
    if isinstance(indices, list | tuple):
        return len(indices)
    if isinstance(indices, str) and indices.startswith("["):
        try:
            parsed = ast.literal_eval(indices)
        except (SyntaxError, ValueError):
            parsed = []
        if isinstance(parsed, list | tuple):
            return len(parsed)
    formula_count = _formula_atom_count(_adsorbate_label(row, adsorbate_column="adsorbate"))
    return min(formula_count, natoms)


def _formula_atom_count(formula: str) -> int:
    if not formula:
        return 0
    count = 0
    for match in re.finditer(r"[A-Z][a-z]?(\d*)", formula):
        suffix = match.group(1)
        count += int(suffix) if suffix else 1
    return count


def _reference_values(
    mapping: pd.Series,
    properties: pd.DataFrame,
    gas_stats: pd.DataFrame,
) -> dict[str, float]:
    reference_kind = str(mapping.get("reference_kind", ""))
    if reference_kind in {"self", "surface_same_index"}:
        reference_name = mapping.get("reference_name")
        reference_atom_index = mapping.get("reference_atom_index")
        if not reference_name or pd.isna(reference_atom_index):
            return _empty_reference_values()
        key = (str(reference_name), int(reference_atom_index))
        if key not in properties.index:
            return _empty_reference_values()
        row = properties.loc[key]
        return {
            "reference_atomic_charge_e": row.get("atomic_charge_e", np.nan),
            "reference_integrated_electron_population": row.get("integrated_electron_population", np.nan),
            "reference_atomic_magnetic_moment_muB": row.get("atomic_magnetic_moment_muB", np.nan),
        }
    if reference_kind == "gas_element_mean":
        key = (str(mapping.get("adsorbate")), str(mapping.get("reference_element")))
        if key not in gas_stats.index:
            return _empty_reference_values()
        row = gas_stats.loc[key]
        return {
            "reference_atomic_charge_e": row.get("gas_atomic_charge_mean_e", np.nan),
            "reference_integrated_electron_population": row.get("gas_integrated_electron_population_mean", np.nan),
            "reference_atomic_magnetic_moment_muB": row.get("gas_atomic_magnetic_moment_mean_muB", np.nan),
        }
    return _empty_reference_values()


def _empty_reference_values() -> dict[str, float]:
    return {
        "reference_atomic_charge_e": np.nan,
        "reference_integrated_electron_population": np.nan,
        "reference_atomic_magnetic_moment_muB": np.nan,
    }


def _delta(value: object, reference: object) -> float:
    value_num = pd.to_numeric(value, errors="coerce")
    ref_num = pd.to_numeric(reference, errors="coerce")
    if pd.isna(value_num) or pd.isna(ref_num):
        return np.nan
    return float(value_num - ref_num)


def _charge_quality_ok(value: object) -> bool:
    text = _clean_text(value).lower()
    return text in {"", "ok"}


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    return "" if text.lower() in {"nan", "none"} else text
