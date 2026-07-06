from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from onepiece.workflow_config import ProjectWorkflowConfig


DEFAULT_ADSORBATES = frozenset(
    {
        "CO2",
        "COOH",
        "HCOO",
        "H2COO",
        "HCOOH",
        "H2CO",
        "H2COH",
        "HCO",
        "CO",
        "CH3O",
        "CH3OH",
        "CH3",
        "CH2",
        "OH",
        "O",
        "Ox2",
        "H",
    }
)


def infer_adsorbate(
    path_value: object,
    name_value: object = "",
    *,
    adsorbates: Iterable[str] = DEFAULT_ADSORBATES,
) -> object:
    """Infer an adsorbate label from path parts first, then calculation name."""
    adsorbate_set = {str(value) for value in adsorbates}
    parts = [part for part in Path(str(path_value or "")).parts]
    for part in parts:
        if part in adsorbate_set:
            return part
    name = str(name_value or "")
    for adsorbate in sorted(adsorbate_set, key=len, reverse=True):
        if f"-{adsorbate}-" in name or name.endswith(f"-{adsorbate}"):
            return adsorbate
    return np.nan


def infer_record_kind(row: pd.Series) -> str:
    """Classify gas/reference rows before adsorbates, surfaces, and bulk rows."""
    path = str(row.get("Path", "") or "")
    name = str(row.get("Name", "") or "")
    record_class = str(row.get("record_class", "") or "").lower()
    existing_kind = str(row.get("record_kind", "") or "").lower()
    path_lower = path.lower()
    name_lower = name.lower()
    if (
        existing_kind in {"gas", "gas_reference"}
        or record_class in {"gas", "gas_reference", "gas-phase", "gas_phase"}
        or "gasphases" in name_lower
        or "gasphase" in name_lower
        or "/gas/" in path_lower
        or "/gasphases/" in path_lower
    ):
        return "gas"
    adsorbate = row.get("adsorbate")
    if not _is_missing_scalar(adsorbate) and str(adsorbate):
        return "adsorbed_surface"
    if "/slabs/" in path_lower:
        return "surface"
    if "/bulk" in path_lower:
        return "bulk"
    return "other"


def add_workflow_classification(
    frame: pd.DataFrame,
    *,
    config: ProjectWorkflowConfig | None = None,
    adsorbate_column: str = "adsorbate",
) -> pd.DataFrame:
    """Add canonical ``adsorbate`` and ``record_kind`` columns."""
    df = frame.copy()
    active_adsorbates = (
        config.active_adsorbate_tokens()
        if config is not None and (config.adsorbate_tokens or config.adsorbates)
        else DEFAULT_ADSORBATES
    )
    inferred_adsorbates = df.apply(
        lambda row: infer_adsorbate(row.get("Path"), row.get("Name"), adsorbates=active_adsorbates),
        axis=1,
    )
    if adsorbate_column in df:
        df[adsorbate_column] = df[adsorbate_column].where(df[adsorbate_column].notna(), inferred_adsorbates)
    else:
        df[adsorbate_column] = inferred_adsorbates
    df["record_kind"] = df.apply(infer_record_kind, axis=1)
    df["workflow_system"] = df.apply(infer_quality_system, axis=1)
    df["workflow_facet"] = df.apply(infer_facet, axis=1)
    return df


def infer_quality_system(row: pd.Series) -> str:
    """Infer a compact material-system label from element counts and row kind."""
    if str(row.get("record_kind", "")).lower() == "gas":
        return "gas"
    elements = {
        element
        for element in ("Cu", "Ni", "Ga", "Zn", "Mg", "O", "Ca", "Al", "Na", "K")
        if numeric_value(row.get(element)) > 0
    }
    if {"Cu", "Ga"} <= elements:
        return "CuGa"
    if {"Ni", "Mg", "O"} <= elements:
        return "NiMgO"
    if {"Cu"} == elements:
        return "Cu"
    if {"Ni"} == elements:
        return "Ni"
    if {"Ca", "O"} <= elements:
        return "CaO"
    if {"Zn", "O"} <= elements:
        return "ZnO"
    if {"Ga", "O"} <= elements:
        return "GaOx"
    return "other"


def infer_facet(row: pd.Series) -> str:
    """Infer a simple facet/index label from calculation path or name."""
    text = f"{row.get('Path', '')}/{row.get('Name', '')}"
    parts = [part for part in Path(str(text)).parts if part]
    for part in parts:
        if part in {"100", "110", "111", "211", "221", "001", "201", "112"}:
            return part
    return ""


def numeric_value(value: object) -> float:
    result = pd.to_numeric(value, errors="coerce")
    return float(result) if pd.notna(result) else 0.0


def calculation_directory(
    row: pd.Series,
    *,
    acf_path_column: str = "acf_path",
    calculation_path_column: str = "Path",
) -> Path | None:
    """Return the directory that should contain calculation files for a row."""
    for column in (acf_path_column, calculation_path_column):
        value = row.get(column)
        if _is_missing_scalar(value):
            continue
        path = Path(str(value))
        if column == acf_path_column and path.name:
            return path.parent
        return path if path.suffix == "" else path.parent
    return None


def add_file_status_columns(
    frame: pd.DataFrame,
    *,
    calculation_path_column: str = "Path",
    filenames: tuple[str, ...] = ("CONTCAR", "OUTCAR", "ACF.dat", "DOSCAR", "CHGCAR", "POTCAR"),
) -> pd.DataFrame:
    """Attach file-presence and missing-file audit columns."""
    df = frame.copy()
    missing_files: list[str] = []
    for filename in filenames:
        column = f"has_{filename.lower().replace('.', '_')}"
        flags = []
        for _, row in df.iterrows():
            directory = calculation_directory(row, calculation_path_column=calculation_path_column)
            flags.append(bool(directory and (directory / filename).exists()))
        df[column] = flags
    for _, row in df.iterrows():
        missing = []
        directory = calculation_directory(row, calculation_path_column=calculation_path_column)
        for filename in filenames:
            if not bool(directory and (directory / filename).exists()):
                missing.append(filename)
        missing_files.append(";".join(missing))
    df["missing_files"] = missing_files
    df["file_status"] = np.where(df["missing_files"].eq(""), "ok", "partial")
    return df


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool | np.bool_):
        return bool(missing)
    return False
