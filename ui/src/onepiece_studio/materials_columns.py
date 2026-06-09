from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from onepiece.adsorption import DEFAULT_ADSORBATE_TOKENS, row_element_count_map


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    name: str
    purpose: str
    columns: tuple[str, ...]


COLUMN_PROFILES: tuple[ColumnProfile, ...] = (
    ColumnProfile(
        name="Overview",
        purpose="Fast identity, dataset, formula, energy and quality scan.",
        columns=(
            "record_type",
            "dataset",
            "Name",
            "Formula",
            "composition_summary",
            "E",
            "energy_per_atom",
            "fmax",
            "quality_flag",
            "source_hdf",
            "source_row",
        ),
    ),
    ColumnProfile(
        name="Surface Stability",
        purpose="Compare surface candidates, coverage, area-normalized energies and clean references.",
        columns=(
            "dataset",
            "Name",
            "Formula",
            "hkl",
            "slabsize",
            "layers",
            "Monolayer_alloy",
            "coverage_label",
            "Area",
            "form_G",
            "form_G_per_Area",
            "form_G_per_alloy",
            "surface_ref",
            "is_clean",
            "quality_flag",
        ),
    ),
    ColumnProfile(
        name="Phase Diagram",
        purpose="Inputs and labels needed to build bulk and surface phase diagrams.",
        columns=(
            "record_type",
            "dataset",
            "Name",
            "Formula",
            "phase_label",
            "Ga_percent",
            "Monolayer_alloy",
            "formation_energy_per_atom",
            "formation_energy_per_atom_numeric",
            "form_G_per_Area",
            "muGa",
            "muZn",
            "mu_Ga",
            "mu_Zn",
            "delta_Ga",
            "delta_Cu",
            "Area",
        ),
    ),
    ColumnProfile(
        name="References",
        purpose="Find clean surfaces, bulk references and later adsorption-energy references.",
        columns=(
            "record_type",
            "dataset",
            "Name",
            "Formula",
            "reference_role",
            "reference_key",
            "is_clean",
            "is_adsorbate_like",
            "adsorbate_guess",
            "hkl",
            "slabsize",
            "surface_ref",
            "E",
            "E_ref",
            "adsorbate_reference_mode",
            "adsorbate_integrated_electrons_delta_vs_ref",
            "adsorbate_charge_delta_vs_ref_e",
            "surface_integrated_electrons_delta_vs_ref",
            "surface_net_charge_delta_vs_ref_e",
            "Cu_ref",
            "Ni_ref",
        ),
    ),
    ColumnProfile(
        name="Quality",
        purpose="Check whether calculations should be included, reviewed or excluded.",
        columns=(
            "quality_flag",
            "dataset",
            "Name",
            "Formula",
            "fmax",
            "has_structure",
            "has_energy",
            "has_area",
            "adsorbate_is_dissociated",
            "adsorbate_desorbed",
            "has_overlapping_atoms",
            "has_unphysical_bonds",
            "min_interatomic_distance",
            "min_bond_ratio",
            "timestamp",
            "human_time",
            "kpts",
            "k1",
            "k2",
            "k3",
            "Path",
        ),
    ),
    ColumnProfile(
        name="Structure",
        purpose="Structure, cell, size and local descriptor context.",
        columns=(
            "record_type",
            "dataset",
            "Name",
            "Formula",
            "n_atoms",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
            "Volume",
            "Volume_per_atom",
            "Area",
            "average_Cu_GCN",
            "average_Ga_GCN",
            "average_Cu_charge",
            "average_Ga_charge",
            "min_Cu_charge",
            "min_Ga_charge",
            "layer_count",
            "slab_thickness",
            "vacuum_thickness",
            "mean_coordination",
            "mean_generalized_coordination",
            "adsorption_site",
            "adsorbate_tilt_deg",
            "surface_reconstruction_rmsd",
            "adsorbate_net_charge_e",
            "surface_net_charge_e",
            "charge_balance_residual_e",
            "metal_d_band_center_eV",
            "metal_d_band_filling",
        ),
    ),
    ColumnProfile(
        name="Provenance",
        purpose="Trace rows back to local files, HDF source and calculation metadata.",
        columns=(
            "row_uid",
            "dataset",
            "source_hdf",
            "source_row",
            "Name",
            "Path",
            "human_time",
            "timestamp",
            "files",
            "parameters",
            "constraints",
        ),
    ),
)


def enrich_materials_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    if "row_uid" not in df.columns:
        if {"source_hdf", "source_row"}.issubset(df.columns):
            df["row_uid"] = df["source_hdf"].astype(str) + "::" + df["source_row"].astype(str)
        else:
            df["row_uid"] = df.index.astype(str)

    df["record_type"] = df.apply(_record_type, axis=1)
    df["is_clean"] = _contains(df, "clean")
    df["is_adsorbate_like"] = _contains_any(df, ["CO", "CO2", "H2O", "OH", "O2", "H2", "ads"])
    df["adsorbate_guess"] = df["Name"].map(_guess_adsorbate) if "Name" in df.columns else ""
    df["reference_role"] = df.apply(_reference_role, axis=1)
    df["reference_key"] = df.apply(_reference_key, axis=1)
    df["composition_summary"] = df.apply(_composition_summary, axis=1)
    df["coverage_label"] = df["Monolayer_alloy"].map(_coverage_label) if "Monolayer_alloy" in df else ""
    df["n_atoms"] = df.apply(_n_atoms, axis=1)
    df["has_structure"] = df.apply(_has_structure, axis=1)
    df["has_energy"] = _has_any_numeric(df, ["E", "form_G", "formation_energy_per_atom"])
    df["has_area"] = _has_any_numeric(df, ["Area"])
    df["energy_per_atom"] = _energy_per_atom(df)
    df["formation_energy_per_atom_numeric"] = pd.to_numeric(
        df["formation_energy_per_atom"], errors="coerce"
    ) if "formation_energy_per_atom" in df else np.nan
    df["phase_label"] = _phase_label(df)
    df["quality_flag"] = df.apply(_quality_flag, axis=1)
    return df


def profile_names() -> list[str]:
    return [profile.name for profile in COLUMN_PROFILES]


def columns_for_profile(name: str, dataframe: pd.DataFrame) -> list[str]:
    profile = next((item for item in COLUMN_PROFILES if item.name == name), COLUMN_PROFILES[0])
    columns = [column for column in profile.columns if column in dataframe.columns]
    columns.extend([column for column in dataframe.columns if column not in columns])
    return columns


def profile_review(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile in COLUMN_PROFILES:
        available = [column for column in profile.columns if column in dataframe.columns]
        missing = [column for column in profile.columns if column not in dataframe.columns]
        rows.append(
            {
                "page_focus": profile.name,
                "purpose": profile.purpose,
                "primary_columns": ", ".join(available[:12]),
                "available_count": len(available),
                "missing_or_future": ", ".join(missing),
            }
        )
    return pd.DataFrame(rows)


def column_context(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    important = {column for profile in COLUMN_PROFILES for column in profile.columns}
    for column in dataframe.columns:
        series = dataframe[column]
        try:
            unique = int(series.nunique(dropna=True))
        except TypeError:
            unique = int(series.dropna().map(repr).nunique())
        sample = ""
        if series.notna().any():
            sample = repr(series.dropna().iloc[0])[:120]
        rows.append(
            {
                "column": column,
                "recommended": column in important,
                "dtype": str(series.dtype),
                "non_null_pct": round(100 * float(series.notna().mean()), 1),
                "unique": unique,
                "sample": sample,
            }
        )
    return pd.DataFrame(rows).sort_values(["recommended", "non_null_pct"], ascending=[False, False])


def _record_type(row: pd.Series) -> str:
    dataset = str(row.get("dataset", "")).lower()
    name = str(row.get("Name", "")).lower()
    hkl = str(row.get("hkl", "")).lower()
    if "bulk" in dataset or "bulk" in name:
        return "bulk"
    if "cluster" in dataset or hkl == "cluster" or "cluster" in name:
        return "cluster"
    if "surface" in dataset or hkl in {"100", "110", "111", "211"} or "surf" in name:
        return "surface"
    return "record"


def _contains(df: pd.DataFrame, token: str) -> pd.Series:
    if "Name" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["Name"].astype(str).str.contains(token, case=False, na=False, regex=False)


def _contains_any(df: pd.DataFrame, tokens: list[str]) -> pd.Series:
    if "Name" not in df.columns:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    names = df["Name"].astype(str)
    for token in tokens:
        mask = mask | names.str.contains(token, case=False, na=False, regex=False)
    return mask


def _guess_adsorbate(name: Any) -> str:
    text = str(name)
    for token in sorted(DEFAULT_ADSORBATE_TOKENS, key=len, reverse=True):
        if re.search(rf"(^|-|_){re.escape(token)}($|-|_)", text, flags=re.IGNORECASE):
            return token
    return ""


def _reference_role(row: pd.Series) -> str:
    if bool(row.get("is_clean", False)) and row.get("record_type") == "surface":
        return "clean surface reference"
    if row.get("record_type") == "bulk":
        return "bulk reference"
    if bool(row.get("is_adsorbate_like", False)):
        return "adsorbate candidate"
    return "candidate"


def _reference_key(row: pd.Series) -> str:
    hkl = str(row.get("hkl", ""))
    slabsize = str(row.get("slabsize", ""))
    formula = str(row.get("Formula", ""))
    record_type = str(row.get("record_type", ""))
    return "|".join([record_type, hkl, slabsize, formula])


def _composition_summary(row: pd.Series) -> str:
    parts = []
    for element, value in sorted(row_element_count_map(row).items()):
        numeric = float(value)
        if numeric != 0:
            parts.append(f"{element}{int(numeric) if numeric.is_integer() else numeric}")
    return " ".join(parts)


def _coverage_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.1f}% ML"


def _n_atoms(row: pd.Series) -> float:
    for column in ["struc", "CONTCAR"]:
        value = row.get(column)
        if hasattr(value, "__len__") and value.__class__.__name__ == "Atoms":
            return float(len(value))
    counts = row_element_count_map(row)
    return float(sum(counts.values())) if counts else np.nan


def _has_structure(row: pd.Series) -> bool:
    return any(row.get(column).__class__.__name__ == "Atoms" for column in ["struc", "CONTCAR"])


def _has_any_numeric(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in columns:
        if column in df.columns:
            mask = mask | pd.to_numeric(df[column], errors="coerce").notna()
    return mask


def _energy_per_atom(df: pd.DataFrame) -> pd.Series:
    if "E" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    n_atoms = df.apply(_n_atoms, axis=1)
    return df["E"] / n_atoms.replace(0, np.nan)


def _phase_label(df: pd.DataFrame) -> pd.Series:
    if "legend" in df.columns:
        label = df["legend"].astype(str)
    elif "Name" in df.columns:
        label = df["Name"].astype(str)
    else:
        label = pd.Series("", index=df.index)
    if "Monolayer_alloy" in df.columns:
        ml = df["Monolayer_alloy"].map(_coverage_label)
        label = label.where(ml == "", ml)
    return label


def _quality_flag(row: pd.Series) -> str:
    if not row.get("has_energy", False):
        return "review: missing energy"
    if not row.get("has_structure", False):
        return "review: missing structure"
    fmax = row.get("fmax")
    if pd.notna(fmax) and float(fmax) > 0.05:
        return "review: high fmax"
    name = str(row.get("Name", "")).lower()
    if any(token in name for token in ["test", "crash", "failed", "broken"]):
        return "review: name flag"
    return "ok"
