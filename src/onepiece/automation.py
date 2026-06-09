from __future__ import annotations

import re

import numpy as np
import pandas as pd

from onepiece.adsorption import (
    ADSORBATE_PATTERN,
    adsorbate_counts_from_structures,
    annotate_adsorbates,
    annotate_copt_paths,
    assign_surface_references,
    atom_counts,
    primary_structure,
)
from onepiece.thermo import is_gas_phase_row

REACTION_STATE_TOKENS = (
    "H2NCH2OH",
    "H2NCH2O",
    "H2NCHOH",
    "H2NCHO",
    "H2CO_OH",
    "H2COOH",
    "HCOO_H",
    "HCO_OH",
    "H2COO",
    "HCOOH",
    "CH3OH",
    "CH3O_H",
    "CH3O",
    "H3COH",
    "H2CO_H",
    "H2CO",
    "CO_NH2",
    "NH2_H",
    "H_HCO",
    "HCO_H",
    "COOH",
    "HCOO",
    "CO2",
    "COH",
    "CHO",
    "HCO",
    "CH3",
    "NH3",
    "NH2",
    "CO",
    "OH",
    "O",
    "H",
)


def annotate_reaction_network(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate reaction-network metadata from names and copt paths."""
    df = annotate_copt_paths(annotate_adsorbates(frame))
    if "reaction_system_name" in df.columns:
        df["reaction_state"] = df["reaction_system_name"].map(_infer_reaction_system_state)
        fallback = df["reaction_state"].astype(str).eq("")
        df.loc[fallback, "reaction_state"] = df.loc[fallback, "Name"].map(_infer_reaction_state)
    else:
        df["reaction_state"] = df["Name"].map(_infer_reaction_state)
    df["reaction_step_initial"] = ""
    df["reaction_step_final"] = ""
    df["reaction_family"] = ""

    copt_mask = df.get("is_copt", pd.Series(False, index=df.index)).fillna(False)
    if copt_mask.any():
        copt_parts = df.loc[copt_mask].apply(
            lambda row: _split_reaction_token(_best_reaction_token(row)),
            axis=1,
        )
        df.loc[copt_mask, "reaction_step_initial"] = copt_parts.map(lambda item: item[0])
        df.loc[copt_mask, "reaction_step_final"] = copt_parts.map(lambda item: item[1])
        df.loc[copt_mask, "reaction_family"] = (
            df.loc[copt_mask, "reaction_step_initial"].astype(str)
            + " -> "
            + df.loc[copt_mask, "reaction_step_final"].astype(str)
        )

    static_mask = ~copt_mask
    df.loc[static_mask, "reaction_step_initial"] = df.loc[static_mask, "reaction_state"]
    df.loc[static_mask, "reaction_step_final"] = df.loc[static_mask, "reaction_state"]
    df.loc[static_mask, "reaction_family"] = df.loc[static_mask, "reaction_state"]
    df["reaction_network_role"] = np.where(copt_mask, "pathway_image", "state")
    return df


def apply_curation_rules(
    frame: pd.DataFrame,
    *,
    energy_column: str = "E",
    static_fmax_max: float = 0.05,
    copt_fmax_max: float = 0.10,
    exclude_name_tokens: list[str] | None = None,
    action: str = "exclude",
    status_column: str = "curation_status",
) -> pd.DataFrame:
    """Apply common DFT curation rules and optionally exclude bad rows."""
    df = annotate_copt_paths(frame.copy())
    exclude_name_tokens = exclude_name_tokens or ["test", "convergence", "failed", "broken"]
    energy = pd.to_numeric(df.get(energy_column), errors="coerce")
    fmax = pd.to_numeric(df.get("fmax"), errors="coerce")
    name_text = df.get("Name", pd.Series("", index=df.index)).astype(str)
    is_gas = df.apply(is_gas_phase_row, axis=1)
    has_structure = df.apply(lambda row: primary_structure(row) is not None, axis=1)
    is_copt = df.get("is_copt", pd.Series(False, index=df.index)).fillna(False)

    token_pattern = "|".join(re.escape(token) for token in exclude_name_tokens if token)
    name_flag = (
        name_text.str.contains(token_pattern, case=False, na=False, regex=True)
        if token_pattern
        else pd.Series(False, index=df.index)
    )

    df["flag_missing_energy"] = energy.isna()
    df["flag_zero_energy"] = energy.eq(0)
    df["flag_missing_structure"] = (~has_structure) & (~is_gas)
    df["flag_high_fmax_static"] = (~is_copt) & fmax.gt(static_fmax_max)
    df["flag_high_fmax_copt"] = is_copt & fmax.gt(copt_fmax_max)
    df["flag_name_tokens"] = name_flag
    flag_columns = [column for column in df.columns if column.startswith("flag_")]
    df["flag_any_bad"] = df[flag_columns].fillna(False).any(axis=1)

    df[status_column] = "ok"
    df.loc[df["flag_any_bad"], status_column] = "review"
    if action == "exclude":
        return df.loc[~df["flag_any_bad"]].copy()
    if action == "mark_excluded":
        df.loc[df["flag_any_bad"], status_column] = "excluded"
    return df


def add_structure_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive structure descriptors useful for catalytic trend analysis."""
    df = assign_surface_references(frame.copy())
    df["primary_atoms"] = df.apply(primary_structure, axis=1)
    df["n_atoms"] = df["primary_atoms"].map(
        lambda atoms: float(len(atoms)) if atoms is not None and atoms.__class__.__name__ == "Atoms" else np.nan
    )
    df["cell_volume"] = df["primary_atoms"].map(
        _safe_cell_volume
    )
    df["surface_ref_atoms"] = pd.Series(index=df.index, dtype=object)
    surface_rows = df.loc[
        df["Name"].astype(str).eq(df["surface_ref_name"].astype(str)) & df["primary_atoms"].notna(),
        ["Name", "primary_atoms"],
    ].drop_duplicates("Name")
    surface_map = surface_rows.set_index("Name")["primary_atoms"].to_dict()
    df["surface_ref_atoms"] = df["surface_ref_name"].map(surface_map)
    df["surface_atom_count"] = df["surface_ref_atoms"].map(
        lambda atoms: float(len(atoms)) if atoms is not None and atoms.__class__.__name__ == "Atoms" else np.nan
    )
    df["adsorbate_counts"] = [
        adsorbate_counts_from_structures(total_atoms, surface_atoms)
        for total_atoms, surface_atoms in zip(df["primary_atoms"], df["surface_ref_atoms"], strict=False)
    ]
    df["adsorbate_formula"] = df["adsorbate_counts"].map(_formula_from_counts)
    df["adsorbate_atom_count"] = df["adsorbate_counts"].map(lambda counts: float(sum(counts.values())))
    df["adsorbate_center_height"] = [
        _adsorbate_center_height(total_atoms, surface_atoms)
        for total_atoms, surface_atoms in zip(df["primary_atoms"], df["surface_ref_atoms"], strict=False)
    ]
    df["surface_top_z"] = df["surface_ref_atoms"].map(_surface_top_z)
    df["adsorbate_height_above_surface"] = df["adsorbate_center_height"] - df["surface_top_z"]
    df["surface_composition"] = df["surface_ref_atoms"].map(lambda atoms: _formula_from_counts(atom_counts(atoms)))
    return df


def _infer_reaction_state(name: object) -> str:
    text = str(name or "")
    if "-copt-" in text:
        return ""
    for token in REACTION_STATE_TOKENS:
        pattern = rf"(^|[-_%]){re.escape(token)}($|[-_%])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return token
    match = ADSORBATE_PATTERN.search(text)
    return match.group(1) if match else ""


def _infer_reaction_system_state(name: object) -> str:
    text = str(name or "")
    if text == "star":
        return "star"
    if text.endswith("gas"):
        return text[:-3]
    if text.endswith("star"):
        return text[:-4]
    return ""


def _best_reaction_token(row: pd.Series) -> str:
    name_text = str(row.get("Name", ""))
    if "-copt-" in name_text:
        tail = name_text.split("-copt-", 1)[1]
        match = re.match(r"(?P<reaction>.+)-(?P<path_id>[^-]+)-(?P<step>\d+)$", tail)
        if match:
            return match.group("reaction")
    return str(row.get("copt_reaction", "") or "")


def _split_reaction_token(token: object) -> tuple[str, str]:
    text = str(token or "")
    if "%" in text:
        left, right = text.split("%", 1)
        return left.strip(), right.strip()
    if "->" in text:
        left, right = text.split("->", 1)
        return left.strip(), right.strip()
    if "_" in text:
        parts = [part for part in text.split("_") if part]
        if len(parts) >= 2:
            midpoint = len(parts) // 2
            return "_".join(parts[:midpoint]), "_".join(parts[midpoint:])
    return text, ""


def _formula_from_counts(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    pieces = []
    for element in sorted(counts):
        count = int(counts[element])
        if count <= 0:
            continue
        pieces.append(f"{element}{'' if count == 1 else count}")
    return "".join(pieces)


def _surface_top_z(atoms: object) -> float:
    if atoms is None or atoms.__class__.__name__ != "Atoms":
        return np.nan
    positions = np.asarray(atoms.get_positions())
    if positions.size == 0:
        return np.nan
    return float(np.max(positions[:, 2]))


def _safe_cell_volume(atoms: object) -> float:
    if atoms is None or atoms.__class__.__name__ != "Atoms":
        return np.nan
    try:
        return float(atoms.get_volume())
    except Exception:
        return np.nan


def _adsorbate_center_height(total_atoms: object, surface_atoms: object) -> float:
    if total_atoms is None or total_atoms.__class__.__name__ != "Atoms":
        return np.nan
    if surface_atoms is None or surface_atoms.__class__.__name__ != "Atoms":
        return np.nan
    total_symbols = list(total_atoms.get_chemical_symbols())
    total_positions = np.asarray(total_atoms.get_positions())
    surface_symbol_counts = atom_counts(surface_atoms)
    used: dict[str, int] = {element: 0 for element in surface_symbol_counts}
    adsorbate_z: list[float] = []
    for symbol, position in zip(total_symbols, total_positions, strict=False):
        if used.get(symbol, 0) < surface_symbol_counts.get(symbol, 0):
            used[symbol] = used.get(symbol, 0) + 1
            continue
        adsorbate_z.append(float(position[2]))
    if not adsorbate_z:
        return np.nan
    return float(np.mean(adsorbate_z))
