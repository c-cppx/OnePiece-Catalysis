from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from onepiece.frame_utils import ensure_name_index, row_name
from onepiece.thermo import add_gibbs_free_energy

DEFAULT_ADSORBATE_TOKENS = (
    "Trimethylamin",
    "H2NCH2OH",
    "H2NCH2O",
    "H2NCHOH",
    "H2NCHO",
    "H2NCOH",
    "H2NCO",
    "H2NHCO_H",
    "H2NHCO",
    "H2CO_OH",
    "H2COOH",
    "HCOO_H",
    "HCO_OH",
    "H2COO",
    "HCOOH",
    "CH3OH",
    "CH3O",
    "H3COH",
    "H2COH",
    "H2CO",
    "CO_NH2",
    "NH2_H",
    "H_HCO",
    "HCO_H",
    "CHOH",
    "HCOH",
    "COOH",
    "HCOO",
    "CO2",
    "COH",
    "CHO",
    "HCO",
    "CH3",
    "NH3",
    "NH2",
    "Me3N",
    "CO",
    "OH",
    "CN",
    "O",
    "H",
)


def _adsorbate_pattern(tokens: tuple[str, ...]) -> re.Pattern[str]:
    ordered = sorted({token for token in tokens if token}, key=len, reverse=True)
    choices = "|".join(re.escape(token) for token in ordered)
    return re.compile(rf"[-_%]({choices})(?:[-_%].*|$)", re.IGNORECASE)


ADSORBATE_PATTERN = _adsorbate_pattern(DEFAULT_ADSORBATE_TOKENS)
FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")
EQUATION_TOKEN_PATTERN = re.compile(r"(?P<sign>[+-]?)\s*(?P<coef>\d*\.?\d*)\s*(?P<species>[A-Za-z0-9_]+)")
NAME_COPT_PATTERN = re.compile(
    r"^(?P<surface_base>.*)-copt-(?P<reaction>.+)-(?P<path_id>[^-]+)-(?P<step>\d+)$"
)
DEFAULT_FORMULA_GAS_SPECIES = ("CO2", "H2", "H2O", "NH3")
NON_METAL_REFERENCE_ELEMENTS = {"C", "H", "O", "N"}


@dataclass(frozen=True)
class GasReferences:
    """Gas-phase reference energies in eV from the same computational setup."""

    co: float = np.nan
    co2: float = np.nan
    ch3oh: float = np.nan
    h2: float = np.nan
    h2o: float = np.nan

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | None) -> GasReferences:
        if values is None:
            return cls()
        normalized = {str(key).lower(): value for key, value in values.items()}
        return cls(
            co=float(normalized.get("co", np.nan)),
            co2=float(normalized.get("co2", np.nan)),
            ch3oh=float(normalized.get("ch3oh", np.nan)),
            h2=float(normalized.get("h2", np.nan)),
            h2o=float(normalized.get("h2o", np.nan)),
        )


def parse_reference_equation(equation: str) -> dict[str, float]:
    """Parse a simple gas-reference equation like ``CO2+H2-H2O``."""
    text = str(equation or "").replace(" ", "")
    if not text:
        return {}
    refs: dict[str, float] = {}
    for match in EQUATION_TOKEN_PATTERN.finditer(text):
        species = str(match.group("species") or "").strip()
        if not species:
            continue
        sign = -1.0 if match.group("sign") == "-" else 1.0
        coef_text = str(match.group("coef") or "").strip()
        coefficient = float(coef_text) if coef_text else 1.0
        refs[species] = refs.get(species, 0.0) + sign * coefficient
    return {species: value for species, value in refs.items() if not np.isclose(value, 0.0)}


def infer_reference_equation_from_formula(formula: str) -> dict[str, float]:
    """Infer gas-reference coefficients from a CHON formula.

    The default basis is the combination of ``CO2``, ``H2``, ``H2O``, and
    ``NH3``. Coefficients may be fractional or negative.
    """
    counts = formula_counts(formula)
    if not counts:
        return {}
    if any(element not in NON_METAL_REFERENCE_ELEMENTS for element in counts):
        counts = {element: count for element, count in counts.items() if element in NON_METAL_REFERENCE_ELEMENTS}
    if not counts:
        return {}
    co2 = float(counts.get("C", 0))
    nh3 = float(counts.get("N", 0))
    h2o = float(counts.get("O", 0) - 2.0 * co2)
    h2 = 0.5 * (float(counts.get("H", 0)) - 2.0 * h2o - 3.0 * nh3)
    refs = {
        "CO2": co2,
        "H2": h2,
        "H2O": h2o,
        "NH3": nh3,
    }
    return {species: value for species, value in refs.items() if not np.isclose(value, 0.0)}


def default_room_temperature_phase(element: str) -> str:
    """Return the default bulk phase label used for metal references."""
    return "fcc"


def infer_adsorbate_recipe(
    adsorbate: str,
    *,
    formula: str | None = None,
    equation: str | None = None,
    basis: str | None = None,
) -> dict[str, object]:
    """Infer a backend recipe for one adsorbate from formula/equation information."""
    label = str(adsorbate or "").strip()
    active_formula = str(formula or label).strip()
    counts = formula_counts(active_formula)
    gas_refs = parse_reference_equation(str(equation)) if equation else infer_reference_equation_from_formula(active_formula)
    bulk_refs = {
        element: float(count)
        for element, count in counts.items()
        if element not in NON_METAL_REFERENCE_ELEMENTS and int(count) > 0
    }
    bulk_phases = {element: default_room_temperature_phase(element) for element in bulk_refs}
    chosen_basis = str(basis or _default_basis_from_counts(counts)).strip() or "C"
    recipe: dict[str, object] = {
        "basis": chosen_basis,
        "gas_refs": gas_refs,
        "formula": active_formula,
    }
    if equation:
        recipe["equation"] = str(equation)
    if bulk_refs:
        recipe["bulk_refs"] = bulk_refs
        recipe["bulk_phases"] = bulk_phases
    return recipe


def infer_adsorption_recipes(
    frame: pd.DataFrame,
    *,
    adsorbate_column: str = "adsorbate",
    adsorbate_formula_column: str = "adsorbate_formula",
    formula_column: str = "Formula",
    equation_columns: tuple[str, ...] = ("equation", "adsorbate_equation", "reaction_equation"),
) -> dict[str, dict[str, object]]:
    """Infer adsorption recipes for all adsorbates that appear in a dataframe."""
    df = frame.copy()
    if adsorbate_column not in df.columns:
        df = annotate_adsorbates(df)
    recipes: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        adsorbate = str(row.get(adsorbate_column, "")).strip()
        if not adsorbate:
            continue
        if adsorbate in recipes:
            continue
        formula_value = str(
            row.get(adsorbate_formula_column)
            or adsorbate
            or row.get(formula_column)
        ).strip()
        equation = next(
            (str(row.get(column)).strip() for column in equation_columns if str(row.get(column, "")).strip()),
            "",
        )
        recipes[adsorbate] = infer_adsorbate_recipe(
            adsorbate,
            formula=formula_value,
            equation=equation or None,
        )
    return recipes


def read_onepiece_hdf(path: Path | str, key: str = "df", dataset_label: str | None = None) -> pd.DataFrame:
    """Read one OnePiece pandas HDF file and attach provenance columns."""
    path = Path(path)
    frame = pd.read_hdf(path, key=key).copy()
    frame["dataset"] = path.stem
    frame["dataset_label"] = dataset_label or path.stem
    frame["source_hdf"] = str(path)
    frame["source_row"] = np.arange(len(frame), dtype=int)
    return frame


def read_onepiece_hdfs(
    hdf_files: Mapping[str, Path | str],
    key: str = "df",
) -> dict[str, pd.DataFrame]:
    """Read several HDF files into source-labeled DataFrames."""
    return {
        label: read_onepiece_hdf(path, key=key, dataset_label=label)
        for label, path in hdf_files.items()
    }


def formula_counts(formula: object) -> dict[str, int]:
    """Parse a simple chemical formula into element counts."""
    if not isinstance(formula, str):
        return {}
    counts: dict[str, int] = {}
    for element, number in FORMULA_PATTERN.findall(formula):
        counts[element] = counts.get(element, 0) + int(number or 1)
    return counts


def count_element(
    row: pd.Series,
    element: str,
    *,
    structure_columns: tuple[str, ...] = ("struc", "CONTCAR", "structure", "atoms"),
) -> float:
    """Read an element count from explicit columns, structures, or Formula."""
    if element in row.index:
        value = pd.to_numeric(row[element], errors="coerce")
        if pd.notna(value):
            return float(value)
    atoms = primary_structure(row, structure_columns=structure_columns)
    if atoms is not None:
        return float(atom_counts(atoms).get(element, 0))
    return float(formula_counts(row.get("Formula")).get(element, 0))


def count_element_in_structure(
    row: pd.Series,
    element: str,
    *,
    structure_columns: tuple[str, ...] = ("struc", "CONTCAR", "structure", "atoms"),
) -> float:
    """Read an element count only from ASE Atoms-bearing structure columns."""
    atoms = primary_structure(row, structure_columns=structure_columns)
    if atoms is not None:
        return float(atom_counts(atoms).get(element, 0))
    return 0.0


def row_element_count_map(
    row: pd.Series,
    *,
    structure_columns: tuple[str, ...] = ("struc", "CONTCAR", "structure", "atoms"),
) -> dict[str, int]:
    """Return all element counts that can be inferred for a row."""
    atoms = primary_structure(row, structure_columns=structure_columns)
    if atoms is not None:
        return atom_counts(atoms)
    return {}


def row_elements(
    row: pd.Series,
    *,
    structure_columns: tuple[str, ...] = ("struc", "CONTCAR", "structure", "atoms"),
) -> tuple[str, ...]:
    """Return the sorted unique element symbols present in a row."""
    return tuple(sorted(row_element_count_map(row, structure_columns=structure_columns)))


def structure_columns_in_frame(frame: pd.DataFrame) -> list[str]:
    """Return dataframe columns that contain ASE Atoms objects in at least one row."""
    columns: list[str] = []
    for column in frame.columns:
        sample = frame[column].dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        if first.__class__.__name__ == "Atoms":
            columns.append(str(column))
    return columns


def get_all_elements(frame: pd.DataFrame, *, structure_column: str | None = None) -> list[str]:
    """Discover all elements present anywhere in a dataset."""
    structure_columns = (
        (structure_column,)
        if structure_column
        else tuple(structure_columns_in_frame(frame))
    )
    if not structure_columns:
        return []
    elements: set[str] = set()
    for _, row in frame.iterrows():
        elements.update(row_elements(row, structure_columns=structure_columns))
    return sorted(elements)


def add_element_count_columns(
    frame: pd.DataFrame,
    elements: list[str] | None = None,
    *,
    structure_column: str | None = None,
) -> pd.DataFrame:
    """Add one count column per element detected in the dataset."""
    result = frame.copy()
    structure_columns = (
        (structure_column,)
        if structure_column
        else tuple(structure_columns_in_frame(result))
    )
    element_list = elements or get_all_elements(result, structure_column=structure_column)
    if not element_list:
        return result
    counts_by_row = result.apply(
        lambda row: row_element_count_map(row, structure_columns=structure_columns),
        axis=1,
    )
    for element in element_list:
        result[element] = counts_by_row.map(lambda counts, element=element: float(counts.get(element, 0)))
    return result


def primary_structure(
    row: pd.Series,
    structure_columns: tuple[str, ...] = ("struc", "CONTCAR", "structure", "atoms"),
):
    """Return the first ASE Atoms-like object found in the preferred columns."""
    for column in structure_columns:
        value = row.get(column)
        if value.__class__.__name__ == "Atoms":
            return value
    return None


def atom_counts(atoms: object) -> dict[str, int]:
    """Count chemical symbols from an ASE Atoms-like object."""
    if atoms is None or atoms.__class__.__name__ != "Atoms":
        return {}
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _default_basis_from_counts(counts: dict[str, int]) -> str:
    for element in ("C", "N", "O", "H"):
        if counts.get(element, 0) > 0:
            return element
    for element in sorted(counts):
        if counts.get(element, 0) > 0:
            return element
    return "C"


def adsorbate_counts_from_structures(total_atoms: object, surface_atoms: object) -> dict[str, int]:
    """Subtract clean-surface stoichiometry from a full adsorbate structure."""
    total_counts = atom_counts(total_atoms)
    surface_counts = atom_counts(surface_atoms)
    elements = set(total_counts) | set(surface_counts)
    counts: dict[str, int] = {}
    for element in sorted(elements):
        difference = total_counts.get(element, 0) - surface_counts.get(element, 0)
        if difference > 0:
            counts[element] = int(difference)
    return counts


def guess_adsorbate(name: object) -> str:
    """Guess the adsorbate token from a calculation name."""
    if not isinstance(name, str):
        return ""
    match = ADSORBATE_PATTERN.search(name)
    return match.group(1) if match else ""


def surface_key_from_name(name: object) -> str:
    """Remove adsorbate and constrained-optimization suffixes from a row name."""
    if not isinstance(name, str):
        return ""
    copt_match = NAME_COPT_PATTERN.match(name)
    if copt_match:
        return copt_match.group("surface_base")
    return ADSORBATE_PATTERN.sub("", name)


def annotate_adsorbates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add adsorbate, surface key, and basic energy columns."""
    df = ensure_name_index(frame)
    df["Name"] = df.get("Name", pd.Series([""] * len(df), index=df.index)).astype(str)
    df["E"] = pd.to_numeric(df.get("E"), errors="coerce")
    df["adsorbate"] = df["Name"].map(guess_adsorbate)
    df["is_adsorbate"] = df["adsorbate"] != ""
    df["surface_key"] = df["Name"].map(surface_key_from_name)
    return df


def choose_surface_references(frame: pd.DataFrame) -> pd.DataFrame:
    """Choose one clean/reference row per surface key in a single source table."""
    df = annotate_adsorbates(frame)
    candidates = df.loc[
        ~df["is_adsorbate"]
        & df["E"].notna()
        & (pd.to_numeric(df["E"], errors="coerce") != 0)
        & ~is_constrained_optimization(df)
    ].copy()
    if candidates.empty:
        return candidates

    candidates["reference_candidate_count"] = candidates.groupby("surface_key")["Name"].transform(
        "count"
    )
    candidates = candidates.sort_values(["surface_key", "E"], ascending=[True, True])
    references = candidates.drop_duplicates("surface_key", keep="first").copy()
    references["reference_ambiguous"] = references["reference_candidate_count"] > 1
    return references


def assign_surface_references(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign clean-surface references before merging multiple HDF sources."""
    df = annotate_adsorbates(frame)
    refs = choose_surface_references(df)
    reference_lookup = refs.set_index("surface_key") if not refs.empty else pd.DataFrame()

    df["surface_ref_name"] = df["surface_key"].map(
        reference_lookup["Name"] if "Name" in reference_lookup else pd.Series(dtype=object)
    )
    df["surface_ref_E"] = df["surface_key"].map(
        reference_lookup["E"] if "E" in reference_lookup else pd.Series(dtype=float)
    )
    df["surface_ref_formula"] = df["surface_key"].map(
        reference_lookup["Formula"] if "Formula" in reference_lookup else pd.Series(dtype=object)
    )
    df["surface_ref_ambiguous"] = df["surface_key"].map(
        reference_lookup["reference_ambiguous"]
        if "reference_ambiguous" in reference_lookup
        else pd.Series(dtype=bool)
    )

    df["surface_ref_status"] = "ok"
    df.loc[df["surface_ref_name"].isna(), "surface_ref_status"] = "missing"
    df.loc[df["surface_ref_ambiguous"].fillna(False), "surface_ref_status"] = "ambiguous"
    df.loc[~df["is_adsorbate"] & (df["surface_ref_status"] == "ok"), "surface_ref_status"] = "self"

    for element in ("C", "H", "O"):
        current = df.apply(lambda row, element=element: count_element(row, element), axis=1)
        ref_counts = (
            refs.set_index("surface_key").apply(
                lambda row, element=element: count_element(row, element),
                axis=1,
            )
            if not refs.empty
            else pd.Series(dtype=float)
        )
        df[f"delta_{element}"] = current - df["surface_key"].map(ref_counts).fillna(0)

    df["delta_E_to_surface_eV"] = df["E"] - df["surface_ref_E"]
    return df


def assign_references_before_merge(
    hdf_files: Mapping[str, Path | str],
    key: str = "df",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read, reference-annotate, and then merge HDF sources."""
    enriched_frames = []
    reference_frames = []
    for label, path in hdf_files.items():
        frame = read_onepiece_hdf(path, key=key, dataset_label=label)
        enriched = assign_surface_references(frame)
        enriched["dataset_label"] = label
        enriched_frames.append(enriched)
        reference_frames.append(
            enriched.loc[
                ~enriched["is_adsorbate"],
                [
                    "dataset_label",
                    "Name",
                    "Formula",
                    "E",
                    "surface_key",
                    "surface_ref_status",
                    "source_hdf",
                    "source_row",
                ],
            ]
        )
    combined = ensure_name_index(pd.concat(enriched_frames, ignore_index=False, sort=False))
    references = ensure_name_index(pd.concat(reference_frames, ignore_index=False, sort=False))
    return combined, references


def add_adsorption_energies(
    frame: pd.DataFrame,
    gas_references_ev: Mapping[str, float] | GasReferences | None = None,
) -> pd.DataFrame:
    """Add CO and methanol-to-methoxy adsorption energy columns.

    CO:
        E_ads,total = E(CO*) - E(*) - n E(CO_gas)
        E_ads,per CO = E_ads,total / n

    Methoxy from methanol:
        * + CH3OH(g) -> CH3O* + 1/2 H2(g)
        E_ads,total = E(CH3O*) + 0.5 n E(H2) - E(*) - n E(CH3OH)
        E_ads,per adsorbate = E_ads,total / n
    """
    refs = gas_references_ev if isinstance(gas_references_ev, GasReferences) else GasReferences.from_mapping(gas_references_ev)
    df = frame.copy()
    for column in ("delta_C", "E", "surface_ref_E"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["n_CO_adsorbates"] = np.where(df["adsorbate"].eq("CO"), df["delta_C"], np.nan)
    df["n_CH3O_adsorbates"] = np.where(df["adsorbate"].eq("CH3O"), df["delta_C"], np.nan)

    valid_co = df["n_CO_adsorbates"].fillna(0) > 0
    df["E_ads_CO_total_eV"] = np.nan
    df["E_ads_CO_eV"] = np.nan
    df.loc[valid_co, "E_ads_CO_total_eV"] = (
        df.loc[valid_co, "E"]
        - df.loc[valid_co, "surface_ref_E"]
        - df.loc[valid_co, "n_CO_adsorbates"] * refs.co
    )
    df.loc[valid_co, "E_ads_CO_eV"] = (
        df.loc[valid_co, "E_ads_CO_total_eV"] / df.loc[valid_co, "n_CO_adsorbates"]
    )

    valid_ch3o = df["n_CH3O_adsorbates"].fillna(0) > 0
    df["E_ads_CH3OH_to_CH3O_total_eV"] = np.nan
    df["E_ads_CH3OH_to_CH3O_eV"] = np.nan
    df.loc[valid_ch3o, "E_ads_CH3OH_to_CH3O_total_eV"] = (
        df.loc[valid_ch3o, "E"]
        + 0.5 * df.loc[valid_ch3o, "n_CH3O_adsorbates"] * refs.h2
        - df.loc[valid_ch3o, "surface_ref_E"]
        - df.loc[valid_ch3o, "n_CH3O_adsorbates"] * refs.ch3oh
    )
    df.loc[valid_ch3o, "E_ads_CH3OH_to_CH3O_eV"] = (
        df.loc[valid_ch3o, "E_ads_CH3OH_to_CH3O_total_eV"]
        / df.loc[valid_ch3o, "n_CH3O_adsorbates"]
    )
    return df


def add_recipe_adsorption_energies(
    frame: pd.DataFrame,
    gas_reference_values: Mapping[str, float] | None,
    recipes: Mapping[str, Mapping[str, object]] | None,
) -> pd.DataFrame:
    """Add adsorption energies for arbitrary adsorbate recipes.

    Each recipe has the shape:
        {
            "basis": "C",
            "gas_refs": {"CO": 1.0, "H2": 1.5},
        }

    The total adsorption energy is:
        E_ads,total = E - E(surface_ref) - n_basis * sum_i coeff_i E(gas_i)

    and the per-adsorbate energy is:
        E_ads = E_ads,total / n_basis
    """
    df = frame.copy()
    if "adsorbate" not in df.columns or "surface_ref_E" not in df.columns:
        df = assign_surface_references(df)
    if "adsorbate" not in df.columns:
        df = annotate_adsorbates(df)
    active_recipes = recipes or infer_adsorption_recipes(df)
    if not active_recipes:
        return df
    df["E"] = pd.to_numeric(df.get("E"), errors="coerce")
    df["surface_ref_E"] = pd.to_numeric(df.get("surface_ref_E"), errors="coerce")

    normalized_gases = {
        str(key): float(value)
        for key, value in (gas_reference_values or {}).items()
        if value is not None and pd.notna(value)
    }

    for label, recipe in active_recipes.items():
        basis = str(recipe.get("basis", "C")).strip() or "C"
        gas_refs = recipe.get("gas_refs", {}) or {}
        n_column = str(recipe.get("count_column", f"n_{label}_adsorbates"))
        total_column = str(recipe.get("total_column", f"E_ads_{label}_total_eV"))
        per_column = str(recipe.get("per_column", f"E_ads_{label}_eV"))

        multiplier = _basis_multiplier(df, basis)
        df[n_column] = np.where(df["adsorbate"].astype(str).eq(str(label)), multiplier, np.nan)
        df[total_column] = np.nan
        df[per_column] = np.nan

        gas_total = 0.0
        missing = False
        for species, coefficient in gas_refs.items():
            if str(species) not in normalized_gases:
                missing = True
                break
            gas_total += float(coefficient) * normalized_gases[str(species)]
        if missing:
            continue

        valid = df["adsorbate"].astype(str).eq(str(label)) & pd.to_numeric(df[n_column], errors="coerce").fillna(0).gt(0)
        df.loc[valid, total_column] = (
            df.loc[valid, "E"]
            - df.loc[valid, "surface_ref_E"]
            - pd.to_numeric(df.loc[valid, n_column], errors="coerce") * gas_total
        )
        df.loc[valid, per_column] = (
            pd.to_numeric(df.loc[valid, total_column], errors="coerce")
            / pd.to_numeric(df.loc[valid, n_column], errors="coerce")
        )

    return df


def add_catalysis_hub_adsorption_energies(
    frame: pd.DataFrame,
    *,
    energy_column: str = "E",
    reaction_id_column: str = "reaction_id",
    system_name_column: str = "reaction_system_name",
    output_column: str = "adsorption_energy",
) -> pd.DataFrame:
    """Compute adsorption energies from Catalysis-Hub reaction-system rows.

    Catalysis-Hub reaction entries often store the surface reference (`star`),
    the gas-phase reference (for example `CO2gas`), and the adsorbate state
    (for example `CO2star`) under the same reaction id. For such rows, the
    adsorption energy is:

    ``E_ads = E(adsorbate*) - E(*) - E(gas)``

    The function adds the necessary helper columns and compares the calculated
    value against the published `reactionEnergy` when that column is present.
    """
    df = frame.copy()
    system_names = df.get(system_name_column, pd.Series("", index=df.index)).astype(str)
    df["cathub_system_kind"] = "other"
    df.loc[system_names.eq("star"), "cathub_system_kind"] = "surface"
    df.loc[system_names.str.endswith("gas", na=False), "cathub_system_kind"] = "gas"
    df.loc[
        system_names.str.endswith("star", na=False) & ~system_names.eq("star"),
        "cathub_system_kind",
    ] = "adsorbate"
    df["cathub_adsorbate"] = np.where(
        df["cathub_system_kind"].isin(["gas", "adsorbate"]),
        system_names.str.replace(r"(gas|star)$", "", regex=True),
        "",
    )

    surface_refs = (
        df.loc[df["cathub_system_kind"].eq("surface"), [reaction_id_column, energy_column]]
        .dropna(subset=[energy_column])
        .drop_duplicates(reaction_id_column)
        .rename(columns={energy_column: "surface_ref_E"})
    )
    gas_refs = (
        df.loc[df["cathub_system_kind"].eq("gas"), [reaction_id_column, "cathub_adsorbate", energy_column]]
        .dropna(subset=[energy_column])
        .drop_duplicates([reaction_id_column, "cathub_adsorbate"])
        .rename(columns={energy_column: "gas_ref_E"})
    )

    df = df.merge(surface_refs, on=reaction_id_column, how="left")
    df = df.merge(gas_refs, on=[reaction_id_column, "cathub_adsorbate"], how="left")
    df[output_column] = np.nan
    valid = (
        df["cathub_system_kind"].eq("adsorbate")
        & pd.to_numeric(df.get(energy_column), errors="coerce").notna()
        & pd.to_numeric(df.get("surface_ref_E"), errors="coerce").notna()
        & pd.to_numeric(df.get("gas_ref_E"), errors="coerce").notna()
    )
    df.loc[valid, output_column] = (
        pd.to_numeric(df.loc[valid, energy_column], errors="coerce")
        - pd.to_numeric(df.loc[valid, "surface_ref_E"], errors="coerce")
        - pd.to_numeric(df.loc[valid, "gas_ref_E"], errors="coerce")
    )
    if "reactionEnergy" in df.columns:
        published = pd.to_numeric(df["reactionEnergy"], errors="coerce")
        df["adsorption_energy_delta_vs_reactionEnergy"] = pd.to_numeric(df[output_column], errors="coerce") - published
    return df


def add_elemental_adsorption_energy(
    frame: pd.DataFrame,
    gas_reference_values: Mapping[str, float] | GasReferences | None,
    *,
    energy_column: str = "E",
    surface_reference_energy_column: str | None = None,
    structure_columns: tuple[str, ...] = ("struc", "CONTCAR", "structure", "atoms"),
    surface_ref_name_column: str = "surface_ref_name",
    output_column: str = "adsorption_energy",
) -> pd.DataFrame:
    """Add a OnePiece-style adsorption-energy column from structure stoichiometry.

    The adsorbate stoichiometry is taken from the difference between the row's
    ASE ``Atoms`` object and the matched clean-surface reference structure.

    The chemical potentials follow the notebook convention used for the methanol
    reaction analysis:

    ``mu_H = 0.5 * E(H2)``
    ``mu_O = E(H2O) - E(H2)``
    ``mu_C = E(CO2) - E(H2O) + 0.5 * E(H2)``

    The adsorption energy is then

    ``E_ads = E - E_surface - n_C * mu_C - n_H * mu_H - n_O * mu_O``
    """
    refs = gas_reference_values if isinstance(gas_reference_values, GasReferences) else GasReferences.from_mapping(gas_reference_values)
    df = frame.copy()
    if surface_ref_name_column not in df.columns or "surface_ref_E" not in df.columns:
        df = assign_surface_references(df)

    reference_column = surface_reference_energy_column or ("surface_ref_E" if energy_column == "E" else f"surface_ref_{energy_column}")
    df[energy_column] = pd.to_numeric(df.get(energy_column), errors="coerce")
    if reference_column not in df.columns:
        reference_lookup = (
            df.loc[df["Name"].astype(str).eq(df[surface_ref_name_column].astype(str)), ["Name", energy_column]]
            .dropna(subset=[energy_column])
            .drop_duplicates("Name")
            .set_index("Name")[energy_column]
        )
        df[reference_column] = df[surface_ref_name_column].map(reference_lookup)
    df[reference_column] = pd.to_numeric(df.get(reference_column), errors="coerce")
    df["primary_atoms"] = df.apply(lambda row: primary_structure(row, structure_columns=structure_columns), axis=1)

    surface_rows = df.loc[
        df["Name"].astype(str).eq(df[surface_ref_name_column].astype(str)) & df["primary_atoms"].notna()
    ][["Name", "primary_atoms"]].drop_duplicates("Name")
    surface_atom_map = surface_rows.set_index("Name")["primary_atoms"].to_dict()
    df["surface_ref_atoms"] = df[surface_ref_name_column].map(surface_atom_map)
    df["adsorbate_counts"] = [
        adsorbate_counts_from_structures(total_atoms, surface_atoms)
        for total_atoms, surface_atoms in zip(df["primary_atoms"], df["surface_ref_atoms"], strict=False)
    ]
    for element in ("C", "H", "O"):
        df[f"{element}_ads"] = df["adsorbate_counts"].map(
            lambda counts, element=element: int(counts.get(element, 0))
        )

    mu_h = 0.5 * refs.h2 if pd.notna(refs.h2) else np.nan
    mu_o = refs.h2o - refs.h2 if pd.notna(refs.h2o) and pd.notna(refs.h2) else np.nan
    mu_c = refs.co2 - refs.h2o + 0.5 * refs.h2 if pd.notna(refs.co2) and pd.notna(refs.h2o) and pd.notna(refs.h2) else np.nan

    df["mu_C_eV"] = mu_c
    df["mu_H_eV"] = mu_h
    df["mu_O_eV"] = mu_o
    df[output_column] = (
        df[energy_column]
        - df[reference_column]
        - df["C_ads"] * mu_c
        - df["H_ads"] * mu_h
        - df["O_ads"] * mu_o
    )
    return df


def add_elemental_adsorption_free_energy(
    frame: pd.DataFrame,
    gas_reference_values: Mapping[str, float] | GasReferences | None,
    *,
    temperature: float | None = None,
    energy_column: str = "E",
    gibbs_column: str = "G",
    structure_columns: tuple[str, ...] = ("struc", "CONTCAR", "structure", "atoms"),
    surface_ref_name_column: str = "surface_ref_name",
    output_column: str = "adsorption_free_energy",
) -> pd.DataFrame:
    """Add a Gibbs adsorption free-energy column from structure stoichiometry.

    This follows the same reference construction as ``add_elemental_adsorption_energy``,
    but uses a Gibbs free-energy column for both adsorbates and gas references.
    """
    df = frame.copy()
    if gibbs_column not in df.columns:
        if temperature is None:
            raise ValueError(f"{gibbs_column} is missing and no temperature was provided to compute it.")
        df = add_gibbs_free_energy(df, temperature=temperature, energy_column=energy_column, output_column=gibbs_column)

    if surface_ref_name_column not in df.columns or "surface_ref_E" not in df.columns:
        df = assign_surface_references(df)

    df[gibbs_column] = pd.to_numeric(df.get(gibbs_column), errors="coerce")
    reference_lookup = (
        df.loc[df["Name"].astype(str).eq(df[surface_ref_name_column].astype(str)), ["Name", gibbs_column]]
        .dropna(subset=[gibbs_column])
        .drop_duplicates("Name")
        .set_index("Name")[gibbs_column]
    )
    df["surface_ref_G"] = df[surface_ref_name_column].map(reference_lookup)
    result = add_elemental_adsorption_energy(
        df,
        gas_reference_values,
        energy_column=gibbs_column,
        structure_columns=structure_columns,
        surface_ref_name_column=surface_ref_name_column,
        output_column=output_column,
    )
    result["surface_ref_G"] = pd.to_numeric(result.get("surface_ref_G"), errors="coerce")
    result["mu_C_G_eV"] = result["mu_C_eV"]
    result["mu_H_G_eV"] = result["mu_H_eV"]
    result["mu_O_G_eV"] = result["mu_O_eV"]
    return result


def _basis_multiplier(frame: pd.DataFrame, basis: str) -> pd.Series:
    delta_column = f"delta_{basis}"
    if delta_column in frame.columns:
        return pd.to_numeric(frame[delta_column], errors="coerce")

    current = frame.apply(lambda row: count_element(row, basis), axis=1)
    ref = frame.get("surface_ref_formula", pd.Series(index=frame.index, dtype=object)).map(
        lambda formula: float(formula_counts(formula).get(basis, 0))
    )
    return pd.to_numeric(current - ref.fillna(0), errors="coerce")


def adsorption_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a focused adsorption-analysis table for UI or notebook use."""
    columns = [
        "dataset_label",
        "Name",
        "Formula",
        "adsorbate",
        "surface_ref_name",
        "surface_ref_formula",
        "surface_ref_status",
        "E",
        "surface_ref_E",
        "delta_E_to_surface_eV",
        "delta_C",
        "delta_H",
        "delta_O",
        "n_CO_adsorbates",
        "E_ads_CO_total_eV",
        "E_ads_CO_eV",
        "n_CH3O_adsorbates",
        "E_ads_CH3OH_to_CH3O_total_eV",
        "E_ads_CH3OH_to_CH3O_eV",
        "fmax",
        "source_hdf",
        "source_row",
    ]
    available = [column for column in columns if column in frame.columns]
    mask = frame["is_adsorbate"] if "is_adsorbate" in frame.columns else pd.Series(True, index=frame.index)
    return frame.loc[mask, available].copy()


def is_constrained_optimization(frame: pd.DataFrame) -> pd.Series:
    """Identify constrained optimization rows, primarily copt path scans."""
    text = _combined_text(frame)
    return text.str.contains(r"(?:^|[-_/])copt(?:$|[-_/])", case=False, regex=True, na=False)


def annotate_copt_paths(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate constrained-optimization metadata from Path or Name."""
    df = ensure_name_index(frame)
    df["Name"] = df.get("Name", pd.Series([""] * len(df), index=df.index)).astype(str)
    df["Path"] = df.get("Path", pd.Series([""] * len(df), index=df.index)).astype(str)
    df["E"] = pd.to_numeric(df.get("E"), errors="coerce")
    df["is_copt"] = is_constrained_optimization(df)

    metadata = df.apply(_parse_copt_metadata, axis=1, result_type="expand")
    for column in metadata.columns:
        df[column] = metadata[column]
    return df


def copt_profile_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Return point-level relative energies for constrained-optimization paths."""
    df = annotate_copt_paths(frame)
    points = df.loc[
        df["is_copt"]
        & df["copt_step"].notna()
        & df["copt_series_id"].notna()
        & df["E"].notna()
        & (df["E"] != 0)
    ].copy()
    if points.empty:
        return points

    points["copt_step"] = points["copt_step"].astype(int)
    points = points.sort_values(["copt_series_id", "copt_step", "E"])
    points = points.drop_duplicates(["copt_series_id", "copt_step"], keep="first")
    first_energy = points.groupby("copt_series_id")["E"].transform("first")
    min_energy = points.groupby("copt_series_id")["E"].transform("min")
    points["relative_E_from_initial_eV"] = points["E"] - first_energy
    points["relative_E_from_min_eV"] = points["E"] - min_energy
    return points


def copt_barrier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize apparent barriers from constrained-optimization path energies."""
    points = copt_profile_points(frame)
    if points.empty:
        return pd.DataFrame(
            columns=[
                "copt_series_id",
                "dataset_label",
                "copt_surface_base",
                "copt_reaction",
                "copt_path_id",
                "n_points",
                "initial_E_eV",
                "final_E_eV",
                "max_E_eV",
                "forward_barrier_eV",
                "reverse_barrier_eV",
                "reaction_energy_eV",
                "ts_step",
                "complete_scan",
            ]
        )

    rows = []
    for series_id, group in points.groupby("copt_series_id", sort=False):
        ordered = group.sort_values("copt_step")
        initial = float(ordered["E"].iloc[0])
        final = float(ordered["E"].iloc[-1])
        max_row = ordered.loc[ordered["E"].idxmax()]
        steps = set(ordered["copt_step"].astype(int))
        rows.append(
            {
                "copt_series_id": series_id,
                "dataset_label": ordered["dataset_label"].iloc[0]
                if "dataset_label" in ordered
                else ordered.get("dataset", pd.Series([""])).iloc[0],
                "copt_surface_base": ordered["copt_surface_base"].iloc[0],
                "copt_reaction": ordered["copt_reaction"].iloc[0],
                "copt_path_id": ordered["copt_path_id"].iloc[0],
                "n_points": int(len(ordered)),
                "initial_E_eV": initial,
                "final_E_eV": final,
                "max_E_eV": float(max_row["E"]),
                "forward_barrier_eV": float(max_row["E"] - initial),
                "reverse_barrier_eV": float(max_row["E"] - final),
                "reaction_energy_eV": float(final - initial),
                "ts_step": int(max_row["copt_step"]),
                "complete_scan": bool({0, 6}.issubset(steps) and len(steps) >= 5),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["copt_reaction", "forward_barrier_eV"], ascending=[True, False]
    )


def _combined_text(frame: pd.DataFrame) -> pd.Series:
    name = frame.get("Name", pd.Series([""] * len(frame), index=frame.index)).astype(str)
    path = frame.get("Path", pd.Series([""] * len(frame), index=frame.index)).astype(str)
    return name + " " + path


def _parse_copt_metadata(row: pd.Series) -> pd.Series:
    path_text = str(row.get("Path", ""))
    name_text = row_name(row)
    dataset = str(row.get("dataset_label", row.get("dataset", "")))
    atoms = primary_structure(row)

    parts = [part for part in Path(path_text).parts if part not in ("/", "")]
    lower_parts = [part.lower() for part in parts]
    if "copt" in lower_parts:
        idx = lower_parts.index("copt")
        surface_base = name_text.split("-copt-", 1)[0] if "-copt-" in name_text else ""
        if not surface_base and idx > 0:
            surface_base = parts[idx - 1]
        reaction = parts[idx + 1] if idx + 1 < len(parts) else ""
        path_id = parts[idx + 2] if idx + 2 < len(parts) else ""
        step = _safe_int(parts[idx + 3] if idx + 3 < len(parts) else None)
    else:
        match = NAME_COPT_PATTERN.match(name_text)
        surface_base = match.group("surface_base") if match else ""
        reaction = match.group("reaction") if match else ""
        path_id = match.group("path_id") if match else ""
        step = _safe_int(match.group("step") if match else None)

    series_id = None
    if surface_base and reaction and path_id:
        series_id = f"{dataset}|{surface_base}|{reaction}|{path_id}"

    initial_state = reaction.split("%", 1)[0] if "%" in reaction else ""
    final_state = reaction.split("%", 1)[1] if "%" in reaction else ""
    fixed_pairs, fixed_lengths = _extract_fixbond_constraints(atoms)

    return pd.Series(
        {
            "copt_surface_base": surface_base or np.nan,
            "copt_reaction": reaction or np.nan,
            "copt_path_id": path_id or np.nan,
            "copt_step": step if step is not None else np.nan,
            "copt_series_id": series_id,
            "copt_initial_state": initial_state or np.nan,
            "copt_final_state": final_state or np.nan,
            "copt_constraint_kind": "FixBondLengths" if fixed_pairs else np.nan,
            "copt_fixed_bond_pairs": fixed_pairs or None,
            "copt_fixed_bond_lengths_A": fixed_lengths or None,
        }
    )


def _safe_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _extract_fixbond_constraints(atoms: object) -> tuple[list[tuple[int, int]], list[float]]:
    if atoms is None or atoms.__class__.__name__ != "Atoms":
        return [], []
    pairs: list[tuple[int, int]] = []
    lengths: list[float] = []
    for constraint in getattr(atoms, "constraints", []) or []:
        if constraint.__class__.__name__ != "FixBondLengths":
            continue
        constraint_pairs = np.asarray(getattr(constraint, "pairs", []), dtype=int)
        for first, second in constraint_pairs:
            pairs.append((int(first), int(second)))
            try:
                lengths.append(float(atoms.get_distance(int(first), int(second), mic=True)))
            except Exception:
                lengths.append(float("nan"))
    return pairs, lengths
