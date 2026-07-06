"""Formula parsing, structure stoichiometry, and adsorbate name detection."""

from __future__ import annotations

import re

import pandas as pd

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
DEFAULT_NAME_DELIMITERS = "-_%"
DEFAULT_REFERENCE_DESCENDANT_MARKERS = ("copt", "ci")


def compile_adsorbate_pattern(
    tokens: tuple[str, ...] = DEFAULT_ADSORBATE_TOKENS,
    *,
    delimiters: str = DEFAULT_NAME_DELIMITERS,
) -> re.Pattern[str]:
    """Compile a delimiter-aware adsorbate token pattern.

    Examples
    --------
    >>> pattern = compile_adsorbate_pattern(("NO3",))
    >>> bool(pattern.search("Cu-111-clean-NO3-1"))
    True
    """
    ordered = sorted({token for token in tokens if token}, key=len, reverse=True)
    if not ordered:
        return re.compile(r"a^")
    choices = "|".join(re.escape(token) for token in ordered)
    delimiter_class = re.escape(delimiters)
    return re.compile(rf"[{delimiter_class}]({choices})(?:[{delimiter_class}].*|$)", re.IGNORECASE)


def compile_reference_descendant_pattern(
    markers: tuple[str, ...] = DEFAULT_REFERENCE_DESCENDANT_MARKERS,
    *,
    delimiters: str = "-",
) -> re.Pattern[str]:
    """Compile a suffix pattern for rows nested below a surface reference.

    Examples
    --------
    >>> pattern = compile_reference_descendant_pattern(("neb",))
    >>> bool(pattern.search("Cu-111-clean-neb-00"))
    True
    """
    ordered = sorted({str(marker) for marker in markers if str(marker)}, key=len, reverse=True)
    if not ordered:
        return re.compile(r"a^")
    choices = "|".join(re.escape(marker) for marker in ordered)
    delimiter_class = re.escape(delimiters)
    return re.compile(rf"[{delimiter_class}](?:{choices})(?:[{delimiter_class}]|$).*$", re.IGNORECASE)


ADSORBATE_PATTERN = compile_adsorbate_pattern(DEFAULT_ADSORBATE_TOKENS)
FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")
NAME_COPT_PATTERN = re.compile(
    r"^(?P<surface_base>.*)-copt-(?P<reaction>.+)-(?P<path_id>[^-]+)-(?P<step>\d+)$"
)
REFERENCE_DESCENDANT_PATTERN = compile_reference_descendant_pattern(DEFAULT_REFERENCE_DESCENDANT_MARKERS)


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


def guess_adsorbate(
    name: object,
    *,
    tokens: tuple[str, ...] = DEFAULT_ADSORBATE_TOKENS,
    pattern: re.Pattern[str] | None = None,
) -> str:
    """Guess the adsorbate token from a calculation name."""
    if not isinstance(name, str):
        return ""
    active_pattern = pattern or (ADSORBATE_PATTERN if tokens == DEFAULT_ADSORBATE_TOKENS else compile_adsorbate_pattern(tokens))
    match = active_pattern.search(name)
    if not match:
        return ""
    token = match.group(1)
    if len(token) == 1 and token != token.upper():
        return ""
    return token


def surface_key_from_name(
    name: object,
    *,
    tokens: tuple[str, ...] = DEFAULT_ADSORBATE_TOKENS,
    adsorbate_pattern: re.Pattern[str] | None = None,
    descendant_markers: tuple[str, ...] = DEFAULT_REFERENCE_DESCENDANT_MARKERS,
    descendant_pattern: re.Pattern[str] | None = None,
) -> str:
    """Remove adsorbate and constrained-optimization suffixes from a row name."""
    if not isinstance(name, str):
        return ""
    copt_match = NAME_COPT_PATTERN.match(name)
    if copt_match:
        return copt_match.group("surface_base")
    active_descendant_pattern = (
        descendant_pattern
        or (
            REFERENCE_DESCENDANT_PATTERN
            if descendant_markers == DEFAULT_REFERENCE_DESCENDANT_MARKERS
            else compile_reference_descendant_pattern(descendant_markers)
        )
    )
    descendant_match = active_descendant_pattern.search(name)
    if descendant_match:
        return name[: descendant_match.start()]
    active_adsorbate_pattern = (
        adsorbate_pattern
        or (ADSORBATE_PATTERN if tokens == DEFAULT_ADSORBATE_TOKENS else compile_adsorbate_pattern(tokens))
    )
    match = active_adsorbate_pattern.search(name)
    if match and len(match.group(1)) == 1 and match.group(1) != match.group(1).upper():
        return name
    return active_adsorbate_pattern.sub("", name)
