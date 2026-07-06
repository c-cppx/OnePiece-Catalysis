"""Reusable dataframe cleaning helpers for adsorption workflows."""

from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd


def assign_surface_reference_descendants(
    frame: pd.DataFrame,
    references: pd.DataFrame,
    *,
    path_columns: Sequence[str] = ("relative_path", "Path"),
    mark_descendants_as_adsorbates: bool = True,
) -> pd.DataFrame:
    """Map rows inside a reference folder back to that clean surface key."""
    df = frame.copy()
    df["surface_key_from_reference_path"] = False
    if references.empty:
        return df

    for column in tuple(str(column) for column in path_columns):
        if column not in df.columns or column not in references.columns:
            continue
        prefixes = _reference_path_prefixes(references, column)
        if not prefixes:
            continue
        paths = df[column].map(_normalize_path_text)
        for prefix, surface_key, reference_name in prefixes:
            descendant = paths.str.startswith(f"{prefix}/", na=False)
            if "Name" in df.columns:
                descendant &= ~df["Name"].astype(str).eq(str(reference_name))
            descendant &= ~df["surface_key_from_reference_path"].astype(bool)
            df.loc[descendant, "surface_key"] = surface_key
            if mark_descendants_as_adsorbates:
                df.loc[descendant, "is_adsorbate"] = True
            df.loc[descendant, "surface_key_from_reference_path"] = True
    return df


def drop_nested_reference_candidates(
    candidates: pd.DataFrame,
    *,
    path_columns: Sequence[str] = ("relative_path", "Path"),
) -> pd.DataFrame:
    """Keep only top-level reference candidates when references are nested."""
    for column in tuple(str(column) for column in path_columns):
        if column not in candidates.columns:
            continue
        paths = candidates[column].map(_normalize_path_text)
        indexed_paths = {index: path for index, path in paths.items() if path}
        if not indexed_paths:
            continue
        nested_indices = {
            index
            for index, path in indexed_paths.items()
            if any(
                index != parent_index and _is_descendant_path(path, parent_path)
                for parent_index, parent_path in indexed_paths.items()
            )
        }
        if nested_indices:
            candidates = candidates.drop(index=list(nested_indices))
    return candidates.copy()


def drop_name_excluded_reference_candidates(
    candidates: pd.DataFrame,
    *,
    substrings: Sequence[str],
    patterns: Sequence[str | re.Pattern[str]],
) -> pd.DataFrame:
    """Drop reference candidates whose names match configured exclusions."""
    if "Name" not in candidates.columns:
        return candidates.copy()
    names = candidates["Name"].fillna("").astype(str)
    excluded = pd.Series(False, index=candidates.index)
    for token in substrings:
        text = str(token)
        if text:
            excluded |= names.str.contains(text, case=False, regex=False, na=False)
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
        excluded |= names.map(lambda value, regex=regex: bool(regex.search(value)))
    return candidates.loc[~excluded].copy()


def _reference_path_prefixes(references: pd.DataFrame, column: str) -> list[tuple[str, str, str]]:
    prefixes: list[tuple[str, str, str]] = []
    for _, row in references.iterrows():
        path = _normalize_path_text(row.get(column))
        surface_key = str(row.get("surface_key", "")).strip()
        name = str(row.get("Name", "")).strip()
        if path and surface_key:
            prefixes.append((path, surface_key, name))
    return sorted(set(prefixes), key=lambda item: len(item[0]))


def _is_descendant_path(path: str, parent_path: str) -> bool:
    return path != parent_path and path.startswith(f"{parent_path}/")


def _normalize_path_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip().replace("\\", "/").rstrip("/")
