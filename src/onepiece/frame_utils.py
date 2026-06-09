from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def ensure_name_index(
    frame: pd.DataFrame,
    *,
    name_column: str = "Name",
    fallback_prefix: str = "row",
) -> pd.DataFrame:
    """Ensure that a dataframe uses the Name column as canonical index.

    The Name column is preserved and the index is set to that same string key.
    Missing values are filled from the previous index or from a fallback prefix.
    """
    df = frame.copy()
    if name_column not in df.columns:
        df[name_column] = [str(value) if str(value).strip() else f"{fallback_prefix}-{i}" for i, value in enumerate(df.index)]
    names = df[name_column].astype(str).str.strip()
    if names.eq("").any():
        rebuilt: list[str] = []
        for position, (index_value, name_value) in enumerate(zip(df.index, names, strict=False)):
            if name_value:
                rebuilt.append(name_value)
                continue
            fallback = str(index_value).strip()
            rebuilt.append(fallback if fallback else f"{fallback_prefix}-{position}")
        names = pd.Series(rebuilt, index=df.index, dtype="string")
        df[name_column] = names
    else:
        df[name_column] = names
    df.index = pd.Index(df[name_column].astype(str), name=name_column)
    return df


def row_name(row: pd.Series, *, name_column: str = "Name") -> str:
    """Return the canonical row name, preferring the Name column over the index."""
    text = str(row.get(name_column, "")).strip()
    if text:
        return text
    index_value = getattr(row, "name", "")
    return str(index_value).strip()


def first_present(values: Iterable[Any]) -> Any:
    """Return the first non-empty value from an iterable."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip() if isinstance(value, str) else value
        if text == "":
            continue
        return value
    return None
