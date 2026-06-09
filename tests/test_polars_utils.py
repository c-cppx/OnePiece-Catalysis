from __future__ import annotations

import pandas as pd
from ase import Atoms

from onepiece._polars import dataframe_is_polars_safe, is_polars_safe_value, series_is_polars_safe


def test_polars_safe_value_accepts_scalar_text_and_numbers() -> None:
    assert is_polars_safe_value("Cu")
    assert is_polars_safe_value(1)
    assert is_polars_safe_value(1.5)
    assert is_polars_safe_value(True)


def test_polars_safe_value_rejects_ase_atoms_objects() -> None:
    assert is_polars_safe_value(Atoms("Cu")) is False


def test_series_is_polars_safe_rejects_object_series_with_atoms() -> None:
    series = pd.Series([Atoms("Cu")], dtype="object")

    assert series_is_polars_safe(series) is False


def test_dataframe_is_polars_safe_accepts_plain_scalar_columns() -> None:
    frame = pd.DataFrame({"Name": ["row"], "E": [-1.0], "dataset": ["base"]})

    assert dataframe_is_polars_safe(frame, ["Name", "E", "dataset"]) is True
