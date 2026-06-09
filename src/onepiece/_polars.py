from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_polars() -> Any | None:
    try:
        import polars as pl
    except Exception:
        return None
    return pl


def dataframe_is_polars_safe(dataframe: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(series_is_polars_safe(dataframe[column]) for column in columns if column in dataframe.columns)


def series_is_polars_safe(series: pd.Series) -> bool:
    if not isinstance(series, pd.Series):
        return False
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return True
    sample = first_non_null(series)
    if sample is None:
        return True
    return is_polars_safe_value(sample)


def first_non_null(series: pd.Series) -> Any | None:
    values = series.dropna()
    if values.empty:
        return None
    return values.iloc[0]


def is_polars_safe_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return True
    except TypeError:
        logger.debug("Could not evaluate pd.isna() for value of type %s.", type(value).__name__)
    if isinstance(value, str | bytes | bool | int | float | np.generic | date | datetime | time | timedelta):
        return True
    return False
