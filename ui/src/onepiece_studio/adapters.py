from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from onepiece.frame_utils import ensure_name_index
from onepiece.sources.core import read_hdf_path


@runtime_checkable
class DatabaseSource(Protocol):
    """Small protocol every OnePiece Studio data source should satisfy."""

    name: str

    def load(self) -> pd.DataFrame:
        """Return the current table as a DataFrame."""


@dataclass(slots=True)
class DataFrameSource:
    dataframe: pd.DataFrame
    name: str = "database"

    def load(self) -> pd.DataFrame:
        return ensure_name_index(self.dataframe.copy())


@dataclass(slots=True)
class HDFSource:
    path: Path | str
    key: str = "df"
    name: str | None = None
    numpy_pickle_compat: bool = True

    def load(self) -> pd.DataFrame:
        return read_hdf_path(
            Path(self.path),
            key=self.key,
            numpy_pickle_compat=self.numpy_pickle_compat,
        )

    @property
    def display_name(self) -> str:
        return self.name or Path(self.path).name


@dataclass(slots=True)
class OnePieceSource:
    """Adapter for OnePiece-like objects without coupling OnePiece Studio to one API."""

    onepiece: Any
    name: str = "onepiece"

    def load(self) -> pd.DataFrame:
        if hasattr(self.onepiece, "to_dataframe"):
            return ensure_name_index(self.onepiece.to_dataframe().copy())
        if hasattr(self.onepiece, "dataframe"):
            return ensure_name_index(self.onepiece.dataframe.copy())
        if hasattr(self.onepiece, "df"):
            return ensure_name_index(self.onepiece.df.copy())
        raise TypeError(
            "OnePieceSource needs an object with to_dataframe(), .dataframe, or .df."
        )


