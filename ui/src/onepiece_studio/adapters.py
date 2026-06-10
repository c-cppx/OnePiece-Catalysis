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


def load_source_cached(source: DatabaseSource) -> pd.DataFrame:
    """Load a source, caching file-backed reads across Streamlit reruns.

    The cache key includes the file's mtime, so an updated file is reread.
    Falls back to a plain load for in-memory sources and unreadable paths
    (those raise their friendly error inside ``load``).
    """
    path = getattr(source, "path", None)
    if path is None:
        return source.load()
    resolved = Path(path)
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except OSError:
        return source.load()
    key = str(getattr(source, "key", "df"))
    return _cached_read(str(resolved), key, mtime_ns).copy()


_CACHED_READ_IMPL = None


def _cached_read(path: str, key: str, mtime_ns: int) -> pd.DataFrame:
    global _CACHED_READ_IMPL
    if _CACHED_READ_IMPL is None:
        import streamlit as st

        @st.cache_resource(max_entries=4, show_spinner="Loading dataset...")
        def _read(path: str, key: str, mtime_ns: int) -> pd.DataFrame:
            return read_hdf_path(Path(path), key=key)

        _CACHED_READ_IMPL = _read
    return _CACHED_READ_IMPL(path, key, mtime_ns)


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


