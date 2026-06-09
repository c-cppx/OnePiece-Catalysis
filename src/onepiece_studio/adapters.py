from __future__ import annotations

import logging
import os
import subprocess  # nosec B404
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from onepiece.frame_utils import ensure_name_index
from onepiece.storage import load_dataset

logger = logging.getLogger(__name__)


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
        source_path = Path(self.path)
        if source_path.is_dir() or source_path.suffix.lower() in {".parquet", ".pq", ".json"}:
            frame, _manifest = load_dataset(source_path)
            return ensure_name_index(frame)
        if self.numpy_pickle_compat:
            _install_numpy_pickle_compat()
        try:
            return ensure_name_index(pd.read_hdf(self.path, key=self.key).copy())
        except Exception as exc:
            try:
                return ensure_name_index(pd.read_hdf(self.path, key=self.key).copy())
            except Exception:
                return ensure_name_index(_read_hdf_with_helper_python(Path(self.path), key=self.key, original_error=exc))

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


def _install_numpy_pickle_compat() -> None:
    """Allow reading HDF files pickled with NumPy 2 from NumPy 1 environments."""
    import ase.constraints  # noqa: F401
    import numpy as np
    import numpy.core as numpy_core

    for module_name in ("scipy.linalg", "sympy", "tables"):
        try:
            __import__(module_name)
        except Exception as exc:
            logger.debug("Optional compatibility preload for %s skipped: %s", module_name, exc)

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def _read_hdf_with_helper_python(path: Path, *, key: str, original_error: Exception) -> pd.DataFrame:
    helper_python = _helper_python_path()
    if helper_python is None:
        raise original_error
    output = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", prefix="onepiece_studio_hdf_").name)
    script = """
from pathlib import Path
import sys
import numpy as np
import pandas as pd
try:
    import numpy.core as numpy_core
    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    import scipy.linalg  # noqa: F401
    import ase.constraints  # noqa: F401
    import sympy  # noqa: F401
except Exception:
    pass
source = Path(sys.argv[1])
key = sys.argv[2]
target = Path(sys.argv[3])
pd.read_hdf(source, key=key).to_pickle(target)
"""
    # The helper executable and arguments are fully constructed in-process.
    completed = subprocess.run(  # nosec B603
        [str(helper_python), "-c", script, str(path), key, str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{original_error}. Helper reader also failed: {detail}")
    # The pickle is a temporary artifact produced by the helper process above.
    return pd.read_pickle(output)  # nosec B301


def _helper_python_path() -> Path | None:
    candidates = []
    configured = os.environ.get("ONEPIECE_STUDIO_HELPER_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path(sys.executable),
            Path("/opt/homebrew/bin/python3"),
            Path("/usr/local/bin/python3"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
