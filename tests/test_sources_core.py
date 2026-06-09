from __future__ import annotations

from pathlib import Path

from onepiece.sources.core import _friendly_hdf_read_error


def test_friendly_hdf_read_error_explains_missing_sympy() -> None:
    message = _friendly_hdf_read_error(
        Path("/tmp/example.hdf"),
        key="df",
        error=ModuleNotFoundError("No module named 'sympy'"),
    )

    assert "missing the optional dependency 'sympy'" in message
    assert "pip install onepiece" in message
    assert "pip install onepiece-studio" in message


def test_friendly_hdf_read_error_explains_missing_key() -> None:
    message = _friendly_hdf_read_error(
        Path("/tmp/example.hdf"),
        key="rows",
        error=KeyError("No object named rows in the file"),
    )

    assert "requested HDF key was not found" in message
    assert "'rows'" in message
