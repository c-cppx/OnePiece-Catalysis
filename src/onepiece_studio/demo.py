from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from ase.build import bulk

from onepiece_studio.adapters import DataFrameSource, HDFSource
from onepiece_studio.config import OnePieceStudioConfig

DEFAULT_LOCAL_HDF = os.environ.get("ONEPIECE_STUDIO_DEFAULT_HDF")


def demo_source() -> tuple[DataFrameSource, OnePieceStudioConfig]:
    atoms = [bulk("Cu", "fcc", a=3.6), bulk("Si", "diamond", a=5.43), bulk("Fe", "bcc", a=2.87)]
    dataframe = pd.DataFrame(
        {
            "id": ["cu-fcc-001", "si-dia-001", "fe-bcc-001"],
            "formula": [atom.get_chemical_formula() for atom in atoms],
            "crystal_system": ["cubic", "cubic", "cubic"],
            "energy_ev": [-3.49, -5.42, -4.28],
            "band_gap_ev": [0.0, 1.12, 0.0],
            "atoms": atoms,
            "preview_image": [
                "examples/assets/cu.png",
                "examples/assets/si.png",
                "examples/assets/fe.png",
            ],
            "status": ["validated", "candidate", "validated"],
        }
    )
    config = OnePieceStudioConfig(
        title="OnePiece Materials Database",
        primary_key="id",
        image_columns=["preview_image"],
        structure_columns=["atoms"],
        asset_root=Path.cwd(),
        metric_columns=["energy_ev", "band_gap_ev"],
    )
    return DataFrameSource(dataframe, name="demo-materials"), config


def empty_source() -> tuple[DataFrameSource, OnePieceStudioConfig]:
    dataframe = pd.DataFrame(
        columns=[
            "Name",
            "Formula",
            "Path",
            "struc",
            "CONTCAR",
            "E",
            "fmax",
        ]
    )
    config = OnePieceStudioConfig(
        title="OnePiece Studio",
        primary_key="Name",
        structure_columns=["struc", "CONTCAR", "structure", "atoms"],
        searchable_columns=[
            "Name",
            "Formula",
            "Path",
            "dataset",
            "dataset_label",
            "source_hdf",
        ],
        metric_columns=["E", "fmax", "a", "b", "c", "gamma", "timestamp"],
    )
    return DataFrameSource(dataframe, name="empty-session"), config


def local_default_source() -> tuple[HDFSource | DataFrameSource, OnePieceStudioConfig]:
    if DEFAULT_LOCAL_HDF:
        path = Path(DEFAULT_LOCAL_HDF).expanduser()
    else:
        path = None
    if path is not None and path.exists():
        source = HDFSource(path, key="df", name=path.name)
        config = OnePieceStudioConfig(
            title="OnePiece Studio: Local Materials Database",
            primary_key="Name",
            structure_columns=["struc", "CONTCAR", "structure", "atoms"],
            searchable_columns=[
                "Name",
                "Formula",
                "Path",
                "human_time",
                "dataset",
                "dataset_label",
                "source_hdf",
            ],
            metric_columns=["E", "fmax", "a", "b", "c", "gamma", "timestamp"],
        )
        return source, config
    return empty_source()
