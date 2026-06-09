from __future__ import annotations

from pathlib import Path

import pandas as pd
from ase import Atoms

from onepiece import (
    DatasetManifest,
    detect_storage_format,
    ensure_name_index,
    ensure_storage_layout,
    load_dataset,
    read_dataset_manifest,
    read_dataset_path,
    resolve_storage_config,
    save_dataset,
)


def test_save_and_load_dataset_roundtrip_with_parquet_and_object_sidecar(tmp_path: Path) -> None:
    config = ensure_storage_layout(resolve_storage_config(tmp_path / ".onepiece"))
    frame = pd.DataFrame(
        {
            "Name": ["calc-a"],
            "E": [-1.23],
            "Path": ["/tmp/calc-a"],
            "struc": [Atoms("Cu2")],
            "input_kpoints_grid": [(4, 4, 1)],
        }
    )

    manifest_path = save_dataset(frame, dataset_id="cugA-study", config=config, storage_format="parquet")
    loaded, manifest = load_dataset(manifest_path.parent)

    assert manifest_path.exists()
    assert isinstance(manifest, DatasetManifest)
    assert manifest.storage_format == "parquet"
    assert detect_storage_format(manifest_path.parent) == "parquet"
    assert loaded.index.name == "Name"
    assert loaded.index.tolist() == ["calc-a"]
    assert loaded.loc["calc-a", "input_kpoints_grid"] == (4, 4, 1)
    assert loaded.loc["calc-a", "struc"].__class__.__name__ == "Atoms"


def test_read_dataset_path_loads_manifest_backed_dataset(tmp_path: Path) -> None:
    config = ensure_storage_layout(resolve_storage_config(tmp_path / ".onepiece"))
    frame = ensure_name_index(pd.DataFrame({"Name": ["row-a"], "E": [1.0]}))
    manifest_path = save_dataset(frame, dataset_id="dataset-a", config=config)

    loaded = read_dataset_path(manifest_path.parent, key="df")

    assert loaded.index.tolist() == ["row-a"]
    assert loaded.loc["row-a", "E"] == 1.0
    reloaded_manifest = read_dataset_manifest(manifest_path)
    assert reloaded_manifest.dataset_id == "dataset-a"


def test_save_and_load_dataset_preserve_duplicate_names(tmp_path: Path) -> None:
    config = ensure_storage_layout(resolve_storage_config(tmp_path / ".onepiece"))
    frame = pd.DataFrame(
        {
            "Name": ["dup", "dup", "unique"],
            "E": [1.0, 2.0, 3.0],
            "struc": [Atoms("H"), Atoms("He"), Atoms("Li")],
        }
    )

    manifest_path = save_dataset(frame, dataset_id="dups", config=config, storage_format="parquet")
    loaded, _manifest = load_dataset(manifest_path.parent)

    assert len(loaded) == 3
    assert loaded["E"].tolist() == [1.0, 2.0, 3.0]
    assert int(loaded.index.duplicated().sum()) == 1
