from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import pandas as pd

from onepiece.adsorption import add_catalysis_hub_adsorption_energies


@dataclass(frozen=True, slots=True)
class SelfTestResult:
    name: str
    passed: bool
    details: dict[str, Any]


def bundled_catalysis_hub_dataset() -> Path:
    resource = files("onepiece.data").joinpath("catalysis_hub_co2_subset.hdf")
    with as_file(resource) as local_path:
        return Path(local_path)


def run_catalysis_hub_self_test(dataset_path: str | Path | None = None) -> SelfTestResult:
    path = Path(dataset_path) if dataset_path is not None else bundled_catalysis_hub_dataset()
    frame = pd.read_hdf(path, key="df").copy()
    analysed = add_catalysis_hub_adsorption_energies(frame)

    adsorption_rows = analysed.loc[
        analysed.get("cathub_system_kind", pd.Series(index=analysed.index, dtype=object)).astype(str).eq("adsorbate")
    ].copy()
    computed = adsorption_rows.loc[pd.to_numeric(adsorption_rows.get("adsorption_energy"), errors="coerce").notna()].copy()

    max_abs_delta = None
    if "adsorption_energy_delta_vs_reactionEnergy" in analysed.columns:
        deltas = pd.to_numeric(analysed["adsorption_energy_delta_vs_reactionEnergy"], errors="coerce").dropna()
        if not deltas.empty:
            max_abs_delta = float(deltas.abs().max())

    passed = (
        len(frame) > 0
        and len(adsorption_rows) > 0
        and len(computed) > 0
        and (max_abs_delta is not None and max_abs_delta < 1e-8)
    )

    details = {
        "dataset_path": str(path),
        "rows": int(len(frame)),
        "adsorbate_rows": int(len(adsorption_rows)),
        "computed_adsorption_rows": int(len(computed)),
        "max_abs_delta_vs_reaction_energy_ev": max_abs_delta,
        "surface_count": int(analysed.get("surfaceComposition", pd.Series(dtype=object)).astype(str).nunique()),
    }
    return SelfTestResult(name="catalysis-hub", passed=passed, details=details)


def format_self_test_result(result: SelfTestResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    lines = [f"[{status}] {result.name} self-test"]
    for key, value in result.details.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)
