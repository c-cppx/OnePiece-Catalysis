from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
PHASE = ROOT / "notebooks" / "phase_diagram_outputs"
CACHE = PHASE / "cuga_full_dataset.pkl"


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close()


def copy_existing_figures() -> None:
    for name in [
        "cuga_phase_diagram_2d_teaching.png",
        "cuga_phase_diagram_2d.png",
        "cuga_surface_100_phase_diagram_2d.png",
        "cuga_surface_110_phase_diagram_2d.png",
        "cuga_surface_111_phase_diagram_2d.png",
        "cuga_surface_211_phase_diagram_2d.png",
        "cuga_bulk_surface_transition_phase_multiplot.png",
    ]:
        src = PHASE / name
        if src.exists():
            shutil.copy2(src, FIGURES / name)


def dataset_overview(df: pd.DataFrame) -> None:
    counts = df["dataset"].value_counts().sort_values()
    plt.figure(figsize=(8.0, 4.4))
    colors = ["#5f6f91", "#2f7f8c", "#86b995", "#d9a441", "#d74f42", "#8a6f3d", "#6d7f72", "#9b6b5d"]
    plt.barh(counts.index, counts.values, color=colors[: len(counts)])
    plt.xlabel("Number of calculations")
    plt.ylabel("Dataset")
    plt.title("Cu/Ga database composition by HDF source")
    for y, value in enumerate(counts.values):
        plt.text(value + 1, y, str(value), va="center", fontsize=9)
    savefig("dataset_counts.png")


def fmax_quality(df: pd.DataFrame) -> None:
    fmax = clean_number(df["fmax"]).dropna()
    plt.figure(figsize=(8.0, 4.4))
    plt.hist(fmax, bins=35, color="#2f7f8c", edgecolor="white")
    plt.axvline(0.05, color="#d74f42", linewidth=2, label="review threshold: 0.05 eV/A")
    plt.xlabel("Maximum residual force fmax [eV/A]")
    plt.ylabel("Number of calculations")
    plt.title("Relaxation-quality distribution")
    plt.legend(frameon=False)
    savefig("fmax_distribution.png")


def bulk_energy_vs_composition(df: pd.DataFrame) -> None:
    bulk = df[df["dataset"].isin(["bulk", "bulk_oxide"])].copy()
    bulk["formation_energy_per_atom_numeric"] = clean_number(bulk["formation_energy_per_atom"])
    bulk = bulk.dropna(subset=["Ga_percent", "formation_energy_per_atom_numeric"])
    plt.figure(figsize=(7.5, 4.8))
    for dataset, group in bulk.groupby("dataset"):
        plt.scatter(
            group["Ga_percent"],
            group["formation_energy_per_atom_numeric"],
            s=48,
            label=dataset,
            alpha=0.85,
        )
    plt.axhline(0, color="#17212b", linewidth=1, alpha=0.55)
    plt.xlabel("Ga content [at. %]")
    plt.ylabel("Formation energy [eV/atom]")
    plt.title("Bulk Cu/Ga formation energies")
    plt.legend(frameon=False)
    savefig("bulk_energy_vs_composition.png")


def surface_energy_vs_coverage(df: pd.DataFrame) -> None:
    surface = df[df["dataset"].str.contains("surface", na=False)].copy()
    surface["form_G_per_Area_numeric"] = clean_number(surface["form_G_per_Area"])
    surface = surface.dropna(subset=["Monolayer_alloy", "form_G_per_Area_numeric"])
    plt.figure(figsize=(8.0, 5.0))
    for dataset, group in surface.groupby("dataset"):
        if len(group) < 2:
            continue
        plt.scatter(
            group["Monolayer_alloy"],
            group["form_G_per_Area_numeric"],
            s=44,
            alpha=0.82,
            label=dataset.replace("surface_", "hkl "),
        )
    plt.axhline(0, color="#17212b", linewidth=1, alpha=0.55)
    plt.xlabel("Ga surface coverage [% monolayer]")
    plt.ylabel("Corrected surface free energy [eV/A$^2$]")
    plt.title("Surface stability candidates across Miller indices")
    plt.legend(frameon=False, ncols=2, fontsize=8)
    savefig("surface_energy_vs_coverage.png")


def stable_fraction_chart() -> None:
    path = PHASE / "cuga_bulk_surface_transition_phase_summary_extended.csv"
    if not path.exists():
        return
    summary = pd.read_csv(path)
    top = summary.sort_values("stable_grid_fraction", ascending=False).head(16)
    labels = top["panel"].astype(str) + " | " + top["Name"].astype(str)
    plt.figure(figsize=(10.0, 6.0))
    plt.barh(labels[::-1], top["stable_grid_fraction"].iloc[::-1] * 100, color="#2f7f8c")
    plt.xlabel("Stable grid fraction [%]")
    plt.ylabel("Phase")
    plt.title("Largest stability fields in the computed T--gas-ratio window")
    savefig("largest_stable_fields.png")


def export_tables(df: pd.DataFrame) -> None:
    dataset_summary = (
        df.groupby("dataset")
        .agg(rows=("Name", "size"), unique_formulas=("Formula", "nunique"), mean_fmax=("fmax", "mean"))
        .reset_index()
    )
    dataset_summary.to_csv(TABLES / "dataset_summary.csv", index=False)

    if (PHASE / "cuga_bulk_surface_transition_phase_summary_extended.csv").exists():
        summary = pd.read_csv(PHASE / "cuga_bulk_surface_transition_phase_summary_extended.csv")
        summary.head(20).to_csv(TABLES / "top_transition_phases.csv", index=False)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(CACHE)
    copy_existing_figures()
    dataset_overview(df)
    fmax_quality(df)
    bulk_energy_vs_composition(df)
    surface_energy_vs_coverage(df)
    stable_fraction_chart()
    export_tables(df)
    print(FIGURES)
    print(TABLES)


if __name__ == "__main__":
    main()
