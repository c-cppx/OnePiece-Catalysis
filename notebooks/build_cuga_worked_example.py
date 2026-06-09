from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from onepiece import add_ase_analysis_descriptors


DATASET = ROOT / "notebooks" / "phase_diagram_outputs" / "cuga_full_dataset.pkl"
DOCS_OUTPUT = ROOT / "docs" / "source" / "_static" / "worked_examples" / "cuga"
TABLE_OUTPUT = ROOT / "notebooks" / "phase_diagram_outputs" / "worked_example_tables"


def main() -> None:
    DOCS_OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)

    frame = pd.read_pickle(DATASET)
    enriched = add_ase_analysis_descriptors(frame, structure_column="struc", include_pdos=False)

    bulk = enriched.loc[enriched["hkl"].isna()].copy()
    surface = enriched.loc[enriched["hkl"].astype(str).isin(["100", "110", "111", "211"])].copy()

    _plot_bulk_formation_vs_ga(bulk)
    _plot_surface_energy_vs_monolayer(surface)
    _plot_gcn_vs_surface_energy(surface)
    _plot_charge_vs_gcn(enriched)
    _plot_slab_geometry_by_facet(surface)
    _write_summary_tables(bulk, surface, enriched)


def _plot_bulk_formation_vs_ga(bulk: pd.DataFrame) -> None:
    data = bulk.copy()
    data["Ga_percent_num"] = pd.to_numeric(data["Ga_percent"], errors="coerce")
    data["formation_energy_per_atom_num"] = pd.to_numeric(data["formation_energy_per_atom"], errors="coerce")
    data = data.dropna(subset=["Ga_percent_num", "formation_energy_per_atom_num"]).copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.scatter(
        data["Ga_percent_num"],
        data["formation_energy_per_atom_num"],
        s=52,
        alpha=0.85,
        color="#2f4858",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Ga content / at.%")
    ax.set_ylabel("Formation energy per atom / eV")
    ax.set_title("Cu/Ga bulk alloys: formation energy versus Ga content")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DOCS_OUTPUT / "bulk_formation_vs_ga.png", dpi=180)
    plt.close(fig)


def _plot_surface_energy_vs_monolayer(surface: pd.DataFrame) -> None:
    data = surface.copy()
    data["Monolayer_alloy_num"] = pd.to_numeric(data["Monolayer_alloy"], errors="coerce")
    data["form_G_per_Area_num"] = pd.to_numeric(data["form_G_per_Area"], errors="coerce")
    data = data.dropna(subset=["Monolayer_alloy_num", "form_G_per_Area_num", "hkl"]).copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    facets = sorted(data["hkl"].astype(str).unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(facets)))
    for color, facet in zip(colors, facets, strict=False):
        subset = data.loc[data["hkl"].astype(str).eq(facet)]
        ax.scatter(
            subset["Monolayer_alloy_num"],
            subset["form_G_per_Area_num"],
            s=48,
            alpha=0.82,
            color=color,
            label=facet,
            edgecolor="black",
            linewidth=0.35,
        )
    ax.set_xlabel("Monolayer alloy coverage")
    ax.set_ylabel("Surface free energy per area / eV A$^{-2}$")
    ax.set_title("Cu/Ga surfaces: free energy versus monolayer coverage")
    ax.legend(title="Facet", frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DOCS_OUTPUT / "surface_energy_vs_monolayer.png", dpi=180)
    plt.close(fig)


def _plot_gcn_vs_surface_energy(surface: pd.DataFrame) -> None:
    data = surface.copy()
    data["average_Ga_GCN_num"] = pd.to_numeric(data["average_Ga_GCN"], errors="coerce")
    data["form_G_per_Area_num"] = pd.to_numeric(data["form_G_per_Area"], errors="coerce")
    data = data.dropna(subset=["average_Ga_GCN_num", "form_G_per_Area_num"]).copy()
    if data.empty:
        return
    color_column = "hkl" if "hkl" in data.columns else None
    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    if color_column is None:
        ax.scatter(data["average_Ga_GCN_num"], data["form_G_per_Area_num"], s=50, alpha=0.85, color="#33658a")
    else:
        facets = sorted(data[color_column].astype(str).unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(facets)))
        for color, facet in zip(colors, facets, strict=False):
            subset = data.loc[data[color_column].astype(str).eq(facet)]
            ax.scatter(
                subset["average_Ga_GCN_num"],
                subset["form_G_per_Area_num"],
                s=50,
                alpha=0.82,
                color=color,
                label=facet,
                edgecolor="black",
                linewidth=0.35,
            )
        ax.legend(title="Facet", frameon=False)
    ax.set_xlabel("Average Ga generalized coordination number")
    ax.set_ylabel("Surface free energy per area / eV A$^{-2}$")
    ax.set_title("Ga coordination versus Cu/Ga surface free energy")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DOCS_OUTPUT / "gcn_vs_surface_energy.png", dpi=180)
    plt.close(fig)


def _plot_charge_vs_gcn(frame: pd.DataFrame) -> None:
    data = frame.copy()
    data["average_Ga_charge_num"] = pd.to_numeric(data["average_Ga_charge"], errors="coerce")
    data["average_Ga_GCN_num"] = pd.to_numeric(data["average_Ga_GCN"], errors="coerce")
    data = data.dropna(subset=["average_Ga_charge_num", "average_Ga_GCN_num"]).copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    kinds = np.where(data["hkl"].isna(), "bulk_like", data["hkl"].astype(str))
    labels = pd.Series(kinds, index=data.index)
    groups = list(dict.fromkeys(labels.astype(str)))
    colors = plt.cm.Dark2(np.linspace(0, 1, len(groups)))
    for color, label in zip(colors, groups, strict=False):
        subset = data.loc[labels.astype(str).eq(label)]
        ax.scatter(
            subset["average_Ga_GCN_num"],
            subset["average_Ga_charge_num"],
            s=48,
            alpha=0.82,
            color=color,
            label=label,
            edgecolor="black",
            linewidth=0.35,
        )
    ax.set_xlabel("Average Ga generalized coordination number")
    ax.set_ylabel("Average Ga charge / e")
    ax.set_title("Ga charge versus coordination across Cu/Ga structures")
    ax.legend(title="Family", frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DOCS_OUTPUT / "charge_vs_gcn.png", dpi=180)
    plt.close(fig)


def _plot_slab_geometry_by_facet(surface: pd.DataFrame) -> None:
    data = surface.copy()
    data["slab_thickness_num"] = pd.to_numeric(data["slab_thickness"], errors="coerce")
    data["vacuum_thickness_num"] = pd.to_numeric(data["vacuum_thickness"], errors="coerce")
    data = data.dropna(subset=["slab_thickness_num", "vacuum_thickness_num", "hkl"]).copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    facets = sorted(data["hkl"].astype(str).unique())
    colors = plt.cm.Paired(np.linspace(0, 1, len(facets)))
    for color, facet in zip(colors, facets, strict=False):
        subset = data.loc[data["hkl"].astype(str).eq(facet)]
        ax.scatter(
            subset["slab_thickness_num"],
            subset["vacuum_thickness_num"],
            s=50,
            alpha=0.82,
            color=color,
            label=facet,
            edgecolor="black",
            linewidth=0.35,
        )
    ax.set_xlabel("Slab thickness / A")
    ax.set_ylabel("Vacuum thickness / A")
    ax.set_title("ASE-derived slab geometry by Cu/Ga surface facet")
    ax.legend(title="Facet", frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DOCS_OUTPUT / "slab_geometry_by_facet.png", dpi=180)
    plt.close(fig)


def _write_summary_tables(bulk: pd.DataFrame, surface: pd.DataFrame, enriched: pd.DataFrame) -> None:
    bulk_summary = (
        bulk.assign(
            Ga_percent_num=pd.to_numeric(bulk["Ga_percent"], errors="coerce"),
            formation_energy_per_atom_num=pd.to_numeric(bulk["formation_energy_per_atom"], errors="coerce"),
        )
        .dropna(subset=["Ga_percent_num", "formation_energy_per_atom_num"])
        .sort_values("formation_energy_per_atom_num")
        [["Name", "Ga_percent", "formation_energy_per_atom", "Formula"]]
        .head(20)
    )
    bulk_summary.to_csv(TABLE_OUTPUT / "cuga_bulk_lowest_formation_energies.csv", index=False)

    surface_summary = (
        surface.assign(
            Monolayer_alloy_num=pd.to_numeric(surface["Monolayer_alloy"], errors="coerce"),
            form_G_per_Area_num=pd.to_numeric(surface["form_G_per_Area"], errors="coerce"),
        )
        .dropna(subset=["hkl", "Monolayer_alloy_num", "form_G_per_Area_num"])
        .sort_values("form_G_per_Area_num")
        [["Name", "hkl", "Monolayer_alloy", "form_G_per_Area", "average_Ga_GCN", "average_Ga_charge"]]
        .head(30)
    )
    surface_summary.to_csv(TABLE_OUTPUT / "cuga_surface_best_candidates.csv", index=False)

    geometry_summary = (
        enriched.dropna(subset=["slab_thickness", "vacuum_thickness"])
        [["Name", "hkl", "slabsize", "slab_thickness", "vacuum_thickness", "mean_coordination"]]
        .sort_values(["hkl", "slabsize", "Name"])
        .head(50)
    )
    geometry_summary.to_csv(TABLE_OUTPUT / "cuga_geometry_summary.csv", index=False)


if __name__ == "__main__":
    main()
