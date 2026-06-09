from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from onepiece import add_catalysis_hub_adsorption_energies, bundled_catalysis_hub_dataset


DOCS_OUTPUT = ROOT / "docs" / "source" / "_static" / "worked_examples" / "catalysis_hub"
NOTEBOOK_OUTPUT = ROOT / "notebooks" / "catalysis_hub_tutorial" / "outputs"


def main() -> None:
    DOCS_OUTPUT.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_OUTPUT.mkdir(parents=True, exist_ok=True)

    frame = pd.read_hdf(bundled_catalysis_hub_dataset(), key="df")
    enriched = add_catalysis_hub_adsorption_energies(frame)
    enriched["surface_label"] = (
        enriched["surfaceComposition"].fillna("").astype(str).replace("", "unknown")
        + "("
        + enriched["facet"].fillna("").astype(str).replace("", "NA")
        + ")"
    )

    _plot_reaction_energy_vs_barrier(enriched)
    _plot_co2_adsorption_parity(enriched)
    _plot_co2_adsorption_heatmap(enriched)
    _plot_hydrogenation_branch_gap(enriched)
    _write_summary_tables(enriched)


def _plot_reaction_energy_vs_barrier(frame: pd.DataFrame) -> None:
    data = frame.dropna(subset=["reactionEnergy", "activationEnergy"]).copy()
    if data.empty:
        return
    top_surfaces = data["surface_label"].value_counts().head(8).index
    filtered = data.loc[data["surface_label"].isin(top_surfaces)]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    palette = plt.cm.tab10(np.linspace(0, 1, len(top_surfaces)))
    for color, label in zip(palette, top_surfaces, strict=False):
        subset = filtered.loc[filtered["surface_label"].eq(label)]
        ax.scatter(
            subset["reactionEnergy"],
            subset["activationEnergy"],
            s=48,
            alpha=0.85,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.axvline(0.0, color="#666666", linewidth=0.9, linestyle="--")
    ax.axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Reaction energy / eV")
    ax.set_ylabel("Activation energy / eV")
    ax.set_title("Catalysis-Hub CO2 subset: reaction energy versus barrier")
    ax.legend(title="Surface", frameon=False, loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DOCS_OUTPUT / "reaction_energy_vs_barrier.png", dpi=180)
    plt.close(fig)


def _plot_co2_adsorption_parity(frame: pd.DataFrame) -> None:
    data = frame.loc[frame["reaction_system_name"].astype(str).eq("CO2star")].copy()
    data = data.dropna(subset=["adsorption_energy", "reactionEnergy"])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.scatter(
        data["reactionEnergy"],
        data["adsorption_energy"],
        s=58,
        alpha=0.9,
        color="#2f4858",
        edgecolor="black",
        linewidth=0.4,
    )
    lo = float(min(data["reactionEnergy"].min(), data["adsorption_energy"].min()))
    hi = float(max(data["reactionEnergy"].max(), data["adsorption_energy"].max()))
    pad = 0.15 * max(1.0, hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#d33f49", linewidth=1.2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Published reaction energy / eV")
    ax.set_ylabel("Reconstructed adsorption energy / eV")
    ax.set_title("CO2 adsorption parity check on Catalysis-Hub rows")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DOCS_OUTPUT / "co2_adsorption_parity.png", dpi=180)
    plt.close(fig)


def _plot_co2_adsorption_heatmap(frame: pd.DataFrame) -> None:
    data = frame.loc[frame["Equation"].astype(str).eq("CO2(g) + * -> CO2*")].copy()
    data = data.dropna(subset=["adsorption_energy"])
    if data.empty:
        return
    pivot = (
        data.groupby(["surfaceComposition", "facet"])["adsorption_energy"]
        .median()
        .unstack("facet")
        .sort_index()
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, max(4.8, 0.42 * len(pivot.index) + 1.5)))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=pivot.min().min(), vmax=pivot.max().max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    ax.set_xlabel("Facet")
    ax.set_ylabel("Surface composition")
    ax.set_title("Median reconstructed CO2 adsorption energy by surface and facet")
    cbar = fig.colorbar(image, ax=ax, shrink=0.9)
    cbar.set_label("Adsorption energy / eV")
    fig.tight_layout()
    fig.savefig(DOCS_OUTPUT / "co2_adsorption_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_hydrogenation_branch_gap(frame: pd.DataFrame) -> None:
    branch_data = frame.loc[
        frame["Equation"].astype(str).isin(
            ["CO2(g) + 0.5H2(g) + * -> HCOO*", "CO2(g) + 0.5H2(g) + * -> COOH*"]
        )
    ].copy()
    if branch_data.empty:
        return
    grouped = (
        branch_data.groupby(["surface_label", "Equation"])["reactionEnergy"]
        .median()
        .unstack("Equation")
        .dropna()
    )
    if grouped.empty:
        return
    grouped["hcoo_minus_cooh_eV"] = (
        grouped["CO2(g) + 0.5H2(g) + * -> HCOO*"] - grouped["CO2(g) + 0.5H2(g) + * -> COOH*"]
    )
    grouped = grouped.sort_values("hcoo_minus_cooh_eV")
    fig, ax = plt.subplots(figsize=(8.6, max(4.8, 0.35 * len(grouped.index) + 1.2)))
    colors = ["#2f4858" if value <= 0 else "#d33f49" for value in grouped["hcoo_minus_cooh_eV"]]
    ax.barh(grouped.index.astype(str), grouped["hcoo_minus_cooh_eV"], color=colors)
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Median reaction-energy gap (HCOO* minus COOH*) / eV")
    ax.set_ylabel("Surface")
    ax.set_title("Hydrogenation branch preference across Catalysis-Hub surfaces")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(DOCS_OUTPUT / "hydrogenation_branch_gap.png", dpi=180)
    plt.close(fig)


def _write_summary_tables(frame: pd.DataFrame) -> None:
    adsorption_rows = frame.loc[frame["reaction_system_name"].astype(str).eq("CO2star")].copy()
    adsorption_rows = adsorption_rows.dropna(subset=["adsorption_energy", "reactionEnergy"])
    adsorption_rows = adsorption_rows[
        [
            "surfaceComposition",
            "facet",
            "reactionEnergy",
            "adsorption_energy",
            "adsorption_energy_delta_vs_reactionEnergy",
        ]
    ].sort_values(["surfaceComposition", "facet"])
    adsorption_rows.to_csv(NOTEBOOK_OUTPUT / "catalysis_hub_co2_adsorption_parity.csv", index=False)

    branch_data = frame.loc[
        frame["Equation"].astype(str).isin(
            ["CO2(g) + 0.5H2(g) + * -> HCOO*", "CO2(g) + 0.5H2(g) + * -> COOH*"]
        )
    ].copy()
    if not branch_data.empty:
        branch_summary = (
            branch_data.groupby(["surface_label", "Equation"])["reactionEnergy"]
            .median()
            .unstack("Equation")
            .dropna()
            .sort_index()
        )
        branch_summary["hcoo_minus_cooh_eV"] = (
            branch_summary["CO2(g) + 0.5H2(g) + * -> HCOO*"]
            - branch_summary["CO2(g) + 0.5H2(g) + * -> COOH*"]
        )
        branch_summary.to_csv(NOTEBOOK_OUTPUT / "catalysis_hub_hydrogenation_branch_summary.csv")


if __name__ == "__main__":
    main()
