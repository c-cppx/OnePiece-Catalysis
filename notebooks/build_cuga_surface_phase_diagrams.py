from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib.colors import ListedColormap

from onepiece.phase_diagrams import build_corrected_phase_expressions, build_phase_field_grid

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "notebooks" / "phase_diagram_outputs" / "cuga_full_dataset.pkl"
OUTPUT_DIR = ROOT / "notebooks" / "phase_diagram_outputs"

FACETS = ("100", "110", "111", "211")

# Keep the same gas-phase constants already used in the Cu/Ga tutorial notebooks.
CONSTANTS = {
    "H2_E": -6.737,
    "H2O_E": -12.062,
    "GA2O3_E": -22.55101755,
    "H2_S": 1.5044e-3,
    "H2O_S": 2.2198e-3,
    "kb": 8.617333262145e-5,
}

T_MIN_K = 300.0
T_MAX_K = 1100.0
N_T = 161
LOG10_RATIO_MIN = -16.0
LOG10_RATIO_MAX = 4.0
N_RATIO = 201
MIN_REGION_LABEL_PERCENT = 2.0
QUALITATIVE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#9edae5",
    "#f7b6d2",
]


@dataclass(frozen=True)
class GridDefinition:
    t_values: np.ndarray
    log10_ratio_values: np.ndarray
    tt: np.ndarray
    yy_log10: np.ndarray
    log_ratio_natural: np.ndarray


def build_grid() -> GridDefinition:
    t_values = np.linspace(T_MIN_K, T_MAX_K, N_T)
    log10_ratio_values = np.linspace(LOG10_RATIO_MIN, LOG10_RATIO_MAX, N_RATIO)
    tt, yy_log10 = np.meshgrid(t_values, log10_ratio_values)
    return GridDefinition(
        t_values=t_values,
        log10_ratio_values=log10_ratio_values,
        tt=tt,
        yy_log10=yy_log10,
        log_ratio_natural=np.log(10.0) * yy_log10,
    )


def mu_h2o_grid(grid: GridDefinition) -> np.ndarray:
    return (
        CONSTANTS["H2O_E"]
        - grid.tt * CONSTANTS["H2O_S"]
        + CONSTANTS["kb"] * grid.tt * grid.log_ratio_natural
    )


def mu_h2_grid(grid: GridDefinition) -> np.ndarray:
    return CONSTANTS["H2_E"] - grid.tt * CONSTANTS["H2_S"]


def mu_ga_grid(grid: GridDefinition) -> np.ndarray:
    """Gallium reference from Ga2O3 + H2/H2O.

    mu_Ga = ( G(Ga2O3) - 3 G(H2O) + 3 G(H2) ) / 2
    """
    return (
        CONSTANTS["GA2O3_E"]
        - 3.0 * mu_h2o_grid(grid)
        + 3.0 * mu_h2_grid(grid)
    ) / 2.0


def cu_ga_reference_expressions() -> dict[str, sp.Basic]:
    """Return symbolic chemical-potential expressions for the Cu/Ga example."""
    t = sp.Symbol("T", real=True)
    kb = sp.Symbol("kb", positive=True)
    x = sp.Symbol("x", positive=True)
    mu_h2 = CONSTANTS["H2_E"] - t * CONSTANTS["H2_S"]
    mu_h2o = CONSTANTS["H2O_E"] - t * CONSTANTS["H2O_S"] + kb * t * sp.log(x)
    mu_ga = (CONSTANTS["GA2O3_E"] - 3.0 * mu_h2o + 3.0 * mu_h2) / 2.0
    return {"mu_Ga": sp.simplify(mu_ga)}


def corrected_surface_energy_grid(df: pd.DataFrame, grid: GridDefinition) -> np.ndarray:
    """Return corrected surface-free-energy grids using the generic phase engine."""
    required = {"form_G", "delta_Ga", "mu_Ga", "Area"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Dataset is missing required columns for surface correction: {missing}")

    corrected = build_corrected_phase_expressions(
        df,
        energy_column="form_G",
        normalized_energy_column=None,
        correction_map={"delta_Ga": ("mu_Ga", cu_ga_reference_expressions()["mu_Ga"])},
        output_column="phase_expression_per_area",
    )
    phase_field = build_phase_field_grid(
        corrected,
        expression_column="phase_expression_per_area",
        x_symbol="x",
        x_values=np.power(10.0, grid.log10_ratio_values),
        t_symbol="T",
        t_values=grid.t_values,
        variables={"kb": CONSTANTS["kb"]},
    )
    return np.transpose(phase_field.energy_grid, (0, 2, 1))


def prepare_facet_candidates(facet_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce repeated structure variants to the best candidate at each coverage."""
    reduced = (
        facet_df.sort_values(["Monolayer_alloy", "form_G_per_Area", "Name"], na_position="last")
        .groupby("Monolayer_alloy", dropna=False, as_index=False)
        .head(1)
        .copy()
    )
    return reduced.sort_values("Monolayer_alloy", na_position="first").reset_index(drop=True)


def prepare_bulk_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    bulk = frame.loc[frame["hkl"].isna()].copy()
    bulk["Ga_percent"] = pd.to_numeric(bulk["Ga_percent"], errors="coerce")
    bulk["formation_energy_per_atom"] = pd.to_numeric(bulk["formation_energy_per_atom"], errors="coerce")
    bulk = bulk.dropna(subset=["Ga_percent", "formation_energy_per_atom"]).copy()
    reduced = (
        bulk.sort_values(["Ga_percent", "formation_energy_per_atom", "Name"], na_position="last")
        .groupby("Ga_percent", dropna=False, as_index=False)
        .head(1)
        .copy()
    )
    return reduced.sort_values("Ga_percent").reset_index(drop=True)


def lower_convex_hull(points: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    hull: list[tuple[float, float, int]] = []
    for point in points:
        while len(hull) >= 2:
            x1, y1, _ = hull[-2]
            x2, y2, _ = hull[-1]
            x3, y3, _ = point
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def build_bulk_reference_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bulk = prepare_bulk_candidates(frame)
    points = [
        (
            float(row["Ga_percent"]),
            float(row["formation_energy_per_atom"]),
            int(i),
        )
        for i, row in bulk.iterrows()
    ]
    hull_points = lower_convex_hull(points)
    stable_indices = [index for _, _, index in hull_points]
    bulk["is_hull_stable"] = False
    bulk.loc[stable_indices, "is_hull_stable"] = True
    hull = bulk.loc[stable_indices].copy().sort_values("Ga_percent").reset_index(drop=True)
    return bulk, hull


def plot_bulk_reference_panel(ax: plt.Axes, bulk: pd.DataFrame, hull: pd.DataFrame) -> None:
    ax.scatter(
        bulk["Ga_percent"],
        bulk["formation_energy_per_atom"],
        s=34,
        color="#98a2b3",
        edgecolor="white",
        linewidth=0.6,
        label="Best candidate per composition",
        zorder=2,
    )
    ax.plot(
        hull["Ga_percent"],
        hull["formation_energy_per_atom"],
        color="#d62728",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Lower convex hull",
        zorder=3,
    )
    for _, row in hull.iterrows():
        label = f"{row['Ga_percent']:.1f}% Ga"
        ax.text(
            float(row["Ga_percent"]),
            float(row["formation_energy_per_atom"]) + 0.003,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111111",
        )
    ax.axhline(0.0, color="#d0d5dd", linewidth=1.0, zorder=1)
    ax.set_title("Bulk alloy reference hull")
    ax.set_xlabel("Ga fraction in bulk [%]")
    ax.set_ylabel("Formation energy [eV/atom]")
    ax.set_facecolor("#f7f7f7")
    ax.legend(loc="lower left", fontsize=8, frameon=True)


def summarize_stable_regions(
    facet_df: pd.DataFrame,
    stable_index: np.ndarray,
    stable_energy: np.ndarray,
    grid: GridDefinition,
    facet: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for phase_id in np.unique(stable_index):
        mask = stable_index == phase_id
        row = facet_df.iloc[int(phase_id)]
        short_label = (
            f"{float(row.get('Monolayer_alloy')):.1f}% ML"
            if pd.notna(row.get("Monolayer_alloy"))
            else str(row.get("Name", phase_id))
        )
        records.append(
            {
                "hkl": facet,
                "phase_id": int(phase_id),
                "Name": row.get("Name", f"phase {phase_id}"),
                "Formula": row.get("Formula", ""),
                "phase_label": f"{short_label} · {row.get('Name', phase_id)}",
                "short_label": short_label,
                "Ga": row.get("Ga", np.nan),
                "Cu": row.get("Cu", np.nan),
                "Monolayer_alloy": row.get("Monolayer_alloy", np.nan),
                "stable_grid_fraction": float(mask.mean()),
                "stable_percent": float(100.0 * mask.mean()),
                "T_min_stable_K": float(grid.tt[mask].min()),
                "T_max_stable_K": float(grid.tt[mask].max()),
                "log10_ratio_min_stable": float(grid.yy_log10[mask].min()),
                "log10_ratio_max_stable": float(grid.yy_log10[mask].max()),
                "min_G_per_Area_eV_A2": float(stable_energy[mask].min()),
            }
        )
    return pd.DataFrame(records).sort_values("stable_grid_fraction", ascending=False)


def plot_phase_map(
    facet_df: pd.DataFrame,
    summary: pd.DataFrame,
    stable_index: np.ndarray,
    grid: GridDefinition,
    facet: str,
) -> None:
    stable_phase_ids = np.unique(stable_index)
    remap = {old: new for new, old in enumerate(stable_phase_ids)}
    stable_compact = np.vectorize(remap.get)(stable_index)
    levels = np.arange(-0.5, len(stable_phase_ids) + 0.5, 1.0)
    xx = grid.tt
    yy = grid.yy_log10

    fig, ax = plt.subplots(figsize=(10.4, 6.2), constrained_layout=True)
    cmap = ListedColormap(QUALITATIVE_COLORS[: len(stable_phase_ids)])
    mesh = ax.pcolormesh(grid.tt, grid.yy_log10, stable_compact, cmap=cmap, shading="auto")
    ax.contour(xx, yy, stable_compact, levels=levels, colors="#ffffff", linewidths=0.55, alpha=0.9)

    ax.set_title(f"Cu/Ga surface phase diagram · hkl {facet}")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel(r"$\log_{10}(p_{H_2O}/p_{H_2})$")
    ax.set_facecolor("#f7f7f7")

    tick_labels = summary.set_index("phase_id").loc[stable_phase_ids, "short_label"].astype(str).tolist()
    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(len(stable_phase_ids)) + 0.5)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("Stable phase coverage")

    summary_by_id = summary.set_index("phase_id")
    for phase_id in stable_phase_ids:
        info = summary_by_id.loc[phase_id]
        if float(info["stable_percent"]) < MIN_REGION_LABEL_PERCENT:
            continue
        mask = stable_index == phase_id
        x_mid = float(np.median(xx[mask]))
        y_mid = float(np.median(yy[mask]))
        label = f"{info['short_label']}\n{info['stable_percent']:.1f}%"
        ax.text(
            x_mid,
            y_mid,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#111111",
            bbox={"facecolor": "white", "alpha": 0.68, "edgecolor": "none", "pad": 1.6},
        )

    fig.savefig(OUTPUT_DIR / f"cuga_surface_{facet}_phase_diagram_2d.png", dpi=220)
    plt.close(fig)


def plot_phase_multiplot(
    phase_maps: dict[str, tuple[pd.DataFrame, pd.DataFrame, np.ndarray]],
    grid: GridDefinition,
    bulk_candidates: pd.DataFrame,
    bulk_hull: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.8), constrained_layout=True)
    axes = axes.ravel()
    plot_bulk_reference_panel(axes[0], bulk_candidates, bulk_hull)

    for ax, facet in zip(axes[1: 1 + len(FACETS)], FACETS, strict=False):
        if facet not in phase_maps:
            ax.axis("off")
            continue

        facet_df, summary, stable_index = phase_maps[facet]
        stable_phase_ids = np.unique(stable_index)
        remap = {old: new for new, old in enumerate(stable_phase_ids)}
        stable_compact = np.vectorize(remap.get)(stable_index)
        cmap = ListedColormap(QUALITATIVE_COLORS[: len(stable_phase_ids)])

        mesh = ax.pcolormesh(grid.tt, grid.yy_log10, stable_compact, cmap=cmap, shading="auto")
        ax.contour(
            grid.tt,
            grid.yy_log10,
            stable_compact,
            levels=np.arange(-0.5, len(stable_phase_ids) + 0.5, 1.0),
            colors="#ffffff",
            linewidths=0.45,
            alpha=0.85,
        )
        ax.set_title(f"hkl {facet}")
        ax.set_xlabel("Temperature T [K]")
        ax.set_ylabel(r"$\log_{10}(p_{H_2O}/p_{H_2})$")
        ax.set_facecolor("#f7f7f7")

        top_rows = summary.head(4).copy()
        top_rows["annotation"] = top_rows["short_label"].astype(str)
        note = "\n".join(
            f"{row['annotation']}: {row['stable_percent']:.1f}%"
            for _, row in top_rows.iterrows()
        )
        ax.text(
            0.02,
            0.02,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#d0d5dd", "boxstyle": "round,pad=0.25"},
        )

        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_ticks(np.arange(len(stable_phase_ids)) + 0.5)
        cbar.set_ticklabels(
            summary.set_index("phase_id").loc[stable_phase_ids, "short_label"].astype(str).tolist()
        )
        cbar.ax.tick_params(labelsize=8)

    if len(axes) > len(FACETS) + 1:
        notes_ax = axes[-1]
        notes_ax.axis("off")
        notes = (
            "Bulk panel:\n"
            "- one best structure per Ga composition\n"
            "- red line marks lower convex hull\n\n"
            "Surface panels:\n"
            "- one best structure per Ga coverage\n"
            "- colors show stable coverage fields\n"
            "- white lines show phase boundaries"
        )
        notes_ax.text(
            0.0,
            0.95,
            notes,
            ha="left",
            va="top",
            fontsize=10,
            color="#344054",
        )

    fig.suptitle(
        "Cu/Ga bulk and surface reference overview\n"
        "Ga reference: (G(Ga2O3) - 3 G(H2O) + 3 G(H2)) / 2",
        fontsize=16,
    )
    fig.savefig(OUTPUT_DIR / "cuga_bulk_surface_phase_overview.png", dpi=220)
    fig.savefig(OUTPUT_DIR / "cuga_surface_phase_diagram_multiplot.png", dpi=220)
    plt.close(fig)


def build_surface_comparison_table(all_summaries: pd.DataFrame) -> pd.DataFrame:
    comparison = all_summaries.copy()
    comparison["panel"] = "Surface hkl " + comparison["hkl"].astype(str)
    comparison["surface_or_bulk"] = np.where(
        comparison["Name"].astype(str).str.contains("clean", case=False, na=False),
        "clean surface",
        "Ga-covered surface",
    )
    comparison["stable_percent"] = comparison["stable_grid_fraction"] * 100.0
    comparison["T_span_K"] = comparison["T_max_stable_K"] - comparison["T_min_stable_K"]
    comparison["log10_ratio_span"] = comparison["log10_ratio_max_stable"] - comparison["log10_ratio_min_stable"]
    comparison["energy_column"] = "G_per_Area_corrected"
    comparison["unit"] = "eV/Å²"
    comparison = comparison.rename(
        columns={
            "T_min_stable_K": "T_min_K",
            "T_max_stable_K": "T_max_K",
            "log10_ratio_min_stable": "log10_ratio_min",
            "log10_ratio_max_stable": "log10_ratio_max",
            "min_G_per_Area_eV_A2": "min_energy",
            "Ga": "Ga_atoms",
            "Cu": "Cu_atoms",
        }
    )
    comparison.insert(0, "rank_in_panel", comparison.groupby("panel").cumcount() + 1)
    ordered = [
        "rank_in_panel",
        "panel",
        "surface_or_bulk",
        "hkl",
        "phase_id",
        "Name",
        "Formula",
        "phase_label",
        "Monolayer_alloy",
        "Cu_atoms",
        "Ga_atoms",
        "stable_grid_fraction",
        "stable_percent",
        "T_min_K",
        "T_max_K",
        "T_span_K",
        "log10_ratio_min",
        "log10_ratio_max",
        "log10_ratio_span",
        "min_energy",
        "unit",
        "energy_column",
    ]
    return comparison[ordered]


def build_bulk_comparison_table(bulk: pd.DataFrame, hull: pd.DataFrame) -> pd.DataFrame:
    table = hull.copy()
    table["panel"] = "Bulk reference"
    table["surface_or_bulk"] = "bulk"
    table["hkl"] = ""
    table["phase_id"] = np.arange(len(table))
    table["phase_label"] = table["Ga_percent"].map(lambda value: f"{value:.1f}% Ga")
    table["Monolayer_alloy"] = np.nan
    table["stable_grid_fraction"] = np.nan
    table["stable_percent"] = np.nan
    table["T_min_K"] = np.nan
    table["T_max_K"] = np.nan
    table["T_span_K"] = np.nan
    table["log10_ratio_min"] = np.nan
    table["log10_ratio_max"] = np.nan
    table["log10_ratio_span"] = np.nan
    table["min_energy"] = table["formation_energy_per_atom"]
    table["unit"] = "eV/atom"
    table["energy_column"] = "formation_energy_per_atom"
    table["Cu_atoms"] = table.get("Cu", np.nan)
    table["Ga_atoms"] = table.get("Ga", np.nan)
    ordered = [
        "panel",
        "surface_or_bulk",
        "hkl",
        "phase_id",
        "Name",
        "Formula",
        "phase_label",
        "Ga_percent",
        "Monolayer_alloy",
        "Cu_atoms",
        "Ga_atoms",
        "stable_grid_fraction",
        "stable_percent",
        "T_min_K",
        "T_max_K",
        "T_span_K",
        "log10_ratio_min",
        "log10_ratio_max",
        "log10_ratio_span",
        "min_energy",
        "unit",
        "energy_column",
    ]
    return table[ordered].reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_pickle(DATASET)
    grid = build_grid()
    bulk_candidates, bulk_hull = build_bulk_reference_summary(frame)
    bulk_candidates.to_csv(OUTPUT_DIR / "cuga_bulk_reference_candidates.csv", index=False)
    bulk_hull.to_csv(OUTPUT_DIR / "cuga_bulk_reference_hull.csv", index=False)
    combined_summaries: list[pd.DataFrame] = []
    phase_maps: dict[str, tuple[pd.DataFrame, pd.DataFrame, np.ndarray]] = {}

    for facet in FACETS:
        facet_df = frame.loc[frame["hkl"].astype(str).eq(facet)].copy()
        if facet_df.empty:
            continue
        facet_df = prepare_facet_candidates(facet_df)
        facet_df.to_csv(OUTPUT_DIR / f"cuga_surface_{facet}_plot_candidates.csv", index=False)

        energy_surfaces = corrected_surface_energy_grid(facet_df, grid)
        stable_index = np.nanargmin(energy_surfaces, axis=0)
        stable_energy = np.nanmin(energy_surfaces, axis=0)

        summary = summarize_stable_regions(
            facet_df=facet_df,
            stable_index=stable_index,
            stable_energy=stable_energy,
            grid=grid,
            facet=facet,
        )
        summary.to_csv(OUTPUT_DIR / f"cuga_surface_{facet}_stable_phases.csv", index=False)
        plot_phase_map(
            facet_df=facet_df,
            summary=summary,
            stable_index=stable_index,
            grid=grid,
            facet=facet,
        )
        phase_maps[facet] = (facet_df, summary, stable_index)
        combined_summaries.append(summary)

    if combined_summaries:
        all_summaries = pd.concat(combined_summaries, ignore_index=True)
        all_summaries.to_csv(OUTPUT_DIR / "cuga_surface_all_stable_phases.csv", index=False)
        surface_comparison = build_surface_comparison_table(all_summaries)
        surface_comparison.to_csv(
            OUTPUT_DIR / "cuga_surface_phase_comparison_summary.csv",
            index=False,
        )
        bulk_comparison = build_bulk_comparison_table(bulk_candidates, bulk_hull)
        combined = pd.concat([bulk_comparison, surface_comparison], ignore_index=True)
        if "rank_in_panel" in combined.columns:
            combined = combined.drop(columns=["rank_in_panel"])
        combined.insert(0, "rank_in_panel", combined.groupby("panel").cumcount() + 1)
        combined.to_csv(
            OUTPUT_DIR / "cuga_bulk_surface_phase_comparison_summary.csv",
            index=False,
        )
        plot_phase_multiplot(phase_maps, grid, bulk_candidates, bulk_hull)


if __name__ == "__main__":
    main()
