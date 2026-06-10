from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from onepiece_studio.adapters import DatabaseSource, HDFSource
from onepiece_studio.config import OnePieceStudioConfig
from onepiece_studio.demo import demo_source, local_default_source
from onepiece_studio.images import resolve_image
from onepiece_studio.materials_columns import columns_for_profile, profile_names
from onepiece_studio.schema import ColumnKind, infer_schema, safe_unique_count
from onepiece_studio.state import (
    PAGE_SIZE,
)
from onepiece_studio.ui.adsorption import render_adsorption_workbench
from onepiece_studio.ui.controlroom import render_controlroom
from onepiece_studio.ui.data_management import render_data_management
from onepiece_studio.ui.data_sources import render_data_source_manager
from onepiece_studio.ui.row_actions import (
    is_atoms,
    open_atoms_in_ase,
    render_action_grid,
    selected_row_summary,
)
from onepiece_studio.ui.workflow_builder import apply_workflow_operations, render_workflow_builder


def run_app(source: DatabaseSource, config: OnePieceStudioConfig) -> None:
    import streamlit as st

    st.set_page_config(page_title=config.title, page_icon="PF", layout="wide")
    _inject_styles(st)

    base_dataframe = source.load()

    source_name = getattr(source, "display_name", source.name)
    source_path = str(getattr(source, "path", source_name))
    st.title(config.title)
    st.caption(f"{len(base_dataframe):,} base records from {source_name}")

    dataframe = render_data_source_manager(
        st,
        base_dataframe,
        str(source_name),
        source_path=source_path,
    )
    st.caption(f"{len(dataframe):,} active source records before Workflow and Controlroom")
    _render_session_onboarding(st, dataframe)

    workflow = apply_workflow_operations(st, dataframe)
    workflow_dataframe = workflow.dataframe
    schema = infer_schema(
        workflow_dataframe,
        image_columns=config.image_columns,
        structure_columns=config.structure_columns,
    )

    workflow_tab, controlroom_tab, data_tab, adsorption_tab, records_tab, charts_tab, schema_tab = st.tabs(
        [
            "Workflow",
            "Controlroom",
            "Data Management",
            "Adsorption & Barriers",
            "Records",
            "Visualize",
            "Schema",
        ]
    )

    with workflow_tab:
        render_workflow_builder(st, dataframe, workflow_dataframe, workflow.messages)

    with controlroom_tab:
        controlroom = render_controlroom(st, workflow_dataframe)

    filtered = controlroom.dataframe

    with data_tab:
        render_data_management(st, workflow_dataframe, filtered)

    with adsorption_tab:
        render_adsorption_workbench(st, filtered, filtered, reference_source=workflow_dataframe)

    with records_tab:
        _render_metrics(st, filtered, config)
        table_column, detail_column = st.columns([0.68, 0.32], gap="large")
        with table_column:
            st.subheader("Filtered Records")
            selected_index = _render_table(st, filtered, config, total_rows=len(workflow_dataframe))

        with detail_column:
            st.subheader("Record detail")
            if selected_index is None and not filtered.empty:
                selected_index = filtered.index[0]
            if selected_index is not None:
                selected_row = workflow_dataframe.loc[selected_index]
                if isinstance(selected_row, pd.DataFrame):
                    # Duplicate index labels return a frame; show the first match.
                    selected_row = selected_row.iloc[0]
                _render_detail(st, selected_row, schema, config)
            else:
                st.info("No record selected.")

    with charts_tab:
        _render_visualizations(st, filtered)

    with schema_tab:
        _render_schema(st, schema)


def _render_filters(st: Any, dataframe: pd.DataFrame, config: OnePieceStudioConfig) -> pd.DataFrame:
    with st.container(border=True):
        search, page_size = st.columns([0.75, 0.25])
        query = search.text_input("Search", placeholder="Formula, id, status...")
        page_size.number_input(
            "Page size",
            min_value=10,
            max_value=500,
            value=config.default_page_size,
            step=10,
            key=PAGE_SIZE,
        )

        categorical_columns = _categorical_columns(
            dataframe,
            exclude=[*config.image_columns, *config.structure_columns],
            max_unique=20,
        )[:6]

        active = dataframe.copy()
        if query:
            requested_search_columns = config.searchable_columns or [
                column for column in dataframe.columns if dataframe[column].dtype == "object"
            ]
            search_columns = [column for column in requested_search_columns if column in dataframe.columns]
            mask = pd.Series(False, index=dataframe.index)
            for column in search_columns:
                mask |= dataframe[column].astype(str).str.contains(query, case=False, na=False)
            active = active[mask]

        if categorical_columns:
            columns = st.columns(len(categorical_columns))
            for widget_column, column in zip(columns, categorical_columns, strict=False):
                options = sorted(dataframe[column].dropna().astype(str).unique())
                selected = widget_column.multiselect(column, options)
                if selected:
                    active = active[active[column].astype(str).isin(selected)]

        numeric_filters = _preferred_numeric_filters(active)
        if numeric_filters:
            with st.expander("Numeric filters", expanded=False):
                numeric_columns = st.columns(min(3, len(numeric_filters)))
                for index, column in enumerate(numeric_filters):
                    finite = active[column].replace([np.inf, -np.inf], np.nan).dropna()
                    if finite.empty:
                        continue
                    minimum = float(finite.min())
                    maximum = float(finite.max())
                    if minimum == maximum:
                        continue
                    selected_min, selected_max = numeric_columns[index % len(numeric_columns)].slider(
                        column,
                        min_value=minimum,
                        max_value=maximum,
                        value=(minimum, maximum),
                    )
                    active = active[
                        active[column].replace([np.inf, -np.inf], np.nan).between(
                            selected_min,
                            selected_max,
                            inclusive="both",
                        )
                    ]

    return active


def _render_session_onboarding(st: Any, dataframe: pd.DataFrame) -> None:
    if not dataframe.empty:
        return
    with st.container(border=True):
        st.markdown("**Welcome to OnePiece Studio**")
        st.caption(
            "This session is empty on purpose. Load the bundled tutorial dataset or your own HDF file from "
            "`Data Sources`, then use `Workflow` to derive adsorption, Gibbs, charge, or geometry columns."
        )
        st.caption(
            "Good first path for new catalysis users: 1) load the bundled tutorial dataset, "
            "2) add `Adsorption + Gibbs analysis starter`, 3) inspect `Records`, 4) compare candidates in `Visualize`."
        )


def _render_metrics(st: Any, dataframe: pd.DataFrame, config: OnePieceStudioConfig) -> None:
    metric_columns = [
        column
        for column in config.metric_columns
        if column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column])
    ][:4]
    if not metric_columns:
        return

    rows = [st.columns(2, gap="small"), st.columns(2, gap="small")]
    slots = [column for row in rows for column in row]
    for widget_column, column in zip(slots, metric_columns, strict=False):
        finite = dataframe[column].replace([np.inf, -np.inf], np.nan).dropna()
        value = f"{finite.mean():.3g}" if not finite.empty else "n/a"
        widget_column.metric(column, value, help="Mean of filtered rows")


def _render_visualizations(st: Any, dataframe: pd.DataFrame) -> None:
    numeric_columns = list(dataframe.select_dtypes(include="number").columns)
    categorical_columns = _categorical_columns(dataframe, exclude=numeric_columns, max_unique=30)
    for preferred in ["surface_ref_name", "surface_ref_status", "adsorbate", "dataset_label"]:
        if preferred in dataframe.columns and preferred not in categorical_columns:
            categorical_columns.append(preferred)
    if not numeric_columns:
        st.info("No numeric columns available for plotting.")
        return

    presets = _chart_presets(dataframe, numeric_columns, categorical_columns)
    preset_labels = list(presets)
    st.write("Analysis preset")
    selected_preset = st.segmented_control(
        "Analysis preset",
        preset_labels,
        default=preset_labels[0],
        label_visibility="collapsed",
    )
    preset = presets[selected_preset]
    st.caption(preset["takeaway"])

    mode = st.segmented_control("Chart type", ["Scatter", "Histogram"], default="Scatter")
    if mode == "Scatter" and len(numeric_columns) >= 2:
        x_column, y_column, color_column = st.columns(3)
        x = x_column.selectbox("x", numeric_columns, index=_column_index(numeric_columns, preset["x"]))
        y = y_column.selectbox("y", numeric_columns, index=_column_index(numeric_columns, preset["y"], fallback=min(1, len(numeric_columns) - 1)))
        color_options = ["None", *categorical_columns]
        color = color_column.selectbox(
            "color",
            color_options,
            index=_column_index(color_options, preset.get("color", "None")),
        )
        kwargs = {
            "x": x,
            "y": y,
            "hover_data": [
                c for c in ["dataset", "source_hdf", "Name", "Formula", "legend"] if c in dataframe
            ],
        }
        if color != "None":
            kwargs["color"] = color
        st.caption(_chart_interpretation(preset["title"], x, y, color))
        try:
            import plotly.express as px

            chart_data = dataframe.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y]).copy()
            chart_data["__onepiece_studio_index"] = chart_data.index.astype(str)
            fig = px.scatter(
                chart_data,
                **kwargs,
                custom_data=["__onepiece_studio_index"],
                title=preset["title"],
                color_discrete_sequence=["#d33f49", "#2f4858", "#33658a", "#8a6f3d", "#3f7d20"],
            )
            fig.update_traces(
                hovertemplate=(
                    f"{x}: %{{x}}<br>"
                    f"{y}: %{{y}}<br>"
                    "row index: %{customdata[0]}<extra></extra>"
                )
            )
            fig.update_layout(
                font_family="Lucifer",
                title_font_family="Lucifer",
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=8, r=8, t=56, b=8),
                xaxis_title=_column_plot_label(x),
                yaxis_title=_column_plot_label(y),
                legend_title_text=_column_plot_label(color) if color != "None" else None,
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#2f343d")))
            event = st.plotly_chart(
                fig,
                width="stretch",
                key="onepiece_studio_visualize_scatter",
                on_select="rerun",
                selection_mode="points",
            )
            _render_plot_selection_actions(st, event, chart_data, dataframe)
        except ImportError:
            st.scatter_chart(dataframe, x=x, y=y, width="stretch")
    else:
        column = st.selectbox("value", numeric_columns, index=_column_index(numeric_columns, preset["y"]))
        st.caption(f"Distribution view for {_column_plot_label(column)}.")
        try:
            import plotly.express as px

            chart_data = dataframe.replace([np.inf, -np.inf], np.nan).dropna(subset=[column])
            fig = px.histogram(
                chart_data,
                x=column,
                nbins=30,
                title=f"Distribution of {_column_plot_label(column)}",
                color_discrete_sequence=["#d33f49"],
            )
            fig.update_layout(
                font_family="Lucifer",
                title_font_family="Lucifer",
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=8, r=8, t=56, b=8),
                xaxis_title=_column_plot_label(column),
                yaxis_title="Count",
            )
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            st.bar_chart(dataframe[column].value_counts().sort_index(), width="stretch")


def _render_plot_selection_actions(
    st: Any,
    event: Any,
    chart_data: pd.DataFrame,
    dataframe: pd.DataFrame,
) -> None:
    selected_index = _selected_plot_index(event)
    if selected_index is None:
        st.caption("Click a scatter point to inspect, exclude, mark review, or open it in ASE.")
        return

    if selected_index not in dataframe.index:
        st.warning(f"Selected row {selected_index} is no longer in the active filtered dataset.")
        return

    row = dataframe.loc[selected_index]
    st.markdown("**Selected calculation**")
    st.dataframe(selected_row_summary(row), hide_index=True, width="stretch")

    render_action_grid(
        st,
        row,
        selected_index,
        key_prefix="plot",
        namespace="onepiece_studio_plot",
        exclude_label="Exclude calculation",
        review_label="Mark review",
        reference_label="Mark reference",
        open_label="Open ASE viewer",
    )


def _selected_plot_index(event: Any) -> int | str | None:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None
    point = points[0]
    customdata = getattr(point, "customdata", None)
    if customdata is None and isinstance(point, dict):
        customdata = point.get("customdata")
    if not customdata:
        return None
    raw = customdata[0]
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw


def _render_schema(st: Any, schema: list[Any]) -> None:
    rows = [
        {
            "column": column.name,
            "kind": column.kind.value,
            "nullable": column.nullable,
            "unique_count": column.unique_count,
            "sample": str(column.sample)[:160],
        }
        for column in schema
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_table(
    st: Any,
    dataframe: pd.DataFrame,
    config: OnePieceStudioConfig,
    *,
    total_rows: int,
) -> int | None:
    page_size = int(st.session_state.get(PAGE_SIZE, config.default_page_size))
    shown_rows = min(len(dataframe), page_size)
    st.caption(
        f"Showing {shown_rows:,} of {len(dataframe):,} filtered rows "
        f"from {total_rows:,} total records."
    )
    if dataframe.empty:
        st.info("No records match the current filters.")
        return None

    visible = dataframe.head(page_size).drop(columns=config.structure_columns, errors="ignore")
    focus = st.selectbox("Column focus", profile_names(), key="onepiece_studio_column_focus")
    visible = visible[columns_for_profile(focus, visible)]
    display = _display_dataframe(visible, image_columns=config.image_columns)
    event = st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        column_config={
            column: st.column_config.ImageColumn(column)
            for column in config.image_columns
            if column in visible.columns
        },
    )
    selected_rows = event.selection.rows
    if not selected_rows:
        return None
    return int(visible.index[selected_rows[0]])


def _categorical_columns(
    dataframe: pd.DataFrame,
    *,
    exclude: list[str],
    max_unique: int,
) -> list[str]:
    columns: list[str] = []
    for column in dataframe.columns:
        if column in exclude or not _is_simple_categorical(dataframe[column]):
            continue
        if safe_unique_count(dataframe[column]) <= max_unique:
            columns.append(column)
    return columns


def _is_simple_categorical(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if series.dtype != "object":
        return False
    sample = _first_non_null(series)
    return isinstance(sample, str | bool)


def _first_non_null(series: pd.Series) -> Any | None:
    values = series.dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _preferred_numeric_filters(dataframe: pd.DataFrame) -> list[str]:
    preferred = [
        "form_G_per_Area",
        "form_G_per_alloy",
        "formation_energy_per_atom",
        "E",
        "fmax",
        "Ga_percent",
        "Zn_percent",
        "Area",
    ]
    return [
        column
        for column in preferred
        if column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column])
    ][:6]


def _chart_presets(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, dict[str, str]]:
    presets = {
        "Surface stability": {
            "x": _first_available(numeric_columns, ["Area", "Ga", "E"]),
            "y": _first_available(numeric_columns, ["form_G_per_Area", "form_G_per_alloy", "E"]),
            "color": _first_available(categorical_columns, ["slabsize", "hkl", "alloy"], default="None"),
            "title": "Surface stability across filtered candidates",
            "takeaway": "Use this view to compare formation energy at the same filtered observation grain.",
        },
        "Coordination vs stability": {
            "x": _first_available(numeric_columns, ["average_Ga_GCN", "average_Cu_GCN", "Ga"]),
            "y": _first_available(numeric_columns, ["form_G_per_alloy", "form_G_per_Area", "E"]),
            "color": _first_available(categorical_columns, ["alloy", "slabsize", "hkl"], default="None"),
            "title": "Coordination descriptors versus stability",
            "takeaway": "Scatter is appropriate here because each point is one structure candidate.",
        },
        "Charge descriptors": {
            "x": _first_available(numeric_columns, ["min_Ga_charge", "average_Ga_charge", "average_Cu_charge"]),
            "y": _first_available(numeric_columns, ["form_G_per_Area", "form_G_per_alloy", "E"]),
            "color": _first_available(categorical_columns, ["slabsize", "alloy", "hkl"], default="None"),
            "title": "Charge descriptor relationship",
            "takeaway": "Use charge descriptors to spot unusual candidates, then inspect exact rows in the table.",
        },
        "Relaxation quality": {
            "x": _first_available(numeric_columns, ["E", "Area", "Ga"]),
            "y": _first_available(numeric_columns, ["fmax", "form_G_per_Area", "E"]),
            "color": _first_available(categorical_columns, ["slabsize", "hkl", "alloy"], default="None"),
            "title": "Relaxation quality check",
            "takeaway": "High force values can flag records that should be reviewed before comparing energies.",
        },
    }
    if "adsorbate_charge_delta_vs_ref_e" in numeric_columns and "E_ads_CO_eV" in numeric_columns:
        presets = {
            "Charge transfer vs adsorption energy": {
                "x": _first_available(
                    numeric_columns,
                    ["adsorbate_charge_delta_vs_ref_e", "adsorbate_net_charge_e", "surface_net_charge_delta_vs_ref_e"],
                ),
                "y": _first_available(
                    numeric_columns,
                    ["E_ads_CO_eV", "E_ads_CH3OH_to_CH3O_eV", "adsorption_energy"],
                ),
                "color": _first_available(
                    categorical_columns,
                    ["adsorption_site", "surface_ref_name", "adsorbate", "dataset_label"],
                    default="None",
                ),
                "title": "Charge transfer versus adsorption energy",
                "takeaway": "Use this to see whether stronger binding correlates with more electron transfer into or out of the adsorbate.",
            },
            **presets,
        }
    if "surface_net_charge_delta_vs_ref_e" in numeric_columns and "adsorbate_height_above_surface" in numeric_columns:
        presets = {
            "Surface polarization vs adsorbate height": {
                "x": _first_available(
                    numeric_columns,
                    ["adsorbate_height_above_surface", "min_adsorbate_surface_distance", "adsorbate_tilt_deg"],
                ),
                "y": _first_available(
                    numeric_columns,
                    ["surface_net_charge_delta_vs_ref_e", "adsorbate_charge_delta_vs_ref_e", "surface_reconstruction_rmsd"],
                ),
                "color": _first_available(
                    categorical_columns,
                    ["adsorption_site", "adsorbate", "surface_ref_name", "dataset_label"],
                    default="None",
                ),
                "title": "Surface polarization versus adsorption geometry",
                "takeaway": "This view is useful when you want to connect geometric lift-off or tilt with charge rearrangement in the slab.",
            },
            **presets,
        }
    if "metal_d_band_center_eV" in numeric_columns:
        presets = {
            "d-band center vs adsorption energy": {
                "x": _first_available(
                    numeric_columns,
                    ["metal_d_band_center_eV", "metal_d_band_filling"],
                ),
                "y": _first_available(
                    numeric_columns,
                    ["E_ads_CO_eV", "adsorption_energy", "E"],
                ),
                "color": _first_available(
                    categorical_columns,
                    ["adsorption_site", "surface_ref_name", "adsorbate", "dataset_label"],
                    default="None",
                ),
                "title": "d-band descriptor versus adsorption energy",
                "takeaway": "Use this as the first electronic-structure screening plot when DOSCAR-based d-band descriptors are available.",
            },
            **presets,
        }
    if "adsorption_site" in categorical_columns and "E_ads_CO_eV" in numeric_columns:
        presets = {
            "Adsorption site families": {
                "x": _first_available(
                    numeric_columns,
                    ["adsorbate_tilt_deg", "min_adsorbate_surface_distance", "adsorbate_height_above_surface"],
                ),
                "y": _first_available(
                    numeric_columns,
                    ["E_ads_CO_eV", "adsorbate_charge_delta_vs_ref_e", "surface_reconstruction_rmsd"],
                ),
                "color": _first_available(
                    categorical_columns,
                    ["adsorption_site", "adsorbate", "surface_ref_name", "dataset_label"],
                    default="None",
                ),
                "title": "Adsorption-site families across filtered candidates",
                "takeaway": "Coloring by site type helps separate top, bridge, hollow, and defect-like binding motifs before you inspect exact structures.",
            },
            **presets,
        }
    if "E_ads_CO_eV" in numeric_columns:
        presets = {
            "Adsorption analysis": {
                "x": _first_available(numeric_columns, ["E_ads_CO_total_eV", "delta_E_to_surface_eV", "E"]),
                "y": _first_available(numeric_columns, ["E_ads_CO_eV", "E_ads_CH3OH_to_CH3O_eV", "E"]),
                "color": _first_available(categorical_columns, ["surface_ref_name", "adsorbate", "dataset_label"], default="None"),
                "title": "Adsorption energy per CO across filtered candidates",
                "takeaway": "Start here after HDF import with adsorption preparation: compare total adsorption energy and per-CO energy, colored by the assigned surface reference.",
            },
            **presets,
        }
    return presets


def _column_plot_label(column: str | None) -> str:
    labels = {
        "E": "Total energy / eV",
        "fmax": "Maximum force / eV/A",
        "Area": "Surface area / A^2",
        "form_G_per_Area": "Formation free energy / eV A^-2",
        "form_G_per_alloy": "Formation free energy per alloy atom / eV",
        "formation_energy_per_atom": "Formation energy per atom / eV",
        "E_ads_CO_total_eV": "CO adsorption energy (total) / eV",
        "E_ads_CO_eV": "CO adsorption energy per CO / eV",
        "E_ads_CH3OH_to_CH3O_eV": "CH3O adsorption energy / eV",
        "adsorption_energy": "Adsorption energy / eV",
        "adsorbate_charge_delta_vs_ref_e": "Adsorbate charge change vs ref / e",
        "adsorbate_net_charge_e": "Adsorbate net charge / e",
        "surface_net_charge_delta_vs_ref_e": "Surface charge change vs clean ref / e",
        "surface_net_charge_e": "Surface net charge / e",
        "adsorbate_height_above_surface": "Adsorbate height above surface / A",
        "min_adsorbate_surface_distance": "Minimum adsorbate-surface distance / A",
        "mean_adsorbate_surface_distance": "Mean adsorbate-surface distance / A",
        "adsorbate_tilt_deg": "Adsorbate tilt / deg",
        "surface_reconstruction_rmsd": "Surface reconstruction RMSD / A",
        "surface_reconstruction_max_displacement": "Maximum surface displacement / A",
        "mean_coordination": "Mean coordination number",
        "mean_generalized_coordination": "Mean generalized coordination number",
        "min_interatomic_distance": "Minimum interatomic distance / A",
        "min_bond_ratio": "Minimum bond-length ratio",
        "metal_d_band_center_eV": "Metal d-band center / eV",
        "metal_d_band_filling": "Metal d-band filling",
        "adsorption_site": "Adsorption site",
        "surface_ref_name": "Surface reference",
        "adsorbate": "Adsorbate",
        "dataset_label": "Dataset",
    }
    if not column or column == "None":
        return "None"
    return labels.get(column, column.replace("_", " "))


def _chart_interpretation(title: str, x: str, y: str, color: str) -> str:
    base = f"{_column_plot_label(x)} on x and {_column_plot_label(y)} on y."
    if color != "None":
        base += f" Points are colored by {_column_plot_label(color)}."
    if "d-band" in title.lower():
        return base + " This is the classic first-pass electronic-structure view for catalyst screening."
    if "charge" in title.lower():
        return base + " Use it to see whether charge transfer tracks binding strength or structural distortion."
    if "site" in title.lower():
        return base + " This helps separate geometric binding motifs before inspecting individual structures."
    if "polarization" in title.lower():
        return base + " This is useful for linking adsorbate geometry to slab charge response."
    return base


def _first_available(columns: list[str], preferred: list[str], *, default: str | None = None) -> str:
    for column in preferred:
        if column in columns:
            return column
    if columns:
        return columns[0]
    return default or "None"


def _column_index(columns: list[str], column: str | None, *, fallback: int = 0) -> int:
    if column in columns:
        return columns.index(column)
    return min(fallback, len(columns) - 1)


def _prioritized_columns(dataframe: pd.DataFrame) -> list[str]:
    priority = [
        "Name",
        "Formula",
        "E",
        "fmax",
        "form_G_per_Area",
        "form_G_per_alloy",
        "formation_energy_per_atom",
        "hkl",
        "slabsize",
        "layers",
        "Area",
        "Ga",
        "Cu",
        "Zn",
        "average_Ga_GCN",
        "average_Ga_charge",
        "Path",
    ]
    ordered = [column for column in priority if column in dataframe.columns]
    ordered.extend([column for column in dataframe.columns if column not in ordered])
    return ordered


def _display_dataframe(dataframe: pd.DataFrame, *, image_columns: list[str]) -> pd.DataFrame:
    display = dataframe.copy()
    for column in display.columns:
        if column in image_columns:
            continue
        if display[column].dtype == "object":
            display[column] = display[column].map(_display_value)
    return display


def _display_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if len(text) <= 240 else f"{text[:237]}..."


def _render_detail(
    st: Any,
    row: pd.Series,
    schema: list[Any],
    config: OnePieceStudioConfig,
) -> None:
    render_action_grid(st, row, row.name, key_prefix="detail", namespace="onepiece_studio_detail")

    for column in config.image_columns:
        if column in row:
            image = resolve_image(row[column], asset_root=config.normalized_asset_root())
            if image.display_uri:
                st.image(image.display_uri, caption=column, width="stretch")
            elif image.raw:
                st.warning(f"Image not found: {image.raw}")

    for column_schema in schema:
        column = column_schema.name
        if column in config.image_columns:
            continue
        value = row[column]
        if column_schema.kind == ColumnKind.STRUCTURE:
            st.code(_format_atoms(value), language="text")
            if is_atoms(value):
                button_key = f"onepiece_studio_ase_view_{column}_{row.name}"
                if st.button("Open in ASE viewer", key=button_key, width="stretch"):
                    try:
                        path = open_atoms_in_ase(value, label=str(row.get("Name", row.name)))
                    except Exception as exc:
                        st.error(f"Could not open ASE viewer: {exc}")
                    else:
                        st.success(f"Opened ASE viewer for {column}. Temporary file: {path}")
        else:
            st.write(f"**{column}**")
            st.write(value)

def _format_atoms(value: Any) -> str:
    if not is_atoms(value):
        return str(value)
    formula = value.get_chemical_formula()
    cell = value.cell.cellpar()
    return (
        f"ASE Atoms: {formula}\n"
        f"Atoms: {len(value)}\n"
        f"Cell: {', '.join(f'{number:.3f}' for number in cell)}"
    )


def _inject_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"], button, input, textarea, select,
        .stMarkdown, .stDataFrame, .stDataFrame *, h1, h2, h3, h4, h5, h6,
        [data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stMetric"],
        [data-testid="stTabs"], [data-testid="stCaptionContainer"] {
            font-family: "Lucifer", system-ui, sans-serif !important;
        }
        .material-icons, .material-symbols-rounded, .material-symbols-outlined,
        [data-testid="stExpanderToggleIcon"], [data-testid="stBaseButton-header"] span {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
        }
        .math, .math *, .MathJax, .MathJax *, .mjx-container, .mjx-container * {
            font-family: initial !important;
        }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        details[data-testid="stExpander"] summary {
            min-height: 2.75rem;
            align-items: center;
        }
        details[data-testid="stExpander"] summary p {
            line-height: 1.2;
            margin: 0;
        }
        div[data-testid="stMetricLabel"] p {
            white-space: normal;
            line-height: 1.15;
        }
        div[data-testid="stMetricValue"] {
            line-height: 1.0;
            font-size: clamp(1.8rem, 2vw, 2.7rem);
        }
        div[data-testid="stMetric"] {
            background: #f7f8fa;
            border: 1px solid #e4e7ec;
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 112px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e4e7ec;
            border-radius: 8px;
            overflow: hidden;
        }
        button[kind], div[data-testid="stButton"] > button {
            min-height: 2.6rem;
            white-space: normal;
        }
        [data-testid="stTabs"] button {
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--hdf")
    parser.add_argument("--key", default="df")
    parser.add_argument("--title")
    args = parser.parse_args()
    if args.demo:
        source, config = demo_source()
        run_app(source, config)
    elif args.hdf:
        from onepiece_studio.ui.welcome import remember_recent_file

        path = args.hdf
        remember_recent_file(path, args.key)
        source = HDFSource(path, key=args.key)
        config = OnePieceStudioConfig(
            title=args.title or f"OnePiece Studio: {source.display_name}",
            primary_key="Name",
            structure_columns=["struc"],
            metric_columns=[
                "E",
                "formation_energy_per_atom",
                "form_G_per_Area",
                "form_G_per_alloy",
            ],
        )
        run_app(source, config)
    else:
        source, config = local_default_source()
        if getattr(source, "name", "") == "empty-session":
            from onepiece_studio.ui.welcome import run_welcome

            run_welcome()
        else:
            run_app(source, config)


if __name__ == "__main__":
    _main()
