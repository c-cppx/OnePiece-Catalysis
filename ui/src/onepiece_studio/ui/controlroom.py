from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from onepiece.adsorption import get_all_elements, row_element_count_map, structure_columns_in_frame
from onepiece.services import DatasetQuery, apply_dataset_query
from onepiece.services import apply_materials_search as backend_apply_materials_search
from onepiece.services import filter_any_token as backend_filter_any_token
from onepiece.services import filter_text as backend_filter_text
from onepiece.services import query_description as backend_query_description
from onepiece_studio.materials_columns import column_context, profile_review
from onepiece_studio.state import (
    CONTROL_DROP_CONVERGENCE,
    CONTROL_DROP_TEST,
    CONTROL_FMAX_MAX,
    CONTROL_MATERIAL_QUERY,
    CONTROL_NUMERIC,
    CONTROL_ROW_KEY,
    CONTROL_SELECTED_FACETS,
    CONTROL_STATUS,
    CONTROL_TEXT_EXCLUDE,
    CONTROL_TEXT_INCLUDE,
    CONTROL_USE_STATUS,
    CONTROL_VISIBLE_STATES,
    CONTROLROOM_ACTIVE_DATAFRAME,
)
from onepiece_studio.ui.row_actions import (
    render_action_grid,
    row_keys,
    selected_row_summary,
)


@dataclass(frozen=True, slots=True)
class ControlroomResult:
    dataframe: pd.DataFrame
    summary: dict[str, Any]


def render_controlroom(st: Any, dataframe: pd.DataFrame) -> ControlroomResult:
    _init_state(st, dataframe)

    st.subheader("Controlroom")
    st.caption(
        "Build the active database view from local OnePiece/pandas rows. "
        "The resulting filter is used by Records and Visualize."
    )

    left, right = st.columns([0.34, 0.66], gap="large")

    with left:
        _render_filter_presets(st, dataframe)
        _render_inclusion_editor(st, dataframe)
        _render_name_filters(st)
        _render_materials_search(st, dataframe)
        _render_domain_filters(st, dataframe)
        _render_numeric_controls(st, dataframe)

    active = _apply_controlroom_filters(st, dataframe)
    with right:
        _render_control_summary(st, dataframe, active)
        _render_commands(st, active)
        _render_preview(st, active)

    return ControlroomResult(
        dataframe=active,
        summary={"rows": len(active), "total_rows": len(dataframe)},
    )


def _init_state(st: Any, dataframe: pd.DataFrame) -> None:
    st.session_state.setdefault(CONTROL_TEXT_INCLUDE, "")
    st.session_state.setdefault(CONTROL_TEXT_EXCLUDE, "")
    st.session_state.setdefault(CONTROL_USE_STATUS, True)
    st.session_state.setdefault(CONTROL_STATUS, {})
    st.session_state.setdefault(CONTROL_SELECTED_FACETS, {})
    st.session_state.setdefault(CONTROL_NUMERIC, {})
    st.session_state.setdefault(CONTROL_MATERIAL_QUERY, {})
    st.session_state.setdefault(CONTROL_FMAX_MAX, None)
    st.session_state.setdefault(CONTROL_DROP_CONVERGENCE, False)
    st.session_state.setdefault(CONTROL_DROP_TEST, False)
    if CONTROL_ROW_KEY not in st.session_state:
        st.session_state[CONTROL_ROW_KEY] = _row_keys(dataframe)


def _render_filter_presets(st: Any, dataframe: pd.DataFrame) -> None:
    st.markdown("**Filter presets**")
    preset_columns = st.columns(2)
    if preset_columns[0].button("All rows", width="stretch"):
        _reset_filters(st)
    if preset_columns[1].button("Surface rows", width="stretch"):
        _set_text_filter(st, include="surf surface slab hkl 100 110 111 211")
    if preset_columns[0].button("Clean refs", width="stretch"):
        _set_text_filter(st, include="clean", exclude="adsorbate convergence test")
    if preset_columns[1].button("Ga rows", width="stretch"):
        _set_text_filter(st, include="Ga")
    if "fmax" in dataframe.columns and preset_columns[0].button("fmax ok", width="stretch"):
        st.session_state[CONTROL_FMAX_MAX] = 0.05
    if preset_columns[1].button("Needs review", width="stretch"):
        _set_status_filter(st, ["review"])


def _render_inclusion_editor(st: Any, dataframe: pd.DataFrame) -> None:
    with st.expander("Inclusion states", expanded=True):
        st.checkbox(
            "Use inclusion state",
            key=CONTROL_USE_STATUS,
            help="Rows marked excluded are removed from the active view.",
        )
        st.multiselect(
            "Visible states",
            ["included", "review", "reference", "excluded"],
            default=st.session_state.get(
                CONTROL_VISIBLE_STATES, ["included", "review", "reference"]
            ),
            key=CONTROL_VISIBLE_STATES,
        )
        row_options = _name_options(dataframe)
        selected = st.selectbox("Mark row", [""] + row_options, index=0)
        new_state = st.segmented_control(
            "Set state",
            ["included", "review", "reference", "excluded"],
            default="included",
        )
        if st.button("Apply state", disabled=not selected, width="stretch"):
            key = selected.split(" | ", 1)[0]
            st.session_state[CONTROL_STATUS][key] = new_state

        status_table = _status_table(st, dataframe)
        if not status_table.empty:
            st.dataframe(status_table, hide_index=True, width="stretch", height=180)


def _render_name_filters(st: Any) -> None:
    with st.expander("Name / text filters", expanded=True):
        st.text_input(
            "Include text",
            key=CONTROL_TEXT_INCLUDE,
            placeholder="e.g. Cu-211 Ga clean",
        )
        st.text_input(
            "Exclude text",
            key=CONTROL_TEXT_EXCLUDE,
            placeholder="e.g. convergence test broken",
        )


def _render_materials_search(st: Any, dataframe: pd.DataFrame) -> None:
    with st.expander("Materials Search", expanded=True):
        st.caption(
            "Structured local search inspired by Materials Project and OQMD: formula, "
            "chemical system, element sets, composition size, structure size and "
            "property windows."
        )
        query = st.session_state.setdefault(CONTROL_MATERIAL_QUERY, {})
        elements = _available_elements(dataframe)

        formula_col, chemsys_col = st.columns(2)
        query["formula"] = formula_col.text_input(
            "Formula contains",
            value=query.get("formula", ""),
            placeholder="e.g. Ni3Ga, CO, CH3O",
            key="onepiece_studio_material_formula",
        )
        query["anonymous_formula"] = formula_col.text_input(
            "Anonymous formula",
            value=query.get("anonymous_formula", ""),
            placeholder="e.g. AB, A2B3, AB2",
            help="Formula pattern with element identities removed. Example: Ni3Ga -> AB3.",
            key="onepiece_studio_material_anonymous_formula",
        )
        query["chemsys"] = chemsys_col.text_input(
            "Chemical system",
            value=query.get("chemsys", ""),
            placeholder="e.g. Ni-Ga-O or C-H-O",
            help="Use '-' or spaces between elements.",
            key="onepiece_studio_material_chemsys",
        )
        query["chemsys_mode"] = chemsys_col.segmented_control(
            "Chemical system mode",
            ["contains all", "exact"],
            default=query.get("chemsys_mode", "contains all"),
            key="onepiece_studio_material_chemsys_mode",
        )

        element_col, exclude_col = st.columns(2)
        query["include_elements"] = element_col.multiselect(
            "Required / optional elements",
            elements,
            default=[element for element in query.get("include_elements", []) if element in elements],
            key="onepiece_studio_material_include_elements",
        )
        query["element_mode"] = element_col.segmented_control(
            "Element mode",
            ["all", "any", "exact"],
            default=query.get("element_mode", "all"),
            help="all = every selected element must appear; any = at least one; exact = only selected elements.",
            key="onepiece_studio_material_element_mode",
        )
        query["exclude_elements"] = exclude_col.multiselect(
            "Exclude elements",
            elements,
            default=[element for element in query.get("exclude_elements", []) if element in elements],
            key="onepiece_studio_material_exclude_elements",
        )
        query["record_types"] = exclude_col.multiselect(
            "Record classes",
            _record_type_options(dataframe),
            default=query.get("record_types", []),
            key="onepiece_studio_material_record_types",
        )

        size_cols = st.columns(3)
        element_counts = _row_element_counts(dataframe)
        min_elements = int(max(0, element_counts.min())) if not dataframe.empty else 0
        max_elements = int(max(element_counts.max(), min_elements)) if not dataframe.empty else 0
        if max_elements > min_elements:
            default_nelements = query.get("nelements", (min_elements, max_elements))
            query["nelements"] = size_cols[0].slider(
                "Number of elements",
                min_value=min_elements,
                max_value=max_elements,
                value=(
                    max(min_elements, min(int(default_nelements[0]), max_elements)),
                    max(min_elements, min(int(default_nelements[1]), max_elements)),
                ),
                key="onepiece_studio_material_nelements",
            )
        else:
            query["nelements"] = None
            size_cols[0].info("Only one element count available.")

        atom_counts = _row_atom_counts(dataframe)
        if atom_counts.notna().any():
            min_atoms = int(max(1, atom_counts.dropna().min()))
            max_atoms = int(max(min_atoms, atom_counts.dropna().max()))
            if max_atoms > min_atoms:
                default_natoms = query.get("natoms", (min_atoms, max_atoms))
                query["natoms"] = size_cols[1].slider(
                    "Number of atoms",
                    min_value=min_atoms,
                    max_value=max_atoms,
                    value=(
                        max(min_atoms, min(int(default_natoms[0]), max_atoms)),
                        max(min_atoms, min(int(default_natoms[1]), max_atoms)),
                    ),
                    key="onepiece_studio_material_natoms",
                )
            else:
                query["natoms"] = None
                size_cols[1].info(f"Atom count fixed at {min_atoms}.")
        else:
            query["natoms"] = None
            size_cols[1].info("No atom-count information.")

        quality_options = _quality_flag_options(dataframe)
        query["quality_flags"] = size_cols[2].multiselect(
            "Quality flags",
            quality_options,
            default=[flag for flag in query.get("quality_flags", []) if flag in quality_options],
            key="onepiece_studio_material_quality_flags",
        )

        property_columns = _materials_property_columns(dataframe)
        query["property_columns"] = st.multiselect(
            "Property windows",
            property_columns,
            default=[column for column in query.get("property_columns", []) if column in property_columns],
            help="Range-filter numeric fields such as energy above hull, formation energy, fmax, band gap, atom counts, etc.",
            key="onepiece_studio_material_property_columns",
        )
        property_ranges = query.get("property_ranges", {})
        updated_ranges = {}
        for column in query["property_columns"]:
            finite = _finite(pd.to_numeric(dataframe[column], errors="coerce"))
            if finite.empty or finite.min() == finite.max():
                continue
            default = property_ranges.get(column, (float(finite.min()), float(finite.max())))
            updated_ranges[column] = st.slider(
                f"{column} range",
                min_value=float(finite.min()),
                max_value=float(finite.max()),
                value=(
                    max(float(finite.min()), min(float(default[0]), float(finite.max()))),
                    max(float(finite.min()), min(float(default[1]), float(finite.max()))),
                ),
                key=f"onepiece_studio_material_property_{column}",
            )
        query["property_ranges"] = updated_ranges

        st.code(_query_description(query), language="text")


def _render_domain_filters(st: Any, dataframe: pd.DataFrame) -> None:
    with st.expander("Materials facets", expanded=True):
        selected: dict[str, list[str]] = {}
        facet_columns = _facet_columns(dataframe)
        for column in facet_columns:
            options = sorted(dataframe[column].dropna().astype(str).unique().tolist())
            current = st.session_state[CONTROL_SELECTED_FACETS].get(column, [])
            selected[column] = st.multiselect(column, options, default=current)
        st.session_state[CONTROL_SELECTED_FACETS] = selected

        st.checkbox("Hide convergence calculations by name", key=CONTROL_DROP_CONVERGENCE)
        st.checkbox("Hide test calculations by name", key=CONTROL_DROP_TEST)


def _render_numeric_controls(st: Any, dataframe: pd.DataFrame) -> None:
    with st.expander("Numeric filters", expanded=False):
        if "fmax" in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe["fmax"]):
            finite = _finite(dataframe["fmax"])
            if not finite.empty:
                default_value = _clamp_float(
                    st.session_state.get(CONTROL_FMAX_MAX),
                    minimum=float(finite.min()),
                    maximum=float(finite.max()),
                    fallback=float(finite.max()),
                )
                value = st.number_input(
                    "Maximum fmax",
                    min_value=float(finite.min()),
                    max_value=float(finite.max()),
                    value=default_value,
                    step=0.01,
                )
                st.session_state[CONTROL_FMAX_MAX] = value

        numeric_columns = _numeric_filter_columns(dataframe)
        selected = st.multiselect(
            "Additional numeric columns",
            numeric_columns,
            default=list(st.session_state[CONTROL_NUMERIC].keys()),
        )
        numeric_state: dict[str, tuple[float, float]] = {}
        for column in selected:
            finite = _finite(dataframe[column])
            if finite.empty or finite.min() == finite.max():
                continue
            default = st.session_state[CONTROL_NUMERIC].get(
                column, (float(finite.min()), float(finite.max()))
            )
            numeric_state[column] = st.slider(
                column,
                min_value=float(finite.min()),
                max_value=float(finite.max()),
                value=(
                    max(float(finite.min()), float(default[0])),
                    min(float(finite.max()), float(default[1])),
                ),
            )
        st.session_state[CONTROL_NUMERIC] = numeric_state


def _clamp_float(value: Any, *, minimum: float, maximum: float, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(fallback)
    return min(maximum, max(minimum, numeric))


def _render_control_summary(st: Any, source: pd.DataFrame, active: pd.DataFrame) -> None:
    top = st.columns(2, gap="small")
    bottom = st.columns(2, gap="small")
    top[0].metric("Active rows", f"{len(active):,}")
    top[1].metric("Total rows", f"{len(source):,}")
    bottom[0].metric("Hidden rows", f"{len(source) - len(active):,}")
    if "dataset" in active.columns:
        bottom[1].metric("Datasets", f"{active['dataset'].nunique():,}")
    else:
        bottom[1].metric("Columns", f"{active.shape[1]:,}")

    if "dataset" in active.columns:
        counts = active["dataset"].astype(str).value_counts().rename_axis("dataset").reset_index(name="rows")
        st.dataframe(counts, hide_index=True, width="stretch", height=220)


def _render_commands(st: Any, dataframe: pd.DataFrame) -> None:
    st.markdown("**Notebook commands**")
    command = st.selectbox(
        "Select command",
        [
            "Show active DataFrame",
            "Top low-energy candidates",
            "Find clean reference rows",
            "Find adsorption-like rows",
            "Find quality problems",
            "Phase-diagram candidate table",
            "Column focus review",
            "Column context table",
            "Export active rows",
        ],
    )

    result = _run_command(command, dataframe)
    if command == "Export active rows":
        csv = dataframe.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download active rows as CSV",
            csv,
            file_name="onepiece_studio_controlroom_active_rows.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        command_height = 520 if command == "Show active DataFrame" else 300
        display = _display_command_frame(result)
        event = st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            height=command_height,
            selection_mode="single-row",
            on_select="rerun",
            key=f"onepiece_studio_command_table_{command}",
        )
        selected_index = _selected_dataframe_index(event, display)
        if selected_index is not None and selected_index in result.index:
            _render_row_actions(st, result.loc[selected_index], selected_index, key_prefix="command")


def _render_preview(st: Any, dataframe: pd.DataFrame) -> None:
    st.markdown("**Active DataFrame**")
    st.caption(f"{len(dataframe):,} active rows")
    display = _display_command_frame(dataframe)
    event = st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=640,
        selection_mode="single-row",
        on_select="rerun",
        key=CONTROLROOM_ACTIVE_DATAFRAME,
    )
    selected_index = _selected_dataframe_index(event, display)
    if selected_index is not None and selected_index in dataframe.index:
        _render_row_actions(st, dataframe.loc[selected_index], selected_index, key_prefix="active")


def _selected_dataframe_index(event: Any, display: pd.DataFrame) -> Any | None:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, dict):
        rows = selection.get("rows")
    if not rows:
        return None
    position = int(rows[0])
    if position >= len(display):
        return None
    return display.index[position]


def _render_row_actions(st: Any, row: pd.Series, index: Any, *, key_prefix: str) -> None:
    st.markdown("**Selected calculation**")
    st.dataframe(selected_row_summary(row), hide_index=True, width="stretch")
    render_action_grid(st, row, index, key_prefix=key_prefix, namespace="onepiece_studio_control")


def _apply_controlroom_filters(st: Any, dataframe: pd.DataFrame) -> pd.DataFrame:
    query = DatasetQuery(
        text_include=st.session_state.get(CONTROL_TEXT_INCLUDE, ""),
        text_exclude=st.session_state.get(CONTROL_TEXT_EXCLUDE, ""),
        drop_convergence=bool(st.session_state.get(CONTROL_DROP_CONVERGENCE, True)),
        drop_test=bool(st.session_state.get(CONTROL_DROP_TEST, True)),
        materials=dict(st.session_state.get(CONTROL_MATERIAL_QUERY, {})),
        selected_facets=dict(st.session_state.get(CONTROL_SELECTED_FACETS, {})),
        fmax_max=st.session_state.get(CONTROL_FMAX_MAX),
        numeric_ranges=dict(st.session_state.get(CONTROL_NUMERIC, {})),
        use_status=bool(st.session_state.get(CONTROL_USE_STATUS, True)),
        visible_states=list(st.session_state.get(CONTROL_VISIBLE_STATES, ["included", "review", "reference"])),
    )
    return apply_dataset_query(
        dataframe,
        query,
        row_key_series=_row_keys(dataframe),
        status_map=st.session_state.get(CONTROL_STATUS, {}),
    )


def _run_command(command: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    if command == "Show active DataFrame":
        return dataframe
    if command == "Top low-energy candidates":
        energy = _first_existing(
            dataframe,
            ["form_G_per_Area", "formation_energy_per_atom", "form_G_per_alloy", "E"],
        )
        if energy:
            return dataframe.sort_values(energy).head(40)
    if command == "Find clean reference rows":
        return _filter_text(dataframe, "clean", include=True).head(80)
    if command == "Find adsorption-like rows":
        tokens = ["CO", "CO2", "H2O", "OH", "O2", "H2", "ads"]
        return _filter_any_token(dataframe, tokens).head(80)
    if command == "Find quality problems":
        problems = dataframe.iloc[0:0].copy()
        if "fmax" in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe["fmax"]):
            problems = pd.concat([problems, dataframe[dataframe["fmax"] > 0.05]])
        problems = pd.concat([problems, _filter_text(dataframe, "crash error fail", include=True)])
        return problems.loc[~problems.index.duplicated(keep="first")].head(80)
    if command == "Phase-diagram candidate table":
        columns = [
            c
            for c in [
                "dataset",
                "source_hdf",
                "source_row",
                "Name",
                "Formula",
                "Ga_percent",
                "Monolayer_alloy",
                "hkl",
                "E",
                "formation_energy_per_atom",
                "form_G_per_Area",
                "form_G_per_alloy",
                "fmax",
            ]
            if c in dataframe.columns
        ]
        return dataframe[columns].head(120)
    if command == "Column focus review":
        return profile_review(dataframe)
    if command == "Column context table":
        return column_context(dataframe)
    return dataframe


def _filter_text(dataframe: pd.DataFrame, text: str, *, include: bool) -> pd.DataFrame:
    return backend_filter_text(dataframe, text, include=include)


def _filter_any_token(dataframe: pd.DataFrame, tokens: list[str]) -> pd.DataFrame:
    return backend_filter_any_token(dataframe, tokens)


def _search_haystack(dataframe: pd.DataFrame) -> pd.Series:
    text_columns = [
        column
        for column in ["dataset", "dataset_label", "Name", "Formula", "legend"]
        if column in dataframe.columns
    ]
    haystack = pd.Series("", index=dataframe.index, dtype="object")
    for column in text_columns:
        haystack = haystack + " " + dataframe[column].astype(str)

    for column in ["Path", "path", "source_hdf"]:
        if column in dataframe.columns:
            haystack = haystack + " " + dataframe[column].astype(str).map(_path_tail)

    if not text_columns and all(column not in dataframe.columns for column in ["Path", "path", "source_hdf"]):
        fallback = [column for column in dataframe.columns if dataframe[column].dtype == "object"][:8]
        for column in fallback:
            haystack = haystack + " " + dataframe[column].astype(str)
    return haystack


def _path_tail(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    parts = re.split(r"[\\/]", text)
    if not parts:
        return text
    tail = parts[-1]
    parent = parts[-2] if len(parts) > 1 else ""
    return f"{parent} {tail}".strip()


def _apply_materials_search(dataframe: pd.DataFrame, query: dict[str, Any]) -> pd.DataFrame:
    return backend_apply_materials_search(dataframe, query)


def _available_elements(dataframe: pd.DataFrame) -> list[str]:
    return get_all_elements(dataframe)


def _row_elements(dataframe: pd.DataFrame) -> pd.Series:
    structure_columns = tuple(structure_columns_in_frame(dataframe))
    return dataframe.apply(
        lambda row: tuple(sorted(row_element_count_map(row, structure_columns=structure_columns))),
        axis=1,
    )


def _row_element_counts(dataframe: pd.DataFrame) -> pd.Series:
    return _row_elements(dataframe).map(len)


def _row_atom_counts(dataframe: pd.DataFrame) -> pd.Series:
    if "n_atoms" in dataframe.columns:
        return pd.to_numeric(dataframe["n_atoms"], errors="coerce")
    structure_columns = tuple(structure_columns_in_frame(dataframe))
    counts_by_row = dataframe.apply(
        lambda row: row_element_count_map(row, structure_columns=structure_columns),
        axis=1,
    )
    if not counts_by_row.empty:
        return counts_by_row.map(lambda counts: float(sum(counts.values())) if counts else np.nan)
    return pd.Series(np.nan, index=dataframe.index)


def _record_type_options(dataframe: pd.DataFrame) -> list[str]:
    return sorted(_record_type_series(dataframe).dropna().unique().tolist())


def _record_type_series(dataframe: pd.DataFrame) -> pd.Series:
    text = pd.Series("", index=dataframe.index)
    for column in ["Name", "Path", "dataset", "dataset_label"]:
        if column in dataframe.columns:
            text = text + " " + dataframe[column].astype(str)
    lower = text.str.lower()
    labels = pd.Series("calculation", index=dataframe.index)
    labels[lower.str.contains("gasphase|gas/", regex=True, na=False)] = "gas_reference"
    labels[lower.str.contains("clean", regex=False, na=False)] = "clean_surface"
    labels[lower.str.contains("co|ch3o|hco|co2|oh", regex=True, na=False)] = "adsorbate"
    labels[lower.str.contains("copt", regex=False, na=False)] = "constrained_optimization"
    if "record_type" in dataframe.columns:
        explicit = dataframe["record_type"].astype("string").str.strip()
        has_explicit = explicit.notna() & ~explicit.str.lower().isin(["", "nan", "none", "nat"])
        labels.loc[has_explicit] = explicit.loc[has_explicit].astype(str)
    return labels


def _quality_flag_options(dataframe: pd.DataFrame) -> list[str]:
    if "quality_flag" not in dataframe.columns:
        return []
    return sorted(dataframe["quality_flag"].dropna().astype(str).unique().tolist())


def _range_is_restrictive(values: pd.Series, bounds: Any) -> bool:
    if not bounds:
        return False
    finite = _finite(pd.to_numeric(values, errors="coerce"))
    if finite.empty:
        return False
    lower, upper = float(bounds[0]), float(bounds[1])
    return lower > float(finite.min()) or upper < float(finite.max())


def _materials_property_columns(dataframe: pd.DataFrame) -> list[str]:
    preferred = [
        "energy_above_hull",
        "e_above_hull",
        "stability",
        "delta_e",
        "formation_energy_per_atom",
        "form_G_per_Area",
        "form_G_per_alloy",
        "energy_per_atom",
        "E",
        "band_gap",
        "density",
        "volume",
        "Volume",
        "Area",
        "fmax",
        "n_atoms",
        "Ga_percent",
        "Monolayer_alloy",
    ]
    numeric = [column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])]
    ordered = [column for column in preferred if column in numeric]
    ordered.extend([column for column in numeric if column not in ordered])
    return ordered[:40]


def _query_description(query: dict[str, Any]) -> str:
    return backend_query_description(query)


def _reset_filters(st: Any) -> None:
    st.session_state[CONTROL_TEXT_INCLUDE] = ""
    st.session_state[CONTROL_TEXT_EXCLUDE] = ""
    st.session_state[CONTROL_SELECTED_FACETS] = {}
    st.session_state[CONTROL_NUMERIC] = {}
    st.session_state[CONTROL_MATERIAL_QUERY] = {}
    st.session_state[CONTROL_FMAX_MAX] = None
    st.session_state[CONTROL_VISIBLE_STATES] = ["included", "review", "reference"]
    st.session_state[CONTROL_DROP_CONVERGENCE] = False
    st.session_state[CONTROL_DROP_TEST] = False


def _set_text_filter(st: Any, *, include: str = "", exclude: str = "") -> None:
    st.session_state[CONTROL_TEXT_INCLUDE] = include
    st.session_state[CONTROL_TEXT_EXCLUDE] = exclude


def _set_status_filter(st: Any, states: list[str]) -> None:
    st.session_state[CONTROL_VISIBLE_STATES] = states


def _facet_columns(dataframe: pd.DataFrame) -> list[str]:
    preferred = [
        "dataset",
        "source_hdf",
        "hkl",
        "slabsize",
        "layers",
        "cluster",
        "convergence",
        "clean",
        "adsorbate",
        "adsorbate_species",
        "Surface Alloy",
        "M",
        "MO",
    ]
    columns = []
    for column in preferred:
        if column in dataframe.columns and _safe_unique_count(dataframe[column]) <= 80:
            columns.append(column)
    return columns[:10]


def _numeric_filter_columns(dataframe: pd.DataFrame) -> list[str]:
    preferred = [
        "E",
        "fmax",
        "formation_energy_per_atom",
        "form_G_per_Area",
        "form_G_per_alloy",
        "Ga_percent",
        "Monolayer_alloy",
        "Area",
        "Ga",
        "Cu",
        "O",
        "H",
        "C",
    ]
    return [
        column
        for column in preferred
        if column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column])
    ]


def _name_options(dataframe: pd.DataFrame) -> list[str]:
    rows = []
    keys = _row_keys(dataframe)
    for index, row in dataframe.head(500).iterrows():
        name = str(row.get("Name", index))
        dataset = str(row.get("dataset", ""))
        rows.append(f"{keys.loc[index]} | {dataset} | {name}")
    return rows


def _row_keys(dataframe: pd.DataFrame) -> pd.Series:
    return row_keys(dataframe)


def _status_table(st: Any, dataframe: pd.DataFrame) -> pd.DataFrame:
    status = st.session_state.get(CONTROL_STATUS, {})
    if not status:
        return pd.DataFrame()
    rows = []
    key_to_index = {key: index for index, key in _row_keys(dataframe).items()}
    for key, state in status.items():
        index = key_to_index.get(key)
        row = dataframe.loc[index] if index is not None else {}
        rows.append(
            {
                "row_key": key,
                "state": state,
                "dataset": row.get("dataset", "") if hasattr(row, "get") else "",
                "Name": row.get("Name", "") if hasattr(row, "get") else "",
            }
        )
    return pd.DataFrame(rows)


def _display_command_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    visible = dataframe.copy()
    priority = [
        "dataset",
        "source_hdf",
        "source_row",
        "Name",
        "Formula",
        "E",
        "fmax",
        "formation_energy_per_atom",
        "form_G_per_Area",
        "form_G_per_alloy",
        "hkl",
        "slabsize",
        "Monolayer_alloy",
        "Ga",
        "Cu",
        "O",
        "Path",
    ]
    ordered = [column for column in priority if column in visible.columns]
    ordered.extend([column for column in visible.columns if column not in ordered])
    visible = visible[ordered]
    for column in visible.columns:
        if visible[column].dtype == "object":
            visible[column] = visible[column].map(_short_value)
    return visible


def _short_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    if isinstance(text, str) and len(text) > 180:
        return text[:177] + "..."
    return text


def _finite(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _first_existing(dataframe: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column]):
            return column
    return None


def _safe_unique_count(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.dropna().map(repr).nunique(dropna=True))
