from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from onepiece_studio.ui.row_actions import row_keys

EDIT_STATE_KEY = "onepiece_studio_cell_edits"
logger = logging.getLogger(__name__)


def init_workbook_state(st: Any) -> None:
    st.session_state.setdefault(EDIT_STATE_KEY, {})


def apply_session_edits(st: Any, dataframe: pd.DataFrame) -> pd.DataFrame:
    edits = st.session_state.get(EDIT_STATE_KEY, {})
    if dataframe.empty or not edits:
        return dataframe

    active = dataframe.copy()
    keys = row_keys(active)
    key_to_index = {key: index for index, key in keys.items()}
    for row_key, column_values in edits.items():
        index = key_to_index.get(row_key)
        if index is None:
            continue
        for column, value in column_values.items():
            if column in active.columns:
                active.at[index, column] = value
    return active


def render_workbook(st: Any, source: pd.DataFrame, active: pd.DataFrame) -> None:
    init_workbook_state(st)
    st.markdown("**Workbook**")
    st.caption(
        "Edit a filtered slice of the active DataFrame like a local spreadsheet. "
        "Changes are session-local and immediately feed back into OnePiece Studio."
    )

    if source.empty:
        st.info("No rows available for workbook editing.")
        return

    page_cols = st.columns([0.26, 0.22, 0.52])
    scope = page_cols[0].segmented_control(
        "Scope",
        ["Active rows", "All source rows"],
        default="Active rows",
        key="onepiece_studio_workbook_scope",
    )
    frame = active if scope == "Active rows" else source
    max_rows = min(500, max(len(frame), 1))
    page_size = page_cols[1].selectbox(
        "Rows in editor",
        [25, 50, 100, 200, 500],
        index=2,
        key="onepiece_studio_workbook_page_size",
    )
    page_size = min(page_size, max_rows)
    total_pages = max(1, (len(frame) - 1) // page_size + 1)
    page = page_cols[2].number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key="onepiece_studio_workbook_page",
    )

    if frame.empty:
        st.info("No rows available in the selected workbook scope.")
        return

    start = (int(page) - 1) * page_size
    stop = min(start + page_size, len(frame))
    window = frame.iloc[start:stop].copy()
    window.insert(0, "row_key", row_keys(window).values)

    editable_columns = _editable_columns(window)
    if not editable_columns:
        st.info("No editable plain-text, numeric, or boolean columns are available in this slice.")
        return

    selected_columns = st.multiselect(
        "Visible columns",
        list(window.columns),
        default=_default_workbook_columns(window),
        key="onepiece_studio_workbook_visible_columns",
    )
    if not selected_columns:
        st.warning("Choose at least one column for the workbook view.")
        return

    display = _display_workbook(window[selected_columns])
    disabled_columns = [column for column in display.columns if column not in editable_columns]
    edited = st.data_editor(
        display,
        width="stretch",
        height=560,
        hide_index=True,
        disabled=disabled_columns,
        key=f"onepiece_studio_workbook_editor_{scope}_{page}_{page_size}",
    )

    apply_cols = st.columns([0.34, 0.33, 0.33])
    if apply_cols[0].button("Apply workbook edits", width="stretch"):
        changed = _store_edits(st, window, edited, editable_columns)
        st.success(f"Stored {changed} cell edits for this session.")
        st.rerun()
    if apply_cols[1].button("Clear edits for visible rows", width="stretch"):
        cleared = _clear_row_edits(st, window["row_key"].tolist())
        st.success(f"Cleared {cleared} stored row edits in this workbook page.")
        st.rerun()
    if apply_cols[2].button("Clear all workbook edits", width="stretch"):
        st.session_state[EDIT_STATE_KEY] = {}
        st.success("Cleared all workbook edits in this session.")
        st.rerun()

    st.caption(
        f"Workbook page {page}/{total_pages}: rows {start + 1} to {stop} of {len(frame):,} "
        f"in the selected scope."
    )


def _editable_columns(dataframe: pd.DataFrame) -> list[str]:
    protected = {
        "row_key",
        "source_hdf",
        "source_row",
        "onepiece_studio_source_id",
        "onepiece_studio_source_label",
    }
    editable: list[str] = []
    for column in dataframe.columns:
        if column in protected:
            continue
        series = dataframe[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            editable.append(column)
            continue
        if series.dtype == "object":
            sample = series.dropna().iloc[0] if series.dropna().any() else ""
            if isinstance(sample, str):
                editable.append(column)
    return editable


def _default_workbook_columns(dataframe: pd.DataFrame) -> list[str]:
    preferred = [
        "row_key",
        "dataset",
        "dataset_label",
        "Name",
        "Formula",
        "E",
        "fmax",
        "quality_flag",
        "surface_ref_status",
    ]
    selected = [column for column in preferred if column in dataframe.columns]
    if len(selected) < 8:
        for column in dataframe.columns:
            if column not in selected:
                selected.append(column)
            if len(selected) >= 8:
                break
    return selected


def _display_workbook(dataframe: pd.DataFrame) -> pd.DataFrame:
    display = dataframe.copy()
    for column in display.columns:
        series = display[column]
        if series.dtype == "object":
            sample = series.dropna().iloc[0] if series.dropna().any() else ""
            if not isinstance(sample, str):
                display[column] = series.map(lambda value: str(value) if value is not None else None)
    return display


def _store_edits(st: Any, original: pd.DataFrame, edited: pd.DataFrame, editable_columns: list[str]) -> int:
    edits = st.session_state.setdefault(EDIT_STATE_KEY, {})
    changed = 0
    original_by_key = original.set_index("row_key", drop=False)
    edited_by_key = edited.set_index("row_key", drop=False)
    for row_key in edited_by_key.index:
        if row_key not in original_by_key.index:
            continue
        source_row = original_by_key.loc[row_key]
        edited_row = edited_by_key.loc[row_key]
        for column in editable_columns:
            if column not in edited_by_key.columns or column not in original_by_key.columns:
                continue
            before = source_row[column]
            after = edited_row[column]
            if _values_equal(before, after):
                continue
            edits.setdefault(row_key, {})[column] = after
            changed += 1
    return changed


def _clear_row_edits(st: Any, row_keys_for_page: list[str]) -> int:
    edits = st.session_state.setdefault(EDIT_STATE_KEY, {})
    cleared = 0
    for row_key in row_keys_for_page:
        if row_key in edits:
            edits.pop(row_key, None)
            cleared += 1
    return cleared


def _values_equal(before: Any, after: Any) -> bool:
    try:
        if pd.isna(before) and pd.isna(after):
            return True
    except TypeError:
        logger.debug("Could not compare workbook values with pd.isna() for %s.", type(before).__name__)
    return before == after
