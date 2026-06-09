from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from onepiece.adsorption import structure_columns_in_frame
from onepiece.workflows import WorkflowResult, apply_operations
from onepiece.workflows import apply_operation as backend_apply_operation
from onepiece_studio.state import (
    WORKFLOW_OPERATIONS,
)

WORKFLOW_GAS_LABELS = ("CO", "CO2", "CH3OH", "H2", "H2O")
logger = logging.getLogger(__name__)


NUMERIC_OPERATORS = {
    "+": "sum",
    "-": "difference",
    "*": "product",
    "/": "ratio",
}


FILTER_OPERATORS = [
    "contains",
    "not contains",
    "equals",
    "not equals",
    ">",
    ">=",
    "<",
    "<=",
    "is not empty",
    "is empty",
]


def apply_workflow_operations(st: Any, dataframe: pd.DataFrame) -> WorkflowResult:
    operations = st.session_state.get(WORKFLOW_OPERATIONS, [])
    return apply_operations(dataframe, operations)


def render_workflow_builder(st: Any, source: pd.DataFrame, active: pd.DataFrame, messages: list[str]) -> None:
    _init_state(st)
    st.subheader("Workflow Builder")
    st.caption(
        "Build a local data-flow pipeline before Controlroom filtering: add derived columns, "
        "filter rows, and keep the operations reproducible."
    )

    if messages:
        for message in messages:
            st.warning(message)

    _render_pipeline_metrics(st, source, active)
    _render_beginner_guidance(st, active)

    add_tab, automation_tab, pipeline_tab, preview_tab = st.tabs(
        ["Add Operation", "Notebook Automation", "Pipeline", "Preview"]
    )
    with add_tab:
        _render_add_operation(st, active)
    with automation_tab:
        _render_notebook_automation(st, active)
    with pipeline_tab:
        _render_pipeline(st)
    with preview_tab:
        _render_preview(st, source, active)


def _init_state(st: Any) -> None:
    st.session_state.setdefault(WORKFLOW_OPERATIONS, [])


def _render_pipeline_metrics(st: Any, source: pd.DataFrame, active: pd.DataFrame) -> None:
    top = st.columns(2, gap="small")
    bottom = st.columns(2, gap="small")
    top[0].metric("Input rows", f"{len(source):,}")
    top[1].metric("Output rows", f"{len(active):,}")
    bottom[0].metric("Input columns", f"{source.shape[1]:,}")
    bottom[1].metric("Output columns", f"{active.shape[1]:,}")


def _render_beginner_guidance(st: Any, dataframe: pd.DataFrame) -> None:
    operations = st.session_state.get(WORKFLOW_OPERATIONS, [])
    if dataframe.empty:
        st.info(
            "Start by loading an HDF file or the bundled tutorial dataset from Data Sources. "
            "Once rows are present, this tab can add adsorption, Gibbs, charge, and geometry workflows."
        )
        return
    if operations:
        return
    with st.container(border=True):
        st.markdown("**Suggested first workflow**")
        st.caption(
            "For a first-day catalysis workflow, a good sequence is: "
            "1) assign adsorption references, 2) add Gibbs energies if thermo data exists, "
            "3) add charge or geometry descriptors, 4) inspect the results in Visualize."
        )
        suggestions = [
            "CO adsorption analysis starter",
            "Adsorption + Gibbs analysis starter",
            "Bader/VASP charge descriptors",
            "ASE geometry, site and QC descriptors",
        ]
        st.caption("Ready-made recipes available below: " + ", ".join(f"`{item}`" for item in suggestions))


def _render_add_operation(st: Any, dataframe: pd.DataFrame) -> None:
    operation_type = st.segmented_control(
        "Operation type",
        ["Standard operation", "Add derived column", "Filter rows", "Flag rows"],
        default="Standard operation",
    )

    if operation_type == "Standard operation":
        _render_standard_operation_builder(st, dataframe)
    elif operation_type == "Add derived column":
        _render_add_derived_column(st, dataframe)
    elif operation_type == "Filter rows":
        _render_add_filter(st, dataframe, keep_as_flag=False)
    else:
        _render_add_filter(st, dataframe, keep_as_flag=True)


def _render_add_derived_column(st: Any, dataframe: pd.DataFrame) -> None:
    numeric_columns = [
        column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    all_columns = list(dataframe.columns)
    mode = st.selectbox(
        "Derived column mode",
        [
            "Arithmetic from two numeric columns",
            "Numeric column and scalar",
            "Text contains flag",
            "Set constant value",
            "Fill missing values",
            "Normalize categorical values",
            "Count element into a column",
            "Count all elements into columns",
            "Group rank",
            "Adsorption-energy columns from dataset references",
            "Custom pandas expression",
        ],
    )
    new_column = st.text_input("New column name", placeholder="e.g. adsorption_energy")

    operation: dict[str, Any] | None = None
    suggested_name = ""
    if mode == "Arithmetic from two numeric columns":
        col1, op_col, col2 = st.columns(3)
        left = col1.selectbox("Left column", numeric_columns)
        operator = op_col.selectbox("Operator", list(NUMERIC_OPERATORS))
        right = col2.selectbox("Right column", numeric_columns, index=min(1, len(numeric_columns) - 1))
        suggested_name = _suggest_derived_name_binary(left, operator, right)
        operation = {
            "kind": "derive_binary",
            "new_column": new_column or suggested_name,
            "left": left,
            "operator": operator,
            "right": right,
            "label": f"{new_column or suggested_name} = {left} {operator} {right}",
        }
    elif mode == "Numeric column and scalar":
        col1, op_col, preset_col, scalar_col = st.columns(4)
        left = col1.selectbox("Column", numeric_columns)
        operator = op_col.selectbox("Operator", list(NUMERIC_OPERATORS))
        gas_refs = _workflow_gas_reference_values(st, dataframe)
        preset_values = {
            f"Gas phase: {label} ({value:.6f} eV)": float(value)
            for label, value in gas_refs.items()
            if value is not None
        }
        preset_options = ["Manual"] + list(preset_values)
        selected_preset = preset_col.selectbox("Scalar preset", preset_options)
        default_scalar = 1.0
        if selected_preset != "Manual":
            default_scalar = preset_values[selected_preset]
        scalar = scalar_col.number_input("Scalar", value=float(default_scalar))
        suggested_name = _suggest_derived_name_scalar(left, operator, scalar)
        operation = {
            "kind": "derive_scalar",
            "new_column": new_column or suggested_name,
            "left": left,
            "operator": operator,
            "scalar": scalar,
            "scalar_preset": selected_preset,
            "label": f"{new_column or suggested_name} = {left} {operator} {scalar:g}",
        }
    elif mode == "Text contains flag":
        col1, col2 = st.columns(2)
        column = col1.selectbox("Text column", all_columns)
        token = col2.text_input("Contains text", placeholder="clean")
        suggested_name = _suggest_contains_name(column, token)
        operation = {
            "kind": "derive_contains",
            "new_column": new_column or suggested_name,
            "column": column,
            "token": token,
            "label": f"{new_column or suggested_name} = {column} contains {token!r}",
        }
    elif mode == "Set constant value":
        col1, col2 = st.columns(2)
        constant_type = col1.selectbox("Value type", ["text", "number", "boolean"])
        if constant_type == "number":
            constant_value = col2.number_input("Constant value", value=0.0)
        elif constant_type == "boolean":
            constant_value = col2.checkbox("Constant value", value=True)
        else:
            constant_value = col2.text_input("Constant value", placeholder="e.g. MgOCu-")
        suggested_name = "constant_value"
        operation = {
            "kind": "derive_constant",
            "new_column": new_column or suggested_name,
            "value": constant_value,
            "label": f"{new_column or suggested_name} = constant {constant_value!r}",
        }
    elif mode == "Fill missing values":
        col1, col2, col3 = st.columns(3)
        column = col1.selectbox("Column", all_columns)
        fill_kind = col2.selectbox("Fill with", ["text", "number", "boolean"])
        if fill_kind == "number":
            fill_value = col3.number_input("Fill value", value=0.0)
        elif fill_kind == "boolean":
            fill_value = col3.checkbox("Fill value", value=False)
        else:
            fill_value = col3.text_input("Fill value", placeholder="e.g. 0 or unknown")
        suggested_name = column
        operation = {
            "kind": "fill_missing",
            "column": column,
            "value": fill_value,
            "label": f"fill missing {column} with {fill_value!r}",
        }
    elif mode == "Normalize categorical values":
        col1, col2, col3 = st.columns(3)
        column = col1.selectbox("Column", all_columns)
        from_value = col2.text_input("From value", placeholder="e.g. CHO")
        to_value = col3.text_input("To value", placeholder="e.g. HCO")
        suggested_name = column
        operation = {
            "kind": "replace_value",
            "column": column,
            "from_value": from_value,
            "to_value": to_value,
            "label": f"replace {column}: {from_value!r} -> {to_value!r}",
        }
    elif mode == "Count element into a column":
        structure_columns = structure_columns_in_frame(dataframe)
        if not structure_columns:
            st.info("This mode needs at least one column containing ASE Atoms objects.")
            operation = None
        else:
            col1, col2 = st.columns(2)
            element = col1.text_input("Element symbol", value="C", max_chars=3)
            structure_column = col2.selectbox("Structure source column", structure_columns)
            suggested_name = f"{element}_count"
            operation = {
                "kind": "count_element",
                "new_column": new_column or suggested_name,
                "element": element,
                "structure_column": structure_column,
                "label": f"{new_column or suggested_name} = count element {element} from {structure_column}",
            }
    elif mode == "Count all elements into columns":
        structure_columns = structure_columns_in_frame(dataframe)
        if not structure_columns:
            st.info("This mode needs at least one column containing ASE Atoms objects.")
            operation = None
        else:
            suggested_name = "all_element_count_columns"
            structure_column = st.selectbox("Structure source column", structure_columns, key="onepiece_studio_count_all_elements_structure_column")
            st.caption(
                "Detects all elements present in the current dataset and adds one count column per element "
                f"using `{structure_column}` as the ASE structure source."
            )
            operation = {
                "kind": "count_all_elements",
                "structure_column": structure_column,
                "label": f"count all detected elements from {structure_column} into columns",
            }
    elif mode == "Group rank":
        if not numeric_columns:
            st.info("This mode needs at least one numeric column.")
            operation = None
        else:
            col1, col2, col3, col4 = st.columns(4)
            value_column = col1.selectbox("Rank values in", numeric_columns)
            candidate_groups = [column for column in all_columns if column != value_column]
            group_columns = col2.multiselect("Group by", candidate_groups, default=candidate_groups[:2] if candidate_groups else [])
            ascending = col3.checkbox("Ascending", value=True)
            method = col4.selectbox("Method", ["min", "dense", "first", "average", "max"])
            suggested_name = _sanitize_identifier(f"{value_column}_rank")
            operation = {
                "kind": "group_rank",
                "new_column": new_column or suggested_name,
                "value_column": value_column,
                "group_columns": group_columns,
                "ascending": ascending,
                "method": method,
                "label": f"{new_column or suggested_name} = rank {value_column} by {group_columns or ['all rows']}",
            }
    elif mode == "Adsorption-energy columns from dataset references":
        suggested_name = "adsorption_columns"
        gas_refs = _workflow_gas_reference_values(st, dataframe)
        st.caption(
            "This operation assigns clean surface references from the current dataset and "
            "adds adsorption columns such as `surface_ref_name`, `surface_ref_E`, "
            "`delta_E_to_surface_eV`, and gas-dependent adsorption energies where possible."
        )
        operation = {
            "kind": "derive_adsorption_columns",
            "gas_references": gas_refs,
            "label": "derive adsorption-energy columns from dataset references",
        }
    else:
        st.caption(
            "Expression can use DataFrame column names directly. Example: "
            "`E / n_atoms` or `form_G / Area`. Use only existing numeric columns."
        )
        expression = st.text_area("Expression", placeholder="E / n_atoms")
        suggested_name = "derived_expression"
        operation = {
            "kind": "derive_expression",
            "new_column": new_column or suggested_name,
            "expression": expression,
            "label": f"{new_column or suggested_name} = {expression}",
        }

    effective_name = new_column or suggested_name
    needs_new_column = operation is not None and operation.get("kind") not in {
        "fill_missing",
        "replace_value",
        "derive_adsorption_columns",
    }
    if not new_column and suggested_name:
        st.caption(f"Using suggested column name: `{suggested_name}`")
    if st.button("Add derived-column operation"):
        if operation is None:
            st.error("Operation is not ready yet.")
        elif needs_new_column and not _valid_new_column(effective_name):
            st.error("Column names must start with a letter or underscore and may only contain letters, numbers, and underscores.")
        else:
            _append_operation(st, operation)
            st.rerun()


def _render_standard_operation_builder(st: Any, dataframe: pd.DataFrame) -> None:
    st.markdown("**Standard operations**")
    st.caption(
        "Choose a ready-made workflow step for common chemistry tasks. The selected recipe is "
        "added to the pipeline as one or several reproducible operations."
    )
    recipe = st.selectbox(
        "Recipe",
        [
            "Assign surface references and adsorption columns",
            "Calculate CO adsorption energy per CO",
            "CO adsorption analysis starter",
            "Adsorption + Gibbs analysis starter",
            "Count all detected elements",
            "Bader/VASP charge descriptors",
            "ASE geometry, site and QC descriptors",
        ],
    )
    gas_refs = _workflow_gas_reference_values(st, dataframe)
    has_co_reference = gas_refs.get("CO") is not None
    operations, description = _standard_operation_recipe(recipe, gas_refs)
    st.info(description)
    if recipe != "Assign surface references and adsorption columns":
        if has_co_reference:
            st.caption(f"Current CO(g) reference: {gas_refs['CO']:.6f} eV")
        else:
            st.caption("No CO(g) reference found yet. `E_ads_CO_eV` will remain empty until one is available.")
    st.caption(f"This recipe will add {len(operations)} workflow step{'s' if len(operations) != 1 else ''}.")
    if st.button(
        "Add standard operation",
        width="stretch",
        disabled="Name" not in dataframe.columns or "E" not in dataframe.columns,
    ):
        _append_operations(st, operations)
        st.rerun()


def _render_notebook_automation(st: Any, dataframe: pd.DataFrame) -> None:
    st.markdown("**Notebook automation blocks**")
    st.caption(
        "Turn recurring notebook-style DataFrame command chains into configurable workflow blocks. "
        "Each block expands into normal OnePiece Studio pipeline operations that stay editable and reproducible."
    )
    block = st.selectbox(
        "Automation block",
        [
            "Source labeling and cleanup",
            "Element counting from formula",
            "Adsorbate normalization map",
            "Recipe-based adsorption energy",
            "VASP charge and projected DOS",
            "ASE geometry, site and QC descriptors",
            "Reaction network builder",
            "Curation engine",
            "Structure descriptor workbench",
            "Ranking within groups",
            "Drop named calculations",
        ],
    )
    all_columns = list(dataframe.columns)
    numeric_columns = [
        column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    name_defaults = [column for column in all_columns if str(column).lower() == "name"] or all_columns
    formula_defaults = [column for column in all_columns if "formula" in str(column).lower()] or all_columns

    if block == "Source labeling and cleanup":
        st.info(
            "For notebook sequences like: set a source label, fill missing element counts with 0, "
            "exclude test/copt rows, and drop zero-energy calculations."
        )
        col1, col2 = st.columns(2)
        constant_column = col1.text_input("Constant column name", value="adsorbate_ref")
        constant_value = col2.text_input("Constant value", placeholder="e.g. MgOCu-")
        fill_columns = st.multiselect(
            "Fill missing values with 0 in these columns",
            all_columns,
            default=[column for column in ["Cu", "Ni", "Ga", "Zn", "Mg", "C", "H", "O", "N"] if column in all_columns],
        )
        name_column = st.selectbox("Name column", name_defaults, key="onepiece_studio_nb_name_cleanup")
        exclude_patterns = st.text_area(
            "Exclude rows whose names contain these tokens",
            value="test\ncopt",
            help="One token per line. Each token becomes a backend filter operation.",
        )
        energy_column = st.selectbox(
            "Energy column for nonzero filter",
            numeric_columns or all_columns,
            index=_column_index(numeric_columns or all_columns, "E"),
            key="onepiece_studio_nb_energy_cleanup",
        )
        drop_zero = st.checkbox("Drop rows where energy equals 0", value=True)
        operations = []
        if constant_column.strip() and constant_value.strip():
            operations.append(
                {
                    "kind": "derive_constant",
                    "new_column": constant_column.strip(),
                    "value": constant_value,
                    "label": f"{constant_column.strip()} = constant {constant_value!r}",
                }
            )
        for column in fill_columns:
            operations.append(
                {
                    "kind": "fill_missing",
                    "column": column,
                    "value": 0.0,
                    "label": f"fill missing {column} with 0.0",
                }
            )
        for token in _split_nonempty_lines(exclude_patterns):
            operations.append(
                {
                    "kind": "filter",
                    "column": name_column,
                    "operator": "not contains",
                    "value": token,
                    "new_column": "",
                    "label": f"filter {name_column} not contains {token!r}",
                }
            )
        if drop_zero:
            operations.append(
                {
                    "kind": "filter",
                    "column": energy_column,
                    "operator": "not equals",
                    "value": "0",
                    "new_column": "",
                    "label": f"filter {energy_column} not equals 0",
                }
            )
        st.caption(f"This block will add {len(operations)} workflow step{'s' if len(operations) != 1 else ''}.")
        if st.button("Add automation block", key="onepiece_studio_nb_cleanup_add", width="stretch"):
            _append_operations(st, operations)
            st.rerun()

    elif block == "Element counting from formula":
        st.info(
            "For notebook steps such as counting C, H, O, N or metal atoms from the `Formula` column "
            "into explicit numeric DataFrame columns."
        )
        col1, col2 = st.columns(2)
        formula_column = col1.selectbox("Formula column", formula_defaults, key="onepiece_studio_nb_formula_count")
        elements_text = col2.text_input("Elements", value="C,H,O,N")
        operations = [
            {
                "kind": "count_element",
                "new_column": f"{element}_count",
                "element": element,
                "formula_column": formula_column,
                "label": f"{element}_count = count element {element} from {formula_column}",
            }
            for element in _split_csv_tokens(elements_text)
        ]
        st.caption(f"This block will add {len(operations)} workflow step{'s' if len(operations) != 1 else ''}.")
        if st.button("Add automation block", key="onepiece_studio_nb_count_add", width="stretch"):
            _append_operations(st, operations)
            st.rerun()

    elif block == "Adsorbate normalization map":
        st.info(
            "For notebook sequences like `unify_adsorbates(...)`, where several adsorbate labels are "
            "systematically renamed to a common standard."
        )
        col1, col2 = st.columns(2)
        column = col1.selectbox(
            "Column to normalize",
            [column for column in all_columns if pd.api.types.is_object_dtype(dataframe[column])] or all_columns,
            index=_column_index(all_columns, "adsorbate"),
            key="onepiece_studio_nb_norm_column",
        )
        mapping_table = col2.data_editor(
            _default_normalization_table(),
            hide_index=True,
            width="stretch",
            num_rows="dynamic",
            key="onepiece_studio_nb_norm_table_editor",
            column_config={
                "from_value": st.column_config.TextColumn("From"),
                "to_value": st.column_config.TextColumn("To"),
            },
        )
        operations = [
            {
                "kind": "replace_value",
                "column": column,
                "from_value": old,
                "to_value": new,
                "label": f"replace {column}: {old!r} -> {new!r}",
            }
            for old, new in _normalization_pairs_from_table(mapping_table)
        ]
        st.caption(f"This block will add {len(operations)} workflow step{'s' if len(operations) != 1 else ''}.")
        if st.button("Add automation block", key="onepiece_studio_nb_norm_add", width="stretch"):
            _append_operations(st, operations)
            st.rerun()

    elif block == "Recipe-based adsorption energy":
        st.info(
            "For notebook functions like `ads_E(...)`: define gas-phase energies and recipes for adsorbates, "
            "then let OnePiece Studio compute total and per-adsorbate adsorption energies in the backend."
        )
        gas_defaults = _workflow_gas_reference_values(st, dataframe)
        col1, col2 = st.columns(2)
        gas_editor = col1.data_editor(
            _default_gas_reference_table(gas_defaults),
            hide_index=True,
            width="stretch",
            num_rows="dynamic",
            key="onepiece_studio_nb_recipe_gases_editor",
            column_config={
                "species": st.column_config.TextColumn("Gas species"),
                "energy_eV": st.column_config.NumberColumn("Energy / eV", format="%.6f"),
            },
        )
        recipe_editor = col2.data_editor(
            _default_recipe_table(),
            hide_index=True,
            width="stretch",
            num_rows="dynamic",
            key="onepiece_studio_nb_recipe_table_editor",
            column_config={
                "adsorbate": st.column_config.TextColumn("Adsorbate"),
                "basis": st.column_config.TextColumn("Basis"),
                "CO": st.column_config.NumberColumn("CO", format="%.3f"),
                "H2": st.column_config.NumberColumn("H2", format="%.3f"),
                "H2O": st.column_config.NumberColumn("H2O", format="%.3f"),
                "CH3OH": st.column_config.NumberColumn("CH3OH", format="%.3f"),
                "CO2": st.column_config.NumberColumn("CO2", format="%.3f"),
                "NH3": st.column_config.NumberColumn("NH3", format="%.3f"),
            },
        )
        gases = _gas_reference_mapping_from_table(gas_editor)
        recipes = _adsorption_recipes_from_table(recipe_editor)
        operation = {
            "kind": "derive_recipe_adsorption",
            "gas_reference_values": gases,
            "recipes": recipes,
            "label": f"derive recipe-based adsorption energies for {list(recipes)}",
        }
        st.caption(
            f"This block will add 1 workflow step and currently defines {len(gases)} gas references and {len(recipes)} recipes."
        )
        if st.button("Add automation block", key="onepiece_studio_nb_recipe_add", width="stretch", disabled=not recipes):
            _append_operation(st, operation)
            st.rerun()

    elif block == "Reaction network builder":
        st.info(
            "Annotate static states and constrained-optimization images into a reaction-network table "
            "with state labels, elementary-step families, and pathway roles."
        )
        operation = {
            "kind": "derive_reaction_network",
            "label": "annotate reaction-network states and copt pathways",
        }
        st.caption(
            "This block will add reaction columns such as `reaction_state`, `reaction_step_initial`, "
            "`reaction_step_final`, `reaction_family`, and `reaction_network_role`."
        )
        if st.button("Add automation block", key="onepiece_studio_nb_reaction_add", width="stretch"):
            _append_operation(st, operation)
            st.rerun()

    elif block == "VASP charge and projected DOS":
        st.info(
            "Read `ACF.dat` or `CHGCAR` files from each calculation folder, derive atomic charge "
            "descriptors, then compare adsorbate-side charge against the matched clean surface and "
            "against gas-phase or valence-electron references where available."
        )
        path_candidates = [column for column in all_columns if "path" in str(column).lower()] or all_columns
        structure_candidates = [column for column in all_columns if "struc" in str(column).lower()] or all_columns
        col1, col2, col3, col4 = st.columns(4)
        calculation_path_column = col1.selectbox(
            "Calculation path column",
            path_candidates,
            index=_column_index(path_candidates, "Path"),
            key="onepiece_studio_nb_vasp_path_column",
        )
        structure_column = col2.selectbox(
            "Structure column",
            structure_candidates,
            index=_column_index(structure_candidates, "struc"),
            key="onepiece_studio_nb_vasp_structure_column",
        )
        charge_source_label = col3.selectbox(
            "Charge source",
            ["ACF.dat (default)", "CHGCAR integration"],
            key="onepiece_studio_nb_vasp_charge_source",
        )
        charge_source = "acf" if charge_source_label.startswith("ACF.dat") else "chgcar"
        add_pdos = col4.checkbox("Also integrate PDOS", value=False, key="onepiece_studio_nb_vasp_add_pdos")

        operations = [
            {
                "kind": "derive_vasp_charge_descriptors",
                "charge_source": charge_source,
                "calculation_path_column": calculation_path_column,
                "structure_column": structure_column,
                "label": f"derive {charge_source.upper()}-preferred charge descriptors and adsorption-style charge references",
            }
        ]
        pdos_integrations: list[dict[str, Any]] = []
        if add_pdos:
            pdos_table = st.data_editor(
                _default_pdos_table(),
                hide_index=True,
                width="stretch",
                num_rows="dynamic",
                key="onepiece_studio_nb_vasp_pdos_editor",
                column_config={
                    "column": st.column_config.TextColumn("Column name"),
                    "elements": st.column_config.TextColumn("Elements (CSV)"),
                    "orbitals": st.column_config.TextColumn("Orbitals (CSV)"),
                    "emin": st.column_config.NumberColumn("E min / eV", format="%.2f"),
                    "emax": st.column_config.NumberColumn("E max / eV", format="%.2f"),
                    "spin": st.column_config.SelectboxColumn(
                        "Spin",
                        options=["sum", "up", "down"],
                    ),
                },
            )
            pdos_integrations = _pdos_integrations_from_table(pdos_table)
            if pdos_integrations:
                operations.append(
                    {
                        "kind": "derive_vasp_pdos_descriptors",
                        "calculation_path_column": calculation_path_column,
                        "structure_column": structure_column,
                        "integrations": pdos_integrations,
                        "label": f"derive projected DOS descriptors for {len(pdos_integrations)} integrations",
                    }
                )
        st.caption(
            "This block will add charge descriptors such as "
            "`adsorbate_net_charge_e`, `surface_net_charge_delta_vs_ref_e`, and "
            "`adsorbate_charge_delta_vs_ref_e`."
        )
        st.caption(
            "When `ACF.dat` is selected, OnePiece Studio uses Bader electron populations by default and "
            "falls back to `CHGCAR` integration only when no `ACF.dat` is available."
        )
        st.caption(
            f"This block will add {len(operations)} workflow step{'s' if len(operations) != 1 else ''}."
        )
        if st.button("Add automation block", key="onepiece_studio_nb_vasp_add", width="stretch"):
            _append_operations(st, operations)
            st.rerun()

    elif block == "ASE geometry, site and QC descriptors":
        st.info(
            "Build ASE-native slab, adsorption-site, local-environment, and quality-control descriptors. "
            "Optionally read DOSCAR to add metal d-band center and filling summaries."
        )
        path_candidates = [column for column in all_columns if "path" in str(column).lower()] or all_columns
        structure_candidates = [column for column in all_columns if "struc" in str(column).lower()] or all_columns
        col1, col2, col3 = st.columns(3)
        calculation_path_column = col1.selectbox(
            "Calculation path column",
            path_candidates,
            index=_column_index(path_candidates, "Path"),
            key="onepiece_studio_nb_ase_path_column",
        )
        structure_column = col2.selectbox(
            "Structure column",
            structure_candidates,
            index=_column_index(structure_candidates, "struc"),
            key="onepiece_studio_nb_ase_structure_column",
        )
        include_pdos = col3.checkbox(
            "Also derive DOSCAR d-band descriptors",
            value=False,
            key="onepiece_studio_nb_ase_include_pdos",
        )
        operation = {
            "kind": "derive_ase_analysis_descriptors",
            "calculation_path_column": calculation_path_column,
            "structure_column": structure_column,
            "include_pdos": include_pdos,
            "label": "derive ASE geometry, adsorption-site, QC, and optional d-band descriptors",
        }
        st.caption(
            "This block will add columns such as `adsorption_site`, `adsorbate_tilt_deg`, "
            "`surface_reconstruction_rmsd`, `adsorbate_is_dissociated`, `adsorbate_desorbed`, "
            "`min_interatomic_distance`, and optionally `metal_d_band_center_eV`."
        )
        if st.button("Add automation block", key="onepiece_studio_nb_ase_analysis_add", width="stretch"):
            _append_operation(st, operation)
            st.rerun()

    elif block == "Curation engine":
        st.info(
            "Apply reproducible DFT quality-control rules: energy and structure presence, static and copt "
            "force thresholds, and name-based exclusion tokens."
        )
        col1, col2, col3 = st.columns(3)
        static_fmax_max = col1.number_input("Static fmax max", value=0.05, min_value=0.0, step=0.01)
        copt_fmax_max = col2.number_input("COPT fmax max", value=0.10, min_value=0.0, step=0.01)
        action = col3.selectbox("Action", ["exclude", "mark_review", "mark_excluded"])
        exclude_name_tokens = st.text_area(
            "Name tokens to flag",
            value="test\nconvergence\nfailed\nbroken",
            help="One token per line. Matching rows are flagged by the curation engine.",
        )
        operation = {
            "kind": "derive_curation",
            "static_fmax_max": float(static_fmax_max),
            "copt_fmax_max": float(copt_fmax_max),
            "exclude_name_tokens": _split_nonempty_lines(exclude_name_tokens),
            "action": action,
            "label": f"curate calculations ({action}) with fmax thresholds {static_fmax_max:g}/{copt_fmax_max:g}",
        }
        if st.button("Add automation block", key="onepiece_studio_nb_curation_add", width="stretch"):
            _append_operation(st, operation)
            st.rerun()

    elif block == "Structure descriptor workbench":
        st.info(
            "Build structure-derived catalytic descriptors from ASE `Atoms`: adsorbate composition, "
            "adsorbate size, cell volume, and height above the matched clean surface."
        )
        operation = {
            "kind": "derive_structure_descriptors",
            "label": "derive structure descriptors from ASE structures and clean references",
        }
        st.caption(
            "This block will add columns such as `adsorbate_formula`, `adsorbate_atom_count`, "
            "`cell_volume`, and `adsorbate_height_above_surface`."
        )
        if st.button("Add automation block", key="onepiece_studio_nb_descriptors_add", width="stretch"):
            _append_operation(st, operation)
            st.rerun()

    elif block == "Ranking within groups":
        st.info(
            "For notebook commands like `groupby(...)[\"E\"].rank(...)`, for example ranking "
            "adsorbates within each surface reference."
        )
        col1, col2, col3, col4 = st.columns(4)
        value_column = col1.selectbox(
            "Rank values in",
            numeric_columns or all_columns,
            index=_column_index(numeric_columns or all_columns, "E"),
            key="onepiece_studio_nb_rank_value",
        )
        default_groups = [column for column in ["adsorbate_ref", "adsorbate", "surface_ref"] if column in all_columns]
        group_columns = col2.multiselect(
            "Group by",
            [column for column in all_columns if column != value_column],
            default=default_groups,
            key="onepiece_studio_nb_rank_groups",
        )
        ascending = col3.checkbox("Ascending", value=True, key="onepiece_studio_nb_rank_asc")
        method = col4.selectbox("Method", ["min", "dense", "first", "average", "max"], key="onepiece_studio_nb_rank_method")
        new_column = st.text_input("Rank column name", value="ranked", key="onepiece_studio_nb_rank_name")
        operation = {
            "kind": "group_rank",
            "new_column": new_column,
            "value_column": value_column,
            "group_columns": group_columns,
            "ascending": ascending,
            "method": method,
            "label": f"{new_column} = rank {value_column} by {group_columns or ['all rows']}",
        }
        if st.button("Add automation block", key="onepiece_studio_nb_rank_add", width="stretch"):
            _append_operation(st, operation)
            st.rerun()

    else:
        st.info(
            "For notebook clean-up cells where known bad, test, or dissociated calculations are "
            "removed by curated matching rules."
        )
        name_column = st.selectbox("Name column", name_defaults, key="onepiece_studio_nb_drop_namecol")
        rules_table = st.data_editor(
            _default_drop_rules_table(),
            hide_index=True,
            width="stretch",
            num_rows="dynamic",
            key="onepiece_studio_nb_drop_rules_editor",
            column_config={
                "pattern": st.column_config.TextColumn("Pattern"),
                "match_mode": st.column_config.SelectboxColumn(
                    "Match mode",
                    options=["exact", "contains", "regex"],
                    required=True,
                ),
                "reason": st.column_config.TextColumn("Reason"),
            },
        )
        rules = _drop_rules_from_table(rules_table)
        operation = {
            "kind": "exclude_by_match_rules",
            "column": name_column,
            "rules": rules,
            "label": f"drop {len(rules)} curated name rules from {name_column}",
        }
        if rules:
            exact_count = sum(1 for rule in rules if rule["match_mode"] == "exact")
            contains_count = sum(1 for rule in rules if rule["match_mode"] == "contains")
            regex_count = sum(1 for rule in rules if rule["match_mode"] == "regex")
            st.caption(
                f"This block will add 1 workflow step with {exact_count} exact, "
                f"{contains_count} contains, and {regex_count} regex rule"
                f"{'' if len(rules) == 1 else 's'}."
            )
        else:
            st.caption("This block will add 0 workflow steps until at least one valid rule is defined.")
        if st.button("Add automation block", key="onepiece_studio_nb_drop_add", width="stretch", disabled=not rules):
            _append_operation(st, operation)
            st.rerun()


def _render_add_filter(st: Any, dataframe: pd.DataFrame, *, keep_as_flag: bool) -> None:
    columns = list(dataframe.columns)
    col1, col2, col3 = st.columns([0.34, 0.28, 0.38])
    column = col1.selectbox("Column", columns)
    operator = col2.selectbox("Condition", FILTER_OPERATORS)
    value = ""
    if operator not in {"is empty", "is not empty"}:
        value = col3.text_input("Value", placeholder="Cu, 0.05, clean")
    flag_column = ""
    if keep_as_flag:
        flag_column = st.text_input("Flag column name", value=f"{column}_flag")
    label = (
        f"{'flag' if keep_as_flag else 'filter'} {column} {operator}"
        + (f" {value!r}" if value else "")
    )
    operation = {
        "kind": "flag_filter" if keep_as_flag else "filter",
        "column": column,
        "operator": operator,
        "value": value,
        "new_column": flag_column,
        "label": label,
    }
    if st.button("Add operation", disabled=keep_as_flag and not _valid_new_column(flag_column)):
        _append_operation(st, operation)
        st.rerun()


def _render_pipeline(st: Any) -> None:
    operations = st.session_state.get(WORKFLOW_OPERATIONS, [])
    if not operations:
        st.info("No workflow operations yet.")
        return

    rows = []
    for index, operation in enumerate(operations):
        rows.append(
            {
                "step": index + 1,
                "enabled": operation.get("enabled", True),
                "kind": operation.get("kind", ""),
                "label": operation.get("label", ""),
                "created_at": operation.get("created_at", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    selected = st.number_input("Step", min_value=1, max_value=len(operations), value=1, step=1)
    index = int(selected) - 1
    controls = st.columns(5)
    if controls[0].button("Toggle enabled", width="stretch"):
        operations[index]["enabled"] = not operations[index].get("enabled", True)
        st.rerun()
    if controls[1].button("Move up", width="stretch", disabled=index == 0):
        operations[index - 1], operations[index] = operations[index], operations[index - 1]
        st.rerun()
    if controls[2].button("Move down", width="stretch", disabled=index == len(operations) - 1):
        operations[index + 1], operations[index] = operations[index], operations[index + 1]
        st.rerun()
    if controls[3].button("Delete step", width="stretch"):
        operations.pop(index)
        st.rerun()
    if controls[4].button("Clear all", width="stretch"):
        st.session_state[WORKFLOW_OPERATIONS] = []
        st.rerun()

    st.download_button(
        "Download workflow JSON",
        pd.Series(operations).to_json(indent=2).encode("utf-8"),
        file_name="onepiece_studio_workflow.json",
        mime="application/json",
    )


def _render_preview(st: Any, source: pd.DataFrame, active: pd.DataFrame) -> None:
    st.markdown("**Data-flow preview**")
    st.caption("The table below is the DataFrame after workflow operations, before Controlroom filters.")
    st.dataframe(_display_preview(active.head(120)), hide_index=True, width="stretch", height=520)

    st.markdown("**Columns created by workflow**")
    created = [
        op.get("new_column")
        for op in st.session_state.get(WORKFLOW_OPERATIONS, [])
        if op.get("new_column") in active.columns and op.get("new_column") not in source.columns
    ]
    if created:
        overview = []
        for column in created:
            series = active[column]
            overview.append(
                {
                    "column": column,
                    "dtype": str(series.dtype),
                    "non_null": int(series.notna().sum()),
                    "sample": repr(series.dropna().iloc[0])[:120] if series.notna().any() else "",
                }
            )
        st.dataframe(pd.DataFrame(overview), hide_index=True, width="stretch")
    else:
        st.info("No workflow-created columns yet.")


def _append_operation(st: Any, operation: dict[str, Any] | None) -> None:
    if not operation:
        return
    operation = dict(operation)
    operation["enabled"] = True
    operation["created_at"] = datetime.now().isoformat(timespec="seconds")
    st.session_state.setdefault(WORKFLOW_OPERATIONS, []).append(operation)


def _append_operations(st: Any, operations: list[dict[str, Any]]) -> None:
    for operation in operations:
        _append_operation(st, operation)


def _apply_operation(dataframe: pd.DataFrame, operation: dict[str, Any]) -> pd.DataFrame:
    return backend_apply_operation(dataframe, operation)


def _valid_new_column(name: str) -> bool:
    return bool(name and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def _column_index(columns: list[str], column: str | None, *, fallback: int = 0) -> int:
    if not columns:
        return 0
    if column in columns:
        return columns.index(column)
    return min(max(fallback, 0), len(columns) - 1)


def _sanitize_identifier(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "derived_column"
    if cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def _suggest_derived_name_binary(left: str, operator: str, right: str) -> str:
    suffix = {
        "+": "plus",
        "-": "minus",
        "*": "times",
        "/": "per",
    }.get(operator, "op")
    return _sanitize_identifier(f"{left}_{suffix}_{right}")


def _suggest_derived_name_scalar(left: str, operator: str, scalar: float) -> str:
    suffix = {
        "+": "plus",
        "-": "minus",
        "*": "times",
        "/": "per",
    }.get(operator, "op")
    scalar_label = str(scalar).replace(".", "_").replace("-", "neg_")
    return _sanitize_identifier(f"{left}_{suffix}_{scalar_label}")


def _workflow_gas_reference_values(st: Any, dataframe: pd.DataFrame) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for label in WORKFLOW_GAS_LABELS:
        state_key = f"onepiece_studio_ads_gas_value_{label}"
        if state_key in st.session_state:
            try:
                values[label] = float(st.session_state[state_key])
                continue
            except (TypeError, ValueError):
                pass

    missing = [label for label in WORKFLOW_GAS_LABELS if label not in values]
    if missing:
        try:
            from onepiece.sources import gas_reference_candidates

            candidates = gas_reference_candidates(dataframe)
            for label in missing:
                frame = candidates.get(label)
                if frame is not None and not frame.empty:
                    values[label] = float(frame.iloc[0]["E"])
        except Exception as exc:
            logger.debug("Could not infer fallback gas references from the dataframe: %s", exc)
    return values




def _standard_operation_recipe(
    recipe: str,
    gas_refs: dict[str, float | None],
) -> tuple[list[dict[str, Any]], str]:
    adsorption_step = {
        "kind": "derive_adsorption_columns",
        "gas_references": gas_refs,
        "label": "assign surface references and derive adsorption columns",
    }
    if recipe == "Assign surface references and adsorption columns":
        return (
            [adsorption_step],
            "Adds `surface_ref_name`, `surface_ref_E`, `delta_E_to_surface_eV`, and the gas-dependent adsorption columns where the required references are available.",
        )
    if recipe == "Calculate CO adsorption energy per CO":
        return (
            [
                {
                    **adsorption_step,
                    "label": "calculate CO adsorption energy per CO from dataset references",
                    "preset": "co_adsorption_per_co",
                }
            ],
            "Calculates `n_CO_adsorbates` and `E_ads_CO_eV` from the dataset references. This is the direct workflow step for CO adsorption-energy analysis.",
        )
    if recipe == "CO adsorption analysis starter":
        return (
            [
                {
                    **adsorption_step,
                    "label": "calculate CO adsorption energy per CO from dataset references",
                    "preset": "co_adsorption_per_co",
                },
                {
                    "kind": "filter",
                    "column": "adsorbate",
                    "operator": "equals",
                    "value": "CO",
                    "new_column": "",
                    "label": "filter adsorbate equals 'CO'",
                },
                {
                    "kind": "filter",
                    "column": "E_ads_CO_eV",
                    "operator": "is not empty",
                    "value": "",
                    "new_column": "",
                    "label": "filter E_ads_CO_eV is not empty",
                },
            ],
            "Builds a ready-to-plot CO workflow: first derive the adsorption columns, then keep only CO rows with a filled `E_ads_CO_eV` value.",
        )
    if recipe == "Adsorption + Gibbs analysis starter":
        return (
            [
                {
                    **adsorption_step,
                    "label": "assign surface references and derive adsorption columns for thermochemistry-ready rows",
                },
                {
                    "kind": "derive_gibbs_free_energy",
                    "temperature": 298.15,
                    "energy_column": "E",
                    "output_column": "G",
                    "label": "derive Gibbs free energies at 298.15 K",
                },
                {
                    "kind": "derive_gibbs_adsorption",
                    "gas_references": gas_refs,
                    "temperature": 298.15,
                    "energy_column": "E",
                    "gibbs_column": "G",
                    "output_column": "adsorption_free_energy",
                    "label": "derive adsorption Gibbs free energies from dataset references",
                },
            ],
            "Builds a thermochemistry-ready workflow: assign clean surface references, derive `G`, then calculate `adsorption_free_energy` where the required row-local thermo columns and gas references are available.",
        )
    if recipe == "Count all detected elements":
        return (
            [
                {
                    "kind": "count_all_elements",
                    "label": "count all detected elements into columns",
                }
            ],
            "Scans the current dataset for all present elements and adds one count column per element. Structure columns are preferred, with Formula and existing element columns as fallback.",
        )
    if recipe == "Bader/VASP charge descriptors":
        return (
            [
                {
                    "kind": "derive_vasp_charge_descriptors",
                    "charge_source": "acf",
                    "calculation_path_column": "Path",
                    "structure_column": "struc",
                    "label": "derive ACF.dat-preferred charge descriptors and adsorption-style charge references",
                }
            ],
            "Reads `ACF.dat` files by default, falls back to `CHGCAR` if needed, and compares adsorbate-side charge against the clean surface reference and against gas-phase or valence references.",
        )
    if recipe == "ASE geometry, site and QC descriptors":
        return (
            [
                {
                    "kind": "derive_ase_analysis_descriptors",
                    "calculation_path_column": "Path",
                    "structure_column": "struc",
                    "include_pdos": False,
                    "label": "derive ASE geometry, adsorption-site, and QC descriptors",
                }
            ],
            "Adds ASE-native descriptors such as slab thickness, coordination, adsorption-site class, dissociation/desorption flags, and surface reconstruction metrics.",
        )
    raise ValueError(f"Unsupported standard operation recipe: {recipe}")


def _suggest_contains_name(column: str, token: str) -> str:
    token_part = token or "match"
    return _sanitize_identifier(f"{column}_{token_part}_flag")


def _display_preview(dataframe: pd.DataFrame) -> pd.DataFrame:
    display = dataframe.copy()
    for column in display.columns:
        if display[column].dtype == "object":
            display[column] = display[column].map(_short_value)
    return display


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
    return text if len(text) <= 180 else text[:177] + "..."


def _split_nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def _split_csv_tokens(text: str) -> list[str]:
    tokens = [token.strip() for token in str(text).split(",")]
    return [token for token in tokens if token]


def _parse_mapping_lines(text: str) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for line in _split_nonempty_lines(text):
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            mappings.append((left, right))
    return mappings


def _default_normalization_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"from_value": "CHO", "to_value": "HCO"},
            {"from_value": "CO_NH2", "to_value": "NH2_CO"},
            {"from_value": "H2NHCO", "to_value": "H2NCHO"},
            {"from_value": "H2NHCO_H", "to_value": "H2NCHO_H"},
        ]
    )


def _normalization_pairs_from_table(table: pd.DataFrame) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if table is None or table.empty:
        return pairs
    for row in table.to_dict("records"):
        old = str(row.get("from_value", "")).strip()
        new = str(row.get("to_value", "")).strip()
        if old and new:
            pairs.append((old, new))
    return pairs


def _default_drop_rules_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pattern": "test", "match_mode": "contains", "reason": "Temporary or exploratory calculations"},
            {"pattern": "copt", "match_mode": "contains", "reason": "Constrained optimization helper rows"},
            {"pattern": "-diss", "match_mode": "contains", "reason": "Known dissociated structures"},
        ]
    )


def _drop_rules_from_table(table: pd.DataFrame) -> list[dict[str, str]]:
    rules: list[dict[str, str]] = []
    if table is None or table.empty:
        return rules
    for row in table.to_dict("records"):
        pattern = str(row.get("pattern", "")).strip()
        match_mode = str(row.get("match_mode", "exact")).strip().lower() or "exact"
        reason = str(row.get("reason", "")).strip()
        if not pattern or match_mode not in {"exact", "contains", "regex"}:
            continue
        rules.append({"pattern": pattern, "match_mode": match_mode, "reason": reason})
    return rules


def _parse_float_mapping_lines(text: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for line in _split_nonempty_lines(text):
        if "=" not in line:
            continue
        left, right = line.split("=", 1)
        key = left.strip()
        value = right.strip()
        if not key:
            continue
        try:
            mapping[key] = float(value)
        except ValueError:
            continue
    return mapping


def _parse_adsorption_recipe_lines(text: str) -> dict[str, dict[str, object]]:
    recipes: dict[str, dict[str, object]] = {}
    for line in _split_nonempty_lines(text):
        if ":" not in line:
            continue
        label, payload = line.split(":", 1)
        label = label.strip()
        if not label:
            continue
        recipe_part, *option_parts = payload.split("|")
        gas_refs: dict[str, float] = {}
        for token in recipe_part.split(","):
            token = token.strip()
            if not token or "=" not in token:
                continue
            species, coefficient = token.split("=", 1)
            species = species.strip()
            try:
                gas_refs[species] = float(coefficient.strip())
            except ValueError:
                continue
        basis = "C"
        for option in option_parts:
            option = option.strip()
            if option.lower().startswith("basis="):
                basis = option.split("=", 1)[1].strip() or "C"
        if gas_refs:
            recipes[label] = {"basis": basis, "gas_refs": gas_refs}
    return recipes


def _default_gas_reference_table(gas_defaults: dict[str, float | None]) -> pd.DataFrame:
    rows = []
    for species in ["CO", "H2", "H2O", "CH3OH", "CO2", "NH3"]:
        rows.append(
            {
                "species": species,
                "energy_eV": gas_defaults.get(species) if gas_defaults.get(species) is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _default_recipe_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"adsorbate": "CO", "basis": "C", "CO": 1.0, "H2": 0.0, "H2O": 0.0, "CH3OH": 0.0, "CO2": 0.0, "NH3": 0.0},
            {"adsorbate": "CH3O", "basis": "C", "CO": 1.0, "H2": 1.5, "H2O": 0.0, "CH3OH": 0.0, "CO2": 0.0, "NH3": 0.0},
            {"adsorbate": "HCO", "basis": "C", "CO": 1.0, "H2": 0.5, "H2O": 0.0, "CH3OH": 0.0, "CO2": 0.0, "NH3": 0.0},
            {"adsorbate": "OH", "basis": "O", "CO": 0.0, "H2": -0.5, "H2O": 1.0, "CH3OH": 0.0, "CO2": 0.0, "NH3": 0.0},
        ]
    )


def _default_pdos_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": "metal_d_pdos_below_ef",
                "elements": "Cu,Ni,Ga,Zn",
                "orbitals": "d",
                "emin": -2.0,
                "emax": 0.0,
                "spin": "sum",
            }
        ]
    )


def _pdos_integrations_from_table(table: pd.DataFrame) -> list[dict[str, Any]]:
    integrations: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        column = str(row.get("column", "")).strip()
        elements = _split_csv_tokens(str(row.get("elements", "")))
        orbitals = _split_csv_tokens(str(row.get("orbitals", "")))
        emin = pd.to_numeric(pd.Series([row.get("emin")]), errors="coerce").iloc[0]
        emax = pd.to_numeric(pd.Series([row.get("emax")]), errors="coerce").iloc[0]
        spin = str(row.get("spin", "sum")).strip() or "sum"
        if not column or not elements or not orbitals or pd.isna(emin) or pd.isna(emax):
            continue
        integrations.append(
            {
                "column": column,
                "elements": elements,
                "orbitals": orbitals,
                "energy_window": [float(emin), float(emax)],
                "spin": spin,
            }
        )
    return integrations


def _gas_reference_mapping_from_table(table: pd.DataFrame) -> dict[str, float]:
    mapping: dict[str, float] = {}
    if table is None or table.empty:
        return mapping
    for row in table.to_dict("records"):
        species = str(row.get("species", "")).strip()
        value = pd.to_numeric(pd.Series([row.get("energy_eV")]), errors="coerce").iloc[0]
        if species and pd.notna(value):
            mapping[species] = float(value)
    return mapping


def _adsorption_recipes_from_table(table: pd.DataFrame) -> dict[str, dict[str, object]]:
    recipes: dict[str, dict[str, object]] = {}
    if table is None or table.empty:
        return recipes
    excluded = {"adsorbate", "basis"}
    for row in table.to_dict("records"):
        adsorbate = str(row.get("adsorbate", "")).strip()
        basis = str(row.get("basis", "C")).strip() or "C"
        if not adsorbate:
            continue
        gas_refs: dict[str, float] = {}
        for key, value in row.items():
            if key in excluded:
                continue
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric) and float(numeric) != 0.0:
                gas_refs[str(key)] = float(numeric)
        if gas_refs:
            recipes[adsorbate] = {"basis": basis, "gas_refs": gas_refs}
    return recipes
