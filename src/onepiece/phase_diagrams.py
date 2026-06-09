from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sympy as sp

DEFAULT_PHASE_SYMBOLS = {
    "x": sp.Symbol("x", positive=True),
    "T": sp.Symbol("T", real=True),
    "kb": sp.Symbol("kb", positive=True),
    "pH2": sp.Symbol("pH2", positive=True),
    "pH2O": sp.Symbol("pH2O", positive=True),
    "pCO2": sp.Symbol("pCO2", positive=True),
    "pCO": sp.Symbol("pCO", positive=True),
    "log": sp.log,
    "ln": sp.log,
}


@dataclass(frozen=True)
class PhaseScanResult:
    stable_frame: pd.DataFrame
    scan_table: pd.DataFrame


@dataclass(frozen=True)
class PhaseFieldResult:
    energy_grid: np.ndarray
    stable_index: np.ndarray
    stable_energy: np.ndarray


@dataclass(frozen=True)
class NamedPhaseFieldResult:
    frame: pd.DataFrame
    expression_column: str
    x_symbol: str
    x_values: np.ndarray
    t_symbol: str
    t_values: np.ndarray
    field: PhaseFieldResult
    stable_summary: pd.DataFrame


@dataclass(frozen=True)
class GroupedPhaseDiagramResult:
    groups: dict[str, NamedPhaseFieldResult]


def default_phase_variables(**overrides: object) -> dict[str, object]:
    """Return a neutral default environment for symbolic phase expressions."""
    variables: dict[str, object] = {
        "x": 1.0,
        "pH2": 1.0,
        "pH2O": 1.0,
        "pCO2": 1.0,
        "pCO": 1.0,
        "T": 500.0,
        "kb": 8.617333262145e-5,
    }
    variables.update(overrides)
    return variables


def phase_symbol_locals(extra: Mapping[str, object] | None = None) -> dict[str, object]:
    locals_dict = dict(DEFAULT_PHASE_SYMBOLS)
    if extra:
        locals_dict.update(extra)
    return locals_dict


def to_sympy_expression(value: object, *, locals_dict: Mapping[str, object] | None = None) -> sp.Basic:
    active_locals = phase_symbol_locals(locals_dict)
    if isinstance(value, sp.Basic):
        return value
    if value is None or pd.isna(value):
        return sp.nan
    if isinstance(value, int | float | np.integer | np.floating):
        return sp.Float(value)
    text = str(value).strip()
    if not text:
        return sp.nan
    return sp.sympify(text, locals=active_locals)


def substitute_variables(
    formula: object,
    *,
    variables: Mapping[str, object] | None,
    locals_dict: Mapping[str, object] | None = None,
) -> object:
    if variables is None:
        raise ValueError("variables must not be None.")
    expr = to_sympy_expression(formula, locals_dict=locals_dict)
    if expr is sp.nan:
        return expr
    try:
        active_locals = phase_symbol_locals(locals_dict)
        for key, value in variables.items():
            key_name = str(key)
            expr = expr.subs(sp.Symbol(key_name), value)
            expr = expr.subs(active_locals.get(key_name, sp.Symbol(key_name)), value)
        return expr
    except Exception:
        return expr


def evaluate_expression(
    expression: object,
    *,
    variable_symbol: str = "x",
    variable_values: Sequence[float] | np.ndarray,
    variables: Mapping[str, object] | None = None,
    locals_dict: Mapping[str, object] | None = None,
) -> np.ndarray:
    values = np.asarray(variable_values, dtype=float)
    expr = to_sympy_expression(expression, locals_dict=locals_dict)
    if variables:
        expr = substitute_variables(expr, variables=variables, locals_dict=locals_dict)
    if expr is sp.nan:
        return np.full(values.shape, np.nan, dtype=float)
    if isinstance(expr, int | float | np.integer | np.floating | sp.Float):
        return np.full(values.shape, float(expr), dtype=float)
    active_locals = phase_symbol_locals(locals_dict)
    symbol = active_locals.get(variable_symbol, sp.Symbol(variable_symbol, positive=True))
    func = sp.lambdify(symbol, expr, modules=["numpy"])
    evaluated = func(values)
    if np.isscalar(evaluated):
        return np.full(values.shape, float(evaluated), dtype=float)
    return np.asarray(evaluated, dtype=float)


def stable_phase_scan(
    frame: pd.DataFrame,
    *,
    expression_column: str,
    variable_symbol: str = "x",
    variable_values: Sequence[float] | np.ndarray | None = None,
    variables: Mapping[str, object] | None = None,
    locals_dict: Mapping[str, object] | None = None,
    sort_by: str | None = None,
) -> PhaseScanResult:
    if variable_values is None:
        variable_values = np.logspace(-10, 10, 1000)
    ordered = frame.copy()
    if sort_by and sort_by in ordered.columns:
        ordered = ordered.sort_values(sort_by).copy()

    x_values = np.asarray(variable_values, dtype=float)
    scan = pd.DataFrame({variable_symbol: x_values})
    energy_columns: list[str] = []
    for index, row in ordered.iterrows():
        column_key = str(index)
        energy_columns.append(column_key)
        scan[column_key] = evaluate_expression(
            row.get(expression_column),
            variable_symbol=variable_symbol,
            variable_values=x_values,
            variables=variables,
            locals_dict=locals_dict,
        )

    scan["minimum"] = scan[energy_columns].min(axis=1)
    scan["phase_minimum"] = scan[energy_columns].idxmin(axis=1)
    stable_indices = [ordered.index[int(idx)] if str(idx).isdigit() else idx for idx in set(scan["phase_minimum"])]
    stable = ordered.loc[stable_indices].copy()
    if sort_by and sort_by in stable.columns:
        stable = stable.sort_values(sort_by).copy()
    return PhaseScanResult(stable_frame=stable, scan_table=scan)


def estimate_phase_scan_slopes(
    frame: pd.DataFrame,
    *,
    expression_column: str,
    temperature: float = 500.0,
    variable_symbol: str = "x",
    locals_dict: Mapping[str, object] | None = None,
) -> pd.Series:
    """Notebook-style symbolic slope estimate on a logarithmic x axis."""
    active_locals = phase_symbol_locals(locals_dict)
    x_symbol = active_locals.get(variable_symbol, sp.Symbol(variable_symbol, positive=True))
    ex = sp.exp(x_symbol)
    slopes: dict[object, object] = {}
    for index, row in frame.iterrows():
        expr = to_sympy_expression(row.get(expression_column), locals_dict=locals_dict)
        if expr is sp.nan:
            slopes[index] = sp.nan
            continue
        try:
            shifted = substitute_variables(expr, variables={"T": temperature}, locals_dict=locals_dict).subs(x_symbol, ex)
            slopes[index] = sp.simplify(sp.diff(shifted, x_symbol))
        except Exception:
            slopes[index] = sp.nan
    return pd.Series(slopes, name="slope")


def solve_phase_boundaries(
    stable_frame: pd.DataFrame,
    *,
    expression_column: str,
    solve_symbol: str = "x",
    sort_by: str | None = None,
    locals_dict: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    ordered = stable_frame.copy()
    if sort_by and sort_by in ordered.columns:
        ordered = ordered.sort_values(sort_by).copy()
    active_locals = phase_symbol_locals(locals_dict)
    symbol = active_locals.get(solve_symbol, sp.Symbol(solve_symbol, positive=True))

    rows: list[dict[str, object]] = []
    previous: pd.Series | None = None
    for _, current in ordered.iterrows():
        if previous is None:
            previous = current
            continue
        left_expr = to_sympy_expression(previous.get(expression_column), locals_dict=locals_dict)
        right_expr = to_sympy_expression(current.get(expression_column), locals_dict=locals_dict)
        delta = sp.simplify(left_expr - right_expr)
        solutions = sp.solve(delta, symbol)
        rows.append(
            {
                "left_name": previous.get("Name", previous.name),
                "right_name": current.get("Name", current.name),
                "left_index": previous.name,
                "right_index": current.name,
                "delta_expression": delta,
                "solutions": solutions,
            }
        )
        previous = current
    return pd.DataFrame(rows)


def build_surface_free_energy_expressions(
    frame: pd.DataFrame,
    *,
    base_column: str = "form_G",
    area_column: str = "Area",
    correction_map: Mapping[str, tuple[str, object]] | None = None,
    output_column: str = "phase_expression_per_area",
    normalize_by_area: bool = True,
    locals_dict: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Build generic symbolic per-area phase expressions.

    `correction_map` maps a delta column to a pair of:
    - stored reference chemical-potential column name
    - target symbolic expression for the new chemical potential

    Example:
        {"delta_Ga": ("mu_Ga", mu_ga_expr)}
    """
    correction_map = correction_map or {}
    df = frame.copy()
    expressions: list[sp.Basic] = []

    for _, row in df.iterrows():
        base_expr = to_sympy_expression(row.get(base_column), locals_dict=locals_dict)
        if base_expr is sp.nan:
            expressions.append(sp.nan)
            continue
        expr = base_expr
        for delta_column, (reference_mu_column, target_mu_expr) in correction_map.items():
            delta_value = pd.to_numeric(pd.Series([row.get(delta_column)]), errors="coerce").fillna(0.0).iloc[0]
            reference_mu = pd.to_numeric(pd.Series([row.get(reference_mu_column)]), errors="coerce").fillna(0.0).iloc[0]
            if float(delta_value) == 0.0:
                continue
            expr = expr + float(delta_value) * (float(reference_mu) - to_sympy_expression(target_mu_expr, locals_dict=locals_dict))

        area_value = pd.to_numeric(pd.Series([row.get(area_column)]), errors="coerce").iloc[0]
        if normalize_by_area and pd.notna(area_value) and float(area_value) != 0.0:
            expr = sp.simplify(expr / float(area_value))
        expressions.append(expr)

    df[output_column] = expressions
    return df


def build_corrected_phase_expressions(
    frame: pd.DataFrame,
    *,
    energy_column: str = "form_G",
    normalized_energy_column: str | None = "form_G_per_Area",
    area_column: str = "Area",
    correction_map: Mapping[str, tuple[str, object]] | None = None,
    output_column: str = "phase_expression",
    divide_by_area: bool | None = None,
    locals_dict: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Build generic corrected symbolic phase expressions.

    Parameters
    ----------
    frame
        Input dataframe containing energy and reference columns.
    energy_column
        Base total free-energy-like column, for example ``form_G``.
    normalized_energy_column
        Optional already-normalized column, for example ``form_G_per_Area``.
        When ``divide_by_area`` is not explicitly set, its presence disables
        area division because the values are already normalized.
    area_column
        Column used for area normalization when needed.
    correction_map
        Maps a delta column to ``(reference_mu_column, target_mu_expression)``.
        Example: ``{"delta_Ga": ("mu_Ga", "(Ga2O3 - 3*H2O + 3*H2)/2")}``
    output_column
        Name of the new symbolic expression column.
    divide_by_area
        If ``None``, infer from ``normalized_energy_column`` availability.
    """
    if divide_by_area is None:
        divide_by_area = normalized_energy_column is None or normalized_energy_column not in frame.columns

    source_column = energy_column
    if not divide_by_area and normalized_energy_column and normalized_energy_column in frame.columns:
        source_column = normalized_energy_column

    corrected = build_surface_free_energy_expressions(
        frame,
        base_column=source_column,
        area_column=area_column,
        correction_map=correction_map,
        output_column=output_column,
        normalize_by_area=divide_by_area,
        locals_dict=locals_dict,
    )
    return corrected


def summarize_phase_field_stability(
    frame: pd.DataFrame,
    *,
    stable_index: np.ndarray,
    stable_energy: np.ndarray,
    x_values: Sequence[float] | np.ndarray,
    t_values: Sequence[float] | np.ndarray,
    group_label: str | None = None,
    x_label: str = "x",
    t_label: str = "T",
    name_column: str = "Name",
) -> pd.DataFrame:
    xx, tt = np.meshgrid(np.asarray(x_values, dtype=float), np.asarray(t_values, dtype=float), indexing="xy")
    records: list[dict[str, object]] = []
    for phase_id in np.unique(stable_index):
        mask = stable_index == phase_id
        row = frame.iloc[int(phase_id)]
        records.append(
            {
                "group": group_label,
                "phase_id": int(phase_id),
                "index": row.name,
                "Name": row.get(name_column, row.name),
                "Formula": row.get("Formula", ""),
                "stable_fraction": float(mask.mean()),
                "stable_percent": float(mask.mean() * 100.0),
                f"{t_label}_min": float(tt[mask].min()),
                f"{t_label}_max": float(tt[mask].max()),
                f"{x_label}_min": float(xx[mask].min()),
                f"{x_label}_max": float(xx[mask].max()),
                "min_energy": float(stable_energy[mask].min()),
            }
        )
    return pd.DataFrame(records).sort_values("stable_fraction", ascending=False).reset_index(drop=True)


def build_phase_field_grid(
    frame: pd.DataFrame,
    *,
    expression_column: str,
    x_symbol: str,
    x_values: Sequence[float] | np.ndarray,
    t_symbol: str = "T",
    t_values: Sequence[float] | np.ndarray = (),
    variables: Mapping[str, object] | None = None,
    locals_dict: Mapping[str, object] | None = None,
) -> PhaseFieldResult:
    x_grid = np.asarray(x_values, dtype=float)
    t_grid = np.asarray(t_values, dtype=float)
    xx, tt = np.meshgrid(x_grid, t_grid, indexing="xy")

    energy_surfaces: list[np.ndarray] = []
    active_locals = phase_symbol_locals(locals_dict)
    x_sym = active_locals.get(x_symbol, sp.Symbol(x_symbol, positive=True))
    t_sym = active_locals.get(t_symbol, sp.Symbol(t_symbol, real=True))
    replacement_symbols = {
        active_locals.get(str(key), sp.Symbol(str(key))): value
        for key, value in (variables or {}).items()
        if str(key) not in {x_symbol, t_symbol}
    }

    for _, row in frame.iterrows():
        expr = to_sympy_expression(row.get(expression_column), locals_dict=locals_dict)
        if replacement_symbols:
            expr = expr.subs(replacement_symbols)
        func = sp.lambdify((t_sym, x_sym), expr, modules=["numpy"])
        values = func(tt, xx)
        if np.isscalar(values):
            values = np.full(xx.shape, float(values), dtype=float)
        energy_surfaces.append(np.asarray(values, dtype=float))

    energy_grid = np.stack(energy_surfaces, axis=0)
    stable_index = np.nanargmin(energy_grid, axis=0)
    stable_energy = np.nanmin(energy_grid, axis=0)
    return PhaseFieldResult(
        energy_grid=energy_grid,
        stable_index=stable_index,
        stable_energy=stable_energy,
    )


def build_surface_phase_diagram(
    frame: pd.DataFrame,
    *,
    correction_map: Mapping[str, tuple[str, object]] | None = None,
    energy_column: str = "form_G",
    normalized_energy_column: str | None = "form_G_per_Area",
    area_column: str = "Area",
    expression_column: str = "phase_expression",
    x_symbol: str = "x",
    x_values: Sequence[float] | np.ndarray | None = None,
    t_symbol: str = "T",
    t_values: Sequence[float] | np.ndarray | None = None,
    variables: Mapping[str, object] | None = None,
    locals_dict: Mapping[str, object] | None = None,
    name_column: str = "Name",
) -> NamedPhaseFieldResult:
    if x_values is None:
        x_values = np.logspace(-10, 10, 1000)
    if t_values is None:
        t_values = np.array([default_phase_variables()["T"]], dtype=float)

    corrected = build_corrected_phase_expressions(
        frame,
        energy_column=energy_column,
        normalized_energy_column=normalized_energy_column,
        area_column=area_column,
        correction_map=correction_map,
        output_column=expression_column,
        locals_dict=locals_dict,
    )
    field = build_phase_field_grid(
        corrected,
        expression_column=expression_column,
        x_symbol=x_symbol,
        x_values=x_values,
        t_symbol=t_symbol,
        t_values=t_values,
        variables=variables,
        locals_dict=locals_dict,
    )
    summary = summarize_phase_field_stability(
        corrected,
        stable_index=field.stable_index,
        stable_energy=field.stable_energy,
        x_values=x_values,
        t_values=t_values,
        x_label=x_symbol,
        t_label=t_symbol,
        name_column=name_column,
    )
    return NamedPhaseFieldResult(
        frame=corrected,
        expression_column=expression_column,
        x_symbol=x_symbol,
        x_values=np.asarray(x_values, dtype=float),
        t_symbol=t_symbol,
        t_values=np.asarray(t_values, dtype=float),
        field=field,
        stable_summary=summary,
    )


def build_grouped_surface_phase_diagrams(
    frame: pd.DataFrame,
    *,
    group_column: str,
    correction_map: Mapping[str, tuple[str, object]] | None = None,
    energy_column: str = "form_G",
    normalized_energy_column: str | None = "form_G_per_Area",
    area_column: str = "Area",
    expression_column: str = "phase_expression",
    x_symbol: str = "x",
    x_values: Sequence[float] | np.ndarray | None = None,
    t_symbol: str = "T",
    t_values: Sequence[float] | np.ndarray | None = None,
    variables: Mapping[str, object] | None = None,
    locals_dict: Mapping[str, object] | None = None,
    name_column: str = "Name",
) -> GroupedPhaseDiagramResult:
    groups: dict[str, NamedPhaseFieldResult] = {}
    for group_value, subset in frame.groupby(group_column, dropna=False):
        label = str(group_value)
        result = build_surface_phase_diagram(
            subset.copy(),
            correction_map=correction_map,
            energy_column=energy_column,
            normalized_energy_column=normalized_energy_column,
            area_column=area_column,
            expression_column=expression_column,
            x_symbol=x_symbol,
            x_values=x_values,
            t_symbol=t_symbol,
            t_values=t_values,
            variables=variables,
            locals_dict=locals_dict,
            name_column=name_column,
        )
        stable_summary = result.stable_summary.copy()
        stable_summary["group"] = label
        groups[label] = NamedPhaseFieldResult(
            frame=result.frame,
            expression_column=result.expression_column,
            x_symbol=result.x_symbol,
            x_values=result.x_values,
            t_symbol=result.t_symbol,
            t_values=result.t_values,
            field=result.field,
            stable_summary=stable_summary,
        )
    return GroupedPhaseDiagramResult(groups=groups)
