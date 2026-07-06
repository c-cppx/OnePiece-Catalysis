from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from onepiece.adsorption import assign_surface_references
from onepiece.atom_tables import ensure_structure_columns
from onepiece.classification import add_file_status_columns, add_workflow_classification
from onepiece.workflow_config import ProjectWorkflowConfig


@dataclass(slots=True)
class CalculationFrame:
    """Small chainable wrapper for calculation dataframe enrichment."""

    dataframe: pd.DataFrame
    config: ProjectWorkflowConfig = field(default_factory=ProjectWorkflowConfig)
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_pickle(
        cls,
        path: str,
        *,
        config: ProjectWorkflowConfig | None = None,
    ) -> CalculationFrame:
        return cls(pd.read_pickle(path), config or ProjectWorkflowConfig())

    def copy(self) -> CalculationFrame:
        return CalculationFrame(self.dataframe.copy(), self.config, list(self.audit_log))

    def ensure_structures(self) -> CalculationFrame:
        return self._with_step(
            "ensure_structures",
            ensure_structure_columns(
                self.dataframe,
                preferred_structure_column=self.config.structure_column,
                fallback_columns=self.config.structure_fallback_columns,
                path_column=self.config.contcar_path_column,
            ),
        )

    def classify_records(self) -> CalculationFrame:
        return self._with_step(
            "classify_records",
            add_workflow_classification(self.dataframe, config=self.config),
        )

    def assign_surface_references(self) -> CalculationFrame:
        return self._with_step(
            "assign_surface_references",
            assign_surface_references(self.dataframe, **self.config.reference_assignment_kwargs()),
        )

    def clean_phase_candidates(
        self,
        *,
        system: str = "",
        phase_set: str = "",
        allowed_elements: tuple[str, ...] = (),
    ) -> CalculationFrame:
        from onepiece.phase_diagrams import clean_phase_candidates

        return self._with_step(
            "clean_phase_candidates",
            clean_phase_candidates(
                self.dataframe,
                system=system,
                phase_set=phase_set,
                allowed_elements=allowed_elements,
                rules=self.config.phase_candidate_rules(),
            ),
        )

    def add_file_status(self) -> CalculationFrame:
        return self._with_step(
            "add_file_status",
            add_file_status_columns(
                self.dataframe,
                calculation_path_column=self.config.calculation_path_column,
            ),
        )

    def _with_step(self, step: str, dataframe: pd.DataFrame) -> CalculationFrame:
        added_columns = [column for column in dataframe.columns if column not in self.dataframe.columns]
        removed_columns = [column for column in self.dataframe.columns if column not in dataframe.columns]
        self.audit_log.append(
            {
                "step": step,
                "rows_before": int(len(self.dataframe)),
                "rows_after": int(len(dataframe)),
                "added_columns": added_columns,
                "removed_columns": removed_columns,
            }
        )
        self.dataframe = dataframe
        return self
