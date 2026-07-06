from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from onepiece.adsorption.formulas import DEFAULT_REFERENCE_DESCENDANT_MARKERS

if TYPE_CHECKING:
    from onepiece.phase_diagrams import PhaseCandidateValidationRules


@dataclass(frozen=True, slots=True)
class ProjectWorkflowConfig:
    """Reusable configuration for project-specific dataframe workflows.

    The package should own generic mechanics, while scripts provide chemical
    system choices such as adsorbate labels, gas references, file column names,
    and validation thresholds.
    """

    adsorbates: frozenset[str] = field(default_factory=frozenset)
    adsorbate_tokens: tuple[str, ...] = ()
    adsorbate_elements: tuple[str, ...] = ("C", "H")
    gas_reference_by_adsorbate: Mapping[str, str] = field(default_factory=dict)
    systems: tuple[str, ...] = ()
    facets: tuple[str, ...] = ()
    structure_column: str = "CONTCAR"
    structure_fallback_columns: tuple[str, ...] = ("struc", "structure", "atoms")
    contcar_path_column: str = "contcar_path"
    acf_path_column: str = "acf_path"
    calculation_path_column: str = "Path"
    reference_path_columns: tuple[str, ...] = ("relative_path", "Path")
    reference_descendant_markers: tuple[str, ...] = DEFAULT_REFERENCE_DESCENDANT_MARKERS
    exclude_reference_name_substrings: tuple[str, ...] = ()
    exclude_reference_name_patterns: tuple[str, ...] = ()
    mapping_mode: str = "surface_same_index_adsorbate_last_element_reference"
    charge_coordinate_max_delta_A: float = 0.05
    charge_balance_residual_max_abs_e: float = 1.0
    phase_candidate_validation_rules: Mapping[str, Any] = field(default_factory=dict)
    output_names: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ProjectWorkflowConfig:
        """Build a config from a plain mapping, such as parsed YAML/JSON."""
        phase_rules = values.get("phase_candidate_validation_rules", values.get("phase_candidate_rules"))
        return cls(
            adsorbates=frozenset(_tuple(values.get("adsorbates", ()))),
            adsorbate_tokens=_tuple(values.get("adsorbate_tokens", ())),
            adsorbate_elements=_tuple(values.get("adsorbate_elements", ("C", "H"))),
            gas_reference_by_adsorbate=_string_mapping(values.get("gas_reference_by_adsorbate")),
            systems=_tuple(values.get("systems", ())),
            facets=_tuple(values.get("facets", ())),
            structure_column=str(values.get("structure_column", "CONTCAR")),
            structure_fallback_columns=_tuple(
                values.get("structure_fallback_columns", ("struc", "structure", "atoms"))
            ),
            contcar_path_column=str(values.get("contcar_path_column", "contcar_path")),
            acf_path_column=str(values.get("acf_path_column", "acf_path")),
            calculation_path_column=str(values.get("calculation_path_column", "Path")),
            reference_path_columns=_tuple(values.get("reference_path_columns", ("relative_path", "Path"))),
            reference_descendant_markers=_tuple(
                values.get("reference_descendant_markers", DEFAULT_REFERENCE_DESCENDANT_MARKERS)
            ),
            exclude_reference_name_substrings=_tuple(values.get("exclude_reference_name_substrings", ())),
            exclude_reference_name_patterns=_tuple(values.get("exclude_reference_name_patterns", ())),
            mapping_mode=str(
                values.get("mapping_mode", "surface_same_index_adsorbate_last_element_reference")
            ),
            charge_coordinate_max_delta_A=float(values.get("charge_coordinate_max_delta_A", 0.05)),
            charge_balance_residual_max_abs_e=float(
                values.get("charge_balance_residual_max_abs_e", 1.0)
            ),
            phase_candidate_validation_rules=_plain_mapping(phase_rules),
            output_names=_string_mapping(values.get("output_names")),
        )

    def active_adsorbate_tokens(self) -> tuple[str, ...]:
        """Return configured adsorbate tokens, falling back to ``adsorbates``."""
        if self.adsorbate_tokens:
            return self.adsorbate_tokens
        return tuple(sorted(self.adsorbates, key=lambda value: (len(value), value), reverse=True))

    def reference_assignment_kwargs(self) -> dict[str, object]:
        """Return keyword arguments for ``assign_surface_references``."""
        kwargs: dict[str, object] = {
            "adsorbate_elements": self.adsorbate_elements,
            "reference_descendant_markers": self.reference_descendant_markers,
            "reference_path_columns": self.reference_path_columns,
            "exclude_reference_name_substrings": self.exclude_reference_name_substrings,
            "exclude_reference_name_patterns": self.exclude_reference_name_patterns,
        }
        active_tokens = self.active_adsorbate_tokens()
        if active_tokens:
            kwargs["adsorbate_tokens"] = active_tokens
        return kwargs

    def phase_candidate_rules(self) -> PhaseCandidateValidationRules:
        """Build phase-candidate validation rules from this config."""
        from onepiece.phase_diagrams import PhaseCandidateValidationRules

        if isinstance(self.phase_candidate_validation_rules, PhaseCandidateValidationRules):
            return self.phase_candidate_validation_rules
        return PhaseCandidateValidationRules.from_mapping(self.phase_candidate_validation_rules)

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for manifests."""
        return {
            "adsorbates": sorted(self.adsorbates),
            "adsorbate_tokens": list(self.adsorbate_tokens),
            "adsorbate_elements": list(self.adsorbate_elements),
            "gas_reference_by_adsorbate": dict(self.gas_reference_by_adsorbate),
            "systems": list(self.systems),
            "facets": list(self.facets),
            "structure_column": self.structure_column,
            "structure_fallback_columns": list(self.structure_fallback_columns),
            "contcar_path_column": self.contcar_path_column,
            "acf_path_column": self.acf_path_column,
            "calculation_path_column": self.calculation_path_column,
            "reference_path_columns": list(self.reference_path_columns),
            "reference_descendant_markers": list(self.reference_descendant_markers),
            "exclude_reference_name_substrings": list(self.exclude_reference_name_substrings),
            "exclude_reference_name_patterns": list(self.exclude_reference_name_patterns),
            "mapping_mode": self.mapping_mode,
            "charge_coordinate_max_delta_A": self.charge_coordinate_max_delta_A,
            "charge_balance_residual_max_abs_e": self.charge_balance_residual_max_abs_e,
            "phase_candidate_validation_rules": _plain_mapping(self.phase_candidate_validation_rules),
            "output_names": dict(self.output_names),
        }


def coerce_project_workflow_config(
    config: ProjectWorkflowConfig | Mapping[str, Any] | None,
) -> ProjectWorkflowConfig:
    """Return a workflow config from ``None``, an existing config, or a mapping."""
    if config is None:
        return ProjectWorkflowConfig()
    if isinstance(config, ProjectWorkflowConfig):
        return config
    return ProjectWorkflowConfig.from_mapping(config)


def _tuple(values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    if isinstance(values, Sequence):
        return tuple(str(value) for value in values)
    return ()


def _string_mapping(values: object) -> dict[str, str]:
    if values is None:
        return {}
    try:
        return {str(key): str(value) for key, value in dict(values).items()}
    except (TypeError, ValueError):
        return {}


def _plain_mapping(values: object) -> dict[str, Any]:
    if values is None:
        return {}
    if hasattr(values, "to_mapping"):
        values = values.to_mapping()
    if not isinstance(values, Mapping):
        try:
            values = dict(values)
        except (TypeError, ValueError):
            return {}
    return {str(key): _plain_value(value) for key, value in values.items()}


def _plain_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, Sequence):
        return [_plain_value(item) for item in value]
    return value
