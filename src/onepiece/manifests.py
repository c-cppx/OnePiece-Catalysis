from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True, slots=True)
class TableArtifact:
    key: str
    path: str
    rows: int | None = None
    index_columns: tuple[str, ...] = ()
    source: str = ""
    status: str = "ready"
    notes: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": self.path,
            "rows": self.rows,
            "index_columns": list(self.index_columns),
            "source": self.source,
            "status": self.status,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class AnalysisManifest:
    artifacts: tuple[TableArtifact, ...]
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "artifacts": {artifact.key: artifact.to_mapping() for artifact in self.artifacts},
            "rows": {artifact.key: artifact.rows for artifact in self.artifacts},
            "config": dict(self.config),
            "validation": dict(self.validation),
        }


def table_artifact(
    key: str,
    path: Path | str,
    *,
    dataframe: pd.DataFrame | None = None,
    index_columns: tuple[str, ...] = (),
    source: str = "",
    status: str = "ready",
    notes: tuple[str, ...] = (),
) -> TableArtifact:
    rows = int(len(dataframe)) if dataframe is not None else None
    return TableArtifact(
        key=str(key),
        path=str(path),
        rows=rows,
        index_columns=tuple(index_columns),
        source=str(source),
        status=str(status),
        notes=tuple(str(note) for note in notes),
    )


def legacy_manifest_payload(
    outputs: Mapping[str, Path | str],
    row_counts: Mapping[str, int],
    *,
    config: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a richer manifest while preserving the previous flat keys."""
    artifacts = tuple(
        TableArtifact(key=key, path=str(path), rows=int(row_counts[key]) if key in row_counts else None)
        for key, path in outputs.items()
    )
    payload = AnalysisManifest(
        artifacts=artifacts,
        config=config or {},
        validation=validation or {},
    ).to_mapping()
    payload.update({key: str(path) for key, path in outputs.items()})
    payload["rows"] = {key: int(value) for key, value in row_counts.items()}
    if extra:
        payload.update(dict(extra))
    return payload


def workflow_stage_manifest_payload(
    *,
    stage_id: str,
    files: Mapping[str, Path | str] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    audit_ledger: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    source_inputs: Mapping[str, Path | str] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a standard machine-readable manifest for one workflow stage."""
    active_files = {str(key): Path(path) for key, path in (files or {}).items()}
    active_tables = dict(tables or {})
    artifacts = tuple(
        table_artifact(
            key,
            path,
            dataframe=active_tables.get(key),
            source="workflow_stage",
            notes=tuple(str(note) for note in notes),
        )
        for key, path in active_files.items()
    )
    payload = AnalysisManifest(
        artifacts=artifacts,
        config=_plain_mapping(config or {}),
        validation=_plain_mapping(validation or {}),
    ).to_mapping()
    payload["schema_version"] = "onepiece.workflow_stage_manifest.v1"
    payload["stage_id"] = str(stage_id)
    payload["files"] = {key: str(path) for key, path in active_files.items()}
    payload["table_rows"] = {
        str(key): int(len(frame))
        for key, frame in active_tables.items()
        if isinstance(frame, pd.DataFrame)
    }
    payload["audit"] = _audit_summary(audit_ledger)
    payload["source_inputs"] = {
        str(key): str(path)
        for key, path in (source_inputs or {}).items()
    }
    payload["notes"] = [str(note) for note in notes]
    return payload


def write_workflow_stage_manifest(
    path: Path | str,
    *,
    stage_id: str,
    files: Mapping[str, Path | str] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    audit_ledger: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    source_inputs: Mapping[str, Path | str] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Write a standard workflow-stage manifest and return its payload."""
    payload = workflow_stage_manifest_payload(
        stage_id=stage_id,
        files=files,
        tables=tables,
        audit_ledger=audit_ledger,
        config=config,
        validation=validation,
        source_inputs=source_inputs,
        notes=notes,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return payload


def _audit_summary(audit_ledger: pd.DataFrame | None) -> dict[str, Any]:
    if audit_ledger is None or audit_ledger.empty:
        return {"rows": 0, "status_counts": {}, "drop_reason_counts": {}}
    status_counts = (
        audit_ledger["status"].astype(str).value_counts(dropna=False).sort_index().to_dict()
        if "status" in audit_ledger.columns
        else {}
    )
    reason_column = "reason_code" if "reason_code" in audit_ledger.columns else "drop_reasons"
    drop_reason_counts = (
        audit_ledger[reason_column].astype(str).value_counts(dropna=False).sort_index().to_dict()
        if reason_column in audit_ledger.columns
        else {}
    )
    return {
        "rows": int(len(audit_ledger)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "drop_reason_counts": {str(key): int(value) for key, value in drop_reason_counts.items()},
    }


def _plain_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(value) for key, value in values.items()}


def _plain_value(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, str) or value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Sequence):
        return [_plain_value(item) for item in value]
    if hasattr(value, "to_mapping"):
        return _plain_value(value.to_mapping())
    return str(value)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)
