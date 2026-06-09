from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from onepiece.sources import restore_source_descriptors, source_descriptors


def build_project_payload(
    *,
    state: dict[str, Any],
    source_rows: int,
    active_rows: int,
    control_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "project_version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "source_rows": int(source_rows),
        "active_rows": int(active_rows),
        "query": deepcopy(control_state),
        "workflow": deepcopy(state.get("onepiece_studio_workflow_operations", [])),
        "row_states": deepcopy(state.get("onepiece_studio_control_status", {})),
        "workbook_edits": deepcopy(state.get("onepiece_studio_cell_edits", {})),
        "sources": source_descriptors(state),
        "saved_views": deepcopy(state.get("onepiece_studio_saved_views", {})),
        "audit_log": deepcopy(state.get("onepiece_studio_audit_log", [])),
    }


def restore_project_payload(state: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    sources = payload.get("sources", [])
    messages = restore_source_descriptors(state, sources)
    control_state = payload.get("query", {})
    for key, value in control_state.items():
        state[key] = deepcopy(value)
    state["onepiece_studio_workflow_operations"] = deepcopy(payload.get("workflow", []))
    state["onepiece_studio_control_status"] = deepcopy(payload.get("row_states", {}))
    state["onepiece_studio_cell_edits"] = deepcopy(payload.get("workbook_edits", {}))
    state["onepiece_studio_saved_views"] = deepcopy(payload.get("saved_views", {}))
    state["onepiece_studio_audit_log"] = deepcopy(payload.get("audit_log", []))
    return messages
