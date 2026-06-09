from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ColumnConfig:
    label: str | None = None
    width: str | None = None
    hidden: bool = False
    pinned: bool = False
    searchable: bool = True
    filterable: bool = True
    format: str | None = None


@dataclass(frozen=True, slots=True)
class OnePieceStudioConfig:
    title: str = "OnePiece Studio Database"
    primary_key: str | None = None
    image_columns: list[str] = field(default_factory=list)
    structure_columns: list[str] = field(default_factory=list)
    asset_root: Path | None = None
    default_page_size: int = 50
    searchable_columns: list[str] | None = None
    columns: dict[str, ColumnConfig] = field(default_factory=dict)
    metric_columns: list[str] = field(default_factory=list)

    def normalized_asset_root(self) -> Path | None:
        return self.asset_root.expanduser().resolve() if self.asset_root else None
