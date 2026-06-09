from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class ImageRef:
    raw: str
    display_uri: str | None
    exists: bool
    is_remote: bool


def resolve_image(value: object, *, asset_root: Path | None = None) -> ImageRef:
    raw = "" if value is None else str(value)
    if not raw or raw.lower() == "nan":
        return ImageRef(raw=raw, display_uri=None, exists=False, is_remote=False)

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        return ImageRef(raw=raw, display_uri=raw, exists=True, is_remote=True)

    path = Path(parsed.path if parsed.scheme == "file" else raw).expanduser()
    if not path.is_absolute() and asset_root is not None:
        path = asset_root / path
    resolved = path.resolve()

    return ImageRef(
        raw=raw,
        display_uri=str(resolved) if resolved.exists() else None,
        exists=resolved.exists(),
        is_remote=False,
    )
