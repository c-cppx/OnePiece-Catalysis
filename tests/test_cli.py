from __future__ import annotations

from pathlib import Path

from onepiece_studio import cli


def test_doctor_report_includes_core_checks() -> None:
    report = cli._installation_report()

    assert "[INFO] OnePiece Studio environment report" in report
    assert "python:" in report
    assert "bundled dataset:" in report


def test_main_tutorial_launches_bundled_dataset(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_call(command: list[str]) -> int:
        captured["command"] = command
        return 0

    dataset = tmp_path / "tutorial.hdf"
    dataset.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(cli, "bundled_catalysis_hub_dataset", lambda: dataset)
    monkeypatch.setattr(cli.subprocess, "call", _fake_call)

    code = cli.main(["tutorial"])

    assert code == 0
    assert captured["command"][-6:] == [
        "--hdf",
        str(dataset),
        "--key",
        "df",
        "--title",
        "OnePiece Studio Tutorial Dataset",
    ]
