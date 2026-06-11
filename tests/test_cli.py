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


def test_status_tags_colored_only_on_tty(monkeypatch) -> None:
    report = "[INFO] header\n[PASS] check one\n[FAIL] check two\n[WARN] hint"

    monkeypatch.setattr(cli, "_use_color", lambda: True)
    colored = cli._colorize_status_tags(report)
    assert "\033[32m[PASS]\033[0m" in colored
    assert "\033[1;31m[FAIL]\033[0m" in colored

    monkeypatch.setattr(cli, "_use_color", lambda: False)
    assert cli._colorize_status_tags(report) == report


def test_no_color_environment_disables_color(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli._use_color() is False
