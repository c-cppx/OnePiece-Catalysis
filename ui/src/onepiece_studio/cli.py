from __future__ import annotations

import argparse
import importlib
import platform
import subprocess  # nosec B404
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path

from onepiece import (
    bundled_catalysis_hub_dataset,
    format_self_test_result,
    run_catalysis_hub_self_test,
)


def _studio_version() -> str:
    try:
        return _package_version("onepiece-studio")
    except PackageNotFoundError:
        return "unknown (package not installed)"


def main(argv: list[str] | None = None) -> int:
    prog_name = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "onepiece-studio"
    parser = argparse.ArgumentParser(prog=prog_name)
    parser.add_argument("--version", action="version", version=f"onepiece-studio {_studio_version()}")
    subparsers = parser.add_subparsers(dest="command", required=False)
    subparsers.add_parser("demo", help="Run the OnePiece Studio Streamlit demo.")
    subparsers.add_parser(
        "tutorial",
        help="Open the bundled Catalysis-Hub tutorial dataset in OnePiece Studio.",
    )
    hdf_parser = subparsers.add_parser("hdf", help="Run OnePiece Studio for a pandas HDF file.")
    hdf_parser.add_argument("path", help="Path to a pandas HDF file.")
    hdf_parser.add_argument("--key", default="df", help="HDF key, defaults to 'df'.")
    hdf_parser.add_argument("--title", default=None, help="Optional UI title.")
    qa_parser = subparsers.add_parser("qa", help="Run bundled package self-tests.")
    qa_parser.add_argument(
        "--dataset",
        default=None,
        help="Optional path to a Catalysis-Hub-style HDF file. Defaults to the bundled reference dataset.",
    )
    subparsers.add_parser("doctor", help="Check whether this environment is ready for OnePiece Studio.")
    args = parser.parse_args(argv)

    if args.command == "qa":
        result = run_catalysis_hub_self_test(args.dataset)
        print(format_self_test_result(result))
        return 0 if result.passed else 1

    if args.command == "doctor":
        report = _installation_report()
        print(report)
        return 0 if "[FAIL]" not in report else 1

    if args.command in {None, "demo", "tutorial", "hdf"}:
        return _run_streamlit_app(args)

    return 0


def _run_streamlit_app(args: argparse.Namespace) -> int:
    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
    ]
    if args.command == "demo":
        command.append("--demo")
    elif args.command == "tutorial":
        command.extend(
            [
                "--hdf",
                str(bundled_catalysis_hub_dataset()),
                "--key",
                "df",
                "--title",
                "OnePiece Studio Tutorial Dataset",
            ]
        )
    elif args.command == "hdf":
        command.extend(["--hdf", args.path, "--key", args.key])
        if args.title:
            command.extend(["--title", args.title])
    # Launches a fixed local Streamlit command assembled by the CLI itself.
    return subprocess.call(command)  # nosec B603


def _installation_report() -> str:
    lines = [
        "[INFO] OnePiece Studio environment report",
        f"- python: {sys.version.split()[0]}",
        f"- platform: {platform.platform()}",
        f"- executable: {sys.executable}",
    ]
    failures = 0
    for module_name in ["streamlit", "pandas", "tables", "sympy", "ase", "plotly"]:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            lines.append(f"[FAIL] import {module_name}: {exc}")
            failures += 1
        else:
            version = getattr(module, "__version__", "unknown")
            lines.append(f"[PASS] import {module_name}: {version}")

    dataset = bundled_catalysis_hub_dataset()
    if dataset.exists():
        lines.append(f"[PASS] bundled dataset: {dataset}")
    else:
        lines.append(f"[FAIL] bundled dataset missing: {dataset}")
        failures += 1

    lines.append(
        "[PASS] ready to launch UI"
        if failures == 0
        else "[WARN] fix the failed checks above before using OnePiece Studio"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
