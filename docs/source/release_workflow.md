# Release Workflow

This page collects the practical release checks for the split `onepiece` /
`onepiece-studio` packaging layout. It is meant as a lightweight release gate
for a scientific Python package family: backend tests, packaged-data QA, build
artifacts, and clean install checks.

## 1. Run The Test Suite

From the repository root:

```bash
python3 -m pip install -e .[dev,docs,release]
python3 scripts/release_check.py
```

This consolidated release check covers:

- Ruff lint for the supported rule set
- backend scientific logic
- workflow execution
- query/filter behavior
- project persistence
- package QA for the bundled Catalysis-Hub dataset
- docs build
- wheel and sdist creation
- wheel install validation

If you want to run only the narrower test suite:

```bash
PYTHONPATH=src python3 -m pytest -q
```

## 2. Run The Built-In Package QA

Even when the repo tests are green, run the installed package self-test:

```bash
onepiece-studio qa
```

This checks that the installed package can:

- find the bundled HDF file
- read it successfully
- reconstruct Catalysis-Hub adsorption energies
- match those values against stored reaction energies

## 3. Build Release Artifacts

Build the backend from the repository root:

```bash
python3 -m build
```

Build the UI distribution from the `ui/` subproject:

```bash
python3 -m build ui
```
These should produce:

- backend wheel and sdist
- UI wheel and sdist

These are the artifacts you would publish to TestPyPI or PyPI.

## 4. Validate A Fresh Install

Use a clean environment and install from the local checkout or the built wheel.

Editable local validation:

```bash
python3 -m venv .venv-release-check
source .venv-release-check/bin/activate
python -m pip install -U pip
python -m pip install -e . --no-build-isolation
python -m pip install -e ./ui --no-build-isolation
```

Then verify:

```bash
onepiece-studio --help
onepiece-studio qa
python -c "import onepiece, onepiece_studio"
```

## 5. Optional UI Smoke Check

For a local release candidate, it is helpful to smoke-test the running UI:

```bash
onepiece-studio demo
```

Then verify the local app responds, for example at:

```text
http://localhost:8503
```

Useful smoke-check targets:

- UI shell loads
- Streamlit health endpoint responds
- demo mode opens without traceback
- one representative HDF file can be loaded

## 6. Minimal Release Checklist

Before tagging or publishing, confirm:

- GitHub Actions CI is green
- Ruff passes
- tests pass
- `onepiece-studio qa` passes
- wheel builds
- sdist builds
- fresh install works
- CLI responds
- docs build succeeds
- `CHANGELOG.md` reflects the release

## 7. Scientific Release Principle

For scientific software, the release standard should be:

- not only “the package installs”
- not only “imports succeed”
- but also “the package still performs a known scientific calculation
  correctly on a bundled reference dataset”

That is why the bundled QA command is part of the release workflow.
