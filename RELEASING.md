# Releasing onepiece-studio

This project is published as the Python package `onepiece-studio`.

## 1. Validate locally

Run the full local release gate:

```bash
python3 -m pip install -e .[dev,docs,release]
python3 scripts/release_check.py
```

This covers:

- `pytest`
- `onepiece-studio qa`
- compile checks
- Sphinx HTML build
- wheel/sdist build
- `twine check`
- fresh wheel install validation

## 2. Build distributions

```bash
python3 -m build
```

This should create:

- `dist/onepiece_studio-<version>.tar.gz`
- `dist/onepiece_studio-<version>-py3-none-any.whl`

## 3. Inspect artifacts

Recommended:

```bash
python3 -m twine check dist/*
```

## 4. Test install from built wheel

```bash
python3 -m venv /tmp/onepiece_studio_release_test
/tmp/onepiece_studio_release_test/bin/python -m pip install dist/*.whl
/tmp/onepiece_studio_release_test/bin/onepiece-studio --help
/tmp/onepiece_studio_release_test/bin/onepiece-studio qa
```

## 5. Publish

```bash
python3 -m twine upload dist/*
```

## Notes

- `onepiece-studio` is the distribution name on package indexes
- `onepiece` and `onepiece_studio` are the import package names
- `onepiece-studio` is the installed console command
