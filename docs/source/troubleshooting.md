# Troubleshooting

## Install Or Launch Problems

If a new macOS user cannot start the package cleanly, run:

```bash
onepiece-studio doctor
```

This checks whether the current Python environment can import the main runtime
dependencies and whether the bundled tutorial dataset is available.

## HDF File Will Not Load

Typical beginner-facing causes are:

- the file path is wrong
- the HDF key is not `df`
- the Python environment is missing `tables`
- the environment is incomplete and cannot import `sympy`

OnePiece Studio now turns these cases into clearer messages in the UI.

## Recommended Recovery

In a fresh environment, the simplest repair path is:

```bash
pip install --upgrade pip
pip install onepiece-studio
onepiece-studio doctor
```

Then launch:

```bash
onepiece-studio tutorial
```

## Before Crawling Your Own Folders

Always confirm first that:

- the tutorial dataset opens
- `onepiece-studio qa` passes
- the `Workflow` tab can add adsorption and Gibbs columns without errors

If all three work, problems in your own project are much more likely to be
dataset-structure issues than package-installation issues.
