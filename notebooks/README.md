# Notebook Layout

This repository keeps two kinds of notebook material side by side:

- maintained package-facing worked examples and generators
- historical or research-specific notebooks that remain useful as reference material

The maintained entry points are:

- `build_cuga_worked_example.py`
- `catalysis_hub_tutorial/build_catalysis_hub_worked_example.py`
- `create_catalysis_hub_onepiece_notebook.py`
- `create_onepiece_phase_tutorial_notebooks.py`
- `chapter6_adsorption_tutorial/create_chapter6_adsorption_tutorial.py`
- `adsorption_barrier_software/create_adsorption_barrier_notebook.py`

Generated notebooks and result tables in this tree are kept so the documentation
and worked examples remain reproducible, but they should be treated as artifacts
derived from the maintained builder scripts above.

If you are updating package behavior, prefer editing the builder scripts and
documentation pages first, then regenerate derived notebook artifacts when
needed.
