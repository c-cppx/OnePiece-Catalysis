# Adsorption Energy and Barrier Software Tutorial

This folder demonstrates the reusable `onepiece.adsorption` software for local
OnePiece/pandas HDF databases.

Main files:

- `../../src/onepiece/adsorption.py`: reusable package module.
- `01_adsorption_energy_and_barrier_software.ipynb`: Jupyter notebook explaining
  the calculation path for every table and plot.
- `run_adsorption_barrier_analysis.py`: terminal-runnable analysis script.
- `create_adsorption_barrier_notebook.py`: regenerates the notebook.
- `outputs/`: generated CSV, pickle, and PNG files.

The implemented software calculates:

- clean surface references assigned before merging HDF sources,
- CO adsorption-energy columns,
- methanol-to-methoxy (`CH3OH -> CH3O* + 1/2 H2`) adsorption-energy columns,
- constrained-optimization profile points from `copt` path scans,
- apparent copt forward/reverse barriers and reaction energies.

Gas-phase references are intentionally left as `NaN` until matching DFT gas
energies are supplied. The provided HDF files contain adsorbed `CH3O`, but no
direct `CH3OH` slab rows.
