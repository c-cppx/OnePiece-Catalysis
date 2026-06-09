# Chapter 6 Adsorption Tutorial

This folder contains a reproducible OnePiece/pandas workflow for the Chapter 6
slab HDF files:

- `01_onepiece_adsorption_energy_CO_CH3OH.ipynb` is the teaching notebook.
- `chapter6_adsorption_workflow.py` is the reusable script version.
- `create_chapter6_adsorption_tutorial.py` regenerates the notebook.
- `outputs/` contains generated CSV, pickle, and plot outputs.

The key workflow is:

1. Read every HDF file with `pd.read_hdf(filename, key="df")`.
2. Assign the clean surface reference inside each individual HDF-derived
   DataFrame.
3. Merge the enriched DataFrames only after reference assignment.
4. Calculate CO adsorption energies after adding a gas-phase CO reference.
5. Calculate methanol-to-methoxy adsorption energies for `CH3O` rows after
   adding gas-phase `CH3OH` and `H2` references.

The provided HDF files contain `CH3O` rows, but no direct `CH3OH` rows. The
notebook therefore uses the transparent reaction convention:

`* + CH3OH(g) -> CH3O* + 1/2 H2(g)`

Gas-phase reference energies are intentionally left as `np.nan` until values
from matching DFT calculations are inserted.
