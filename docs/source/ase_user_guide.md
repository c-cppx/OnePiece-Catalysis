# ASE User Guide

This guide is for readers who are already comfortable with ASE and want to
understand what `onepiece-studio` adds on top of that workflow.

The short version is:

- ASE remains the structure model
- OnePiece adds reproducible dataframe-level scientific operations
- OnePiece Studio turns those operations into a local, inspectable workbench

If you already think in terms of `Atoms`, slabs, adsorbates, VASP folders, and
post-processing scripts, this package should feel familiar rather than
surprising.

## The Core Mental Model

`onepiece-studio` does **not** replace ASE. It treats ASE as the structure
authority and builds a dataframe-centric analysis layer around it.

That means a typical row still looks like an ASE user would expect:

- one row = one calculation, structure, path image, or reference state
- one structure column = an ASE `Atoms` object
- one path column = a calculation folder on disk
- several scalar columns = energies, forces, compositions, descriptors

The package becomes useful when you want to do all of these at once:

- keep the local structure objects alive
- add reference-matched adsorption energetics
- derive chemical descriptors
- curate bad calculations reproducibly
- compare many structures as a database rather than one folder at a time

## What OnePiece Adds Beyond Raw ASE

ASE is excellent at:

- representing structures
- reading and writing files
- visual inspection
- calculator integration
- local geometric analysis

OnePiece is meant to help with the things ASE does not try to solve directly:

- project-wide dataframe workflows
- clean-surface reference matching
- adsorption-energy bookkeeping
- Gibbs free-energy bookkeeping
- reaction-network annotations
- reproducible row-state curation
- packaging those analysis steps into a saved project

In other words:

- ASE answers: "what is this structure?"
- OnePiece answers: "how does this structure sit inside a larger catalytic dataset?"

## Where To Go Next

- [ASE Structures In DataFrames](ase_structures_in_dataframes.md)
- [VASP Charge And DOS Workflows](vasp_charge_and_dos.md)
- [ASE To UI Workflow Mapping](ase_to_ui_workflow_mapping.md)
- [OnePiece Backend API Contract](onepiece_backend_api.md)
