# Column Review For OnePiece Studio Materials Pages

The Cu/Ga combined local dataset currently has 274 rows and 90 original columns.
Not every page should show the same columns. OnePiece Studio should use column profiles:
each page shows the columns that answer the scientific question for that page,
while the full raw row remains available in the detail inspector.

## Recommended Page Profiles

### Overview

Purpose: fast scan of identity, dataset, formula, energy and quality.

Show first:

- `record_type`
- `dataset`
- `Name`
- `Formula`
- `composition_summary`
- `E`
- `energy_per_atom`
- `fmax`
- `quality_flag`
- `source_hdf`
- `source_row`

Why: this is the closest equivalent to the first result table in OQMD or
Materials Project. It tells the user what the row is, where it came from and
whether it looks trustworthy.

### Surface Stability

Purpose: compare surface candidates and decide which rows belong in surface
stability and phase-diagram work.

Show first:

- `dataset`
- `Name`
- `Formula`
- `hkl`
- `slabsize`
- `layers`
- `Monolayer_alloy`
- `coverage_label`
- `Area`
- `form_G`
- `form_G_per_Area`
- `form_G_per_alloy`
- `surface_ref`
- `is_clean`
- `quality_flag`

Why: surface work needs geometry, coverage, area-normalized energy and reference
context in one table.

### Phase Diagram

Purpose: expose the columns used to build bulk and surface phase diagrams.

Show first:

- `record_type`
- `dataset`
- `Name`
- `Formula`
- `phase_label`
- `Ga_percent`
- `Monolayer_alloy`
- `formation_energy_per_atom`
- `formation_energy_per_atom_numeric`
- `form_G_per_Area`
- `muGa`
- `muZn`
- `mu_Ga`
- `mu_Zn`
- `delta_Ga`
- `delta_Cu`
- `Area`

Why: these are the columns needed to understand what is plotted and how the
thermodynamic correction is made.

### References

Purpose: find clean surfaces, bulk references and later adsorption-energy
references.

Show first:

- `record_type`
- `dataset`
- `Name`
- `Formula`
- `reference_role`
- `reference_key`
- `is_clean`
- `is_adsorbate_like`
- `adsorbate_guess`
- `hkl`
- `slabsize`
- `surface_ref`
- `E`
- `E_ref`
- `Cu_ref`
- `Ni_ref`

Why: adsorption energies will depend on matching target rows to reference rows.
This page should make that matching visible before any calculation is accepted.

### Quality

Purpose: decide which calculations should be included, reviewed or excluded.

Show first:

- `quality_flag`
- `dataset`
- `Name`
- `Formula`
- `fmax`
- `has_structure`
- `has_energy`
- `has_area`
- `timestamp`
- `human_time`
- `kpts`
- `k1`
- `k2`
- `k3`
- `Path`

Why: quality review should be independent of the thermodynamic analysis. Bad
rows should be found before plotting phase diagrams.

### Structure

Purpose: inspect cell, size and descriptor context.

Show first:

- `record_type`
- `dataset`
- `Name`
- `Formula`
- `n_atoms`
- `a`
- `b`
- `c`
- `alpha`
- `beta`
- `gamma`
- `Volume`
- `Volume_per_atom`
- `Area`
- `average_Cu_GCN`
- `average_Ga_GCN`
- `average_Cu_charge`
- `average_Ga_charge`
- `min_Cu_charge`
- `min_Ga_charge`

Why: this profile connects the row table with ASE structure information and
computed descriptors.

### Provenance

Purpose: trace every row back to local files and source data.

Show first:

- `row_uid`
- `dataset`
- `source_hdf`
- `source_row`
- `Name`
- `Path`
- `human_time`
- `timestamp`
- `files`
- `parameters`
- `constraints`

Why: local research workflows need source traceability more than public database
polish.

## Derived Columns Added

OnePiece Studio now creates several context columns at load time:

- `row_uid`: stable row identifier, usually `source_hdf::source_row`
- `record_type`: `bulk`, `surface`, `cluster` or generic `record`
- `is_clean`: true when `Name` looks like a clean reference
- `is_adsorbate_like`: true when `Name` suggests `CO`, `CO2`, `H2O`, `OH`, etc.
- `adsorbate_guess`: guessed adsorbate token from `Name`
- `reference_role`: clean surface reference, bulk reference, adsorbate candidate or candidate
- `reference_key`: matching key from record type, hkl, slab size and formula
- `composition_summary`: compact non-zero elemental composition
- `coverage_label`: human-readable monolayer coverage
- `n_atoms`: atom count from ASE structure when available, otherwise element counts
- `has_structure`
- `has_energy`
- `has_area`
- `energy_per_atom`
- `formation_energy_per_atom_numeric`
- `phase_label`
- `quality_flag`

## Columns To Create Later

These should be added when the workflows mature:

- `include_state`: persistent local state: included, excluded, review, reference
- `exclude_reason`: bad relaxation, duplicate, wrong termination, wrong reference
- `calculation_status`: converged, crashed, incomplete, unknown
- `surface_family`: clean, alloyed, adsorbate, oxide, reconstruction
- `adsorption_target`: boolean for rows intended for adsorption energy
- `matched_clean_reference_uid`
- `matched_adsorbate_reference_uid`
- `reference_match_status`: ok, ambiguous, missing, manual
- `adsorption_energy`
- `adsorption_energy_per_adsorbate`
- `adsorption_energy_unit`
- `phase_diagram_include`
- `phase_region_fraction`
- `runner_up_delta_G`
- `notebook_source`
- `derived_at`
- `derived_from_hash`

The key design rule is that derived values must be traceable back to source rows
and reference rows, never silently computed.
