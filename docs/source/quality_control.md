# Quality Control And Package QA

`onepiece-studio` now ships with a small built-in quality-control path so a new
installation can verify that the package is working on a real scientific
dataset, not only on synthetic unit-test fixtures.

## What Is Included

The package contains a bundled reference HDF file:

- `catalysis_hub_co2_subset.hdf`

This file is a local Catalysis-Hub subset used to validate backend adsorption
analysis on realistic reaction-database rows.

## Run The Installed Self-Test

After installing the package, run:

```bash
onepiece-studio qa
```

This command uses the bundled reference dataset automatically.

You can also point the QA command to a specific local HDF file:

```bash
onepiece-studio qa --dataset "/path/to/catalysis_hub_subset.hdf"
```

## What The QA Command Verifies

The self-test checks all of these points:

1. the packaged HDF file exists and can be found through the installed package
2. the HDF file can be loaded with `pandas`
3. the Catalysis-Hub rows can be classified into gas, surface, and adsorbate
   roles
4. adsorption energies can be reconstructed from the linked rows
5. the reconstructed adsorption energies match the stored `reactionEnergy`
   values to numerical precision

This is a much better release check than only verifying imports or CLI help,
because it tests a real scientific workflow end to end.

## Typical Successful Output

Example:

```text
[PASS] catalysis-hub self-test
- dataset_path: .../catalysis_hub_co2_subset.hdf
- rows: 133
- adsorbate_rows: 34
- computed_adsorption_rows: 9
- max_abs_delta_vs_reaction_energy_ev: 1.7763568394002505e-14
- surface_count: 16
```

Interpretation:

- `rows`: total rows in the bundled dataset
- `adsorbate_rows`: rows classified as adsorbate systems
- `computed_adsorption_rows`: rows for which adsorption energies could be
  reconstructed with the available matching references
- `max_abs_delta_vs_reaction_energy_ev`: agreement between reconstructed
  adsorption energies and the stored Catalysis-Hub reaction energies

The last value should be very small. Values near machine precision indicate that
the backend reconstruction is behaving correctly.

## Automated Test Coverage

The package test suite also includes checks for this QA path:

- bundled dataset presence
- backend self-test pass/fail result
- CLI-level usability after installation

So the QA path exists in two layers:

- a **user-facing installed command**: `onepiece-studio qa`
- an **automated package test** inside the repo

## Why This Matters

For scientific software, “it imports” is not enough.

A useful release should demonstrate that:

- the packaged data ships correctly
- the analysis backend still performs the intended scientific calculation
- a new user can validate the install without writing custom code

That is exactly what the bundled QA command is for.
