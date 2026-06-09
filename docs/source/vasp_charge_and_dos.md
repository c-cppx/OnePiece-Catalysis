# VASP Charge And DOS Workflows

This page explains the new VASP-facing parts of the package from an ASE user's
point of view.

The goal is not to turn OnePiece into a general-purpose VASP parser library.
The goal is to make common catalytic post-processing reproducible across a whole
dataset.

## What Is Supported

The backend currently supports:

- reading `CHGCAR`
- integrating electron populations per atom from charge density
- estimating atom-resolved charges using `POTCAR` or `OUTCAR` valence data
- converting CHGCAR into labeled `xarray.Dataset` objects
- reading `DOSCAR`
- integrating total DOS over an energy window
- integrating site-projected DOS over an energy window
- converting DOSCAR and PDOS into labeled `xarray.Dataset` objects
- attaching those results back onto dataframe rows

Relevant public functions include:

- `onepiece.read_chgcar(...)`
- `onepiece.integrate_atomic_electron_populations(...)`
- `onepiece.read_vasp_valence_electrons(...)`
- `onepiece.compute_atomic_charges(...)`
- `onepiece.add_atomic_charge_descriptors(...)`
- `onepiece.add_adsorbate_charge_descriptors(...)`
- `onepiece.chgcar_to_xarray(...)`
- `onepiece.chgcar_planar_average(...)`
- `onepiece.chgcar_line_profile(...)`
- `onepiece.read_doscar(...)`
- `onepiece.integrate_total_dos(...)`
- `onepiece.integrate_projected_dos(...)`
- `onepiece.add_projected_dos_descriptors(...)`
- `onepiece.doscar_to_xarray(...)`
- `onepiece.doscar_integrated_pdos(...)`
- `onepiece.doscar_orbital_band_center(...)`

## CHGCAR: What The Package Actually Does

ASE already provides a useful `VaspChargeDensity` reader, and OnePiece builds on
that rather than replacing it.

```python
from onepiece import read_chgcar

chgcar = read_chgcar("/path/to/CHGCAR")
print(chgcar.grid_shape)
print(chgcar.voxel_volume)
```

The package then integrates charge density on a voxel grid by assigning each
voxel to its nearest atom.

That gives:

- `integrated_electron_populations`
- `total_integrated_electrons`

This is intentionally simple and reproducible. It is useful for trend analysis
and row-to-row comparisons, but it is **not** a replacement for a full Bader or
other topological charge partitioning workflow.

## From Electron Population To Charge

To convert integrated electrons into charges, the package needs reference
valence electron counts.

It looks for those in:

- sibling `POTCAR`
- sibling `OUTCAR`

If found, the package computes:

```text
atomic_charge = reference_valence_electrons - integrated_electron_population
```

This is the same kind of "reference-minus-observed" bookkeeping used elsewhere
in the package for adsorption energies and Gibbs energies.

## Adsorption-Style Charge Comparison

For catalytic slab datasets, the most useful quantity is not just the raw
charge on an atom. It is how that charge changes relative to a reference.

The backend now provides an adsorption-style charge workflow:

1. match a clean surface reference
2. identify which atoms belong to the adsorbate and which belong to the slab
3. compare adsorbate-side and surface-side quantities separately
4. compare the adsorbate either to:
   - a matching gas-phase reference row
   - or neutral valence-electron reference from `POTCAR` / `OUTCAR`

This yields columns such as:

- `adsorbate_integrated_electrons`
- `adsorbate_reference_integrated_electrons`
- `adsorbate_integrated_electrons_delta_vs_ref`
- `adsorbate_net_charge_e`
- `adsorbate_charge_delta_vs_ref_e`
- `surface_integrated_electrons_delta_vs_ref`
- `surface_net_charge_delta_vs_ref_e`
- `charge_balance_residual_e`

These are much closer to the kinds of questions catalytic researchers actually
ask:

- how much charge moved onto the adsorbate?
- how much polarization stayed on the slab?
- does the charge transfer track adsorption strength or site geometry?

## Example: DataFrame Enrichment

```python
from onepiece import add_adsorbate_charge_descriptors

enriched = add_adsorbate_charge_descriptors(
    frame,
    calculation_path_column="Path",
    structure_column="struc",
)
```

This assumes each row points to a calculation directory and that a `CHGCAR`
file lives there. If the directory also contains `POTCAR` or `OUTCAR`, the
charge normalization step can be completed automatically.

## DOSCAR: Total DOS And Projected DOS

For DOS, the package uses ASE's `VaspDos` helper and wraps it in a more
dataset-friendly interface.

```python
from onepiece import read_doscar, integrate_total_dos

doscar = read_doscar("/path/to/DOSCAR")
occupied_states = integrate_total_dos(doscar, energy_window=(-2.0, 0.0))
```

The reader shifts energies to the Fermi level by default, so windows like
`(-2.0, 0.0)` naturally mean "occupied states up to `E_F`".

## Site-Projected DOS

Projected DOS can be integrated by atom selection and orbital selection:

```python
from onepiece import integrate_projected_dos, read_doscar

doscar = read_doscar("/path/to/DOSCAR")
metal_d = integrate_projected_dos(
    doscar,
    atom_indices=[0, 1, 2, 3],
    orbitals=["d"],
    energy_window=(-2.0, 0.0),
)
```

Or attached back to a dataset:

```python
from onepiece import add_projected_dos_descriptors

frame = add_projected_dos_descriptors(
    frame,
    [
        {
            "column": "cu_d_pdos_below_ef",
            "elements": ["Cu"],
            "orbitals": ["d"],
            "energy_window": (-2.0, 0.0),
            "spin": "sum",
        }
    ],
    calculation_path_column="Path",
    structure_column="struc",
)
```

## Recommended Interpretation

For ASE/VASP users, the safest way to interpret these quantities is:

- use them for comparisons within one consistent workflow
- use them to rank or cluster states
- use them to correlate with adsorption energy, barriers, or geometry
- do not oversell them as method-independent absolute charges

That is especially true for charge descriptors derived from CHGCAR partitioning.

## How This Appears In The UI

The UI now exposes this through:

- a standard workflow recipe:
  - `VASP charge descriptors from CHGCAR`
- a notebook-automation block:
  - `VASP charge and projected DOS`

Those operations still run in the backend as normal dataframe transforms. The UI
only collects parameters and visualizes the result.

For the chemistry-facing plotting layer that now sits on top of these
descriptors, see:

- [Recommended Analysis Views](recommended_analysis_views.md)
- [Xarray For VASP Grids And DOS](xarray_vasp.md)
