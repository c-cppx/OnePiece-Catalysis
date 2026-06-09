# Xarray For VASP Grids And DOS

OnePiece now includes an `xarray` layer for the parts of VASP output that are
genuinely labeled multidimensional scientific data:

- `CHGCAR` as a 3D voxel field
- `DOSCAR` as an energy-resolved dataset
- projected DOS as an `atom x orbital x energy` object

This sits next to the existing `pandas` layer rather than replacing it.

- `pandas` remains the right tool for row-wise dataset operations
- `xarray` is now the right tool for labeled 3D fields and labeled spectral
  tensors

## Public Functions

For charge-density grids:

- `onepiece.chgcar_to_xarray(...)`
- `onepiece.chgcar_planar_average(...)`
- `onepiece.chgcar_plane_integrated_electrons(...)`
- `onepiece.chgcar_cumulative_axis_profile(...)`
- `onepiece.chgcar_line_profile(...)`

For DOS data:

- `onepiece.doscar_to_xarray(...)`
- `onepiece.doscar_select_energy_window(...)`
- `onepiece.doscar_integrated_pdos(...)`
- `onepiece.doscar_orbital_band_center(...)`

## Example

```python
from onepiece import (
    chgcar_to_xarray,
    chgcar_planar_average,
    doscar_to_xarray,
    doscar_orbital_band_center,
)

chg = chgcar_to_xarray("/path/to/CHGCAR")
z_profile = chgcar_planar_average(chg, axis="z")

dos = doscar_to_xarray("/path/to/DOSCAR")
metal_d_center = doscar_orbital_band_center(
    dos,
    atom_indices=[0, 1, 2, 3],
    orbitals=["d"],
    energy_window=(-6.0, 1.0),
)
```

## Why This Matters

This makes the code clearer in exactly the places where raw arrays become hard
to trust:

- selecting one dimension out of a 3D charge grid
- comparing several reductions of the same field
- summing projected DOS over selected atoms and orbitals
- keeping energy windows explicit in descriptor calculations

For a longer worked example, see the generated homework report in:

- `docs/reports/xarray_vasp_homework/`
