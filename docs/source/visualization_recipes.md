# Visualization Recipes

These are good notebook and OnePiece Studio examples for the supplied databases.

## Bulk Alloy Stability

Use `CuGabulk.hdf` or `CuGabulk_oxide.hdf`.

```python
bulk = pd.read_hdf("CuGabulk.hdf", key="df")

bulk[["Name", "Ga_percent", "Zn_percent", "formation_energy_per_atom"]].sort_values(
    "formation_energy_per_atom"
).head(10)
```

OnePiece Studio plot:

- `x = Ga_percent`
- `y = formation_energy_per_atom`
- `color = alloy`

Interpretation: look for alloy compositions with negative formation energy and compare Ga- and
Zn-rich candidates.

## Surface Formation Energy By Facet

Load the individual surface files:

```python
from pathlib import Path
import pandas as pd

root = Path("path/to/surface-alloy-datasets")
surface_files = [
    "CuGasurf_100.hdf",
    "CuGasurf_110.hdf",
    "CuGasurf_111.hdf",
    "CuGasurf_211.hdf",
]

surfaces = []
for filename in surface_files:
    frame = pd.read_hdf(root / filename, key="df")
    frame["dataset"] = filename.replace(".hdf", "")
    surfaces.append(frame)

surf = pd.concat(surfaces, ignore_index=True)
```

Analyze:

```python
surf.groupby("dataset")["form_G_per_Area"].describe()
```

OnePiece Studio plot:

- `x = Area`
- `y = form_G_per_Area`
- `color = slabsize`

Interpretation: compare stability per area while watching for slab-size convergence.

## Coordination Versus Stability

```python
df[["Name", "average_Ga_GCN", "average_Cu_GCN", "form_G_per_alloy"]].dropna().sort_values(
    "form_G_per_alloy"
)
```

OnePiece Studio plot:

- `x = average_Ga_GCN`
- `y = form_G_per_alloy`
- `color = alloy`

Interpretation: low-coordinated alloy atoms may be chemically active but not always stable.

## Charge Descriptors

```python
charge_cols = [column for column in df.columns if "charge" in column]
df[["Name", *charge_cols]].head()
```

OnePiece Studio plot:

- `x = min_Ga_charge`
- `y = form_G_per_Area`
- `color = slabsize`

Interpretation: charge transfer can flag unusually reactive or under-coordinated structures.

## ASE/VASP-Enriched Adsorption Views

When the dataset already contains:

- adsorption energies
- `CHGCAR`-based charge descriptors
- ASE geometry descriptors
- optional `DOSCAR`-based d-band descriptors

the recommended visual sequence is:

1. `Adsorption analysis`
2. `Adsorption site families`
3. `Charge transfer vs adsorption energy`
4. `Surface polarization vs adsorbate height`
5. `d-band center vs adsorption energy`

Those views are now documented in more detail here:

- [Recommended Analysis Views](recommended_analysis_views.md)

## Calculation Quality

```python
df[["Name", "fmax", "E", "human_time"]].sort_values("fmax", ascending=False).head(10)
```

OnePiece Studio plot:

- histogram of `fmax`

Interpretation: large `fmax` values indicate structures that may need re-relaxation before comparing
energies.
