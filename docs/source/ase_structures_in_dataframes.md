# ASE Structures In DataFrames

For ASE users, the most important design choice in this package is simple:

> the `Atoms` object is still the primary structure representation, even when
> the dataset is being treated like a table.

This page explains what that means in practice.

## One Row, One Local Scientific Object

In a OnePiece dataset, the row is not just a spreadsheet row. It is usually a
structured scientific record containing:

- a structure (`Atoms`)
- a local path (`Path`)
- an identifier (`Name`)
- energies and convergence metadata
- optionally reference assignments, charge descriptors, or reaction metadata

Typical example:

```python
import pandas as pd

df = pd.read_hdf("my_dataset.hdf", key="df")
row = df.iloc[0]

atoms = row["struc"]
name = row["Name"]
energy = row["E"]
path = row["Path"]
```

This is why the package is comfortable with both:

- dataframe operations over many rows
- structure-aware logic inside individual rows

## Why The DataFrame Layer Matters

If you already use ASE heavily, you may wonder why not just keep everything as
lists of `Atoms` plus ad hoc Python dictionaries.

The dataframe layer becomes valuable when you need to:

- sort or filter many calculations by scalar criteria
- compare several catalysts or several adsorbates at once
- add columns derived from common rules
- save a whole analysis state reproducibly
- return later to exactly the same filtered subset

So the dataframe is not there to flatten away the structure. It is there to
organize many structures at once.

## Structure Columns

The package typically expects one or more of these columns:

- `struc`
- `CONTCAR`
- `structure`
- `atoms`

Backend helpers usually resolve the first usable one automatically.

For example, many OnePiece functions use a `primary_structure(...)` helper
internally so they can work against older or differently named datasets without
rewriting every operation.

## Structure-Derived Quantities

Because the structure object stays available, backend functions can derive
quantities that are not just text parsing.

Examples already in the package:

- adsorbate composition from total structure minus clean surface reference
- adsorbate height above the surface
- surface atom count
- cell volume
- CHGCAR-based integrated electron populations
- CHGCAR-based charge deltas relative to surface and gas references

This is the key point for ASE users:

> the package prefers structure-difference logic over fragile naming heuristics
> whenever the structure is available.

## Surface Reference Matching

Many catalytic workflows need a clean surface reference before anything else can
be computed.

OnePiece uses a two-level approach:

1. infer the likely clean-surface identity from row naming/path patterns
2. whenever possible, compare actual `Atoms` objects after the reference is
   assigned

That lets the package do things like:

- compute adsorption stoichiometry from `Atoms - surface_ref.Atoms`
- map CHGCAR charge populations onto the adsorbate atoms only
- separate surface polarization from adsorbate charge transfer

## What The UI Does With Structure Columns

Once structure columns exist, OnePiece Studio can:

- show structure summaries in record views
- open selected structures in ASE
- use structure-aware workflows like adsorption references or charge descriptors
- expose descriptors derived from the `Atoms` rather than only from text columns

The UI does not reinterpret the structure on its own. It calls backend
operations that already know how to work with ASE objects.

## A Good Dataset Shape For ASE Users

If you are assembling your own dataset, a very usable minimum is:

```python
required_columns = [
    "Name",
    "Path",
    "E",
    "Formula",
    "struc",
]
```

Then add as available:

- `fmax`
- `record_class`
- thermochemistry columns like `E_ZPE`, `Cv_vib`, `S_vib`
- image columns for structure or DOS thumbnails
- any precomputed descriptors you already trust

## Practical Advice

If a value could be derived either by:

- fragile string parsing from `Name`
- or structure-aware logic from `Atoms`

prefer the structure-aware route.

That principle is one of the strongest reasons to use OnePiece rather than a
purely tabular materials dashboard.
