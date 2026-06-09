# ASE To UI Workflow Mapping

If you already work comfortably in ASE and notebooks, the best way to
understand OnePiece Studio is to map familiar scripting steps onto backend
operations and UI actions.

## The Familiar ASE Workflow

A common catalytic notebook flow looks like this:

1. load structures and metadata from files
2. inspect a few `Atoms` objects directly
3. identify clean surface references
4. compute adsorption energies or other derived quantities
5. filter out broken calculations
6. compare subsets in plots or summary tables
7. go back to individual structures when a point looks interesting

OnePiece Studio keeps the same rhythm, but makes it reproducible and persistent.

## Mapping: Notebook Habit -> OnePiece Layer

### Read structures from disk

Notebook:

```python
df = pd.read_hdf("dataset.hdf", key="df")
atoms = df.loc[0, "struc"]
```

Package:

- backend: `onepiece.read_hdf_path(...)`
- frontend: `onepiece-studio hdf ...`

### Assign clean references

Notebook:

```python
from onepiece import assign_surface_references

df = assign_surface_references(df)
```

UI:

- Workflow Builder
- standard operation:
  - `Assign surface references and adsorption columns`

### Compute adsorption energies

Notebook:

```python
from onepiece import add_adsorption_energies

df = add_adsorption_energies(df, {"CO": -14.0, "H2": -6.0})
```

UI:

- Workflow Builder
- standard operations or notebook-automation blocks

### Compute structure-derived descriptors

Notebook:

```python
from onepiece import add_structure_descriptors

df = add_structure_descriptors(df)
```

UI:

- Notebook Automation
- `Structure descriptor workbench`

### Compute CHGCAR-based charge descriptors

Notebook:

```python
from onepiece import add_adsorbate_charge_descriptors

df = add_adsorbate_charge_descriptors(df, calculation_path_column="Path")
```

UI:

- Notebook Automation
- `VASP charge and projected DOS`

### Compute PDOS descriptors

Notebook:

```python
from onepiece import add_projected_dos_descriptors

df = add_projected_dos_descriptors(
    df,
    [
        {
            "column": "metal_d_pdos_below_ef",
            "elements": ["Cu", "Ni", "Ga", "Zn"],
            "orbitals": ["d"],
            "energy_window": (-2.0, 0.0),
        }
    ],
)
```

UI:

- same `VASP charge and projected DOS` block
- optional PDOS integrations table

### Inspect a suspicious structure

Notebook:

```python
atoms = df.loc[idx, "struc"]
view(atoms)
```

UI:

- select the row or scatter point
- use `Open ASE`

## Why The UI Is Useful Even For Strong ASE Users

The UI is not meant to replace scripting fluency. It helps in three places
where strong ASE users still often lose time:

### 1. Reproducible state

Instead of re-running ad hoc cells and remembering the order, the workflow is
stored as an explicit list of operations.

### 2. Multi-row inspection

ASE is excellent for one structure at a time. The UI is better when you need to
scan:

- dozens of related states
- many adsorption candidates
- a filtered subset after several curation rules

### 3. Shared analysis

A saved OnePiece Studio project is easier to reopen and discuss than a
half-finished notebook with hidden in-memory state.

## The Right Division Of Labor

The most productive way to use the package is usually:

- keep ASE for direct structure reasoning
- keep notebooks for exploratory or publication-specific analysis
- use OnePiece for stable, repeated, dataset-wide transforms
- use OnePiece Studio when the question is:
  - "what is the active dataset state right now?"
  - "which rows pass the current logic?"
  - "which structure does this plotted point come from?"

That division of labor fits very naturally into an ASE-centered research style.
