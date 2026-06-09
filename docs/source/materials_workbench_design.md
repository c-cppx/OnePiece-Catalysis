# Local Materials Workbench Design

OnePiece Studio should behave less like a generic DataFrame viewer and more like a local
materials-science workbench for OnePiece, pandas and ASE data. The inspiration is
the workflow quality of OQMD and Materials Project, but the deployment target is
strictly local: HDF files, local image paths, local structure files, local
notebooks and local Python package APIs.

## Product Principle

The UI should make the next scientific action obvious:

- inspect calculation quality
- filter out invalid or unwanted calculations
- compare structures and energies
- build phase diagrams
- trace every plotted phase back to the exact row in the HDF file
- derive adsorption energies from reference rows in the same dataset

The UI is not a public database portal. It is a reproducible local analysis
surface for research datasets.

## Reference Patterns To Adapt

Materials Project-style patterns:

- global search by formula, name, id or elements
- left-side filters for composition and properties
- selectable result columns
- detail panels for structure, summary and properties
- phase-diagram views backed by computed entries

OQMD-style patterns:

- direct entry search
- composition and stability tables
- phase diagram and ground-state shortcuts
- structure visualization close to the entry table
- clear distinction between stable and unstable entries

OnePiece Studio should adapt these patterns without copying the public-website model. The
local workflow should keep file provenance, notebook outputs and one-off research
columns visible.

## Missing UI Perspectives

### 1. Calculation Inclusion State

Every row should have an analysis state independent of the original HDF file:

- `included`
- `excluded`
- `review`
- `reference`

The state should be stored in a local overlay file, for example
`.onepiece_studio/overlays/<dataset_hash>.parquet`, so the HDF input remains unchanged.

Useful exclusion reasons:

- bad relaxation
- duplicate structure
- wrong stoichiometry
- wrong surface termination
- non-reference clean surface
- test calculation
- manually excluded

Phase diagrams and tables should have a visible toggle:

`Use included rows only`

This is essential because one bad or irrelevant calculation can change a phase
field.

### 2. Materials-Specific Filter Sidebar

Generic numeric filters are not enough. OnePiece Studio needs domain filters grouped by
scientific meaning:

- Identity: `Name`, `Formula`, dataset, path, calculation id
- Composition: elements, `Ga`, `Cu`, `Zn`, coverage, `Monolayer_alloy`
- Surface: `hkl`, slab size, layers, termination, clean/alloyed/adsorbate
- Energy: `E`, `form_G`, `form_G_per_Area`, `formation_energy_per_atom`
- Quality: `fmax`, convergence flags, warnings, missing structure
- Provenance: source HDF, run folder, calculator, notebook/script version
- References: clean surface candidates, gas references, bulk references

The filter sidebar should show row counts after each filter, like a local search
engine for calculations.

### 3. Phase Diagram Workspace

A phase diagram view should be a first-class workflow, not just a generated
figure. It should contain:

- axis controls: `T`, `log10(pH2O/pH2)`, chemical potential presets
- included/excluded row controls
- stable phase map
- transition table
- row detail for the selected phase
- "why stable here?" energy comparison at clicked condition
- download buttons for CSV, PNG, HTML and notebook

For each clicked point `(T, ratio)`, OnePiece Studio should show:

- winning row
- runner-up rows
- energy difference to runner-up
- energy formula used
- constants used
- source HDF row index

### 4. Structure-Centric Comparison

ASE structures should become visual objects:

- structure preview in the row detail panel
- side-by-side comparison for pinned rows
- clean surface vs alloyed surface vs adsorbate structure
- export to CIF/XYZ/POSCAR when possible

The UI should still work when only image paths are available. In that case,
image columns become the structure preview fallback.

### 5. Reference Resolution For Adsorption Energies

Future adsorption-energy workflows need a local reference resolver. The important
point is that reference rows live in the same dataset and can be found through
the `Name` column.

Recommended model:

```text
E_ads = E_adsorbate_slab - E_reference_surface - n_adsorbate * E_adsorbate_reference
```

The UI should let the user define reference rules:

- current row pattern: contains adsorbate name
- clean surface reference: same `hkl`, same slab size, `Name` contains `clean`
- gas/molecule reference: `Name` equals or contains selected reference token
- optional coverage normalization

The resolver should preview matches before calculation:

| target row | matched clean surface | matched adsorbate reference | status |
| --- | --- | --- | --- |
| `Cu-211-O-...` | `Cu-211-clean-...` | `O2` | ok |
| `Cu-111-OH-...` | `Cu-111-clean-...` | `H2O/H2` | review |

Ambiguous references should not silently calculate. They should be marked
`review`.

### 6. Provenance And Audit Trail

Every derived table should store:

- source HDF path
- HDF key
- row index
- calculation timestamp if available
- script or notebook name
- constants used
- filter and exclusion state
- hash of source rows used

This is what makes local work trustworthy.

## Proposed OnePiece Studio Layout

### Top Bar

- dataset selector
- global search
- active row count
- save/load view preset

### Left Sidebar

- dataset facets
- materials filters
- inclusion/exclusion state filters
- phase-diagram parameter controls

### Main Work Area

Tabs:

- `Records`
- `Phase Diagram`
- `Structures`
- `References`
- `Adsorption`
- `Visualize`
- `Provenance`

### Right Inspector

Always shows details for the selected row or selected phase region:

- key properties
- structure/image preview
- inclusion state
- reference matches
- source row and source HDF

## OnePiece Integration Contract

OnePiece Studio should keep the current loose adapter API, but add optional richer hooks.

Minimum already supported:

```python
OnePieceSource(obj)  # obj.to_dataframe(), obj.dataframe or obj.df
```

Recommended richer contract:

```python
class OnePieceMaterialsSource:
    def to_dataframe(self) -> pd.DataFrame: ...
    def metadata(self) -> dict: ...
    def structure_for(self, row_id): ...
    def source_path_for(self, row_id) -> str: ...
    def reference_candidates(self, row, kind: str) -> pd.DataFrame: ...
```

OnePiece Studio should not require these methods. If they exist, the UI becomes smarter; if
not, it falls back to DataFrame-only mode.

## Priority Roadmap

### Phase 1: Local Materials Browser

- persistent row inclusion/exclusion overlay
- domain filter sidebar
- configurable visible columns
- row pinning and compare mode
- image and structure preview

### Phase 2: Phase Diagram Workbench

- phase diagram tab inside Streamlit
- dynamic recalculation from filtered rows
- stable phase table linked to source rows
- clicked-condition energy comparison
- export of table, plot and notebook

### Phase 3: Reference Resolver

- clean-surface matching by `Name`, `hkl`, slab size and formula
- gas/molecule reference matching
- ambiguity handling
- reference preview table

### Phase 4: Adsorption Energy Workbench

- rule-based adsorption-energy calculation
- coverage normalization
- grouped adsorption-energy plots
- export derived DataFrame to HDF/CSV/parquet

## Design Standard

The UI should be dense, calm and scientific:

- no landing page
- filters always visible
- tables and plots linked
- row provenance visible
- visual state for included/excluded/reference rows
- every derived number traceable back to source rows

This is the key difference between a nice DataFrame UI and a useful local
materials workbench.
