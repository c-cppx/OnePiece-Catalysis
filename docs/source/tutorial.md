# Tutorial: Install, Launch, And Use onepiece-studio

## 1. Install The Package

For backend-only Python workflows:

```bash
pip install onepiece
```

For the full UI workbench:

```bash
pip install onepiece-studio
```

For development in a local checkout:

```bash
pip install -e .[dev]
```

## 2. Run The Built-In Package QA

The package ships with a bundled Catalysis-Hub reference dataset and a
self-test command.

Run:

```bash
onepiece-studio qa
```

This verifies that:

- the packaged reference dataset is present
- the backend can read the HDF file
- Catalysis-Hub adsorption energies can be reconstructed
- the reconstructed values match the stored reaction energies to numerical
  precision

Typical successful output looks like:

```text
[PASS] catalysis-hub self-test
- rows: 133
- adsorbate_rows: 34
- computed_adsorption_rows: 9
```

## 3. Launch The UI

### Bundled Tutorial Dataset

```bash
onepiece-studio tutorial
```

This is the recommended first launch for new catalysis users because it opens a
known-good bundled adsorption dataset.

### Demo Mode

```bash
onepiece-studio demo
```

### Local HDF Mode

```bash
onepiece-studio hdf "/path/to/database.hdf" --key df --title "Local Database"
```

The UI runs locally in the browser through Streamlit. In this project the most
common address is:

```text
http://localhost:8503
```

## 4. Understand The Workflow Model

OnePiece Studio is designed around a simple idea:

1. a local dataset is loaded into a `pandas.DataFrame`
2. OnePiece applies scientific DataFrame operations in the backend
3. OnePiece Studio lets you inspect, filter, visualize, and save that work

The UI is not meant to replace scientific thinking. It is meant to make the
DataFrame workflow reproducible and easier to inspect.

## 5. First-Day Beginner Path

For a student who is just joining a computational catalysis project, the
recommended path is:

1. launch `onepiece-studio tutorial`
2. open `Data Sources` and note how the bundled example is represented
3. in `Workflow`, add `Adsorption + Gibbs analysis starter`
4. go to `Records` and inspect the added `G` and `adsorption_free_energy`
   columns
5. go to `Visualize` and start with the `Adsorption analysis` preset

This sequence keeps the student inside backend-driven DataFrame operations while
giving them immediate visual feedback.

For a more explicit first-day onboarding page, see:

- [First Day Guide For A Bachelor Student](first_day_student.md)

## 6. Typical First Actions In The UI

### Search And Filter

Use the Controlroom to:

- search by `Name`, `Formula`, dataset, or source path
- filter by composition, materials-system logic, numeric windows, or row state
- keep only rows that belong in the active analysis set

```{image} _static/screenshots/records.png
:alt: OnePiece Studio records view
:class: screenshot
```

### Visualize

Use the Visualize tab to build scatter plots and comparison plots from numeric
columns.

Good first plots are:

- energy versus composition
- formation energy versus surface area
- adsorption energy versus descriptor columns
- quality metrics such as `fmax`

If you are working with ASE/VASP-enriched adsorption datasets, the most useful
recommended views are documented in:

- [Recommended Analysis Views](recommended_analysis_views.md)

```{image} _static/screenshots/visualize.png
:alt: OnePiece Studio visualization view
:class: screenshot
```

### Inspect The Schema

Before a workflow becomes large, use the Schema tab to confirm:

- which columns are numeric
- which columns are object-like
- which columns contain ASE structures
- which columns contain missing values

```{image} _static/screenshots/schema.png
:alt: OnePiece Studio schema view
:class: screenshot
```

## 7. Work With Scientific State, Not Just Tables

The package supports:

- workflow operations
- saved views
- source blocks
- row-state curation
- workbook edits
- project save/load

This is what turns the software from a passive table viewer into a real local
scientific workbench.

## 8. Use The Existing Dataset Tutorials As Worked Examples

The package docs still include Cu/Ga-oriented pages because they are useful as
worked scientific examples:

- how phase-like analysis columns are interpreted
- how structure, energy, and provenance coexist in one DataFrame
- how OnePiece Studio maps domain-specific columns into useful views

Those pages are best read as examples of how to shape your own local dataset
for the workbench.
