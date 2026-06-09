# Recommended Analysis Views

This page collects the most useful **analysis presets** now built into the
`Visualize` tab for adsorption, charge-transfer, and DOS-driven workflows.

It is written for users who already think in terms of ASE `Atoms`, slab
references, adsorbates, and VASP post-processing outputs, but want a fast,
repeatable way to scan a whole dataset before drilling into individual rows.

## Why These Views Exist

The UI can plot any numeric columns against each other, but a chemistry-facing
workbench should do a little more than that.

These presets are meant to answer common catalytic questions quickly:

- does adsorption strength correlate with charge transfer?
- is slab polarization coupled to adsorbate lift-off or tilt?
- do site families separate into distinct energetic clusters?
- do d-band descriptors track adsorption trends in the expected direction?
- are odd structures chemically interesting or just broken?

The backend still owns the calculations. These views only help you ask the
right first questions.

## 1. Adsorption Analysis

Default use case:

- after assigning clean surface references
- after adding adsorption energies such as `E_ads_CO_eV`

Typical axes:

- `x = E_ads_CO_total_eV`
- `y = E_ads_CO_eV`
- `color = surface_ref_name`

Use this when:

- you want a first ranking of adsorption strength
- you want to compare the same adsorbate across different surfaces
- you need to catch rows where the total and per-adsorbate normalization do not
  behave as expected

Interpretation:

- more negative adsorption energy usually means stronger binding
- grouped color families often reveal surface-specific energetic regimes
- obvious outliers should usually be checked in `Records` and then in ASE

## 2. Charge Transfer Versus Adsorption Energy

Default use case:

- after `CHGCAR`-based charge enrichment
- when `adsorbate_charge_delta_vs_ref_e` is available

Typical axes:

- `x = adsorbate_charge_delta_vs_ref_e`
- `y = E_ads_CO_eV`
- `color = adsorption_site` or `surface_ref_name`

Use this when:

- you want to test whether stronger adsorption correlates with electron gain or
  electron loss on the adsorbate
- you want to separate site effects from purely electronic effects

Interpretation:

- this is often the first useful plot after building charge descriptors
- a monotonic trend is not guaranteed, but clusters are often chemically
  meaningful
- compare this view with the site-colored view below before making a mechanistic
  claim

## 3. Surface Polarization Versus Adsorbate Height

Default use case:

- after both structure descriptors and charge descriptors are available

Typical axes:

- `x = adsorbate_height_above_surface`
- `y = surface_net_charge_delta_vs_ref_e`
- `color = adsorption_site`

Alternative x choices that often help:

- `min_adsorbate_surface_distance`
- `adsorbate_tilt_deg`

Use this when:

- you want to connect geometry with slab-side charge response
- you suspect that weak binding is really a geometrical lift-off effect
- you want to identify structures that polarize the slab unusually strongly

Interpretation:

- large height with small slab polarization often points to weak interaction
- strong slab response at modest height can indicate a chemically engaged site
- this is especially useful for distinguishing electronic response from pure
  steric separation

## 4. Adsorption Site Families

Default use case:

- after the ASE geometry analysis block has classified site types

Typical axes:

- `x = adsorbate_tilt_deg`
- `y = E_ads_CO_eV`
- `color = adsorption_site`

Alternative y choices:

- `adsorbate_charge_delta_vs_ref_e`
- `surface_reconstruction_rmsd`

Use this when:

- you want to compare top, bridge, hollow, and defect-like motifs directly
- you need a quick site-resolved map before inspecting representative
  structures

Interpretation:

- this plot is often more informative than a raw site table because geometry and
  energetics are visible together
- if one site family contains the worst outliers, inspect those rows in ASE
  before treating them as real chemistry

## 5. d-Band Center Versus Adsorption Energy

Default use case:

- after `DOSCAR`-based PDOS enrichment
- when `metal_d_band_center_eV` is available

Typical axes:

- `x = metal_d_band_center_eV`
- `y = E_ads_CO_eV`
- `color = adsorption_site` or `surface_ref_name`

Use this when:

- you want an electronic-structure screening view
- you want to compare a classical descriptor against actual adsorption results

Interpretation:

- this is a good first-pass descriptor plot, not a final mechanistic proof
- use it to find whether the dataset roughly follows expected d-band logic
- when the trend breaks, compare the same points in:
  - `Charge transfer versus adsorption energy`
  - `Adsorption site families`
  - the record detail / ASE view

## 6. Quality-Control Views

Some of the most useful chemistry plots are actually QC plots.

Recommended columns:

- `fmax`
- `min_interatomic_distance`
- `min_bond_ratio`
- `surface_reconstruction_rmsd`
- `adsorbate_is_dissociated`
- `adsorbate_desorbed`

Good first checks:

- histogram of `fmax`
- scatter of `surface_reconstruction_rmsd` versus adsorption energy
- scatter of `min_interatomic_distance` versus adsorption energy

Use these when:

- you want to decide whether an energetic outlier is still chemically credible
- you need to separate interesting chemistry from broken geometry

## Suggested Reading Order

For a charge- and DOS-enriched adsorption dataset, a good sequence is:

1. `Adsorption analysis`
2. `Adsorption site families`
3. `Charge transfer versus adsorption energy`
4. `Surface polarization versus adsorbate height`
5. `d-band center versus adsorption energy`
6. QC views for the interesting outliers

That progression usually moves from:

- broad thermochemistry
- to geometry
- to charge transfer
- to electronic structure
- and finally to sanity checking

## How This Relates To ASE Work

These plots are not meant to replace direct structure inspection.

They are best used like this:

1. enrich the dataframe with ASE/VASP descriptors
2. scan trends in `Visualize`
3. click outliers or clusters
4. inspect the selected rows in `Records`
5. open representative structures in ASE

That division of labor keeps the workflow fast without giving up chemical
judgment.

For a broader local materials-style example using a Cu/Ga ASE dataset rather
than a reaction database, see:

- [Cu/Ga Worked Example](cuga_worked_example.md)
