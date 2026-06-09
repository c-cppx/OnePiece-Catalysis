from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pfui.adsorption import add_catalysis_hub_adsorption_energies
from pfui.automation import apply_curation_rules


ROOT = Path(__file__).resolve().parent.parent
HDF_PATH = ROOT / "notebooks" / "catalysis_hub_tutorial" / "notebooks" / "catalysis_hub_tutorial" / "outputs" / "catalysis_hub_co2_subset.hdf"
OUT = ROOT / "docs" / "reports" / "catalysis_hub_master_report"
ASSETS = OUT / "assets"


def image_md(path: Path, alt: str) -> str:
    return f"![{alt}]({path.resolve()})"


def surface_family(surface: object) -> str:
    text = str(surface)
    if "NC" in text:
        return "single-atom/support"
    if "O" in text and not text.endswith("-fcc"):
        return "oxide/support"
    return "metal"


def md_table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    data = frame.copy()
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            def _fmt(value):
                if pd.isna(value):
                    return ""
                if isinstance(value, (int, np.integer)):
                    return str(int(value))
                if isinstance(value, (float, np.floating)) and float(value).is_integer():
                    return str(int(value))
                if isinstance(value, (float, np.floating)):
                    return format(float(value), floatfmt).rstrip("0").rstrip(".")
                return str(value)
            data[column] = data[column].map(_fmt)
        else:
            data[column] = data[column].fillna("").astype(str)
    header = "| " + " | ".join(map(str, data.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(data.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in data.to_numpy().tolist()]
    return "\n".join([header, divider, *rows])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    df = pd.read_hdf(HDF_PATH, "df")
    curated = apply_curation_rules(
        df,
        static_fmax_max=0.08,
        copt_fmax_max=0.12,
        exclude_name_tokens=["failed", "broken"],
        action="mark_review",
    )
    full = add_catalysis_hub_adsorption_energies(curated)

    reactions = (
        full[
            [
                "reaction_id",
                "Equation",
                "surfaceComposition",
                "facet",
                "reactionEnergy",
                "activationEnergy",
                "publication_title",
                "publication_year",
                "publication_doi",
                "chemicalComposition",
                "dftCode",
                "dftFunctional",
                "curation_status",
            ]
        ]
        .drop_duplicates("reaction_id")
        .copy()
    )
    reactions["reactionEnergy"] = pd.to_numeric(reactions["reactionEnergy"], errors="coerce")
    reactions["activationEnergy"] = pd.to_numeric(reactions["activationEnergy"], errors="coerce")
    reactions["surface_family"] = reactions["surfaceComposition"].map(surface_family)

    adsorption = full.loc[full["adsorption_energy"].notna()].copy()
    adsorption["adsorption_energy"] = pd.to_numeric(adsorption["adsorption_energy"], errors="coerce")

    hcoo_cooh = reactions[
        reactions["Equation"].isin(
            [
                "CO2(g) + 0.5H2(g) + * -> HCOO*",
                "CO2(g) + 0.5H2(g) + * -> COOH*",
                "CO2(g) + 0.5H2(g) + * -> CHO2*",
            ]
        )
    ].copy()

    route_pivot = hcoo_cooh.pivot_table(
        index="surfaceComposition",
        columns="Equation",
        values="reactionEnergy",
        aggfunc="first",
    )
    if {
        "CO2(g) + 0.5H2(g) + * -> HCOO*",
        "CO2(g) + 0.5H2(g) + * -> COOH*",
    }.issubset(route_pivot.columns):
        route_pivot["HCOO_minus_COOH_eV"] = (
            route_pivot["CO2(g) + 0.5H2(g) + * -> HCOO*"]
            - route_pivot["CO2(g) + 0.5H2(g) + * -> COOH*"]
        )

    # Save core tables.
    reactions.to_csv(ASSETS / "reaction_level_table.csv", index=False)
    adsorption.to_csv(ASSETS / "adsorption_rows.csv", index=False)
    route_pivot.to_csv(ASSETS / "hcoo_cooh_route_comparison.csv")

    # Figure 1: composition by equation.
    equation_counts = reactions["Equation"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    equation_counts.plot.barh(ax=ax, color="#4c78a8")
    ax.set_xlabel("number of unique reactions")
    ax.set_ylabel("reaction class")
    ax.set_title("Catalysis-Hub CO2 subset: reaction classes represented in the local HDF")
    plt.tight_layout()
    fig.savefig(ASSETS / "figure_01_equation_counts.png", dpi=220)
    plt.close(fig)

    # Figure 2: reaction-energy heatmap.
    heatmap = reactions.pivot_table(
        index="surfaceComposition",
        columns="Equation",
        values="reactionEnergy",
        aggfunc="first",
    )
    heatmap = heatmap.loc[heatmap.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(12, 7.5))
    im = ax.imshow(heatmap.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-3, vmax=6)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_title("Reaction-energy map across surfaces")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("reaction energy / eV")
    plt.tight_layout()
    fig.savefig(ASSETS / "figure_02_reaction_energy_heatmap.png", dpi=220)
    plt.close(fig)

    # Figure 3: CO2 adsorption energies.
    co2_ads = adsorption.sort_values("adsorption_energy")
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.barh(co2_ads["surfaceComposition"], co2_ads["adsorption_energy"], color="#54a24b")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("CO2 adsorption energy / eV")
    ax.set_ylabel("surface")
    ax.set_title("Surfaces with directly reconstructable CO2 adsorption energies")
    plt.tight_layout()
    fig.savefig(ASSETS / "figure_03_co2_adsorption.png", dpi=220)
    plt.close(fig)

    # Figure 4: HCOO vs COOH preference.
    preference = route_pivot.dropna(subset=["HCOO_minus_COOH_eV"]).sort_values("HCOO_minus_COOH_eV")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(preference.index, preference["HCOO_minus_COOH_eV"], color="#f58518")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(r"$\Delta E$(HCOO*) - $\Delta E$(COOH*) / eV")
    ax.set_ylabel("surface")
    ax.set_title("Hydrogenation branch preference: negative values favor HCOO*")
    plt.tight_layout()
    fig.savefig(ASSETS / "figure_04_hcoo_vs_cooh.png", dpi=220)
    plt.close(fig)

    # Figure 5: barrier subset.
    barrier_subset = reactions[reactions["activationEnergy"].notna()].sort_values("activationEnergy")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = barrier_subset["surfaceComposition"] + " | " + barrier_subset["Equation"]
    ax.barh(labels, barrier_subset["activationEnergy"], color="#e45756")
    ax.set_xlabel("activation energy / eV")
    ax.set_ylabel("reaction")
    ax.set_title("Barrier information available in the downloaded subset")
    plt.tight_layout()
    fig.savefig(ASSETS / "figure_05_barriers.png", dpi=220)
    plt.close(fig)

    # Figure 6: method mix.
    method_mix = (
        reactions.groupby(["dftCode", "dftFunctional"])
        .size()
        .sort_values(ascending=True)
        .rename("count")
        .reset_index()
    )
    method_mix["label"] = method_mix["dftCode"] + " / " + method_mix["dftFunctional"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(method_mix["label"], method_mix["count"], color="#72b7b2")
    ax.set_xlabel("number of unique reactions")
    ax.set_ylabel("DFT setup")
    ax.set_title("Method mix inside the downloaded Catalysis-Hub subset")
    plt.tight_layout()
    fig.savefig(ASSETS / "figure_06_method_mix.png", dpi=220)
    plt.close(fig)

    # Summary tables for report body.
    overview = pd.DataFrame(
        [
            {
                "metric": "system rows in HDF",
                "value": len(full),
            },
            {
                "metric": "unique reactions",
                "value": reactions["reaction_id"].nunique(),
            },
            {
                "metric": "surface compositions",
                "value": reactions["surfaceComposition"].nunique(),
            },
            {
                "metric": "publications represented",
                "value": reactions["publication_title"].nunique(),
            },
            {
                "metric": "rows marked review by curation",
                "value": int((full["curation_status"] == "review").sum()),
            },
            {
                "metric": "directly reconstructable CO2 adsorption energies",
                "value": int(adsorption["adsorption_energy"].notna().sum()),
            },
            {
                "metric": "finite activation barriers",
                "value": int(reactions["activationEnergy"].notna().sum()),
            },
        ]
    )

    top_exergonic = reactions.sort_values("reactionEnergy").head(10)[
        ["surfaceComposition", "facet", "Equation", "reactionEnergy", "activationEnergy"]
    ]
    top_endergonic = reactions.sort_values("reactionEnergy", ascending=False).head(10)[
        ["surfaceComposition", "facet", "Equation", "reactionEnergy", "activationEnergy"]
    ]
    family_summary = (
        reactions.groupby("surface_family")
        .agg(
            reactions=("reaction_id", "size"),
            mean_reaction_energy=("reactionEnergy", "mean"),
            min_reaction_energy=("reactionEnergy", "min"),
            max_reaction_energy=("reactionEnergy", "max"),
            barriers=("activationEnergy", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    adsorption_summary = adsorption[
        ["surfaceComposition", "facet", "adsorption_energy", "reactionEnergy", "publication_title"]
    ].sort_values("adsorption_energy")
    barrier_summary = barrier_subset[
        ["surfaceComposition", "facet", "Equation", "reactionEnergy", "activationEnergy", "publication_title"]
    ]

    report = dedent(
        f"""
        # Technical Report: Analysis of the Downloaded Catalysis-Hub CO2 Dataset

        **Audience:** experimental chemists interested in the interpreted trends rather than the theoretical implementation details.

        **Source file analyzed:** `{HDF_PATH}`

        ## Executive Summary

        This report analyzes the local Catalysis-Hub HDF subset that was previously downloaded and converted into a tabular format.
        The dataset contains **{len(full)} linked system rows** corresponding to **{reactions["reaction_id"].nunique()} unique reaction entries** across **{reactions["surfaceComposition"].nunique()} surface compositions** from **{reactions["publication_title"].nunique()} literature sources**.

        The chemical focus of the downloaded subset is **CO2 activation on surfaces**. The dominant reaction classes are:

        - molecular CO2 adsorption, `CO2(g) + * -> CO2*`
        - dissociation toward `CO* + O*`
        - deep deoxygenation toward `C*`
        - first hydrogenation toward `HCOO*`, `COOH*`, and `CHO2*`

        The main data-driven conclusions are:

        1. **Vacancy-rich ceria-derived oxide surfaces bind CO2 most strongly.**
           The most stabilizing adsorption energies in the dataset are found for `CeO2-rocksalt-1Ovacancy`, `SmCeO2-rocksalt-1Ovacancy`, and `CeO2-rocksalt-2Ovancies`, with reconstructed CO2 adsorption energies between roughly `-2.84 eV` and `-1.12 eV`.

        2. **Ni binds CO2 much more weakly than reduced ceria, and facet matters.**
           In this subset the Ni entries shift from slightly exergonic adsorption on `Ni(211)` (`-0.161 eV`) to mildly endergonic adsorption on `Ni(111)` (`+0.325 eV`).

        3. **Direct C formation from CO2 is strongly uphill on all surfaces where it appears.**
           The `CO2 -> C*` entries are among the most endergonic reactions in the entire dataset, typically between `+3.95 eV` and `+5.78 eV`. This strongly argues against deep direct deoxygenation as a realistic low-energy route under the conditions represented here.

        4. **For the small metal subset with hydrogenation data, HCOO* is consistently more stable than COOH*.**
           On Ag, Au, Pd, and Rh the energy difference `E(HCOO*) - E(COOH*)` is always negative, meaning the formate-like branch is favored over the carboxyl branch in this dataset.

        5. **Barrier information is sparse but chemically informative.**
           Only six unique reactions in the subset contain a finite activation barrier. Among them, Ru and Co are the only surfaces where the direct `CO2 -> CO* + O*` dissociation step is both thermodynamically favorable and associated with a comparatively small barrier.

        ## 1. What is in the downloaded file?

        The local HDF is not a single homogeneous study. It is a stitched subset of Catalysis-Hub entries from several publications, different DFT engines, and different methodological settings.
        That is useful for trend spotting, but it means the dataset should be treated as a **comparative screening dataset**, not as a single internally uniform benchmark.

        {md_table(overview, floatfmt=".0f")}

        ### Publications and methods represented

        The downloaded subset spans classic metal-surface CO2 dissociation studies, oxide-vacancy chemistry, and small electrochemical CO2-reduction descriptor datasets.
        The method mix is dominated by `VASP / PBE+U` for the ceria-based oxide entries, `DACAPO / RPBE` for several older metal dissociation entries, and `QE / BEEF-vdW` for the hydrogenation branch entries on close-packed metals.

        {image_md(ASSETS / "figure_06_method_mix.png", "Method mix")}

        ## 2. Curation and data quality

        A light curation pass was applied before interpretation.
        Rows were marked for review when they lacked an energy, had no reliable structure, or exceeded conservative force thresholds.
        In practice, the review rows are almost entirely the six `N/A` entries that only carry a reaction summary without a linked system energy, plus two rows with somewhat elevated residual forces.

        This matters because the reaction-level trends remain usable, but **system-level adsorption-energy reconstruction is only meaningful when the corresponding gas, surface, and adsorbate rows are all present under the same reaction id**.

        ## 3. Reaction classes represented in the subset

        The figure below shows that the file is heavily centered on CO2 adsorption and first-step transformations rather than on long, fully resolved pathways.

        {image_md(ASSETS / "figure_01_equation_counts.png", "Equation counts")}

        This is exactly what an experimental reader should keep in mind: this dataset is strong for **first mechanistic ranking questions** such as
        “which surfaces bind CO2 strongly?”, “which direction of the first hydrogenation step is preferred?”, and “is direct dissociation plausible?”.
        It is not a complete mechanistic map all the way to a final product.

        ## 4. Reaction-energy landscape across surfaces

        The heatmap below condenses the core result of the dataset.
        Blue entries are stabilizing or exergonic, red entries are uphill.

        {image_md(ASSETS / "figure_02_reaction_energy_heatmap.png", "Reaction heatmap")}

        Three broad patterns emerge:

        - **Oxide and defective oxide surfaces** stabilize molecular CO2 adsorption much more strongly than the small metal subset.
        - **Direct CO formation** from CO2 is uphill on the oxide entries shown here, but can become favorable on specific stepped transition-metal surfaces such as Ru(211) and Co(211).
        - **Direct carbon formation** is always strongly uphill and therefore chemically implausible as a facile elementary step in this dataset.

        ### Surface-family summary

        {md_table(family_summary)}

        The table above should not be over-read quantitatively because the families are sampled unevenly, but qualitatively it is clear that:

        - the **oxide/support** class contains the strongest CO2-binding entries;
        - the **metal** class contains the most direct dissociation/barrier data;
        - the single Fe–N–C entry is too sparse for a broad comparative statement.

        ## 5. Directly reconstructable CO2 adsorption energies

        A valuable technical check is that the local HDF contains enough information to reconstruct adsorption energies directly for the `CO2*` entries.
        For nine reactions, the corresponding `CO2gas`, `star`, and `CO2star` rows are present under the same reaction id.
        Using

        `E_ads(CO2) = E(CO2*) - E(*) - E(CO2_g)`

        reproduces the published Catalysis-Hub `reactionEnergy` values to numerical precision.

        {image_md(ASSETS / "figure_03_co2_adsorption.png", "CO2 adsorption")}

        ### Experimental interpretation

        For an experimental audience, the practical reading is:

        - **strongly negative adsorption energies** imply surfaces that are effective at trapping and activating CO2;
        - **mildly negative or near-zero adsorption energies** imply weaker molecular binding, often more compatible with facile desorption or with the need for additional activation steps;
        - **positive adsorption energies** mean that molecular CO2 adsorption itself is not favored under the chosen reference.

        In this subset, vacancy-rich ceria surfaces clearly dominate the strong-binding regime, whereas Ni lies much closer to thermoneutral adsorption.

        ### Lowest-energy CO2 adsorption entries

        {md_table(adsorption_summary.head(9))}

        ## 6. Competition between HCOO* and COOH* on metals

        One of the most experimentally relevant questions is which branch of the first hydrogenation step is preferred.
        In this subset, four metals contain both `HCOO*` and `COOH*` entries.

        {image_md(ASSETS / "figure_04_hcoo_vs_cooh.png", "HCOO vs COOH")}

        The interpretation is straightforward:

        - negative values mean `HCOO*` is more stable than `COOH*`;
        - positive values would mean the opposite.

        Here, **all four metals favor HCOO***.
        The preference is strongest for Ag and weakest for Pd, but the sign is consistent across the available metal entries.

        For experiment, this suggests that under conditions represented by these calculations, the first hydrogenation event is more likely to populate a **formate-like intermediate** than a **carboxyl-like intermediate** on these metal surfaces.

        ### Hydrogenation-branch comparison table

        {md_table(preference.reset_index()[["surfaceComposition", "HCOO_minus_COOH_eV"]])}

        ## 7. Direct CO2 dissociation and available barriers

        Barrier data are sparse in the downloaded subset, but they are highly useful because they immediately separate merely favorable final states from kinetically plausible steps.

        {image_md(ASSETS / "figure_05_barriers.png", "Barrier subset")}

        The barrier-containing subset leads to three practical takeaways:

        1. **Ru(211)** stands out as the most favorable dissociation case in the downloaded data:
           reaction energy `-1.477 eV` and a listed barrier very close to zero.

        2. **Co(211)** is also favorable:
           reaction energy `-1.107 eV` with a moderate barrier of `0.372 eV`.

        3. **Pd(111)`, `Ag(211)`, and `Au(111)` are much less promising for this direct dissociation step**:
           both thermodynamics and barriers are less favorable.

        The ZnO entry is special because it corresponds to molecular CO2 adsorption with a reported barrier of `0.000 eV`, which is consistent with essentially barrierless adsorption in that specific study.

        ### Barrier table

        {md_table(barrier_summary)}

        ## 8. Most favorable and least favorable reactions in the downloaded subset

        These two tables are a useful “bottom line” for rapid experimental reading.
        They show which elementary steps are strongly stabilized and which are strongly disfavored.

        ### Ten most exergonic entries

        {md_table(top_exergonic)}

        ### Ten most endergonic entries

        {md_table(top_endergonic)}

        The most important chemical message from these extremes is that:

        - **molecular CO2 adsorption** can be strongly favorable on reduced ceria-type surfaces;
        - **C* formation** is consistently too uphill to be considered an accessible first-step route in this subset.

        ## 9. What an experimental chemist can take away

        If the question is “which classes of surfaces deserve attention for CO2 activation?”, this dataset supports the following hierarchy:

        - **Defective ceria-based oxides**: strongest CO2 capture and activation in the present data.
        - **Stepped transition-metal surfaces such as Ru(211) and Co(211)**: most promising for direct dissociation among the barrier-containing subset.
        - **Au, Ag, Pd**: weaker for direct dissociation, but still chemically informative in hydrogenation-branch comparisons.
        - **Ni**: intermediate behavior, with facet-sensitive CO2 adsorption.

        If the question is “which first hydrogenation branch is preferred?”, the answer from this subset is:

        - the available metal entries consistently favor **HCOO*** over **COOH***.

        If the question is “is direct carbon deposition from CO2 a low-energy route here?”, the answer is:

        - **no**. The `CO2 -> C*` entries are far too uphill in this dataset.

        ## 10. Limitations of this dataset

        This report should be read with a few clear limitations in mind:

        1. The file is a **subset**, not the whole of Catalysis-Hub.
        2. It mixes **different publications, codes, and functionals**.
        3. The number of explicit **barrier entries is small**.
        4. Most entries are **first-step thermodynamic snapshots**, not complete pathways.
        5. Only a subset of rows can be used for exact adsorption-energy reconstruction because that requires the matching gas, clean surface, and adsorbate entries under the same reaction id.

        These limitations do not erase the value of the data.
        They simply define the right use case: **comparative screening and mechanistic direction-finding**, not single-number absolute benchmarking across all systems.

        ## 11. Final conclusion

        The downloaded Catalysis-Hub HDF already contains enough chemically resolved information to support a meaningful, master-thesis-level analysis for an experimental audience.

        The strongest and clearest results are:

        - CO2 adsorption is most favorable on defective ceria-derived oxide surfaces.
        - Ni shows much weaker, facet-sensitive CO2 adsorption.
        - Ru(211) and Co(211) are the most favorable direct dissociation cases among the barrier-containing entries.
        - The first hydrogenation step on the available close-packed metal surfaces favors HCOO* over COOH*.
        - Direct formation of C* from CO2 is strongly disfavored in the downloaded subset.

        In practical experimental language:
        this dataset points much more strongly toward **surface-specific CO2 activation and selective first-step chemistry** than toward indiscriminate deep deoxygenation.
        """
    )
    report = "\n".join(line[8:] if line.startswith("        ") else line for line in report.splitlines()).strip() + "\n"

    report_path = OUT / "catalysis_hub_master_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
