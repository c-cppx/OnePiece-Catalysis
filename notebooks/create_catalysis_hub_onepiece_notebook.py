from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "catalysis_hub_tutorial"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook(title: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [md(f"# {title}")] + cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


COMMON_SETUP = """
from __future__ import annotations

from io import StringIO
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from ase.io import read

from onepiece import add_structure_descriptors, annotate_reaction_network, apply_curation_rules

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

ROOT = Path.cwd()
OUT = ROOT / "notebooks" / "catalysis_hub_tutorial" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

GRAPHQL_URL = "https://api.catalysis-hub.org/graphql"
"""


def make_notebook():
    cells = [
        md(
            """
This notebook shows how we can use **Catalysis-Hub** as an internet data source,
convert the returned data into a **local HDF database**, and then analyze the
result with our **OnePiece Studio-compatible backend tools**.

The design goal is practical:

1. fetch a chemically meaningful subset from Catalysis-Hub,
2. flatten the GraphQL result into a table,
3. reconstruct ASE structures from the returned CIF payload where available,
4. save the result locally as `key='df'` HDF,
5. perform first-pass catalysis analysis on the local DataFrame.

Important note:
Catalysis-Hub already stores published reaction and adsorption energies. In this
notebook we therefore treat the database as a **reaction-energy source** and
structure archive rather than recomputing all adsorption energies from scratch.
"""
        ),
        code(COMMON_SETUP),
        md(
            """
## 1. Define a GraphQL query

Catalysis-Hub exposes a GraphQL API. Their documentation shows that the
`reactions` table can be filtered by `reactants`, `products`, `facet`,
`chemicalComposition`, and energy fields, and that the `systems` table can be
queried through each reaction node.

For a first local adsorption/reaction HDF we fetch a focused subset around
`CO2`-containing chemistry together with reaction systems and structure payloads.
"""
        ),
        code(
            r"""
QUERY = '''
{
  reactions(first: 40, reactants: "CO2") {
    totalCount
    edges {
      node {
        id
        Equation
        chemicalComposition
        surfaceComposition
        facet
        sites
        reactants
        products
        reactionEnergy
        activationEnergy
        dftCode
        dftFunctional
        pubId
        publication {
          title
          year
          doi
        }
        reactionSystems {
          name
          aseId
          energyCorrection
          systems {
            id
            uniqueId
            Formula
            energy
            fmax
            Cifdata
            Adsorbate
            Substrate
          }
        }
      }
    }
  }
}
'''
print(QUERY)
"""
        ),
        md(
            """
## 2. Fetch the remote JSON

This is a normal Python `requests.post(...)` GraphQL call. If you want a larger
dataset later, increase the `first:` limit or page through the API in batches.
"""
        ),
        code(
            """
response = requests.post(GRAPHQL_URL, json={"query": QUERY}, timeout=120)
response.raise_for_status()
payload = response.json()

if "errors" in payload:
    raise RuntimeError(payload["errors"])

payload.keys()
"""
        ),
        code(
            """
payload["data"]["reactions"]["totalCount"]
"""
        ),
        md(
            """
## 3. Flatten the GraphQL response into a local analysis table

Each Catalysis-Hub reaction can have several linked systems. For local HDF work
it is often more convenient to create one row per linked system while repeating
the important reaction metadata on every row.
"""
        ),
        code(
            """
def _safe_json_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


rows = []
reaction_edges = payload["data"]["reactions"]["edges"]
for edge in reaction_edges:
    reaction = edge["node"]
    publication = reaction.get("publication") or {}
    for reaction_system in reaction.get("reactionSystems") or []:
        system = reaction_system.get("systems") or {}
        rows.append(
            {
                "reaction_id": reaction.get("id"),
                "Equation": reaction.get("Equation"),
                "chemicalComposition": reaction.get("chemicalComposition"),
                "surfaceComposition": reaction.get("surfaceComposition"),
                "facet": reaction.get("facet"),
                "reactionEnergy": reaction.get("reactionEnergy"),
                "activationEnergy": reaction.get("activationEnergy"),
                "dftCode": reaction.get("dftCode"),
                "dftFunctional": reaction.get("dftFunctional"),
                "pubId": reaction.get("pubId"),
                "publication_title": publication.get("title"),
                "publication_year": publication.get("year"),
                "publication_doi": publication.get("doi"),
                "sites": _safe_json_text(reaction.get("sites")),
                "reactants": _safe_json_text(reaction.get("reactants")),
                "products": _safe_json_text(reaction.get("products")),
                "reaction_system_name": reaction_system.get("name"),
                "reaction_system_aseId": reaction_system.get("aseId"),
                "energyCorrection": reaction_system.get("energyCorrection"),
                "system_id": system.get("id"),
                "system_uniqueId": system.get("uniqueId"),
                "Formula": system.get("Formula"),
                "E": system.get("energy"),
                "fmax": system.get("fmax"),
                "Cifdata": system.get("Cifdata"),
                "Adsorbate": system.get("Adsorbate"),
                "Substrate": system.get("Substrate"),
            }
        )

cathub = pd.DataFrame(rows)
cathub.head()
"""
        ),
        md(
            """
## 4. Reconstruct ASE structures where CIF data is available

Catalysis-Hub exposes `Cifdata` for systems. We use ASE to turn that text back
into an `Atoms` object so that later local descriptor workflows can operate on
the structures directly.
"""
        ),
        code(
            """
def cif_to_atoms(cif_text):
    text = "" if cif_text is None else str(cif_text).strip()
    if not text:
        return None
    try:
        return read(StringIO(text), format="cif")
    except Exception:
        return None


cathub["struc"] = cathub["Cifdata"].map(cif_to_atoms)
cathub["has_structure"] = cathub["struc"].map(lambda value: value is not None)
cathub["record_class"] = np.where(
    cathub["reaction_system_name"].astype(str).str.contains("TS|transition", case=False, na=False),
    "transition_state",
    "reaction_system",
)
cathub["Name"] = (
    "cathub-"
    + cathub["surfaceComposition"].astype(str).fillna("surface")
    + "-"
    + cathub["reaction_system_name"].astype(str).fillna("system")
    + "-"
    + cathub["reaction_system_aseId"].astype(str).fillna("ase")
)

cathub[["Name", "Formula", "E", "reactionEnergy", "activationEnergy", "has_structure"]].head(10)
"""
        ),
        md(
            """
## 5. Save a local HDF file

This is the key bridge into your local OnePiece Studio workflow. Once the remote
query is flattened into a DataFrame, it becomes just another local HDF source.
"""
        ),
        code(
            """
hdf_path = OUT / "catalysis_hub_co2_subset.hdf"
cathub.to_hdf(hdf_path, key="df", mode="w")
hdf_path
"""
        ),
        md(
            """
## 6. First local OnePiece-style analysis

Catalysis-Hub is reaction-centric, so the first analyses are most naturally:

- which surfaces appear most often,
- which reactions are most exergonic,
- where finite activation barriers are available,
- which linked structures are suitable for deeper local inspection.
"""
        ),
        code(
            """
summary = pd.Series(
    {
        "rows": len(cathub),
        "unique_reactions": cathub["reaction_id"].nunique(),
        "unique_surfaces": cathub["surfaceComposition"].nunique(),
        "rows_with_structure": int(cathub["has_structure"].sum()),
        "rows_with_barrier": int(pd.to_numeric(cathub["activationEnergy"], errors="coerce").notna().sum()),
    }
)
summary
"""
        ),
        code(
            """
surface_table = (
    cathub.groupby("surfaceComposition", dropna=False)
    .agg(
        rows=("reaction_id", "size"),
        mean_reaction_energy=("reactionEnergy", "mean"),
        mean_barrier=("activationEnergy", "mean"),
    )
    .sort_values("rows", ascending=False)
)
surface_table.head(15)
"""
        ),
        code(
            """
best_reactions = (
    cathub[["Equation", "surfaceComposition", "facet", "reactionEnergy", "activationEnergy", "publication_year"]]
    .drop_duplicates()
    .sort_values("reactionEnergy")
)
best_reactions.head(20)
"""
        ),
        code(
            """
fig, ax = plt.subplots()
surface_table.head(12)["rows"].sort_values().plot.barh(ax=ax, color="#4c78a8")
ax.set_xlabel("number of linked reaction-system rows")
ax.set_ylabel("surface composition")
ax.set_title("Most represented surfaces in the fetched Catalysis-Hub subset")
plt.tight_layout()
plt.show()
"""
        ),
        code(
            """
plot_df = cathub[["reactionEnergy", "activationEnergy", "surfaceComposition"]].copy()
plot_df["reactionEnergy"] = pd.to_numeric(plot_df["reactionEnergy"], errors="coerce")
plot_df["activationEnergy"] = pd.to_numeric(plot_df["activationEnergy"], errors="coerce")
plot_df = plot_df.dropna(subset=["reactionEnergy", "activationEnergy"])

fig, ax = plt.subplots()
ax.scatter(plot_df["reactionEnergy"], plot_df["activationEnergy"], s=70, alpha=0.75, color="#f58518")
ax.set_xlabel("reaction energy / eV")
ax.set_ylabel("activation energy / eV")
ax.set_title("Barrier-energy landscape in the Catalysis-Hub subset")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
## 7. Add local reaction-network annotations

Even though the upstream source is reaction-centric, we can still enrich the
rows with our local OnePiece reaction-network logic. This is especially useful when
names contain state-like labels or when you later merge these rows with local
trajectory or slab calculations.
"""
        ),
        code(
            """
cathub_network = annotate_reaction_network(cathub)
cathub_network[["Name", "reaction_state", "reaction_family", "reaction_network_role"]].head(12)
"""
        ),
        md(
            """
## 8. Optional structure-descriptor pass

This pass only adds useful descriptor columns to rows where an ASE structure is
actually available. Catalysis-Hub often links reaction systems and structures,
but the completeness depends on the queried subset.
"""
        ),
        code(
            """
descriptor_input = cathub_network.copy()
descriptor_input["surface_ref_name"] = np.where(
    descriptor_input["reaction_system_name"].astype(str).eq("star"),
    descriptor_input["Name"],
    np.nan,
)
descriptor_input["surface_ref_E"] = np.where(
    descriptor_input["reaction_system_name"].astype(str).eq("star"),
    descriptor_input["E"],
    np.nan,
)

descriptor_frame = add_structure_descriptors(descriptor_input)
descriptor_frame[
    ["Name", "reaction_system_name", "Formula", "has_structure", "cell_volume", "adsorbate_formula", "adsorbate_atom_count"]
].head(15)
"""
        ),
        md(
            """
## 9. Curation pass for local reuse

Before reusing the downloaded HDF in OnePiece Studio, it is helpful to flag rows that are
missing energies, structures, or convergence information.
"""
        ),
        code(
            """
curated = apply_curation_rules(
    cathub_network,
    static_fmax_max=0.10,
    copt_fmax_max=0.15,
    exclude_name_tokens=["failed", "broken"],
    action="mark_review",
)

curated["curation_status"].value_counts(dropna=False)
"""
        ),
        code(
            """
curated_hdf = OUT / "catalysis_hub_co2_subset_curated.hdf"
curated.to_hdf(curated_hdf, key="df", mode="w")
curated_hdf
"""
        ),
        md(
            """
## 10. What we achieved

At this point we have:

- fetched a real subset from the Catalysis-Hub GraphQL API,
- converted it into a local DataFrame,
- reconstructed ASE structures where possible,
- written a local HDF file,
- performed first-pass catalysis analysis with a OnePiece-compatible workflow.

This local HDF can now be loaded into OnePiece Studio exactly like your own project HDFs.
"""
        ),
    ]
    return notebook("Catalysis-Hub to Local HDF with OnePiece-Compatible Analysis", cells)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "01_catalysis_hub_onepiece_analysis.ipynb"
    with path.open("w", encoding="utf-8") as handle:
        nbf.write(make_notebook(), handle)
    readme = OUT / "README.md"
    readme.write_text(
        "# Catalysis-Hub Tutorial\n\n"
        "- `01_catalysis_hub_onepiece_analysis.ipynb` fetches a Catalysis-Hub subset,\n"
        "  converts it to a local HDF file, and analyzes it with OnePiece Studio-compatible tools.\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
