from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib
import numpy as np
from ase import Atoms

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from onepiece import (
    ChgcarData,
    DoscarData,
    chgcar_cumulative_axis_profile,
    chgcar_line_profile,
    chgcar_planar_average,
    chgcar_plane_integrated_electrons,
    chgcar_to_xarray,
    doscar_orbital_band_center,
    doscar_to_xarray,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "reports" / "xarray_vasp_homework"
ASSETS_DIR = REPORT_DIR / "assets"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    chgcar = _synthetic_chgcar()
    doscar = _synthetic_doscar()
    chg_ds = chgcar_to_xarray(chgcar)
    dos_ds = doscar_to_xarray(doscar)

    figures = _build_figures(chg_ds, dos_ds)
    markdown = _build_markdown(figures)
    latex = _build_latex(figures)

    (REPORT_DIR / "xarray_vasp_homework.md").write_text(markdown)
    (REPORT_DIR / "xarray_vasp_homework.tex").write_text(latex)


def _synthetic_chgcar() -> ChgcarData:
    atoms = Atoms(
        "Cu2COH2",
        positions=[
            [1.5, 1.5, 0.7],
            [4.5, 1.5, 0.7],
            [3.0, 1.5, 2.4],
            [3.8, 1.5, 2.8],
            [2.3, 1.1, 3.0],
            [2.3, 1.9, 3.0],
        ],
        cell=[[6.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 6.0]],
        pbc=True,
    )
    nx, ny, nz = 36, 18, 36
    x = np.linspace(0.0, 1.0, nx, endpoint=False) + 0.5 / nx
    y = np.linspace(0.0, 1.0, ny, endpoint=False) + 0.5 / ny
    z = np.linspace(0.0, 1.0, nz, endpoint=False) + 0.5 / nz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    density = 0.25 + 0.15 * np.cos(2 * np.pi * Z)
    spin = np.zeros_like(density)
    scaled_positions = atoms.get_scaled_positions(wrap=False)
    widths = {
        "Cu": (0.06, 0.08, 0.05, 0.12, 0.01),
        "C": (0.05, 0.05, 0.04, 0.18, -0.04),
        "O": (0.04, 0.05, 0.04, 0.16, -0.05),
        "H": (0.035, 0.04, 0.035, 0.08, 0.03),
    }
    for symbol, center in zip(atoms.get_chemical_symbols(), scaled_positions, strict=False):
        sx, sy, sz, amp, spin_amp = widths[symbol]
        dx = ((X - center[0] + 0.5) % 1.0) - 0.5
        dy = ((Y - center[1] + 0.5) % 1.0) - 0.5
        dz = ((Z - center[2] + 0.5) % 1.0) - 0.5
        gaussian = np.exp(-0.5 * ((dx / sx) ** 2 + (dy / sy) ** 2 + (dz / sz) ** 2))
        density += amp * gaussian
        spin += spin_amp * gaussian

    voxel_volume = float(atoms.get_volume() / density.size)
    return ChgcarData(
        atoms=atoms,
        charge_density=density.astype(float),
        spin_density=spin.astype(float),
        voxel_volume=voxel_volume,
        source_path="synthetic/CuGa_CHGCAR",
    )


def _synthetic_doscar() -> DoscarData:
    energies = np.linspace(-8.0, 4.0, 500)
    total = (
        1.7 * np.exp(-0.5 * ((energies + 2.3) / 0.9) ** 2)
        + 1.2 * np.exp(-0.5 * ((energies + 0.8) / 0.6) ** 2)
        + 0.3 * np.exp(-0.5 * ((energies - 1.1) / 0.5) ** 2)
    )
    integrated = np.concatenate(
        [[0.0], np.cumsum(0.5 * (total[1:] + total[:-1]) * np.diff(energies))]
    )

    site_dos = np.zeros((4, 4, energies.size), dtype=float)
    site_dos[:, 0, :] = energies
    site_dos[0, 3, :] = 0.9 * np.exp(-0.5 * ((energies + 2.0) / 0.9) ** 2)
    site_dos[1, 3, :] = 1.1 * np.exp(-0.5 * ((energies + 1.6) / 0.8) ** 2)
    site_dos[2, 3, :] = 0.6 * np.exp(-0.5 * ((energies + 0.2) / 0.5) ** 2)
    site_dos[3, 3, :] = 0.25 * np.exp(-0.5 * ((energies - 0.4) / 0.6) ** 2)
    site_dos[0, 1, :] = 0.10 * np.exp(-0.5 * ((energies + 4.5) / 1.2) ** 2)
    site_dos[1, 1, :] = 0.10 * np.exp(-0.5 * ((energies + 4.2) / 1.0) ** 2)
    site_dos[2, 2, :] = 0.35 * np.exp(-0.5 * ((energies + 1.0) / 0.7) ** 2)
    site_dos[3, 2, :] = 0.25 * np.exp(-0.5 * ((energies - 0.2) / 0.7) ** 2)

    return DoscarData(
        energies=energies,
        total_dos=total[np.newaxis, :],
        integrated_total_dos=integrated[np.newaxis, :],
        site_dos=site_dos,
        efermi=0.0,
        source_path="synthetic/CuGa_DOSCAR",
        orbital_columns={"s": 1, "p": 2, "d": 3},
    )


def _build_figures(chg_ds, dos_ds):
    figures = []
    z_index = chg_ds.sizes["z"] // 2
    charge_slice = chg_ds["charge_density"].isel(z=z_index)
    spin_slice = chg_ds["spin_density"].isel(z=z_index)
    planar = chgcar_planar_average(chg_ds, axis="z")
    plane_electrons = chgcar_plane_integrated_electrons(chg_ds, axis="z")
    cumulative = chgcar_cumulative_axis_profile(chg_ds, axis="z")
    line = chgcar_line_profile(
        chg_ds,
        start_frac=(0.15, 0.5, 0.15),
        stop_frac=(0.7, 0.5, 0.62),
        n_points=240,
    )

    figures.append(
        _save_plot(
            "page_01_charge_density_slice.png",
            "Charge density slice from CHGCAR as a labeled xarray field",
            _plot_charge_slice,
            charge_slice,
            "viridis",
            "Charge density (e/Å$^3$)",
        )
    )
    figures.append(
        _save_plot(
            "page_02_spin_density_slice.png",
            "Spin-density slice from the same grid using an additional data variable",
            _plot_charge_slice,
            spin_slice,
            "coolwarm",
            "Spin density (arb. units)",
        )
    )
    figures.append(
        _save_plot(
            "page_03_planar_average.png",
            "Planar average along z computed by averaging over x and y",
            _plot_line,
            planar["z"].to_numpy(),
            planar.to_numpy(),
            "Fractional z",
            "Mean charge density (arb. units)",
        )
    )
    figures.append(
        _save_plot(
            "page_04_plane_electrons.png",
            "Electrons per z-slice from density summation times voxel volume",
            _plot_line,
            plane_electrons["z"].to_numpy(),
            plane_electrons.to_numpy(),
            "Fractional z",
            "Electrons per slice",
        )
    )
    figures.append(
        _save_plot(
            "page_05_cumulative_profile.png",
            "Cumulative electron profile that tracks where charge accumulates through the slab normal",
            _plot_line,
            cumulative["z"].to_numpy(),
            cumulative.to_numpy(),
            "Fractional z",
            "Cumulative electrons",
        )
    )
    figures.append(
        _save_plot(
            "page_06_line_profile.png",
            "Line interpolation through the 3D CHGCAR grid in fractional coordinates",
            _plot_line,
            line["distance_A"].to_numpy(),
            line.to_numpy(),
            "Distance along path (Å)",
            "Interpolated charge density",
        )
    )
    figures.append(
        _save_plot(
            "page_07_total_dos.png",
            "Total DOS with an energy coordinate referenced to the Fermi level",
            _plot_total_dos,
            dos_ds["energy"].to_numpy(),
            dos_ds["total_dos"].sel(spin="total").to_numpy(),
        )
    )
    figures.append(
        _save_plot(
            "page_08_integrated_total_dos.png",
            "Integrated total DOS as an energy-ordered cumulative observable",
            _plot_line,
            dos_ds["energy"].to_numpy(),
            dos_ds["integrated_total_dos"].sel(spin="total").to_numpy(),
            "Energy - E_F (eV)",
            "Integrated DOS",
        )
    )
    figures.append(
        _save_plot(
            "page_09_atom_resolved_d_pdos.png",
            "Atom-resolved d-projected DOS extracted from the orbital dimension",
            _plot_atom_resolved_d_pdos,
            dos_ds,
        )
    )

    centers = [
        float(doscar_orbital_band_center(dos_ds, atom_indices=[idx], orbitals=["d"], energy_window=(-6.0, 1.5)))
        for idx in range(4)
    ]
    figures.append(
        _save_plot(
            "page_10_band_centers.png",
            "d-band centers from xarray-selected atom and orbital subspaces",
            _plot_band_centers,
            centers,
        )
    )
    return figures


def _save_plot(filename, caption, plot_fn, *args):
    path = ASSETS_DIR / filename
    fig = plt.figure(figsize=(8.3, 5.2))
    ax = fig.add_subplot(111)
    plot_fn(ax, *args)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"filename": filename, "path": path, "caption": caption}


def _plot_charge_slice(ax, data_array, cmap, colorbar_label):
    image = ax.imshow(data_array.T, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    plt.colorbar(image, ax=ax, label=colorbar_label)


def _plot_line(ax, x, y, xlabel, ylabel):
    ax.plot(x, y, color="#1f77b4", linewidth=2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def _plot_total_dos(ax, x, y):
    ax.plot(x, y, color="#2ca02c", linewidth=2.0)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Energy - E_F (eV)")
    ax.set_ylabel("DOS")
    ax.grid(alpha=0.25)


def _plot_atom_resolved_d_pdos(ax, dataset):
    energy = dataset["energy"].to_numpy()
    for atom in dataset["atom"].to_numpy():
        signal = dataset["site_projected_dos"].sel(atom=atom, orbital="d").to_numpy()
        ax.plot(energy, signal, linewidth=1.8, label=f"atom {int(atom)}")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Energy - E_F (eV)")
    ax.set_ylabel("Projected DOS")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)


def _plot_band_centers(ax, centers):
    labels = ["atom 0", "atom 1", "atom 2", "atom 3"]
    ax.bar(labels, centers, color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759"])
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("d-band center (eV)")


def _build_markdown(figures) -> str:
    pages = [
        (
            "1. Why xarray is useful for CHGCAR",
            "The CHGCAR reader already gives us a 3D numerical field, but xarray adds labels for each axis. "
            "That means the charge density is no longer just an anonymous ndarray. It becomes a data object with "
            "named dimensions x, y, and z, coordinate values at voxel centers, and metadata for the cell and atomic positions.",
        ),
        (
            "2. Multiple variables in one dataset",
            "The same CHGCAR-derived dataset can hold both charge density and spin density. "
            "This is an important difference from a plain DataFrame because both variables share the same 3D coordinates and can be sliced consistently.",
        ),
        (
            "3. Planar averages",
            "A common surface-science task is to collapse a 3D field along the slab normal. "
            "With xarray, we can average over named dimensions and keep the remaining coordinate automatically.",
        ),
        (
            "4. Charge per slice",
            "A second useful reduction is the electron content of each z slice. "
            "This is obtained from the density sum multiplied by voxel volume and is a better physical quantity than a raw array sum.",
        ),
        (
            "5. Cumulative electron profile",
            "The cumulative profile is useful when we want to ask where electrons accumulate relative to the metal surface and adsorbate region.",
        ),
        (
            "6. Interpolated line profile",
            "Not every scientifically meaningful path is parallel to a grid axis. "
            "xarray interpolation lets us sample a path through the 3D field while keeping the coordinate bookkeeping explicit.",
        ),
        (
            "7. DOSCAR as a labeled energy dataset",
            "DOSCAR data are naturally organized along an energy axis. "
            "Representing total DOS with xarray makes energy-window operations and Fermi-level alignment easier to reason about.",
        ),
        (
            "8. Integrated DOS",
            "Because the energy coordinate is explicit, the integrated total DOS is easy to plot and compare with selected energy windows.",
        ),
        (
            "9. Atom- and orbital-resolved PDOS",
            "Projected DOS is where xarray becomes especially useful. "
            "The data carry atom, orbital, and energy labels simultaneously, so selection stays readable and less error-prone.",
        ),
        (
            "10. d-band centers from selected subspaces",
            "OnePiece can now calculate orbital band centers from xarray-selected PDOS subspaces. "
            "This is a practical bridge from raw DOSCAR data to catalysis descriptors that experimental collaborators may want to correlate with trends.",
        ),
    ]

    lines = [
        "# Xarray For CHGCAR And DOSCAR In OnePiece",
        "",
        "This homework-style note shows how `xarray` is used in OnePiece to turn VASP CHGCAR and DOSCAR data into labeled scientific datasets.",
        "",
        "The figures below were generated from the backend xarray functions added in version `0.7.0`.",
        "",
    ]
    for (title, text), figure in zip(pages, figures, strict=False):
        lines.extend(
            [
                f"## {title}",
                "",
                text,
                "",
                f"![{title}]({figure['path']})",
                "",
                f"*Figure:* {figure['caption']}",
                "",
            ]
        )
    return "\n".join(lines)


def _build_latex(figures) -> str:
    sections = [
        (
            "Why xarray is useful for CHGCAR",
            "The CHGCAR reader gives a three-dimensional scalar field. In OnePiece, xarray adds named axes, coordinate values at voxel centers, and metadata describing the simulation cell and atom positions. This makes later reductions and selections much easier to read and to validate.",
        ),
        (
            "Multiple variables in one dataset",
            "Charge density and spin density share the same grid. xarray keeps both variables in one dataset with the same coordinates, which is more natural than carrying several unrelated ndarrays.",
        ),
        (
            "Planar averages",
            "A planar average is a named-dimension reduction over x and y. This is a common slab-analysis step because the remaining z coordinate follows the surface normal.",
        ),
        (
            "Charge per slice",
            "Summing density over a plane and multiplying by voxel volume yields electrons per slice. The xarray-based helper keeps the axis label and therefore remains self-explanatory in later code.",
        ),
        (
            "Cumulative electron profile",
            "The cumulative profile integrates the slice electrons along z. It is useful for seeing how much electronic population lies below or above an interfacial region.",
        ),
        (
            "Interpolated line profile",
            "A catalytic analysis often needs a profile along a bond axis or through an adsorption site. The line-profile helper samples the 3D field between two fractional points and returns both the sampled values and the traveled distance in angstrom.",
        ),
        (
            "DOSCAR as a labeled energy dataset",
            "For DOSCAR, xarray treats energy as the primary scientific coordinate. Total DOS and integrated DOS become energy-indexed observables instead of anonymous columns.",
        ),
        (
            "Integrated DOS",
            "The integrated total DOS remains useful as a sanity check because it reflects the cumulative state count. The explicit energy coordinate also makes energy-window selection transparent.",
        ),
        (
            "Atom- and orbital-resolved PDOS",
            "Projected DOS adds atom and orbital dimensions. This is where xarray is especially convincing because atom and orbital selections stay readable and composable.",
        ),
        (
            "d-band centers from selected subspaces",
            "A final practical descriptor is the d-band center. In OnePiece, it is derived from an xarray-selected PDOS subspace, which makes the calculation traceable: first choose atom and orbital labels, then integrate over the chosen energy range.",
        ),
    ]
    body = []
    for (title, text), figure in zip(sections, figures, strict=False):
        body.append(
            dedent(
                f"""
                \\section*{{{title}}}
                {text}

                \\begin{{figure}}[h]
                \\centering
                \\includegraphics[width=0.92\\textwidth]{{assets/{figure["filename"]}}}
                \\caption*{{{figure["caption"]}}}
                \\end{{figure}}
                \\clearpage
                """
            ).strip()
        )
    body_text = "\n\n".join(body)
    return dedent(
        f"""
        \\documentclass[11pt,a4paper]{{article}}
        \\usepackage[margin=2.2cm]{{geometry}}
        \\usepackage{{graphicx}}
        \\usepackage{{float}}
        \\usepackage{{parskip}}
        \\usepackage{{amsmath}}
        \\title{{Xarray for CHGCAR and DOSCAR in OnePiece}}
        \\author{{Claude Coppex}}
        \\date{{June 8, 2026}}
        \\begin{{document}}
        \\maketitle
        This homework-style report explains how \\texttt{{xarray}} is used inside OnePiece to analyze labeled three-dimensional charge-density fields and energy-resolved density-of-states data from VASP. The emphasis is on scientifically meaningful labeled reductions rather than on general plotting alone.
        \\clearpage
        {body_text}
        \\end{{document}}
        """
    ).strip() + "\n"


if __name__ == "__main__":
    main()
