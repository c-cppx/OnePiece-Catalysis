from __future__ import annotations

import numpy as np
from ase import Atoms

from onepiece import (
    ChgcarData,
    DoscarData,
    chgcar_cumulative_axis_profile,
    chgcar_line_profile,
    chgcar_planar_average,
    chgcar_plane_integrated_electrons,
    chgcar_to_xarray,
    doscar_integrated_pdos,
    doscar_orbital_band_center,
    doscar_select_energy_window,
    doscar_to_xarray,
)


def test_chgcar_to_xarray_exposes_labeled_grid_and_spin_density() -> None:
    atoms = Atoms(
        "HeBe",
        positions=[[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]],
        cell=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pbc=True,
    )
    chgcar = ChgcarData(
        atoms=atoms,
        charge_density=np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=float),
        spin_density=np.array([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=float),
        voxel_volume=0.5,
        source_path="synthetic/CHGCAR",
    )

    dataset = chgcar_to_xarray(chgcar)

    assert dataset["charge_density"].dims == ("x", "y", "z")
    assert dataset["spin_density"].dims == ("x", "y", "z")
    assert np.allclose(dataset["x"].to_numpy(), [0.25, 0.75])
    assert dataset.attrs["chemical_symbols"] == ["He", "Be"]


def test_chgcar_profiles_return_expected_planar_and_cumulative_quantities() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    chgcar = ChgcarData(
        atoms=atoms,
        charge_density=np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=float,
        ),
        voxel_volume=0.125,
        source_path="synthetic/CHGCAR",
    )

    planar = chgcar_planar_average(chgcar, axis="z")
    plane_electrons = chgcar_plane_integrated_electrons(chgcar, axis="z")
    cumulative = chgcar_cumulative_axis_profile(chgcar, axis="z")

    assert np.allclose(planar.to_numpy(), [4.0, 5.0])
    assert np.allclose(plane_electrons.to_numpy(), [2.0, 2.5])
    assert np.allclose(cumulative.to_numpy(), [2.0, 4.5])


def test_chgcar_line_profile_uses_fractional_interpolation() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    chgcar = ChgcarData(
        atoms=atoms,
        charge_density=np.array([[[1.0]], [[3.0]]], dtype=float),
        voxel_volume=1.0,
        source_path="synthetic/CHGCAR",
    )

    profile = chgcar_line_profile(
        chgcar,
        start_frac=(0.25, 0.5, 0.5),
        stop_frac=(0.75, 0.5, 0.5),
        n_points=3,
    )

    assert np.allclose(profile.to_numpy(), [1.0, 2.0, 3.0])
    assert np.allclose(profile["distance_A"].to_numpy(), [0.0, 0.25, 0.5])


def test_doscar_to_xarray_exposes_energy_atom_and_orbital_dimensions() -> None:
    doscar = _synthetic_doscar()

    dataset = doscar_to_xarray(doscar)

    assert dataset["total_dos"].dims == ("spin", "energy")
    assert dataset["site_projected_dos"].dims == ("atom", "orbital", "energy")
    assert dataset["orbital"].to_numpy().tolist() == ["s", "p", "d"]
    assert dataset.attrs["spin_polarized"] is False


def test_doscar_xarray_helpers_integrate_and_locate_band_center() -> None:
    doscar = _synthetic_doscar()
    dataset = doscar_to_xarray(doscar)

    window = doscar_select_energy_window(dataset, energy_window=(-1.0, 1.0))
    integrated = doscar_integrated_pdos(
        dataset,
        atom_indices=[0],
        orbitals=["d"],
        energy_window=(-1.0, 1.0),
    )
    center = doscar_orbital_band_center(
        dataset,
        atom_indices=[0],
        orbitals=["d"],
        energy_window=(-1.0, 1.0),
    )

    assert np.allclose(window["energy"].to_numpy(), [-1.0, 0.0, 1.0])
    assert np.isclose(float(integrated), 3.0)
    assert np.isclose(float(center), 0.0)


def _synthetic_doscar() -> DoscarData:
    energies = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    site_dos = np.zeros((2, 4, 5), dtype=float)
    site_dos[:, 0, :] = energies
    site_dos[0, 3, :] = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=float)
    site_dos[1, 3, :] = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=float)
    return DoscarData(
        energies=energies,
        total_dos=np.array([[0.0, 1.0, 2.0, 1.0, 0.0]], dtype=float),
        integrated_total_dos=np.array([[0.0, 0.5, 2.0, 3.5, 4.0]], dtype=float),
        site_dos=site_dos,
        efermi=0.0,
        source_path="synthetic/DOSCAR",
        orbital_columns={"s": 1, "p": 2, "d": 3},
    )
