from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp.vasp_auxiliary import VaspChargeDensity
from ase.io import write

import onepiece.dftdataframe_import as import_mod
from onepiece.dftdataframe_import import (
    add_input_parameter_checks,
    crawl_root_to_frame,
    crawl_root_to_hdf,
    enrich_electronic_summaries,
)


def test_crawl_root_to_frame_builds_onepiece_dataframe_from_final_traj(tmp_path: Path) -> None:
    root = tmp_path / "calcs"
    calc_a = root / "row-a"
    calc_b = root / "row-b"
    calc_a.mkdir(parents=True)
    calc_b.mkdir(parents=True)

    _write_traj(
        calc_a / "final.traj",
        Atoms("Cu2", positions=[(0, 0, 0), (1.8, 0, 0)]),
        energy=-10.0,
        forces=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=float),
    )
    _write_traj(
        calc_b / "final.traj",
        Atoms("Cu4O", positions=[(0, 0, 0), (1.8, 0, 0), (0, 1.8, 0), (1.8, 1.8, 0), (0.9, 0.9, 1.2)]),
        energy=-25.0,
        forces=np.array(
            [
                [0.0, 0.0, 0.02],
                [0.0, 0.0, -0.02],
                [0.0, 0.01, 0.0],
                [0.0, -0.01, 0.0],
                [0.0, 0.0, 0.03],
            ],
            dtype=float,
        ),
    )
    (calc_a / "out.txt").write_text(
        "\n".join(
            [
                "E_ZPE = 0.0100 eV",
                "S_vib = 0.1000 eV/K",
            ]
        )
    )
    (calc_b / "out.txt").write_text(
        "\n".join(
            [
                "E_ZPE = 0.0200 eV",
                "S_vib = 0.2000 eV/K",
            ]
        )
    )

    progress_events: list[tuple[int, int, str]] = []
    frame = crawl_root_to_frame(
        root,
        query="Cu == 2",
        verbose=True,
        progress_callback=lambda completed, total, path: progress_events.append((completed, total, path)),
    )

    assert frame["Name"].tolist() == ["row-a"]
    assert frame["Cu"].tolist() == [2.0]
    assert frame["Formula"].tolist() == ["Cu2"]
    assert np.isclose(frame.loc[frame.index[0], "E"], -10.0)
    assert np.isclose(frame.loc[frame.index[0], "fmax"], 0.01)
    assert frame["S_vib"].tolist() == [0.1]
    assert frame["E_ZPE"].tolist() == [0.01]
    assert frame["structure_source"].tolist() == ["final.traj"]
    assert frame["entropy_data_available"].tolist() == [True]
    assert frame["has_final_traj"].tolist() == [True]
    assert frame["has_outcar"].tolist() == [False]
    assert frame.index.name == "Name"
    assert frame.index.tolist() == ["row-a"]
    assert len(progress_events) == 2
    assert progress_events[0][0] == 1
    assert progress_events[-1][1] == 2


def test_crawl_root_to_hdf_writes_output(tmp_path: Path) -> None:
    calc_dir = tmp_path / "calcs" / "row-a"
    calc_dir.mkdir(parents=True)
    _write_traj(
        calc_dir / "final.traj",
        Atoms("Cu2", positions=[(0, 0, 0), (1.8, 0, 0)]),
        energy=-10.0,
        forces=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=float),
    )

    output = tmp_path / "crawled.hdf"
    returned = crawl_root_to_hdf(tmp_path / "calcs", output)

    assert returned == output
    loaded = pd.read_hdf(output, key="df")
    assert loaded["Cu"].tolist() == [2.0]
    assert loaded["Name"].tolist() == ["row-a"]


def test_crawl_root_to_frame_tolerates_missing_entropies_file(tmp_path: Path) -> None:
    calc_dir = tmp_path / "calcs" / "row-a"
    calc_dir.mkdir(parents=True)
    _write_traj(
        calc_dir / "final.traj",
        Atoms("Cu2", positions=[(0, 0, 0), (1.8, 0, 0)]),
        energy=-10.0,
        forces=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=float),
    )

    frame = crawl_root_to_frame(tmp_path / "calcs")

    assert frame["Name"].tolist() == ["row-a"]
    assert frame["entropy_source_file"].tolist() == [str(calc_dir / "out.txt")]
    assert frame["entropy_data_available"].tolist() == [False]


def test_crawl_root_to_frame_reads_per_folder_out_txt_thermochemistry(tmp_path: Path) -> None:
    calc_dir = tmp_path / "calcs" / "row-a"
    calc_dir.mkdir(parents=True)
    _write_traj(
        calc_dir / "final.traj",
        Atoms("Cu2", positions=[(0, 0, 0), (1.8, 0, 0)]),
        energy=-10.0,
        forces=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=float),
    )
    (calc_dir / "out.txt").write_text(
        "\n".join(
            [
                "E_ZPE                = 0.1200 eV",
                "Cv_trans (0 -> T)    = 0.0100 eV",
                "Cv_rot (0 -> T)      = 0.0200 eV",
                "Cv_vib (0 -> T)      = 0.0300 eV",
                "S_trans              = 0.0010 eV/K",
                "S_rot                = 0.0020 eV/K",
                "S_vib                = 0.0030 eV/K",
            ]
        )
    )

    frame = crawl_root_to_frame(tmp_path / "calcs")

    row = frame.iloc[0]
    assert bool(row["entropy_data_available"]) is True
    assert row["entropy_source_file"] == str(calc_dir / "out.txt")
    assert np.isclose(row["E_ZPE"], 0.12)
    assert np.isclose(row["Cv_trans"], 0.01)
    assert np.isclose(row["Cv_rot"], 0.02)
    assert np.isclose(row["Cv_vib"], 0.03)
    assert np.isclose(row["S_trans"], 0.001)
    assert np.isclose(row["S_rot"], 0.002)
    assert np.isclose(row["S_vib"], 0.003)


def test_crawl_root_to_frame_falls_back_to_contcar_when_final_traj_is_missing(tmp_path: Path) -> None:
    calc_dir = tmp_path / "calcs" / "row-a"
    calc_dir.mkdir(parents=True)
    atoms = Atoms("CuO", positions=[(0, 0, 0), (1.8, 0, 0.9)], cell=[5.0, 5.0, 8.0], pbc=True)
    write(calc_dir / "CONTCAR", atoms, format="vasp")

    frame = crawl_root_to_frame(tmp_path / "calcs")

    row = frame.iloc[0]
    assert row["Name"] == "row-a"
    assert row["Formula"] == "CuO"
    assert row["structure_source"] == "CONTCAR"
    assert bool(row["has_contcar"]) is True
    assert row["contcar_path"] == str(calc_dir / "CONTCAR")
    assert np.isclose(row["a"], 5.0)
    assert np.isclose(row["c"], 8.0)


def test_crawl_root_to_frame_reads_chgcar_and_doscar_summaries(tmp_path: Path) -> None:
    calc_dir = tmp_path / "calcs" / "row-a"
    calc_dir.mkdir(parents=True)
    atoms = Atoms(
        "HeBe",
        positions=[[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]],
        cell=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pbc=True,
    )
    _write_traj(
        calc_dir / "final.traj",
        atoms,
        energy=-10.0,
        forces=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=float),
    )
    _write_chgcar(calc_dir / "CHGCAR", atoms, np.array([[[1.0]], [[3.0]]], dtype=float))
    _write_synthetic_doscar(calc_dir / "DOSCAR")

    frame = crawl_root_to_frame(tmp_path / "calcs")

    row = frame.iloc[0]
    assert bool(row["chgcar_read_ok"]) is True
    assert row["chgcar_grid_shape"] == (2, 1, 1)
    assert np.isclose(row["chgcar_voxel_volume"], 1.0)
    assert np.isclose(row["chgcar_total_integrated_electrons"], 4.0)
    assert bool(row["doscar_read_ok"]) is True
    assert np.isclose(row["doscar_natoms"], 2.0)
    assert bool(row["doscar_spin_polarized"]) is False
    assert np.isclose(row["doscar_efermi"], 0.0)
    assert np.isclose(row["doscar_energy_min"], -2.0)
    assert np.isclose(row["doscar_energy_max"], 2.0)
    assert np.isclose(row["doscar_total_dos_below_ef"], 2.0)


def test_enrich_electronic_summaries_runs_as_separate_parallel_stage(tmp_path: Path) -> None:
    calc_dir = tmp_path / "calcs" / "row-a"
    calc_dir.mkdir(parents=True)
    atoms = Atoms(
        "HeBe",
        positions=[[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]],
        cell=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pbc=True,
    )
    _write_traj(
        calc_dir / "final.traj",
        atoms,
        energy=-10.0,
        forces=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=float),
    )
    _write_chgcar(calc_dir / "CHGCAR", atoms, np.array([[[1.0]], [[3.0]]], dtype=float))
    _write_synthetic_doscar(calc_dir / "DOSCAR")

    base = crawl_root_to_frame(tmp_path / "calcs", read_electronic_files=False)
    assert "chgcar_read_ok" not in base.columns

    progress_events: list[tuple[int, int, str]] = []
    enriched = enrich_electronic_summaries(
        base,
        workers=2,
        progress_callback=lambda completed, total, path: progress_events.append((completed, total, path)),
    )

    row = enriched.iloc[0]
    assert bool(row["chgcar_read_ok"]) is True
    assert bool(row["doscar_read_ok"]) is True
    assert len(progress_events) == 1
    assert progress_events[0][0] == 1
    assert progress_events[0][1] == 1


def test_crawl_root_to_frame_reads_incar_and_kpoints_inputs_early(tmp_path: Path) -> None:
    calc_dir = tmp_path / "calcs" / "row-a"
    calc_dir.mkdir(parents=True)
    atoms = Atoms("Cu2", positions=[(0, 0, 0), (1.8, 0, 0)], cell=[5.0, 5.0, 8.0], pbc=True)
    _write_traj(
        calc_dir / "final.traj",
        atoms,
        energy=-10.0,
        forces=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=float),
    )
    (calc_dir / "INCAR").write_text("ENCUT = 450\nISMEAR = 1\nSIGMA = 0.2\nPREC = Accurate\n")
    (calc_dir / "KPOINTS").write_text("kpoints\n0\nGamma\n4 4 1\n0 0 0\n")

    frame = crawl_root_to_frame(tmp_path / "calcs", read_electronic_files=False)

    row = frame.iloc[0]
    assert row["input_parameter_source"] == "INCAR"
    assert np.isclose(row["input_encut"], 450.0)
    assert np.isclose(row["input_ismear"], 1.0)
    assert np.isclose(row["input_sigma"], 0.2)
    assert row["input_prec"] == "Accurate"
    assert row["input_kpoints_mode"] == "gamma"
    assert row["input_kpoints_grid"] == (4, 4, 1)
    assert bool(row["input_kpoints_present"]) is True


def test_crawl_root_to_frame_reads_outcar_frequencies(tmp_path: Path) -> None:
    calc_dir = tmp_path / "calcs" / "row-a"
    calc_dir.mkdir(parents=True)
    atoms = Atoms("Cu2", positions=[(0, 0, 0), (1.8, 0, 0)], cell=[5.0, 5.0, 8.0], pbc=True)
    _write_traj(
        calc_dir / "final.traj",
        atoms,
        energy=-10.0,
        forces=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=float),
    )
    (calc_dir / "OUTCAR").write_text(
        "\n".join(
            [
                "  1 f  =   3.000 THz   18.0 2PiTHz   100.0 cm-1   12.0 meV",
                "  2 f/i=   1.500 THz    9.0 2PiTHz    50.0 cm-1    6.0 meV",
            ]
        )
    )

    frame = crawl_root_to_frame(tmp_path / "calcs", read_electronic_files=False)
    row = frame.iloc[0]

    assert row["frequency_source_file"] == str(calc_dir / "OUTCAR")
    assert row["frequencies_cm1"] == [100.0, -50.0]
    assert row["frequencies_mev"] == [12.0, -6.0]
    assert np.isclose(row["frequency_count"], 2.0)
    assert np.isclose(row["imaginary_frequency_count"], 1.0)
    assert np.isclose(row["lowest_frequency_cm1"], -50.0)


def test_crawl_root_to_frame_reuses_base_cache(tmp_path: Path, monkeypatch) -> None:
    calc_dir = tmp_path / "calcs" / "row-a"
    calc_dir.mkdir(parents=True)
    _write_traj(
        calc_dir / "final.traj",
        Atoms("Cu2", positions=[(0, 0, 0), (1.8, 0, 0)]),
        energy=-10.0,
        forces=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=float),
    )
    cache_dir = tmp_path / "cache"

    first = crawl_root_to_frame(tmp_path / "calcs", read_electronic_files=False, cache_dir=cache_dir)
    assert first.index.tolist() == ["row-a"]

    monkeypatch.setattr(import_mod, "_read_structure", lambda _path: (_ for _ in ()).throw(AssertionError("cache miss")))
    second = crawl_root_to_frame(tmp_path / "calcs", read_electronic_files=False, cache_dir=cache_dir)

    assert second.index.tolist() == ["row-a"]


def test_enrich_electronic_summaries_reuses_cache(tmp_path: Path, monkeypatch) -> None:
    calc_dir = tmp_path / "calcs" / "row-a"
    calc_dir.mkdir(parents=True)
    atoms = Atoms(
        "HeBe",
        positions=[[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]],
        cell=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pbc=True,
    )
    _write_traj(
        calc_dir / "final.traj",
        atoms,
        energy=-10.0,
        forces=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=float),
    )
    _write_chgcar(calc_dir / "CHGCAR", atoms, np.array([[[1.0]], [[3.0]]], dtype=float))
    _write_synthetic_doscar(calc_dir / "DOSCAR")

    base = crawl_root_to_frame(tmp_path / "calcs", read_electronic_files=False)
    cache_dir = tmp_path / "cache"
    enriched = enrich_electronic_summaries(base, cache_dir=cache_dir)
    assert bool(enriched.iloc[0]["chgcar_read_ok"]) is True

    monkeypatch.setattr(
        import_mod,
        "_electronic_summary_from_path",
        lambda _calc_dir, verbose=False: (_ for _ in ()).throw(AssertionError("electronic cache miss")),
    )
    cached = enrich_electronic_summaries(base, cache_dir=cache_dir)
    assert bool(cached.iloc[0]["doscar_read_ok"]) is True


def test_add_input_parameter_checks_marks_dataset_reference_consistency() -> None:
    frame = pd.DataFrame(
        {
            "input_encut": [450.0, 450.0, 400.0],
            "input_kpoints_grid": [(4, 4, 1), (4, 4, 1), (3, 3, 1)],
        }
    )

    checked = add_input_parameter_checks(frame)

    assert np.isclose(checked.loc[0, "encut_reference_value"], 450.0)
    assert checked.loc[0, "kpoints_reference_grid"] == "(4, 4, 1)"
    assert bool(checked.loc[0, "input_settings_ok"]) is True
    assert bool(checked.loc[2, "input_settings_ok"]) is False


def _write_traj(path: Path, atoms: Atoms, *, energy: float, forces: np.ndarray) -> None:
    atoms = atoms.copy()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    write(path, atoms)


def _write_chgcar(path: Path, atoms: Atoms, density: np.ndarray) -> None:
    writer = VaspChargeDensity(None)
    writer.atoms = [atoms]
    writer.chg = [density]
    writer.write(path, format="chgcar")


def _write_synthetic_doscar(path: Path) -> None:
    lines = [
        "2   generated DOSCAR\n",
        "header line 2\n",
        "header line 3\n",
        "header line 4\n",
        "header line 5\n",
        "  2.000000 -2.000000 5 0.000000 0.000000\n",
    ]
    energies = [-2.0, -1.0, 0.0, 1.0, 2.0]
    total = [0.0, 1.0, 2.0, 1.0, 0.0]
    integrated = [0.0, 0.5, 2.0, 3.5, 4.0]
    for energy, dos_value, int_value in zip(energies, total, integrated, strict=False):
        lines.append(f"{energy:10.6f} {dos_value:10.6f} {int_value:10.6f}\n")

    atom_blocks = [
        {
            "s": [0.0, 0.0, 0.0, 0.0, 0.0],
            "p": [0.0, 0.0, 0.0, 0.0, 0.0],
            "d": [0.0, 1.0, 2.0, 1.0, 0.0],
        },
        {
            "s": [0.0, 0.0, 0.0, 0.0, 0.0],
            "p": [0.0, 0.0, 0.0, 0.0, 0.0],
            "d": [0.0, 0.5, 1.0, 0.5, 0.0],
        },
    ]
    for block in atom_blocks:
        lines.append("  2.000000 -2.000000 5 0.000000 0.000000\n")
        for idx, energy in enumerate(energies):
            lines.append(
                f"{energy:10.6f} {block['s'][idx]:10.6f} {block['p'][idx]:10.6f} {block['d'][idx]:10.6f}\n"
            )

    path.write_text("".join(lines))
