"""K-point result reporting must follow occupations and SCF state."""
import numpy as np
import pytest

from src.gvector import GVector
from src.kpoint_scf import KPointSCF
from src.lattice import Lattice
from src.constants import HA_TO_EV


def result_solver(n_electrons, occupations):
    lattice = Lattice.cubic(2 * np.pi)
    gvec = GVector(lattice, 1.1)
    solver = KPointSCF(
        lattice, gvec, lattice.volume, n_electrons, len(occupations),
        np.zeros(gvec.npw, dtype=complex))
    solver.setup_kpoints(1, 1, 1, use_symmetry=False)
    kdata = solver.kpoints[0]
    kdata.evals = np.array([-0.5, 0.25, 0.5, 1.0])[:len(occupations)]
    kdata.occupations = np.asarray(occupations, dtype=float)
    solver.total_energy = 0.0
    solver.energy_label = "Internal energy"
    solver.internal_energy = 0.0
    solver.smearing_correction = 0.0
    return solver


def test_print_results_uses_occupations_for_float_electron_count(capsys):
    solver = result_solver(2.0, [2.0, 0.0, 0.0, 0.0])

    gap = solver.print_results()

    assert gap == pytest.approx(0.75 * HA_TO_EV, abs=1e-8)
    assert "Band gap:" in capsys.readouterr().out


def test_print_results_reports_no_empty_bands(capsys):
    solver = result_solver(8, [2.0, 2.0, 2.0, 2.0])

    assert solver.print_results() is None
    assert "there are no empty bands" in capsys.readouterr().out


def test_print_results_does_not_report_a_gap_for_partial_occupations(capsys):
    solver = result_solver(1.5, [1.5, 0.0, 0.0, 0.0])

    assert solver.print_results() is None
    assert "partially occupied bands" in capsys.readouterr().out


def test_print_results_before_run_has_clear_error():
    lattice = Lattice.cubic(2 * np.pi)
    gvec = GVector(lattice, 1.1)
    solver = KPointSCF(lattice, gvec, lattice.volume, 2.0, 2,
                       np.zeros(gvec.npw, dtype=complex))

    with pytest.raises(RuntimeError, match=r"call run\(\) first"):
        solver.print_results()
