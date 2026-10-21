import numpy as np
import pytest

from src.gvector import GVector
from src.kpoint_scf import KPointSCF
from src.lattice import Lattice
from src.smearing import FermiDirac, Gaussian, MethfesselPaxton


def make_two_level_solver():
    lattice = Lattice.cubic(2 * np.pi)
    gvec = GVector(lattice, 1.1)
    solver = KPointSCF(
        lattice, gvec, lattice.volume, 2, 2, np.zeros(gvec.npw, complex)
    )
    solver.setup_kpoints(1, 1, 1, use_symmetry=False)
    kdata = solver.kpoints[0]
    kdata.evals = np.array([0.0, 0.2])
    kdata.evecs = np.zeros((solver.npw, 2), dtype=complex)
    kdata.evecs[0, 0] = 1
    kdata.evecs[1, 1] = 1
    kdata.kg_squared = np.zeros(solver.npw)
    solver.rho_r = np.ones(solver.fft_shape)
    solver.rho_g = np.zeros(solver.density_gvec.npw, dtype=complex)
    return solver, kdata


def test_fermi_dirac_total_energy_matches_two_level_free_energy(capsys):
    solver, kdata = make_two_level_solver()
    temperature = 10000.0
    solver._update_occupations("fd", temperature=temperature)
    kbt = FermiDirac(temperature).kbt

    # One level at 0 and one at delta have mu=delta/2 by symmetry.
    delta = kdata.evals[1] - kdata.evals[0]
    f_lower = 1.0 / (1.0 + np.exp(-delta / (2 * kbt)))
    f_upper = 1.0 - f_lower
    np.testing.assert_allclose(kdata.occupations, 2 * np.array([f_lower, f_upper]))
    analytic_internal = 2 * delta * f_upper
    binary_entropy = -(
        f_lower * np.log(f_lower) + (1 - f_lower) * np.log(1 - f_lower)
    )
    analytic_correction = -4 * kbt * binary_entropy

    total = solver._compute_total_energy()

    assert solver.energy_label == "Helmholtz free energy"
    assert solver.internal_energy is not None
    assert solver.smearing_correction == pytest.approx(analytic_correction)
    assert solver.free_energy == total
    assert total == pytest.approx(solver.internal_energy + analytic_correction)
    assert analytic_internal + analytic_correction == pytest.approx(
        2 * delta - 4 * kbt * np.log1p(np.exp(delta / (2 * kbt)))
    )
    solver.print_results()
    output = capsys.readouterr().out
    assert "Helmholtz free energy:" in output
    assert "Internal energy:" in output
    assert "Smearing correction:" in output


@pytest.mark.parametrize(
    "smearing",
    [
        FermiDirac(temperature=1e-5),
        Gaussian(sigma=1e-12),
        MethfesselPaxton(sigma=1e-12, order=1),
        MethfesselPaxton(sigma=1e-12, order=2),
    ],
)
def test_smearing_correction_vanishes_as_width_goes_to_zero(smearing):
    correction = np.sum(smearing.energy_correction([0.0], 0.0))
    assert abs(correction) < 1e-9


@pytest.mark.parametrize(
    "scheme,label",
    [
        ("fd", "Helmholtz free energy"),
        ("gaussian", "Smearing-corrected energy"),
        ("mp", "Smearing-corrected energy"),
        ("fixed", "Internal energy"),
    ],
)
def test_scf_labels_energy_objective_for_each_occupation_scheme(scheme, label):
    solver, _ = make_two_level_solver()

    solver._update_occupations(scheme, temperature=10000.0, sigma=0.05)
    total = solver._compute_total_energy()

    assert solver.energy_label == label
    assert np.isfinite(solver.smearing_correction)
    assert total == pytest.approx(solver.internal_energy + solver.smearing_correction)
    if scheme == "fd":
        assert solver.free_energy == total
    else:
        assert solver.free_energy is None


@pytest.mark.parametrize("order", [1, 2])
def test_methfessel_paxton_correction_is_consistent_with_occupations(order):
    smearing = MethfesselPaxton(sigma=0.3, order=order)
    x = np.array([-0.7, 0.2, 1.1])
    step = 1e-6
    energies = x * smearing.sigma
    derivative = (
        smearing.energy_correction(energies + step * smearing.sigma, 0)
        - smearing.energy_correction(energies - step * smearing.sigma, 0)
    ) / (2 * step)
    occupation_derivative = (
        smearing.occupation_array(energies + step * smearing.sigma, 0)
        - smearing.occupation_array(energies - step * smearing.sigma, 0)
    ) / (2 * step)

    np.testing.assert_allclose(
        derivative, -smearing.sigma * x * occupation_derivative, atol=2e-9
    )
