"""Output-state and convergence invariants; see book SCF and energy chapters."""
import numpy as np
import pytest
from src.lattice import Lattice
from src.gvector import GVector
from src.scf import SCFSolver
from src.kpoint_scf import KPointSCF
from src.hamiltonian import g_to_r
from src.hartree import compute_hartree_energy
from src.xc import lda_xc


def make_solver():
    lattice = Lattice.cubic(2*np.pi)
    g = GVector(lattice, 1.1)
    v = np.zeros(g.npw, complex)
    v[np.abs(g.miller[:, 0]) == 1] = -.3
    s = KPointSCF(lattice, g, lattice.volume, 2, 2, v)
    s.setup_kpoints(1, 1, 1)
    # Independent exact solve keeps this test about SCF bookkeeping.
    def exact_solve(kdata):
        identity = np.eye(s.npw, dtype=complex)
        matrix = np.column_stack([s._apply_hamiltonian(c, kdata) for c in identity.T])
        values, vectors = np.linalg.eigh(matrix)
        kdata.evals = values[:s.n_bands]
        kdata.evecs = vectors[:, :s.n_bands]
    s._solve_eigenvalue = exact_solve
    return s


def test_scf_first_iteration_energy_matches_output_orbitals():
    s = make_solver()
    with pytest.warns(RuntimeWarning, match='did not converge'):
        energy = s.run(max_iter=1, verbose=False, smearing='fixed')
    assert not s.converged and s.iterations == 1
    k = s.kpoints[0]
    rho = s._compute_density()
    np.testing.assert_allclose(s.rho_r, rho, atol=1e-14)
    np.testing.assert_allclose(s._g_to_r(s.rho_g), rho, atol=1e-14)
    # Direct energy, independent of the solver's eigenvalue-sum bookkeeping.
    kinetic = sum(occ*np.vdot(k.evecs[:, n], k.kg_squared*k.evecs[:, n]).real
                  for n, occ in enumerate(k.occupations))
    v = np.fft.ifftn(s.gvec.map_to_fft_grid(s.vloc_g, s.fft_shape)).real*s.n_fft
    _, exc = lda_xc(rho)
    direct = kinetic + np.sum(rho*(v+exc))*s.volume/s.n_fft
    direct += compute_hartree_energy(s._r_to_g(rho), s.density_gvec.norms, s.volume)
    assert energy == pytest.approx(direct, abs=2e-13)


def test_scf_density_retains_doubled_frequency():
    s = SCFSolver(Lattice.cubic(2*np.pi), 5, 1, 2)
    s.evecs = np.zeros((s.npw, 1), complex)
    s.evecs[np.all(s.gvec.miller == [3, 0, 0], axis=1), 0] = 1/np.sqrt(2)
    s.evecs[np.all(s.gvec.miller == [-3, 0, 0], axis=1), 0] = 1/np.sqrt(2)
    rho_g = s._compute_density()
    mode = np.all(s.density_gvec.miller == [6, 0, 0], axis=1)
    assert rho_g[mode][0] == pytest.approx(1/s.volume, abs=1e-14)
    rho = 2*np.abs(g_to_r(s.evecs[:, 0], s.gvec, s.fft_shape, s.volume))**2
    np.testing.assert_allclose(s._g_to_r_density(rho_g), rho, atol=1e-14)


def test_gamma_jellium_converges_and_invalid_capacity_rejected():
    np.random.seed(9)
    s = SCFSolver(Lattice.cubic(4), 1.5, 2, 2)
    energy = s.run(max_iter=30, verbose=False)
    assert np.isfinite(energy) and s.converged
    assert s.density_residual < 1e-7
    assert s.rho_r.mean()*s.volume == pytest.approx(2, abs=1e-12)
    with pytest.raises(ValueError):
        SCFSolver(Lattice.cubic(4), 1.5, 1, 3)


def test_kpoint_density_condition_prevents_false_energy_convergence():
    s = make_solver()
    with pytest.warns(RuntimeWarning, match='did not converge'):
        s.run(max_iter=2, tol=1e3, density_tol=1e-20, verbose=False, smearing='fixed')
    assert not s.converged
    assert s.density_residual > 1e-20
