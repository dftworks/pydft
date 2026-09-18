"""Exact Fourier and occupation checks; see book appendix validation section."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_hermite
from src.lattice import Lattice
from src.gvector import GVector
from src.hamiltonian import Hamiltonian
from src.density import compute_density, compute_density_gradient
from src.smearing import (FermiDirac, Gaussian, MethfesselPaxton,
                          FixedOccupation, find_fermi_level, create_smearing)
from src.mixing import BroydenMixer, PulayMixer


def test_fcc_basis_is_complete():
    g = GVector(Lattice.fcc(10.26), 20)
    # Independent over-wide cube, testing completeness rather than membership.
    m = np.array(np.meshgrid(*([np.arange(-12, 13)] * 3), indexing='ij')).reshape(3, -1).T
    keep = np.sum((m @ g.lattice.reciprocal_vectors)**2, axis=1) / 2 < 20
    assert set(map(tuple, g.miller)) == set(map(tuple, m[keep]))
    assert g.npw == 1139


def test_local_convolution_has_no_wrapped_matrix_element():
    g = GVector(Lattice.cubic(2*np.pi), 5)
    h = Hamiltonian(g, g.lattice.volume)
    x = np.arange(h.fft_shape[0]) * 2*np.pi/h.fft_shape[0]
    h.set_local_potential(np.broadcast_to(np.cos(2*x)[:, None, None], h.fft_shape))
    psi = np.zeros(g.npw, complex)
    psi[np.all(g.miller == [3, 0, 0], axis=1)] = 1
    result = h.apply(psi)
    # Fourier identity cos(2x)e^(3ix)=(e^(5ix)+e^(ix))/2.
    assert abs(result[np.all(g.miller == [-3, 0, 0], axis=1)][0]) < 1e-13
    assert result[np.all(g.miller == [1, 0, 0], axis=1)][0] == pytest.approx(.5, abs=1e-13)


def test_density_gradient_includes_modes_outside_orbital_cutoff():
    lattice = Lattice([[2*np.pi, 0, 0], [1, 2*np.pi, 0], [0, 0, 2*np.pi]])
    g = GVector(lattice, 1)
    shape = (16, 16, 16)
    phase = 2*np.pi*3*np.arange(16)[:, None, None]/16
    rho = np.broadcast_to(2+np.sin(phase), shape)
    gradient, _ = compute_density_gradient(rho, g, shape, lattice)
    for d in range(3):
        expected = np.broadcast_to(3*lattice.reciprocal_vectors[0, d]*np.cos(phase), shape)
        np.testing.assert_allclose(gradient[d], expected, atol=2e-14)  # FFT roundoff.


@pytest.mark.parametrize('order', [1, 2])
def test_mp_symmetry_and_cancelled_moments(order):
    smearing = MethfesselPaxton(sigma=1, order=order)
    x = np.linspace(-4, 4, 81)
    np.testing.assert_allclose(smearing.occupation_array(x, 0)
                               + smearing.occupation_array(-x, 0), 1, atol=2e-15)
    assert smearing.occupation(0, 0) == .5
    assert np.min(smearing.occupation_array(x, 0)) < 0
    # Independent broadened delta: Gaussian times even Hermite polynomials.
    import math
    def delta(x):
        return np.exp(-x*x)/np.sqrt(np.pi)*sum(
            (-1)**n*eval_hermite(2*n, x)/(4**n*math.factorial(n))
            for n in range(order+1))
    for power in range(2, 2*order+1, 2):
        # Integration by parts relates the occupation correction to its delta.
        moment = quad(lambda x: x**power*delta(x), -8, 8, epsabs=1e-11)[0]
        assert abs(moment) < 1e-10
    for x0 in [.3, 1.1]:
        step = 1e-5
        derivative = -(smearing.occupation(x0+step, 0)-smearing.occupation(x0-step, 0))/(2*step)
        assert derivative == pytest.approx(delta(x0), abs=1e-9)


def test_fermi_search_charge_degeneracy_and_broadening():
    _, occ = find_fermi_level(np.array([[0., 0.], [0., 1.]]), [.25, .75], 1., FixedOccupation())
    np.testing.assert_allclose(occ, [[.8, .8], [.8, 0]], atol=1e-14)
    _, occ = find_fermi_level(np.array([0.]), 1., .1, Gaussian(100))
    assert occ.sum() == pytest.approx(.1, abs=1e-10)
    with pytest.raises(ValueError, match='Electron count'):
        find_fermi_level(np.array([0., 1.]), 1., 6, FermiDirac())
    with pytest.raises(RuntimeError, match='did not converge'):
        find_fermi_level(np.array([0.]), 1., .1, Gaussian(), max_iter=1)


def test_entropy_endpoints_and_invalid_smearing():
    assert FermiDirac().entropy(np.array([0., 1.])) == 0
    assert FermiDirac().entropy(np.array([.5])) > 0
    with pytest.raises(ValueError):
        create_smearing('typo')
    with pytest.raises(ValueError):
        Gaussian(0)


@pytest.mark.parametrize('electrons', [0., 2.])
def test_fermi_endpoint_mu_matches_occupations(electrons):
    fd = FermiDirac(temperature=1e6)
    mu, occ = find_fermi_level(np.array([0.]), 1., electrons, fd)
    np.testing.assert_allclose(occ, 2*fd.occupation_array(np.array([0.]), mu), atol=1e-14)
    assert occ.sum() == pytest.approx(electrons, abs=1e-10)


@pytest.mark.parametrize('mixer_type', [BroydenMixer, PulayMixer])
def test_nonlinear_mixers_accept_real_arrays(mixer_type):
    mixer = mixer_type()
    rho = np.array([1.])
    for _ in range(3):
        rho = mixer.mix(rho.real, np.array([2.]))
        assert np.isfinite(rho).all()
        np.testing.assert_allclose(rho.imag, 0, atol=1e-14)


def test_signed_occupations_preserve_charge():
    g = GVector(Lattice.cubic(2*np.pi), 1)
    psi = np.eye(g.npw, 2, dtype=complex)
    rho = compute_density(psi, np.array([2.1, -.1]), g, g.get_fft_grid_size(), g.lattice.volume)
    assert rho.mean()*g.lattice.volume == pytest.approx(2., abs=1e-13)
