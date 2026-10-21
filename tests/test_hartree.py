import numpy as np
import pytest
from src.hartree import (compute_hartree_potential, compute_hartree_energy,
                         compute_hartree_stress)
from src.lattice import Lattice

def test_hartree_potential_g0():
    """Verify that Hartree potential at G=0 is set to zero."""
    rhog = np.ones(5, dtype=complex)
    g_norms = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    
    vhg = compute_hartree_potential(rhog, g_norms)
    
    assert vhg[0] == 0.0
    assert np.all(vhg[1:] != 0)

def test_hartree_energy_scaling():
    """Verify Hartree energy scaling with density and volume."""
    volume = 100.0
    rhog = np.array([1.0, 0.5, 0.5j], dtype=complex)
    g_norms = np.array([0.0, 1.0, 2.0])
    
    e1 = compute_hartree_energy(rhog, g_norms, volume)
    
    # Doubling density should quadruple energy (E ~ |rho|^2)
    e2 = compute_hartree_energy(2.0 * rhog, g_norms, volume)
    np.testing.assert_allclose(e2, 4.0 * e1, rtol=1e-10)
    
    # Doubling volume should double energy
    e3 = compute_hartree_energy(rhog, g_norms, 2.0 * volume)
    np.testing.assert_allclose(e3, 2.0 * e1, rtol=1e-10)


def test_hartree_stress_matches_fixed_electron_count_strain_derivative():
    vectors = np.array([[4.0, 0, 0], [3.6, 1.7, 0], [0.7, 0.3, 4.1]])
    miller = np.array([
        [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, -1, 0],
    ])
    amplitudes = np.array([
        2.0, 0.8 + 0.2j, 0.8 - 0.2j, -0.3 + 0.1j, -0.3 - 0.1j,
        0.4 - 0.15j, 0.4 + 0.15j, 0.2 + 0.1j, 0.2 - 0.1j,
    ])
    lattice = Lattice(vectors)
    g_cart = miller @ lattice.reciprocal_vectors
    g_norms = np.linalg.norm(g_cart, axis=1)
    rhog = amplitudes / lattice.volume
    stress = compute_hartree_stress(rhog, g_cart, g_norms, lattice.volume)

    def strained_energy(strain):
        strained = Lattice(vectors @ (np.eye(3) + strain).T)
        strained_g = miller @ strained.reciprocal_vectors
        strained_norms = np.linalg.norm(strained_g, axis=1)
        # Keep the electron count fixed as the volume changes.
        strained_rhog = amplitudes / strained.volume
        return compute_hartree_energy(
            strained_rhog, strained_norms, strained.volume)

    h = 1e-5
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1)]:
        strain = np.zeros((3, 3))
        strain[i, j] = h
        derivative = -(strained_energy(strain) - strained_energy(-strain))
        derivative /= 2 * h * lattice.volume
        assert stress[i, j] == pytest.approx(derivative, abs=2e-8)
