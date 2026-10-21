import numpy as np

from src.gvector import GVector
from src.lattice import Lattice
from src.silicon_bandstructure import compute_kpoint_kinetic_energies


def test_silicon_band_kinetic_energies_include_half_factor_at_gamma_and_k():
    gvec = GVector(Lattice.cubic(10.0), 1.0)

    gamma_energies = compute_kpoint_kinetic_energies(np.zeros(3), gvec)
    first_shell = np.isclose(gamma_energies, 0.5 * (2 * np.pi / 10.0) ** 2)
    assert first_shell.sum() == 6
    np.testing.assert_allclose(gamma_energies[first_shell], 0.197392088, rtol=1e-8)

    k_cart = np.array([0.13, 0.07, 0.04])
    energies = compute_kpoint_kinetic_energies(k_cart, gvec)
    expected = 0.5 * np.sum((gvec.cart + k_cart) ** 2, axis=1)
    np.testing.assert_allclose(energies, expected)
