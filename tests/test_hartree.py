import numpy as np
import pytest
from src.hartree import compute_hartree_potential, compute_hartree_energy

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
