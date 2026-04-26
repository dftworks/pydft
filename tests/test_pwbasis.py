import numpy as np
import pytest
from src.lattice import Lattice
from src.gvector import GVector
from src.pwbasis import PWBasis, create_fft_workspace

def test_pwbasis_generation():
    """Test that PWBasis correctly handles k-points."""
    lattice = Lattice.cubic(10.0)
    ecut = 5.0
    gvec = GVector(lattice, ecut)
    
    # Gamma point
    k_gamma = np.array([0.0, 0.0, 0.0])
    pw_gamma = PWBasis(gvec, k_gamma)
    
    # Gamma point kinetic energy should match gvec exactly
    np.testing.assert_allclose(pw_gamma.get_kg(), 0.5 * gvec.norms**2)
    assert pw_gamma.get_npw() == gvec.npw
    
    # Non-Gamma point
    k_point = np.array([0.1, 0.2, 0.3])
    pw_k = PWBasis(gvec, k_point)
    
    # Kinetic energy |k+G|^2 / 2
    kg_expected = 0.5 * np.sum((gvec.cart + k_point)**2, axis=1)
    np.testing.assert_allclose(pw_k.get_kg(), kg_expected)

def test_create_fft_workspace():
    """Test creation of FFT workspace arrays."""
    lattice = Lattice.cubic(10.0)
    ecut = 5.0
    gvec = GVector(lattice, ecut)
    
    fft_shape, work_r, work_g = create_fft_workspace(gvec, factor=2.0)
    
    assert work_r.shape == fft_shape
    assert work_g.shape == fft_shape
    assert work_r.dtype == complex
    assert work_g.dtype == complex
