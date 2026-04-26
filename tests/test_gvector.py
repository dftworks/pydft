import numpy as np
import pytest
from src.lattice import Lattice
from src.gvector import GVector

def test_gvector_generation():
    """Test that GVector generates correct reciprocal lattice vectors."""
    lattice = Lattice.cubic(10.0)
    ecut = 5.0
    
    gvec = GVector(lattice, ecut)
    
    # Check that npw is reasonable
    assert gvec.npw > 0
    
    # Check that G=0 is the first vector
    assert np.allclose(gvec.cart[0], [0, 0, 0])
    assert np.allclose(gvec.norms[0], 0.0)
    assert np.allclose(gvec.miller[0], [0, 0, 0])
    
    # Check that all G-vectors satisfy the energy cutoff
    # 0.5 * |G|^2 < ecut
    kinetic_energies = 0.5 * gvec.norms**2
    assert np.all(kinetic_energies < ecut + 1e-10)
    
def test_fft_grid_mapping():
    """Test mapping to and from the FFT grid."""
    lattice = Lattice.cubic(10.0)
    ecut = 5.0
    gvec = GVector(lattice, ecut)
    
    fft_shape = gvec.get_fft_grid_size(factor=2.0)
    
    # Create some dummy data in G-space
    data_g = np.random.rand(gvec.npw) + 1j * np.random.rand(gvec.npw)
    
    # Map to FFT grid and back
    grid = gvec.map_to_fft_grid(data_g, fft_shape)
    recovered_data_g = gvec.map_from_fft_grid(grid)
    
    # Should recover original data exactly
    np.testing.assert_allclose(recovered_data_g, data_g)
