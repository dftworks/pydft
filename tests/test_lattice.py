import numpy as np
import pytest
from src.lattice import Lattice

def test_lattice_reciprocal_orthogonality():
    """Verify that a_i . b_j = 2*pi * delta_ij."""
    # Test with a non-trivial triclinic lattice
    vectors = np.array([
        [10.0, 0.0, 0.0],
        [2.0, 8.0, 0.0],
        [1.0, 1.0, 12.0]
    ])
    lat = Lattice(vectors)
    
    a = lat.vectors
    b = lat.reciprocal_vectors
    
    for i in range(3):
        for j in range(3):
            dot = np.dot(a[i], b[j])
            if i == j:
                np.testing.assert_allclose(dot, 2.0 * np.pi, rtol=1e-10)
            else:
                np.testing.assert_allclose(dot, 0.0, atol=1e-10)

def test_lattice_volume():
    """Verify lattice volume calculation."""
    vectors = np.eye(3) * 5.0
    lat = Lattice(vectors)
    np.testing.assert_allclose(lat.volume, 125.0, rtol=1e-12)
    
def test_cubic_lattice():
    """Test cubic lattice helper."""
    lat = Lattice.cubic(10.0)
    np.testing.assert_allclose(lat.vectors, np.eye(3) * 10.0)
    np.testing.assert_allclose(lat.reciprocal_vectors, np.eye(3) * (2.0 * np.pi / 10.0))
