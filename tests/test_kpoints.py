import numpy as np
import pytest
from src.lattice import Lattice
from src.kpoints import monkhorst_pack, automatic_kpoints, GammaPoint

def test_gamma_point():
    """Test Gamma point generator."""
    lattice = Lattice.cubic(10.0)
    kpts = GammaPoint(lattice)
    
    assert kpts.get_nk() == 1
    assert np.allclose(kpts.get_k_frac(0), [0.0, 0.0, 0.0])
    assert np.allclose(kpts.get_weight(0), 1.0)

def test_monkhorst_pack_mesh():
    """Test Monkhorst-Pack k-point mesh generation."""
    lattice = Lattice.cubic(10.0)
    
    # 2x2x2 mesh without shift
    kpts = monkhorst_pack(lattice, 2, 2, 2, shift=False)
    assert kpts.get_nk() == 8
    
    # Weights should sum to 1
    total_weight = sum(kpts.get_weight(i) for i in range(kpts.get_nk()))
    assert np.isclose(total_weight, 1.0)
    
    # 1x1x1 mesh with shift
    kpts_shifted = monkhorst_pack(lattice, 1, 1, 1, shift=True)
    assert kpts_shifted.get_nk() == 1
    assert np.allclose(kpts_shifted.get_k_frac(0), [0.5, 0.5, 0.5])

def test_automatic_kpoints():
    """Test automatic k-point mesh generation based on spacing."""
    lattice = Lattice.cubic(10.0)  # b = 2pi/10 = 0.628
    
    kpts = automatic_kpoints(lattice, kspacing=0.2)
    # n = ceil(0.628 / 0.2) = 4
    assert kpts.mesh == (3, 3, 3) or kpts.mesh == (4, 4, 4)
    assert kpts.get_nk() > 0
