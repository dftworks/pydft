import numpy as np
import pytest
from src.lattice import Lattice
from src.gvector import GVector
from src.kpoints import (monkhorst_pack, automatic_kpoints, GammaPoint,
                         get_kpg_norms)
from src.kpoint_scf import generate_monkhorst_pack

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


def test_kpg_norms_match_direct_cartesian_norms():
    gvector = GVector(Lattice.fcc(8.0), 2.0)
    k_cart = np.array([0.07, -0.11, 0.03])

    actual = get_kpg_norms(k_cart, gvector)
    expected = np.sum((gvector.cart + k_cart)**2, axis=1)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("mesh", [(2, 2, 2), (3, 2, 1), (3, 3, 3)])
@pytest.mark.parametrize("shift", [False, True])
def test_monkhorst_pack_apis_share_shift_semantics(mesh, shift):
    lattice = Lattice.cubic(10.0)
    kpoints = monkhorst_pack(lattice, *mesh, shift=shift)
    scf_points, scf_weights = generate_monkhorst_pack(
        *mesh, shift=(float(shift),) * 3
    )

    # Compare modulo reciprocal lattice translations in a canonical cell.
    def canonical(points):
        points = np.round((points + 0.5) % 1.0 - 0.5, 12)
        return points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]

    np.testing.assert_allclose(canonical(kpoints.k_frac), canonical(scf_points))
    np.testing.assert_allclose(kpoints.weights, scf_weights)
