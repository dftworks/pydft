import numpy as np

from src.lattice import Lattice
from src.pseudopotential import create_periodic_gaussian_model_potential


def test_periodic_gaussian_potential_is_smooth_and_translation_invariant():
    lattice = Lattice(np.array([
        [5.0, 0.0, 0.0],
        [1.1, 4.2, 0.0],
        [0.4, 0.7, 6.0],
    ]))
    positions = np.array([[0.17, 0.29, 0.41], [0.63, 0.52, 0.08]])
    grid = (16, 18, 20)

    potential = create_periodic_gaussian_model_potential(
        lattice, positions, grid, depth=1.2, r_c=1.1)
    shifted = create_periodic_gaussian_model_potential(
        lattice, positions + [[1, -2, 3], [-4, 2, -1]], grid,
        depth=1.2, r_c=1.1)

    assert potential.shape == grid
    assert np.isfinite(potential).all()
    assert potential.min() < potential.max() < 0
    np.testing.assert_allclose(shifted, potential, atol=1e-12, rtol=1e-12)

    # Compare one grid point with a direct Cartesian sum over periodic images.
    index = (4, 8, 12)
    frac = np.array(index) / grid
    direct = 0.0
    for atom in positions:
        for i in range(-4, 5):
            for j in range(-4, 5):
                for k in range(-4, 5):
                    displacement = (frac - atom + [i, j, k]) @ lattice.vectors
                    direct -= 1.2 * np.exp(-displacement @ displacement / 1.1**2)
    np.testing.assert_allclose(potential[index], direct, atol=1e-12, rtol=1e-12)
