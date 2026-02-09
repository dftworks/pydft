import numpy as np
from src.real_spherical import real_spherical_harmonic


def test_l0_m0_constant():
    value = real_spherical_harmonic(0, 0, 0.2, 0.3, 0.9)
    assert np.isclose(value, 1 / np.sqrt(4 * np.pi))


def test_l1_m1_real_part_matches_expected_ratio():
    theta = np.linspace(0.1, 2.0, 20)
    phi = np.linspace(0.0, 2 * np.pi, 20)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    result = real_spherical_harmonic(1, 1, x, y, z)
    denom = np.sqrt(x * x + y * y + z * z)
    basis = x * np.sqrt(3 / (8 * np.pi)) / denom
    # We expect a constant ratio across the sampled directions (global sign may differ)
    ratio = result / basis
    ratio_val = ratio[0, 0]
    assert np.allclose(ratio, ratio_val, atol=1e-6)
