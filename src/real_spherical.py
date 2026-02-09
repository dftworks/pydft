"""Utilities for real spherical harmonics."""

import numpy as np
from scipy.special import sph_harm_y


def real_spherical_harmonic(l, m, x, y, z):
    """Compute a real spherical harmonic for vectors (x, y, z)."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    r = np.sqrt(x * x + y * y + z * z)
    r_safe = np.where(r < 1e-15, 1.0, r)

    theta = np.arccos(z / r_safe)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    if m == 0:
        ylm = sph_harm_y(l, 0, theta, phi)
        res = np.real(ylm)
    elif m > 0:
        ylm_pos = sph_harm_y(l, m, theta, phi)
        ylm_neg = sph_harm_y(l, -m, theta, phi)
        res = np.real((ylm_pos + ((-1) ** m) * ylm_neg) / np.sqrt(2))
    else:
        ylm_pos = sph_harm_y(l, -m, theta, phi)
        ylm_neg = sph_harm_y(l, m, theta, phi)
        res = np.real((ylm_pos - ((-1) ** (-m)) * ylm_neg) / (1j * np.sqrt(2)))

    if l == 0:
        res = np.where(r < 1e-15, 1.0 / np.sqrt(4 * np.pi), res)
    else:
        res = np.where(r < 1e-15, 0.0, res)

    return res if len(res) > 1 else res[0]
