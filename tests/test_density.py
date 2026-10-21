import numpy as np
import pytest

from types import SimpleNamespace

from src.density import compute_density, initial_density_atomic
from src.lattice import Lattice


def test_compute_density_rejects_invalid_evec_rank():
    evecs = np.zeros((2, 3, 4, 5))
    occupations = np.zeros((3,))
    with pytest.raises(ValueError, match="ndim 2 or 3"):
        compute_density(
            evecs=evecs,
            occupations=occupations,
            gvector=None,
            fft_shape=(2, 2, 2),
            volume=1.0,
        )


def test_compute_density_rejects_single_k_occupation_shape():
    evecs = np.zeros((4, 2))
    occupations = np.zeros((1, 2))
    with pytest.raises(ValueError, match="single-k input"):
        compute_density(
            evecs=evecs,
            occupations=occupations,
            gvector=None,
            fft_shape=(2, 2, 2),
            volume=1.0,
        )


def test_compute_density_rejects_invalid_k_weight_shape():
    evecs = np.zeros((2, 4, 3))
    occupations = np.zeros((2, 3))
    with pytest.raises(ValueError, match="k_weights must have shape"):
        compute_density(
            evecs=evecs,
            occupations=occupations,
            gvector=None,
            fft_shape=(2, 2, 2),
            volume=1.0,
            k_weights=np.ones(3),
        )


@pytest.mark.parametrize("lattice_constant", [8.0, 16.0])
def test_atomic_density_width_is_independent_of_cell_size(lattice_constant):
    lattice = Lattice.cubic(lattice_constant)
    shape = (64, 64, 64)
    rho = initial_density_atomic(
        [SimpleNamespace(zion=1.0)], [[0.5, 0.5, 0.5]],
        SimpleNamespace(lattice=lattice), shape, lattice.volume)

    axis = np.arange(shape[0]) / shape[0]
    dx = axis - 0.5
    dx -= np.round(dx)
    x_cart = dx * lattice_constant
    x_grid = np.broadcast_to(x_cart[:, None, None], shape)
    rms_width = np.sqrt(np.sum(rho * x_grid**2) / np.sum(rho))

    assert rms_width == pytest.approx(1.0, abs=0.02)


def test_atomic_density_uses_skew_lattice_metric():
    vectors = np.array([[12.0, 0.0, 0.0], [3.0, 11.0, 0.0], [1.0, 2.0, 13.0]])
    lattice = Lattice(vectors)
    shape = (48, 48, 48)
    rho = initial_density_atomic(
        [SimpleNamespace(zion=1.0)], [[0.5, 0.5, 0.5]],
        SimpleNamespace(lattice=lattice), shape, lattice.volume)

    axes = [np.arange(n) / n for n in shape]
    grid_frac = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    dfrac = grid_frac - 0.5
    dfrac -= np.round(dfrac)
    dcart = dfrac @ vectors
    covariance = np.einsum('ijk,ijka,ijkb->ab', rho, dcart, dcart) / rho.sum()

    np.testing.assert_allclose(covariance, np.eye(3), atol=0.04)
