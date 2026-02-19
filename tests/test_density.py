import numpy as np
import pytest

from src.density import compute_density


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
