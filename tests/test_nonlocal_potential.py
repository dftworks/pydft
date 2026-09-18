"""Translated projectors and Fourier phases; see book UPF chapter."""
from pathlib import Path
import numpy as np
import pytest
from src.lattice import Lattice
from src.gvector import GVector
from src.upf_reader import UPFPseudopotential
from src.nonlocal_potential import NonlocalPotential


@pytest.fixture
def psp():
    return UPFPseudopotential(Path(__file__).resolve().parents[1]/'pot/Si-sr.upf')


def test_local_potential_is_centered_on_atom(psp):
    lattice = Lattice.cubic(8)
    g = GVector(lattice, 2)
    potential = NonlocalPotential(psp, [[.25, 0, 0]], g, lattice.volume)
    v = np.fft.ifftn(g.map_to_fft_grid(potential.vloc_g, (32, 32, 32))).real
    assert np.unravel_index(v.argmin(), v.shape) == (8, 0, 0)


def test_nonlocal_wrapper_translation_and_hermiticity(psp):
    lattice = Lattice.cubic(8)
    g = GVector(lattice, 1)
    origin = NonlocalPotential(psp, [[0, 0, 0]], g, lattice.volume)
    tau = np.array([.23, .17, .31])
    shifted = NonlocalPotential(psp, [tau], g, lattice.volume)
    k = np.array([.03, .07, .02])
    rng = np.random.default_rng(6)
    psi = rng.normal(size=g.npw)+1j*rng.normal(size=g.npw)
    phi = rng.normal(size=g.npw)+1j*rng.normal(size=g.npw)
    phase = np.exp(-1j*((g.cart+k) @ (tau @ lattice.vectors)))
    applied = origin.apply_vnl_k(psi, k)
    np.testing.assert_allclose(applied, origin.apply_vnl(psi, k_cart=k), atol=1e-12)
    np.testing.assert_allclose(shifted.apply_vnl_k(phase*psi, k), phase*applied,
                               atol=1e-12)  # Floating-point projector contractions.
    assert np.vdot(phi, applied) == pytest.approx(
        np.vdot(origin.apply_vnl_k(phi, k), psi), abs=1e-12)
