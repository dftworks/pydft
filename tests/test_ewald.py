"""Energy derivative invariants for Ewald; see pydft-book/ewald.tex."""
import numpy as np
import pytest
from src.lattice import Lattice
from src.ewald import Ewald

A = np.array([[4., 0, 0], [3.6, 1.7, 0], [.7, .3, 4.1]])
POSITIONS = np.array([[.03, .94, .1], [.8, .07, .6]])


def test_ewald_forces_stress_and_periodicity():
    lattice = Lattice(A)
    ewald = Ewald(lattice, POSITIONS, [1, 2], eta=.7)
    h = 1e-5
    # Central differences at fixed eta: 2e-8 allows sum truncation and FD roundoff.
    for d in range(3):
        shift = np.eye(3)[d] @ np.linalg.inv(A) * h
        plus, minus = POSITIONS.copy(), POSITIONS.copy()
        plus[0] += shift
        minus[0] -= shift
        force = -(Ewald(lattice, plus, [1, 2], eta=.7).energy
                  - Ewald(lattice, minus, [1, 2], eta=.7).energy)/(2*h)
        assert ewald.forces[0, d] == pytest.approx(force, abs=2e-8)
    # Diagonal and off-diagonal strains, fractional coordinates held fixed.
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1)]:
        strain = np.zeros((3, 3)); strain[i, j] = h
        plus = Ewald(Lattice(A @ (np.eye(3)+strain).T), POSITIONS, [1, 2], eta=.7)
        minus = Ewald(Lattice(A @ (np.eye(3)-strain).T), POSITIONS, [1, 2], eta=.7)
        stress = -(plus.energy-minus.energy)/(2*h*lattice.volume)
        assert ewald.stress[i, j] == pytest.approx(stress, abs=2e-8)
    shifted = Ewald(lattice, POSITIONS+[[1, -2, 0], [-3, 1, 4]], [1, 2], eta=.7)
    assert shifted.energy == pytest.approx(ewald.energy, abs=1e-12)
    np.testing.assert_allclose(ewald.forces.sum(axis=0), 0, atol=1e-12)


def test_ewald_independent_of_split_parameter():
    energies = [Ewald(Lattice(A), POSITIONS, [1, 2], eta=eta).energy for eta in [.4, 1.2]]
    assert energies[0] == pytest.approx(energies[1], abs=2e-10)
