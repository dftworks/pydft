"""Standalone energy helpers share the package's units and weight conventions."""
import numpy as np
import pytest

from src.gvector import GVector
from src.lattice import Lattice
from src.pwbasis import PWBasis
from src.smearing import FermiDirac, compute_band_energy as compute_smearing_band_energy
from src.total_energy import (TotalEnergy, compute_band_energy,
                              compute_kinetic_energy)


def test_total_energy_subtracts_positive_fermi_dirac_ts():
    smearing = FermiDirac(temperature=1000.0)
    ts = smearing.entropy(np.array([0.2, 0.8]))
    energy = TotalEnergy()
    energy.e_kinetic = 1.25
    energy.e_entropy = ts

    assert energy.compute_total(method="direct") == pytest.approx(1.25 - ts)


def test_kinetic_helper_accepts_pwbasis_kinetic_energies():
    lattice = Lattice.cubic(2 * np.pi)
    gvec = GVector(lattice, 1.1)
    basis = PWBasis(gvec, k_cart=np.array([0.1, 0.0, 0.0]))
    evecs = np.zeros((gvec.npw, 1), dtype=complex)
    evecs[0, 0] = 1.0

    kinetic = compute_kinetic_energy(evecs, np.array([2.0]), basis.kg)

    assert kinetic == pytest.approx(2.0 * basis.kg[0])
    assert compute_kinetic_energy(
        evecs, np.array([2.0]), basis.kg, k_weights=0.5
    ) == pytest.approx(kinetic / 2)


def test_scalar_k_weights_are_normalized_consistently():
    eigenvalues = np.array([[1.0, 2.0], [3.0, 5.0]])
    occupations = np.ones_like(eigenvalues)
    explicit = np.array([0.5, 0.5])

    expected = 0.5 * (eigenvalues[0].sum() + eigenvalues[1].sum())
    assert compute_band_energy(eigenvalues, occupations) == pytest.approx(expected)
    assert compute_smearing_band_energy(eigenvalues, occupations) == pytest.approx(expected)
    assert compute_band_energy(eigenvalues, occupations, explicit) == pytest.approx(expected)
    assert compute_smearing_band_energy(
        eigenvalues, occupations, explicit) == pytest.approx(expected)
    one_k_eigenvalues = eigenvalues[0]
    one_k_occupations = occupations[0]
    assert compute_band_energy(
        one_k_eigenvalues, one_k_occupations, k_weights=0.5
    ) == pytest.approx(1.5)
    assert compute_smearing_band_energy(
        one_k_eigenvalues, one_k_occupations, weights=0.5
    ) == pytest.approx(1.5)


def test_kinetic_scalar_k_weight_matches_explicit_average():
    lattice = Lattice.cubic(2 * np.pi)
    gvec = GVector(lattice, 1.1)
    evecs = np.zeros((2, gvec.npw, 1), dtype=complex)
    evecs[:, 0, 0] = 1.0
    occupations = np.full((2, 1), 2.0)
    kg = np.stack([
        PWBasis(gvec, np.array([0.0, 0.0, 0.0])).kg,
        PWBasis(gvec, np.array([0.2, 0.0, 0.0])).kg,
    ])

    scalar = compute_kinetic_energy(evecs, occupations, kg)
    explicit = compute_kinetic_energy(
        evecs, occupations, kg, k_weights=np.array([0.5, 0.5]))

    assert scalar == pytest.approx(explicit)
