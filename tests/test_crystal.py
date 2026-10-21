"""Crystal species charge assignment must be explicit and consistent."""
import numpy as np
import pytest

from src.crystal import Crystal
from src.lattice import Lattice


def test_unknown_element_requires_explicit_ionic_charge():
    crystal = Crystal(Lattice.cubic(5.0))

    with pytest.raises(ValueError, match="pass zion explicitly"):
        crystal.add_atom("Xe", [0.0, 0.0, 0.0])

    crystal.add_atom("Xe", [0.0, 0.0, 0.0], zion=8)
    np.testing.assert_array_equal(crystal.get_charges(), [8])


def test_conflicting_charge_for_existing_species_is_rejected():
    crystal = Crystal(Lattice.cubic(5.0))
    crystal.add_atom("Si", [0.0, 0.0, 0.0], zion=4)

    with pytest.raises(ValueError, match="conflicting zion"):
        crystal.add_atom("Si", [0.5, 0.5, 0.5], zion=12)


def test_atoms_without_repeated_zion_reuse_existing_species_charge():
    crystal = Crystal(Lattice.cubic(5.0))
    crystal.add_atom("Si", [0.0, 0.0, 0.0], zion=3)
    crystal.add_atom("Si", [0.5, 0.5, 0.5])

    np.testing.assert_array_equal(crystal.get_charges(), [3, 3])
