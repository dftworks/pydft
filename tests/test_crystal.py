"""Crystal species charge assignment must be explicit and consistent."""
import numpy as np
import pytest

from src.crystal import Crystal, read_xyz
from src.lattice import Lattice
from src.constants import ANG_TO_BOHR


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


def test_read_xyz_defaults_to_angstrom_and_wraps_positions(tmp_path):
    xyz = tmp_path / "dimer.xyz"
    xyz.write_text("2\nSi dimer\nSi 0 0 0\nSi 2.35 0 0\n")
    lattice = Lattice.cubic(20.0)

    crystal = read_xyz(xyz, lattice)

    bond_length = np.linalg.norm(np.diff(crystal.get_positions_cart(), axis=0)[0])
    assert bond_length == pytest.approx(2.35 * ANG_TO_BOHR)
    assert np.all((crystal.get_positions() >= 0) & (crystal.get_positions() < 1))
    np.testing.assert_array_equal(crystal.get_charges(), [4, 4])


def test_read_xyz_accepts_bohr_and_explicit_charges(tmp_path):
    xyz = tmp_path / "unknown.xyz"
    xyz.write_text("2\nunknown element\nXx 11 0 0\nXx 13 0 0\n")
    lattice = Lattice.cubic(10.0)

    crystal = read_xyz(
        xyz, lattice, units="bohr", zion_by_element={"Xx": 2})

    positions = crystal.get_positions_cart()
    assert np.linalg.norm(positions[1] - positions[0]) == pytest.approx(2.0)
    assert np.all((crystal.get_positions() >= 0) & (crystal.get_positions() < 1))
    np.testing.assert_array_equal(crystal.get_charges(), [2, 2])


def test_read_xyz_rejects_unknown_units(tmp_path):
    xyz = tmp_path / "empty.xyz"
    xyz.write_text("0\nempty\n")

    with pytest.raises(ValueError, match="units must be"):
        read_xyz(xyz, Lattice.cubic(10.0), units="nm")
