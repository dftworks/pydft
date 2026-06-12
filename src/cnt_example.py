#!/usr/bin/env python3
"""
Carbon Nanotube (CNT) DFT Calculation Example

This script demonstrates a plane-wave DFT calculation for a (4,0) zigzag
carbon nanotube -- a 1D periodic material made by rolling up a graphene sheet.

The key computational idea:
- The nanotube axis is along z (periodic direction)
- Large vacuum in x and y prevents interaction between periodic images
- K-point mesh: 1 x 1 x n3 (dense only along the tube axis)

A (4,0) zigzag CNT is chosen because it is the smallest zigzag tube that
gives a reasonable structure (8 atoms per unit cell, diameter ~ 3.1 Angstrom).

Nanotube classification:
- Zigzag (n,0): metallic if n is a multiple of 3, else semiconducting
- (4,0) is NOT a multiple of 3, so it is expected to be semiconducting
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import HA_TO_EV, BOHR_TO_ANG, ANG_TO_BOHR, TWOPI
from src.lattice import Lattice
from src.crystal import Crystal
from src.gvector import GVector
from src.xc import lda_xc, compute_xc_energy, compute_xc_potential_energy
from src.hartree import compute_hartree_potential, compute_hartree_energy
from src.ewald import Ewald
from src.smearing import create_smearing, find_fermi_level
from src.hamiltonian import Hamiltonian, g_to_r
from src.eigensolver import PCGEigensolver, random_initial_guess
from src.mixing import LinearMixer


def create_cnt_crystal(n=4, m=0, vacuum=18.0):
    """
    Create the unit cell for an (n,m) carbon nanotube.

    The tube axis is along z. The translational period along z is T,
    and the tube is centred in a tetragonal box with vacuum in x and y.

    For a zigzag (n,0) tube:
        Diameter:  d = n * a_CC * sqrt(3) / pi
        Period:    T = sqrt(3) * a_CC   (= a_graphene)
        Atoms per cell: 2*n

    where a_CC = 1.42 Angstrom is the C-C bond length.

    Args:
        n, m: Chiral indices (only zigzag (n,0) is implemented)
        vacuum: Vacuum spacing in Bohr (applied in both x and y)

    Returns:
        Crystal object, translational period T in Bohr
    """
    if m != 0:
        raise NotImplementedError(
            "Only zigzag (n,0) nanotubes are implemented in this example.")

    a_cc_ang = 1.42  # C-C bond length in Angstrom
    a_cc = a_cc_ang * ANG_TO_BOHR

    # Tube diameter and translational period
    diameter = n * a_cc * np.sqrt(3.0) / np.pi
    T = np.sqrt(3.0) * a_cc  # period along the tube axis

    # Tetragonal supercell: tube centred in the xy-plane
    box_xy = diameter + vacuum  # side length in x and y

    lattice = Lattice(np.array([
        [box_xy, 0.0, 0.0],
        [0.0, box_xy, 0.0],
        [0.0, 0.0, T],
    ]))

    crystal = Crystal(lattice)

    # Generate atom positions by rolling up the graphene strip.
    # A zigzag (n,0) tube has 2*n atoms per translational period,
    # arranged in two rings at z = 0 and z = T/3.
    radius = diameter / 2.0

    for i in range(n):
        # Ring 1 at z = 0
        theta1 = 2.0 * np.pi * i / n
        x1 = radius * np.cos(theta1) + box_xy / 2.0
        y1 = radius * np.sin(theta1) + box_xy / 2.0
        z1 = 0.0

        # Convert to fractional
        fx1 = x1 / box_xy
        fy1 = y1 / box_xy
        fz1 = z1 / T

        crystal.add_atom('C', [fx1, fy1, fz1], zion=4)

        # Ring 2 at z = T/3, rotated by pi/n
        theta2 = 2.0 * np.pi * (i + 0.5) / n
        x2 = radius * np.cos(theta2) + box_xy / 2.0
        y2 = radius * np.sin(theta2) + box_xy / 2.0
        z2 = T / 3.0

        fx2 = x2 / box_xy
        fy2 = y2 / box_xy
        fz2 = z2 / T

        crystal.add_atom('C', [fx2, fy2, fz2], zion=4)

    return crystal, T


def create_cnt_potential(lattice, crystal, fft_shape,
                         v_depth=1.2, r_c=1.2):
    """
    Create a model potential for the CNT atoms.

    Uses a Gaussian potential centred on each atom, consistent with
    the graphene and silicon examples.

    Args:
        lattice: Lattice object
        crystal: Crystal object with atom positions
        fft_shape: FFT grid shape
        v_depth: Depth of potential well (Hartree)
        r_c: Characteristic radius (Bohr)
    """
    n1, n2, n3 = fft_shape
    v_ext = np.zeros(fft_shape, dtype=float)

    x = np.linspace(0, 1, n1, endpoint=False)
    y = np.linspace(0, 1, n2, endpoint=False)
    z = np.linspace(0, 1, n3, endpoint=False)

    atoms_frac = crystal.get_positions()
    a, b, c = lattice.vectors

    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            for k, zk in enumerate(z):
                grid_frac = np.array([xi, yj, zk])

                for atom_frac in atoms_frac:
                    d_frac = grid_frac - atom_frac
                    d_frac = d_frac - np.round(d_frac)

                    d_cart = d_frac[0]*a + d_frac[1]*b + d_frac[2]*c
                    r2 = np.sum(d_cart**2)

                    v_ext[i, j, k] += -v_depth * np.exp(-r2 / r_c**2)

    return v_ext


class CarbonNanotubeSCF:
    """
    SCF solver for a carbon nanotube.

    This example highlights:
    - 1D periodic geometry (tube axis along z, vacuum in x and y)
    - K-point mesh dense only along the tube axis (1 x 1 x n3)
    - More atoms per cell than silicon or graphene (2*n for a (n,0) tube)
    """

    def __init__(self, n=4, ecut=3.0, n_extra_bands=4,
                 kpts_mesh=(1, 1, 4)):
        """
        Initialize CNT SCF calculation.

        Args:
            n: Chiral index for a (n,0) zigzag tube
            ecut: Energy cutoff in Hartree
            n_extra_bands: Extra empty bands beyond the occupied ones
            kpts_mesh: K-point mesh (1, 1, n3)
        """
        print("=" * 60)
        print(f"Carbon Nanotube ({n},0) DFT Calculation (Educational)")
        print("=" * 60)

        # Create crystal structure
        self.crystal, self.T = create_cnt_crystal(n=n)
        self.lattice = self.crystal.lattice
        self.volume = self.lattice.volume
        self.natoms = self.crystal.natoms

        diameter_ang = (n * 1.42 * np.sqrt(3.0) / np.pi)

        print(f"\nNanotube: ({n},0) zigzag")
        print(f"Atoms per cell: {self.natoms}")
        print(f"Diameter: {diameter_ang:.3f} Ang")
        print(f"Period along axis: {self.T:.4f} Bohr "
              f"({self.T * BOHR_TO_ANG:.4f} Ang)")
        print(f"Cell volume: {self.volume:.4f} Bohr^3")

        # Number of electrons (4 valence per C)
        self.n_electrons = 4 * self.natoms
        n_occ = self.n_electrons // 2
        self.n_bands = n_occ + n_extra_bands

        # Parameters
        self.ecut = ecut

        # Generate G-vectors
        self.gvec = GVector(self.lattice, ecut)
        self.npw = self.gvec.npw

        # FFT grid
        self.fft_shape = self.gvec.get_fft_grid_size()
        self.n_fft = np.prod(self.fft_shape)

        print(f"\nEnergy cutoff: {ecut:.1f} Ha ({ecut * HA_TO_EV:.1f} eV)")
        print(f"Plane waves: {self.npw}")
        print(f"FFT grid: {self.fft_shape}")
        print(f"Bands: {self.n_bands} ({n_occ} occupied + "
              f"{n_extra_bands} empty)")
        print(f"Electrons: {self.n_electrons}")

        # K-points (dense along tube axis only)
        from src.kpoints import monkhorst_pack
        self.kpts = monkhorst_pack(self.lattice, *kpts_mesh)
        self.nk = self.kpts.nk
        print(f"K-points: {kpts_mesh} mesh = {self.nk} points")

        # External potential (model pseudopotential)
        print("\nBuilding model pseudopotential...")
        self.v_ext = create_cnt_potential(
            self.lattice, self.crystal, self.fft_shape)

        # Ion-ion energy (Ewald)
        positions = self.crystal.get_positions()
        charges = self.crystal.get_charges()
        self.ewald = Ewald(self.lattice, positions, charges, self.gvec)
        print(f"Ewald ion-ion energy: {self.ewald.energy:.6f} Ha")

        # Hamiltonian
        self.hamiltonian = Hamiltonian(self.gvec, self.volume)

        # Mixer - simple linear mixing. The CNT cell (many atoms, vacuum in
        # two directions) is the stiffest system in the examples; a small
        # alpha keeps the density update gentle enough to avoid sloshing.
        self.mixer = LinearMixer(alpha=0.1)

        # Smearing
        self.smearing = create_smearing('gaussian', sigma=0.02)

        # Storage
        self.evecs = None
        self.evals = None
        self.rho_r = None
        self.rho_g = None
        self.occupations = None

    def run(self, max_iter=50, tol=1e-4, verbose=True):
        """
        Run SCF calculation.

        Returns:
            Total energy in Hartree
        """
        print("\n" + "-" * 60)
        print("Starting SCF iteration...")
        print("-" * 60)

        # Initialize density (uniform)
        rho_0 = self.n_electrons / self.volume
        self.rho_r = np.full(self.fft_shape, rho_0, dtype=float)
        self.rho_g = self._r_to_g_density(self.rho_r)

        # Initialize eigenvectors
        self.evecs = random_initial_guess(self.npw, self.n_bands)
        self.evals = np.zeros(self.n_bands)

        # Fixed occupations
        n_occ = self.n_electrons // 2
        self.occupations = np.zeros(self.n_bands)
        self.occupations[:n_occ] = 2.0

        # SCF loop
        energy_old = 0.0
        converged = False
        energy_history = []

        if verbose:
            print(f"{'Iter':>4} {'E_total (Ha)':>16} {'dE (Ha)':>12} "
                  f"{'dE (eV)':>12}")
            print("-" * 50)

        for scf_iter in range(1, max_iter + 1):
            self._build_potential()

            eigensolver = PCGEigensolver(self.npw, self.n_bands)
            eigensolver.solve(
                ham_apply=self.hamiltonian.apply,
                ham_diag=self.hamiltonian.get_diagonal(),
                evecs=self.evecs,
                evals=self.evals,
                tol=1e-8,
                max_iter=200
            )

            rho_new = self._compute_density()

            energy = self._compute_total_energy()
            energy_history.append(energy)

            de = abs(energy - energy_old)

            if verbose:
                print(f"{scf_iter:4d} {energy:16.8f} {de:12.2e} "
                      f"{de * HA_TO_EV:12.2e}")

            if de < tol:
                converged = True
                print("-" * 50)
                print(f"SCF CONVERGED in {scf_iter} iterations")
                break

            if scf_iter > 10:
                recent = energy_history[-6:]
                if len(recent) == 6:
                    odd_vals = [recent[0], recent[2], recent[4]]
                    even_vals = [recent[1], recent[3], recent[5]]
                    if np.std(odd_vals) < tol and np.std(even_vals) < tol:
                        energy = np.mean(recent)
                        converged = True
                        print("-" * 50)
                        print(f"SCF oscillating - using average energy "
                              f"after {scf_iter} iterations")
                        break

            rho_mixed = self.mixer.mix(self.rho_g, rho_new)
            self.rho_g = rho_mixed
            self.rho_r = self._g_to_r_density(self.rho_g)

            energy_old = energy

        if not converged:
            print("-" * 50)
            print(f"Warning: SCF not converged after {max_iter} iterations")
            if len(energy_history) > 5:
                energy = np.mean(energy_history[-6:])

        self._print_results(energy)

        return energy

    def _r_to_g_density(self, rho_r):
        rho_fft = np.fft.fftn(rho_r) / self.n_fft
        return self.gvec.map_from_fft_grid(rho_fft)

    def _g_to_r_density(self, rho_g):
        rho_fft = self.gvec.map_to_fft_grid(rho_g, self.fft_shape)
        rho_r = np.fft.ifftn(rho_fft) * self.n_fft
        return np.real(rho_r)

    def _build_potential(self):
        v_hartree_g = compute_hartree_potential(self.rho_g, self.gvec.norms)
        v_hartree_fft = self.gvec.map_to_fft_grid(v_hartree_g, self.fft_shape)
        v_hartree_r = np.real(np.fft.ifftn(v_hartree_fft) * self.n_fft)

        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        v_xc_r, self._exc_r = lda_xc(rho_real)

        v_local_r = v_hartree_r + v_xc_r + self.v_ext

        self._v_hartree_r = v_hartree_r
        self._v_xc_r = v_xc_r

        self.hamiltonian.set_local_potential(v_local_r)

    def _compute_density(self):
        rho_r = np.zeros(self.fft_shape, dtype=float)

        for i in range(self.n_bands):
            if self.occupations[i] < 1e-10:
                continue
            psi_r = g_to_r(self.evecs[:, i], self.gvec,
                           self.fft_shape, self.volume)
            rho_r += self.occupations[i] * np.abs(psi_r)**2

        return self._r_to_g_density(rho_r)

    def _compute_total_energy(self):
        e_band = np.sum(self.occupations * self.evals)
        e_hartree = compute_hartree_energy(
            self.rho_g, self.gvec.norms, self.volume)

        rho_real = np.maximum(np.real(self.rho_r), 1e-20)
        e_xc = compute_xc_energy(rho_real, self._exc_r,
                                 self.volume, self.n_fft)
        e_vxc = compute_xc_potential_energy(
            rho_real, self._v_xc_r, self.volume, self.n_fft)

        e_total = e_band - e_hartree + e_xc - e_vxc + self.ewald.energy
        return e_total

    def _print_results(self, energy):
        print("\n" + "=" * 60)
        print("CALCULATION RESULTS")
        print("=" * 60)

        n_occ = self.n_electrons // 2

        print(f"\nMaterial: ({self.crystal.natoms // 2},0) Carbon Nanotube")
        print(f"Total energy: {energy:.8f} Ha ({energy * HA_TO_EV:.6f} eV)")
        print(f"Energy per atom: {energy/self.natoms:.8f} Ha "
              f"({energy/self.natoms * HA_TO_EV:.6f} eV)")
        print(f"Ewald energy: {self.ewald.energy:.8f} Ha")

        print(f"\nEigenvalues (Gamma point):")
        print(f"{'Band':>6} {'Energy (Ha)':>14} {'Energy (eV)':>14} "
              f"{'Occ':>8}")
        print("-" * 44)
        n_show = min(self.n_bands, n_occ + 4)
        for i in range(n_show):
            marker = " <-- HOMO" if i == n_occ - 1 else \
                     " <-- LUMO" if i == n_occ else ""
            print(f"{i+1:6d} {self.evals[i]:14.6f} "
                  f"{self.evals[i]*HA_TO_EV:14.6f} "
                  f"{self.occupations[i]:8.2f}{marker}")
        if n_show < self.n_bands:
            print(f"  ... ({self.n_bands - n_show} more bands)")

        if n_occ < self.n_bands:
            vbm = self.evals[n_occ - 1]
            cbm = self.evals[n_occ]
            gap = cbm - vbm
            print(f"\nHOMO-LUMO gap: {gap:.4f} Ha ({gap * HA_TO_EV:.4f} eV)")
            if self.crystal.natoms // 2 % 3 == 0:
                print("(Zigzag tube with n divisible by 3: "
                      "expected to be metallic)")
            else:
                print("(Zigzag tube with n not divisible by 3: "
                      "expected to be semiconducting)")

        print("=" * 60)


def main():
    """Run carbon nanotube calculation."""
    # Fixed seed for reproducible SCF output (see silicon example).
    np.random.seed(42)
    print("\n" + "#" * 60)
    print("# Educational Carbon Nanotube DFT Calculation")
    print("#" * 60)

    print("\nCarbon nanotube classification:")
    print("  A (n,m) CNT is metallic if (n - m) mod 3 = 0")
    print("  Otherwise it is semiconducting")
    print("  Zigzag tubes have m = 0")
    print("  Armchair tubes have n = m")

    # Create and run (4,0) zigzag CNT
    # Using lower ecut and fewer bands for speed (educational demo)
    cnt_scf = CarbonNanotubeSCF(
        n=4,
        ecut=3.0,              # Low cutoff for speed
        n_extra_bands=4,       # 4 empty bands above occupied ones
        kpts_mesh=(1, 1, 4)    # Dense only along tube axis
    )

    energy = cnt_scf.run(max_iter=60, tol=1e-4)

    print("\n" + "#" * 60)
    print("# Calculation Complete!")
    print("#" * 60 + "\n")

    return energy


if __name__ == "__main__":
    main()
